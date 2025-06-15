"""mae_single_frame_poc.py (scratch version)
----------------------------------------------------------------------
A *from--scratch* proof--of--concept **Masked Auto--Encoder** in PyTorch that
uses a **ViT--Tiny** backbone but implements its *own* patch embedding,
masking logic, and lightweight decoder.  No timm MAE helpers are used ---
we only borrow ``torchvision`` for basic layers.

Main simplifications vs. full MAE paper
--------------------------------------
* **ViT--Tiny** config (embed_dim = 192, heads = 3, depth = 12).
* Decoder shares the same width (192) instead of using a smaller width.
* Only *one* target camera (default ``CAM_FRONT``), RGB cropped to
  224224 so we get exactly **1414 = 196** patches.
* Reconstruction loss is **L2** on *unnormalised* RGB patches.

You can train on NuScenes with:
```
python mae_single_frame_poc.py --nusc_root /path/to/nuscenes --epochs 30 \
       --batch 128 --mask_ratio 0.75 --device cuda
```

Expected (tiny) GPU cost: -1.4 GB for bs = 128 on an A5000.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# ---------------------------------------------------------------------------
# 0.  Patch--embedding & positional embeddings
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Split image into non--overlapping patches and embed them."""

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 192):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W)
        x = self.proj(x)                 # (B,embed_dim,H/ps,W/ps)
        x = x.flatten(2).transpose(1, 2) # (B,N,embed_dim)
        return x


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Sin--cos positional embeddings like the original MAE code."""
    import math
    # create grid of coordinates
    coords_h = torch.arange(grid_size, dtype=torch.float32)
    coords_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(coords_h, coords_w, indexing="ij")  # two (G,G) maps

    # flatten to positions
    pos_h = grid[0].flatten()  # (N,)
    pos_w = grid[1].flatten()  # (N,)

    # get 1D embeddings for each axis
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, pos_h)  # (1,N,embed_dim/2)
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, pos_w)  # (1,N,embed_dim/2)

    # concatenate on the embedding dimension
    pos_embed = torch.cat([emb_h, emb_w], dim=2)  # (1,N,embed_dim)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim: int, pos):
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)  # (embed_dim/2,)
    pos = pos.flatten()            # (G*G,)
    out = torch.einsum('m,n->mn', pos, omega)
    emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
    return emb.unsqueeze(0)        # (1,N,embed_dim)

# ---------------------------------------------------------------------------
# 1.  ViT--Tiny encoder (transformer blocks only, no patch--embedding)
# ---------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                activation='gelu', batch_first=True,
                norm_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.layers:
            x = blk(x)
        return self.norm(x)  # (B,N_visible+1,embed_dim)

# ---------------------------------------------------------------------------
# 2.  MAE model (mask, encode visible, decode & reconstruct)
# ---------------------------------------------------------------------------

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=192,
                 depth=12, num_heads=3, decoder_depth=4, mask_ratio=0.75):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token (optional for MAE; we keep it for completeness)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            get_2d_sincos_pos_embed(embed_dim, int(num_patches ** 0.5)),
            requires_grad=False)  # (1,N,embed_dim)

        self.encoder = ViTEncoder(embed_dim, depth, num_heads)

        # ---- decoder ----
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # clone positional embeddings for decoder and register as buffer for proper device placement
        self.register_buffer("decoder_pos_embed", self.pos_embed.clone())
        self.decoder = ViTEncoder(embed_dim, decoder_depth, num_heads)
        self.decoder_pred = nn.Linear(embed_dim, patch_size * patch_size * 3, bias=True)

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = num_patches
        self.initialize_weights()

    # ---------------------------------------------------------------------
    # helper functions
    # ---------------------------------------------------------------------
    def initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pred.weight, std=0.02)
        nn.init.zeros_(self.decoder_pred.bias)

    # ---------------------------------------------------------------------
    def random_masking(self, x, mask_ratio) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate binary mask. 0 = keep, 1 = mask."""
        B, N, _ = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)  # noise ~ U[0,1)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(2)))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    # ---------------------------------------------------------------------
    def forward(self, imgs: torch.Tensor):  # (B,3,H,W)
        # Patchify
        x = self.patch_embed(imgs)  # (B,N,embed)
        B, N, C = x.shape

        # Add positional encodings
        x = x + self.pos_embed

        # Random masking
        x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # prepend CLS if desired (not used for reconstruction)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_masked = torch.cat([cls_tokens, x_masked], dim=1)

        # Encode visible patches
        latent = self.encoder(x_masked)  # (B,1+N_vis,embed)

        # ---- decoder: embed + add mask tokens
        latent = self.decoder_embed(latent[:, 1:])  # drop CLS for simplicity

        # insert mask tokens
        mask_tokens = self.mask_token.expand(B, ids_restore.shape[1] + 1 - latent.shape[1], -1)
        combined = torch.cat([latent, mask_tokens], dim=1)  # (B,N,embed)
        # reorder to original positions
        combined = torch.gather(combined, dim=1,
                                index=ids_restore.unsqueeze(-1).repeat(1, 1, combined.size(2)))

        # DEBUG
        '''
        masked_embeddings = combined[0, mask[0]==1]    # (N_mask, D)
        print("masked embeddings shape:", masked_embeddings.shape)
        print("masked embeddings  mean/std:", 
        masked_embeddings.mean().item(), 
        masked_embeddings.std().item())
        '''

        # add positional embeddings
        combined = combined + self.decoder_pos_embed
        decoded = self.decoder(combined)
        pred = self.decoder_pred(decoded)  # (B,N,patch*patch*3)

        # DEBUG 2
        # now inspect masked slots with pos‐embeddings included
        me = combined[0, mask[0]==1]        # [N_mask, D]
        print("▶︎ masked embeddings w/ pos:", me.mean().item(), me.std().item())

        # also inspect visible slots to ensure they’re not constant
        ve = combined[0, mask[0]==0]        # [N_vis, D]
        print("▶︎ visible embeddings w/ pos:", ve.mean().item(), ve.std().item())

        # finally run your decoder head on just the first few masked tokens
        small = me[:5]                      # pick 5 masked tokens
        out = self.decoder_pred(self.decoder(small.unsqueeze(0)))[0]  # [5, P*P*3]
        print("▶︎ raw pred patch stats:", out.mean().item(), out.std().item())


        # loss: compute only on masked patches
        img_patches = self.patchify(imgs)  # (B,N,P)
        loss = (pred - img_patches) ** 2
        loss = loss.mean(dim=-1)  # per patch MSE
        loss = (loss * mask).sum() / mask.sum()  # mean over masked patches

        return loss, pred, mask

    def patchify(self, imgs):
        P = self.patch_size
        B, C, H, W = imgs.shape
        assert H == W == self.img_size
        imgs = imgs.reshape(B, C, H // P, P, W // P, P)
        imgs = imgs.permute(0, 2, 4, 3, 5, 1)  # (B, H/P, W/P, P, P, C)
        return imgs.reshape(B, self.num_patches, P * P * C)

# ---------------------------------------------------------------------------
# 3.  Lightning wrapper
# ---------------------------------------------------------------------------

class LitMAE(pl.LightningModule):
    def __init__(self, mask_ratio: float = 0.75, lr: float = 1.5e-4):
        super().__init__()
        self.net = MaskedAutoencoderViT(mask_ratio=mask_ratio)
        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        loss, _, _ = self(batch)
        # log training loss and learning rate
        self.log("train_loss", loss, prog_bar=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, _):
        loss, _, _ = self(batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.95))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sched}

# ---------------------------------------------------------------------------
# 4.  Minimal data pipeline (single CAM_FRONT)
# ---------------------------------------------------------------------------

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nuscenes_dataset import NuScenesDataset  # must be importable


def preprocess(img_np, hw=(224, 224)):
    img = torch.from_numpy(img_np.astype("float32") / 255.).permute(2, 0, 1)
    img = T.functional.resize(img, hw, interpolation=T.InterpolationMode.BICUBIC)
    return img  # **no normalisation** --- MAE uses raw pixels


class CollateFrontCam:
    def __init__(self, hw=(224, 224), cam: str = "CAM_FRONT"):
        self.hw = hw; self.cam = cam
    def __call__(self, samples):
        imgs = []
        for s in samples:
            imgs.append(preprocess(s["sensor_data"][self.cam]["img"], self.hw))
        return torch.stack(imgs, 0)

# ---------------------------------------------------------------------------
# 5.  Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb_project", default="mae_poc", help="W&B project name")
    ap.add_argument("--wandb_entity", default=None, help="W&B entity/user")
    ap.add_argument("--nusc_root", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=1.5e-4, help="Learning rate for optimizer")
    args = ap.parse_args()

    train_ds = NuScenesDataset(args.nusc_root, split="train", version="v1.0-trainval",
                               future_seconds=3, future_hz=1)
    val_ds = NuScenesDataset(args.nusc_root, split="val", version="v1.0-trainval",
                             future_seconds=3, future_hz=1)
    collate = CollateFrontCam()
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate)

    # Lightning model and logging
    model = LitMAE(mask_ratio=args.mask_ratio, lr=args.lr)
    # set up loggers: W&B
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name='mae_poc',
        save_dir='/scratch/gautschi/mgagvani/wandb',
        log_model=True
    )
    # log gradients, parameter histogram and model topology
    wandb_logger.watch(model, log='all')
    # also save best model checkpoint to scratch
    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss', mode='min', save_top_k=1,
        dirpath='/scratch/gautschi/mgagvani/wandb',
        filename='mae_poc_best-{epoch:02d}-{val_loss:.4f}'
    )
    callbacks = [checkpoint_cb]

    # define logger list
    logger = [wandb_logger]

    # Determine accelerator, devices, and strategy for multi-GPU
    import torch as _torch
    ngpus = _torch.cuda.device_count() if args.device.startswith('cuda') else 0
    accelerator = 'gpu' if ngpus > 0 else 'cpu'
    devices = ngpus if ngpus > 0 else None
    strategy = 'ddp' if ngpus > 1 else "auto"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    # Start training
    trainer.fit(model, train_dl, val_dl)

if __name__ == '__main__':
    main()
