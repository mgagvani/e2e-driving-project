from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import timm

# ---------------------------------------------------------------------------
# 1.  MAE with TIMM-pretrained ViT encoder
# ---------------------------------------------------------------------------

# (Keep your imports at the top)

class MaskedAutoencoderViT(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        decoder_depth: int = 4,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_embed.proj.kernel_size[0]
        self.num_patches = (self.backbone.patch_embed.img_size[0] // self.patch_size) ** 2

        self.decoder_embed = nn.Linear(embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Properly register the decoder position embedding as a buffer
        # Don't clone directly - register a zero tensor first
        self.register_buffer("decoder_pos_embed", torch.zeros(1, self.num_patches + 1, embed_dim))
        
        n_heads = self.backbone.blocks[0].attn.num_heads
        decoder_layers = [
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
                activation='gelu', batch_first=True, norm_first=True,
            ) for _ in range(decoder_depth)
        ]
        self.decoder = nn.Sequential(*decoder_layers, nn.LayerNorm(embed_dim))
        self.decoder_pred = nn.Linear(embed_dim, self.patch_size * self.patch_size * 3)

        # init loss function (LPIPS)
        self.lpips_loss = LearnedPerceptualImagePatchSimilarity(
            normalize=True, 
            net_type="vgg", 
            reduction="mean"
        )
        self.lpips_factor = 0.1 # scaling factor

        self._init_decoder_weights()
        
        # Initialize decoder pos embed after model creation
        self._init_decoder_pos_embed()

    def _init_decoder_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.decoder_pred.weight, std=0.02)
        nn.init.zeros_(self.decoder_pred.bias)
    
    def _init_decoder_pos_embed(self):
        # Copy the backbone's positional embedding to decoder
        # This ensures they start on the same device
        with torch.no_grad():
            if hasattr(self.backbone, 'pos_embed') and self.backbone.pos_embed is not None:
                self.decoder_pos_embed.copy_(self.backbone.pos_embed)

    def random_masking(self, x: torch.Tensor):
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        
        # Create noise tensor on the same device as input
        noise = torch.rand(B, N, device=x.device, dtype=x.dtype)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        
        # Gather visible patches
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # Create mask on the same device
        mask = torch.ones([B, N], device=x.device, dtype=x.dtype)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def patchify(self, imgs: torch.Tensor):
        B, C, H, W = imgs.shape
        P = self.patch_size
        x = imgs.reshape(B, C, H // P, P, W // P, P).permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(B, self.num_patches, -1)
    
    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Unpatchify the patches back to images.
        Input: (B, num_patches, patch_size*patch_size*3)
        Output: (B, 3, H, W)
        """
        B = patches.shape[0]
        P = self.patch_size
        h = w = int(self.num_patches ** 0.5)
    
        # Reshape to (B, h, w, P, P, 3)
        x = patches.reshape(B, h, w, P, P, 3)
    
        # Permute to (B, 3, h, P, w, P)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    
        # Reshape to (B, 3, h*P, w*P) = (B, 3, H, W)
        imgs = x.reshape(B, 3, h * P, w * P)
        return imgs

    def forward(self, imgs: torch.Tensor):
        # Ensure input is on the right device (should be handled by Lightning)
        device = next(self.parameters()).device
        imgs = imgs.to(device)
        
        # Patch embedding
        x = self.backbone.patch_embed(imgs)
        B, N, D = x.shape
        
        # Add class token
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add positional embedding
        x = x + self.backbone.pos_embed
        
        # Random masking (remove class token for masking)
        x_visible, mask, ids_restore = self.random_masking(x[:, 1:, :])
        
        # Add class token back
        x_visible = torch.cat((cls_token, x_visible), dim=1)
        
        # Encoder forward pass
        encoded = self.backbone.norm(self.backbone.blocks(x_visible))
        
        # Decoder
        decoder_input = self.decoder_embed(encoded[:, 1:, :])  # Remove class token
        
        # Add mask tokens
        num_masked = self.num_patches - decoder_input.shape[1]
        
        # Ensure mask tokens are on the right device
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        
        # Concatenate visible and mask tokens
        dec_tokens_shuffled = torch.cat([decoder_input, mask_tokens], dim=1)
        
        # Unshuffle
        dec_tokens = torch.gather(
            dec_tokens_shuffled, 1, 
            ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )
        
        # Add positional embedding (make sure devices match)
        dec_tokens = dec_tokens + self.decoder_pos_embed[:, 1:, :].to(dec_tokens.device)
        
        # Decoder forward pass
        decoded = self.decoder(dec_tokens)
        pred = self.decoder_pred(decoded)
        
        # Compute loss
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        mse_loss = (loss * mask).sum() / mask.sum()

        # Compute LPIPS loss
        reconstructed_imgs = torch.clamp(self.unpatchify(pred), 0., 1.)
        lpips_loss = self.lpips_loss(reconstructed_imgs, imgs)
        total_loss = mse_loss + self.lpips_factor * lpips_loss
        
        return (total_loss, lpips_loss, mse_loss), pred, mask
# ---------------------------------------------------------------------------
# 2.  Lightning wrapper
# ---------------------------------------------------------------------------

class LitMAE(MaskedAutoencoderViT, pl.LightningModule):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        freeze_encoder: bool,
        mask_ratio: float,
        lr: float,
        # We need decoder_depth for the super().__init__ call
        decoder_depth: int = 4,
    ):
        # Call the MaskedAutoencoderViT constructor first
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            decoder_depth=decoder_depth,
            mask_ratio=mask_ratio,
        )
        # save_hyperparameters() will now save all args passed to this init
        self.save_hyperparameters()

        self.img_size = 224  # Assuming fixed image size for simplicity

        assert 0 < self.hparams.mask_ratio < 1, "mask_ratio must be in (0, 1)"

        if self.hparams.freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False

    # The forward method is now inherited directly from MaskedAutoencoderViT
    # We only need the step and optimizer logic here.

    def training_step(self, batch, batch_idx):
        imgs = batch
        # self(imgs) now calls the inherited forward method
        losses, _, _ = self(imgs)
        loss, lpips_loss, mse_loss = losses
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_lpips_loss", lpips_loss, prog_bar=True)
        self.log("train_mse_loss", mse_loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs = batch
        losses, _, _ = self(imgs)
        loss, lpips_loss, mse_loss = losses
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_lpips_loss", lpips_loss, prog_bar=True)
        self.log("val_mse_loss", mse_loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # self.parameters() now correctly gathers parameters from the inherited model
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-2
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.trainer.max_epochs
        )
        return {"optimizer": opt, "lr_scheduler": sched}

# ---------------------------------------------------------------------------
# 3.  Data pipeline
# ---------------------------------------------------------------------------

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from nuscenes_dataset import NuScenesDataset

def build_transforms(hw=(224,224)):
    return T.Compose([
        T.ToPILImage(), 
        T.Resize(hw, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])

class CollateFrontCam:
    def __init__(self, hw=(224,224), cam: str="CAM_FRONT"):
        self.transform = build_transforms(hw)
        self.cam = cam
    def __call__(self, samples):
        imgs = [self.transform(s["sensor_data"][self.cam]["img"]) for s in samples]
        return torch.stack(imgs,0)

# ---------------------------------------------------------------------------
# 4.  Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nusc_root", required=True)
    ap.add_argument("--backbone", default="vit_base_patch16_clip_224.laion2b_ft_in12k_in1k")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--mask_ratio", type=float, default=0.75)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    # datasets
    train_ds = NuScenesDataset(args.nusc_root, split="train", version="v1.0-trainval", future_seconds=3, future_hz=1)
    val_ds   = NuScenesDataset(args.nusc_root, split="val",   version="v1.0-trainval", future_seconds=3, future_hz=1)
    collate  = CollateFrontCam(hw=(224,224), cam="CAM_FRONT")
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate)

    # lightning
    model = LitMAE(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        freeze_encoder=args.freeze_encoder,
        mask_ratio=args.mask_ratio,
        lr=args.lr,
    )
    logger = WandbLogger(project="mae_poc", log_model=True)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1,
        dirpath="/scratch/gautschi/mgagvani/wandb/")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        logger=logger,
        callbacks=[checkpoint],
        log_every_n_steps=5,
        strategy=DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else None,
    )
    trainer.fit(model, train_dl, val_dl)

if __name__ == '__main__':
    main()
