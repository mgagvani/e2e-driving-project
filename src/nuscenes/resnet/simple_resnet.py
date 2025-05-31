"""baseline_resnet_mlp.py
Simple trajectory baseline for NuScenes
============================================================================
* **Backbone:** pre -trained ResNet from timm (`resnet18` by default).
* **Head:** 2 -layer MLP that takes `[vision_features ‖ command]` → flattened way -points.
* **Loss:** L2 on way -points.  Metrics: mean L2 and collision -rate stub.

"""

from __future__ import annotations

import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


###############################################################################
#                               Model pieces                                  #
###############################################################################
class ResNetBackbone(nn.Module):
    """Wrapper around a frozen timm ResNet that outputs a **global** feature vector."""

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        out_dim: int | None = None,
    ):
        super().__init__()
        self.net = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.net.eval()  # keep in eval mode even during training (BN uses pre -trained stats)
        for p in self.net.parameters():
            p.requires_grad = False
        self.out_dim = out_dim or self.net.num_features

    def forward(self, x: torch.Tensor):  # x: (B,3,H,W)
        return self.net(x)  # (B, out_dim)


class WaypointMLP(nn.Module):
    """Very small MLP: (feat+cmd) → flattened way -points (N_pts×2)."""

    def __init__(self, in_dim: int, hidden_dim: int, num_waypoints: int):
        super().__init__()
        self.out_dim = num_waypoints * 2  # (x,y) per waypoint
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class ResNetMLPBaseline(nn.Module):
    """Full baseline: image → features; concat command → way -points."""

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        cmd_dim: int = 3,
        hidden_dim: int = 256,
        num_waypoints: int = 12,
    ):
        super().__init__()
        self.backbone = ResNetBackbone(model_name, pretrained)
        in_dim = self.backbone.out_dim + cmd_dim
        self.head = WaypointMLP(in_dim, hidden_dim, num_waypoints)
        self.num_waypoints = num_waypoints

    def forward(
        self, img: torch.Tensor, cmd: torch.Tensor
    ):  # img: (B,3,H,W)  cmd: (B,3)
        feat = self.backbone(img)  # (B, F)
        x = torch.cat([feat, cmd], dim=-1)  # (B, F+3)
        out = self.head(x)  # (B, N*2)
        return out.view(out.size(0), self.num_waypoints, 2)  # (B,N,2)


###############################################################################
#                             Lightning wrapper                               #
###############################################################################
class LitResNetMLP(pl.LightningModule):
    def __init__(self, model: ResNetMLPBaseline, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

        # give Lightning an example batch of (img, cmd)
        self.example_input_array = (
            torch.zeros(1, 3, 224, 224, dtype=torch.float32),  # a dummy image
            torch.zeros(1, 3, dtype=torch.float32),            # a dummy command
        )

    # --------------- metrics ---------------
    @staticmethod
    def l2_error(pred: torch.Tensor, gt: torch.Tensor):  # both (B,N,2)
        return torch.sqrt(
            ((pred - gt) ** 2).sum(dim=-1)
        ).mean()  # mean over pts & batch

    @staticmethod
    def collision_rate(pred: torch.Tensor) -> torch.Tensor:
        """Dummy placeholder - returns 0. Replace with proper collision check."""
        return torch.tensor(0.0, device=pred.device)
    
    # --------------- optimizers -----------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,  # Added weight decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # Reduce when validation loss plateaus
            factor=0.1,  # Reduce learning rate by a factor of 10
            patience=5,  # Number of epochs with no improvement after which learning rate will be reduced
            verbose=True,  # Print a message when the learning rate is reduced            
        )

        monitor = "val_loss"  # Monitor validation loss for scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": monitor,  # Specify the metric to monitor
            "interval": "epoch",  # Update every epoch
            "frequency": 1,  # Every epoch
            },
        }

    # --------------- forward / step ---------------
    def forward(self, img, cmd):
        return self.model(img, cmd)

    def _shared_step(self, batch, stage: str):
        img, cmd, gt_wp = batch  # img (B,3,H,W), cmd (B,3), gt (B,N,2)
        pred = self(img, cmd)
        l2 = self.l2_error(pred, gt_wp)
        loss = l2
        coll = self.collision_rate(pred)
    
        # Log metrics
        log_dict = {f"{stage}_loss": loss, f"{stage}_l2": l2, f"{stage}_coll": coll}
    
        # Log learning rate (only during training)
        if stage == "train":
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            log_dict["lr"] = current_lr
    
        self.log_dict(log_dict, prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        self._shared_step(batch, "val")


###############################################################################
#                     Minimal dataset/collate for baseline                     #
###############################################################################
# Assumes your NuScenesDataset (or a thin wrapper) returns:
#   img_rgb (H×W×3 uint8),  command (3,),  trajectory (N,3) in ego -frame.
# We drop Z coordinate and use (x,y).

from torchvision.transforms.functional import resize as tv_resize


def preprocess_img(img_np, hw=(224, 224)):
    img = torch.from_numpy(img_np.astype("float32") / 255.0).permute(2, 0, 1)  # (C,H,W)
    H, W = img.shape[1:]
    out_h, out_w = hw

    # Scale up output height to original image height
    scale = H / out_h
    crop_w = int(out_w * scale)
    crop_h = H

    # Center crop
    x1 = (W - crop_w) // 2
    y1 = 0
    img_cropped = img[:, y1:y1+crop_h, x1:x1+crop_w]

    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )

    img_normalized = normalize(img_cropped)  # Normalize to ImageNet stats

    # Resize cropped region to output size using bicubic interpolation
    return tv_resize(img_normalized, hw, interpolation=3)  # 3 = bicubic


class BaselineCollate:
    def __init__(self, hw=(224, 224), cam="CAM_FRONT", num_waypoints=6):
        self.hw = hw
        self.cam = cam
        self.num_waypoints = num_waypoints

    def __call__(self, samples):
        imgs, cmds, wps = [], [], []
        for sample in samples:
            sd = sample["sensor_data"]
            traj = sample["trajectory"]
            cmd = torch.from_numpy(sample["command"]).squeeze(0)
            img_np = sd[self.cam]["img"]
            imgs.append(preprocess_img(img_np, self.hw))
            cmds.append(cmd.to(dtype=torch.float32))
            wps.append(torch.tensor(traj[:self.num_waypoints, :2], dtype=torch.float32))
        return (torch.stack(imgs,0), torch.stack(cmds,0), torch.stack(wps,0))


def make_baseline_collate(hw=(224, 224), cam="CAM_FRONT", num_waypoints=6):
    return BaselineCollate(hw, cam, num_waypoints)


###############################################################################
#                             Quick standalone test                            #
###############################################################################
if __name__ == "__main__":
    import os
    import sys

    import numpy as np
    from torch.utils.data import DataLoader, Dataset

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from nuscenes_dataset import NuScenesDataset

    model = ResNetMLPBaseline(
        model_name="resnet34.a1_in1k",
        pretrained=True,
        cmd_dim=3,
        hidden_dim=512,
        num_waypoints=3,
    )
    lit = LitResNetMLP(model)

    p = argparse.ArgumentParser()
    p.add_argument("--nusc_root", default="/scratch/gautschi/mgagvani/nuscenes")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=4e-6)
    p.add_argument("--cam", default="CAM_FRONT")
    args = p.parse_args()

    train_dataset = NuScenesDataset(
        nuscenes_path=args.nusc_root,
        version="v1.0-trainval",
        future_seconds=3,
        future_hz=1,
        split="train",
    )
    test_dataset = NuScenesDataset(
        nuscenes_path=args.nusc_root,
        version="v1.0-trainval",
        future_seconds=3,
        future_hz=1,
        split="val",
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=16,
        collate_fn=make_baseline_collate(hw=(224, 224), cam=args.cam, num_waypoints=3),
        pin_memory=True,
        pin_memory_device="cuda",
        persistent_workers=True,
    )
    val_dl = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=16,
        collate_fn=make_baseline_collate(hw=(224, 224), cam=args.cam, num_waypoints=3),
        pin_memory=True,
        pin_memory_device="cuda",
        persistent_workers=True,
    )

    tb_logger = TensorBoardLogger(
        save_dir="/scratch/gautschi/mgagvani/runs",
        name="resnet_mlp",
        default_hp_metric=False,
        version=None,
        log_graph=True,
    )

    lit.hparams.lr = args.lr
    lit.hparams.weight_decay = 1e-2  # copied from ad_mlp, trainable head is same

    pl.Trainer(
        max_epochs=args.epochs,
        precision="32-true",
        devices=-1,
        accelerator="gpu",
        log_every_n_steps=10,
        default_root_dir="/scratch/gautschi/mgagvani/runs/resnet_e2e",
        logger= tb_logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                dirpath='./',  # save to current dir
                filename="resnet_mlp-{epoch:02d}-{val_loss:.3f}",
            ),
        ],
    ).fit(lit, train_dl, val_dl)

    # lit.trainer.save_checkpoint("resnet_mlp.ckpt")
