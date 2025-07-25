"""baseline_resnet_mlp.py
Simple trajectory baseline for NuScenes
============================================================================
* **Backbone:** pre -trained ResNet from timm (`resnet18` by default).
* **Head:** 2 -layer MLP that takes `[vision_features ??? command]` ??? flattened way -points.
* **Loss:** L2 on way -points.  Metrics: mean L2 and collision -rate stub.

"""

from __future__ import annotations

import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
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
    """Very small MLP: (feat+cmd) ??? flattened way -points (N_pts??2)."""

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
    """Full baseline: image ??? features; concat command ??? way -points."""

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

class SimpleMultiCam(nn.Module):
    """multi camera"""

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        cmd_dim: int = 3,
        hidden_dim: int = 256,
        num_waypoints: int = 12,
        ncams: int = 6
    ):
        super().__init__()
        self.backbone = ResNetBackbone(model_name, pretrained) # shared across cameras since its frozen
        in_dim = self.backbone.out_dim * ncams + cmd_dim
        self.head = WaypointMLP(in_dim, hidden_dim, num_waypoints)
        self.num_waypoints = num_waypoints

    def forward(
        self, imgs: torch.Tensor, cmd: torch.Tensor
    ):
        """imgs: (B, ncams, 3, H, W)  cmd: (B,3)"""
        B, ncams, C, H, W = imgs.shape
        imgs = imgs.view(B * ncams, C, H, W)
        feat = self.backbone(imgs)  # (B*ncams, F)
        feat = feat.view(B, ncams, -1).view(B, -1)  # (B, ncams*F)
        x = torch.cat([feat, cmd], dim=-1)  # (B, ncams*F+3)
        out = self.head(x)  # (B, N*2)
        return out.view(out.size(0), self.num_waypoints, 2)  # (B,N,2)


class MoE(nn.Module):
    """Mixture-of-Experts model with one expert per camera"""
    def __init__(self,
                 backbone_name: str = "resnet18",
                 pretrained: bool = True,
                 cmd_dim: int = 3,
                 hidden_dim: int = 256,
                 num_waypoints: int = 12,
                 ncams: int = 6):
        super().__init__()
        # one expert per camera
        self.experts = nn.ModuleList([
            ResNetBackbone(backbone_name, pretrained) for _ in range(ncams)
        ])
        feature_dim = self.experts[0].out_dim
        # gating network consumes command to produce weights per expert
        self.gating = nn.Sequential(
            nn.Linear(cmd_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, ncams),
        )
        # final head: fused features + command -> waypoints
        self.head = WaypointMLP(feature_dim + cmd_dim, hidden_dim, num_waypoints)
        self.num_waypoints = num_waypoints

    def forward(self, imgs: torch.Tensor, cmd: torch.Tensor):
        # imgs: (B, ncams, 3, H, W), cmd: (B, cmd_dim)
        B, ncams, C, H, W = imgs.shape
        # extract features per expert
        feats = []
        for i, expert in enumerate(self.experts):
            feats.append(expert(imgs[:, i]))  # (B, feature_dim)
        feats = torch.stack(feats, dim=1)      # (B, ncams, feature_dim)
        # compute gating weights
        gate_logits = self.gating(cmd)         # (B, ncams)
        gates = torch.softmax(gate_logits, dim=1).unsqueeze(-1)  # (B, ncams, 1)
        # fuse features
        fused = (feats * gates).sum(dim=1)     # (B, feature_dim)
        # combine with command
        x = torch.cat([fused, cmd], dim=-1)    # (B, feature_dim+cmd_dim)
        # predict waypoints
        out = self.head(x)                     # (B, num_waypoints*2)
        # reshape flattened waypoints to (B, N, 2)
        return out.view(out.size(0), self.num_waypoints, 2)
        


###############################################################################
#                             Lightning wrapper                               #
###############################################################################
class LitResNetMLP(pl.LightningModule):
    def __init__(self, model: ResNetMLPBaseline, lr: float = 1e-3, weight_decay: float = 1e-2):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

        # give Lightning an example batch of (img, cmd)
        self.example_input_array = (
            torch.zeros(1, 3, 448, 448, dtype=torch.float32),  # a dummy image
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
            # verbose=True,  # Print a message when the learning rate is reduced            
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

class LitSimpleMultiCam(LitResNetMLP):
    """Lightning wrapper for the multi-camera ResNetMLP baseline."""
    def __init__(self, model: SimpleMultiCam, lr: float = 1e-3, weight_decay: float = 1e-2):
        super().__init__(model, lr)
        # Override hyperparameters to include weight_decay
        self.save_hyperparameters(ignore=["model"])
        self.example_input_array = (
            torch.zeros(1, 6, 3, 448, 448, dtype=torch.float32),  # a dummy image batch
            torch.zeros(1, 3, dtype=torch.float32),                # a dummy command
        )
    
    def forward(self, imgs, cmd):
        return self.model(imgs, cmd)
    
    def _shared_step(self, batch, stage: str):
        imgs, cmd, gt_wp = batch  # imgs (B,ncams,3,H,W), cmd (B,3), gt (B,N,2)
        pred = self(imgs, cmd)
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


class LitMoE(LitSimpleMultiCam):
    """Lightning wrapper for the Mixture-of-Experts model"""
    def __init__(self, model: MoE, lr: float = 1e-3, weight_decay: float = 1e-2):
        super().__init__(model, lr)
        # Override hyperparameters to include weight_decay
        self.save_hyperparameters(ignore=["model"])
        
    def configure_optimizers(self):
        # Use CosineAnnealingLR instead of ReduceLROnPlateau for MoE
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


###############################################################################
#                     Minimal dataset/collate for baseline                     #
###############################################################################
# Assumes your NuScenesDataset (or a thin wrapper) returns:
#   img_rgb (H??W??3 uint8),  command (3,),  trajectory (N,3) in ego -frame.
# We drop Z coordinate and use (x,y).

from torchvision.transforms.functional import resize as tv_resize


def preprocess_img(img_np, hw=(448, 448)):
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
        std=[0.229, 0.448, 0.225]
    )

    img_normalized = normalize(img_cropped)  # Normalize to ImageNet stats

    # Resize cropped region to output size using bicubic interpolation
    return tv_resize(img_normalized, hw, interpolation=3)  # 3 = bicubic


class BaselineCollate:
    def __init__(self, hw=(448, 448), cam="CAM_FRONT", num_waypoints=6):
        self.hw = hw
        self.cam = cam
        self.num_waypoints = num_waypoints

    def __call__(self, samples):
        imgs, cmds, wps, scene_starts = [], [], [], []
        for sample in samples:
            sd = sample["sensor_data"]
            traj = sample["trajectory"]
            cmd = torch.from_numpy(sample["command"]).squeeze(0)
            img_np = sd[self.cam]["img"]
            scene_start = sample.get("scene_start", False)
            imgs.append(preprocess_img(img_np, self.hw))
            cmds.append(cmd)
            wps.append(torch.tensor(traj[:self.num_waypoints, :2], dtype=torch.float32))
            scene_starts.append(scene_start)
        return (
            torch.stack(imgs, 0),
            torch.stack(cmds, 0),
            torch.stack(wps, 0),
            torch.tensor(scene_starts, dtype=torch.bool),
        )


def make_baseline_collate(hw=(448, 448), cam="CAM_FRONT", num_waypoints=6):
    return BaselineCollate(hw, cam, num_waypoints)

class MultiCamCollate:
    def __init__(self, hw=(448, 448), cams=None, num_waypoints=6):
        self.hw = hw
        # default to typical 6 NuScenes cameras if none provided
        self.cams = cams if cams is not None else [
            "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"
        ]
        self.num_waypoints = num_waypoints

    def __call__(self, samples):
        imgs, cmds, wps = [], [], []
        for sample in samples:
            sd = sample["sensor_data"]
            traj = sample["trajectory"]
            cmd = torch.from_numpy(sample["command"]).squeeze(0).to(dtype=torch.float32)
            # collect and preprocess each camera image
            cam_imgs = []
            for cam in self.cams:
                img_np = sd[cam]["img"]
                cam_imgs.append(preprocess_img(img_np, self.hw))
            # stack cams dimension: (ncams,3,H,W)
            imgs.append(torch.stack(cam_imgs, dim=0))
            wps.append(torch.tensor(traj[:self.num_waypoints, :2], dtype=torch.float32))
            cmds.append(cmd)
        # batch stack: imgs (B,ncams,3,H,W), cmds (B,3), wps (B,N,2)
        return (torch.stack(imgs, 0), torch.stack(cmds, 0), torch.stack(wps, 0))


def make_multicam_collate(hw=(448, 448), cams=None, num_waypoints=6):
    return MultiCamCollate(hw, cams, num_waypoints)


###############################################################################
#                             Quick standalone test                            #
###############################################################################
if __name__ == "__main__":
    import os
    import sys
    import argparse
    import numpy as np
    from torch.utils.data import DataLoader

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from nuscenes_dataset import NuScenesDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["simple", "moe"], default="simple", help="Model type to train")
    parser.add_argument("--nusc_root", required=True)
    parser.add_argument("--backbone", default="timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_waypoints", type=int, default=12)
    parser.add_argument("--ncams", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cams", nargs='+', default=None, help="list of camera names for multicam collate")
    args = parser.parse_args()

    # Create datasets
    train_ds = NuScenesDataset(args.nusc_root, split="train", version="v1.0-trainval", future_seconds=3, future_hz=1)
    val_ds = NuScenesDataset(args.nusc_root, split="val", version="v1.0-trainval", future_seconds=3, future_hz=1)
    
    collate = make_multicam_collate(hw=(448, 448), cams=args.cams, num_waypoints=args.num_waypoints)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, 
                         collate_fn=collate, pin_memory=True, pin_memory_device="cuda", persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, 
                       collate_fn=collate, pin_memory=True, pin_memory_device="cuda", persistent_workers=True)

    # Create model based on choice
    if args.model == "simple":
        model = SimpleMultiCam(
            model_name=args.backbone,
            pretrained=args.pretrained,
            cmd_dim=3,
            hidden_dim=args.hidden_dim,
            num_waypoints=args.num_waypoints,
            ncams=args.ncams,
        )
        lit = LitSimpleMultiCam(model, lr=args.lr)
        project_name = "simple_multicam"
    else:  # moe
        model = MoE(
            backbone_name=args.backbone,
            pretrained=args.pretrained,
            cmd_dim=3,
            hidden_dim=args.hidden_dim,
            num_waypoints=args.num_waypoints,
            ncams=args.ncams,
        )
        lit = LitMoE(model, lr=args.lr)
        project_name = "moe_poc"

    # Setup logging and training
    logger = WandbLogger(project=project_name, log_model=True)
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath="/scratch/gautschi/mgagvani/wandb"
    )
    strategy = DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else None

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        logger=logger,
        callbacks=[checkpoint],
        strategy=strategy,
        log_every_n_steps=10,
    )
    trainer.fit(lit, train_dl, val_dl)

