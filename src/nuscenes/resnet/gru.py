import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

# other modules
from simple_resnet import ResNetBackbone, LitResNetMLP, make_baseline_collate

class MonocularGRU(nn.Module):
    def __init__(self, feature_extractor="resnet18", gru_hidden_dim=512, mlp_hidden_dim=512, cmd_dim=3, n_wp=3):
        super().__init__()
        self.feature_extractor = ResNetBackbone(model_name=feature_extractor)

        # backbone - very simple 
        # features --> gru --> mlp --> output!

        self.gru = nn.GRU(
            input_size = self.feature_extractor.out_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden_dim+cmd_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, n_wp * 2)  # each wp is (x, y)
        )

    def forward(self, img: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        vision_feats = self.feature_extractor(img)  # (B, self.feature_extractor.out_dim)
        temporal_out, h_n = self.gru(vision_feats.unsqueeze(1))  # (B, 1, gru_hidden_dim)
        temporal = h_n[-1]  # (B, gru_hidden_dim)
        mlp_in_feats = torch.cat((temporal, cmd), dim=1)  # (B, gru_hidden_dim + cmd_dim)
        return self.mlp(mlp_in_feats) 
    
class LitMonocularGRU(pl.LightningModule):
    def __init__(self, model: MonocularGRU, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.model = model
        self.save_hyperparameters()  # Save hyperparameters for logging and checkpointing
        self.hparams.lr = lr
        self.hparams.weight_decay = weight_decay

        self.example_input_array = (
            torch.zeros(1, 3, 448, 448, dtype=torch.float32),  # a dummy image
            torch.zeros(1, 3, dtype=torch.float32),            # a dummy command
        )

        self.l2_error = LitResNetMLP.l2_error # same staticmethod

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

    def forward(self, img: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
        return self.model(img, cmd)
    
    def _shared_step(self, batch, stage: str):
        img, cmd, gt_wp = batch  # img (B,3,H,W), cmd (B,3), gt (B,N,2)
        pred = self(img, cmd)
        batch = pred.shape[0] 
        n_wp = gt_wp.shape[1]
        pred = pred.view(batch, n_wp, 2)  
        l2 = self.l2_error(pred, gt_wp)
        loss = l2
    
        # Log metrics
        log_dict = {f"{stage}_loss": loss, f"{stage}_l2": l2}
    
        # Log learning rate (only during training)
        if stage == "train":
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            log_dict["lr"] = current_lr
    
        self.log_dict(log_dict, prog_bar=True)
        return loss

    def training_step(self, batch, _):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, _):
        return self._shared_step(batch, "val")

if __name__ == "__main__":
    import os, sys, argparse
    import numpy as np
    from torch.utils.data import DataLoader

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from nuscenes_dataset import NuScenesDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--nusc_root", default="/scratch/gautschi/mgagvani/nuscenes")
    parser.add_argument("--backbone", default="timm/eva02_base_patch14_448.mim_in22k_ft_in22k_in1k")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gru_dim", type=int, default=512)
    parser.add_argument("--mlp_dim", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # we keep the number of waypoints fixed at 3 - one per second for the next 3 seconds
    train_ds = NuScenesDataset(args.nusc_root, split="train", version="v1.0-trainval", future_seconds=3, future_hz=1)
    val_ds = NuScenesDataset(args.nusc_root, split="val", version="v1.0-trainval", future_seconds=3, future_hz=1)

    collate = make_baseline_collate(hw=(448, 448), cam="CAM_FRONT", num_waypoints=3)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, 
                         collate_fn=collate, pin_memory=True, pin_memory_device="cuda", persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, 
                       collate_fn=collate, pin_memory=True, pin_memory_device="cuda", persistent_workers=True)
    
    model = MonocularGRU(feature_extractor=args.backbone, gru_hidden_dim=args.gru_dim, mlp_hidden_dim=args.mlp_dim, cmd_dim=3,
                         n_wp=3)
    lit = LitMonocularGRU(model, lr=args.lr, weight_decay=1e-4)

    project_name = "mono-gru"
    logger = WandbLogger(project=project_name, log_model=True)
    checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        dirpath="/scratch/gautschi/mgagvani/wandb"
    )
    strat = DDPStrategy(find_unused_parameters=True) if torch.cuda.device_count() > 1 else "auto"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        logger=logger,
        callbacks=[checkpoint],
        strategy=strat,
        log_every_n_steps=10,
    )
    trainer.fit(lit, train_dl, val_dl)


        




