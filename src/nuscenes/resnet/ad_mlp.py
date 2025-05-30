import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class AD_MLP(nn.Module):
    """
    Reproduction of AD-MLP architecture from the paper:
    https://arxiv.org/pdf/2305.10430
    """

    def __init__(self, in_dim: int, out_dim: int = 6):
        '''
        Args:
            in_dim: Input dimension of the MLP. (9 + 3*future_steps)
            out_dim: Output dimension of the MLP, default is 6 (3 future steps with x,y coordinates).
        '''
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn(x)
    
class LitAD_MLP(pl.LightningModule):
    """
    Lightning module for training the AD_MLP model.
    """
    def __init__(self, model: AD_MLP, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.max_epochs = 100
        self.save_hyperparameters(ignore=["model"])

        # for log_graph
        self.in_dim = model.nn[0].in_features  # Input dimension of the first layer
        self.example_input_array = torch.zeros((1, self.in_dim), dtype=torch.float32)  # Example input for logging

    # --------------- metrics ---------------
    @staticmethod
    def l2_error(pred: torch.Tensor, gt: torch.Tensor):  # both (B,N,2)
        return torch.sqrt(
            ((pred - gt) ** 2).sum(dim=-1)
        ).mean()  # mean over pts & batch
    
    @staticmethod
    def l1_error(pred: torch.Tensor, gt: torch.Tensor): 
        return torch.abs(pred - gt).mean() 

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

        # The paper uses cosine annealing but I could not reproduce
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.max_epochs,
        #    eta_min=1e-8,  # Minimum learning rate
        # )
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, gt_wp = batch  # x: input features, gt_wp: ground truth waypoints (B,N,2)
        pred = self(x)
        # Reshape pred to match gt_wp if needed
        if pred.dim() == 2 and gt_wp.dim() == 3:
            pred = pred.view(pred.size(0), -1, 2)  # (B, N, 2)
        elif pred.dim() == 2 and gt_wp.dim() == 2:
            # Both are flattened, keep as is
            pass
        
        # Use L1 error as the loss function
        l1 = self.l1_error(pred, gt_wp)
        l2 = self.l2_error(pred, gt_wp)
        loss = l1  # Use L1 loss as the main loss function
        
        self.log_dict(
            {f"{stage}_loss": loss, f"{stage}_l1": l1, f"{stage}_l2": l2},
            prog_bar=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")


###############################################################################
#                                   Train                                     #
###############################################################################
if __name__ == "__main__":
    import os
    import sys

    import numpy as np
    from torch.utils.data import DataLoader, Dataset

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from nuscenes_dataset import NuScenesDataset

    # Create AD-MLP model
    # Input dimension: ego states (past_traj: 4*3=12, velocity: 3, acceleration: 3) + command: 3 = 21
    # Output dimension: 3 waypoints * 2 coordinates = 6
    model = AD_MLP(
        in_dim=21,  # 12 (past trajectory) + 3 (velocity) + 3 (acceleration) + 3 (command)
        out_dim=6   # 3 waypoints * 2 coordinates
    )
    lit = LitAD_MLP(model)

    # Update the model instantiation and arguments
    p = argparse.ArgumentParser()
    p.add_argument("--nusc_root", default="/scratch/gautschi/mgagvani/nuscenes")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=4e-6)  # Updated to match paper
    p.add_argument("--weight_decay", type=float, default=1e-2)  # Added weight decay
    p.add_argument("--num_waypoints", type=int, default=3)
    args = p.parse_args()

    # Log hyperparameters
    print(f"Model: AD-MLP")
    print(f"Input dim: 21 (ego states + command)")
    print(f"Output dim: {args.num_waypoints * 2}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Batch size: {args.batch}")

    train_dataset = NuScenesDataset(
        nuscenes_path=args.nusc_root,
        version="v1.0-trainval",
        future_seconds=3,
        future_hz=1,
        past_frames=4,
        split="train",
        get_img_data=False,  # No images needed for AD-MLP
    )
    
    val_dataset = NuScenesDataset(
        nuscenes_path=args.nusc_root, 
        version="v1.0-trainval",
        future_seconds=3,
        future_hz=1,
        split="val",
        get_img_data=False
    )

    # Custom collate function for AD-MLP (ego states + command → waypoints)
    def ad_mlp_collate(samples):
        features, wps = [], []
        for sample in samples:
            # Extract ego trajectory, velocity, acceleration, and command
            ego_traj = torch.tensor(sample["ego_trajectory"], dtype=torch.float32)        # (past_frames, 3)
            velocity = torch.tensor(sample["ego_velocity"], dtype=torch.float32)          # (3,)
            acceleration = torch.tensor(sample["ego_acceleration"], dtype=torch.float32)  # (3,)
            cmd = torch.from_numpy(sample["command"]).squeeze(0).float()                  # (3,)
            traj = sample["trajectory"]  # numpy array (N, 3)
            
            # Flatten past trajectory: (past_frames, 3) → (past_frames * 3,)
            past_traj_flat = ego_traj.flatten()
            
            # Concatenate features: past_traj + velocity + acceleration + command
            feature_vec = torch.cat([past_traj_flat, velocity, acceleration, cmd], dim=0)
            features.append(feature_vec)
            
            # Waypoints: first num_waypoints and xy coords
            wps.append(torch.tensor(traj[:args.num_waypoints, :2], dtype=torch.float32))
        
        return (torch.stack(features, 0), torch.stack(wps, 0))

    # if num_workers == batch size, i think it'll reduce time waiting
    train_dl = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=14,
        collate_fn=ad_mlp_collate,
        persistent_workers=True,
        prefetch_factor=3,
        pin_memory=True,
        pin_memory_device="cuda",
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False, # we want to be consistent.
        num_workers=14,
        persistent_workers=True,
        prefetch_factor=3,
        collate_fn=ad_mlp_collate,
        pin_memory=True,
        pin_memory_device="cuda",
    )

    tb_logger = TensorBoardLogger(
        save_dir='/scratch/gautschi/mgagvani/runs/ad_mlp_e2e',
        name='logs',
        version=None,
        log_graph=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision="32-true",
        devices=-1,
        accelerator="gpu",
        log_every_n_steps=10,
        default_root_dir="/scratch/gautschi/mgagvani/runs/ad_mlp_e2e",
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                dirpath='./', # save to current dir
                filename="ad_mlp-{epoch:02d}-{val_loss:.3f}",
            ),
        ],
    )
    
    # Update model with correct learning rate
    lit.hparams.lr = args.lr
    lit.hparams.weight_decay = args.weight_decay
    
    trainer.fit(lit, train_dl, val_dl)

