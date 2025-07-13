import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, BatchSampler
import torchvision.transforms as T
from collections import deque

# other modules
from simple_resnet import ResNetBackbone, LitResNetMLP, make_baseline_collate

class MonocularGRU(nn.Module):
    def __init__(self, feature_extractor="resnet18", gru_hidden_dim=512, mlp_hidden_dim=512, cmd_dim=3, n_wp=3):
        super().__init__()
        self.feature_extractor = ResNetBackbone(model_name=feature_extractor)

        # backbone - very simple 
        # features --> gru --> mlp --> output!

        self.gru = nn.GRU(
            input_size = self.feature_extractor.out_dim + cmd_dim,  # ResNet output + command dimension
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, n_wp * 2)  # each wp is (x, y)
        )

        self.register_buffer("h", torch.zeros(1, 0, gru_hidden_dim)) 
        self.window = 10
        self.step = 0

    @torch.no_grad()
    def _resize_state(self, B, device):
        if self.h.size(1) != B:                         # ragged batch or first call
            self.h = torch.zeros(1, B, self.gru.hidden_size, device=device)

    @torch.no_grad()
    def _mask_beginnings(self, start):
        if not start.any():
            return

        # If start is multi-dimensional, reduce to batch dimension
        if len(start.shape) > 1:
            batch_mask = start.any(dim=1)
        else:
            batch_mask = start

        # Create an index tensor for the batch elements to reset
        if batch_mask.any():
            reset_indices = torch.nonzero(batch_mask).squeeze(1)
            self.h[:, reset_indices, :] = 0.0

    def forward(self, img: torch.Tensor, cmd: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        B, T = img.shape[:2]  # Batch size, Sequence length
        self._resize_state(B, img.device)
        self._mask_beginnings(start)
        
        # Process entire sequence but only keep the final output
        for t in range(T):
            if t % self.window == 0:
                self.h = self.h.detach()
            
            vision_feats = self.feature_extractor(img[:, t])
            x_t = torch.cat([vision_feats, cmd[:, t]], dim=-1).unsqueeze(1)
            out, self.h = self.gru(x_t, self.h)
        
        # Only return the final timestep's prediction
        return self.mlp(out.squeeze(1))  # (B, n_wp*2)
    
class WindowBatchSampler(BatchSampler):
    def __init__(self, dataset, seq_len=10, stride=1, drop_last=False):
        self.indices = list(range(len(dataset)))
        self.seq_len = seq_len
        self.stride  = stride
        self.drop_last = drop_last

    def __iter__(self):
        for start in range(0, len(self.indices) - self.seq_len + 1, self.stride):
            batch = self.indices[start:start + self.seq_len]
            yield batch

    def __len__(self):
        L = (len(self.indices) - self.seq_len) // self.stride + 1
        return L if not self.drop_last else max(L - 1, 0)
    
class SequenceCollate:
    def __init__(self, hw=(448, 448), cam="CAM_FRONT", num_wps=6):
        self.hw, self.cam, self.num_wps = hw, cam, num_wps
        self.normalize = T.Normalize(mean=[0.485,0.456,0.406],
                                     std=[0.229,0.224,0.225])

    def preprocess_img(self, img_np):
        img = torch.from_numpy(img_np.astype("float32") / 255.).permute(2,0,1)
        H,W = img.shape[1:]
        out_h,out_w = self.hw
        scale = H / out_h
        crop_w = int(out_w * scale)
        x1 = (W - crop_w)//2
        img = img[:, : , x1:x1+crop_w]                # center crop
        img = self.normalize(img)
        return T.functional.resize(img, self.hw, interpolation=3)

    def __call__(self, window_samples):
        """window_samples is length T list from BatchSampler"""
        imgs, cmds, wps, start_flags = [], [], [], []
    
        # mark first frame in window as episode start
        for i, sample_data in enumerate(window_samples):
            # Access sample dictionary without unpacking fixed number of values
            sample = sample_data  # If this is directly a dict
        
            # If sample_data is a tuple/list containing the dict as its last element
            if isinstance(sample_data, (tuple, list)):
                sample = sample_data[-1]
        
            sd = sample["sensor_data"]
            traj = sample["trajectory"]
            cmd = torch.from_numpy(sample["command"]).squeeze(0)
            img_np = sd[self.cam]["img"]

            imgs.append(self.preprocess_img(img_np))
            cmds.append(cmd)
            wps.append(torch.tensor(traj[:self.num_wps, :2], dtype=torch.float32))
            start_flags.append(i == 0)  # True only for first step

        # stack into (T,C,H,W) then move T to dim1 to get (1,T,C,H,W)
        imgs = torch.stack(imgs).unsqueeze(0)        # (1,T,C,H,W)
        cmds = torch.stack(cmds).unsqueeze(0)         # (1,T,3)
        wps = torch.stack(wps).unsqueeze(0)         # (1,T,N_wp,2)
        start = torch.tensor(start_flags).unsqueeze(0) # (1,T)
        return imgs, cmds, wps, start.bool()
    
class LitMonocularGRU(pl.LightningModule):
    def __init__(self, model: MonocularGRU, lr: float = 1e-3, weight_decay: float = 1e-4):
        super().__init__()
        self.model = model
        self.save_hyperparameters()  # Save hyperparameters for logging and checkpointing
        self.hparams.lr = lr
        self.hparams.weight_decay = weight_decay

        self.example_input_array = (
            torch.zeros(1, 1, 3, 448, 448, dtype=torch.float32),  # (B,T,C,H,W)
            torch.zeros(1, 1, 3, dtype=torch.float32),            # (B,T,3)
            torch.zeros(1, 1, dtype=torch.bool)                   # (B,T)
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

    def forward(self, img: torch.Tensor, cmd: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
        return self.model(img, cmd, start)
    
    def _shared_step(self, batch, stage: str):
        img, cmd, gt_wp, start = batch  # img: (B,T,C,H,W), cmd: (B,T,3), gt_wp: (B,T,N,2)
        pred = self(img, cmd, start)    # pred: (B, N*2)
        
        # Reshape prediction and get final ground truth waypoints
        pred = pred.view(-1, gt_wp.shape[2], 2)  # (B, N, 2)
        gt_final = gt_wp[:, -1, :, :]  # Last timestep's waypoints
        
        l2 = self.l2_error(pred, gt_final)
        loss = l2
    
        self.log_dict({
            f"{stage}_loss": loss,
            f"{stage}_l2": l2
        }, prog_bar=True)
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

    train_sampler = WindowBatchSampler(train_ds, seq_len=10, stride=8, drop_last=True)
    val_sampler = WindowBatchSampler(val_ds, seq_len=10, stride=8, drop_last=True)

    sequence_collate = SequenceCollate(hw=(448, 448), cam="CAM_FRONT", num_wps=3)

    train_dl = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=sequence_collate,
        num_workers=args.workers,
        pin_memory=True
    )

    val_dl = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        collate_fn=sequence_collate,
        num_workers=args.workers,
        pin_memory=True
    )
    
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
