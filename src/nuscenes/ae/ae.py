"""ae.py – 3‑D VAE + Lightning **with true sliding‑window clips**
================================================================
Changes from previous version
----------------------------
1. **SequenceDataset** wrapper streams overlapping T‑frame clips: `[i‑T+1 … i]`.
2. **Collate** just stacks `(C,T,H,W)` that the wrapper already assembles – *no replication*.
3. All other model/training code untouched.

Usage
-----
```bash
python ae.py --seq_len 4 --stride 1 --epochs 10
```
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision.transforms.functional import resize as tv_resize
import argparse, os, sys

################################################################################
#                                Core VAE                                     #
################################################################################
class Conv3dEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256, hidden_dims=(32,64,128)):
        super().__init__()
        c_prev=in_channels; layers=[]
        for c in hidden_dims:
            layers+= [nn.Conv3d(c_prev,c,(3,4,4),stride=(1,2,2),padding=(1,1,1),bias=False),
                      nn.BatchNorm3d(c), nn.ReLU(inplace=True)]
            c_prev=c
        self.encoder=nn.Sequential(*layers)
        self.fc_mu, self.fc_logvar = nn.Linear(c_prev,latent_dim), nn.Linear(c_prev,latent_dim)
    def forward(self,x):
        h=self.encoder(x).mean(dim=[2,3,4])
        return self.fc_mu(h), self.fc_logvar(h)

class Conv3dDecoder(nn.Module):
    def __init__(self,out_channels=3,latent_dim=256,hidden_dims=(128,64,32),out_size=(4,180,320)):
        super().__init__(); T,H,W=out_size
        assert H%4==0 and W%4==0
        self.seed_shape=(hidden_dims[0],T,H//4,W//4)
        self.fc=nn.Linear(latent_dim,int(torch.prod(torch.tensor(self.seed_shape))))
        c_prev=hidden_dims[0]; layers=[]
        for c in hidden_dims[1:]:
            layers+=[nn.ConvTranspose3d(c_prev,c,(1,4,4),stride=(1,2,2),padding=(0,1,1),bias=False),
                     nn.BatchNorm3d(c), nn.ReLU(inplace=True)]
            c_prev=c
        layers.append(nn.Conv3d(c_prev,out_channels,3,padding=1))
        self.decoder=nn.Sequential(*layers)
    def forward(self,z):
        B=z.size(0); h=self.fc(z).view(B,*self.seed_shape)
        return torch.sigmoid(self.decoder(h))

class SpatioTemporalVAE(nn.Module):
    def __init__(self,in_channels=3,latent_dim=256,video_shape=(4,180,320)):
        super().__init__(); self.video_shape=video_shape
        self.encoder=Conv3dEncoder(in_channels,latent_dim)
        self.decoder=Conv3dDecoder(in_channels,latent_dim,out_size=video_shape)
    def reparameterise(self,mu,logvar):
        std=torch.exp(0.5*logvar); eps=torch.randn_like(std); return mu+eps*std
    def forward(self,x):
        mu,logvar=self.encoder(x); z=self.reparameterise(mu,logvar); recon=self.decoder(z)
        return z,mu,logvar,recon
    @staticmethod
    def loss_fn(x,recon,mu,logvar,beta=1.):
        rec=F.mse_loss(recon,x); kld=-0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
        return rec+beta*kld,rec,kld

################################################################################
#                           Lightning Module                                   #
################################################################################
class LitVAE(pl.LightningModule):
    def __init__(self, vae: SpatioTemporalVAE, lr=1e-4, beta=4., warmup_steps=5000):
        super().__init__()
        self.model = vae
        self.save_hyperparameters(ignore=['vae'])
        self.warmup_steps = warmup_steps
        self.beta_final = beta
        self.register_buffer("step", torch.tensor(0))

    def _current_beta(self):
        return min(1.0 * self.step / self.warmup_steps, 1.0) * self.beta_final

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        clip, _ = batch
        _, mu, logvar, recon = self.model(clip)
        beta = self._current_beta() if stage == 'train' else self.beta_final
        loss, rec, kld = self.model.loss_fn(clip, recon, mu, logvar, beta)
        log_dict = {f"{stage}_loss": loss, f"{stage}_rec": rec, f"{stage}_kld": kld}
        if stage == 'train':
            log_dict[f"{stage}_beta"] = beta
        self.log_dict(log_dict, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        self.step += 1
        return self._step(batch, 'train')

    def validation_step(self, batch, idx):
        self._step(batch, 'val')

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}

################################################################################
#                    Dataset wrappers & collate                                #
################################################################################

def preprocess_rgb(img_np:np.ndarray, hw:tuple[int,int]):
    tensor=torch.from_numpy(img_np.astype(np.float32)/255.).permute(2,0,1)
    return tv_resize(tensor, hw, interpolation=2)

class SequenceDataset(Dataset):
    """Wrap a *frame‑level* NuScenesDataset to yield sliding‑window clips.
    Args:
        base_ds: returns (sensor_data, trajectory) **for one frame**.
        seq_len: number of frames per clip (T).
        stride:  step between successive windows (e.g., 1 for [1‑4],[2‑5]…).
        cam:     camera key.
        hw:      resize target (H,W).
    """
    def __init__(self, base_ds:Dataset, seq_len:int=4, stride:int=1,
                 cam:str='CAM_FRONT', hw:tuple[int,int]=(180,320)):
        assert seq_len>0 and stride>0
        self.base, self.T, self.stride, self.cam, self.hw = base_ds, seq_len, stride, cam, hw
        self.max_start = len(base_ds) - seq_len
        if self.max_start < 0:
            raise ValueError("Dataset shorter than seq_len")
    def __len__(self):
        return (self.max_start // self.stride) + 1
    def __getitem__(self, idx):
        start = idx * self.stride
        clips = []
        # we also grab trajectory of **last** frame (can adapt as needed)
        for i in range(start, start+self.T):
            sensor_data, trajectory = self.base[i]
            img_np = sensor_data[self.cam]['img']
            clips.append(preprocess_rgb(img_np, self.hw))
        clip_tensor = torch.stack(clips, 1)  # (C,T,H,W)
        return clip_tensor, torch.from_numpy(trajectory).float()

# simple collate: stack tensors the wrapper already prepared
_collate = lambda batch:(torch.stack([b[0] for b in batch],0), torch.stack([b[1] for b in batch],0))

################################################################################
#                               CLI script                                     #
################################################################################
if __name__=='__main__':
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))
    from nuscenes_dataset import NuScenesDataset

    p=argparse.ArgumentParser()
    p.add_argument('--nusc_root'); p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch',type=int,default=4); p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--beta',type=float,default=4.0); p.add_argument('--cam',default='CAM_FRONT')
    p.add_argument('--seq_len',type=int,default=4); p.add_argument('--stride',type=int,default=1)
    p.add_argument('--hw',type=int,nargs=2,default=[80,160]); args=p.parse_args()

    frame_ds = NuScenesDataset(args.nusc_root, version="v1.0-trainval")  # yields single frames
    clip_ds  = SequenceDataset(frame_ds, seq_len=args.seq_len, stride=args.stride,
                               cam=args.cam, hw=tuple(args.hw))
    train_dl = DataLoader(clip_ds, batch_size=args.batch, shuffle=True, num_workers=8, collate_fn=_collate)
    val_dl   = DataLoader(clip_ds, batch_size=args.batch, shuffle=False,num_workers=8, collate_fn=_collate)

    vae = SpatioTemporalVAE(video_shape=(args.seq_len,*args.hw))
    lit = LitVAE(vae, lr=args.lr, beta=args.beta)

    pl.Trainer(max_epochs=args.epochs, precision='16-mixed', devices=-1, accelerator='gpu',
               log_every_n_steps=10, default_root_dir='/scratch/gautschi/mgagvani/runs/vae').fit(lit, train_dl, val_dl)

    lit.trainer.save_checkpoint('vae_nuscenes.ckpt')
