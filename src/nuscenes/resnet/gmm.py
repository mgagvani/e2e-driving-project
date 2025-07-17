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
import math

from simple_resnet import ResNetBackbone

class WaypointMDN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_wp: int, K: int):
        super().__init__()
        self.K, self.num_wp = K, num_wp
        self.param_dim = 6 * K                 # pi, mu, sigma, rho (packed together)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.param_dim * num_wp),
        )

    def forward(self, x):
        B = x.size(0)
        raw = self.mlp(x)                      # (B, num_wp*6K)
        raw = raw.view(B, self.num_wp, self.K, 6)

        logit_pi   = raw[..., 0]               # (B,N,K)
        mu         = raw[..., 1:3]             # (B,N,K,2)
        log_sigma  = raw[..., 3:5]             # (B,N,K,2)
        rho_raw    = raw[..., 5]               # (B,N,K)

        pi   = torch.softmax(logit_pi, dim=-1)
        sigma= torch.exp(log_sigma)
        rho  = torch.tanh(rho_raw)

        return {"pi": pi, "mu": mu, "sigma": sigma, "rho": rho}

class LitWaypointMDN(pl.LightningModule):
    def __init__(self, cmd_dim=3, num_wp=3, hidden=256, K=5, lr=2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = ResNetBackbone()
        self.head = WaypointMDN(self.backbone.out_dim + cmd_dim,
                                hidden, num_wp, K)
        
    @staticmethod
    def mdn_log_prob(params, y):
        pi, mu, sigma, rho = params["pi"], params["mu"], params["sigma"], params["rho"]
        y = y.unsqueeze(2)             # (B,N,1,2)
        diff = y - mu
        sx, sy = sigma[..., 0], sigma[..., 1]
        z = (diff[..., 0]/sx)**2 + (diff[..., 1]/sy)**2 - 2*rho*diff[..., 0]*diff[..., 1]/(sx*sy)
        denom = 2*(1 - rho**2)
        log_gauss = -torch.log(2*math.pi) - torch.log(sx*sy) - 0.5*torch.log(1 - rho**2) - 0.5*z/denom
        return torch.logsumexp(torch.log(pi) + log_gauss, dim=-1)  # (B,N)
    
    def forward(self, img, cmd):
        feat = self.backbone(img)
        x = torch.cat([feat, cmd], -1)
        return self.head(x)

    def step(self, batch, stage):
        img, cmd, gt_wp = batch 
        params = self(img, cmd)
        nll = -self.mdn_log_prob(params, gt_wp).mean()
        self.log(f"{stage}_nll", nll, prog_bar=True)
        return nll

    def training_step(self, batch, _):
        return self.step(batch, "train")

    def validation_step(self, batch, _):
        self.step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)