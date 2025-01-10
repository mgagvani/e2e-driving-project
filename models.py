import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

METADRIVE_OBS_FLATTENED_SIZE = 1152
FORZA_OBS_FLATTENED_SIZE = 2496

class PilotNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.nn6 = nn.Linear(FORZA_OBS_FLATTENED_SIZE, 100)
        self.nn7 = nn.Linear(100, 50)
        self.nn8 = nn.Linear(50, 10)
        self.nn9 = nn.Linear(10, 2) # throttle, steer

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) 
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.nn6(x))
        x = F.relu(self.nn7(x))
        x = F.relu(self.nn8(x))
        x = self.nn9(x)

        return x
    

class MegaPilotNet(nn.Module):
    def __init__(self, drop=0.5):
        super().__init__()

        self.activ = nn.SiLU
        self.drop = drop
        
        # Feature extraction 
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            self.activ(),
            nn.Conv2d(24, 36, 5, stride=2),
            self.activ(),
            nn.Conv2d(36, 48, 5, stride=2),
            self.activ(),
            nn.Conv2d(48, 64, 3),
            self.activ(),
            nn.Conv2d(64, 64, 3),
            self.activ()
        )
        
        # Pooling and flatten 
        self.pool = nn.Sequential(
            nn.MaxPool2d(2, 2), # 2x2 pooling
            nn.Flatten(),
            nn.Dropout(self.drop)
        )
        
        # Control output block
        self.control = nn.Sequential(
            nn.Linear(384, 100),
            self.activ(),
            nn.Linear(100, 50),
            self.activ(),
            nn.Linear(50, 10),
            self.activ(),
            nn.Linear(10, 2)  # throttle, steer
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.control(x)
        return x

class MultiCamWaypointNet(nn.Module):
    def __init__(self, drop=0.5):
        super().__init__()
        self.act = nn.ReLU
        self.drop = drop

        # For input shape: (3, 168, 3*224)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            self.act(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            self.act(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3),
            self.act(),
            nn.Conv2d(128, 128, kernel_size=3),
            self.act(),
            nn.MaxPool2d(2, 2)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(self.drop),
            nn.Linear(34944, 512),  # Adjust based on final spatial dims
            self.act(),
            nn.Linear(512, 256),
            self.act(),
            nn.Linear(256, 64),
            self.act(),
            nn.Linear(64, 8)  # steer, throttle, w1_x, w1_y, w2_x, w2_y, w3_x, w3_y
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class SplitCamWaypointNet(nn.Module):
    def __init__(self, drop=0.5):
        super().__init__()
        self.act = nn.LeakyReLU
        self.drop = drop

        # 3 feature extractors for each cam (shape 3, 168, 224)
        def make_extractor():
            return nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2),
                self.act(),
                nn.Conv2d(32, 64, kernel_size=5, stride=2),
                self.act(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3),
                self.act(),
                nn.Conv2d(128, 128, kernel_size=3),
                self.act(),
                nn.MaxPool2d(2, 2),
                nn.Flatten()
            )
        self.extractor_left = make_extractor()
        self.extractor_center = make_extractor()
        self.extractor_right = make_extractor()

        # Heads: each tries to regress 8 outputs (steer, throttle, w1_x, w1_y, w2_x, w2_y, w3_x, w3_y)
        def make_head(in_features=9856): 
            return nn.Sequential(
                nn.Dropout(self.drop),
                nn.Linear(in_features, 256),
                self.act(),
                nn.Linear(256, 8)
            )
        self.head_left = make_head()
        self.head_center = make_head()
        self.head_right = make_head()

        # Gating network to produce weights for each camera
        # We'll form a vector [feat_left, feat_center, feat_right] for gating input
        self.gate_fc = nn.Sequential(
            nn.Linear(3 * 9856, 3),    # produce 3 gating logits
        )

        # Final FC for the weighted sum of extracted features
        self.final_fc = nn.Sequential(
            nn.Dropout(self.drop),
            nn.Linear(9856, 256),
            self.act(),
            nn.Linear(256, 64),
            self.act(),
            nn.Linear(64, 8)
        )

    def forward(self, x, return_split_heads=True, return_gates=True):
        # x shape: (B, 3, 168, 3*224)
        # Split into left, center, right chunks (width 224 each)
        x_left   = x[:, :, :, 0:224]
        x_center = x[:, :, :, 224:448]
        x_right  = x[:, :, :, 448:]

        feat_left   = self.extractor_left(x_left)
        feat_center = self.extractor_center(x_center)
        feat_right  = self.extractor_right(x_right)

        # Each head's raw predictions
        pred_left = self.head_left(feat_left)
        pred_center = self.head_center(feat_center)
        pred_right = self.head_right(feat_right)

        # Gating
        concat_feats = torch.cat([feat_left, feat_center, feat_right], dim=1)   # B x (3*feat_dim)
        gating_logits = self.gate_fc(concat_feats)                             # B x 3
        gating_weights = F.softmax(gating_logits, dim=1)                       # B x 3

        # Weighted sum of features
        # Expand gating_weights to match feature dims (B x 3 x feat_dim)
        gating_weights_expanded = gating_weights.unsqueeze(-1).expand(-1, -1, feat_left.size(1))
        stacked_feats = torch.stack([feat_left, feat_center, feat_right], dim=1)  # B x 3 x feat_dim
        fused_feats = (gating_weights_expanded * stacked_feats).sum(dim=1)       # B x feat_dim

        # Final FC
        out = self.final_fc(fused_feats)  # shape (B, 8)

        # Optionally return heads' predictions and gating weights
        if return_split_heads or return_gates:
            results = [out]
            if return_split_heads:
                results += [pred_left, pred_center, pred_right]
            if return_gates:
                results += [gating_weights]
            return results

        return out



