import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from transformers import AutoProcessor, SiglipVisionModel 

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
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activ = nn.SiLU()
        
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = self.activ(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        return self.activ(out)
    
class SuperPilot(nn.Module):
    def __init__(self, drop=0.3):
        super().__init__()
        self.activ = nn.SiLU
        self.drop = drop
        
        # Feature extraction with residual blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.BatchNorm2d(24),
            self.activ()
        )
        
        self.res1 = ResBlock(24, 36, stride=2)
        self.res2 = ResBlock(36, 48, stride=2)
        self.res3 = ResBlock(48, 64)
        self.res4 = ResBlock(64, 64)
        
        # Global average pooling instead of max pooling
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(self.drop)
        )
        
        # Control output block with wider layers
        self.control = nn.Sequential(
            nn.Linear(64, 256),
            self.activ(),
            nn.Dropout(self.drop),
            nn.Linear(256, 128),
            self.activ(),
            nn.Dropout(self.drop),
            nn.Linear(128, 64),
            self.activ(),
            nn.Linear(64, 2)  # throttle, steer
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.pool(x)
        x = self.control(x)
        return x

class SigLIPPilot(nn.Module):
    def __init__(self):
        super().__init__()

        # init vision model
        # (this is extremely overpowered)
        model_id = "google/siglip-so400m-patch14-384"
        self.vision_model = SiglipVisionModel.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)

        # init control model
        self.nn6 = nn.Linear(1152, 100) # 1152 is the output embedding size
        self.nn7 = nn.Linear(100, 50)
        self.nn8 = nn.Linear(50, 10)
        self.nn9 = nn.Linear(10, 2) # throttle, steer
    
    def forward(self, x):   
        # TODO: implement batch. embeddings are weird when batched

        # get vision embeddings
        vision_output = self.model(
            **self.processor(images=x, return_tensors="pt").to("cuda")
        ).pooler_output

        y = F.relu(self.nn6(vision_output))
        y = F.relu(self.nn7(y))
        y = F.relu(self.nn8(y))
        y = self.nn9(y)

        return y