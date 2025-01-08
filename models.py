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
            nn.Linear(128*8*20, 512),  # Adjust based on final spatial dims
            self.act(),
            nn.Linear(512, 256),
            self.act(),
            nn.Linear(256, 64),
            self.act(),
            nn.Linear(64, 8)  # throttle, steer, w1_x, w1_y, w2_x, w2_y, w3_x, w3_y
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

