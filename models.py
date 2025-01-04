import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from tqdm import tqdm
import pandas as pd
import timm

METADRIVE_OBS_FLATTENED_SIZE = 1152
FORZA_OBS_FLATTENED_SIZE = 2496

N_BINS = 5

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

class FeatureExtractorPilot(nn.Module):
    def __init__(self):
        super().__init__()

        feature_extractor = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)

        for param in feature_extractor.parameters():
            param.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # 1x1x1280

        flattened_size = 1280 * 1 * 1

        self.fc1 = nn.Linear(flattened_size, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 2) # throttle, steer

        self.feature_extractor = feature_extractor

    def forward(self, x):
        x = self.feature_extractor.forward_features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)

        return x

class ResNet18Pilot(nn.Module):
    def __init__(self):
        super().__init__()
        # get model - we want the feature extracto!
        self.resnet = models.resnet18(pretrained=True)
        # freeze.
        for layer in self.resnet.parameters():
            layer.requires_grad = False

        # set trainable layers
        for layer in self.resnet.layer4.parameters():
            layer.requires_grad = True    
        self.resnet.fc = nn.Linear(512, 2*N_BINS) # throttle, steer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
        
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 2, N_BINS) # (reshape to batchsize, throttle, steer)
        # apply softmax to throttle and steer, then return
        return F.softmax(x, dim=2) # (batchsize, 2, steer/throttle)