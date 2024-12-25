import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from transformers import AutoProcessor, SiglipVisionModel 
class PilotNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.nn6 = nn.Linear(1152, 100)
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