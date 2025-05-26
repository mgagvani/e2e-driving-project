from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(inputs)
        weights = F.softmax(gate_logits, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            expert_output = expert(inputs)
            results += weights[:, i, None] * expert_output

        return results
    
class TinyWaypointNet(nn.Module):
    """
    TinyWaypointNets are the experts in the MoE structure.
    Based on the autoencoder output, they are inferenced on a camera.
    Then the outputs are combined with a weighted sum.
    """
    def __init__(self):
        super().__init__()

        
