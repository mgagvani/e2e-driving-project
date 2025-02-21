import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .weight_noise import noise_fn


class RandLinear(nn.Module):
    def __init__(self, in_features, out_features, init_s=1e-4, bias=True):
        super(RandLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_s = init_s
        self.mu_weight = Parameter(torch.Tensor(out_features, in_features))
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_features))
            self.sigma_bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('sigma_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mu_weight.size(1))
        self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(self.init_s)
        if self.mu_bias is not None:
            self.mu_bias.data.uniform_(-stdv, stdv)
            self.sigma_bias.data.fill_(self.init_s)

    def forward(self, input, sample=True):
        if not sample:
            out = F.linear(input, self.mu_weight, self.mu_bias)
            return out

        eps_weight = torch.randn_like(self.sigma_weight)
        weight = self.mu_weight + torch.exp(self.sigma_weight) * eps_weight

        biasp = None
        if self.mu_bias is not None:
            eps_bias = torch.randn_like(self.sigma_bias)
            biasp = self.mu_bias + torch.exp(self.sigma_bias) * eps_bias
        out = F.linear(input, weight, biasp)
        return out
