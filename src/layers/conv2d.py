import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from .weight_noise import noise_fn


class RandConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, init_s=-1e10, stride=1, padding=0, dilation=1,
                 groups=1, bias=True):
        super(RandConv2d, self).__init__()
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2
            assert kernel_size[0] == kernel_size[1]
            kernel_size = kernel_size[0]
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.init_s = init_s
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self.mu_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.sigma_weight = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.mu_bias = Parameter(torch.Tensor(out_channels))
            self.sigma_bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('mu_bias', None)
            self.register_parameter('sigma_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        n *= self.kernel_size ** 2
        stdv = 1.0 / math.sqrt(n)
        self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(self.init_s)
        if self.mu_bias is not None:
            self.mu_bias.data.uniform_(-stdv, stdv)
            self.sigma_bias.data.fill_(self.init_s)

    def forward(self, input, sample=True):
        if not sample:
            out = F.conv2d(input, self.mu_weight, self.mu_bias, self.stride, self.padding, self.dilation, self.groups)
            return out

        eps_weight = torch.ones(self.sigma_weight.size(), device=input.device).normal_().type(self.sigma_weight.dtype)
        weight = self.mu_weight + torch.exp(self.sigma_weight) * eps_weight

        biasp = None
        if self.mu_bias is not None:
            eps_bias = torch.ones(self.sigma_bias.size(), device=input.device).normal_().type(self.sigma_bias.dtype)
            biasp = self.mu_bias + torch.exp(self.sigma_bias) * eps_bias

        out = F.conv2d(input, weight, biasp, self.stride, self.padding, self.dilation, self.groups)
        return out
