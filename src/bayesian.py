import copy
import math
import argparse

import torch
import torch.nn as nn
from torch.nn import BatchNorm2d

from src.layers.conv2d import RandConv2d
from src.layers.linear import RandLinear

convert_params = 0.0
overall_params = 0.0
kl_count = 0
module_count = 0


def set_sigma_module_for_unet(module, sigma_blocks):
    """
    Args:
        module: nn.Module, current module
        sigma_blocks: a list as shape [a, b, c] where a, b, c are sigma for upblock, midblock and downblock
    Returns:
        new module with same structure as module and parameter as corresponding sigma
    """
    new_module = copy.deepcopy(module)
    for i, key in enumerate(new_module._modules):
        print(key, i)
        for param in new_module._modules[key].parameters():
            with torch.no_grad():
                param.fill_(sigma_blocks[i])
    return new_module


def add_bayesian_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--convert_conv",
        action="store_false",
        dest="skip_conv",
        default=True
    )
    parser.add_argument(
        "--convert_linear",
        action="store_false",
        dest="skip_linear",
        default=True
    )
    parser.add_argument(
        "--bayes_only",
        action="store_true",
    )
    parser.add_argument(
        "--init_sigma",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--prior_sigma",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--lambda1",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--init_mu_from_module",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--test_sigma",
        type=float,
        default=-1.0
    )
    parser.add_argument(
        "--samplings",
        type=int,
        default=1
    )


def convert_with_config(module, args):
    return convert(module, init_sigma=args.init_sigma, init_mu_from_module=args.init_mu_from_module,
                   skip_Conv=args.skip_conv, skip_Linear=args.skip_linear)


def convert(module, init_sigma=0.02, init_mu_from_module=True, **kwargs):
    global convert_params, overall_params, module_count
    is_base = not any(module.children())
    if is_base:
        overall_params += sum(p.numel() for p in module.parameters())
        if isinstance(module, torch.nn.Conv2d):
            if kwargs.get('skip_Conv'):
                print(f'Skipping Conv, params: {sum(p.numel() for p in module.parameters())}')
                return copy.deepcopy(module)
            bayesian_module = RandConv2d(in_channels=module.in_channels, out_channels=module.out_channels,
                                         kernel_size=module.kernel_size, stride=module.stride, padding=module.padding,
                                         dilation=module.dilation, groups=module.groups, bias=module.bias is not None,
                                         init_s=init_sigma)
            module_count += 1
            convert_params += sum(p.numel() for p in module.parameters())
        elif isinstance(module, torch.nn.Linear):
            if kwargs.get('skip_Linear'):
                print(f'Skipping Linear, params: {sum(p.numel() for p in module.parameters())}')
                return copy.deepcopy(module)
            bayesian_module = RandLinear(in_features=module.in_features, out_features=module.out_features,
                                         bias=module.bias is not None,
                                         init_s=init_sigma)
            module_count += 1
            convert_params += sum(p.numel() for p in module.parameters())
        else:
            return copy.deepcopy(module)  # not a layer to be converted into Bayesian
        if init_mu_from_module:
            bayesian_module.mu_weight.data.copy_(module.weight.data)
            if module.bias is not None:
                bayesian_module.mu_bias.data.copy_(module.bias.data)
        return bayesian_module

    else:
        new_module = copy.deepcopy(module)
        for key in module._modules:
            new_module._modules[key] = convert(module._modules[key], init_sigma=init_sigma, **kwargs)
        return new_module


def log_gaussian_loss(output, target, logsigma):
    # p(output) = N(target, sigma^2)
    exponent = - 0.5 * (target - output) ** 2 / torch.exp(logsigma) ** 2
    log_coeff = - logsigma - 0.5 * math.log(2 * torch.pi)

    return - (log_coeff + exponent).sum()


def cal_KL(mu1, logsigma1, mu2, logsigma2):
    """
    Compute KL divergence KL(P1 || P2) where:
      P1 ~ N(mu1, exp(logsigma1)^2)
      P2 ~ N(mu2, exp(logsigma2)^2)

    :param mu1: Mean of the first Gaussian
    :param logsigma1: Log-standard deviation of the first Gaussian
    :param mu2: Mean of the second Gaussian
    :param logsigma2: Log-standard deviation of the second Gaussian
    :return: KL divergence value(s)
    """
    # Calculate the KL divergence term by term:
    # log(sigma2/sigma1) = logsigma2 - logsigma1
    term1 = logsigma2 - logsigma1

    # sigma1^2 = exp(2*logsigma1) and sigma2^2 = exp(2*logsigma2)
    # Compute the second term: (sigma1^2 + (mu1 - mu2)^2) / (2 * sigma2^2)
    term2 = (torch.exp(2 * logsigma1) + (mu1 - mu2) ** 2) / (2 * torch.exp(2 * logsigma2))

    # Final KL divergence calculation:
    kl = (term1 + term2 - 0.5).sum()
    return kl


def cal_KL_modules(curr_module, prior_mu, prior_sigma):
    """
    Args:
        curr_module: nn.Module, current module
        prior_module: nn.Module, prior module
        prior_sigma: torch.tensor or nn.Module
    Returns:
    """
    global kl_count
    is_base = not any(curr_module.children())
    if is_base:
        if (isinstance(curr_module, RandConv2d) or isinstance(curr_module, RandLinear)):
            kl_count += 1
            kl_weight = cal_KL(curr_module.mu_weight, curr_module.sigma_weight, prior_mu, prior_sigma)
            if curr_module.mu_bias is not None:
                kl_bias = cal_KL(curr_module.mu_bias, curr_module.sigma_bias, prior_mu, prior_sigma)
            else:
                kl_bias = 0
            return kl_weight + kl_bias
        else:
            return 0.0  # not a layer to be converted into Bayesian
    else:
        kl = torch.tensor(0.0, device="cuda")
        for key in curr_module._modules:
            kl += cal_KL_modules(curr_module=curr_module._modules[key],
                                 prior_mu=prior_mu,
                                 prior_sigma=prior_sigma)
        return kl


def pred_sample(module, x, samplings=10):
    pred_mus = []
    pred_sigmas = []
    with torch.no_grad():
        for i in range(samplings):
            pred = module(x)
            pred_mu, pred_logsigma = pred.chunk(2, dim=-1)
            pred_mus.append(pred_mu)
            pred_sigmas.append(torch.exp(pred_logsigma))

    Ha = torch.mean(torch.stack(pred_sigmas, dim=0), dim=0)
    He = torch.mean(torch.stack([i ** 2 for i in pred_mus], dim=0), dim=0) - torch.mean(
        torch.stack(pred_mus, dim=0) ** 2)
    return pred, Ha, He


if __name__ == '__main__':
    from src.models import PilotNet, MegaPilotNet, MultiCamWaypointNet, WaypointNet, SplitCamWaypointNet

    init_s = math.log(0.02)
    m1 = WaypointNet()
    # m1.load(xxx)
    with torch.no_grad():
        bm1 = convert(m1, init_sigma=init_s, skip_Conv=True, skip_Linear=False)

        # Convert the last linear layer into Bayesian layer with additional logsigma output to model a distribution
        previous_layer = bm1.fc[-1]
        in_features = previous_layer.in_features
        out_features = previous_layer.out_features
        bm1.fc[-1] = RandLinear(in_features=in_features, out_features=out_features * 2,
                                bias=bm1.fc[-1].mu_bias is not None,
                                init_s=init_s)
        bm1.fc[-1].mu_weight[:out_features, :] = previous_layer.mu_weight
        bm1.fc[-1].mu_bias[:out_features] = previous_layer.mu_bias

    # train
    # rand_input = torch.randn(10, 3, 168, 224)
    # rand_target = torch.randn(10, 8)
    # output = bm1(rand_input)
    # output, output_logsigma = output.chunk(2, dim=-1)
    # loss = log_gaussian_loss(output, rand_target, output_logsigma)
    # kl_loss = cal_KL_modules(m1, 0, math.log(0.02))
    # loss += kl_loss / len(data_loader)
    # loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()

    # uncertainty estimation

    rand_input = torch.randn(10, 3, 168, 224)
    pred, Ha, He = pred_sample(module=bm1, x=rand_input)
