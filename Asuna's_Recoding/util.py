import os
import random
import math
import numpy as np
from scipy.special import logsumexp

import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.utils import make_grid
from collections import OrderedDict


# ------------------------- Utility Classes -------------------------
class AverageMeter:
    """Tracks and computes the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TotalAverage:
    """Tracks a weighted average."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.mass = 0.0
        self.sum = 0.0
        self.avg = 0.0

    def update(self, val, mass=1):
        self.val = val
        self.mass += mass
        self.sum += val * mass
        self.avg = self.sum / self.mass


class MovingAverage:
    """Tracks a moving average with inertia."""
    def __init__(self, inertia=0.9):
        self.inertia = inertia
        self.reset()

    def reset(self):
        self.avg = 0.0

    def update(self, val):
        self.avg = self.inertia * self.avg + (1 - self.inertia) * val


# ------------------------- Initialization Functions -------------------------
def setup_runtime(seed=0, cuda_dev_id=[0]):
    """Initialize CUDA, CuDNN, and set random seeds."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cuda_dev_id))
    
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_pytorch_defaults(module, version='041'):
    """Applies weight initialization based on version."""
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        n = module.in_features if isinstance(module, nn.Linear) else module.in_channels
        stdv = 1. / math.sqrt(n)
        module.weight.data.uniform_(-stdv, stdv)
        if module.bias is not None:
            module.bias.data.uniform_(-stdv, stdv)
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)) and module.affine:
        module.weight.data.uniform_()
        module.bias.data.zero_()


def weight_init(module):
    """Applies custom weight initialization."""
    module.apply(lambda m: init_pytorch_defaults(m, version='041'))


# ------------------------- Training and Evaluation -------------------------
def accuracy(output, target, topk=(1,)):
    """Computes precision@k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))
        return results


def warmup_batchnorm(model, data_loader, device, batches=100):
    """Warms up running stats for batchnorm layers."""
    model.train()
    for i, (images, _) in enumerate(data_loader):
        if i == batches:
            break
        images = images.to(device)
        _ = model(images)


def absorb_bn(module, bn_module):
    """Absorbs batch normalization into preceding layers."""
    w = module.weight.data
    b = module.bias.data if module.bias is not None else torch.zeros_like(w[:, 0])

    invstd = (bn_module.running_var + bn_module.eps).pow(-0.5)
    w.mul_(invstd.view(-1, 1, 1, 1))
    b.add_(-bn_module.running_mean).mul_(invstd)
    
    if bn_module.affine:
        w.mul_(bn_module.weight.data.view(-1, 1, 1, 1))
        b.mul_(bn_module.weight.data).add_(bn_module.bias.data)
    
    bn_module.reset_parameters()


# ------------------------- Model Helpers -------------------------
class View(nn.Module):
    """Adapts shape to flatten input."""
    def forward(self, x):
        return x.view(x.size(0), -1)


def prep_model(model, model_path):
    """Loads model weights and adjusts layers."""
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items()))
    for param in model.features.parameters():
        param.requires_grad = False
    return model