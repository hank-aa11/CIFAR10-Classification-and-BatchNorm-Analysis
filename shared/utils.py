"""Shared utilities used by both parts."""
import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum, self.cnt = 0.0, 0

    def update(self, v, n=1):
        self.sum += float(v) * n
        self.cnt += n

    @property
    def avg(self):
        return self.sum / max(self.cnt, 1)


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')