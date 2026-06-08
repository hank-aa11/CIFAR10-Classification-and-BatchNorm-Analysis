"""Loss functions used in the ablations."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        n_classes = logits.size(-1)
        log_p = F.log_softmax(logits, dim=-1)
        if target.dim() == 1:
            with torch.no_grad():
                t = torch.zeros_like(log_p).fill_(
                    self.smoothing / (n_classes - 1))
                t.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        else:
            t = (target * (1 - self.smoothing) +
                 self.smoothing / n_classes)
        return -(t * log_p).sum(-1).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class SoftTargetCrossEntropy(nn.Module):
    """For mixup / soft label inputs."""
    def forward(self, logits, target):
        return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()