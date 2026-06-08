"""VGG-A family for CIFAR-10 (32x32). Standard, BatchNorm, NoisyBN, Dropout."""
import numpy as np
import torch
from torch import nn
from utils.nn import init_weights_


def get_number_of_parameters(model):
    return sum(int(np.prod(p.shape)) for p in model.parameters())


# ----------------------------- VGG-A -----------------------------------------
class VGG_A(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(inp_ch, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        # classifier[0,2,4] are Linear; classifier[4] is the final FC.
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True),
            nn.Linear(512, 512), nn.ReLU(True),
            nn.Linear(512, num_classes))
        if init_weights:
            for m in self.modules():
                init_weights_(m)

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


# ----------------------------- VGG-A + BN ------------------------------------
class VGG_A_BatchNorm(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True):
        super().__init__()

        def cb(ic, oc):
            return [nn.Conv2d(ic, oc, 3, padding=1),
                    nn.BatchNorm2d(oc), nn.ReLU(True)]
        layers = (cb(inp_ch, 64) + [nn.MaxPool2d(2)]
                  + cb(64, 128) + [nn.MaxPool2d(2)]
                  + cb(128, 256) + cb(256, 256) + [nn.MaxPool2d(2)]
                  + cb(256, 512) + cb(512, 512) + [nn.MaxPool2d(2)]
                  + cb(512, 512) + cb(512, 512) + [nn.MaxPool2d(2)])
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True),
            nn.Linear(512, 512), nn.ReLU(True),
            nn.Linear(512, num_classes))
        if init_weights:
            for m in self.modules():
                init_weights_(m)

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


# ----------------- BONUS: VGG-A + Noisy BN (Santurkar Fig. 8) ----------------
class NoisyBatchNorm2d(nn.Module):
    """BN followed by a per-step random affine. Re-introduces strong
    distribution shift while preserving BN's optimisation benefits."""
    def __init__(self, num_features, mean_shift=2.0, log_scale_std=1.0):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.mean_shift = mean_shift
        self.log_scale_std = log_scale_std

    def forward(self, x):
        x = self.bn(x)
        if self.training:
            C = x.size(1)
            shift = torch.randn(1, C, 1, 1, device=x.device) * self.mean_shift
            scale = torch.exp(torch.randn(1, C, 1, 1, device=x.device)
                              * self.log_scale_std)
            x = x * scale + shift
        return x


class VGG_A_NoisyBN(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True,
                 mean_shift=2.0, log_scale_std=1.0):
        super().__init__()

        def cb(ic, oc):
            return [nn.Conv2d(ic, oc, 3, padding=1),
                    NoisyBatchNorm2d(oc, mean_shift, log_scale_std),
                    nn.ReLU(True)]
        layers = (cb(inp_ch, 64) + [nn.MaxPool2d(2)]
                  + cb(64, 128) + [nn.MaxPool2d(2)]
                  + cb(128, 256) + cb(256, 256) + [nn.MaxPool2d(2)]
                  + cb(256, 512) + cb(512, 512) + [nn.MaxPool2d(2)]
                  + cb(512, 512) + cb(512, 512) + [nn.MaxPool2d(2)])
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True),
            nn.Linear(512, 512), nn.ReLU(True),
            nn.Linear(512, num_classes))
        if init_weights:
            for m in self.modules():
                init_weights_(m)

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


# ----------------------------- VGG-A + Dropout -------------------------------
class VGG_A_Dropout(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights=True, p=0.5):
        super().__init__()

        def cb(ic, oc):
            return [nn.Conv2d(ic, oc, 3, padding=1), nn.ReLU(True)]
        self.features = nn.Sequential(
            *cb(inp_ch, 64), nn.MaxPool2d(2),
            *cb(64, 128), nn.MaxPool2d(2),
            *cb(128, 256), *cb(256, 256), nn.MaxPool2d(2),
            *cb(256, 512), *cb(512, 512), nn.MaxPool2d(2),
            *cb(512, 512), *cb(512, 512), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Dropout(p), nn.Linear(512, 512), nn.ReLU(True),
            nn.Dropout(p), nn.Linear(512, 512), nn.ReLU(True),
            nn.Linear(512, num_classes))
        if init_weights:
            for m in self.modules():
                init_weights_(m)

    def forward(self, x):
        return self.classifier(self.features(x).view(x.size(0), -1))


if __name__ == '__main__':
    for cls in (VGG_A, VGG_A_BatchNorm, VGG_A_NoisyBN, VGG_A_Dropout):
        print(f'{cls.__name__:20s} params = {get_number_of_parameters(cls()):,}')