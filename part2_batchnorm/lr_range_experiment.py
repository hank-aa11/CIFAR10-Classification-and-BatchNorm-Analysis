"""Bonus 1: trainable learning-rate range. Reproduces Fig. 5 of Santurkar et al."""
import os
import json
import torch
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 5
LR_LIST = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
RES_DIR = './results'
FIG_DIR = './figures'
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


@torch.no_grad()
def acc(model, loader):
    model.eval(); c = t = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        c += (model(x).argmax(1) == y).sum().item(); t += y.size(0)
    return c / t


def run(model_cls, lr, tl, vl):
    model = model_cls().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    diverged = False
    for ep in range(EPOCHS):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            l = crit(model(x), y)
            if not torch.isfinite(l):
                diverged = True; break
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
            opt.step()
        if diverged:
            break
    return 0.1 if diverged else acc(model, vl)


def main():
    tl = get_cifar_loader(train=True)
    vl = get_cifar_loader(train=False)
    res = {'lrs': LR_LIST, 'standard': [], 'bn': []}
    for lr in LR_LIST:
        a = run(VGG_A, lr, tl, vl)
        res['standard'].append(a)
        print(f'VGG-A      lr={lr:<8g} acc={a:.4f}')
        b = run(VGG_A_BatchNorm, lr, tl, vl)
        res['bn'].append(b)
        print(f'VGG-A+BN   lr={lr:<8g} acc={b:.4f}')
    json.dump(res, open(os.path.join(RES_DIR, 'lr_range.json'), 'w'),
              indent=2)

    plt.figure(figsize=(7, 4))
    plt.semilogx(LR_LIST, res['standard'], 'o-', c='tab:red',
                 label='Standard VGG')
    plt.semilogx(LR_LIST, res['bn'], 's-', c='tab:blue',
                 label='Standard VGG + BN')
    plt.axhline(0.1, ls='--', c='gray', lw=0.8, label='chance')
    plt.xlabel('Learning rate')
    plt.ylabel(f'Val acc after {EPOCHS} ep')
    plt.title('Trainable LR range: BN tolerates much larger LRs')
    plt.grid(True, which='both', alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'lr_range.png'), dpi=150)


if __name__ == '__main__':
    main()