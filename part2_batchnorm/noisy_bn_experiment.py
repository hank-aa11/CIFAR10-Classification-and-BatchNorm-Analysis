"""Bonus 2: Noisy BN. Reproduces Fig. 8 of Santurkar et al. — BN's benefit
is not explained by reduced internal covariate shift."""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm, VGG_A_NoisyBN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 15
LR = 1e-3
RES_DIR = './results'
FIG_DIR = './figures'
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def train_track(model, name, tl, vl):
    model = model.to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    losses, accs = [], []
    for ep in range(EPOCHS):
        model.train(); s, n = 0., 0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            l = crit(model(x), y); l.backward(); opt.step()
            s += l.item() * y.size(0); n += y.size(0)
        losses.append(s / n)
        model.eval(); c = t = 0
        with torch.no_grad():
            for x, y in vl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                c += (model(x).argmax(1) == y).sum().item(); t += y.size(0)
        accs.append(c / t)
        print(f'[{name}] ep {ep+1}/{EPOCHS} loss={losses[-1]:.4f} '
              f'acc={accs[-1]:.4f}')
    return losses, accs


def main():
    tl = get_cifar_loader(train=True)
    vl = get_cifar_loader(train=False)
    L_std, A_std = train_track(VGG_A(), 'VGG-A', tl, vl)
    L_bn, A_bn = train_track(VGG_A_BatchNorm(), 'VGG-A + BN', tl, vl)
    L_noi, A_noi = train_track(
        VGG_A_NoisyBN(mean_shift=2.0, log_scale_std=1.0),
        'VGG-A + NoisyBN', tl, vl)
    np.savez(os.path.join(RES_DIR, 'noisy_bn.npz'),
             L_std=L_std, A_std=A_std,
             L_bn=L_bn, A_bn=A_bn,
             L_noi=L_noi, A_noi=A_noi)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(L_std, c='tab:red', label='Standard')
    ax[0].plot(L_bn, c='tab:blue', label='+ BN')
    ax[0].plot(L_noi, c='tab:green', ls='--', label='+ Noisy BN')
    ax[0].set(xlabel='Epoch', ylabel='Train loss',
              title='Noisy BN trains nearly as well as BN')
    ax[0].grid(True, alpha=0.3); ax[0].legend()
    ax[1].plot(A_std, c='tab:red', label='Standard')
    ax[1].plot(A_bn, c='tab:blue', label='+ BN')
    ax[1].plot(A_noi, c='tab:green', ls='--', label='+ Noisy BN')
    ax[1].set(xlabel='Epoch', ylabel='Val acc',
              title='Reducing ICS is not the cause of BN’s benefit')
    ax[1].grid(True, alpha=0.3); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'noisy_bn.png'), dpi=150)


if __name__ == '__main__':
    main()