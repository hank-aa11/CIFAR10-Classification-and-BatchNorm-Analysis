"""Section 2.2: VGG-A vs VGG-A-BN at the same LR, plotting train loss + val acc."""
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
EPOCHS = 30
LR = 1e-3
FIG_DIR = './figures'
RES_DIR = './results'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); c, t = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        c += (model(x).argmax(1) == y).sum().item(); t += y.size(0)
    return c / t


def train_one(model, name, tl, vl):
    model = model.to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    losses, accs = [], []
    for ep in range(EPOCHS):
        model.train(); s, n = 0., 0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y); loss.backward(); opt.step()
            s += loss.item() * y.size(0); n += y.size(0)
        losses.append(s / n)
        accs.append(evaluate(model, vl))
        print(f'[{name}] ep {ep+1}/{EPOCHS}  loss={losses[-1]:.4f}  '
              f'val_acc={accs[-1]:.4f}')
    return losses, accs


def main():
    tl = get_cifar_loader(train=True)
    vl = get_cifar_loader(train=False)
    a_loss, a_acc = train_one(VGG_A(), 'VGG_A', tl, vl)
    b_loss, b_acc = train_one(VGG_A_BatchNorm(), 'VGG_A_BN', tl, vl)

    json.dump({'vgg_a_loss': a_loss, 'vgg_a_acc': a_acc,
               'vgg_bn_loss': b_loss, 'vgg_bn_acc': b_acc},
              open(os.path.join(RES_DIR, 'compare_bn.json'), 'w'),
              indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(a_loss, label='VGG-A')
    ax[0].plot(b_loss, label='VGG-A + BN')
    ax[0].set(xlabel='Epoch', ylabel='Train loss', title='Training loss')
    ax[0].grid(True, alpha=0.3); ax[0].legend()
    ax[1].plot(a_acc, label='VGG-A')
    ax[1].plot(b_acc, label='VGG-A + BN')
    ax[1].set(xlabel='Epoch', ylabel='Val accuracy',
              title='Validation accuracy')
    ax[1].grid(True, alpha=0.3); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'bn_compare.png'), dpi=150)


if __name__ == '__main__':
    main()