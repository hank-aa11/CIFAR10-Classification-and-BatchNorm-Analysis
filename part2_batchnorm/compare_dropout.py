"""Bonus 4: BN vs Dropout — rules out 'BN is just regularisation'."""
import os
import json
import torch
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm, VGG_A_Dropout

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 30
LR = 1e-3
FIG_DIR = './figures'
RES_DIR = './results'
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)


@torch.no_grad()
def evaluate(model, vl):
    model.eval(); c = t = 0
    for x, y in vl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        c += (model(x).argmax(1) == y).sum().item(); t += y.size(0)
    return c / t


def train(model, name, tl, vl):
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
        losses.append(s / n); accs.append(evaluate(model, vl))
        print(f'[{name}] ep {ep+1}/{EPOCHS} loss={losses[-1]:.4f} '
              f'acc={accs[-1]:.4f}')
    return losses, accs


def main():
    tl = get_cifar_loader(train=True)
    vl = get_cifar_loader(train=False)
    runs = [(VGG_A, 'VGG-A', 'tab:red'),
            (VGG_A_Dropout, 'VGG-A + Dropout', 'tab:orange'),
            (VGG_A_BatchNorm, 'VGG-A + BN', 'tab:blue')]
    out = {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for cls, name, c in runs:
        l, a = train(cls(), name, tl, vl)
        out[name] = {'loss': l, 'acc': a}
        ax[0].plot(l, c=c, label=name)
        ax[1].plot(a, c=c, label=name)
    json.dump(out, open(os.path.join(RES_DIR, 'compare_dropout.json'), 'w'),
              indent=2)
    ax[0].set(xlabel='Epoch', ylabel='Train loss', title='Training loss')
    ax[1].set(xlabel='Epoch', ylabel='Val acc',
              title='Validation accuracy')
    for a_ in ax:
        a_.grid(True, alpha=0.3); a_.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'compare_dropout.png'), dpi=150)


if __name__ == '__main__':
    main()