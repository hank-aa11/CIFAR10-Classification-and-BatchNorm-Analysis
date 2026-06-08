"""Bonus 3: visualise activation distribution evolution at selected layers."""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 4
LR = 1e-3
PROBE_EVERY = 50
PROBE_LAYERS = [3, 7, 14]
FIG_DIR = './figures'
os.makedirs(FIG_DIR, exist_ok=True)


def collect_percentiles(model, probe_x, layer_indices):
    feats = {}
    hooks = []
    for li in layer_indices:
        h = model.features[li].register_forward_hook(
            lambda m, i, o, k=li: feats.setdefault(k, o.detach()))
        hooks.append(h)
    model.eval()
    with torch.no_grad():
        _ = model(probe_x.to(DEVICE))
    for h in hooks:
        h.remove()
    out = {}
    for k, v in feats.items():
        flat = v.flatten().cpu().numpy()
        out[k] = np.percentile(flat, [15, 50, 85])
    return out


def train_and_probe(model_cls, name, tl, probe_x):
    model = model_cls().to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    history = {li: [] for li in PROBE_LAYERS}
    step = 0
    for ep in range(EPOCHS):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
            if step % PROBE_EVERY == 0:
                p = collect_percentiles(model, probe_x, PROBE_LAYERS)
                model.train()
                for li in PROBE_LAYERS:
                    history[li].append(p[li])
            step += 1
        print(f'[{name}] ep {ep+1}/{EPOCHS}')
    return {li: np.array(history[li]) for li in PROBE_LAYERS}


def main():
    tl = get_cifar_loader(train=True)
    probe_x, _ = next(iter(tl))
    h_std = train_and_probe(VGG_A, 'standard', tl, probe_x)
    h_bn = train_and_probe(VGG_A_BatchNorm, 'bn', tl, probe_x)

    fig, ax = plt.subplots(2, len(PROBE_LAYERS),
                           figsize=(4 * len(PROBE_LAYERS), 6))
    for col, li in enumerate(PROBE_LAYERS):
        for row, (h, name, c) in enumerate(
                [(h_std, 'Standard', 'tab:red'),
                 (h_bn, 'With BN', 'tab:blue')]):
            arr = h[li]
            xs = np.arange(arr.shape[0]) * PROBE_EVERY
            ax[row, col].fill_between(xs, arr[:, 0], arr[:, 2],
                                       alpha=0.3, color=c)
            ax[row, col].plot(xs, arr[:, 1], color=c, lw=1.0)
            ax[row, col].set_title(f'{name}, layer {li}')
            ax[row, col].set_xlabel('Step')
            ax[row, col].set_ylabel('Activation')
            ax[row, col].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'ics_evolution.png'), dpi=150)


if __name__ == '__main__':
    main()