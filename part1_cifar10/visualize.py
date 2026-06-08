"""Visualization: filters, feature maps, Grad-CAM, 1D/2D loss landscape, CM."""
import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.utils import get_device
from shared.data_cifar import CIFAR10_CLASSES


# ----------------------------- filter visualization --------------------------
def visualize_first_conv_filters(model, save_path):
    """Plot the first conv layer's filters."""
    first_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break
    if first_conv is None:
        return
    w = first_conv.weight.detach().cpu().numpy()
    n = w.shape[0]
    cols = 8
    rows = (n + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(cols, rows))
    for i in range(rows * cols):
        a = ax[i // cols, i % cols] if rows > 1 else ax[i % cols]
        a.axis('off')
        if i < n:
            f = w[i]
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            if f.shape[0] == 3:
                a.imshow(f.transpose(1, 2, 0))
            else:
                a.imshow(f[0], cmap='gray')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ----------------------------- feature maps ----------------------------------
@torch.no_grad()
def visualize_feature_maps(model, x_sample, save_path, layer_name='layer1'):
    """Save first 16 channels of the output of the named module."""
    device = get_device()
    model.eval().to(device)
    target = dict(model.named_modules())[layer_name]
    feats = {}
    h = target.register_forward_hook(lambda m, i, o: feats.setdefault('o', o))
    _ = model(x_sample.unsqueeze(0).to(device))
    h.remove()
    fmap = feats['o'][0].cpu().numpy()
    n = min(16, fmap.shape[0])
    fig, ax = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(16):
        a = ax[i // 8, i % 8]
        a.axis('off')
        if i < n:
            a.imshow(fmap[i], cmap='viridis')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ----------------------------- Grad-CAM --------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.feat = None
        self.grad = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.feat = o

    def _bwd(self, m, gi, go):
        self.grad = go[0]

    def __call__(self, x, cls=None):
        x = x.requires_grad_(True)
        logits = self.model(x)
        if cls is None:
            cls = logits.argmax(1)
        score = logits.gather(1, cls.view(-1, 1)).sum()
        self.model.zero_grad()
        score.backward()
        # weights = global-avg-pool of grads
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.feat).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear',
                            align_corners=False)
        cam = cam.squeeze(1)
        cam = (cam - cam.amin(dim=(1, 2), keepdim=True)) / (
            cam.amax(dim=(1, 2), keepdim=True)
            - cam.amin(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach().cpu().numpy(), cls.detach().cpu().numpy()


def visualize_gradcam(model, target_layer, sample_loader, save_path,
                      n_images=8):
    device = get_device()
    model.to(device)
    cam = GradCAM(model, target_layer)
    xs, ys = next(iter(sample_loader))
    xs, ys = xs[:n_images].to(device), ys[:n_images]
    cams, preds = cam(xs)
    fig, ax = plt.subplots(2, n_images, figsize=(2 * n_images, 4))
    for i in range(n_images):
        img = xs[i].detach().cpu().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax[0, i].imshow(img)
        ax[0, i].set_title(f'gt={CIFAR10_CLASSES[ys[i].item()]}\n'
                           f'pred={CIFAR10_CLASSES[preds[i]]}', fontsize=8)
        ax[0, i].axis('off')
        ax[1, i].imshow(img)
        ax[1, i].imshow(cams[i], cmap='jet', alpha=0.45)
        ax[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ----------------------------- 1D loss landscape -----------------------------
@torch.no_grad()
def loss_landscape_1d(model, loader, save_path, alphas=None, max_batches=8):
    """Loss along a random direction.

    Important:
    Only perturb trainable weight parameters with dim >= 2.
    Do NOT perturb BatchNorm running_mean / running_var.
    """
    device = get_device()
    model.eval().to(device)

    if alphas is None:
        alphas = np.linspace(-0.5, 0.5, 31)

    # Save original trainable parameters only
    base = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
    }

    # Only perturb weight tensors, not bias / BN affine parameters
    direction = {}
    for name, p in model.named_parameters():
        if p.requires_grad and p.dim() >= 2:
            d = torch.randn_like(p)

            # Filter-wise normalization for Conv / Linear weights
            if p.dim() >= 2:
                d_flat = d.flatten(1)
                p_flat = p.detach().flatten(1)
                d_norm = d_flat.norm(dim=1, keepdim=True)
                p_norm = p_flat.norm(dim=1, keepdim=True)
                scale = p_norm / (d_norm + 1e-10)
                d = d * scale.view(-1, *([1] * (p.dim() - 1)))

            direction[name] = d

    criterion = nn.CrossEntropyLoss()
    losses = []

    for a in alphas:
        # Apply perturbation
        for name, p in model.named_parameters():
            if name in direction:
                p.copy_(base[name] + float(a) * direction[name])
            else:
                p.copy_(base[name])

        total_loss = 0.0
        total_num = 0

        for bi, (x, y) in enumerate(loader):
            if bi >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if torch.isfinite(loss):
                total_loss += loss.item() * y.size(0)
                total_num += y.size(0)

        if total_num == 0:
            losses.append(np.nan)
        else:
            losses.append(total_loss / total_num)

    # Restore original parameters
    for name, p in model.named_parameters():
        p.copy_(base[name])

    losses_np = np.array(losses, dtype=np.float64)

    plt.figure(figsize=(6, 4))
    plt.plot(alphas, losses_np, marker='o')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Loss')
    plt.title('1D loss landscape')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f'[1D landscape] finite points: {np.isfinite(losses_np).sum()} / {len(losses_np)}')
    return alphas, losses_np


# ----------------------------- 2D loss landscape -----------------------------
@torch.no_grad()
def loss_landscape_2d(model, loader, save_path,
                      n=15, span=0.5, max_batches=4):
    """2D loss landscape.

    Important:
    Only perturb trainable weight parameters with dim >= 2.
    Do NOT perturb BatchNorm running_mean / running_var.
    """
    device = get_device()
    model.eval().to(device)

    base = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
    }

    def make_direction():
        direction = {}
        for name, p in model.named_parameters():
            if p.requires_grad and p.dim() >= 2:
                d = torch.randn_like(p)

                d_flat = d.flatten(1)
                p_flat = p.detach().flatten(1)
                d_norm = d_flat.norm(dim=1, keepdim=True)
                p_norm = p_flat.norm(dim=1, keepdim=True)
                scale = p_norm / (d_norm + 1e-10)
                d = d * scale.view(-1, *([1] * (p.dim() - 1)))

                direction[name] = d
        return direction

    d1 = make_direction()
    d2 = make_direction()

    alphas = np.linspace(-span, span, n)
    betas = np.linspace(-span, span, n)
    grid = np.zeros((n, n), dtype=np.float64)

    criterion = nn.CrossEntropyLoss()

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            for name, p in model.named_parameters():
                if name in d1:
                    p.copy_(base[name] + float(a) * d1[name] + float(b) * d2[name])
                else:
                    p.copy_(base[name])

            total_loss = 0.0
            total_num = 0

            for bi, (x, y) in enumerate(loader):
                if bi >= max_batches:
                    break
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                if torch.isfinite(loss):
                    total_loss += loss.item() * y.size(0)
                    total_num += y.size(0)

            if total_num == 0:
                grid[i, j] = np.nan
            else:
                grid[i, j] = total_loss / total_num

        print(f'2D landscape row {i + 1}/{n}')

    # Restore original parameters
    for name, p in model.named_parameters():
        p.copy_(base[name])

    finite_count = np.isfinite(grid).sum()
    print(f'[2D landscape] finite points: {finite_count} / {grid.size}')

    np.savez(save_path.replace('.png', '.npz'),
             grid=grid, alphas=alphas, betas=betas)

    # Fill NaN only for plotting
    plot_grid = grid.copy()
    if np.isfinite(plot_grid).any():
        max_finite = np.nanmax(plot_grid[np.isfinite(plot_grid)])
        plot_grid = np.nan_to_num(plot_grid, nan=max_finite,
                                  posinf=max_finite, neginf=max_finite)
    else:
        plot_grid = np.zeros_like(plot_grid)

    A, B = np.meshgrid(alphas, betas, indexing='ij')

    fig = plt.figure(figsize=(11, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    cs = ax1.contour(A, B, plot_grid, levels=20, cmap='viridis')
    ax1.clabel(cs, inline=True, fontsize=7)
    ax1.set_xlabel(r'$\alpha$')
    ax1.set_ylabel(r'$\beta$')
    ax1.set_title('2D loss landscape contour')

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(A, B, plot_grid, cmap='viridis',
                     edgecolor='none', alpha=0.9)
    ax2.set_xlabel(r'$\alpha$')
    ax2.set_ylabel(r'$\beta$')
    ax2.set_zlabel('Loss')
    ax2.set_title('2D loss landscape surface')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ----------------------------- confusion matrix ------------------------------
@torch.no_grad()
def plot_confusion_matrix(model, loader, save_path,
                          classes=CIFAR10_CLASSES):
    device = get_device()
    model.eval().to(device)
    K = len(classes)
    cm = np.zeros((K, K), dtype=int)
    for x, y in loader:
        x = x.to(device)
        p = model(x).argmax(1).cpu().numpy()
        for yi, pi in zip(y.numpy(), p):
            cm[yi, pi] += 1
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f'{cm_norm[i, j]:.2f}', ha='center', va='center',
                    color='white' if cm_norm[i, j] > 0.5 else 'black',
                    fontsize=6)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Normalised confusion matrix')
    plt.colorbar(im, ax=ax)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()