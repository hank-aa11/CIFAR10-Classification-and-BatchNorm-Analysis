"""Test-Time Augmentation. Provides a small but consistent boost on CIFAR-10."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def tta_evaluate(model, loader, device='cuda'):
    """5-crop + horizontal flip = 10 views per image."""
    model.eval().to(device)
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        views = [x, torch.flip(x, dims=[-1])]
        x_pad = F.pad(x, [4, 4, 4, 4], mode='reflect')
        for dy in (0, 8):
            for dx in (0, 8):
                views.append(x_pad[..., dy:dy + 32, dx:dx + 32])
        views.append(x_pad[..., 4:36, 4:36])
        probs = sum(F.softmax(model(v), dim=1) for v in views) / len(views)
        correct += (probs.argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total