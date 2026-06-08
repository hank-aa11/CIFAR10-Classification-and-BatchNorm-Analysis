"""CIFAR-10 loaders shared by both parts."""
import torch
import torchvision
import torchvision.transforms as T

CIFAR_MEAN_NORMAL = (0.4914, 0.4822, 0.4465)
CIFAR_STD_NORMAL = (0.2470, 0.2435, 0.2616)
CIFAR_MEAN_HALF = (0.5, 0.5, 0.5)
CIFAR_STD_HALF = (0.5, 0.5, 0.5)
CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')


def build_transforms(train, strong=True,
                     mean=CIFAR_MEAN_NORMAL, std=CIFAR_STD_NORMAL):
    if train:
        ops = [T.RandomCrop(32, padding=4, padding_mode='reflect'),
               T.RandomHorizontalFlip()]
        if strong:
            ops += [T.ColorJitter(0.2, 0.2, 0.2),
                    T.RandAugment(num_ops=2, magnitude=9)]
        ops += [T.ToTensor(), T.Normalize(mean, std)]
        if strong:
            ops += [T.RandomErasing(p=0.25)]
        return T.Compose(ops)
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


def get_loaders(batch_size=128, num_workers=4, root='./data',
                strong_aug=True,
                mean=CIFAR_MEAN_NORMAL, std=CIFAR_STD_NORMAL):
    train_set = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True,
        transform=build_transforms(True, strong_aug, mean, std))
    test_set = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True,
        transform=build_transforms(False, False, mean, std))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def mixup(x, y, alpha=0.2, num_classes=10):
    if alpha <= 0:
        return x, torch.nn.functional.one_hot(y, num_classes).float()
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    idx = torch.randperm(x.size(0), device=x.device)
    y_oh = torch.nn.functional.one_hot(y, num_classes).float()
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = lam * y_oh + (1 - lam) * y_oh[idx]
    return mixed_x, mixed_y