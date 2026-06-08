"""Generate all Part-1 figures and run TTA evaluation."""
import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.utils import get_device, ensure_dirs
from shared.data_cifar import get_loaders
from part1_cifar10.models import preact_resnet18
from part1_cifar10.tta import tta_evaluate
from part1_cifar10.visualize import (visualize_first_conv_filters,
                                      visualize_feature_maps,
                                      visualize_gradcam,
                                      loss_landscape_1d,
                                      loss_landscape_2d,
                                      plot_confusion_matrix)
from part1_cifar10.trainer import evaluate


def main():
    ensure_dirs('./figures')
    device = get_device()
    train_loader, test_loader = get_loaders(batch_size=128, num_workers=4,
                                            strong_aug=False)

    model = preact_resnet18().to(device)
    ckpt_paths = ['./checkpoints/resnet18_swa.pt',
                  './checkpoints/resnet18_best.pt']
    for p in ckpt_paths:
        if os.path.isfile(p):
            print(f'loading {p}')
            model.load_state_dict(torch.load(p, map_location=device))
            break
    else:
        print('No checkpoint found; visualising untrained model.')

    # ---- 1. plain test accuracy ----
    _, acc = evaluate(model, test_loader, device)
    print(f'test acc       : {acc:.4f}')

    # ---- 2. TTA accuracy ----
    tta_acc = tta_evaluate(model, test_loader, device=device)
    print(f'test acc (TTA) : {tta_acc:.4f}')

    # ---- 3. visualisations ----
    visualize_first_conv_filters(model, './figures/first_conv_filters.png')

    x_sample, _ = next(iter(test_loader))
    visualize_feature_maps(model, x_sample[0],
                           './figures/feature_maps.png',
                           layer_name='layer1')

    # Grad-CAM on last conv block
    visualize_gradcam(model, model.layer4[-1].conv2,
                      test_loader, './figures/gradcam.png')

    # 1D and 2D loss landscape
    loss_landscape_1d(model, test_loader, './figures/landscape_1d.png')
    loss_landscape_2d(model, test_loader, './figures/landscape_2d.png',
                      n=11, span=0.5, max_batches=4)

    plot_confusion_matrix(model, test_loader, './figures/confusion_matrix.png')


if __name__ == '__main__':
    main()