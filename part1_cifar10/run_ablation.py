"""Run all ablation experiments required by tasks 4 & 5."""
import os
import sys
import json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.utils import set_seed, count_params, ensure_dirs
from shared.data_cifar import get_loaders
from part1_cifar10.models import BasicCNN
from part1_cifar10.optimizers import MyMomentumSGD, MyAdamW
from part1_cifar10.losses import LabelSmoothingCrossEntropy, FocalLoss
from part1_cifar10.trainer import train_one


def main():
    set_seed(42)
    ensure_dirs('./checkpoints', './logs', './figures')
    train_loader, test_loader = get_loaders(batch_size=128, num_workers=4)

    EPOCHS = 30
    summary = {}

    # ---------- (4a) width ablation ----------
    for w in [(32, 64, 128), (64, 128, 256), (96, 192, 384)]:
        name = f'width_{w[0]}_{w[1]}_{w[2]}'
        model = BasicCNN(channels=w, activation='relu')
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                              weight_decay=5e-4, nesterov=True)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        h = train_one(name, model, train_loader, test_loader, opt,
                      nn.CrossEntropyLoss(), sched, epochs=EPOCHS)
        summary[name] = {'best_acc': h['best_acc'],
                         'params': count_params(model)}

    # ---------- (4b) loss ablation ----------
    losses = {
        'ce': nn.CrossEntropyLoss(),
        'ce_l2': nn.CrossEntropyLoss(),  # plus weight_decay; via optimizer
        'label_smooth_0.1': LabelSmoothingCrossEntropy(0.1),
        'focal_2.0': FocalLoss(2.0),
    }
    for ln, crit in losses.items():
        name = f'loss_{ln}'
        model = BasicCNN(channels=(64, 128, 256))
        wd = 5e-3 if ln == 'ce_l2' else 5e-4
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                              weight_decay=wd, nesterov=True)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        h = train_one(name, model, train_loader, test_loader, opt,
                      crit, sched, epochs=EPOCHS)
        summary[name] = {'best_acc': h['best_acc'],
                         'params': count_params(model)}

    # ---------- (4c) activation ablation ----------
    for a in ['relu', 'gelu', 'silu', 'mish', 'leaky_relu']:
        name = f'act_{a}'
        model = BasicCNN(channels=(64, 128, 256), activation=a)
        opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                              weight_decay=5e-4, nesterov=True)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        h = train_one(name, model, train_loader, test_loader, opt,
                      nn.CrossEntropyLoss(), sched, epochs=EPOCHS)
        summary[name] = {'best_acc': h['best_acc'],
                         'params': count_params(model)}

    # ---------- (5a) library optimizers ----------
    for opt_name in ['sgd_nesterov', 'adam', 'adamw']:
        name = f'opt_{opt_name}'
        model = BasicCNN(channels=(64, 128, 256))
        if opt_name == 'sgd_nesterov':
            opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                  weight_decay=5e-4, nesterov=True)
        elif opt_name == 'adam':
            opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                                   weight_decay=5e-4)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                    weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        h = train_one(name, model, train_loader, test_loader, opt,
                      nn.CrossEntropyLoss(), sched, epochs=EPOCHS,
                      use_amp=False)
        summary[name] = {'best_acc': h['best_acc']}

    # ---------- (5b) hand-rolled optimizers ----------
    for opt_name, opt_cls in [('my_momentum_sgd', MyMomentumSGD),
                              ('my_adamw', MyAdamW)]:
        name = f'opt_{opt_name}'
        model = BasicCNN(channels=(64, 128, 256))
        if opt_name == 'my_momentum_sgd':
            opt = opt_cls(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
        else:
            opt = opt_cls(model.parameters(), lr=1e-3, weight_decay=1e-2)
        h = train_one(name, model, train_loader, test_loader, opt,
                      nn.CrossEntropyLoss(), scheduler=None,
                      epochs=EPOCHS, use_amp=False)
        summary[name] = {'best_acc': h['best_acc']}

    json.dump(summary, open('./logs/ablation_summary.json', 'w'), indent=2)
    print('=== ABLATION SUMMARY ===')
    for k, v in summary.items():
        print(f'  {k:30s}  {v}')


if __name__ == '__main__':
    main()