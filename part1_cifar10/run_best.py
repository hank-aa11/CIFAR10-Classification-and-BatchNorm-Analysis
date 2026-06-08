"""Train the best model: PreAct ResNet-18 + Mixup + EMA + AMP + cosine + SWA."""
import os
import sys
import json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.utils import set_seed, count_params, ensure_dirs
from shared.data_cifar import get_loaders
from part1_cifar10.models import preact_resnet18
from part1_cifar10.trainer import train_one, train_with_swa


def main():
    set_seed(42)
    ensure_dirs('./checkpoints', './logs', './figures')
    train_loader, test_loader = get_loaders(batch_size=128, num_workers=4)

    # ---- Stage 1: long training with mixup/EMA/AMP/cosine ----
    name = 'resnet18'
    model = preact_resnet18()
    print(f'{name} params = {count_params(model):,}')
    EPOCHS = 200

    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)

    warmup_epochs = 5
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        import math
        prog = (ep - warmup_epochs) / max(1, EPOCHS - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    h = train_one(name, model, train_loader, test_loader, opt,
                  nn.CrossEntropyLoss(), sched, epochs=EPOCHS,
                  use_ema=True, ema_decay=0.999, use_amp=True,
                  mixup_alpha=0.2)
    print(f'best test_acc (stage 1): {h["best_acc"]:.4f}')

    # ---- Stage 2: SWA on a fresh model for cleaner comparison ----
    swa_model = preact_resnet18()
    h2 = train_with_swa('resnet18_swa', swa_model, train_loader, test_loader,
                        epochs=200, swa_start=120, base_lr=0.1,
                        weight_decay=5e-4, mixup_alpha=0.2)
    print(f'best swa_acc: {h2["best_swa"]:.4f}')

    json.dump({'stage1': h, 'stage2': h2},
              open('./logs/best_summary.json', 'w'),
              indent=2, default=float)


if __name__ == '__main__':
    main()