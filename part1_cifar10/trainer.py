"""Training loops with EMA, AMP, Mixup, and SWA support."""
import os
import json
import copy
import time
import torch
import torch.nn as nn
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shared.utils import AverageMeter, get_device
from shared.data_cifar import mixup
from .losses import SoftTargetCrossEntropy


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ps, pm in zip(self.shadow.parameters(), model.parameters()):
            ps.data.mul_(self.decay).add_(pm.data, alpha=1 - self.decay)
        # also copy buffers (BN running stats)
        for bs, bm in zip(self.shadow.buffers(), model.buffers()):
            bs.data.copy_(bm.data)


@torch.no_grad()
def evaluate(model, loader, device=None):
    device = device or get_device()
    model.eval().to(device)
    crit = nn.CrossEntropyLoss()
    loss_m, acc_m = AverageMeter(), AverageMeter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        loss_m.update(loss.item(), y.size(0))
        acc_m.update(acc.item(), y.size(0))
    return loss_m.avg, acc_m.avg


def train_one(name, model, train_loader, test_loader,
              optimizer, criterion, scheduler=None,
              epochs=50, device=None, save_dir='./checkpoints',
              log_dir='./logs', use_ema=False, ema_decay=0.999,
              use_amp=True, mixup_alpha=0.0, log_every=1):
    """Single-run training. Returns history dict."""
    device = device or get_device()
    model = model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    ema = EMA(model, decay=ema_decay) if use_ema else None
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == 'cuda')
    soft_crit = SoftTargetCrossEntropy()

    history = {'train_loss': [], 'test_loss': [], 'test_acc': [],
               'ema_acc': [], 'lr': []}
    best_acc = 0.0

    for ep in range(epochs):
        model.train()
        loss_m = AverageMeter()
        t0 = time.time()
        for x, y in tqdm(train_loader, leave=False,
                         desc=f'{name} ep{ep+1}/{epochs}'):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            if mixup_alpha > 0:
                xm, ym = mixup(x, y, alpha=mixup_alpha, num_classes=10)
                with torch.cuda.amp.autocast(enabled=use_amp and device.type == 'cuda'):
                    logits = model(xm)
                    loss = soft_crit(logits, ym)
            else:
                with torch.cuda.amp.autocast(enabled=use_amp and device.type == 'cuda'):
                    logits = model(x)
                    loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            loss_m.update(loss.item(), y.size(0))

        if scheduler is not None:
            scheduler.step()

        test_loss, test_acc = evaluate(model, test_loader, device)
        ema_acc = (evaluate(ema.shadow, test_loader, device)[1]
                   if ema is not None else 0.0)
        cur_lr = optimizer.param_groups[0]['lr'] if hasattr(
            optimizer, 'param_groups') else optimizer.lr

        history['train_loss'].append(loss_m.avg)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['ema_acc'].append(ema_acc)
        history['lr'].append(cur_lr)

        if ep % log_every == 0:
            print(f'[{name}] ep {ep+1}/{epochs}  '
                  f'train_loss={loss_m.avg:.4f}  test_acc={test_acc:.4f}  '
                  f'ema_acc={ema_acc:.4f}  lr={cur_lr:.2e}  '
                  f'time={time.time()-t0:.1f}s')

        # save best
        ref_acc = max(test_acc, ema_acc)
        if ref_acc > best_acc:
            best_acc = ref_acc
            ckpt = (ema.shadow if (ema is not None and ema_acc >= test_acc)
                    else model)
            torch.save(ckpt.state_dict(),
                       os.path.join(save_dir, f'{name}_best.pt'))

    history['best_acc'] = best_acc
    json.dump(history, open(os.path.join(log_dir, f'{name}.json'), 'w'),
              indent=2, default=float)
    print(f'[{name}] DONE  best_acc={best_acc:.4f}')
    return history


# --------------------------------- SWA ---------------------------------------
def train_with_swa(name, model, train_loader, test_loader,
                   epochs=200, swa_start=120, base_lr=0.1,
                   weight_decay=5e-4, device=None,
                   save_dir='./checkpoints', log_dir='./logs',
                   use_amp=True, mixup_alpha=0.2):
    """SGD + cosine + SWA. Saves best SWA checkpoint."""
    from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
    device = device or get_device()
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=weight_decay,
                          nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=swa_start)
    swa_model = AveragedModel(model)
    swa_sched = SWALR(opt, swa_lr=0.01, anneal_epochs=5)
    crit = nn.CrossEntropyLoss()
    soft_crit = SoftTargetCrossEntropy()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == 'cuda')

    history = {'test_acc': [], 'swa_acc': []}
    best_swa = 0.0
    for ep in range(epochs):
        model.train()
        for x, y in tqdm(train_loader, leave=False,
                         desc=f'{name} ep{ep+1}/{epochs}'):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            if mixup_alpha > 0:
                xm, ym = mixup(x, y, mixup_alpha, 10)
                with torch.cuda.amp.autocast(enabled=use_amp and device.type == 'cuda'):
                    loss = soft_crit(model(xm), ym)
            else:
                with torch.cuda.amp.autocast(enabled=use_amp and device.type == 'cuda'):
                    loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        if ep >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            sched.step()

        _, acc = evaluate(model, test_loader, device)
        history['test_acc'].append(acc)
        if ep >= swa_start:
            update_bn(train_loader, swa_model, device=device)
            _, sacc = evaluate(swa_model, test_loader, device)
            history['swa_acc'].append(sacc)
            print(f'[{name}] ep {ep+1}/{epochs}  acc={acc:.4f}  '
                  f'swa_acc={sacc:.4f}')
            if sacc > best_swa:
                best_swa = sacc
                torch.save(swa_model.module.state_dict(),
                           os.path.join(save_dir, f'{name}_swa.pt'))
        else:
            print(f'[{name}] ep {ep+1}/{epochs}  acc={acc:.4f}')

    torch.save(model.state_dict(),
               os.path.join(save_dir, f'{name}_final.pt'))
    history['best_swa'] = best_swa
    json.dump(history, open(os.path.join(log_dir, f'{name}_swa.json'), 'w'),
              indent=2, default=float)
    return history