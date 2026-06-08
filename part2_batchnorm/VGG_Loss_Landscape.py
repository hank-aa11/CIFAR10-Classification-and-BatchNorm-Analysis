"""Section 2.3: train VGG_A and VGG_A_BatchNorm at multiple LRs and record
per-step quantities used to draw landscape / smoothness figures."""
import os
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128
NUM_WORKERS = 4
EPOCHS = 20
SEED = 2020
DATA_ROOT = './data/'
RESULT_DIR = './results'
LR_LIST = [1e-3, 2e-3, 1e-4, 5e-4]

os.makedirs(RESULT_DIR, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def flat_grad(model):
    return torch.cat([p.grad.detach().reshape(-1)
                      for p in model.parameters() if p.grad is not None])


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def train_record(model_cls, lr, epochs, train_loader, val_loader, tag):
    set_seed(SEED)
    model = model_cls().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    losses, grad_norms, cls_grad_norms = [], [], []
    grad_diffs, param_steps = [], []
    val_accs, ep_train_losses = [], []

    prev_g, prev_p = None, flat_params(model).clone()

    for ep in range(epochs):
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, leave=False,
                         desc=f'{tag} lr={lr} ep{ep+1}/{epochs}'):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()

            g = flat_grad(model).clone()
            cls_g = model.classifier[4].weight.grad.detach().reshape(-1)
            losses.append(loss.item())
            grad_norms.append(g.norm().item())
            cls_grad_norms.append(cls_g.norm().item())
            grad_diffs.append(0.0 if prev_g is None
                              else (g - prev_g).norm().item())

            optimizer.step()

            new_p = flat_params(model)
            param_steps.append((new_p - prev_p).norm().item())
            prev_p, prev_g = new_p.clone(), g
            running += loss.item()

        ep_train_losses.append(running / len(train_loader))
        val_accs.append(evaluate(model, val_loader))
        print(f'[{tag} lr={lr}] ep {ep+1}/{epochs}  '
              f'loss={ep_train_losses[-1]:.4f}  val_acc={val_accs[-1]:.4f}')

    fn = os.path.join(RESULT_DIR, f'{tag}_lr{lr}.npz')
    np.savez(fn,
             losses=np.array(losses),
             grad_norms=np.array(grad_norms),
             cls_grad_norms=np.array(cls_grad_norms),
             grad_diffs=np.array(grad_diffs),
             param_steps=np.array(param_steps),
             val_accs=np.array(val_accs),
             ep_train_losses=np.array(ep_train_losses),
             tag=tag, lr=lr)
    print(f'  saved -> {fn}')


def main():
    print('device =', DEVICE)
    train_loader = get_cifar_loader(root=DATA_ROOT, batch_size=BATCH_SIZE,
                                    train=True, num_workers=NUM_WORKERS)
    val_loader = get_cifar_loader(root=DATA_ROOT, batch_size=BATCH_SIZE,
                                  train=False, shuffle=False,
                                  num_workers=NUM_WORKERS)
    for lr in LR_LIST:
        train_record(VGG_A, lr, EPOCHS, train_loader, val_loader, tag='VGG_A')
    for lr in LR_LIST:
        train_record(VGG_A_BatchNorm, lr, EPOCHS, train_loader, val_loader,
                     tag='VGG_A_BN')


if __name__ == '__main__':
    main()