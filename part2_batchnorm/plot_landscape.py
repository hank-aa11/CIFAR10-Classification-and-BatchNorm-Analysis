"""Generate the three Santurkar-style figures from the npz records."""
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

RESULT_DIR = './results'
FIG_DIR = './figures'
LR_LIST = [1e-3, 2e-3, 1e-4, 5e-4]
os.makedirs(FIG_DIR, exist_ok=True)


def load_runs(tag):
    L, G, D, P = [], [], [], []
    for lr in LR_LIST:
        d = np.load(os.path.join(RESULT_DIR, f'{tag}_lr{lr}.npz'))
        L.append(d['losses']); G.append(d['grad_norms'])
        D.append(d['grad_diffs']); P.append(d['param_steps'])
    n = min(len(x) for x in L)
    return (np.stack([x[:n] for x in L]),
            np.stack([x[:n] for x in G]),
            np.stack([x[:n] for x in D]),
            np.stack([x[:n] for x in P]))


def smooth(arr, k=50):
    if k <= 1:
        return arr
    out = np.zeros_like(arr)
    kernel = np.ones(k) / k
    pad = np.repeat(arr[:, :1], k - 1, axis=1)
    arr_p = np.concatenate([pad, arr], axis=1)
    for i in range(arr.shape[0]):
        out[i] = np.convolve(arr_p[i], kernel, mode='valid')[:arr.shape[1]]
    return out


def band(ax, A, B, lab_a='Standard VGG', lab_b='Standard VGG + BN',
         smooth_k=50, ylog=False):
    A_s, B_s = smooth(A, smooth_k), smooth(B, smooth_k)
    xa, xb = np.arange(A_s.shape[1]), np.arange(B_s.shape[1])
    ax.fill_between(xa, A_s.min(0), A_s.max(0),
                    color='tab:red', alpha=0.30, label=lab_a)
    ax.fill_between(xb, B_s.min(0), B_s.max(0),
                    color='tab:blue', alpha=0.30, label=lab_b)
    ax.plot(xa, A_s.mean(0), color='tab:red', lw=0.8)
    ax.plot(xb, B_s.mean(0), color='tab:blue', lw=0.8)
    if ylog:
        ax.set_yscale('log')
    ax.set_xlabel('Steps'); ax.grid(True, alpha=0.3); ax.legend()


def main():
    L_a, G_a, D_a, P_a = load_runs('VGG_A')
    L_b, G_b, D_b, P_b = load_runs('VGG_A_BN')

    fig, ax = plt.subplots(figsize=(8, 4))
    band(ax, L_a, L_b)
    ax.set_ylabel('Loss')
    ax.set_title('Loss landscape (max/min over LRs)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'loss_landscape.png'), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    band(ax, G_a, G_b)
    ax.set_ylabel(r'$\|\nabla L\|_2$')
    ax.set_title('Gradient predictiveness')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'grad_predictiveness.png'), dpi=150)
    plt.close()

    eps = 1e-12
    R_a = D_a / (P_a + eps); R_a[:, 0] = R_a[:, 1]
    R_b = D_b / (P_b + eps); R_b[:, 0] = R_b[:, 1]
    fig, ax = plt.subplots(figsize=(8, 4))
    band(ax, R_a, R_b, ylog=True)
    ax.set_ylabel(r'$\|\Delta\nabla L\| \;/\; \|\Delta\theta\|$')
    ax.set_title(r'Effective $\beta$-smoothness')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'beta_smoothness.png'), dpi=150)
    plt.close()

    print('saved figures to', FIG_DIR)


if __name__ == '__main__':
    main()