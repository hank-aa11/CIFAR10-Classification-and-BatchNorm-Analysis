"""Hand-rolled optimizers using only torch tensor primitives."""
import torch


class MyMomentumSGD:
    """SGD with Nesterov momentum and decoupled weight decay."""

    def __init__(self, params, lr=0.1, momentum=0.9,
                 weight_decay=5e-4, nesterov=True):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.state = {}                       # lazy init in step()

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            if self.weight_decay != 0:
                g = g + self.weight_decay * p.data
            key = id(p)
            buf = self.state.get(key)
            if buf is None or buf.device != p.device:
                buf = torch.zeros_like(p.data)
                self.state[key] = buf
            buf.mul_(self.momentum).add_(g)
            update = g + self.momentum * buf if self.nesterov else buf
            p.data.add_(update, alpha=-self.lr)


class MyAdamW:
    """Adam + decoupled weight decay (Loshchilov & Hutter 2019)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=1e-2):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}                       # lazy init in step()

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            key = id(p)
            st = self.state.get(key)
            if st is None or st['m'].device != p.device:
                st = {'m': torch.zeros_like(p.data),
                      'v': torch.zeros_like(p.data),
                      't': 0}
                self.state[key] = st
            st['t'] += 1
            t = st['t']
            m, v = st['m'], st['v']
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            m_hat = m / (1 - self.b1 ** t)
            v_hat = v / (1 - self.b2 ** t)
            if self.weight_decay != 0:
                p.data.add_(p.data, alpha=-self.lr * self.weight_decay)
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps),
                            value=-self.lr)