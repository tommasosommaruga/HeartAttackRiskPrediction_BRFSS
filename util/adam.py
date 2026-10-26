# -*- coding: utf-8 -*-
"""Adam optimizer in pure NumPy."""

import numpy as np


class Adam:
    """Adam optimizer with optional decoupled weight decay (AdamW).

    Args:
        params: list of Param.
        lr: learning rate.
        betas: decay rates of the first and second moment estimates.
        eps: small constant for numerical stability.
        weight_decay: decoupled L2 penalty applied to the weights.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad.fill(0.0)

    def step(self):
        self.t += 1
        bias1 = 1 - self.beta1**self.t
        bias2 = 1 - self.beta2**self.t
        for p, m, v in zip(self.params, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * p.grad
            v *= self.beta2
            v += (1 - self.beta2) * p.grad**2
            if self.weight_decay:
                p.data -= self.lr * self.weight_decay * p.data
            p.data -= self.lr * (m / bias1) / (np.sqrt(v / bias2) + self.eps)
