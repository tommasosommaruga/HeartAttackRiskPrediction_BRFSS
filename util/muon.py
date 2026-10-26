# -*- coding: utf-8 -*-
"""Muon optimizer (MomentUm Orthogonalized by Newton-schulz) in pure NumPy."""

import numpy as np

from util.adam import Adam


def newton_schulz(g, steps=5, eps=1e-7):
    """Approximate the orthogonal factor U V^T of g = U S V^T.

    Uses the quintic Newton-Schulz iteration from the Muon reference
    implementation; the coefficients are tuned to converge fast, not exactly.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g / (np.linalg.norm(g) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T  # iterate on the smaller Gram matrix
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if transposed else x


class Muon:
    """Muon optimizer.

    Params flagged with `muon=True` (hidden 2D weights) receive a Nesterov
    momentum update that is orthogonalized with Newton-Schulz. All the other
    params (biases, BatchNorm, output layer) are handled by an internal Adam,
    as in the reference Muon recipe.

    Args:
        params: list of Param.
        lr: learning rate for the Muon params.
        momentum: momentum coefficient.
        nesterov: use Nesterov momentum.
        ns_steps: number of Newton-Schulz iterations.
        weight_decay: decoupled weight decay for the Muon params.
        adam_lr, adam_betas, adam_weight_decay: settings of the internal Adam.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        weight_decay=0.0,
        adam_lr=1e-3,
        adam_betas=(0.9, 0.999),
        adam_weight_decay=0.0,
    ):
        params = list(params)
        self.muon_params = [p for p in params if p.muon and p.data.ndim == 2]
        other = [p for p in params if not (p.muon and p.data.ndim == 2)]
        self.adam = Adam(other, lr=adam_lr, betas=adam_betas, weight_decay=adam_weight_decay)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.ns_steps = ns_steps
        self.weight_decay = weight_decay
        self.buffers = [np.zeros_like(p.data) for p in self.muon_params]

    def zero_grad(self):
        for p in self.muon_params:
            p.grad.fill(0.0)
        self.adam.zero_grad()

    def step(self):
        for p, buf in zip(self.muon_params, self.buffers):
            buf *= self.momentum
            buf += p.grad
            g = p.grad + self.momentum * buf if self.nesterov else buf
            update = newton_schulz(g, self.ns_steps)
            # W has shape (fan_in, fan_out): scale up "wide" layers so the
            # update RMS does not shrink with the aspect ratio
            scale = max(1.0, p.data.shape[1] / p.data.shape[0]) ** 0.5
            if self.weight_decay:
                p.data -= self.lr * self.weight_decay * p.data
            p.data -= self.lr * scale * update
        self.adam.step()
