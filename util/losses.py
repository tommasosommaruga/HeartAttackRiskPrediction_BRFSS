"""
Binary cross-entropy computed on the logits.

loss_fn(logits, y) -> (loss, grad), where y is in {0, 1}, loss is the
scalar mean and grad (N,) is d loss / d logits.
"""

import numpy as np


def _log_sigmoid(z):
    """Stable log(sigmoid(z)) = -log(1 + e^-z)."""
    return -np.logaddexp(0.0, -z)


class BCELoss:
    """Binary cross-entropy, with an optional weight on the positive class.

    Args:
        pos_weight: weight of the positive examples (>1 helps with imbalance).
    """

    def __init__(self, pos_weight=1.0):
        self.pos_weight = pos_weight

    def __call__(self, logits, y):
        weights = np.where(y == 1, self.pos_weight, 1.0)
        losses = -weights * (y * _log_sigmoid(logits) + (1 - y) * _log_sigmoid(-logits))
        probs = np.exp(_log_sigmoid(logits))
        grad = weights * (probs - y) / y.shape[0]
        return losses.mean(), grad
