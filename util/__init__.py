# -*- coding: utf-8 -*-
"""Optimizers, losses and metrics, each selectable by name."""

from util.adam import Adam
from util.losses import BCELoss, FocalLoss, MSELoss
from util.metrics import (
    THRESHOLD_METRICS,
    best_threshold,
    classification_metrics,
    threshold_table,
)

OPTIMIZERS = {"adam": Adam}
LOSSES = {"bce": BCELoss, "focal": FocalLoss, "mse": MSELoss}


def build_optimizer(name, params, **kwargs):
    """Create an optimizer by name ("adam"). It exposes zero_grad() / step()."""
    if name.lower() not in OPTIMIZERS:
        raise ValueError("unknown optimizer {!r}, choose from {}".format(name, list(OPTIMIZERS)))
    return OPTIMIZERS[name.lower()](params, **kwargs)


def build_loss(name, **kwargs):
    """Create a loss by name ("bce", "focal" or "mse"). Calling it returns (loss, grad)."""
    if name.lower() not in LOSSES:
        raise ValueError("unknown loss {!r}, choose from {}".format(name, list(LOSSES)))
    return LOSSES[name.lower()](**kwargs)
