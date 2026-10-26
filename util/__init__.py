# -*- coding: utf-8 -*-
"""Optimizers. Adam and Muon share the interface zero_grad() / step()."""

from util.adam import Adam
from util.muon import Muon


def build_optimizer(name, params, **kwargs):
    """Create an optimizer by name ("adam" or "muon") so they can be swapped."""
    optimizers = {"adam": Adam, "muon": Muon}
    if name.lower() not in optimizers:
        raise ValueError("unknown optimizer {!r}, choose 'adam' or 'muon'".format(name))
    return optimizers[name.lower()](params, **kwargs)
