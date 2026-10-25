"""Readonly diagnostics for the BRFSS training matrix."""

import matplotlib.pyplot as plt
import numpy as np


def missing_count(x):
    """Count NaNs in each column. Returns shape (D,)."""
    return np.isnan(x).sum(axis=0)


def missing_rate(x):
    """Fraction of NaNs in each column. Returns shape (D,) in [0, 1]."""
    return missing_count(x) / x.shape[0]


def count_unique(x):
    """Count distinct finite values per column. NaNs are ignored."""
    return np.array(
        [np.unique(col[~np.isnan(col)]).size for col in x.T],
        dtype=int,
    )


def constant_mask(x):
    """True for columns with one finite value, or all NaN."""
    with np.errstate(all="ignore"):
        same = np.nanmax(x, axis=0) == np.nanmin(x, axis=0)
    all_nan = np.isnan(x).all(axis=0)
    return same | all_nan


def classify_columns(x, continuous_min_unique=20, max_std=1e-8):
    """Split columns by how many distinct values they have.

    Nearly constant means nanstd is below `max_std` (includes true constants).
    """
    n_unique = count_unique(x)
    with np.errstate(all="ignore"):
        std = np.nanstd(x, axis=0)
    std = np.where(np.isnan(std), 0.0, std)

    constant = n_unique <= 1
    binary = n_unique == 2
    continuous = n_unique >= continuous_min_unique
    ordinal = (~constant) & (~binary) & (~continuous)
    quasi_constant = std < max_std

    return {
        "constant": constant,
        "binary": binary,
        "ordinal": ordinal,
        "continuous": continuous,
        "quasi_constant": quasi_constant,
    }


def class_proportions(y):
    """Return the fraction of each class label (- 1 and +1)."""
    return {
        -1: float(np.mean(y == -1)),
        1: float(np.mean(y == 1)),
    }


def plot_class_balance(y):
    """Bar chart of label counts for -1 and 1."""
    labels = np.array([-1, 1])
    counts = np.array([(y == label).sum() for label in labels])
    fig, ax = plt.subplots()
    ax.bar(labels.astype(str), counts)
    ax.set_xlabel("label")
    ax.set_ylabel("count")
    ax.set_title("class balance")
    return ax


def plot_missing_rates(rates):
    """Bar chart of per-column missing fractions."""
    fig, ax = plt.subplots()
    ax.bar(np.arange(rates.shape[0]), rates)
    ax.set_xlabel("column")
    ax.set_ylabel("missing rate")
    ax.set_title("missingness by feature")
    return ax
