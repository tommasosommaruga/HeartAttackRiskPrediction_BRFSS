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
    """Return the fraction of each class label (minus 1 and plus 1)."""
    return {-1: float(np.mean(y == -1)), 1: float(np.mean(y == 1))}


def plot_class_balance(y, title="class balance"):
    """Bar chart of label counts."""
    fig, ax = plt.subplots()
    ax.bar(["-1", "1"], [(y == -1).sum(), (y == 1).sum()])
    ax.set_title(title)
    return ax


def plot_missing_rates(rates, title="missing rate by column"):
    """Bar chart of per column missing fractions."""
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(rates)), rates)
    ax.set_title(title)
    return ax


def plot_feature_boxplots(x, cols, sample=5000, seed=42, title="feature boxplots"):
    """Matplotlib boxplots for a few columns (row subsample)."""
    cols = np.asarray(cols, dtype=int)
    rng = np.random.default_rng(seed)
    rows = rng.choice(x.shape[0], size=min(sample, x.shape[0]), replace=False)
    data = [x[rows, j][np.isfinite(x[rows, j])] for j in cols]
    fig, ax = plt.subplots()
    ax.boxplot(data, showfliers=False)
    ax.set_xticklabels(cols)
    ax.set_title(title)
    return ax


def missing_rate_by_class(x, y):
    """Missing rate per column within each class. Dict label -> (D,)."""
    return {int(c): missing_rate(x[y == c]) for c in np.unique(y)}


def plot_missing_rate_by_class(rates_neg, rates_pos, title="missingness by class"):
    """Scatter class -1 vs class 1 missing rates."""
    fig, ax = plt.subplots()
    ax.scatter(rates_neg, rates_pos, s=10, alpha=0.5)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("missing rate (-1)")
    ax.set_ylabel("missing rate (1)")
    ax.set_title(title)
    return ax


def mean_diff_by_class(x, y, col_mask=None, max_missing=0.3):
    """Standardized class mean gap for low missing columns.

    gap = (mean_1 - mean_-1) / nanstd(column). Comparable across features.

    Output: cols (K,), gaps (K,).
    """
    if col_mask is None:
        col_mask = np.ones(x.shape[1], dtype=bool)
    cols = np.flatnonzero(col_mask & (missing_rate(x) <= max_missing))
    with np.errstate(all="ignore"):
        gap = np.nanmean(x[y == 1][:, cols], axis=0) - np.nanmean(
            x[y == -1][:, cols], axis=0
        )
        scale = np.nanstd(x[:, cols], axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return cols, gap / scale


def drop_summary(rates, constant, high_missing):
    """List of (col, missing_rate, reason) for columns to drop."""
    rows = []
    for j in np.flatnonzero(constant | high_missing):
        reason = "constant" if constant[j] else "high_missing"
        if constant[j] and high_missing[j]:
            reason = "constant+high_missing"
        rows.append((int(j), float(rates[j]), reason))
    return rows
