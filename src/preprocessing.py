"""Fit cleaning stats on the fit split only. Apply them to val and test with `transform`.

Only IEEE NaN is missing. Drop constant and high missing columns, median impute, then standardize.
"""

import numpy as np

from src.eda import constant_mask, missing_rate


def stratified_split(y, val_frac, rng):
    """Stratified fit and val indices so each class keeps the same ratio.

    Input: y 1D labels (any hashable values), val_frac in (0, 1), rng Generator.
    Output: fit_idx (N_fit,), val_idx (N_val,).
    """
    fit_parts, val_parts = [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_val = int(val_frac * len(idx))
        val_parts.append(idx[:n_val])
        fit_parts.append(idx[n_val:])
    fit_idx = np.concatenate(fit_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(fit_idx)
    rng.shuffle(val_idx)
    return fit_idx, val_idx


def column_medians(x):
    """Median of each column, ignoring NaN. All NaN column falls back to 0.

    Input: x (N, D).
    Output: medians (D,).
    """
    med = np.nanmedian(x, axis=0)
    return np.where(np.isfinite(med), med, 0.0)


def impute_medians(x, medians):
    """Fill NaNs with the given column medians. Leaves `x` unchanged.

    Input: x (N, D), medians (D,).
    Output: filled copy (N, D).
    """
    z = x.copy()
    nan = np.isnan(z)
    z[nan] = np.take(medians, np.where(nan)[1])
    return z


def standardization_stats(x, eps):
    """Mean and std of an imputed matrix. Std below `eps` becomes 1.

    Input: x (N, D) with no NaNs, eps float.
    Output: means (D,), scales (D,).
    """
    means = np.mean(x, axis=0)
    scales = np.std(x, axis=0)
    scales = np.where(scales < eps, 1.0, scales)
    return means, scales


def standardize(x, means, scales):
    """Return (x - means) / scales using stored stats.

    Input: x (N, D), means (D,), scales (D,).
    Output: standardized array (N, D).
    """
    return (x - means) / scales


def high_missing_mask(x, max_missing_rate=0.5):
    """True for columns whose missing rate is above `max_missing_rate`.

    Input: x (N, D), max_missing_rate in [0, 1].
    Output: bool mask (D,).
    """
    return missing_rate(x) > max_missing_rate


def fit_preprocessor(x, eps, max_missing_rate=0.5):
    """Fit drop mask, medians, means, and scales on the fit matrix `x`.

    Drops constant columns and columns flagged by `high_missing_mask`.

    Input: x (N, D), eps float, max_missing_rate in [0, 1].
    Output: dict with drop (D,), and medians, means, scales each (D_kept,).
    """
    drop = constant_mask(x) | high_missing_mask(x, max_missing_rate)
    kept = x[:, ~drop]
    medians = column_medians(kept)
    filled = impute_medians(kept, medians)
    means, scales = standardization_stats(filled, eps)
    return {"drop": drop, "medians": medians, "means": means, "scales": scales}


def transform(x, state):
    """Drop, impute, and standardize using a fitted `state`. Do not refit.

    Input: x (N, D), state from fit_preprocessor.
    Output: cleaned array (N, D_kept).
    """
    kept = x[:, ~state["drop"]]
    filled = impute_medians(kept, state["medians"])
    return standardize(filled, state["means"], state["scales"])
