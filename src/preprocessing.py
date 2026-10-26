"""Fit cleaning stats on the fit split only. Apply them to val and test with `transform`.

Only IEEE NaN is missing. Drop constant and high missing columns, impute, then standardize.
Imputation is median, mean, or mode. Stats are stored in `state` and never refit in `transform`.
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


def impute_values(x, fill):
    """Fill NaNs with the given column values. Leaves `x` unchanged.

    Input: x (N, D), fill (D,).
    Output: filled copy (N, D).
    """
    z = x.copy()
    nan = np.isnan(z)
    z[nan] = np.take(fill, np.where(nan)[1])
    return z


def impute_medians(x, medians):
    """Fill NaNs with the given column medians. Leaves `x` unchanged.

    Input: x (N, D), medians (D,).
    Output: filled copy (N, D).
    """
    return impute_values(x, medians)


def column_means(x):
    """Mean of each column, ignoring NaN. All NaN column falls back to 0.

    Input: x (N, D).
    Output: means (D,).
    """
    with np.errstate(all="ignore"):
        means = np.nanmean(x, axis=0)
    return np.where(np.isfinite(means), means, 0.0)


def column_modes(x):
    """Most frequent finite value of each column. All NaN column falls back to 0.

    Input: x (N, D).
    Output: modes (D,).
    """
    modes = np.zeros(x.shape[1], dtype=float)
    for j in range(x.shape[1]):
        col = x[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        values, counts = np.unique(finite, return_counts=True)
        modes[j] = values[counts.argmax()]
    return modes


def column_fill(x, impute):
    """Column fill values for `impute` in {median, mean, mode}.

    Input: x (N, D), impute string.
    Output: fill (D,).
    """
    if impute == "median":
        return column_medians(x)
    if impute == "mean":
        return column_means(x)
    if impute == "mode":
        return column_modes(x)
    raise ValueError("impute must be 'median', 'mean', or 'mode'")


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


def fit_preprocessor(x, eps, max_missing_rate=0.5, impute="median"):
    """Fit drop mask, fill values, means, and scales on the fit matrix `x`.

    Drops constant columns and columns flagged by `high_missing_mask`.
    `impute` is median, mean, or mode. `medians` in the state is the fill
    vector for that method, so older callers can keep reading that key.

    Input: x (N, D), eps float, max_missing_rate in [0, 1], impute string.
    Output: dict with drop (D,), impute, and fill, medians, means, scales (D_kept,).
    """
    drop = constant_mask(x) | high_missing_mask(x, max_missing_rate)
    kept = x[:, ~drop]
    fill = column_fill(kept, impute)
    filled = impute_values(kept, fill)
    means, scales = standardization_stats(filled, eps)
    return {
        "drop": drop,
        "impute": impute,
        "fill": fill,
        "medians": fill,
        "means": means,
        "scales": scales,
    }


def transform(x, state):
    """Drop, impute, and standardize using a fitted `state`. Do not refit.

    Input: x (N, D), state from fit_preprocessor.
    Output: cleaned array (N, D_kept).
    """
    kept = x[:, ~state["drop"]]
    fill = state["fill"] if "fill" in state else state["medians"]
    filled = impute_values(kept, fill)
    return standardize(filled, state["means"], state["scales"])
