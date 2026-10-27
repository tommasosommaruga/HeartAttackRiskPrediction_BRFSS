"""Log transform of the long-tailed columns, fitted on the fit split only.

Standardization only shifts and rescales, so a few huge values (e.g. 53460
minutes of activity a week) stay huge and squeeze all the normal values
together. log(1 + x) compresses the tail and keeps the order of the values.
"""

import numpy as np


def skewness(col):
    """Sample skewness of a 1D array, ignoring NaN (0 for a constant column)."""
    v = col[~np.isnan(col)]
    std = v.std() if v.size else 0.0
    if std == 0:
        return 0.0
    return np.mean(((v - v.mean()) / std) ** 3)


def fit_log_columns(x, min_unique=10, min_skew=2.0):
    """Mask of the columns that get log(1 + x).

    A column qualifies if its values are >= 0, it has at least `min_unique`
    distinct values (not a yes/no or 1-5 answer) and a right tail with
    skewness above `min_skew`.

    Input: x (N, D) fit matrix (NaN allowed).
    Output: bool mask (D,).
    """
    mask = np.zeros(x.shape[1], dtype=bool)
    for j in range(x.shape[1]):
        v = x[:, j][~np.isnan(x[:, j])]
        if v.size == 0 or v.min() < 0 or np.unique(v).size < min_unique:
            continue
        mask[j] = skewness(v) > min_skew
    return mask


def apply_log(x, mask):
    """Copy of x with log(1 + x) on the masked columns (NaN stays NaN)."""
    z = x.copy()
    z[:, mask] = np.log1p(z[:, mask])
    return z
