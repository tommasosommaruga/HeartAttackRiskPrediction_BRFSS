"""Fit cleaning stats on the fit split only. Apply them to val and test with `transform`.

Only IEEE NaN is missing. Drop constant columns, median impute, then standardize.
"""

import numpy as np


def column_medians(x):
    """Median of each column, ignoring NaN. All NaN column falls back to 0.

    Input: x (N, D).
    Output: medians (D,).
    """
    raise NotImplementedError


def impute_medians(x, medians):
    """Fill NaNs with the given column medians. Leaves `x` unchanged.

    Input: x (N, D), medians (D,).
    Output: filled copy (N, D).
    """
    raise NotImplementedError


def standardization_stats(x, eps):
    """Mean and std of an imputed matrix. Std below `eps` becomes 1.

    Input: x (N, D) with no NaNs, eps float.
    Output: means (D,), scales (D,).
    """
    raise NotImplementedError


def standardize(x, means, scales):
    """Return (x - means) / scales using stored stats.

    Input: x (N, D), means (D,), scales (D,).
    Output: standardized array (N, D).
    """
    raise NotImplementedError


def fit_preprocessor(x, eps):
    """Fit drop mask, medians, means, and scales on the fit matrix `x`.

    Input: x (N, D), eps float.
    Output: dict with drop (D,), and medians, means, scales each (D_kept,).
    """
    raise NotImplementedError


def transform(x, state):
    """Drop, impute, and standardize using a fitted `state`. Do not refit.

    Input: x (N, D), state from fit_preprocessor.
    Output: cleaned array (N, D_kept).
    """
    raise NotImplementedError
