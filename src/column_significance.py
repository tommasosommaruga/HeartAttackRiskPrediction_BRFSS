"""Rank columns by the relation that disappears when the column is left out.

One weighted least squares fit, with a bias column, produces every
leave one column score. Removing column j and refitting raises the residual
sum of squares by w_j^2 / (G^{-1})_{jj}. That increase is the part of the
target this column still carries after the other columns have been used.
Dividing by the reduced residual sum of squares gives a partial R^2.

Rows use `positive_class_weights`, so the relation matches the training loss.
"""

import csv

import numpy as np

from src.positive_class_weights import positive_class_weights


def leave_one_column_relation(x, y, sample_weights=None):
    """Partial R^2 of each column from one leave one column calculation.

    Input: x (N, D) standardized, y (N,) in {-1, +1} or {0, 1},
    sample_weights (N,) or None. None builds `positive_class_weights`.
    Output: partial_r2 (D,), delta_rss (D,). Both are large when dropping
    the column hurts the weighted fit.
    """
    features = np.asarray(x, dtype=float)
    labels = np.asarray(y, dtype=float).reshape(-1)
    n_rows, n_features = features.shape
    if labels.shape[0] != n_rows:
        raise ValueError("y must have one label per row")
    if sample_weights is None:
        sample_weights = positive_class_weights(labels)
    weights = np.asarray(sample_weights, dtype=float).reshape(-1)
    if weights.shape[0] != n_rows:
        raise ValueError("sample_weights must have one value per row")

    scale = np.sqrt(weights)
    scaled_x = features * scale[:, np.newaxis]
    scaled_y = labels * scale
    gram = np.empty((n_features + 1, n_features + 1), dtype=float)
    gram[0, 0] = np.dot(scale, scale)
    gram[0, 1:] = scale @ scaled_x
    gram[1:, 0] = gram[0, 1:]
    gram[1:, 1:] = scaled_x.T @ scaled_x
    rhs = np.empty(n_features + 1, dtype=float)
    rhs[0] = np.dot(scale, scaled_y)
    rhs[1:] = scaled_x.T @ scaled_y

    solved = _solvable_gram(gram)
    coefficients = np.linalg.solve(solved, rhs)
    inverse_diag = np.diag(np.linalg.inv(solved))
    # Weighted residual sum of squares at these coefficients. When `solved`
    # is `gram`, this equals ||y||^2 - b·w and the column deltas below match
    # a fresh fit with that column removed.
    residual = (
        np.dot(scaled_y, scaled_y)
        - 2.0 * np.dot(rhs, coefficients)
        + float(coefficients @ gram @ coefficients)
    )
    residual = max(float(residual), 0.0)
    delta = coefficients[1:] ** 2 / inverse_diag[1:]
    delta = np.maximum(delta, 0.0)
    partial_r2 = delta / (residual + delta)
    return partial_r2, delta


def write_column_significance(
    path, original_columns, marginal_gap, partial_r2, kept, feature_names=None
):
    """Write one row per cleaned column, strongest partial R^2 first.

    Input: path, original_columns (D,), marginal_gap (D,), partial_r2 (D,),
    kept (D,) bool, feature_names length D or None. The marginal gap is the
    class mean difference. The partial R^2 is the leave one column relation.
    This file is a report. The caller decides which rows stay in the model.
    """
    original_columns = np.asarray(original_columns)
    order = np.argsort(-np.asarray(partial_r2), kind="mergesort")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["original_column", "marginal_gap", "partial_r2", "kept"]
        if feature_names is not None:
            header = ["feature", *header]
        writer.writerow(header)
        for index in order:
            row = [
                int(original_columns[index]),
                f"{float(marginal_gap[index]):.6f}",
                f"{float(partial_r2[index]):.6e}",
                int(bool(kept[index])),
            ]
            if feature_names is not None:
                row = [feature_names[index], *row]
            writer.writerow(row)


def _solvable_gram(gram):
    """Return `gram`, or a diagonally nudged copy when it is singular."""
    try:
        np.linalg.cholesky(gram)
        return gram
    except np.linalg.LinAlgError:
        nudged = gram.copy()
        nudge = 1e-8 * np.trace(nudged) / nudged.shape[0]
        nudged.flat[:: nudged.shape[0] + 1] += nudge
        return nudged
