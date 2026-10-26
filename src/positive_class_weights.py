"""Give target 1 more weight in the training loss than the negative class.

Missing a positive MICHD case is worse than a false alarm, and positives are
rare. `positive_class_weights` sets the total weight of target 1 to
`POSITIVE_CLASS_COST` times the negative class. The original rows are kept.

Linear models keep the course solvers. Each row is scaled by the square root
of its weight, which is weighted least squares. Logistic models use a weighted
negative log likelihood, because scaling rows would change the sigmoid input.
"""

import numpy as np

from implementations import (
    least_squares,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    ridge_regression,
)

# Total weight of target 1, divided by the total weight of the other class.
# 1 would make the classes weigh the same. Above 1, a missed positive costs more.
POSITIVE_CLASS_COST = 2.0

_SCORE_CLIP = 40.0


def positive_class_weights(y, positive_cost=POSITIVE_CLASS_COST):
    """Per row weights so target 1 has `positive_cost` times the other total.

    Labels may be {-1, +1} or {0, 1}. The positive label is 1. Every other
    label shares the remaining mass. Weights then average to 1, so step sizes
    and penalties stay on the same scale as an unweighted loss.

    Input: y (N,), positive_cost > 0.
    Output: weights (N,).
    """
    labels = np.asarray(y).reshape(-1)
    if not np.isfinite(positive_cost) or positive_cost <= 0:
        raise ValueError("positive_cost must be a finite value above 0")
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = int(labels.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("both target 1 and the other class are required")
    # Class totals are positive_cost and 1 before the mean is set to 1.
    raw = np.where(positive, positive_cost / n_pos, 1.0 / n_neg)
    return raw / raw.mean()


def sqrt_weight_rows(y, tx, sample_weights):
    """Scale labels and rows by the square root of `sample_weights`.

    Unweighted least squares on the scaled arrays is weighted least squares
    on the original ones. The bias column is scaled too.

    Input: y (N,), tx (N, D), sample_weights (N,) averaging to 1.
    Output: y_weighted (N,), tx_weighted (N, D).
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    tx = np.asarray(tx, dtype=float)
    scale = np.sqrt(_as_weights(sample_weights, y.shape[0]))
    return y * scale, tx * scale[:, np.newaxis]


def fit_positive_class_weights(
    kind, y_pm1, y01, tx, sample_weights, lambda_, gamma, max_iters
):
    """Fit one course model with `sample_weights` from `positive_class_weights`.

    `y_pm1` is {-1, +1} for least squares, ridge, and gradient descent.
    `y01` is {0, 1} for logistic regression. `sample_weights` follows the
    same row order. Returns `(w, loss)` like the course functions. The
    logistic loss is the weighted negative log likelihood. The linear loss
    is the course loss on the square root scaled rows.

    Input: kind string, y_pm1 (N,), y01 (N,), tx (N, D), sample_weights (N,),
    lambda_ float, gamma float, max_iters int.
    Output: w (D,), loss float.
    """
    y_pm1 = np.asarray(y_pm1, dtype=float).reshape(-1)
    y01 = np.asarray(y01, dtype=float).reshape(-1)
    tx = np.asarray(tx, dtype=float)
    sample_weights = _as_weights(sample_weights, y_pm1.shape[0])
    if kind in ("least_squares", "ridge", "gd", "sgd"):
        y_w, tx_w = sqrt_weight_rows(y_pm1, tx, sample_weights)
        if kind == "least_squares":
            return least_squares(y_w, tx_w)
        if kind == "ridge":
            return ridge_regression(y_w, tx_w, lambda_)
        w0 = np.zeros(tx.shape[1])
        if kind == "gd":
            return mean_squared_error_gd(y_w, tx_w, w0, max_iters, gamma)
        return mean_squared_error_sgd(y_w, tx_w, w0, max_iters, gamma)
    if kind == "logistic":
        return _logistic_regression_weighted(y01, tx, sample_weights, 0.0, gamma, max_iters)
    if kind == "reg_logistic":
        return _logistic_regression_weighted(
            y01, tx, sample_weights, lambda_, gamma, max_iters
        )
    raise ValueError("unknown model: " + kind)


def _as_weights(sample_weights, n_rows):
    weights = np.asarray(sample_weights, dtype=float).reshape(-1)
    if weights.shape[0] != n_rows:
        raise ValueError("sample_weights must have one value per row")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("sample_weights must be finite and non negative")
    return weights


def _clipped_sigmoid(scores):
    # exp overflows for large negative scores. The clip keeps the sigmoid finite.
    clipped = np.clip(scores, -_SCORE_CLIP, _SCORE_CLIP)
    return 1.0 / (1.0 + np.exp(-clipped))


def _weighted_logistic_loss(y, tx, w, sample_weights):
    """Mean weighted negative log likelihood. Penalty is not included."""
    scores = np.clip(tx @ w, -_SCORE_CLIP, _SCORE_CLIP)
    softplus = np.maximum(scores, 0.0) + np.log1p(np.exp(-np.abs(scores)))
    nll = softplus - y * scores
    return float(np.mean(sample_weights * nll))


def _weighted_logistic_gradient(y, tx, w, sample_weights, lambda_):
    """Gradient of the weighted logistic loss plus `2 * lambda_ * w`."""
    residual = sample_weights * (_clipped_sigmoid(tx @ w) - y)
    gradient = tx.T @ residual / y.shape[0]
    if lambda_ != 0.0:
        gradient = gradient + 2.0 * lambda_ * w
    return gradient


def _logistic_regression_weighted(y, tx, sample_weights, lambda_, gamma, max_iters):
    """Gradient descent for weighted logistic regression. Stays quiet."""
    w = np.zeros(tx.shape[1], dtype=float)
    threshold = 1e-8
    previous = None
    for _ in range(max_iters):
        loss = _weighted_logistic_loss(y, tx, w, sample_weights)
        gradient = _weighted_logistic_gradient(y, tx, w, sample_weights, lambda_)
        w = w - gamma * gradient
        if previous is not None and abs(loss - previous) < threshold:
            break
        previous = loss
    return w, _weighted_logistic_loss(y, tx, w, sample_weights)
