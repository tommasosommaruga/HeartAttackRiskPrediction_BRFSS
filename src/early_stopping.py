"""Stop gradient methods when the validation score stops rising.

Least squares and ridge have no steps. The other models may run up to
`MAX_ITERS`. The weights kept are the checkpoint with the best validation
score. That score is AUC, or class-1 F1 when the config asks for it.

Full batch updates can run on a CUDA device when CuPy is installed and
`use_gpu` is true. Stochastic gradient descent stays on the CPU because each
step uses one row.
"""

import contextlib
import io

import numpy as np

from implementations import compute_loss, mean_squared_error_sgd
from src.positive_class_weights import _weighted_logistic_loss, sqrt_weight_rows

MAX_ITERS = {
    "gd": 2000,
    "sgd": 200000,
    "logistic": 500,
    "reg_logistic": 500,
}
EVAL_EVERY = {
    "gd": 25,
    "sgd": 5000,
    "logistic": 10,
    "reg_logistic": 10,
}
PATIENCE = 5
MIN_AUC_GAIN = 1e-4
ITERATIVE = ("gd", "sgd", "logistic", "reg_logistic")


def validation_auc(scores, y_pm1):
    """Area under the ROC curve for labels in {-1, +1}.

    Input: scores (N,), y_pm1 (N,).
    Output: float in [0, 1]. Tied scores share the average rank.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(y_pm1).reshape(-1)
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0 or not np.isfinite(scores).all():
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    sum_pos = float(ranks[positive].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def validation_f1(scores, y_pm1):
    """Best class-1 F1 over a quantile grid of thresholds.

    Input: scores (N,), y_pm1 (N,) in {-1, +1}.
    Output: float in [0, 1]. The grid matches the cutoff used at the end of
    training, so stopping follows the same F1 the submission is scored on.
    """
    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(y_pm1).reshape(-1)
    if scores.shape[0] != labels.shape[0] or not np.isfinite(scores).all():
        return 0.0
    positive = labels == 1
    n_pos = int(positive.sum())
    if n_pos == 0 or n_pos == labels.size:
        return 0.0
    grid = np.linspace(0.0, 1.0, 201)
    candidates = np.unique(np.concatenate([np.quantile(scores, grid), np.array([0.0])]))
    best = 0.0
    for threshold in candidates:
        predicted = scores >= threshold
        tp = float(np.sum(predicted & positive))
        if tp == 0.0:
            continue
        fp = float(np.sum(predicted & ~positive))
        precision = tp / (tp + fp)
        recall = tp / n_pos
        f1 = 2.0 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1
    return best


def fit_iterative(
    kind,
    y_pm1,
    y01,
    tx,
    sample_weights,
    lambda_,
    gamma,
    max_iters,
    tx_val=None,
    y_val=None,
    patience=PATIENCE,
    eval_every=None,
    use_gpu=False,
    min_auc_gain=None,
    metric="auc",
):
    """Fit `kind` for up to `max_iters` steps.

    When `tx_val` and `y_val` are given, stop after `patience` checks that
    do not raise the validation score by `MIN_AUC_GAIN`. `metric` is `auc`
    or `f1`. The returned weights are the best checkpoint, and the third
    value is the step count of that checkpoint. Without validation data,
    every requested step is run.

    Output: w (D,), loss float, steps int.
    """
    if kind not in ITERATIVE:
        raise ValueError("early stopping applies only to iterative models")
    if eval_every is None:
        eval_every = EVAL_EVERY[kind]
    if min_auc_gain is None:
        min_auc_gain = MIN_AUC_GAIN
    y_pm1 = np.asarray(y_pm1, dtype=float).reshape(-1)
    y01 = np.asarray(y01, dtype=float).reshape(-1)
    tx = np.asarray(tx, dtype=float)
    xp = _array_module(use_gpu and kind != "sgd")
    if kind in ("logistic", "reg_logistic"):
        return _logistic_fit(
            y01, tx, sample_weights, lambda_, gamma, max_iters,
            tx_val, y_val, patience, eval_every, xp, min_auc_gain, metric,
        )
    y_w, tx_w = sqrt_weight_rows(y_pm1, tx, sample_weights)
    if kind == "gd":
        return _gd_fit(
            y_w, tx_w, gamma, max_iters, tx_val, y_val, patience, eval_every, xp,
            min_auc_gain, metric,
        )
    return _sgd_fit(
        y_w, tx_w, gamma, max_iters, tx_val, y_val, patience, eval_every,
        min_auc_gain, metric,
    )


def _logistic_fit(
    y, tx, sample_weights, lambda_, gamma, max_iters,
    tx_val, y_val, patience, eval_every, xp, min_auc_gain, metric,
):
    y_x = _to_module(y, xp)
    tx_x = _to_module(tx, xp)
    weights_x = _to_module(sample_weights, xp)
    w = xp.zeros(tx.shape[1], dtype=float)
    return _checkpoint_loop(
        w,
        max_iters,
        eval_every,
        patience,
        tx_val,
        y_val,
        xp,
        lambda weights, step: _logistic_chunk(
            y_x, tx_x, weights, weights_x, gamma, lambda_, step, xp,
        ),
        lambda weights: _weighted_logistic_loss(
            y, tx, _to_numpy(weights, xp), sample_weights,
        ),
        min_auc_gain,
        metric,
    )


def _gd_fit(
    y, tx, gamma, max_iters, tx_val, y_val, patience, eval_every, xp,
    min_auc_gain, metric,
):
    y_x = _to_module(y, xp)
    tx_x = _to_module(tx, xp)
    w = xp.zeros(tx.shape[1], dtype=float)
    n_rows = y.shape[0]

    def chunk(weights, step):
        for _ in range(step):
            residual = y_x - tx_x @ weights
            gradient = -(1.0 / n_rows) * (tx_x.T @ residual)
            weights = weights - gamma * gradient
        return weights

    def loss_of(weights):
        return float(compute_loss(y, tx, _to_numpy(weights, xp)))

    return _checkpoint_loop(
        w, max_iters, eval_every, patience, tx_val, y_val, xp, chunk, loss_of,
        min_auc_gain, metric,
    )


def _sgd_fit(
    y, tx, gamma, max_iters, tx_val, y_val, patience, eval_every, min_auc_gain,
    metric,
):
    w = np.zeros(tx.shape[1], dtype=float)

    def chunk(weights, step):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            updated, _loss = mean_squared_error_sgd(y, tx, weights, step, gamma)
        return updated

    def loss_of(weights):
        return float(compute_loss(y, tx, weights))

    return _checkpoint_loop(
        w, max_iters, eval_every, patience, tx_val, y_val, np, chunk, loss_of,
        min_auc_gain, metric,
    )


def _logistic_chunk(y, tx, w, sample_weights, gamma, lambda_, n_steps, xp):
    n_rows = y.shape[0]
    for _ in range(n_steps):
        scores = xp.clip(tx @ w, -40.0, 40.0)
        sigma = 1.0 / (1.0 + xp.exp(-scores))
        residual = sample_weights * (sigma - y)
        gradient = tx.T @ residual / n_rows
        if lambda_ != 0.0:
            gradient = gradient + 2.0 * lambda_ * w
        w = w - gamma * gradient
    return w


def _checkpoint_loop(
    w, max_iters, eval_every, patience, tx_val, y_val, xp, chunk, loss_of,
    min_auc_gain, metric,
):
    if tx_val is None or y_val is None:
        w = chunk(w, max_iters)
        numpy_w = _to_numpy(w, xp)
        return numpy_w, float(loss_of(numpy_w)), int(max_iters)
    if metric not in ("auc", "f1"):
        raise ValueError("early-stopping metric must be auc or f1")

    def rank_score(weights):
        scores = np.asarray(tx_val) @ weights
        if metric == "f1":
            return validation_f1(scores, y_val)
        return validation_auc(scores, y_val)

    best_w = _to_numpy(w, xp).copy()
    best_score = rank_score(best_w)
    best_step = 0
    stalls = 0
    step = 0
    while step < max_iters:
        take = min(eval_every, max_iters - step)
        w = chunk(w, take)
        step += take
        numpy_w = _to_numpy(w, xp)
        if not np.isfinite(numpy_w).all():
            break
        score = rank_score(numpy_w)
        if score > best_score + min_auc_gain:
            best_score = score
            best_w = numpy_w.copy()
            best_step = step
            stalls = 0
        else:
            stalls += 1
            if stalls >= patience:
                break
    return best_w, float(loss_of(best_w)), int(best_step)


def _array_module(use_gpu):
    if not use_gpu:
        return np
    # Local import: CuPy is optional and is absent from the course environment.
    try:
        import cupy as cp
    except ImportError:
        print("CuPy is not installed. Full batch updates stay on the CPU.")
        return np
    return cp


def _to_module(array, xp):
    if xp is np:
        return np.asarray(array, dtype=float)
    return xp.asarray(array, dtype=float)


def _to_numpy(array, xp):
    if xp is np:
        return np.asarray(array, dtype=float)
    return xp.asnumpy(array)
