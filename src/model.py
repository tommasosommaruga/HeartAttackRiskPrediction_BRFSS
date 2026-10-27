"""Weighted logistic regression (FISTA) and classification scores."""

import numpy as np

SCORE_CLIP = 40.0
POWER_STEPS = 30
LIPSCHITZ_MARGIN = 1.1


def positive_class_weights(positive, positive_cost):
    """Row weights so positives total positive_cost times the negatives."""
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    raw = np.where(positive, positive_cost / n_pos, 1.0 / n_neg)
    return raw / raw.mean()


def logistic_scores(x, theta):
    """Linear scores from features and weights (bias in theta[0])."""
    return x @ theta[1:].astype(np.float32) + np.float32(theta[0])


def largest_curvature(x, weights, rng):
    """Power iteration estimate of the top Hessian eigenvalue."""
    vector = rng.standard_normal(x.shape[1] + 1)
    value = 0.0
    for _ in range(POWER_STEPS):
        vector /= np.linalg.norm(vector)
        weighted = weights * logistic_scores(x, vector)
        image = np.concatenate([[weighted.sum()], x.T @ weighted]) / x.shape[0]
        value = float(vector @ image)
        vector = image
    return value


def logistic_gradient(x, y01, weights, theta, l2):
    """Gradient of the weighted logistic loss plus L2 on non bias weights."""
    scores = np.clip(logistic_scores(x, theta), -SCORE_CLIP, SCORE_CLIP)
    residual = (weights * (1.0 / (1.0 + np.exp(-scores)) - y01)).astype(np.float32)
    gradient = np.empty(theta.size)
    gradient[0] = residual.sum() / x.shape[0]
    gradient[1:] = (x.T @ residual) / x.shape[0] + 2.0 * l2 * theta[1:]
    return gradient


def fit_logistic(x, y01, weights, l2, max_iters, rng):
    """FISTA fit of weighted logistic regression with L2."""
    weights = weights.astype(np.float32)
    y01 = y01.astype(np.float32)
    step = 1.0 / (0.25 * LIPSCHITZ_MARGIN * largest_curvature(x, weights, rng) + 2.0 * l2)
    theta = np.zeros(x.shape[1] + 1)
    rate = float(np.sum(weights * y01) / np.sum(weights))
    theta[0] = np.log(rate / (1.0 - rate))
    look = theta.copy()
    momentum = 1.0
    for _ in range(max_iters):
        moved = look - step * logistic_gradient(x, y01, weights, look, l2)
        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum ** 2)) / 2.0
        look = moved + ((momentum - 1.0) / next_momentum) * (moved - theta)
        theta, momentum = moved, next_momentum
    return theta


def fit_logistic_sgd(tx, y01, max_iters, gamma, batch_size, rng, l2=0.0):
    """Mini batch SGD logistic regression on a matrix that already has a bias column."""
    n, d = tx.shape
    w = np.zeros(d)
    y01 = y01.astype(float)
    for _ in range(max_iters):
        idx = rng.integers(0, n, size=batch_size)
        xb, yb = tx[idx], y01[idx]
        scores = np.clip(xb @ w, -SCORE_CLIP, SCORE_CLIP)
        residual = 1.0 / (1.0 + np.exp(-scores)) - yb
        grad = xb.T @ residual / batch_size + 2.0 * l2 * w
        w -= gamma * grad
    return w


def classification_scores(scores, y_pm1, threshold):
    """Accuracy, F1, precision, and recall for labels in {-1, +1}."""
    pred = np.where(scores >= threshold, 1, -1)
    positive = y_pm1 == 1
    tp = float(np.sum((pred == 1) & positive))
    fp = float(np.sum((pred == 1) & ~positive))
    fn = float(np.sum((pred == -1) & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": float(np.mean(pred == y_pm1)),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def choose_threshold(scores, y_pm1):
    """Score cut that maximizes F1 (ties keep higher accuracy)."""
    candidates = np.concatenate([np.quantile(scores, np.linspace(0.0, 1.0, 201)), [0.0]])
    best_threshold, best_f1, best_accuracy = 0.0, -1.0, -1.0
    for threshold in np.unique(candidates):
        metrics = classification_scores(scores, y_pm1, float(threshold))
        tied = metrics["f1"] == best_f1 and metrics["accuracy"] > best_accuracy
        if metrics["f1"] > best_f1 or tied:
            best_f1, best_accuracy = metrics["f1"], metrics["accuracy"]
            best_threshold = float(threshold)
    return best_threshold, best_f1


def auc(score, positive):
    """ROC AUC with average ranks for ties."""
    _values, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2.0)[inverse]
    n_pos = positive.sum()
    n_neg = positive.size - n_pos
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(score, positive):
    """Area under the precision recall curve."""
    hits = positive[np.argsort(-score, kind="stable")]
    hit_ranks = np.flatnonzero(hits) + 1
    return float(np.mean(np.arange(1, hit_ranks.size + 1) / hit_ranks))
