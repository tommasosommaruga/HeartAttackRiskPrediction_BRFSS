
"""Classification metrics. The positive class is rare, so accuracy alone is not enough."""

import numpy as np


def classification_metrics(probs, y, threshold=0.5):
    """Accuracy, balanced accuracy, precision, recall and F1 of the positive class.

    Args:
        probs: shape=(N,), predicted probabilities of the positive class.
        y: shape=(N,), labels in {0, 1}.
        threshold: probability above which a prediction is positive.

    Returns:
        dict with the metrics.
    """
    pred = probs >= threshold
    truth = y == 1
    tp = np.sum(pred & truth)
    precision = tp / max(np.sum(pred), 1)
    recall = tp / max(np.sum(truth), 1)
    specificity = np.mean(~pred[~truth]) if np.any(~truth) else 0.0
    return {
        "accuracy": np.mean(pred == truth),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


# Thresholds tried when the decision threshold is picked on validation
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)

# Names accepted by --threshold to pick it on validation, and the metric each one maximizes
THRESHOLD_METRICS = {"best_f1": "f1", "best_bal_acc": "balanced_accuracy"}


def threshold_table(probs, y, grid=THRESHOLD_GRID):
    """Metrics for each threshold of the grid (list of dicts, with "threshold" and "pos_rate")."""
    rows = []
    for t in grid:
        m = classification_metrics(probs, y, t)
        m["threshold"] = t
        m["pos_rate"] = np.mean(probs >= t)
        rows.append(m)
    return rows


def best_threshold(probs, y, metric="f1", grid=THRESHOLD_GRID):
    """Threshold of the grid that maximizes `metric`. Returns (threshold, metrics)."""
    best = max(threshold_table(probs, y, grid), key=lambda r: r[metric])
    return best["threshold"], best
