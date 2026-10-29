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


def best_threshold(probs, y, grid=THRESHOLD_GRID):
    """Threshold of the grid that maximizes F1. Returns (threshold, metrics)."""
    rows = [classification_metrics(probs, y, t) for t in grid]
    best = max(range(len(grid)), key=lambda i: rows[i]["f1"])
    return grid[best], rows[best]
