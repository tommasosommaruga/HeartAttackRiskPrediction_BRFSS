"""Cross validation and submission for the one hot MI pipeline.

    python -m src.experiments
    python run.py
"""

import csv
import json
import os
import sys
import time

import numpy as np

import src.model as model
import src.preprocessing as prep
from helpers import create_csv_submission

CONFIG_PATH = prep.CONFIG_PATH
RESULTS_PATH = "pipeline_experiments.csv"
IMPORTANCE_PATH = "permutation_importance.csv"
SUBMISSION_DIR = "submissions"
METRICS = (
    "pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy",
    "threshold", "n_features", "nonzero", "seconds",
)


def read_json(path):
    """Load a JSON object from path."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_arrays():
    """Training matrix and labels (cached by prep.load_raw)."""
    x, _x_test, y, _train_ids, _test_ids = prep.load_raw()
    return x, y


def load_test_arrays():
    """Test ids and features."""
    _x, x_test, _y, _train_ids, test_ids = prep.load_raw()
    return test_ids, x_test


def fit_and_tune(data, train_rows, y, cfg, rng):
    """Fit on most of train_rows; choose the F1 cutoff on the rest."""
    positive = y == 1
    fit_local, tune_local = prep.stratified_split(
        positive[train_rows], float(cfg["tune_fraction"]), rng
    )
    fit_rows, tune_rows = train_rows[fit_local], train_rows[tune_local]
    state, x_fit = prep.fit(data, fit_rows, positive[fit_rows], cfg)
    theta = model.fit_logistic(
        x_fit,
        positive[fit_rows],
        model.positive_class_weights(positive[fit_rows], float(cfg["positive_class_cost"])),
        float(cfg["l2_penalty"]),
        int(cfg["max_iters"]),
        rng,
    )
    del x_fit
    tune_scores = model.logistic_scores(prep.apply(data, tune_rows, state), theta)
    threshold, _ = model.choose_threshold(tune_scores, y[tune_rows])
    return state, theta, threshold, tune_rows, tune_scores


def run_fold(data, train_rows, test_rows, y, cfg, rng):
    """Fit on one outer training fold and score its test fold."""
    started = time.time()
    positive = y == 1
    state, theta, threshold, _tune_rows, _tune_scores = fit_and_tune(
        data, train_rows, y, cfg, rng
    )
    x_test = prep.apply(data, test_rows, state)
    test_scores = model.logistic_scores(x_test, theta)
    metrics = model.classification_scores(test_scores, y[test_rows], threshold)
    metrics.update({
        "pr_auc": model.average_precision(test_scores, positive[test_rows]),
        "roc_auc": model.auc(test_scores, positive[test_rows]),
        "threshold": threshold,
        "n_features": x_test.shape[1],
        "nonzero": int(np.count_nonzero(theta[1:])),
        "seconds": time.time() - started,
    })
    return metrics, (state, theta, x_test, test_scores)


def permutation_importance(fitted, positive, rng):
    """PR and ROC AUC drop when each raw column is shuffled."""
    state, theta, x_test, scores = fitted
    weights = theta[1:].astype(np.float32)
    base_pr = model.average_precision(scores, positive)
    base_roc = model.auc(scores, positive)
    order = rng.permutation(x_test.shape[0])
    sources = state["sources"]
    rows = []
    for source in np.unique(sources[sources >= 0]):
        columns = np.flatnonzero(sources == source)
        block = x_test[:, columns]
        shuffled = scores + (block[order] - block) @ weights[columns]
        rows.append({
            "column": prep.ensure_header()[0][source],
            "n_columns": columns.size,
            "pr_auc_drop": base_pr - model.average_precision(shuffled, positive),
            "roc_auc_drop": base_roc - model.auc(shuffled, positive),
        })
    rows.sort(key=lambda row: -row["pr_auc_drop"])
    return rows


def write_rows(rows, path):
    """Write a list of dicts as CSV."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: ("%.5f" % value if isinstance(value, float) else value)
                for key, value in row.items()
            })


def submit():
    """Fit on labelled rows and write submissions/weighted_logistic.csv."""
    cfg = read_json(CONFIG_PATH)
    x, y = load_arrays()
    test_ids, x_test = load_test_arrays()
    data = prep.make_data(x, cfg)
    rng = np.random.default_rng(int(cfg["seed"]))
    state, theta, threshold, tune_rows, tune_scores = fit_and_tune(
        data, np.arange(y.size), y, cfg, rng
    )
    tune_metrics = model.classification_scores(tune_scores, y[tune_rows], threshold)
    data["x"] = x_test
    test_scores = model.logistic_scores(
        prep.apply(data, np.arange(x_test.shape[0]), state), theta
    )
    y_pred = np.where(test_scores >= threshold, 1, -1)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    path = os.path.join(SUBMISSION_DIR, "weighted_logistic.csv")
    create_csv_submission(test_ids, y_pred, path)
    print(
        "tune f1 %.4f  prec %.4f  recall %.4f  threshold %.4f  cols %d"
        % (
            tune_metrics["f1"],
            tune_metrics["precision"],
            tune_metrics["recall"],
            threshold,
            state["sources"].size,
        ),
        flush=True,
    )
    print(
        "test rows %d  predicted positive %.2f%%  wrote %s"
        % (y_pred.size, 100.0 * np.mean(y_pred == 1), path),
        flush=True,
    )


def main():
    """Run five fold CV and write metrics plus permutation importance."""
    cfg = read_json(CONFIG_PATH)
    x, y = load_arrays()
    positive = y == 1
    data = prep.make_data(x, cfg)
    n_folds = int(cfg["n_folds"])
    fold = prep.stratified_folds(positive, n_folds, np.random.default_rng(int(cfg["seed"])))
    print(
        "%d rows, %d raw columns, %.2f%% positive, %d folds"
        % (x.shape[0], x.shape[1], 100.0 * positive.mean(), n_folds),
        flush=True,
    )
    results, importance = [], []
    for index in range(n_folds):
        rng = np.random.default_rng(int(cfg["seed"]) + index)
        train_rows = np.flatnonzero(fold != index)
        test_rows = np.flatnonzero(fold == index)
        metrics, fitted = run_fold(data, train_rows, test_rows, y, cfg, rng)
        results.append({"fold": index, **metrics})
        print(
            "fold %d  pr_auc %.4f  f1 %.4f  cols %d  %.0fs"
            % (index, metrics["pr_auc"], metrics["f1"], metrics["n_features"], metrics["seconds"]),
            flush=True,
        )
        if index == 0:
            importance = permutation_importance(fitted, positive[test_rows], rng)
            write_rows(importance, IMPORTANCE_PATH)
        del fitted
    means = {key: float(np.mean([row[key] for row in results])) for key in METRICS}
    stds = {key: float(np.std([row[key] for row in results])) for key in METRICS}
    write_rows(
        results
        + [{"fold": "mean", **means}, {"fold": "std", **stds}],
        RESULTS_PATH,
    )
    print(
        "mean  pr_auc %.4f  roc_auc %.4f  f1 %.4f  prec %.4f  recall %.4f  cols %d"
        % (
            means["pr_auc"],
            means["roc_auc"],
            means["f1"],
            means["precision"],
            means["recall"],
            int(means["n_features"]),
        )
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["submit"]:
        submit()
    else:
        main()
