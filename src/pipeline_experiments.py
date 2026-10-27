"""Compare preprocessing pipelines on one stratified cross validation.

Every experiment sees the same outer folds. Inside each outer training fold
the rows are split again into a fit part and a tune part. Medians, levels,
bins, scales, duplicate checks, feature rankings, and the L1 fit use the fit
part only. The F1 threshold is chosen on the tune part. The outer fold is
used once, for the reported metrics.

From the project folder:

    python -m src.pipeline_experiments                 # experiments in the config
    python -m src.pipeline_experiments onehot codebook # only these

Settings are in config/experiments.json. Writes `pipeline_experiments.csv`
with one row per experiment and fold plus the mean and the standard
deviation, and `permutation_importance.csv` for fold 0 of the experiments in
`importance_experiments`.

The model is the same everywhere: weighted logistic regression with a
positive class cost, an L2 penalty, and an optional L1 penalty, fit by
accelerated proximal gradient (FISTA). Only the preprocessing changes.
"""

import csv
import json
import os
import sys
import time

import numpy as np

import src.codebook_features as codebook
import src.features as features
import src.preprocessing as prep
from helpers import load_csv_data
from run import choose_threshold, classification_scores, restore_required
from src.feature_audit import auc, categories
from src.positive_class_weights import positive_class_weights

CONFIG_PATH = os.path.join("config", "experiments.json")
FEATURES_PATH = os.path.join("config", "features.json")
DATA_DIR = os.path.join("data", "dataset")
CACHE_PATH = os.path.join("data", "train_arrays.npz")
RESULTS_PATH = "pipeline_experiments.csv"
IMPORTANCE_PATH = "permutation_importance.csv"

# Answered only by people who had the event, or started because of it.
LEAK_COLUMNS = ("HAREHAB1", "STREHAB1", "CVDASPRN", "ASPUNSAF")
AGE_INDEX = codebook.INDEX["_AGE80"]
SCORE_CLIP = 40.0
POWER_STEPS = 30
# The power iteration estimate of the largest curvature is a lower bound.
# The margin keeps the FISTA step below 1 / L.
LIPSCHITZ_MARGIN = 1.1

# name: (pipeline kind, options)
EXPERIMENTS = {
    "codebook": ("codebook", {}),
    "all_median_flags": ("numeric", {"columns": "all"}),
    "low_missing_median_flags": ("numeric", {"columns": "low_missing"}),
    "missing_flags_only": ("numeric", {"columns": "flags_only"}),
    "missing_flags_top_sources": ("numeric", {"columns": "flags_top"}),
    "onehot": ("onehot", {}),
    "onehot_mi_top": ("onehot", {"mi_top": True}),
    "onehot_l1": ("onehot", {"l1": True}),
    "onehot_age_interactions": ("onehot", {"interactions": True}),
    "codebook_plus_onehot": ("union", {}),
}
METRICS = (
    "pr_auc", "roc_auc", "f1", "precision", "recall", "accuracy",
    "threshold", "n_features", "nonzero", "seconds",
)


# Data


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_arrays():
    """Training matrix and labels. The first call caches them as .npz."""
    if os.path.exists(CACHE_PATH):
        cached = np.load(CACHE_PATH)
        return cached["x"], cached["y"]
    x, _x_test, y, _train_ids, _test_ids = load_csv_data(DATA_DIR)
    np.savez(CACHE_PATH, x=x, y=y)
    return x, y


def generic_columns(use_leak_columns):
    """Raw columns the generic pipelines may use: no survey machinery, no leaks."""
    excluded = {
        name for name, (_action, text) in codebook.DECISIONS.items()
        if "Survey machinery" in text
    }
    if not use_leak_columns:
        excluded |= set(LEAK_COLUMNS)
    return np.array(
        [index for index, name in enumerate(codebook.HEADER) if name not in excluded],
        dtype=int,
    )


def codebook_design(x, features_cfg):
    """Codebook recode with the same missing flags as `run.prepare`.

    The flag choice reads missing rates only, never labels, so computing it
    once on every training row does not leak the outer fold labels.
    """
    codebook.BMI_CAP = float(features_cfg["bmi_cap"])
    base, base_names, _source = codebook.transform_codebook(x, flag_names=())
    always = [name for name, _flag in codebook.MISSING_FLAGS]
    flags = codebook.choose_flag_names(
        base, base_names, float(features_cfg["missing_flag_min_rate"]), always
    )
    values, names, source = codebook.transform_codebook(x, flag_names=flags)
    return {"x": values, "names": names, "source": source}


def stratified_folds(positive, n_folds, rng):
    """Fold id per row. Each fold keeps the positive rate of the whole file."""
    fold = np.empty(positive.size, dtype=int)
    for label in (False, True):
        rows = np.flatnonzero(positive == label)
        rng.shuffle(rows)
        fold[rows] = np.arange(rows.size) % n_folds
    return fold


# Scores and rankings


def average_precision(score, positive):
    """Area under the precision recall curve, as the mean precision at each hit."""
    hits = positive[np.argsort(-score, kind="stable")]
    hit_ranks = np.flatnonzero(hits) + 1
    return float(np.mean(np.arange(1, hit_ranks.size + 1) / hit_ranks))


def categorical_mutual_information(codes, positive):
    """Mutual information in nats between integer codes and a binary label."""
    joint = np.bincount(
        codes * 2 + positive.astype(int), minlength=2 * (codes.max() + 1)
    ).reshape(-1, 2) / codes.size
    expected = joint.sum(axis=1, keepdims=True) @ joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    return float(np.sum(joint[nonzero] * np.log(joint[nonzero] / expected[nonzero])))


def source_mutual_information(x, rows, columns, positive):
    """Mutual information of each raw column, binned with missing as a category."""
    return np.array([
        categorical_mutual_information(categories(x[rows, column]), positive)
        for column in columns
    ])


def binary_mutual_information(raw, positive):
    """Mutual information of every 0 or 1 column with the label, vectorized."""
    n_rows = float(raw.shape[0])
    n_pos = float(positive.sum())
    on = raw.sum(axis=0, dtype=np.int64).astype(float)
    on_pos = raw[positive].sum(axis=0, dtype=np.int64).astype(float)
    cells = (
        (on_pos, on, n_pos),
        (on - on_pos, on, n_rows - n_pos),
        (n_pos - on_pos, n_rows - on, n_pos),
        (n_rows - on - n_pos + on_pos, n_rows - on, n_rows - n_pos),
    )
    information = np.zeros(raw.shape[1])
    for count, row_total, class_total in cells:
        with np.errstate(divide="ignore", invalid="ignore"):
            term = count / n_rows * np.log(count * n_rows / (row_total * class_total))
        information += np.where(count > 0, term, 0.0)
    return information


# Shared preprocessing steps


def usable_columns(x, rows, columns, variance_floor):
    """Columns with at least one value and nonzero variance on `rows`.

    Output: kept columns and their missing rates.
    """
    sub = x[np.ix_(rows, columns)]
    rate = np.isnan(sub).mean(axis=0)
    usable = rate < 1.0
    variance = np.zeros(columns.size)
    variance[usable] = np.nanvar(sub[:, usable], axis=0)
    usable &= variance > variance_floor
    return columns[usable], rate[usable]


def distinct_missing_flags(x, rows, columns):
    """Columns whose missing pattern is not constant and not already seen.

    One skipped survey module blanks many columns at once. Their flags are
    identical, so only the first one is kept.
    """
    kept, seen = [], set()
    for column in columns:
        mask = np.isnan(x[rows, column])
        if not mask.any() or mask.all():
            continue
        key = np.packbits(mask).tobytes()
        if key in seen:
            continue
        seen.add(key)
        kept.append(column)
    return np.array(kept, dtype=int)


def distinct_columns(raw):
    """Positions of 0 or 1 columns that are not constant and not duplicates."""
    counts = raw.sum(axis=0, dtype=np.int64)
    kept, seen = [], set()
    for position in range(raw.shape[1]):
        if counts[position] == 0 or counts[position] == raw.shape[0]:
            continue
        key = np.packbits(raw[:, position]).tobytes()
        if key in seen:
            continue
        seen.add(key)
        kept.append(position)
    return np.array(kept, dtype=int)


def fit_scale_in_place(matrix, eps):
    means, scales = prep.standardization_stats(matrix, eps)
    scale_in_place(matrix, means, scales)
    return means, scales


def scale_in_place(matrix, means, scales):
    matrix -= means.astype(np.float32)
    matrix /= scales.astype(np.float32)
    return matrix


# Pipeline: median imputation plus missing flags


def numeric_columns(option, data, rows, positive, cfg):
    """Value columns and flag columns for one numeric experiment."""
    present, rate = usable_columns(data["x"], rows, data["columns"], data["eps"])
    with_missing = present[rate > 0]
    if option == "all":
        return present, with_missing
    if option == "low_missing":
        low = rate < float(cfg["high_missing_rate"])
        return present[low], present[low & (rate > 0)]
    if option == "flags_only":
        return np.zeros(0, dtype=int), with_missing
    if option == "flags_top":
        information = source_mutual_information(data["x"], rows, present, positive)
        top = present[np.argsort(-information)[: int(cfg["top_sources"])]]
        return np.sort(top), with_missing
    raise ValueError("unknown numeric option: " + option)


def numeric_matrix(x, rows, state):
    values = prep.impute_values(x[np.ix_(rows, state["value_columns"])], state["medians"])
    flags = np.isnan(x[np.ix_(rows, state["flag_columns"])])
    return np.hstack([values, flags]).astype(np.float32)


def fit_numeric(data, rows, positive, cfg, options):
    value_columns, flag_candidates = numeric_columns(options["columns"], data, rows, positive, cfg)
    state = {
        "value_columns": value_columns,
        "medians": prep.column_medians(data["x"][np.ix_(rows, value_columns)]),
        "flag_columns": distinct_missing_flags(data["x"], rows, flag_candidates),
    }
    matrix = numeric_matrix(data["x"], rows, state)
    state["means"], state["scales"] = fit_scale_in_place(matrix, data["eps"])
    state["sources"] = np.concatenate([state["value_columns"], state["flag_columns"]])
    return state, matrix


def apply_numeric(data, rows, state):
    matrix = numeric_matrix(data["x"], rows, state)
    return scale_in_place(matrix, state["means"], state["scales"])


# Pipeline: every column as levels, without trusting its codes


def fit_levels(column, min_count, bins):
    """Levels of one raw column.

    Any value held by at least `min_count` rows is its own level, so codes
    such as 7, 9, 88, or 7777 get their own weight instead of a magnitude.
    The remaining values are cut into quantile bins, or pooled when rare.
    Missing is a level when it is common enough.
    """
    finite = np.isfinite(column)
    observed = column[finite]
    values, counts = np.unique(observed, return_counts=True)
    frequent = values[counts >= min_count]
    rest = observed[~np.isin(observed, frequent)]
    edges = None
    if rest.size >= min_count and np.unique(rest).size > 1:
        edges = np.unique(np.quantile(rest, np.linspace(0.0, 1.0, bins + 1)[1:-1]))
    has_missing = int((~finite).sum()) >= min_count
    width = frequent.size + int(has_missing)
    width += edges.size + 1 if edges is not None else int(rest.size > 0)
    return {
        "frequent": frequent,
        "edges": edges,
        "other": rest.size > 0,
        "missing": has_missing,
        "width": width,
    }


def encode_levels(column, level, out):
    """Write the 0 or 1 level columns of one raw column into `out`."""
    position = 0
    for value in level["frequent"]:
        out[:, position] = column == value
        position += 1
    finite = np.isfinite(column)
    rest = finite & ~np.isin(column, level["frequent"])
    if level["edges"] is not None:
        rest_rows = np.flatnonzero(rest)
        bins = np.searchsorted(level["edges"], column[rest_rows], side="right")
        out[rest_rows, position + bins] = 1
        position += level["edges"].size + 1
    elif level["other"]:
        out[:, position] = rest
        position += 1
    if level["missing"]:
        out[:, position] = ~finite


def onehot_raw(x, rows, state):
    """Every level of every column, as uint8 to keep memory low."""
    widths = [level["width"] for level in state["levels"]]
    raw = np.zeros((rows.size, sum(widths)), dtype=np.uint8)
    start = 0
    for column, level in zip(state["columns"], state["levels"]):
        encode_levels(x[rows, column], level, raw[:, start : start + level["width"]])
        start += level["width"]
    return raw


def onehot_matrix(x, rows, state, selected_raw=None):
    """Selected levels, then their products with the age cutoff flag."""
    if selected_raw is None:
        selected_raw = onehot_raw(x, rows, state)[:, state["selected"]]
    blocks = [selected_raw]
    if state["interactions"].size:
        older = (x[rows, AGE_INDEX] >= state["age_cutoff"]).astype(np.uint8)
        blocks.append(selected_raw[:, state["interactions"]] * older[:, np.newaxis])
    return np.hstack(blocks).astype(np.float32)


def fit_onehot(data, rows, positive, cfg, options):
    x = data["x"]
    present, _rate = usable_columns(x, rows, data["columns"], data["eps"])
    min_count = max(1, int(round(float(cfg["min_level_share"]) * rows.size)))
    levels = [fit_levels(x[rows, column], min_count, int(cfg["quantile_bins"])) for column in present]
    state = {
        "columns": present,
        "levels": levels,
        "age_cutoff": float(cfg["age_cutoff"]),
        "interactions": np.zeros(0, dtype=int),
    }
    raw = onehot_raw(x, rows, state)
    level_sources = np.repeat(present, [level["width"] for level in levels])
    selected = distinct_columns(raw)
    if options.get("mi_top"):
        information = binary_mutual_information(raw[:, selected], positive)
        selected = np.sort(selected[np.argsort(-information)[: int(cfg["top_onehot"])]])
    state["selected"] = selected
    sources = level_sources[selected]
    if options.get("interactions"):
        information = source_mutual_information(x, rows, present, positive)
        top = present[np.argsort(-information)[: int(cfg["interaction_sources"])]]
        top = top[top != AGE_INDEX]
        state["interactions"] = np.flatnonzero(np.isin(sources, top))
    matrix = onehot_matrix(x, rows, state, raw[:, selected])
    del raw
    state["sources"] = np.concatenate([sources, sources[state["interactions"]]])
    state["means"], state["scales"] = fit_scale_in_place(matrix, data["eps"])
    return state, matrix


def apply_onehot(data, rows, state):
    matrix = onehot_matrix(data["x"], rows, state)
    return scale_in_place(matrix, state["means"], state["scales"])


# Pipeline: the current codebook recode and clinical terms


def fit_codebook(data, rows, _positive, _cfg, _options):
    features_cfg = data["features"]
    design = data["codebook"]
    eps = data["eps"]
    state_prep = prep.fit_preprocessor(
        design["x"][rows], eps, float(features_cfg["max_missing_rate"]), features_cfg["impute"]
    )
    raw_names, kept_sources = codebook.columns_after_drop(
        design["names"], design["source"], state_prep["drop"]
    )
    cleaned, expanded_names = restore_required(prep.transform(design["x"][rows], state_prep), raw_names)
    clinical = features.fit_clinical_terms(cleaned, expanded_names, eps)
    matrix = features.transform_clinical_terms(cleaned, clinical).astype(np.float32)
    # Restored zero columns and the squares and products have no single source.
    extra = matrix.shape[1] - kept_sources.size
    state = {
        "prep": state_prep,
        "raw_names": raw_names,
        "clinical": clinical,
        "sources": np.concatenate([kept_sources, np.full(extra, -1, dtype=int)]),
    }
    return state, matrix


def apply_codebook(data, rows, state):
    cleaned, _names = restore_required(
        prep.transform(data["codebook"]["x"][rows], state["prep"]), state["raw_names"]
    )
    return features.transform_clinical_terms(cleaned, state["clinical"]).astype(np.float32)


# Dispatch


def fit_pipeline(name, data, rows, positive, cfg):
    kind, options = EXPERIMENTS[name]
    if kind == "union":
        state_cb, matrix_cb = fit_codebook(data, rows, positive, cfg, {})
        state_oh, matrix_oh = fit_onehot(data, rows, positive, cfg, {})
        state = {
            "kind": kind,
            "parts": (state_cb, state_oh),
            "sources": np.concatenate([state_cb["sources"], state_oh["sources"]]),
        }
        return state, np.hstack([matrix_cb, matrix_oh])
    fit = {"codebook": fit_codebook, "numeric": fit_numeric, "onehot": fit_onehot}[kind]
    state, matrix = fit(data, rows, positive, cfg, options)
    state["kind"] = kind
    return state, matrix


def apply_pipeline(data, rows, state):
    kind = state["kind"]
    if kind == "union":
        state_cb, state_oh = state["parts"]
        return np.hstack([apply_codebook(data, rows, state_cb), apply_onehot(data, rows, state_oh)])
    apply = {"codebook": apply_codebook, "numeric": apply_numeric, "onehot": apply_onehot}[kind]
    return apply(data, rows, state)


# Model


def logistic_scores(x, theta):
    return x @ theta[1:].astype(np.float32) + np.float32(theta[0])


def largest_curvature(x, weights, rng):
    """Power iteration for the top eigenvalue of [1 X]^T diag(weights) [1 X] / n."""
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
    scores = np.clip(logistic_scores(x, theta), -SCORE_CLIP, SCORE_CLIP)
    residual = (weights * (1.0 / (1.0 + np.exp(-scores)) - y01)).astype(np.float32)
    gradient = np.empty(theta.size)
    gradient[0] = residual.sum() / x.shape[0]
    gradient[1:] = (x.T @ residual) / x.shape[0] + 2.0 * l2 * theta[1:]
    return gradient


def fit_logistic(x, y01, weights, l2, l1, max_iters, rng):
    """FISTA on the weighted logistic loss with L2, and L1 by soft thresholding.

    The bias is not penalized. It starts at the weighted log odds.
    """
    weights = weights.astype(np.float32)
    y01 = y01.astype(np.float32)
    step = 1.0 / (0.25 * LIPSCHITZ_MARGIN * largest_curvature(x, weights, rng) + 2.0 * l2)
    theta = np.zeros(x.shape[1] + 1)
    weighted_rate = float(np.sum(weights * y01) / np.sum(weights))
    theta[0] = np.log(weighted_rate / (1.0 - weighted_rate))
    look = theta.copy()
    momentum = 1.0
    for _ in range(max_iters):
        moved = look - step * logistic_gradient(x, y01, weights, look, l2)
        moved[1:] = np.sign(moved[1:]) * np.maximum(np.abs(moved[1:]) - step * l1, 0.0)
        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum ** 2)) / 2.0
        look = moved + ((momentum - 1.0) / next_momentum) * (moved - theta)
        theta, momentum = moved, next_momentum
    return theta


# Evaluation


def run_fold(name, data, train_rows, test_rows, y, cfg, rng):
    """Fit on one outer training fold and score its outer test fold."""
    started = time.time()
    positive = y == 1
    fit_local, tune_local = prep.stratified_split(
        positive[train_rows], float(cfg["tune_fraction"]), rng
    )
    fit_rows, tune_rows = train_rows[fit_local], train_rows[tune_local]
    state, x_fit = fit_pipeline(name, data, fit_rows, positive[fit_rows], cfg)
    l1 = float(cfg["l1_penalty"]) if EXPERIMENTS[name][1].get("l1") else 0.0
    theta = fit_logistic(
        x_fit,
        positive[fit_rows],
        positive_class_weights(positive[fit_rows], float(cfg["positive_class_cost"])),
        float(cfg["l2_penalty"]),
        l1,
        int(cfg["max_iters"]),
        rng,
    )
    del x_fit
    tune_scores = logistic_scores(apply_pipeline(data, tune_rows, state), theta)
    threshold, _tune_f1 = choose_threshold(tune_scores, y[tune_rows])
    x_test = apply_pipeline(data, test_rows, state)
    test_scores = logistic_scores(x_test, theta)
    metrics = classification_scores(test_scores, y[test_rows], threshold)
    metrics.update({
        "pr_auc": average_precision(test_scores, positive[test_rows]),
        "roc_auc": auc(test_scores, positive[test_rows]),
        "threshold": threshold,
        "n_features": x_test.shape[1],
        "nonzero": int(np.count_nonzero(theta[1:])),
        "seconds": time.time() - started,
    })
    return metrics, (state, theta, x_test, test_scores)


def permutation_importance(name, fitted, positive, rng):
    """Drop in PR AUC and ROC AUC when one raw column is shuffled.

    All encoded columns from the same raw column move together, so a
    column split into levels is judged as one feature. Shuffling the
    encoded block equals encoding a shuffled raw column, because each row
    is encoded on its own.
    """
    state, theta, x_test, scores = fitted
    weights = theta[1:].astype(np.float32)
    base_pr = average_precision(scores, positive)
    base_roc = auc(scores, positive)
    order = rng.permutation(x_test.shape[0])
    sources = state["sources"]
    rows = []
    for source in np.unique(sources[sources >= 0]):
        columns = np.flatnonzero(sources == source)
        block = x_test[:, columns]
        shuffled = scores + (block[order] - block) @ weights[columns]
        rows.append({
            "experiment": name,
            "column": codebook.HEADER[source],
            "n_columns": columns.size,
            "pr_auc_drop": base_pr - average_precision(shuffled, positive),
            "roc_auc_drop": base_roc - auc(shuffled, positive),
        })
    rows.sort(key=lambda row: -row["pr_auc_drop"])
    return rows


def summary_rows(results):
    out = []
    for name in dict.fromkeys(row["experiment"] for row in results):
        folds = [row for row in results if row["experiment"] == name]
        for label, reduce in (("mean", np.mean), ("std", np.std)):
            summary = {"experiment": name, "fold": label}
            summary.update({key: float(reduce([row[key] for row in folds])) for key in METRICS})
            out.append(summary)
    return out


def write_rows(rows, path):
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


def print_summary(results):
    means = [row for row in summary_rows(results) if row["fold"] == "mean"]
    means.sort(key=lambda row: -row["pr_auc"])
    print("%-28s %7s %7s %7s %7s %7s %6s" % (
        "experiment", "pr_auc", "roc_auc", "f1", "prec", "recall", "cols"))
    for row in means:
        print("%-28s %7.4f %7.4f %7.4f %7.4f %7.4f %6d" % (
            row["experiment"], row["pr_auc"], row["roc_auc"], row["f1"],
            row["precision"], row["recall"], row["n_features"]))


def main(names):
    cfg = read_json(CONFIG_PATH)
    features_cfg = read_json(FEATURES_PATH)
    names = names or list(cfg["experiments"])
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        raise SystemExit("unknown experiments: " + ", ".join(unknown))
    x, y = load_arrays()
    positive = y == 1
    data = {
        "x": x,
        "columns": generic_columns(bool(cfg["use_leak_columns"])),
        "eps": float(features_cfg["eps"]),
        "features": features_cfg,
    }
    if any(EXPERIMENTS[name][0] in ("codebook", "union") for name in names):
        data["codebook"] = codebook_design(x, features_cfg)
    n_folds = int(cfg["n_folds"])
    fold = stratified_folds(positive, n_folds, np.random.default_rng(int(cfg["seed"])))
    print("%d rows, %d raw columns, %.2f%% positive, %d folds" % (
        x.shape[0], x.shape[1], 100.0 * positive.mean(), n_folds), flush=True)
    results, importance = [], []
    for name in names:
        for index in range(n_folds):
            # Same tune split and start for every experiment on a given fold.
            rng = np.random.default_rng(int(cfg["seed"]) + index)
            train_rows, test_rows = np.flatnonzero(fold != index), np.flatnonzero(fold == index)
            metrics, fitted = run_fold(name, data, train_rows, test_rows, y, cfg, rng)
            results.append({"experiment": name, "fold": index, **metrics})
            print("%s fold %d  pr_auc %.4f  f1 %.4f  cols %d  %.0fs" % (
                name, index, metrics["pr_auc"], metrics["f1"],
                metrics["n_features"], metrics["seconds"]), flush=True)
            if index == 0 and name in cfg["importance_experiments"]:
                importance.extend(permutation_importance(name, fitted, positive[test_rows], rng))
                write_rows(importance, IMPORTANCE_PATH)
            del fitted
        write_rows(results + summary_rows(results), RESULTS_PATH)
    print_summary(results)


if __name__ == "__main__":
    main(sys.argv[1:])
