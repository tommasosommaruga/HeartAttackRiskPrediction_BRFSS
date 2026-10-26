"""Train the models named in the config and keep the best class-1 score.

From the project folder:

    python run.py

config/train.json lists the models, the class weight, the grids, early
stopping, and an optional age or sex split. config/features.json sets
imputation, the BMI cap, and the missing-column rules. The script fits every
listed model, writes one submission per model, and copies the best validation
F1 to submissions/best.csv. The cutoff for every model is the validation F1
of target 1.
"""

import os
import shutil
from pathlib import Path

import numpy as np

import src.codebook_features as codebook
import src.column_significance as column_significance
import src.early_stopping as early
import src.features as features
import src.positive_class_weights as weights
import src.preprocessing as prep
import src.train_config as train_config
from helpers import create_csv_submission, load_csv_data
from implementations import calculate_loss, compute_loss


ROOT = Path(__file__).resolve().parent
REQUIRED_COLUMNS = (
    "_AGE80",
    "_BMI5",
    "SEX",
    "_SMOKER3",
    "diab_yes",
    "GENHLTH",
    "_RFCHOL",
)


def add_bias(x):
    return np.c_[np.ones(x.shape[0]), x]


def classification_scores(scores, y_pm1, threshold):
    """Accuracy, F1, precision, and recall for labels in {-1, +1}."""
    pred = np.where(scores >= threshold, 1, -1)
    positive = y_pm1 == 1
    negative = ~positive
    tp = float(np.sum((pred == 1) & positive))
    fp = float(np.sum((pred == 1) & negative))
    fn = float(np.sum((pred == -1) & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "accuracy": float(np.mean(pred == y_pm1)),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def threshold_for_rate(scores, rate):
    """Score cutoff whose share of rows at or above it is about `rate`."""
    if rate <= 0.0:
        return float(np.max(scores) + 1.0)
    if rate >= 1.0:
        return float(np.min(scores))
    return float(np.quantile(scores, 1.0 - rate))


def choose_threshold(scores, y_pm1):
    """Cut that maximizes validation F1. A tie keeps the higher accuracy."""
    grid = np.linspace(0.0, 1.0, 201)
    candidates = np.concatenate([np.quantile(scores, grid), np.array([0.0])])
    best_threshold, best_f1, best_accuracy = 0.0, -1.0, -1.0
    for threshold in np.unique(candidates):
        metrics = classification_scores(scores, y_pm1, float(threshold))
        better_f1 = metrics["f1"] > best_f1
        tied = metrics["f1"] == best_f1 and metrics["accuracy"] > best_accuracy
        if better_f1 or tied:
            best_f1 = metrics["f1"]
            best_accuracy = metrics["accuracy"]
            best_threshold = float(threshold)
    return best_threshold, best_f1


def use_gpu(train):
    flag = os.environ.get("USE_GPU", "")
    if flag == "1":
        return True
    if flag == "0":
        return False
    return bool(train["use_gpu"])


def parameter_grid(kind, train):
    """Hyperparameter combinations searched on the validation fold."""
    grids = train["grids"]
    caps = train["early_stopping"]["max_iters"]
    if kind == "least_squares":
        return ({"lambda": 0.0, "gamma": 0.0, "max_iters": 0},)
    if kind == "ridge":
        return tuple(
            {"lambda": lam, "gamma": 0.0, "max_iters": 0}
            for lam in grids["ridge_lambda"]
        )
    if kind == "gd":
        return tuple(
            {"lambda": 0.0, "gamma": gamma, "max_iters": caps["gd"]}
            for gamma in grids["gd_gamma"]
        )
    if kind == "sgd":
        return tuple(
            {"lambda": 0.0, "gamma": gamma, "max_iters": caps["sgd"]}
            for gamma in grids["sgd_gamma"]
        )
    if kind == "logistic":
        return tuple(
            {"lambda": 0.0, "gamma": gamma, "max_iters": caps["logistic"]}
            for gamma in grids["logistic_gamma"]
        )
    if kind == "reg_logistic":
        return tuple(
            {
                "lambda": lam,
                "gamma": gamma,
                "max_iters": caps["reg_logistic"],
            }
            for lam in grids["reg_logistic_lambda"]
            for gamma in grids["logistic_gamma"]
        )
    raise ValueError("unknown model: " + kind)


def train_quiet(kind, y_pm1, y01, tx, lambda_, gamma, max_iters, train, tx_val=None, y_val=None):
    """Fit one model. Iterative models stop when validation AUC stalls."""
    sample_weights = weights.positive_class_weights(y_pm1, float(train["positive_class_cost"]))
    stop = train["early_stopping"]
    if kind in early.ITERATIVE:
        if kind == "sgd":
            np.random.seed(int(train["seed"]))
        return early.fit_iterative(
            kind,
            y_pm1,
            y01,
            tx,
            sample_weights,
            lambda_,
            gamma,
            max_iters,
            tx_val,
            y_val,
            patience=int(stop["patience"]),
            eval_every=int(stop["eval_every"][kind]),
            use_gpu=use_gpu(train),
            min_auc_gain=float(stop["min_auc_gain"]),
            metric=str(stop.get("metric", "auc")),
        )
    w, loss = weights.fit_positive_class_weights(
        kind, y_pm1, y01, tx, sample_weights, lambda_, gamma, max_iters,
    )
    return w, loss, 0


def finite_fit(w, loss):
    return bool(np.isfinite(loss)) and bool(np.isfinite(w).all())


def select_params(kind, y_fit, y01_fit, tx_fit, y_val, tx_val, train):
    """Pick the grid point with the best validation F1."""
    best = None
    n_skipped = 0
    grid = parameter_grid(kind, train)
    for params in grid:
        w, loss, stopped = train_quiet(
            kind,
            y_fit,
            y01_fit,
            tx_fit,
            params["lambda"],
            params["gamma"],
            params["max_iters"],
            train,
            tx_val,
            y_val,
        )
        if not finite_fit(w, loss):
            n_skipped += 1
            continue
        scores = tx_val @ w
        threshold, f1 = choose_threshold(scores, y_val)
        metrics = classification_scores(scores, y_val, threshold)
        better = best is None or f1 > best["f1"]
        tied = (
            best is not None
            and f1 == best["f1"]
            and metrics["accuracy"] > best["accuracy"]
        )
        if better or tied:
            best = {
                "lambda": params["lambda"],
                "gamma": params["gamma"],
                "max_iters": stopped,
                "threshold": threshold,
                "f1": f1,
                "accuracy": metrics["accuracy"],
                "w": w,
            }
    if best is None:
        raise RuntimeError("every setting diverged for " + kind)
    best["n_tried"] = len(grid)
    best["n_skipped"] = n_skipped
    return best


def restore_required(cleaned, names):
    """Put back a required column the group drop removed, as zeros."""
    missing = [name for name in REQUIRED_COLUMNS if name not in names]
    if not missing:
        return cleaned, list(names)
    zeros = np.zeros((cleaned.shape[0], len(missing)))
    return np.column_stack([cleaned, zeros]), list(names) + missing


def _crossed_groups(x_train_cb, x_test_cb, names, bands):
    """Sex crossed with age bands. Missing sex joins the women. Missing age
    joins the last band, which must be open on the right.
    """
    sex_train = x_train_cb[:, names.index("SEX")]
    sex_test = x_test_cb[:, names.index("SEX")]
    age_train = x_train_cb[:, names.index("_AGE80")]
    age_test = x_test_cb[:, names.index("_AGE80")]
    groups = []
    for sex_label, is_male in (("female", False), ("male", True)):
        if is_male:
            sex_train_mask = sex_train == 1
            sex_test_mask = sex_test == 1
        else:
            sex_train_mask = sex_train != 1
            sex_test_mask = sex_test != 1
        for band_label, low, high in bands:
            groups.append(
                (
                    sex_label + "_" + band_label,
                    sex_train_mask & _age_band(age_train, low, high),
                    sex_test_mask & _age_band(age_test, low, high),
                )
            )
    return tuple(groups)


def _sex_age_groups(x_train_cb, x_test_cb, names):
    """Six groups: sex crossed with under 55, 55–64, and 65 or older."""
    return _crossed_groups(
        x_train_cb,
        x_test_cb,
        names,
        (
            ("under_55", None, 55.0),
            ("age_55_64", 55.0, 65.0),
            ("age_65_plus", 65.0, None),
        ),
    )


def _sex_age65_groups(x_train_cb, x_test_cb, names, cutoff):
    """Four groups: sex crossed with under `cutoff` and `cutoff` or older."""
    return _crossed_groups(
        x_train_cb,
        x_test_cb,
        names,
        (
            ("under_65", None, float(cutoff)),
            ("age_65_plus", float(cutoff), None),
        ),
    )


def _age_band(age, low, high):
    """Rows with age in [low, high). A missing bound is open on that side."""
    finite = np.isfinite(age)
    if low is None:
        return finite & (age < high)
    if high is None:
        return (~finite) | (age >= low)
    return finite & (age >= low) & (age < high)


def split_groups(split_name, x_train_cb, x_test_cb, names, train):
    """Boolean train and test masks for each piece of one split."""
    if split_name == "sex":
        sex_train = x_train_cb[:, names.index("SEX")]
        sex_test = x_test_cb[:, names.index("SEX")]
        groups = (
            ("male", sex_train == 1, sex_test == 1),
            ("female", sex_train == 0, sex_test == 0),
        )
    elif split_name == "sex_age":
        groups = _sex_age_groups(x_train_cb, x_test_cb, names)
    elif split_name == "sex_age65":
        groups = _sex_age65_groups(
            x_train_cb, x_test_cb, names, float(train.get("age_cutoff", 65))
        )
    elif split_name == "age65":
        cutoff = float(train.get("age_cutoff", 65))
        age_train = x_train_cb[:, names.index("_AGE80")]
        age_test = x_test_cb[:, names.index("_AGE80")]
        groups = (
            (
                "under_65",
                np.isfinite(age_train) & (age_train < cutoff),
                np.isfinite(age_test) & (age_test < cutoff),
            ),
            (
                "age_65_plus",
                (~np.isfinite(age_train)) | (age_train >= cutoff),
                (~np.isfinite(age_test)) | (age_test >= cutoff),
            ),
        )
    else:
        raise ValueError("unknown split: " + split_name)
    train_cover = np.zeros(x_train_cb.shape[0], dtype=bool)
    test_cover = np.zeros(x_test_cb.shape[0], dtype=bool)
    for _label, train_mask, test_mask in groups:
        train_cover |= train_mask
        test_cover |= test_mask
    if not train_cover.all() or not test_cover.all():
        raise RuntimeError(split_name + " left some rows in no group")
    return groups


def search_state(x, drop, fit_rows, impute, eps):
    """Fill and scale from the fit rows, using a drop mask fixed on the group."""
    kept = x[fit_rows][:, ~drop]
    fill = prep.column_fill(kept, impute)
    means, scales = prep.standardization_stats(prep.impute_values(kept, fill), eps)
    return {
        "drop": drop,
        "impute": impute,
        "fill": fill,
        "medians": fill,
        "means": means,
        "scales": scales,
    }


def fit_terms(cleaned, names, eps, features_cfg):
    """Clinical terms, or those plus the degree-2 map of the risk columns."""
    if features_cfg.get("risk_polynomial"):
        return features.fit_risk_terms(cleaned, names, eps)
    return features.fit_clinical_terms(cleaned, names, eps)


def fit_clinical(x, rows, state, names, source, eps, features_cfg):
    """Fit the feature map on `rows` and return the matrix plus the map pieces."""
    raw_names, _source = codebook.columns_after_drop(names, source, state["drop"])
    cleaned, expanded_names = restore_required(prep.transform(x[rows], state), raw_names)
    clinical = fit_terms(cleaned, expanded_names, eps, features_cfg)
    tx = add_bias(features.transform_clinical_terms(cleaned, clinical))
    return tx, raw_names, clinical, expanded_names


def transform_with(x, rows, state, raw_names, names, source, clinical, expanded_names):
    """Transform `rows` with a state and clinical map fit elsewhere."""
    got, _source = codebook.columns_after_drop(names, source, state["drop"])
    if got != raw_names:
        raise RuntimeError("column set changed between the search and the refit")
    cleaned, got_names = restore_required(prep.transform(x[rows], state), raw_names)
    if got_names != expanded_names:
        raise RuntimeError("restored columns changed between the search and the refit")
    return add_bias(features.transform_clinical_terms(cleaned, clinical))


def train_pooled(kind, bundle, features_cfg, train):
    """One model on every labelled row. Returns the validation metrics and path."""
    chosen = select_params(
        kind,
        bundle["y_fit"],
        bundle["y01_fit"],
        bundle["tx_fit"],
        bundle["y_val"],
        bundle["tx_val"],
        train,
    )
    threshold = chosen["threshold"]
    val_metrics = classification_scores(bundle["tx_val"] @ chosen["w"], bundle["y_val"], threshold)
    w_full, full_loss, _stopped = train_quiet(
        kind,
        bundle["y_pm1"],
        bundle["y01"],
        bundle["tx_train"],
        chosen["lambda"],
        chosen["gamma"],
        chosen["max_iters"],
        train,
    )
    if not finite_fit(w_full, full_loss):
        raise RuntimeError("full refit diverged for " + kind)
    y_pred = np.where(bundle["tx_test"] @ w_full >= threshold, 1, -1)
    path = Path(train["submission_dir"]) / f"{kind}_mode_expanded.csv"
    create_csv_submission(bundle["ids_test"], y_pred, str(path))
    print_result(kind, "all", val_metrics, chosen, y_pred, path)
    return result_row(kind, "all", val_metrics, chosen, y_pred, path)


def train_split(kind, split_name, bundle, features_cfg, train):
    """One model inside each group of `split_name`, with its own F1 cutoff."""
    eps = float(features_cfg["eps"])
    impute = features_cfg["impute"]
    max_missing = float(features_cfg["max_missing_rate"])
    x_train_cb = bundle["x_train_cb"]
    x_test_cb = bundle["x_test_cb"]
    names = bundle["names"]
    source = bundle["source"]
    y_pm1 = bundle["y_pm1"]
    y01 = bundle["y01"]
    val_pred = np.zeros(y_pm1.shape[0])
    test_pred = np.zeros(x_test_cb.shape[0], dtype=int)
    groups = split_groups(split_name, x_train_cb, x_test_cb, names, train)
    reported_design = False
    for label, train_mask, test_mask in groups:
        train_rows = np.flatnonzero(train_mask)
        test_rows = np.flatnonzero(test_mask)
        state_columns = prep.fit_preprocessor(x_train_cb[train_rows], eps, max_missing, impute)
        if train.get("use_validation", True):
            group_fit = np.intersect1d(bundle["fit_idx"], train_rows)
            group_val = np.intersect1d(bundle["val_idx"], train_rows)
            if not np.any(y_pm1[group_fit] == 1) or not np.any(y_pm1[group_val] == 1):
                raise RuntimeError(label + " is missing target 1 in fit or validation")
            drop = state_columns["drop"]
            state_group = search_state(x_train_cb, drop, group_fit, impute, eps)
            tx_fit, raw_names, clinical, expanded_names = fit_clinical(
                x_train_cb, group_fit, state_group, names, source, eps, features_cfg
            )
            tx_val = transform_with(
                x_train_cb, group_val, state_group, raw_names, names, source, clinical, expanded_names
            )
            chosen = select_params(
                kind, y_pm1[group_fit], y01[group_fit], tx_fit, y_pm1[group_val], tx_val, train
            )
            score_rows = group_val
            score_tx = tx_val
            score_y = y_pm1[group_val]
        else:
            fixed = train.get("fixed_group_params", {}).get(label)
            if fixed is None:
                raise RuntimeError("missing fixed_group_params for " + label)
            chosen = {
                "lambda": float(fixed["lambda"]),
                "gamma": float(fixed["gamma"]),
                "max_iters": int(fixed["max_iters"]),
            }
            raw_names = None
            score_rows = train_rows
            score_tx = None
            score_y = y_pm1[train_rows]
        tx_all, raw_all, clinical_all, expanded_all = fit_clinical(
            x_train_cb, train_rows, state_columns, names, source, eps, features_cfg
        )
        if raw_names is not None and raw_all != raw_names:
            raise RuntimeError(label + " kept different columns on the full group")
        if not reported_design:
            print(
                f"{split_name} design {tx_all.shape[1] - 1} columns "
                f"({clinical_all['n_squares']} squares, {clinical_all['n_products']} products)",
                flush=True,
            )
            reported_design = True
        tx_test = transform_with(
            x_test_cb, test_rows, state_columns, raw_all, names, source, clinical_all, expanded_all
        )
        w_full, full_loss, _stopped = train_quiet(
            kind,
            y_pm1[train_rows],
            y01[train_rows],
            tx_all,
            chosen["lambda"],
            chosen["gamma"],
            chosen["max_iters"],
            train,
        )
        if not finite_fit(w_full, full_loss):
            raise RuntimeError("full refit diverged for " + kind + " " + label)
        if score_tx is None:
            threshold, _f1 = choose_threshold(tx_all @ w_full, score_y)
            chosen["threshold"] = threshold
            group_scores = tx_all @ w_full
        else:
            threshold = chosen["threshold"]
            group_scores = score_tx @ chosen["w"]
        group_pred = np.where(group_scores >= threshold, 1.0, -1.0)
        val_pred[score_rows] = group_pred
        full_scores = tx_all @ w_full
        if train.get("use_validation", True):
            held_out_rate = float(np.mean(group_scores >= threshold))
            submit_threshold = threshold_for_rate(full_scores, held_out_rate)
        else:
            held_out_rate = float(np.mean(group_pred == 1))
            submit_threshold = threshold
        test_pred[test_rows] = np.where(tx_test @ w_full >= submit_threshold, 1, -1)
        group_metrics = classification_scores(group_pred, score_y, 0.0)
        print(
            f"{kind:16} {split_name:10} {label:18} "
            f"f1 {group_metrics['f1']:.3f} rec {group_metrics['recall']:.3f} "
            f"prec {group_metrics['precision']:.3f} acc {group_metrics['accuracy']:.3f} "
            f"gamma={chosen['gamma']:.4g} iters={chosen['max_iters']} "
            f"rate={held_out_rate:.3f}",
            flush=True,
        )
    if np.any(test_pred == 0):
        raise RuntimeError(split_name + " left a test row without a prediction")
    if train.get("use_validation", True):
        scored = bundle["val_idx"]
    else:
        scored = np.arange(y_pm1.shape[0])
    if np.any(val_pred[scored] == 0):
        raise RuntimeError(split_name + " left a scored row without a prediction")
    overall = classification_scores(val_pred[scored], y_pm1[scored], 0.0)
    tag = split_name + ("_poly" if features_cfg.get("risk_polynomial") else "")
    suffix = "" if train.get("use_validation", True) else "_all"
    path = Path(train["submission_dir"]) / f"{kind}_{tag}{suffix}.csv"
    create_csv_submission(bundle["ids_test"], test_pred, str(path))
    print_result(kind, split_name, overall, {"lambda": 0, "gamma": 0, "max_iters": 0, "n_tried": 0, "n_skipped": 0, "threshold": 0}, test_pred, path)
    return result_row(kind, split_name, overall, {"lambda": 0.0, "gamma": 0.0, "max_iters": 0, "n_tried": 0, "n_skipped": 0, "threshold": 0.0}, test_pred, path)


def result_row(kind, split_name, metrics, chosen, y_pred, path):
    return {
        "model": kind,
        "split": split_name,
        "val_f1": metrics["f1"],
        "val_recall": metrics["recall"],
        "val_precision": metrics["precision"],
        "val_accuracy": metrics["accuracy"],
        "test_pos": float(np.mean(y_pred == 1)),
        "lambda": chosen["lambda"],
        "gamma": chosen["gamma"],
        "max_iters": chosen["max_iters"],
        "threshold": chosen["threshold"],
        "path": str(path),
    }


def print_result(kind, split_name, metrics, chosen, y_pred, path):
    print(
        f"{kind:16} {split_name:8} overall  "
        f"f1 {metrics['f1']:.3f} rec {metrics['recall']:.3f} "
        f"prec {metrics['precision']:.3f} acc {metrics['accuracy']:.3f} "
        f"test_pos={float(np.mean(y_pred == 1)):.3f} {path}",
        flush=True,
    )


def prepare(features_cfg, train):
    """Load the files, recode them, and build the pooled design matrices."""
    codebook.BMI_CAP = float(features_cfg["bmi_cap"])
    eps = float(features_cfg["eps"])
    impute = features_cfg["impute"]
    max_missing = float(features_cfg["max_missing_rate"])
    x_train, x_test, y_train, _ids_train, ids_test = load_csv_data(str(ROOT / "data" / "dataset"))
    y01 = (y_train == 1).astype(float)
    y_pm1 = y_train.astype(float)
    if train.get("use_validation", True):
        rng = np.random.default_rng(int(train["seed"]))
        fit_idx, val_idx = prep.stratified_split(y01, float(train["val_fraction"]), rng)
    else:
        fit_idx = np.arange(y01.shape[0])
        val_idx = np.arange(0, dtype=int)
    base_train, base_names, _base_source = codebook.transform_codebook(x_train, flag_names=())
    always_flags = [name for name, _flag in codebook.MISSING_FLAGS]
    flag_names = codebook.choose_flag_names(
        base_train, base_names, float(features_cfg["missing_flag_min_rate"]), always_flags
    )
    x_train_cb, names, source = codebook.transform_codebook(x_train, flag_names=flag_names)
    x_test_cb, names_test, _source_test = codebook.transform_codebook(x_test, flag_names=flag_names)
    if names != names_test:
        raise RuntimeError("train and test codebook columns differ")
    print(
        f"codebook {len(base_names)} columns, {len(flag_names)} missing indicators",
        flush=True,
    )
    bundle = {
        "x_train_cb": x_train_cb,
        "x_test_cb": x_test_cb,
        "names": names,
        "source": source,
        "y_pm1": y_pm1,
        "y01": y01,
        "fit_idx": fit_idx,
        "val_idx": val_idx,
        "ids_test": ids_test,
    }
    if not train.get("train_pooled", True):
        return bundle
    state_fit = prep.fit_preprocessor(x_train_cb[fit_idx], eps, max_missing, impute)
    x_fit = prep.transform(x_train_cb[fit_idx], state_fit)
    x_val = prep.transform(x_train_cb[val_idx], state_fit)
    names_fit, source_fit = codebook.columns_after_drop(names, source, state_fit["drop"])
    partial_r2, _delta = column_significance.leave_one_column_relation(x_fit, y_pm1[fit_idx])
    column_significance.write_column_significance(
        str(ROOT / "column_significance.csv"),
        source_fit,
        features.association_with_target(x_fit, y_pm1[fit_idx]),
        partial_r2,
        np.ones(len(names_fit), dtype=bool),
        feature_names=names_fit,
    )
    map_fit = fit_terms(x_fit, names_fit, eps, features_cfg)
    state_full = prep.fit_preprocessor(x_train_cb, eps, max_missing, impute)
    names_full, _source_full = codebook.columns_after_drop(names, source, state_full["drop"])
    if names_full != names_fit:
        raise RuntimeError("fit fold and full sample kept different codebook columns")
    x_all = prep.transform(x_train_cb, state_full)
    map_full = fit_terms(x_all, names_full, eps, features_cfg)
    bundle.update(
        {
            "y_fit": y_pm1[fit_idx],
            "y_val": y_pm1[val_idx],
            "y01_fit": y01[fit_idx],
            "tx_fit": add_bias(features.transform_clinical_terms(x_fit, map_fit)),
            "tx_val": add_bias(features.transform_clinical_terms(x_val, map_fit)),
            "tx_train": add_bias(features.transform_clinical_terms(x_all, map_full)),
            "tx_test": add_bias(
                features.transform_clinical_terms(prep.transform(x_test_cb, state_full), map_full)
            ),
        }
    )
    print(f"pooled design {bundle['tx_fit'].shape[1] - 1} columns", flush=True)
    return bundle


def print_table(rows, score_name):
    print()
    print(f"{'model':16} {'split':8} {score_name:7} {'recall':7} {'precision':9} {'accuracy':8} file")
    ordered = sorted(rows, key=lambda row: (-row["val_f1"], -row["val_accuracy"]))
    for row in ordered:
        print(
            f"{row['model']:16} {row['split']:8} {row['val_f1']:.3f}   "
            f"{row['val_recall']:.3f}   {row['val_precision']:.3f}     "
            f"{row['val_accuracy']:.3f}    {row['path']}"
        )
    return ordered[0]


def main():
    os.chdir(ROOT)
    np.seterr(divide="ignore", over="ignore", invalid="ignore")
    features_cfg, train = train_config.load()
    submission_dir = ROOT / train["submission_dir"]
    submission_dir.mkdir(parents=True, exist_ok=True)
    models = tuple(train["models"])
    splits = tuple(train.get("group_splits", []))
    print(
        f"models={','.join(models)} splits={','.join(splits) or 'none'} "
        f"pooled={bool(train.get('train_pooled', True))} "
        f"class_cost={float(train['positive_class_cost']):g}",
        flush=True,
    )
    bundle = prepare(features_cfg, train)
    rows = []
    if train.get("train_pooled", True):
        for kind in models:
            rows.append(train_pooled(kind, bundle, features_cfg, train))
    for split_name in splits:
        for kind in models:
            rows.append(train_split(kind, split_name, bundle, features_cfg, train))
    if not rows:
        raise RuntimeError("config/train.json selects no model and no split")
    score_name = "val F1" if train.get("use_validation", True) else "train F1"
    best = print_table(rows, score_name)
    best_path = submission_dir / "best.csv"
    shutil.copyfile(best["path"], best_path)
    print()
    print(
        f"best {best['model']} {best['split']} "
        f"{score_name} {best['val_f1']:.3f} recall {best['val_recall']:.3f} "
        f"precision {best['val_precision']:.3f}"
    )
    print(f"submit {best_path}")


if __name__ == "__main__":
    main()
