"""Score every raw BRFSS column against the label without trusting its codes.

Each column is treated as categories: its distinct values when there are
few, quantile bins otherwise, and missing as one more category. The positive
rate of each category is learned on one half of the rows and scored on the
other half, so the AUC is out of fold. Codes such as 7, 9, 88, or 7777 then
count only through the positive rate they carry, never as magnitudes.

`auc_within_age_sex` compares people only inside one age band and sex, so it
shows what a column adds beyond the two strongest demographic facts.

The missingness AUC uses only whether the cell is blank. A high value means
the skip pattern of the survey itself tells the label apart.
"""

import csv
import os

import numpy as np

from helpers import load_csv_data
from src.codebook_features import HEADER

MAX_CATEGORIES = 60
QUANTILE_BINS = 20
# Pseudo count that pulls rare categories toward the overall positive rate.
PRIOR_STRENGTH = 20.0


def auc(score, positive):
    """Rank AUC with average ranks for ties."""
    _values, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    average_rank = np.cumsum(counts) - (counts - 1) / 2.0
    ranks = average_rank[inverse]
    n_pos = positive.sum()
    n_neg = positive.size - n_pos
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def categories(column):
    """Integer category per row. Missing is its own category."""
    finite = np.isfinite(column)
    distinct = np.unique(column[finite])
    codes = np.full(column.shape[0], -1, dtype=int)
    if distinct.size <= MAX_CATEGORIES:
        codes[finite] = np.searchsorted(distinct, column[finite])
    else:
        edges = np.unique(np.quantile(column[finite], np.linspace(0, 1, QUANTILE_BINS + 1)))
        codes[finite] = np.searchsorted(edges[1:-1], column[finite], side="right")
    return codes + 1


def stratified_auc(score, positive, strata):
    """AUC inside each stratum, averaged with weight n_pos * n_neg.

    Pairs are only compared within one stratum, so a column that only
    restates age or sex scores near 0.5.
    """
    total = 0.0
    weight_sum = 0.0
    for stratum in np.unique(strata):
        rows = strata == stratum
        n_pos = positive[rows].sum()
        weight = float(n_pos) * float(rows.sum() - n_pos)
        if weight == 0:
            continue
        total += weight * auc(score[rows], positive[rows])
        weight_sum += weight
    return total / weight_sum


def out_of_fold_auc(codes, positive, half, strata):
    """Category positive rate fit on one half, scored on the other.

    Output: plain AUC and AUC within `strata`, each averaged over both halves.
    """
    prior = positive.mean()
    plain = []
    within = []
    for fit_rows in (half, ~half):
        size = codes.max() + 1
        hits = np.bincount(codes[fit_rows], weights=positive[fit_rows], minlength=size)
        totals = np.bincount(codes[fit_rows], minlength=size)
        rate = (hits + PRIOR_STRENGTH * prior) / (totals + PRIOR_STRENGTH)
        score_rows = ~fit_rows
        score = rate[codes[score_rows]]
        plain.append(auc(score, positive[score_rows]))
        within.append(stratified_auc(score, positive[score_rows], strata[score_rows]))
    return float(np.mean(plain)), float(np.mean(within))


def audit(x, y, seed=0):
    positive = y == 1
    half = np.random.default_rng(seed).random(x.shape[0]) < 0.5
    prior = positive.mean()
    # Fourteen age bands times two sexes.
    strata = categories(x[:, HEADER.index("_AGEG5YR")]) * 10 + categories(x[:, HEADER.index("SEX")])
    rows = []
    for index, name in enumerate(HEADER):
        column = x[:, index]
        missing = ~np.isfinite(column)
        missing_auc = auc(missing.astype(float), positive) if 0 < missing.mean() < 1 else 0.5
        plain_auc, within_auc = out_of_fold_auc(categories(column), positive, half, strata)
        rows.append({
            "column": name,
            "index": index,
            "missing_rate": missing.mean(),
            "auc": plain_auc,
            "auc_within_age_sex": within_auc,
            "missing_auc": max(missing_auc, 1.0 - missing_auc),
            "pos_rate_missing": positive[missing].mean() if missing.any() else np.nan,
            "pos_rate_present": positive[~missing].mean() if (~missing).any() else np.nan,
            "prior": prior,
        })
    rows.sort(key=lambda row: -row["auc_within_age_sex"])
    return rows


def write_audit(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: ("%.4f" % value if isinstance(value, float) else value)
                for key, value in row.items()
            })


if __name__ == "__main__":
    x_train, _x_test, y_train, _train_ids, _test_ids = load_csv_data(
        os.path.join("data", "dataset")
    )
    write_audit(audit(x_train, y_train), "feature_audit.csv")
