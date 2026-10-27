"""One preprocessing path: drop survey/leak columns, one hot levels, MI top.

    data = make_data(x, cfg)
    state, x_fit = fit(data, fit_rows, y[fit_rows] == 1, cfg)
    x_out = apply(data, other_rows, state)
"""

import os

import numpy as np

DATA_DIR = os.path.join("data", "dataset")

SURVEY_COLUMNS = (
    "_STATE", "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO",
    "_PSU", "CTELENUM", "PVTRESD1", "COLGHOUS", "STATERES", "CELLFON3", "LADULT",
    "NUMADULT", "NUMMEN", "NUMWOMEN", "CTELNUM1", "CELLFON2", "CADULT", "PVTRESD2",
    "CCLGHOUS", "CSTATE", "LANDLINE", "HHADULT", "NUMHHOL2", "NUMPHON2", "CPDEMO1",
    "QSTVER", "QSTLANG", "MSCODE", "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE",
    "_CLLCPWT", "_LLCPWT", "_DUALUSE", "_DUALCOR",
)
LEAK_COLUMNS = ("HAREHAB1", "STREHAB1", "CVDASPRN", "ASPUNSAF")


def load_header(data_dir=DATA_DIR):
    """Column names from x_train.csv, without Id."""
    with open(os.path.join(data_dir, "x_train.csv"), encoding="utf-8") as handle:
        return tuple(handle.readline().strip().split(",")[1:])


HEADER = None
INDEX = None


def ensure_header(data_dir=DATA_DIR):
    """Load HEADER and INDEX once, when the CSV is available."""
    global HEADER, INDEX
    if HEADER is None:
        HEADER = load_header(data_dir)
        INDEX = {name: index for index, name in enumerate(HEADER)}
    return HEADER, INDEX


def usable_columns(x, rows, columns, eps):
    """Columns with at least one value and variance above eps on rows."""
    sub = x[np.ix_(rows, columns)]
    rate = np.isnan(sub).mean(axis=0)
    usable = rate < 1.0
    variance = np.zeros(columns.size)
    variance[usable] = np.nanvar(sub[:, usable], axis=0)
    usable &= variance > eps
    return columns[usable], rate[usable]


def distinct_columns(raw):
    """Positions of non constant, non duplicate 0/1 columns."""
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


def binary_mutual_information(raw, positive):
    """Mutual information of each 0/1 column with the label."""
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


def stratified_split(y, val_frac, rng):
    """Fit and val indices that keep each class ratio."""
    fit_parts, val_parts = [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_val = int(val_frac * len(idx))
        val_parts.append(idx[:n_val])
        fit_parts.append(idx[n_val:])
    fit_idx = np.concatenate(fit_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(fit_idx)
    rng.shuffle(val_idx)
    return fit_idx, val_idx


def stratified_folds(positive, n_folds, rng):
    """Fold id per row with the same positive rate in each fold."""
    fold = np.empty(positive.size, dtype=int)
    for label in (False, True):
        rows = np.flatnonzero(positive == label)
        rng.shuffle(rows)
        fold[rows] = np.arange(rows.size) % n_folds
    return fold


def fit_levels(column, min_count, bins):
    """Frequent values, quantile bins for the rest, and optional missing."""
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
    """Write one hot level columns for one raw column into out."""
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


def _onehot_raw(x, rows, state):
    """All levels of every kept column as uint8."""
    widths = [level["width"] for level in state["levels"]]
    raw = np.zeros((rows.size, sum(widths)), dtype=np.uint8)
    start = 0
    for column, level in zip(state["columns"], state["levels"]):
        encode_levels(x[rows, column], level, raw[:, start : start + level["width"]])
        start += level["width"]
    return raw


def make_data(x, cfg):
    """Raw matrix plus the columns every pipeline may use."""
    header, _index = ensure_header()
    excluded = set(SURVEY_COLUMNS)
    if not cfg.get("use_leak_columns"):
        excluded |= set(LEAK_COLUMNS)
    columns = np.array(
        [index for index, name in enumerate(header) if name not in excluded], dtype=int
    )
    return {"x": x, "columns": columns, "eps": float(cfg.get("eps", 1e-12))}


def fit(data, rows, positive, cfg):
    """Fit one hot levels and keep the top_onehot by mutual information."""
    x = data["x"]
    present, _rate = usable_columns(x, rows, data["columns"], data["eps"])
    min_count = max(1, int(round(float(cfg["min_level_share"]) * rows.size)))
    levels = [fit_levels(x[rows, column], min_count, int(cfg["quantile_bins"])) for column in present]
    state = {"columns": present, "levels": levels}
    raw = _onehot_raw(x, rows, state)
    level_sources = np.repeat(present, [level["width"] for level in levels])
    candidates = distinct_columns(raw)
    information = binary_mutual_information(raw[:, candidates], positive)
    order = np.argsort(-information)[: int(cfg["top_onehot"])]
    picked = candidates[order]
    picked_mi = information[order]
    sort = np.argsort(picked)
    state["selected"] = picked[sort]
    state["mi"] = picked_mi[sort]
    state["sources"] = level_sources[state["selected"]]
    matrix = raw[:, state["selected"]].astype(np.float32)
    del raw
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales < data["eps"], 1.0, scales)
    matrix -= means
    matrix /= scales
    state["means"], state["scales"] = means, scales
    return state, matrix


def apply(data, rows, state):
    """Apply a fitted state. Does not refit."""
    matrix = _onehot_raw(data["x"], rows, state)[:, state["selected"]].astype(np.float32)
    matrix -= state["means"].astype(np.float32)
    matrix /= state["scales"].astype(np.float32)
    return matrix
