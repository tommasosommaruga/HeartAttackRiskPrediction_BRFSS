"""Clean the BRFSS answer codes before the usual preprocessing.

BRFSS stores non-answers as numbers: 7 / 77 / 777 / 7777 = "don't know",
9 / 99 / 999 / 9999 = "refused", and 88 / 888 (555 for food questions) =
"none". Left as they are, "no bad days" (88) looks worse than 30 bad days.
This module turns "don't know" / "refused" into NaN, "none" into 0, converts
WEIGHT2 / HEIGHT3 to one unit, drops administrative columns and one-hot
encodes nominal columns. Rules are fitted on the fit split only.
"""

import numpy as np

# Interview date and record identifiers: no information about the person
ADMIN_COLUMNS = ["FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "SEQNO", "_PSU"]

# Categories without an order: numeric codes become one-hot columns
NOMINAL_COLUMNS = ["_STATE", "MARITAL", "EMPLOY1", "_RACE"]

# Frequencies written as "unit digit + amount" (e.g. 105 = 5 per day,
# 205 = 5 per week): times per month of one unit, for each first digit
WEEKS_PER_MONTH = 30.0 / 7.0
FREQUENCY_COLUMNS = {
    # 1xx per week, 2xx per month
    "ALCDAY5": {1: WEEKS_PER_MONTH, 2: 1.0},
    "EXEROFT1": {1: WEEKS_PER_MONTH, 2: 1.0},
    "EXEROFT2": {1: WEEKS_PER_MONTH, 2: 1.0},
    "STRENGTH": {1: WEEKS_PER_MONTH, 2: 1.0},
    # 1xx per day, 2xx per week, 3xx per month (300 = less than once a month)
    "FRUITJU1": {1: 30.0, 2: WEEKS_PER_MONTH, 3: 1.0},
    "FRUIT1": {1: 30.0, 2: WEEKS_PER_MONTH, 3: 1.0},
    "FVBEANS": {1: 30.0, 2: WEEKS_PER_MONTH, 3: 1.0},
    "FVGREEN": {1: 30.0, 2: WEEKS_PER_MONTH, 3: 1.0},
    "FVORANG": {1: 30.0, 2: WEEKS_PER_MONTH, 3: 1.0},
    "VEGETAB1": {1: 30.0, 2: WEEKS_PER_MONTH, 3: 1.0},
    # same, plus 4xx per year
    "BLDSUGAR": {1: 30.0, 2: WEEKS_PER_MONTH, 3: 1.0, 4: 1.0 / 12.0},
    "FEETCHK2": {1: 30.0, 2: WEEKS_PER_MONTH, 3: 1.0, 4: 1.0 / 12.0},
}
# Durations written as hours and minutes, H:MM (130 = 1 h 30 min)
DURATION_COLUMNS = ["EXERHMM1", "EXERHMM2"]

# Version of the cleaning rules; 2 added the rules for computed variables
CODES_VERSION = 2

# Codes by number of digits of the column maximum: (to NaN, to 0)
CODES_BY_DIGITS = {
    2: ([77, 99], [88]),
    3: ([777, 999], [555, 888]),
    4: ([7777, 9999], [8888]),
    5: ([77777, 99999, 99900], []),
}


def read_column_names(csv_path):
    """Feature names from the CSV header, without the Id column."""
    with open(csv_path) as f:
        return f.readline().strip().split(",")[1:]


def _is_computed(name):
    """CDC computed variables (_AGE80, PADUR1_, ...) have no "none" codes."""
    return name.startswith("_") or name.endswith("_")


def _column_rules(col, computed=False):
    """Codes of one column: (values to NaN, values to 0).

    A code only counts when the value right below it is absent, so real
    values are kept (e.g. 77 years in _AGE80, 7 = "student" in EMPLOY1).
    Computed variables only use "don't know" codes (e.g. 900 in DROCDY3_):
    in them 555 or 888 are real values, never "none".
    """
    vals = np.unique(col[~np.isnan(col)])
    if vals.size == 0 or np.any(np.mod(vals, 1) != 0) or vals.max() < 7:
        return [], []
    present = set(vals.tolist())

    def isolated(c):
        return c in present and (c - 1) not in present

    digits = len(str(int(vals.max())))
    if digits == 1:
        # small categorical answers: 9 = refused, 7 = don't know (if 6 is unused)
        to_nan = [9] if vals.max() == 9 and vals.size <= 10 else []
        if to_nan and isolated(7):
            to_nan.append(7)
        return to_nan, []
    nan_codes, zero_codes = CODES_BY_DIGITS.get(digits, ([], []))
    if computed:
        nan_codes, zero_codes = nan_codes + ([900] if digits == 3 else []), []
    return [c for c in nan_codes if isolated(c)], [c for c in zero_codes if isolated(c)]


def _convert_units(x, names):
    """WEIGHT2 to pounds and HEIGHT3 to inches (in place, codes already NaN)."""
    if "WEIGHT2" in names:
        w = x[:, names.index("WEIGHT2")]
        kg = (w >= 9000) & (w < 9999)
        w[kg] = (w[kg] - 9000) * 2.20462
    if "HEIGHT3" in names:
        h = x[:, names.index("HEIGHT3")]
        ft_in = (h >= 200) & (h < 1000)
        h[ft_in] = (h[ft_in] // 100) * 12 + h[ft_in] % 100
        cm = (h >= 9000) & (h < 9999)
        h[cm] = (h[cm] - 9000) / 2.54


def _fix_frequencies(x, names):
    """Mixed-unit frequencies to times per month, H:MM durations to minutes (in place).

    Run after the code cleaning: "none" is already 0 and "don't know" NaN.
    """
    for name, per_month in FREQUENCY_COLUMNS.items():
        if name not in names:
            continue
        col = x[:, names.index(name)]
        unit = np.floor(col / 100)
        for digit, factor in per_month.items():
            sel = unit == digit
            col[sel] = (col[sel] - 100 * digit) * factor
    for name in DURATION_COLUMNS:
        if name in names:
            col = x[:, names.index(name)]
            col[:] = np.floor(col / 100) * 60 + np.mod(col, 100)


def fit_code_cleaner(x, names, fix_units=False, version=CODES_VERSION):
    """Learn the code rules and the nominal categories on the fit matrix.

    Input: x (N, D) raw features, names list of D column names, fix_units
    to also convert the mixed-unit frequencies (see _fix_frequencies),
    version of the rules (1 = before the computed-variable rules, kept so
    that older checkpoints get exactly the data they were trained on).
    Output: state dict for apply_code_cleaner.
    """
    rules = [
        _column_rules(x[:, j], version >= 2 and _is_computed(names[j]))
        for j in range(x.shape[1])
    ]
    remove = np.array([n in ADMIN_COLUMNS or n in NOMINAL_COLUMNS for n in names])
    nominal = {}
    for n in NOMINAL_COLUMNS:
        if n in names:
            j = names.index(n)
            col = x[:, j].copy()
            col[np.isin(col, rules[j][0])] = np.nan
            nominal[j] = np.unique(col[~np.isnan(col)])
    return {
        "names": list(names),
        "rules": rules,
        "remove": remove,
        "nominal": nominal,
        "fix_units": fix_units,
    }


def apply_code_cleaner(x, state):
    """Apply the fitted rules. Leaves `x` unchanged.

    Output: numeric (N, D_numeric) with codes cleaned, ready for the usual
    preprocessing; onehot (N, K) 0/1 columns of the nominal features
    (all zeros for a NaN or a category unseen in the fit split).
    """
    z = x.copy()
    for j, (to_nan, to_zero) in enumerate(state["rules"]):
        if to_nan:
            z[np.isin(z[:, j], to_nan), j] = np.nan
        if to_zero:
            z[np.isin(z[:, j], to_zero), j] = 0.0
    _convert_units(z, state["names"])
    if state["fix_units"]:
        _fix_frequencies(z, state["names"])
    blocks = [(z[:, j][:, None] == cats[None, :]).astype(float) for j, cats in state["nominal"].items()]
    onehot = np.hstack(blocks) if blocks else np.zeros((x.shape[0], 0))
    return z[:, ~state["remove"]], onehot


def describe(state):
    """Short summary of what the cleaner changes (for logs and the report)."""
    n_nan = sum(1 for r in state["rules"] if r[0])
    n_zero = sum(1 for r in state["rules"] if r[1])
    n_onehot = sum(len(c) for c in state["nominal"].values())
    return "codes->NaN in {} columns, none->0 in {}, removed {} columns, {} one-hot columns".format(
        n_nan, n_zero, int(state["remove"].sum()), n_onehot
    )
