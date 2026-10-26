"""Feature maps for the linear models.

`fit_clinical_terms` is what section 5 trains on, including logistic regression.
It keeps every codebook column and adds squares of age and BMI plus products
of age with sex, smoking, diagnosed diabetes, general health, and high
cholesterol, and of BMI with smoking. The logistic sigmoid does not replace
those products. It only bends the final score.

`fit_feature_map` is the earlier partial-R^2 selector. It is not the gate
for the submission matrix. A binary column is not squared there, because a
bias term already spans that square.
"""

import numpy as np

from src.column_significance import leave_one_column_relation
from src.preprocessing import standardize, standardization_stats


def association_with_target(x, y):
    """Absolute gap between the class means of each column.

    Input: x (N, D) already standardized, y (N,) with positive label 1.
    Output: gaps (D,). A larger value means the column separates target 1
    from the other class by more standard deviations.
    """
    labels = np.asarray(y).reshape(-1)
    if labels.shape[0] != x.shape[0]:
        raise ValueError("y must have one label per row")
    positive = labels == 1
    if not np.any(positive) or np.all(positive):
        raise ValueError("both target 1 and the other class are required")
    gap = np.mean(x[positive], axis=0) - np.mean(x[~positive], axis=0)
    return np.abs(gap)


def fit_feature_map(
    x,
    y,
    n_keep,
    n_interact,
    min_partial_r2,
    eps,
    source_columns=None,
):
    """Choose columns, squares, and products, and store their standardization.

    A column is kept when leaving it out raises the weighted residual enough
    to give a partial R^2 of at least `min_partial_r2`. At most `n_keep`
    survivors are kept, strongest first. Pairwise products use the
    `n_interact` strongest of those. Squares are added for every kept column
    that takes more than two values.

    Input: x (N, D) standardized, y (N,), counts and floors as above,
    eps for a zero standard deviation, source_columns (D,) original indices.
    Output: state dict consumed by `transform_feature_map`.
    """
    if n_keep < 1:
        raise ValueError("n_keep must be at least 1")
    if n_interact < 0:
        raise ValueError("n_interact must be at least 0")
    if not np.isfinite(min_partial_r2) or min_partial_r2 < 0:
        raise ValueError("min_partial_r2 must be finite and at least 0")

    marginal_gap = association_with_target(x, y)
    partial_r2, _delta = leave_one_column_relation(x, y)
    order = np.argsort(-partial_r2, kind="mergesort")
    eligible = order[partial_r2[order] >= min_partial_r2]
    if eligible.size == 0:
        raise ValueError("no feature passed the partial R^2 floor")
    keep = np.asarray(eligible[:n_keep], dtype=int)
    n_pairs_src = min(n_interact, keep.size)
    selected = x[:, keep]
    square_pos = _nonlinear_positions(selected)

    expanded = _expand(selected, square_pos, n_pairs_src)
    means, scales = standardization_stats(expanded, eps)
    if source_columns is None:
        original = keep.copy()
    else:
        source_columns = np.asarray(source_columns)
        original = source_columns[keep]

    return {
        "keep": keep,
        "square_pos": square_pos,
        "n_interact": n_pairs_src,
        "means": means,
        "scales": scales,
        "marginal_gap": marginal_gap,
        "partial_r2": partial_r2,
        "scores": partial_r2,
        "selected_scores": partial_r2[keep],
        "original_columns": original,
        "n_linear": int(keep.size),
        "n_squares": int(square_pos.size),
        "n_products": int(n_pairs_src * (n_pairs_src - 1) // 2),
    }


def transform_feature_map(x, state):
    """Apply a fitted feature map. Do not refit the selection.

    Input: x (N, D) standardized with the same preprocessor as the fit,
    state from `fit_feature_map`.
    Output: expanded and standardized array (N, D_expanded).
    """
    selected = x[:, state["keep"]]
    expanded = _expand(selected, state["square_pos"], state["n_interact"])
    return standardize(expanded, state["means"], state["scales"])


def _nonlinear_positions(selected):
    """Positions, inside the selected block, of columns with 3 or more values."""
    positions = []
    for j in range(selected.shape[1]):
        if np.unique(selected[:, j]).size > 2:
            positions.append(j)
    return np.asarray(positions, dtype=int)


CLINICAL_SQUARE_NAMES = ("age80_sq", "bmi5_sq")
CLINICAL_PRODUCT_NAMES = (
    "age_x_sex",
    "age_x_smoker",
    "age_x_diab_yes",
    "age_x_genhlth",
    "age_x_rfchol",
    "bmi_x_smoker",
)
# Columns that move heart-disease risk in BRFSS and in this file's partial R^2.
# Degree 2 is built on these only. The other codebook columns stay linear.
RISK_NAMES = (
    "_AGE80",
    "GENHLTH",
    "_BMI5",
    "_SMOKER3",
    "SEX",
    "_RFCHOL",
    "CVDSTRK3",
    "_RFHYPE5",
    "diab_yes",
    "CHCCOPD1",
    "CHCKIDNY",
    "BPMEDS",
    "employ_unable",
)


def fit_clinical_terms(x, names, eps):
    """Add the codebook nonlinear terms and store their standardization.

    The terms are fixed. They are the square of `_AGE80`, the square of
    `_BMI5`, and the products of age with sex, `_SMOKER3`, `diab_yes`,
    `GENHLTH`, and `_RFCHOL`, plus `_BMI5` with `_SMOKER3`. Nothing is
    dropped because of a column score.

    Input: x (N, D) standardized codebook columns, names length D, eps for a
    zero standard deviation.
    Output: state dict consumed by `transform_clinical_terms`.
    """
    squares, products = _clinical_layout(names)
    if len(squares) != len(CLINICAL_SQUARE_NAMES):
        raise ValueError("square names do not match the layout")
    if len(products) != len(CLINICAL_PRODUCT_NAMES):
        raise ValueError("product names do not match the layout")
    expanded = _clinical_expand(x, squares, products)
    means, scales = standardization_stats(expanded, eps)
    expanded_names = list(names) + list(CLINICAL_SQUARE_NAMES)
    expanded_names.extend(CLINICAL_PRODUCT_NAMES)
    return {
        "squares": squares,
        "products": products,
        "means": means,
        "scales": scales,
        "n_linear": int(x.shape[1]),
        "n_squares": len(squares),
        "n_products": len(products),
        "expanded_names": expanded_names,
    }


def fit_risk_terms(x, names, eps):
    """Clinical terms plus a degree-2 map of the heart-disease columns.

    Squares are added for every risk column with three or more distinct
    values. Pairwise products cover every pair among the risk columns that
    are present. Pairs and squares already built by the clinical terms are
    not repeated. A risk column dropped as constant inside a group is
    skipped. The expanded block is standardized once.

    Input: x (N, D) standardized codebook columns, names length D, eps for a
    zero standard deviation.
    Output: state dict consumed by `transform_clinical_terms`.
    """
    squares, products = _clinical_layout(names)
    square_names = list(CLINICAL_SQUARE_NAMES)
    product_names = list(CLINICAL_PRODUCT_NAMES)
    seen_squares = set(squares)
    seen_pairs = {frozenset(pair) for pair in products}
    position = {name: index for index, name in enumerate(names)}
    present = [name for name in RISK_NAMES if name in position]
    for name in present:
        index = position[name]
        if index in seen_squares or np.unique(x[:, index]).size <= 2:
            continue
        squares.append(index)
        square_names.append(_short_name(name) + "_sq")
        seen_squares.add(index)
    for left_at, left_name in enumerate(present):
        for right_name in present[left_at + 1 :]:
            pair = frozenset((position[left_name], position[right_name]))
            if pair in seen_pairs:
                continue
            products.append((position[left_name], position[right_name]))
            product_names.append(_short_name(left_name) + "_x_" + _short_name(right_name))
            seen_pairs.add(pair)
    expanded = _clinical_expand(x, squares, products)
    means, scales = standardization_stats(expanded, eps)
    return {
        "squares": squares,
        "products": products,
        "means": means,
        "scales": scales,
        "n_linear": int(x.shape[1]),
        "n_squares": len(squares),
        "n_products": len(products),
        "expanded_names": list(names) + square_names + product_names,
    }


def _short_name(name):
    """Stable piece of an expanded-column name."""
    return name.lower().lstrip("_")


def transform_clinical_terms(x, state):
    """Apply a fitted clinical expansion. Do not refit the standardization.

    Input: x (N, D) standardized with the same codebook columns as the fit,
    state from `fit_clinical_terms`.
    Output: expanded and standardized array. Squares and products stored in
    `state` are appended to the codebook columns.
    """
    expanded = _clinical_expand(x, state["squares"], state["products"])
    return standardize(expanded, state["means"], state["scales"])


def _clinical_layout(names):
    """Column positions for the clinical squares and products."""
    position = {name: index for index, name in enumerate(names)}
    required = (
        "_AGE80",
        "_BMI5",
        "SEX",
        "_SMOKER3",
        "diab_yes",
        "GENHLTH",
        "_RFCHOL",
    )
    missing = [name for name in required if name not in position]
    if missing:
        raise ValueError("clinical terms need " + ", ".join(missing))
    squares = [position["_AGE80"], position["_BMI5"]]
    products = [
        (position["_AGE80"], position["SEX"]),
        (position["_AGE80"], position["_SMOKER3"]),
        (position["_AGE80"], position["diab_yes"]),
        (position["_AGE80"], position["GENHLTH"]),
        (position["_AGE80"], position["_RFCHOL"]),
        (position["_BMI5"], position["_SMOKER3"]),
    ]
    return squares, products


def _clinical_expand(x, squares, products):
    """Linear columns, then the two squares, then the age and BMI products."""
    extra = [x[:, index] ** 2 for index in squares]
    extra.extend(x[:, left] * x[:, right] for left, right in products)
    return np.column_stack([x, *extra])


def _expand(selected, square_pos, n_interact):
    """Linear columns, then squares, then pairwise products of the strongest."""
    n_rows, n_keep = selected.shape
    n_squares = square_pos.shape[0]
    n_products = n_interact * (n_interact - 1) // 2
    out = np.empty((n_rows, n_keep + n_squares + n_products), dtype=float)
    out[:, :n_keep] = selected
    if n_squares:
        out[:, n_keep : n_keep + n_squares] = selected[:, square_pos] ** 2
    col = n_keep + n_squares
    for a in range(n_interact):
        left = selected[:, a]
        for b in range(a + 1, n_interact):
            out[:, col] = left * selected[:, b]
            col += 1
    return out
