"""Load the feature and training settings used by section 5.

`config/features.json` chooses imputation, the BMI cap, and which mostly
empty columns stay. `config/train.json` chooses the models, the class
weight, the grids, and early stopping. Edit those files and rerun
`python run.py` from the project root.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "config" / "features.json"
TRAIN_PATH = ROOT / "config" / "train.json"

_FEATURE_KEYS = (
    "max_missing_rate",
    "missing_flag_min_rate",
    "impute",
    "eps",
    "bmi_cap",
)
_TRAIN_KEYS = (
    "models",
    "positive_class_cost",
    "val_fraction",
    "seed",
    "submission_dir",
    "use_gpu",
    "early_stopping",
    "grids",
)
_MODEL_NAMES = (
    "least_squares",
    "ridge",
    "gd",
    "sgd",
    "logistic",
    "reg_logistic",
)


def load():
    """Return `(features, train)` dicts checked against the expected keys."""
    features = _read(FEATURES_PATH, _FEATURE_KEYS)
    train = _read(TRAIN_PATH, _TRAIN_KEYS)
    if features["impute"] not in ("median", "mean", "mode"):
        raise ValueError("impute must be 'median', 'mean', or 'mode'")
    if not 0.0 <= features["max_missing_rate"] <= 1.0:
        raise ValueError("max_missing_rate must be between 0 and 1")
    if not 0.0 <= features["missing_flag_min_rate"] <= 1.0:
        raise ValueError("missing_flag_min_rate must be between 0 and 1")
    if features["bmi_cap"] <= 0:
        raise ValueError("bmi_cap must be above 0")
    if not train["models"]:
        raise ValueError("models must list at least one model")
    unknown = [name for name in train["models"] if name not in _MODEL_NAMES]
    if unknown:
        raise ValueError("unknown models: " + ", ".join(unknown))
    if train["positive_class_cost"] <= 0:
        raise ValueError("positive_class_cost must be above 0")
    if not 0.0 < train["val_fraction"] < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    return features, train


def _read(path, required):
    if not path.is_file():
        raise FileNotFoundError(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    missing = [key for key in required if key not in loaded]
    if missing:
        raise ValueError(str(path) + " is missing " + ", ".join(missing))
    return loaded
