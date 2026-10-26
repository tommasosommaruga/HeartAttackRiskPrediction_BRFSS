"""Train the models listed in config/train.json.

Run this from the project folder:

    python train_early_stop.py

Which models, the class weight, the grids, and early stopping are in
config/train.json. Imputation, the BMI cap, and the missing-column rules
are in config/features.json. Edit those files to change a run. The default
list is logistic regression, the model with the best class-1 validation F1.

Full batch gradient descent and logistic regression can use an NVIDIA GPU
when CuPy is installed:

    USE_GPU=1 python train_early_stop.py

Stochastic gradient descent stays on the CPU, because each step uses one row.
Without CuPy the same file runs on the CPU. Copy the whole project, including
data/dataset, config, and run.ipynb. This script runs the submission cell in
that notebook. That cell recodes columns from the BRFSS codebook before the
mode fill, keeps sparse health items, then adds the clinical nonlinear terms.
"""

import json
import os
from pathlib import Path

import numpy as np

import src.column_significance as column_significance
import src.features as features
import src.positive_class_weights as weights
import src.preprocessing as prep
from helpers import create_csv_submission, load_csv_data
from implementations import calculate_loss, compute_loss


def submission_cell_source():
    """Return the notebook cell that defines train_quiet and writes submissions."""
    notebook = json.loads(Path("run.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "def train_quiet" in source:
            return source
    raise RuntimeError("submission cell not found in run.ipynb")


def main():
    if os.environ.get("USE_GPU", "") == "1":
        print("USE_GPU=1. Full batch steps will try CuPy.")
    else:
        print("CPU run. On a CUDA machine with CuPy, use USE_GPU=1.")
    x_train, x_test, y_train, _ids_train, ids_test = load_csv_data("data/dataset")
    shared = {
        "__name__": "__main__",
        "np": np,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "ids_test": ids_test,
        "calculate_loss": calculate_loss,
        "compute_loss": compute_loss,
        "create_csv_submission": create_csv_submission,
        "column_significance": column_significance,
        "features": features,
        "weights": weights,
        "prep": prep,
    }
    exec(compile(submission_cell_source(), "run.ipynb", "exec"), shared)


if __name__ == "__main__":
    main()
