
"""
Inference with a trained checkpoint: evaluate it on the validation split and
write the predictions on the test set.
"""

import argparse

import numpy as np

from helpers import create_csv_submission
from loader import DATA_PATH, load_datasets
from neural_net import MLP
from util import THRESHOLD_METRICS, classification_metrics, threshold_table


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a checkpoint and predict the test set.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default=DATA_PATH)
    p.add_argument(
        "--threshold",
        default="checkpoint",
        help="'checkpoint' (the one chosen in training, 0.5 for old checkpoints), "
        "a number, or best_f1 / best_bal_acc to pick it on the validation split",
    )
    p.add_argument("--submission", default=None, help="output csv (default: next to the checkpoint)")
    return p.parse_args()


def main():
    args = parse_args()
    model = MLP.load(args.checkpoint)
    meta = MLP.load_meta(args.checkpoint)
    if meta:
        print("checkpoint trained with: " + ", ".join("{}={}".format(k, v) for k, v in meta.items()))
    else:
        print("checkpoint without metadata: assuming val_fraction=0.2, max_missing_rate=0.5, "
              "no missing indicators, seed=42, no sub-sampling")

    _, val_ds, test_ds, test_ids = load_datasets(
        args.data,
        val_fraction=meta.get("val_fraction", 0.2),
        max_missing_rate=meta.get("max_missing_rate", 0.5),
        missing_indicators=meta.get("missing_indicators", False),
        clean_codes=meta.get("clean_codes", False),
        fix_units=meta.get("fix_units", False),
        log_skewed=meta.get("log_skewed", False),
        codes_version=meta.get("codes_version", 1),
        seed=meta.get("seed", 42),
        sub_sample=meta.get("sub_sample", False),
    )
    if test_ds.x.shape[1] != model.config["input_dim"]:
        raise ValueError(
            "the data has {} features but the model expects {}: the preprocessing "
            "differs from training".format(test_ds.x.shape[1], model.config["input_dim"])
        )

    if args.threshold == "checkpoint":
        threshold = float(meta.get("threshold", 0.5))
        print("threshold from the checkpoint: {}".format(threshold))
    elif args.threshold in THRESHOLD_METRICS:
        threshold = None  # picked on validation below
    else:
        threshold = float(args.threshold)

    if val_ds is not None:
        probs = model.predict_proba(val_ds.x)
        print("\nvalidation: {} rows, {:.1%} positive, mean predicted probability {:.3f}".format(
            len(val_ds), val_ds.y.mean(), probs.mean()
        ))
        rows = threshold_table(probs, val_ds.y)
        print("threshold  pred_pos  accuracy  bal_acc  precision  recall    f1")
        for r in rows:
            print("  {:5.2f}    {:6.1%}    {:.4f}   {:.4f}    {:.4f}   {:.4f}  {:.4f}".format(
                r["threshold"], r["pos_rate"], r["accuracy"], r["balanced_accuracy"],
                r["precision"], r["recall"], r["f1"],
            ))
        if threshold is None:
            threshold = max(rows, key=lambda r: r[THRESHOLD_METRICS[args.threshold]])["threshold"]
            print("\n{} on validation -> threshold {}".format(args.threshold, threshold))
        m = classification_metrics(probs, val_ds.y, threshold)
        print("validation @ {}: ".format(threshold) + ", ".join(
            "{}={:.4f}".format(k, v) for k, v in m.items()
        ))
    elif threshold is None:
        raise ValueError("no validation split in this checkpoint: use the saved or a numeric threshold")

    y_pred = model.predict_labels(test_ds.x, threshold)
    out = args.submission or args.checkpoint.replace(".npz", "_submission.csv")
    create_csv_submission(test_ids, y_pred, out)
    print("test: {} rows, {:.1%} predicted positive -> {}".format(
        len(y_pred), np.mean(y_pred == 1), out
    ))


if __name__ == "__main__":
    main()
