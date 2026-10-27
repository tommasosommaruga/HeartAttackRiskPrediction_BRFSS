# -*- coding: utf-8 -*-
"""
Train the MLP: dataset -> loader -> model -> loss + optimizer -> epochs.

Run from the project root:

    python train.py --loss bce --epochs 100
    python train.py --loss bce --optimizer adam --pos-weight 5 --threshold best_bal_acc
    python train.py --final --threshold 0.45      # all labelled data, writes the submission

python train.py --help lists every option.
"""

import argparse
import os
import time

import numpy as np

from helpers import create_csv_submission
from loader import DATA_PATH, DataLoader, load_datasets
from neural_net import MLP
from src.brfss_codes import CODES_VERSION
from util import (
    LOSSES,
    OPTIMIZERS,
    THRESHOLD_METRICS,
    best_threshold,
    build_loss,
    build_optimizer,
    classification_metrics,
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def train_one_epoch(model, optimizer, loss_fn, loader):
    """One pass over the loader. Returns the mean training loss."""
    total, count = 0.0, 0
    for xb, yb in loader:
        if xb.shape[0] < 2:
            continue  # BatchNorm needs at least two samples
        optimizer.zero_grad()
        loss, grad = loss_fn(model.forward(xb, training=True), yb)
        model.backward(grad)
        optimizer.step()
        total += loss * xb.shape[0]
        count += xb.shape[0]
    return total / count


def evaluate(model, loss_fn, dataset, threshold=0.5):
    """Loss and classification metrics of the model on a labelled Dataset.

    `threshold` is a number, or a key of THRESHOLD_METRICS ("best_f1",
    "best_bal_acc") to use the threshold that maximizes that metric on this
    dataset. The threshold actually used is returned under "threshold".
    """
    logits = model.predict_logits(dataset.x)
    loss, _ = loss_fn(logits, dataset.y)
    probs = 1.0 / (1.0 + np.exp(-logits))
    if threshold in THRESHOLD_METRICS:
        threshold, metrics = best_threshold(probs, dataset.y, THRESHOLD_METRICS[threshold])
    else:
        metrics = classification_metrics(probs, dataset.y, threshold)
    metrics["threshold"] = float(threshold)
    metrics["loss"] = loss
    return metrics


def train(
    model,
    optimizer,
    loss_fn,
    train_loader,
    epochs,
    val_ds=None,
    threshold=0.5,
    patience=10,
    checkpoint=None,
    checkpoint_meta=None,
    verbose=True,
):
    """Train for a number of epochs, evaluating on val_ds after each one.

    With a validation set, the weights with the lowest validation loss are
    kept (and saved to `checkpoint`, if given, together with the decision
    threshold chosen at that epoch); training stops early when the
    validation loss has not improved for `patience` epochs, and the best
    weights are restored at the end. Without a validation set, the weights
    of the last epoch are saved with the given numeric threshold.
    `checkpoint_meta` is stored in the file next to the weights (see MLP.save).

    Returns:
        history: dict with the per-epoch "train_loss", validation metrics
        "val", "best_epoch" (1-based, None without validation) and the
        decision "threshold" of the returned model.
    """
    meta = dict(checkpoint_meta or {})
    history = {"train_loss": [], "val": [], "best_epoch": None, "threshold": threshold}
    best_loss, best_state, bad_epochs = np.inf, None, 0
    for epoch in range(epochs):
        start = time.time()
        train_loss = train_one_epoch(model, optimizer, loss_fn, train_loader)
        history["train_loss"].append(train_loss)
        msg = "epoch {}/{}: train_loss={:.4f}".format(epoch + 1, epochs, train_loss)
        if val_ds is not None:
            m = evaluate(model, loss_fn, val_ds, threshold)
            history["val"].append(m)
            msg += ", val_loss={:.4f} | @{:.2f}: bal_acc={:.4f}, f1={:.4f}".format(
                m["loss"], m["threshold"], m["balanced_accuracy"], m["f1"]
            )
            if m["loss"] < best_loss:
                best_loss, best_state, bad_epochs = m["loss"], model.state_dict(), 0
                history["best_epoch"] = epoch + 1
                history["threshold"] = m["threshold"]
                if checkpoint:
                    meta["threshold"] = m["threshold"]
                    model.save(checkpoint, meta)
                msg += " *best*"
            else:
                bad_epochs += 1
        if verbose:
            print(msg + " ({:.1f}s)".format(time.time() - start))
        if val_ds is not None and bad_epochs >= patience:
            if verbose:
                print("early stopping: no improvement for {} epochs".format(patience))
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            best = history["val"][history["best_epoch"] - 1]
            print("best epoch {}: val_loss={:.4f}, threshold={:.2f}, bal_acc={:.4f}, f1={:.4f}{}".format(
                history["best_epoch"], best_loss, best["threshold"],
                best["balanced_accuracy"], best["f1"],
                ", weights saved to " + checkpoint if checkpoint else "",
            ))
    elif checkpoint:
        meta["threshold"] = float(threshold)
        model.save(checkpoint, meta)
        if verbose:
            print("weights of the last epoch saved to " + checkpoint)
    return history


def parse_threshold(value):
    """--threshold: a number in (0, 1) or one of THRESHOLD_METRICS."""
    if value in THRESHOLD_METRICS:
        return value
    try:
        t = float(value)
    except ValueError:
        t = -1.0
    if not 0.0 < t < 1.0:
        raise argparse.ArgumentTypeError(
            "use a number in (0, 1) or one of {}".format(", ".join(THRESHOLD_METRICS))
        )
    return t


def parse_pos_weight(value):
    """--pos-weight: a positive number, or "auto" for n_negatives / n_positives."""
    if value == "auto":
        return value
    w = float(value)
    if w <= 0:
        raise argparse.ArgumentTypeError("pos-weight must be positive")
    return w


def parse_args():
    p = argparse.ArgumentParser(description="Train the MLP on the project data.")
    # what to train with
    p.add_argument("--loss", default="bce", choices=sorted(LOSSES))
    p.add_argument("--optimizer", default="adam", choices=sorted(OPTIMIZERS))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=10,
                   help="stop after this many epochs without a better val loss")
    p.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "mlp_best.npz"),
                   help="where the best weights are saved")
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--weight-decay", type=float, default=0.0)
    # loss options
    p.add_argument("--pos-weight", type=parse_pos_weight, default="auto",
                   help="bce: weight of the positives; 'auto' = n_negatives / n_positives (~10)")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=0.5)
    # network
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    p.add_argument("--no-batch-norm", action="store_true")
    # data
    p.add_argument("--data", default=DATA_PATH)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--max-missing-rate", type=float, default=0.5,
                   help="drop columns with more NaNs than this (1.0 keeps all)")
    p.add_argument("--missing-indicators", action="store_true",
                   help="add a 0/1 'was missing' column for every feature with NaNs")
    p.add_argument("--no-clean-codes", dest="clean_codes", action="store_false",
                   help="keep the raw BRFSS codes (by default 77/99 -> NaN, 88 -> 0, "
                   "nominal columns one-hot)")
    p.add_argument("--fix-units", action="store_true",
                   help="convert mixed-unit frequencies (105 = 5/day, 205 = 5/week, ...) "
                   "to times per month and H:MM durations to minutes")
    p.add_argument("--log-skewed", action="store_true",
                   help="apply log(1 + x) to the long-tailed non-negative columns")
    p.add_argument("--sub-sample", action="store_true", help="keep 1 row out of 50")
    p.add_argument("--seed", type=int, default=42)
    # prediction
    p.add_argument("--threshold", type=parse_threshold, default="best_f1",
                   help="decision threshold: a number, or best_f1 / best_bal_acc picked on validation")
    p.add_argument("--final", action="store_true", help="train on all data, write the submission")
    p.add_argument("--submission", default="submissions/submission_mlp.csv")
    args = p.parse_args()
    if args.final and args.threshold in THRESHOLD_METRICS:
        p.error("--final has no validation set to pick the threshold on: pass a number, "
                "e.g. the 'threshold' printed for the best epoch of a validation run")
    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # 1) datasets
    val_fraction = 0.0 if args.final else args.val_fraction
    train_ds, val_ds, test_ds, test_ids = load_datasets(
        args.data,
        val_fraction=val_fraction,
        max_missing_rate=args.max_missing_rate,
        missing_indicators=args.missing_indicators,
        clean_codes=args.clean_codes,
        fix_units=args.fix_units,
        log_skewed=args.log_skewed,
        seed=args.seed,
        sub_sample=args.sub_sample,
        verbose=True,
    )
    print("train: {} ({:.2%} positive), val: {} ({}), test: {}, features: {}".format(
        len(train_ds), train_ds.y.mean(),
        len(val_ds) if val_ds else 0,
        "{:.2%} positive".format(val_ds.y.mean()) if val_ds else "-",
        len(test_ds), train_ds.x.shape[1],
    ))

    # 2) loader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, seed=args.seed)

    # 3) model, loss, optimizer
    model = MLP(train_ds.x.shape[1], args.hidden, batch_norm=not args.no_batch_norm, seed=args.seed)

    pos_weight = args.pos_weight
    if pos_weight == "auto":
        n_pos = train_ds.y.sum()
        pos_weight = (len(train_ds) - n_pos) / n_pos
    loss_kwargs = {
        "bce": {"pos_weight": pos_weight},
        "focal": {"gamma": args.focal_gamma, "alpha": args.focal_alpha},
        "mse": {},
    }[args.loss]
    loss_fn = build_loss(args.loss, **loss_kwargs)
    if args.loss == "bce":
        print("bce pos_weight: {:.2f}".format(pos_weight))

    opt_kwargs = {"weight_decay": args.weight_decay}
    if args.lr is not None:
        opt_kwargs["lr"] = args.lr
    optimizer = build_optimizer(args.optimizer, model.parameters(), **opt_kwargs)

    # 4) epochs (keeps the best weights on validation, stops early)
    checkpoint_dir = os.path.dirname(args.checkpoint)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    # the data settings let infer.py rebuild the same preprocessing;
    # the rest records how the weights were obtained
    meta = {
        "val_fraction": val_fraction,
        "max_missing_rate": args.max_missing_rate,
        "missing_indicators": args.missing_indicators,
        "clean_codes": args.clean_codes,
        "fix_units": args.fix_units,
        "codes_version": CODES_VERSION,
        "log_skewed": args.log_skewed,
        "seed": args.seed,
        "sub_sample": args.sub_sample,
        "loss": args.loss,
        "optimizer": args.optimizer,
        "lr": optimizer.lr,
        "batch_size": args.batch_size,
        "pos_weight": pos_weight if args.loss == "bce" else 1.0,
        "weight_decay": args.weight_decay,
        "threshold_rule": str(args.threshold),
    }
    history = train(
        model, optimizer, loss_fn, train_loader, args.epochs, val_ds,
        args.threshold, args.patience, args.checkpoint, meta,
    )

    if args.final:
        y_pred = model.predict_labels(test_ds.x, history["threshold"])
        create_csv_submission(test_ids, y_pred, args.submission)
        print("saved {} ({} positive predictions, threshold {})".format(
            args.submission, np.sum(y_pred == 1), history["threshold"]
        ))


if __name__ == "__main__":
    main()
