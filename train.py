# -*- coding: utf-8 -*-
"""
Train the MLP: dataset -> loader -> model -> loss + optimizer -> epochs.

Run from the project root:

    python train.py --epochs 50                   # validation run, picks the threshold
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
from util import Adam, BCELoss, best_threshold, classification_metrics

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


def evaluate(model, loss_fn, dataset, threshold=None):
    """Loss and classification metrics on a labelled Dataset.

    With threshold=None the threshold that maximizes F1 is picked.
    """
    logits = model.predict_logits(dataset.x)
    loss, _ = loss_fn(logits, dataset.y)
    probs = 1.0 / (1.0 + np.exp(-logits))
    if threshold is None:
        threshold, metrics = best_threshold(probs, dataset.y)
    else:
        metrics = classification_metrics(probs, dataset.y, threshold)
    metrics["threshold"] = float(threshold)
    metrics["loss"] = loss
    return metrics


def train(model, optimizer, loss_fn, train_loader, epochs, val_ds=None, threshold=None,
          patience=10, checkpoint=None, verbose=True):
    """Train for a number of epochs, evaluating on val_ds after each one.

    With a validation set, the weights with the lowest validation loss are
    kept (and saved to `checkpoint`), and training stops after `patience`
    epochs without improvement. Without one, the last weights are kept.

    Returns:
        history: dict with the per-epoch "train_loss", validation metrics
        "val", "best_epoch" (1-based, None without validation) and the
        decision "threshold" of the returned model.
    """
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
                    model.save(checkpoint)
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
        model.save(checkpoint)
        if verbose:
            print("weights of the last epoch saved to " + checkpoint)
    return history


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
    # training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=10,
                   help="stop after this many epochs without a better val loss")
    p.add_argument("--checkpoint", default=os.path.join(CHECKPOINT_DIR, "mlp_best.npz"),
                   help="where the best weights are saved")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--pos-weight", type=parse_pos_weight, default="auto",
                   help="weight of the positives in the BCE; 'auto' = n_negatives / n_positives (~10)")
    # network
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    p.add_argument("--no-batch-norm", action="store_true")
    # data
    p.add_argument("--data", default=DATA_PATH)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--sub-sample", action="store_true", help="keep 1 row out of 50")
    p.add_argument("--seed", type=int, default=42)
    # prediction
    p.add_argument("--threshold", type=float, default=None,
                   help="decision threshold in (0, 1); by default the best F1 on validation")
    p.add_argument("--final", action="store_true", help="train on all data, write the submission")
    p.add_argument("--submission", default="submissions/submission_mlp.csv")
    args = p.parse_args()
    if args.threshold is not None and not 0.0 < args.threshold < 1.0:
        p.error("--threshold must be in (0, 1)")
    if args.final and args.threshold is None:
        p.error("--final has no validation set to pick the threshold on: pass --threshold, "
                "e.g. the 'threshold' printed for the best epoch of a validation run")
    return args


def main():
    args = parse_args()

    # 1) datasets
    val_fraction = 0.0 if args.final else args.val_fraction
    train_ds, val_ds, test_ds, test_ids = load_datasets(
        args.data, val_fraction, args.seed, args.sub_sample
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
    print("bce pos_weight: {:.2f}".format(pos_weight))
    loss_fn = BCELoss(pos_weight)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 4) epochs (keeps the best weights on validation, stops early)
    checkpoint_dir = os.path.dirname(args.checkpoint)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    history = train(
        model, optimizer, loss_fn, train_loader, args.epochs, val_ds,
        args.threshold, args.patience, args.checkpoint,
    )

    if args.final:
        y_pred = model.predict_labels(test_ds.x, history["threshold"])
        create_csv_submission(test_ids, y_pred, args.submission)
        print("saved {} ({} positive predictions, threshold {})".format(
            args.submission, np.sum(y_pred == 1), history["threshold"]
        ))


if __name__ == "__main__":
    main()
