# -*- coding: utf-8 -*-
"""
Training and prediction for the MLP in neural_net.py.

Example:
    from neural_net import MLP
    from util import build_optimizer
    from train import train, predict_labels

    model = MLP(input_dim=x_fit.shape[1], hidden_dims=[128, 64], seed=42)
    opt = build_optimizer("muon", model.parameters())   # or "adam"
    history = train(model, opt, x_fit, y_fit, epochs=10, batch_size=256,
                    x_val=x_val, y_val=y_val)
    y_pred = predict_labels(model, x_test)
"""

import numpy as np


def bce_with_logits(logits, y, pos_weight=1.0):
    """Binary cross-entropy computed on logits, numerically stable.

    Args:
        logits: shape=(N,), raw model outputs.
        y: shape=(N,), labels in {0, 1}.
        pos_weight: weight of the positive class (>1 helps with imbalance).

    Returns:
        loss: scalar mean loss.
        grad: shape=(N,), gradient of the loss w.r.t. the logits.
    """
    softplus = np.logaddexp(0.0, logits)  # stable log(1 + e^z)
    weights = np.where(y == 1, pos_weight, 1.0)
    losses = weights * (softplus - y * logits)
    probs = np.exp(logits - softplus)  # stable sigmoid
    grad = weights * (probs - y) / y.shape[0]
    return losses.mean(), grad


def predict_logits(model, x, batch_size=4096):
    """Raw outputs (logits) for each row of x, in evaluation mode."""
    logits = [
        model.forward(x[i : i + batch_size], training=False)
        for i in range(0, x.shape[0], batch_size)
    ]
    return np.concatenate(logits)


def predict_proba(model, x):
    """Probability of the positive class for each row of x."""
    return 1.0 / (1.0 + np.exp(-predict_logits(model, x)))


def predict_labels(model, x, threshold=0.5):
    """Predictions in {-1, 1}, the format required by create_csv_submission."""
    return np.where(predict_proba(model, x) >= threshold, 1, -1)


def train(
    model,
    optimizer,
    x,
    y,
    epochs=10,
    batch_size=256,
    pos_weight=1.0,
    x_val=None,
    y_val=None,
    seed=42,
    verbose=True,
):
    """Train the model with mini-batches.

    Args:
        model: an MLP.
        optimizer: Adam or Muon built on model.parameters().
        x: shape=(N, D) training features (already preprocessed).
        y: shape=(N,) labels in {0, 1}.
        epochs: number of passes over the data.
        batch_size: mini-batch size (keep it > 1 because of BatchNorm).
        pos_weight: weight of the positive class in the loss.
        x_val, y_val: optional validation set, evaluated after every epoch.
        seed: seed for the mini-batch shuffling.
        verbose: print the losses after every epoch.

    Returns:
        history: dict with the per-epoch "train_loss" and "val_loss".
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(epochs):
        perm = rng.permutation(n)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            if idx.shape[0] < 2:
                continue  # BatchNorm needs at least two samples
            optimizer.zero_grad()
            logits = model.forward(x[idx], training=True)
            loss, grad = bce_with_logits(logits, y[idx], pos_weight)
            model.backward(grad)
            optimizer.step()
            total += loss * idx.shape[0]
        history["train_loss"].append(total / n)

        msg = "epoch {}/{}: train_loss={:.4f}".format(epoch + 1, epochs, total / n)
        if x_val is not None:
            val_loss, _ = bce_with_logits(predict_logits(model, x_val), y_val, pos_weight)
            history["val_loss"].append(val_loss)
            msg += ", val_loss={:.4f}".format(val_loss)
        if verbose:
            print(msg)
    return history
