
"""
Gradient-boosted decision trees for binary classification, in pure NumPy.


Run from the project root (same data options as train.py):

    python gbdt.py
    python gbdt.py --max-depth 6 --lr 0.05 --n-trees 1000 --max-missing-rate 1.0 --missing-indicators
"""

import argparse
import time

import numpy as np

from helpers import create_csv_submission
from loader import DATA_PATH, load_datasets
from util import THRESHOLD_METRICS, best_threshold, classification_metrics




def fit_bins(x, max_bins=64):
    """Quantile bin edges of each column, computed on the training matrix."""
    qs = np.linspace(0, 1, max_bins + 1)[1:-1]
    return [np.unique(np.quantile(x[:, j], qs)) for j in range(x.shape[1])]


def apply_bins(x, edges):
    """Bin index of every value, transposed to (D, N) uint8 for fast row access."""
    out = np.empty((x.shape[1], x.shape[0]), dtype=np.uint8)
    for j, e in enumerate(edges):
        out[j] = np.searchsorted(e, x[:, j], side="right")
    return out




class Tree:
    """Binary tree stored in flat arrays; leaves have feature == -1."""

    def __init__(self):
        self.feature, self.bin, self.left, self.right, self.value = [], [], [], [], []

    def add_node(self):
        for a in (self.feature, self.bin, self.left, self.right, self.value):
            a.append(-1 if a is not self.value else 0.0)
        return len(self.value) - 1

    def finalize(self):
        self.feature = np.array(self.feature)
        self.bin = np.array(self.bin)
        self.left = np.array(self.left)
        self.right = np.array(self.right)
        self.value = np.array(self.value)
        self.depth = 0
        stack = [(0, 0)]
        while stack:
            node, d = stack.pop()
            self.depth = max(self.depth, d)
            if self.feature[node] >= 0:
                stack += [(self.left[node], d + 1), (self.right[node], d + 1)]

    def predict(self, xb):
        """Leaf value for each column of xb (D, N) of bin indices."""
        n = xb.shape[1]
        node = np.zeros(n, dtype=np.int64)
        cols = np.arange(n)
        for _ in range(self.depth):
            feat = self.feature[node]
            inner = feat >= 0
            if not inner.any():
                break
            go_left = xb[np.maximum(feat, 0), cols] <= self.bin[node]
            node = np.where(inner, np.where(go_left, self.left[node], self.right[node]), node)
        return self.value[node]


def histogram(xb_rows, g, h, n_bins):
    """Per-feature, per-bin sums of g and h. xb_rows is (D, n). Returns (D, n_bins, 2)."""
    hist = np.empty((xb_rows.shape[0], n_bins, 2))
    for j in range(xb_rows.shape[0]):
        hist[j, :, 0] = np.bincount(xb_rows[j], weights=g, minlength=n_bins)
        hist[j, :, 1] = np.bincount(xb_rows[j], weights=h, minlength=n_bins)
    return hist


def best_split(hist, features, reg_lambda, min_child_weight):
    """Best (gain, feature, bin) over the allowed features; split is bin <= b goes left."""
    G = hist[features, :, 0]
    H = hist[features, :, 1]
    g_tot, h_tot = G[0].sum(), H[0].sum()
    GL, HL = np.cumsum(G, axis=1)[:, :-1], np.cumsum(H, axis=1)[:, :-1]
    GR, HR = g_tot - GL, h_tot - HL
    gain = GL**2 / (HL + reg_lambda) + GR**2 / (HR + reg_lambda) - g_tot**2 / (h_tot + reg_lambda)
    gain[(HL < min_child_weight) | (HR < min_child_weight)] = -np.inf
    k = np.argmax(gain)
    f, b = np.unravel_index(k, gain.shape)
    return 0.5 * gain[f, b], features[f], b


def build_tree(xb, g, h, rows, features, p, n_bins):
    """Grow one tree depth-first on the sampled rows and features."""
    tree = Tree()
    root = tree.add_node()
    stack = [(root, rows, histogram(xb[:, rows], g[rows], h[rows], n_bins), 0)]
    while stack:
        node, idx, hist, depth = stack.pop()
        g_sum, h_sum = hist[0, :, 0].sum(), hist[0, :, 1].sum()
        tree.value[node] = -g_sum / (h_sum + p.reg_lambda)
        if depth >= p.max_depth or h_sum < 2 * p.min_child_weight:
            continue
        gain, f, b = best_split(hist, features, p.reg_lambda, p.min_child_weight)
        if not gain > p.gamma:
            continue
        go_left = xb[f, idx] <= b
        left_idx, right_idx = idx[go_left], idx[~go_left]
        # histogram of the smaller child only; the other one is parent - smaller
        small, large = (left_idx, right_idx) if len(left_idx) <= len(right_idx) else (right_idx, left_idx)
        hist_small = histogram(xb[:, small], g[small], h[small], n_bins)
        hist_large = hist - hist_small
        left, right = tree.add_node(), tree.add_node()
        tree.feature[node], tree.bin[node] = f, b
        tree.left[node], tree.right[node] = left, right
        hist_left, hist_right = (hist_small, hist_large) if small is left_idx else (hist_large, hist_small)
        stack += [(left, left_idx, hist_left, depth + 1), (right, right_idx, hist_right, depth + 1)]
    tree.finalize()
    return tree





def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def weighted_logloss(z, y, pos_weight):
    w = np.where(y == 1, pos_weight, 1.0)
    return np.mean(w * (np.logaddexp(0.0, z) - y * z))


class GBDT:
    """Boosted trees on the logistic loss. Call fit, then predict_proba."""

    def __init__(self, p):
        self.p = p

    def fit(self, x, y, x_val=None, y_val=None, verbose=True):
        p = self.p
        rng = np.random.default_rng(p.seed)
        self.edges = fit_bins(x, p.max_bins)
        n_bins = max(len(e) for e in self.edges) + 1
        xb = apply_bins(x, self.edges)
        xb_val = apply_bins(x_val, self.edges) if x_val is not None else None
        w = np.where(y == 1, p.pos_weight, 1.0)

        # start from the weighted log-odds of the positive class
        rate = np.sum(w * y) / np.sum(w)
        self.base = np.log(rate / (1 - rate))
        self.trees = []
        z = np.full(len(y), self.base)
        z_val = np.full(len(y_val), self.base) if x_val is not None else None
        best_loss, best_round, history = np.inf, 0, []
        n_feat = max(1, int(round(p.colsample * x.shape[1])))
        n_rows = max(2, int(round(p.subsample * len(y))))

        for t in range(p.n_trees):
            start = time.time()
            prob = sigmoid(z)
            g, h = w * (prob - y), w * prob * (1 - prob)
            rows = np.sort(rng.choice(len(y), n_rows, replace=False)) if p.subsample < 1 else np.arange(len(y))
            features = np.sort(rng.choice(x.shape[1], n_feat, replace=False))
            tree = build_tree(xb, g, h, rows, features, p, n_bins)
            tree.value = tree.value * p.lr
            self.trees.append(tree)
            z += tree.predict(xb)

            msg = "tree {}/{}: train_loss={:.4f}".format(t + 1, p.n_trees, weighted_logloss(z, y, p.pos_weight))
            if x_val is not None:
                z_val += tree.predict(xb_val)
                val_loss = weighted_logloss(z_val, y_val, p.pos_weight)
                history.append(val_loss)
                msg += ", val_loss={:.4f}".format(val_loss)
                if val_loss < best_loss - 1e-6:
                    best_loss, best_round = val_loss, t + 1
                    msg += " *best*"
            if verbose and ((t + 1) % p.log_every == 0 or t == 0):
                print(msg + " ({} nodes, {:.1f}s)".format(len(tree.value), time.time() - start))
            if x_val is not None and t + 1 - best_round >= p.patience:
                if verbose:
                    print("early stopping: no improvement for {} trees".format(p.patience))
                break

        if x_val is not None:
            self.trees = self.trees[:best_round]
            if verbose:
                print("best number of trees: {} (val_loss={:.4f})".format(best_round, best_loss))
        return history

    def predict_logits(self, x):
        xb = apply_bins(x, self.edges)
        z = np.full(x.shape[0], self.base)
        for tree in self.trees:
            z += tree.predict(xb)
        return z

    def predict_proba(self, x):
        return sigmoid(self.predict_logits(x))




def parse_args():
    p = argparse.ArgumentParser(description="Gradient-boosted trees on the project data.")
    # boosting
    p.add_argument("--n-trees", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.1, help="shrinkage of every tree")
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--min-child-weight", type=float, default=5.0,
                   help="minimum hessian sum in a leaf (larger = more conservative)")
    p.add_argument("--reg-lambda", type=float, default=1.0, help="L2 penalty on the leaf values")
    p.add_argument("--gamma", type=float, default=0.0, help="minimum gain to make a split")
    p.add_argument("--subsample", type=float, default=0.8, help="fraction of rows per tree")
    p.add_argument("--colsample", type=float, default=0.8, help="fraction of features per tree")
    p.add_argument("--max-bins", type=int, default=64)
    p.add_argument("--pos-weight", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=30, help="trees without a better val loss")
    p.add_argument("--log-every", type=int, default=10)
    # data (same as train.py)
    p.add_argument("--data", default=DATA_PATH)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--max-missing-rate", type=float, default=0.5)
    p.add_argument("--missing-indicators", action="store_true")
    p.add_argument("--no-clean-codes", dest="clean_codes", action="store_false")
    p.add_argument("--fix-units", action="store_true")
    p.add_argument("--log-skewed", action="store_true")
    p.add_argument("--sub-sample", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    # prediction
    p.add_argument("--threshold", default="best_f1", help="best_f1 / best_bal_acc or a number")
    p.add_argument("--submission", default="submissions/submission_gbdt.csv")
    args = p.parse_args()
    if args.max_bins > 255:
        p.error("--max-bins must be <= 255 (bins are stored as uint8)")
    return args


def main():
    args = parse_args()
    train_ds, val_ds, test_ds, test_ids = load_datasets(
        args.data,
        val_fraction=args.val_fraction,
        max_missing_rate=args.max_missing_rate,
        missing_indicators=args.missing_indicators,
        clean_codes=args.clean_codes,
        fix_units=args.fix_units,
        log_skewed=args.log_skewed,
        seed=args.seed,
        sub_sample=args.sub_sample,
        verbose=True,
    )
    print("train: {}, val: {}, test: {}, features: {}".format(
        len(train_ds), len(val_ds) if val_ds else 0, len(test_ds), train_ds.x.shape[1]
    ))

    start = time.time()
    model = GBDT(args)
    model.fit(train_ds.x, train_ds.y,
              val_ds.x if val_ds else None, val_ds.y if val_ds else None)
    print("training time: {:.0f}s".format(time.time() - start))

    if args.threshold in THRESHOLD_METRICS:
        if val_ds is None:
            raise ValueError("no validation split: pass a numeric --threshold")
        threshold, _ = best_threshold(model.predict_proba(val_ds.x), val_ds.y, THRESHOLD_METRICS[args.threshold])
    else:
        threshold = float(args.threshold)
    if val_ds is not None:
        m = classification_metrics(model.predict_proba(val_ds.x), val_ds.y, threshold)
        print("validation @ {}: ".format(threshold) + ", ".join("{}={:.4f}".format(k, v) for k, v in m.items()))

    y_pred = np.where(model.predict_proba(test_ds.x) >= threshold, 1, -1)
    create_csv_submission(test_ids, y_pred, args.submission)
    print("test: {} rows, {:.1%} predicted positive -> {}".format(
        len(y_pred), np.mean(y_pred == 1), args.submission
    ))


if __name__ == "__main__":
    main()
