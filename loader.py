"""
Data loading for the network: CSV -> preprocessing -> Dataset -> DataLoader.

"""

import os

import numpy as np

from helpers import load_csv_data
import src.brfss_codes as codes
import src.preprocessing as prep
import src.transforms as transforms

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


class Dataset:
    """Features with (optional) labels, indexable like an array.

    Args:
        x: shape=(N, D) features.
        y: shape=(N,) labels in {0, 1}, or None for the test set.
    """

    def __init__(self, x, y=None):
        if y is not None and x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of rows")
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


class DataLoader:
    """Iterates over a Dataset in mini-batches.

    Args:
        dataset: a Dataset.
        batch_size: number of rows per batch.
        shuffle: reshuffle the rows at the start of every epoch.
        drop_last: drop the final batch if it is smaller than batch_size.
        seed: seed of the shuffling.
    """

    def __init__(self, dataset, batch_size=256, shuffle=True, drop_last=False, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for b in range(len(self)):
            yield self.dataset[order[b * self.batch_size : (b + 1) * self.batch_size]]


def load_raw(data_path=DATA_PATH, cache=True):
    """Load the CSVs with load_csv_data, caching them in a .npz file."""
    cache_file = os.path.join(data_path, "raw_cache.npz")
    if cache and os.path.exists(cache_file):
        d = np.load(cache_file)
        return d["x_train"], d["x_test"], d["y_train"], d["train_ids"], d["test_ids"]
    if not os.path.exists(os.path.join(data_path, "x_train.csv")):
        raise FileNotFoundError(
            "no x_train.csv in {!r}: download the data or pass the right path".format(data_path)
        )
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(data_path)
    if cache:
        np.savez(
            cache_file,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            train_ids=train_ids,
            test_ids=test_ids,
        )
    return x_train, x_test, y_train, train_ids, test_ids


def load_datasets(
    data_path=DATA_PATH,
    val_fraction=0.1,
    max_missing_rate=0.5,
    missing_indicators=False,
    clean_codes=True,
    fix_units=False,
    log_skewed=False,
    codes_version=codes.CODES_VERSION,
    seed=42,
    sub_sample=False,
    cache=True,
    verbose=False,
):
    """Build the train / validation / test Datasets, ready for the network.

    Returns:
        train_ds, val_ds, test_ds: Dataset objects.
        test_ids: ids of the test rows, for create_csv_submission.
    """
    x_train, x_test, y_train, _, test_ids = load_raw(data_path, cache)
    if sub_sample:
        x_train, y_train = x_train[::50], y_train[::50]
    y01 = (y_train == 1).astype(float)

    if val_fraction > 0:
        rng = np.random.default_rng(seed)
        fit_idx, val_idx = prep.stratified_split(y01, val_fraction, rng)
    else:
        fit_idx, val_idx = np.arange(len(y01)), None

    if fix_units and not clean_codes:
        raise ValueError("fix_units needs clean_codes: the codes must be cleaned first")

    x_fit = x_train[fit_idx]
    code_state = None
    if clean_codes:
        names = codes.read_column_names(os.path.join(data_path, "x_train.csv"))
        code_state = codes.fit_code_cleaner(x_fit, names, fix_units, codes_version)
        if verbose:
            print("clean codes: " + codes.describe(code_state))

    def split_codes(x):
        """(numeric features, one-hot block) after the optional code cleaning."""
        if code_state is None:
            return x, None
        return codes.apply_code_cleaner(x, code_state)

    x_fit_num, _ = split_codes(x_fit)
    log_mask = None
    if log_skewed:
        log_mask = transforms.fit_log_columns(x_fit_num)
        x_fit_num = transforms.apply_log(x_fit_num, log_mask)
        if verbose:
            print("log(1+x) on {} long-tailed columns".format(int(log_mask.sum())))
    state = prep.fit_preprocessor(x_fit_num, eps=1e-12, max_missing_rate=max_missing_rate)
    mask = fit_missing_indicators(x_fit_num, state) if missing_indicators else None

    def prepare(x):
        x_num, onehot = split_codes(x)
        if log_mask is not None:
            x_num = transforms.apply_log(x_num, log_mask)
        x_clean = prep.transform(x_num, state)
        if mask is not None:
            x_clean = add_missing_indicators(x_num, x_clean, state, mask)
        if onehot is not None:
            x_clean = np.hstack([x_clean, onehot])
        return x_clean

    train_ds = Dataset(prepare(x_fit), y01[fit_idx])
    val_ds = None
    if val_idx is not None:
        val_ds = Dataset(prepare(x_train[val_idx]), y01[val_idx])
    test_ds = Dataset(prepare(x_test))
    return train_ds, val_ds, test_ds, test_ids

