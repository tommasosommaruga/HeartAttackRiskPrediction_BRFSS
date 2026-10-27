"""
Data loading for the models: preprocessed arrays -> Dataset -> DataLoader.

The preprocessing lives in src/preprocessing.py (get_data).
"""

import numpy as np

import src.preprocessing as prep

DATA_PATH = prep.DATA_DIR


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
        seed: seed of the shuffling.
    """

    def __init__(self, dataset, batch_size=256, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.dataset)
        order = self.rng.permutation(n) if self.shuffle else np.arange(n)
        for b in range(len(self)):
            yield self.dataset[order[b * self.batch_size : (b + 1) * self.batch_size]]


def load_datasets(data_path=DATA_PATH, val_fraction=0.1, seed=42, sub_sample=False, cfg=None):
    """Train / validation / test Datasets from prep.get_data.

    Returns:
        train_ds, val_ds (None if val_fraction=0), test_ds: Dataset objects.
        test_ids: ids of the test rows, for create_csv_submission.
    """
    x_tr, y_tr, x_va, y_va, x_te, test_ids = prep.get_data(
        data_path, cfg, val_fraction, seed, sub_sample
    )
    val_ds = Dataset(x_va, y_va) if x_va is not None else None
    return Dataset(x_tr, y_tr), val_ds, Dataset(x_te), test_ids
