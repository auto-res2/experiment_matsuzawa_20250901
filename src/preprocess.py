"""src/preprocess.py
Synthetic data generation & DataLoader helpers.  We purposefully *do not* rely
on scikit-learn so that the entire project remains lightweight; everything is
implemented with NumPy and PyTorch only.

A binary classification problem is generated as follows:
  • Class-0 samples are drawn from N([-1, -1],  I)
  • Class-1 samples are drawn from N([+1, +1],  I)
  • A configurable amount of label-noise can be injected via *config["data"]["flip_prob"]*

The routine returns PyTorch DataLoaders for the train/val/test splits.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

# The DataLoader objects are (re-)generated every call – this is acceptable as
# the dataset is tiny (<1 MB) and avoids serialisation overhead.


def _generate_dataset(cfg: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic binary classification data according to *cfg*"""
    n_total: int = cfg["data"]["n_samples"]
    dim: int = cfg["data"]["dim"]
    noise: float = cfg["data"].get("flip_prob", 0.0)

    n_half = n_total // 2
    mean0, mean1 = -np.ones(dim), np.ones(dim)
    X0 = np.random.randn(n_half, dim) + mean0
    X1 = np.random.randn(n_total - n_half, dim) + mean1
    y0 = np.zeros(n_half, dtype=np.int64)
    y1 = np.ones(n_total - n_half, dtype=np.int64)

    X = np.vstack([X0, X1]).astype(np.float32)
    y = np.concatenate([y0, y1])

    # Label noise (optional)
    if noise > 0:
        mask = np.random.rand(n_total) < noise
        y[mask] = 1 - y[mask]

    # shuffle
    idx = np.random.permutation(n_total)
    X, y = X[idx], y[idx]

    return torch.from_numpy(X), torch.from_numpy(y)


def make_dataloaders(cfg: Mapping) -> Dict[str, DataLoader]:
    """Return dict with keys train/val/test."""
    batch_size: int = cfg["optim"]["batch_size"]
    val_frac: float = cfg["data"]["val_split"]
    test_frac: float = cfg["data"]["test_split"]

    X, y = _generate_dataset(cfg)
    dataset = TensorDataset(X, y)

    n_total = len(dataset)
    n_val = int(val_frac * n_total)
    n_test = int(test_frac * n_total)
    n_train = n_total - n_val - n_test

    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    loader_kwargs = dict(batch_size=batch_size, shuffle=True, drop_last=False)
    return {
        "train": DataLoader(train_ds, **loader_kwargs),
        "val":   DataLoader(val_ds,   **loader_kwargs),
        "test":  DataLoader(test_ds,  **loader_kwargs),
    }
