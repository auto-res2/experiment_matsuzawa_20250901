"""
preprocess.py – Synthetic data generation & preprocessing utilities
Generates a simple 1-D regression dataset: y = 2x + ε.
The function returns PyTorch tensors already split into train/val/test.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from .utils import set_seed


def make_dataset(cfg: Dict) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    """Generate and return (train, val, test) datasets as tensor tuples."""
    set_seed(cfg.get("seed", 0))

    n_total: int = cfg["n_total"]
    x = np.random.uniform(-1.0, 1.0, size=(n_total, 1)).astype(np.float32)
    noise = np.random.normal(0.0, cfg["noise_std"], size=(n_total, 1)).astype(np.float32)
    y = 2.0 * x + noise

    # shuffle and split
    idx = np.random.permutation(n_total)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    train_idx, val_idx, test_idx = idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]

    to_tensor = lambda arr: torch.from_numpy(arr)  # noqa: E731
    train_xy = (to_tensor(x[train_idx]), to_tensor(y[train_idx]))
    val_xy = (to_tensor(x[val_idx]), to_tensor(y[val_idx]))
    test_xy = (to_tensor(x[test_idx]), to_tensor(y[test_idx]))

    return train_xy, val_xy, test_xy
