"""
preprocess.py – Synthetic data generation & preprocessing utilities
Generates a simple 1-D regression dataset: y = 2x + ε.
The function returns PyTorch tensors already split into train/val/test.
"""
from __future__ import annotations

import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

# -----------------------------------------------------------------------------
# Re-usable utility – falls back to local definition if src.utils is unavailable
# -----------------------------------------------------------------------------
try:
    from .utils import set_seed  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – local fallback for robustness

    def set_seed(seed: int | None = None) -> None:  # noqa: D401
        """Set random seeds for reproducibility (torch / numpy / python)."""
        if seed is None:
            return
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
