"""src/preprocess.py
Very small data-generation & preprocessing helper so that the experiment is
completely self-contained.  We synthesise a 20-dimensional binary
classification dataset with scikit-learn and then store it as PyTorch tensors
in RAM (no filesystem footprint necessary).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from sklearn.datasets import make_classification

# -----------------------------------------------------------------------------
#  Public helper – called by train.py / evaluate.py
# -----------------------------------------------------------------------------

def load_preprocessed_data(cfg: dict) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Generate (or regenerate) a deterministic synthetic dataset.

    The function keeps the seed fixed so that different stages of the pipeline
    see exactly the same data splits.
    """
    rng = np.random.default_rng(cfg.get("data_seed", 0))

    x, y = make_classification(
        n_samples=cfg["dataset_size"],
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=rng.integers(0, 10_000),
    )

    # Normalise to zero-mean, unit-variance (common practice)
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)

    # Train/val split ----------------------------------------------------------
    n_train = int(cfg["train_split"] * len(x))
    x_train, y_train = x[:n_train], y[:n_train]
    x_val,   y_val   = x[n_train:], y[n_train:]

    # Cast to torch tensors ----------------------------------------------------
    x_train = torch.as_tensor(x_train, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.long)
    x_val   = torch.as_tensor(x_val,   dtype=torch.float32)
    y_val   = torch.as_tensor(y_val,   dtype=torch.long)

    return (x_train, y_train), (x_val, y_val)
