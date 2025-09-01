"""src/preprocess.py
Dataset downloading / preprocessing utilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def make_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/val/test *DataLoader*s according to *cfg*.

    The loaders are cached under ./data automatically by *torchvision*.
    """
    batch_size = cfg.get("batch_size", 128)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),  # gray → RGB for CNN
        ]
    )

    root = Path("data")
    ds_full = datasets.MNIST(root, download=True, train=True, transform=transform)
    ds_test = datasets.MNIST(root, download=True, train=False, transform=transform)

    n_train = int(0.9 * len(ds_full))
    n_val = len(ds_full) - n_train
    ds_train, ds_val = random_split(ds_full, [n_train, n_val])

    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2)
    loader_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)

    return loader_train, loader_val, loader_test
