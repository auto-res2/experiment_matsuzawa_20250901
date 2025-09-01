# src/preprocess.py
"""Data-loading & simple preprocessing helper.
Returns train/validation/test PyTorch dataloaders for CIFAR-10.
The dataset is automatically downloaded to ``data/`` if necessary.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_root: Path | str = "data",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Download CIFAR-10 (if missing) and return train/val/test loaders."""

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])
    trainset_full = CIFAR10(root=data_root, train=True, download=True, transform=transform)
    testset = CIFAR10(root=data_root, train=False, download=True, transform=transform)

    # hold out 5k examples for validation
    train_size = len(trainset_full) - 5000
    val_size = 5000
    trainset, valset = random_split(trainset_full, [train_size, val_size])

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
