"""src/preprocess.py
--------------------
Download and prepare the MNIST dataset used by the experiment.  The function
`prepare_datasets` returns the training- and test-sets so that the rest of the
code does not need to touch the on-disk representation.

All data are stored inside the project in the directory `data/` as required by
the specification.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple

from torchvision.datasets import MNIST
from torchvision import transforms


def prepare_datasets(data_root: Path) -> Tuple[MNIST, MNIST]:
    """Download (if necessary) and return (train_ds, test_ds)."""
    data_root.mkdir(exist_ok=True)
    tfm = transforms.ToTensor()
    train_ds = MNIST(data_root, download=True, train=True, transform=tfm)
    test_ds = MNIST(data_root, download=True, train=False, transform=tfm)
    return train_ds, test_ds
