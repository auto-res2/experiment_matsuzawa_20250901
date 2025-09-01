"""
preprocess.py
~~~~~~~~~~~~~
For the MNIST experiment there is no heavy preprocessing step besides
downloading the dataset.  However, this module keeps the project layout
consistent with the *Instructions* and would be a good place to add
future data-manipulation logic (e.g., sub-sampling, augmentations, or
feature extraction).

The current function simply ensures that the dataset is present on disk
so that the training/evaluation routines do not need to trigger the
network download twice.
"""
from __future__ import annotations

from pathlib import Path
import torchvision
from torchvision import datasets, transforms

from .train import PROJECT_ROOT


def prepare_datasets() -> None:
    """Ensure that MNIST is downloaded and ready on disk."""
    data_root = PROJECT_ROOT / "data"
    data_root.mkdir(exist_ok=True, parents=True)
    tfm = transforms.ToTensor()
    for split in (True, False):
        datasets.MNIST(root=data_root, train=split, download=True, transform=tfm)


if __name__ == "__main__":
    prepare_datasets()
    print("Datasets prepared ✔")
