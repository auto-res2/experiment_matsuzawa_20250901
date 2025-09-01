"""src/preprocess.py
For MNIST nothing more than downloading is required, but this file keeps the
pre-processing logic separate so that it mirrors a realistic project layout.
It can be extended with dataset-specific cleaning / augmentation steps later.
"""
from __future__ import annotations

from pathlib import Path

from torchvision.datasets import MNIST
from torchvision import transforms as T

DATA_DIR = Path("data")


def preprocess_data() -> None:
    """Download / cache the MNIST dataset under `data/`."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tf = T.Compose([T.ToTensor()])
    # The following calls are idempotent – torchvision skips re-download.
    MNIST(root=str(DATA_DIR), train=True, download=True, transform=tf)
    MNIST(root=str(DATA_DIR), train=False, download=True, transform=tf)
    print("✅  MNIST dataset is ready (stored in ./data)")


if __name__ == "__main__":
    preprocess_data()