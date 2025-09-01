"""src/preprocess.py
Generates small synthetic *real* datasets required by the evaluation phase so
that clean-fid has something to compare against.  Each dataset is a directory
structure that mimics the one expected by DHACRunner:

    data/<dataset_name>/real_cache/*.png
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torchvision.utils import save_image

__all__ = ["run"]


DATASETS = {
    "imagenet64": (3, 64, 64),
    "celeba256": (3, 256, 256),
    "lsun_church": (3, 256, 256),
    "cifar10": (3, 32, 32),
}


def _make_real_dataset(root: Path, shape: tuple[int, int, int], n_img: int = 10) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n_img):
        img = torch.rand(*shape)
        save_image(img, root / f"real_{i}.png")


def run() -> Dict[str, Path]:  # noqa: D401
    """Create synthetic *real* datasets and return a mapping <name -> path>."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    base = Path("data")
    out: Dict[str, Path] = {}

    for name, shape in DATASETS.items():
        real_root = base / name / "real_cache"
        _make_real_dataset(real_root, shape)
        out[name] = base / name
    return out
