"""src/evaluate.py
-------------------
Functions for evaluating a trained model.  Currently only top-1 accuracy on the
MNIST test-set is implemented because that is sufficient to illustrate the
pipeline.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from .train import MLP, DEVICE


def evaluate(model: nn.Module, batch_size: int = 256) -> float:
    """Return classification accuracy on the MNIST test-set."""
    test_ds = MNIST(Path("data"), download=False, train=False,
                    transform=transforms.ToTensor())
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    total, correct = 0, 0
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb).argmax(dim=1)
            total += yb.size(0)
            correct += (preds == yb).sum().item()
    return correct / total
