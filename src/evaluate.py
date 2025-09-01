"""src/evaluate.py
Evaluation helper – computes accuracy on the held-out **test** set.  The test
loader is built by `src.preprocess.make_dataloaders` so we re-use that helper
here to avoid code duplication.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader

from .preprocess import make_dataloaders


def evaluate(model: nn.Module, config: Dict) -> float:
    """Return classification accuracy on the test set."""
    device = next(model.parameters()).device
    loaders = make_dataloaders(config)
    test_loader: DataLoader = loaders["test"]

    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu()
            correct += (preds == yb).sum().item()
    acc = correct / len(test_loader.dataset)
    return acc
