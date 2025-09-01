"""
evaluate.py – evaluation utilities used by src.main.  Currently provides a
single *evaluate_model* helper that computes overall accuracy on the test
set and returns both the numeric accuracy and per-sample predictions.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["evaluate_model"]


def evaluate_model(
    model: nn.Module,
    test_ds: TensorDataset,
    batch_size: int = 64,
    device: torch.device | str | None = None,
) -> Tuple[float, np.ndarray]:
    """Compute accuracy of *model* on *test_ds*.

    Returns
    -------
    acc : float – classification accuracy in the range [0, 1].
    preds : np.ndarray – raw integer predictions for every sample.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    all_preds = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            all_preds.append(pred.cpu().numpy())
            correct += (pred == yb).sum().item()
            total += yb.size(0)

    acc = correct / total
    return acc, np.concatenate(all_preds)
