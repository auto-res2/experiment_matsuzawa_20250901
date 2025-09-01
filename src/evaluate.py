# src/evaluate.py
"""Evaluation utilities.
Loads a checkpoint produced by ``src.train`` and evaluates it on the test
split of CIFAR-10.  Also saves a confusion-matrix heat-map PDF suitable for
inclusion in academic papers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .train import SimpleCNN  # re-use architecture


@torch.inference_mode()
def evaluate_model(
    ckpt_path: Path | str,
    test_loader: DataLoader,
    device: str | torch.device = "cuda",
) -> Tuple[float, np.ndarray]:
    """Return (accuracy, confusion_matrix)."""
    device = torch.device(device)
    model = SimpleCNN().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    preds_all, targets_all = [], []
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1).cpu().numpy()
        preds_all.append(preds)
        targets_all.append(labels.numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    acc = (preds_all == targets_all).mean()
    cm = confusion_matrix(targets_all, preds_all)

    _plot_confusion_matrix(cm, test_loader.dataset.classes)
    return acc, cm


# -------------------------------------------------------------
# Helper – plot confusion matrix
# -------------------------------------------------------------

def _plot_confusion_matrix(cm: np.ndarray, class_names):
    out_dir = Path(".research/iteration1/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.title("CIFAR-10 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.pdf", format="pdf")
    plt.close()
