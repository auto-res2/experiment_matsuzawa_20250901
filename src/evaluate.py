"""src/evaluate.py
Evaluation utilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import torch
from torch.utils.data import DataLoader

from .train import SimpleCNN


def _plot_confusion_matrix(cm: np.ndarray, classes: list[str], pdf_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)


def evaluate(model: torch.nn.Module, data_loader: DataLoader, save_dir: Path) -> float:
    """Return classification accuracy (percentage) and save a confusion matrix."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu()
            y_true.append(y)
            y_pred.append(preds)
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    acc = 100.0 * (y_true == y_pred).mean()

    cm = confusion_matrix(y_true, y_pred)
    class_names = [str(l) for l in unique_labels(y_true, y_pred)]
    save_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion_matrix(cm, class_names, save_dir / "confusion_matrix.pdf")
    return acc
