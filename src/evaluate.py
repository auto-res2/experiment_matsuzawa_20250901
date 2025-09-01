"""src/evaluate.py
Evaluate the trained model on the held-out test set and produce a confusion
matrix figure (PDF).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .preprocess import maybe_prepare_data, TARGET_NAMES
from .train import IrisNet, MODELS_DIR, IMAGES_DIR


# ----------------------------------------------------------------------------

def evaluate(model_path: Path | None = None) -> Tuple[float, Path]:
    """Return (accuracy, confusion_matrix_path)."""
    # ensure data present
    _, test_npz = maybe_prepare_data()
    X_test = torch.tensor(test_npz["x"], dtype=torch.float32)
    y_test = torch.tensor(test_npz["y"], dtype=torch.long)

    if model_path is None:
        model_path = MODELS_DIR / "iris_net.pt"
    checkpoint = torch.load(model_path, map_location="cpu")

    # Recreate network with the same hidden dimension that was used for training
    hidden_dim: int = checkpoint.get("cfg", {}).get("hidden_dim", 16)
    model = IrisNet(hidden_dim=hidden_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        preds = model(X_test).argmax(dim=1).numpy()
        true = y_test.numpy()
        acc = (preds == true).mean()

    cm = confusion_matrix(true, preds, labels=range(len(TARGET_NAMES)))

    # plot confusion matrix
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=TARGET_NAMES,
        yticklabels=TARGET_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = IMAGES_DIR / "confusion_matrix.pdf"
    plt.savefig(cm_path, bbox_inches="tight")

    print(f"[evaluate] accuracy={acc * 100:.2f}%  confusion-matrix saved → {cm_path}")
    return acc, cm_path
