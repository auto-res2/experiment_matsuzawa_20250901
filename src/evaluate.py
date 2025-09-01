"""
evaluate.py
~~~~~~~~~~~
Evaluate a trained MNIST model and create a confusion-matrix PDF saved to
``.research/iteration1/images/confusion_matrix.pdf``.
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .train import SmallCNN, PROJECT_ROOT, MODEL_PATH  # relative import

RESEARCH_IMG_DIR = PROJECT_ROOT / ".research" / "iteration1" / "images"
RESEARCH_IMG_DIR.mkdir(parents=True, exist_ok=True)


def _load_test_loader(batch_size: int = 256) -> DataLoader:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = datasets.MNIST(root=PROJECT_ROOT / "data", train=False, download=True, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def evaluate_model(device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu") -> dict[str, float]:
    test_dl = _load_test_loader()
    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for imgs, labels in test_dl:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            preds = logits.argmax(1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    cm = confusion_matrix(y_true, y_pred)
    acc = cm.trace() / cm.sum()
    _plot_confusion_matrix(cm)
    print(f"Test accuracy: {acc:.4f}")
    return {"accuracy": acc}


def _plot_confusion_matrix(cm: Sequence[Sequence[int]]):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("MNIST Confusion Matrix")
    plt.tight_layout()
    out_path = RESEARCH_IMG_DIR / "confusion_matrix.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion-matrix figure to {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    evaluate_model()
