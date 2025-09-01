"""
evaluate.py – model evaluation & visualisation
Loads the trained model and prints the test accuracy as well as a confusion
matrix that is stored as a PDF suitable for inclusion in papers.
"""
from __future__ import annotations

from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from .train import SimpleMLP

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
IMG_DIR = ROOT / ".research" / "iteration11" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(model_path: Path | None = None) -> None:
    if model_path is None:
        model_path = MODEL_DIR / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError("Trained model .pt file not found. Run training first.")

    data = torch.load(DATA_DIR / "dataset.pt")
    x_test = data["x_test"]
    y_test = data["y_test"]

    model = SimpleMLP(in_dim=x_test.shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(x_test)
        preds = logits.argmax(dim=1)

    cm = confusion_matrix(y_test.numpy(), preds.numpy())
    report = classification_report(y_test.numpy(), preds.numpy(), digits=3)

    print("\n==== TEST RESULTS ====")
    print(report)

    # ---- figure ----
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    fig_path = IMG_DIR / "confusion_matrix.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    print(f"Confusion matrix saved → {fig_path.relative_to(ROOT)}")
