"""
main.py – entry-point that is executed via ``python -m src.main`` from the
project root.  The script wires together the three internal modules
(preprocess, train, evaluate), prints a succinct experiment description
to stdout, and stores publication-ready PDF figures under
``.research/iteration14/images`` as mandated by the Instructions.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")  # head-less environments (e.g., CI, SSH)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import TensorDataset

from .preprocess import preprocess
from .train import train_model
from .evaluate import evaluate_model

# ----------------------------------------------------------------------------
#  Directories used throughout the experiment
# ----------------------------------------------------------------------------
_PLOT_DIR = Path(".research/iteration14/images")
_PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
#  Helper to create TensorDataset objects from numpy arrays
# ----------------------------------------------------------------------------

def _to_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:  # type: ignore[name-defined]
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))  # type: ignore[attr-defined]


# ----------------------------------------------------------------------------
#  MAIN orchestrator
# ----------------------------------------------------------------------------

def main() -> None:
    # 1) ---------------------------------------------------------------------
    print("====================  IRIS – SimpleNet Benchmark  ====================\n")
    print("Task      : Iris flower classification (3-class).")
    print("Model     : 1 hidden-layer MLP (16 units, ReLU).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device    : {device}\n")

    # 2) ---------------------------------------------------------------------
    print("[Stage] Pre-processing …")
    x_tr, y_tr, x_val, y_val, x_te, y_te = preprocess()
    train_ds, val_ds, test_ds = _to_dataset(x_tr, y_tr), _to_dataset(x_val, y_val), _to_dataset(x_te, y_te)
    print(f"          train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)} samples")

    # 3) ---------------------------------------------------------------------
    print("[Stage] Training …")
    model, tr_curve, val_curve = train_model(train_ds, val_ds, epochs=120, lr=1e-2, batch_size=32, seed=7, device=device)

    # 4) ---------------------------------------------------------------------
    print("[Stage] Evaluation …")
    acc, _ = evaluate_model(model, test_ds, batch_size=64, device=device)
    print(f"Test accuracy: {acc * 100:.2f} %\n")

    # 5) ---------------------------------------------------------------------
    print("[Stage] Plotting …")
    sns.set_theme(style="whitegrid")

    # Loss curves ------------------------------------------------------------
    plt.figure(figsize=(6, 3))
    plt.plot(tr_curve, label="train")
    plt.plot(val_curve, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.tight_layout()
    fig_loss = _PLOT_DIR / "training_curves.pdf"
    plt.savefig(fig_loss, bbox_inches="tight")

    # Accuracy bar -----------------------------------------------------------
    plt.figure(figsize=(2.5, 3))
    sns.barplot(x=["SimpleNet"], y=[acc * 100], palette="muted")
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    for p in plt.gca().patches:
        h = p.get_height()
        plt.gca().annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2, h), ha="center", va="bottom")
    plt.tight_layout()
    fig_acc = _PLOT_DIR / "test_accuracy.pdf"
    plt.savefig(fig_acc, bbox_inches="tight")

    print("Figures saved →", fig_loss, fig_acc)
    print("\n✔  Experiment finished successfully.")


# ----------------------------------------------------------------------------
#  CLI entry-point
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
