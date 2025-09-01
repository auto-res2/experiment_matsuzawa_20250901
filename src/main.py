"""src/main.py
Entry point for the mini-experiment.  Execute with
    python -m src.main
from the project root as required in the instructions.

The pipeline performs the following steps:
    1. Read YAML hyper-parameters (config/config.yaml).
    2. Pre-process or generate data (handled by src.preprocess).
    3. Train a simple logistic-regression model (src.train).
    4. Evaluate the model (src.evaluate).
    5. Save a high-quality PDF figure that visualises the training data and
       decision boundary to .research/iteration12/images.

All console outputs include enough detail so that a reviewer can reproduce the
results without digging into intermediate files.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import yaml

from .train import train_model
from .evaluate import evaluate_model
from .preprocess import load_preprocessed_data

# ---------------------------------------------------------------------
#  Fixed project-root level paths
# ---------------------------------------------------------------------
ROOT_DIR        = Path(__file__).resolve().parent.parent
CONFIG_PATH     = ROOT_DIR / "config" / "config.yaml"
MODEL_DIR       = ROOT_DIR / "models"
IMG_DIR         = ROOT_DIR / ".research" / "iteration12" / "images"

for _d in (MODEL_DIR, IMG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
#  Helper – plot training samples & decision boundary
# ---------------------------------------------------------------------

def _plot_2d_projection(model: torch.nn.Module, cfg: dict):
    """Project the 20-dimensional problem down to 2 PCA components for visuals."""
    from sklearn.decomposition import PCA  # local import to keep requirements small
    (x_train, y_train), _ = load_preprocessed_data(cfg)

    # Fit PCA on training set and transform both train & grid
    pca   = PCA(n_components=2)
    x_2d  = pca.fit_transform(x_train.numpy())

    # Build a grid over principal-component space
    x_min, x_max = x_2d[:, 0].min() - 1.0, x_2d[:, 0].max() + 1.0
    y_min, y_max = x_2d[:, 1].min() - 1.0, x_2d[:, 1].max() + 1.0
    xx, yy       = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_flat    = np.c_[xx.ravel(), yy.ravel()]  # (N, 2)

    # Map the grid back to 20-D input space via inverse PCA approx
    grid_20d = torch.as_tensor(pca.inverse_transform(grid_flat), dtype=torch.float32)
    with torch.no_grad():
        zz = model(grid_20d).softmax(dim=1)[:, 1].view(xx.shape).numpy()

    # ------------------------------------------------------------------
    plt.figure(figsize=(5, 4))
    cs = plt.contourf(xx, yy, zz, cmap="RdBu", alpha=0.6, levels=20)
    plt.colorbar(cs, fraction=0.046, pad=0.04)
    # Overlay the training points
    plt.scatter(x_2d[y_train == 0, 0], x_2d[y_train == 0, 1], s=12, c="black", label="class 0")
    plt.scatter(x_2d[y_train == 1, 0], x_2d[y_train == 1, 1], s=12, c="white", edgecolors="black", label="class 1")
    plt.legend(frameon=False, fontsize=8)
    plt.title("Decision boundary (PCA projection)")
    plt.tight_layout()

    fig_path = IMG_DIR / "decision_boundary.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"[main] figure saved → {fig_path}")


# ---------------------------------------------------------------------
#  Main orchestrator – invoked via `python -m src.main`
# ---------------------------------------------------------------------

def run_pipeline():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Could not find configuration file at {CONFIG_PATH}")

    print("======================  Mini-Experiment  ======================")
    print(f"Config file : {CONFIG_PATH}")
    print(f"Model dir   : {MODEL_DIR}")
    print("==============================================================\n")

    # ------------------------------------------------------------------
    # 1. Train ----------------------------------------------------------
    # ------------------------------------------------------------------
    model, train_metrics = train_model(CONFIG_PATH, MODEL_DIR)
    print("\n[main] Training finished ⏱️  – summary:")
    for k, v in train_metrics.items():
        print(f"    {k:<15}: {v}")

    # ------------------------------------------------------------------
    # 2. Evaluate -------------------------------------------------------
    # ------------------------------------------------------------------
    ckpt_path = MODEL_DIR / "model.pt"
    eval_metrics = evaluate_model(CONFIG_PATH, ckpt_path)
    print("\n[main] Evaluation results:")
    for k, v in eval_metrics.items():
        print(f"    {k:<15}: {v}")

    # ------------------------------------------------------------------
    # 3. Visualisation --------------------------------------------------
    # ------------------------------------------------------------------
    _plot_2d_projection(model, yaml.safe_load(CONFIG_PATH.read_text()))

    print("\nAll done – you can now open the PDF in .research/iteration12/images 🎉")


if __name__ == "__main__":
    run_pipeline()
