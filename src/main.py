"""src/main.py
Entry-point launched via `python -m src.main` as required.  It orchestrates the
full pipeline:
  1. Load YAML configuration (config/config.yaml).  If the file does not exist,
     a default configuration is used and written to disk so the user can edit
     it later.
  2. Pre-processing & data loaders are set up (src.preprocess).
  3. Model is trained (src.train) and stored in ./models/.
  4. The trained model is evaluated on the test split (src.evaluate).
  5. A high-quality PDF figure showing the training & validation loss curves is
     saved to ./.research/iteration24/images/loss_curve.pdf.
  6. All key metrics are printed to *stdout* for quick inspection.

The code purposefully remains concise but follows best practices (deterministic
seeds, directory creation, relative imports, proper PDF back-end handling).
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict

import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend suitable for headless CI
import matplotlib.pyplot as plt

from .train import train
from .evaluate import evaluate

# ──────────────────────────────────────────────────────────────────────────────
# Directories
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # project root (one level above src/)
CONFIG_DIR = ROOT / "config"
MODELS_DIR = ROOT / "models"
IMG_DIR = ROOT / ".research" / "iteration24" / "images"

for d in (CONFIG_DIR, MODELS_DIR, IMG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Default configuration – will be written to config/config.yaml on first run
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_CFG: Dict = {
    "seed": 42,
    "data": {
        "n_samples": 2000,
        "dim": 2,
        "val_split": 0.1,
        "test_split": 0.1,
        "flip_prob": 0.05,  # label noise probability
    },
    "model": {
        "hidden_dim": 32,
    },
    "optim": {
        "lr": 1e-3,
        "epochs": 20,
        "batch_size": 128,
    },
    "misc": {
        "verbose": True,
    },
}

CONFIG_FILE = CONFIG_DIR / "config.yaml"
if not CONFIG_FILE.exists():
    CONFIG_FILE.write_text(yaml.safe_dump(_DEFAULT_CFG))
    print(f"[INFO] Default configuration written to {CONFIG_FILE}")

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ──────────────────────────────────────────────────────────────────────────────
# Main routine
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # 1) Load config ----------------------------------------------------------
    cfg: Dict = yaml.safe_load(CONFIG_FILE.read_text())

    # 2) Reproducibility ------------------------------------------------------
    set_seeds(cfg["seed"])

    # 3) Train ---------------------------------------------------------------
    model, train_loss, val_loss = train(cfg)

    # 4) Persist model --------------------------------------------------------
    model_path = MODELS_DIR / "simple_net.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, model_path)

    # 5) Evaluate -------------------------------------------------------------
    acc = evaluate(model, cfg)

    # 6) Visualise ------------------------------------------------------------
    epochs = np.arange(1, len(train_loss) + 1)
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(epochs, train_loss, label="train", lw=2)
    plt.plot(epochs, val_loss,   label="val",   lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Training curves – SimpleNet")
    plt.legend()
    plt.grid(True, ls=":", alpha=0.6)
    plt.tight_layout()
    pdf_path = IMG_DIR / "loss_curve.pdf"
    plt.savefig(pdf_path, format="pdf")

    # 7) Print summary --------------------------------------------------------
    print("\n================  EXPERIMENT SUMMARY  ================")
    print(f"Seed                  : {cfg['seed']}")
    print(f"Train samples         : {int((1-cfg['data']['val_split']-cfg['data']['test_split'])*cfg['data']['n_samples'])}")
    print(f"Validation samples    : {int(cfg['data']['val_split']*cfg['data']['n_samples'])}")
    print(f"Test samples          : {int(cfg['data']['test_split']*cfg['data']['n_samples'])}")
    print(f"Final validation loss : {val_loss[-1]:.4f}")
    print(f"Test accuracy         : {acc*100:.2f} %")
    print(f"Model saved to        : {model_path}")
    print(f"Loss curve PDF        : {pdf_path}")
    print("======================================================\n")


if __name__ == "__main__":
    # Enable running via  `python src/main.py` for convenience.
    # The canonical invocation requested by the Instructions is
    #       python -m src.main
    # which also works thanks to the module guard above.
    main()
