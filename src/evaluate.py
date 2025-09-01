"""src/evaluate.py
Evaluation utilities used by src.main.  The code expects a trained model that
adheres to the simple forward-API defined in train.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from .preprocess import load_preprocessed_data
from .train import LogisticRegression

# -------------------------------------------------------------
#  Evaluation entry point – called by src.main
# -------------------------------------------------------------

def evaluate_model(cfg_path: Path, model_ckpt: Path) -> Dict[str, float]:
    """Load a checkpoint and compute accuracy on the validation set.

    Returns a small dict so that src.main can pretty-print results.
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    (_, _), (x_val, y_val) = load_preprocessed_data(cfg)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=cfg["batch_size"], shuffle=False)

    model = LogisticRegression(in_dim=x_val.shape[1], out_dim=len(torch.unique(y_val)))
    model.load_state_dict(torch.load(model_ckpt, map_location="cpu"))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total   += yb.size(0)

    return {
        "val_accuracy": correct / total if total else 0.0,
        "num_val_samples": total,
    }
