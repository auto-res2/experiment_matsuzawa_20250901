"""src/train.py
Train a simple feed-forward neural network on the pre-processed dataset and
save the trained model together with a training-loss figure (PDF).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # head-less backend
import matplotlib.pyplot as plt
import yaml

# Relative import – obey project layout
from .preprocess import DATA_DIR, maybe_prepare_data

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
MODELS_DIR = Path("models")
IMAGES_DIR = Path(".research/iteration10/images")  # ← updated per requirements
CONFIG_PATH = Path("config/config.yaml")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Simple feed-forward classifier
# -----------------------------------------------------------------------------
class IrisNet(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 16, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    # sensible defaults if no config provided
    return {
        "epochs": 50,
        "batch_size": 32,
        "lr": 1e-2,
        "weight_decay": 0.0,
        "hidden_dim": 16,
        "seed": 42,
    }


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
# Main train function called by src.main
# -----------------------------------------------------------------------------

def train() -> Tuple[float, Path]:
    """Entry point used by src.main.  Returns (final_accuracy, model_path)."""
    cfg = _load_config()
    _set_seed(cfg.get("seed", 0))

    train_npz, test_npz = maybe_prepare_data()  # ensure data present

    X_train = torch.tensor(train_npz["x"], dtype=torch.float32)
    y_train = torch.tensor(train_npz["y"], dtype=torch.long)
    X_test = torch.tensor(test_npz["x"], dtype=torch.float32)
    y_test = torch.tensor(test_npz["y"], dtype=torch.long)

    model = IrisNet(hidden_dim=cfg["hidden_dim"]).to(torch.device("cpu"))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # mini-batch training
    losses = []
    N = X_train.shape[0]
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            batch_x = X_train[idx]
            batch_y = y_train[idx]

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        if epoch % 10 == 0 or epoch == epochs:
            print(f"[train] epoch {epoch:>3}/{epochs}  loss={loss.item():.4f}")

    # Plot training loss
    plt.figure(figsize=(4, 3))
    plt.plot(range(1, epochs + 1), losses, marker="o", linewidth=1.5)
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy")
    plt.tight_layout()
    loss_fig_path = IMAGES_DIR / "training_loss.pdf"
    plt.savefig(loss_fig_path, bbox_inches="tight")

    # Save model (include config for downstream use)
    model_path = MODELS_DIR / "iris_net.pt"
    torch.save({"model_state": model.state_dict(), "cfg": cfg}, model_path)
    print(f"[train] model saved to {model_path.resolve()}")

    # quick accuracy on train set (for logging)
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
    print(f"[train] test accuracy after training ≈ {acc * 100:.2f}%")

    # store small JSON summary next to model (optional)
    summary_path = model_path.with_suffix(".json")
    summary_path.write_text(json.dumps({"accuracy": acc, **cfg}, indent=2))

    return acc, model_path
