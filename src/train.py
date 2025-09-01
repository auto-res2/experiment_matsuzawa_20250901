"""
train.py – model training
This module trains a very small feed-forward neural network on the
(pre-)processed data that is written by `preprocess.py`.
The trained model is stored under `models/model.pt`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from .preprocess import preprocess

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
CONFIG_DIR = ROOT / "config"


class SimpleMLP(nn.Module):
    """A tiny two-layer perceptron."""

    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    cfg_path = CONFIG_DIR / "default.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fp:
            return yaml.safe_load(fp)
    # sensible defaults
    return {
        "batch_size": 64,
        "epochs": 50,
        "lr": 1e-3,
        "hidden": 32,
    }


def build_loaders(x_train: torch.Tensor, y_train: torch.Tensor,
                  x_val: torch.Tensor, y_val: torch.Tensor,
                  batch_size: int) -> Tuple[DataLoader, DataLoader]:
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training API (called from src.main)
# ---------------------------------------------------------------------------

def train() -> Path:
    """High-level training wrapper.
    Returns the path of the saved model.
    """
    cfg = load_config()

    # make sure data exists – if not, run preprocessing
    data_pt = DATA_DIR / "dataset.pt"
    if not data_pt.exists():
        preprocess()

    data = torch.load(data_pt)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]

    model = SimpleMLP(in_dim=x_train.shape[1], hidden=cfg["hidden"], out_dim=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    train_loader, val_loader = build_loaders(x_train, y_train, x_val, y_val,
                                             cfg["batch_size"])

    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                pred_lbl = logits.argmax(dim=1)
                correct += (pred_lbl == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1:02d}/{cfg['epochs']}  loss={epoch_loss:.4f}  val_acc={val_acc:.3f}")

    model_path = MODEL_DIR / "model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved → {model_path.relative_to(ROOT)}")
    return model_path
