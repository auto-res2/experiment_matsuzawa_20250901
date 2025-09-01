"""src/train.py
Training script for a tiny feed-forward network on a synthetic binary
classification task.  The training routine is deliberately lightweight so
that CI (or a laptop without GPU) can finish it within a few seconds while
still exercising the full pipeline (pre-processing → training → evaluation →
visualisation).

The trained model is stored in ./models/simple_net.pt
Loss curves are returned to the caller so that `src.main` can create high-
quality PDF figures afterwards.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .preprocess import make_dataloaders  # relative import, see Instructions

# ──────────────────────────────────────────────────────────────────────────────
# Simple feed-forward classifier
# ──────────────────────────────────────────────────────────────────────────────

class SimpleNet(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (N, D) → (N, 2)
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Training routine
# ──────────────────────────────────────────────────────────────────────────────

def train(config: dict) -> Tuple[nn.Module, List[float], List[float]]:
    """Train *SimpleNet* according to *config* and return (model, train_loss, val_loss).

    The caller (src.main) is responsible for serialising the model and for any
    further processing/visualisation.  Returning the loss curves keeps this
    function unit-test friendly.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Data -----------------------------------------------------------------
    loaders = make_dataloaders(config)
    train_loader: DataLoader = loaders["train"]
    val_loader:   DataLoader = loaders["val"]

    # 2) Model ----------------------------------------------------------------
    model = SimpleNet(input_dim=config["data"]["dim"], hidden_dim=config["model"]["hidden_dim"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=config["optim"]["lr"])

    epochs = config["optim"]["epochs"]
    train_losses, val_losses = [], []

    # 3) Training loop --------------------------------------------------------
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # validation
        model.eval()
        with torch.no_grad():
            vloss = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vloss += criterion(model(xb), yb).item() * xb.size(0)
            vloss /= len(val_loader.dataset)
            val_losses.append(vloss)

        if config["misc"]["verbose"]:
            print(f"Epoch {epoch:02d}/{epochs} | train {epoch_loss:.4f} | val {vloss:.4f}")

    dt = time.perf_counter() - t0
    if config["misc"]["verbose"]:
        print(f"[train] finished in {dt:.2f} s → final val-loss {val_losses[-1]:.4f}")

    return model, train_losses, val_losses
