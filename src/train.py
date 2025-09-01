"""src/train.py
-----------------
Contains the training loop used by `src.main`.  The trainer is deliberately
kept very small so that the whole project can be executed on a single GPU/CPU
within the evaluation time-limit while still demonstrating the full pipeline
(pre-processing → training → evaluation → visualisation).

The example trains a simple two-layer MLP on the MNIST classification task
(downloaded automatically by `src.preprocess`).  All parameters such as the
number of epochs, learning-rate, etc. are provided by the `config.yaml` file
loaded in `src.main`.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

from .preprocess import prepare_datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """A very small two-layer perceptron for 28×28 images."""

    def __init__(self, in_dim: int = 28 * 28, hidden: int = 256, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 – simple forward
        x = x.flatten(start_dim=1)
        return self.net(x)


def train_model(cfg: Dict) -> Tuple[nn.Module, List[float], List[float]]:
    """Train the model and return (model, train_losses, val_losses)."""

    # ------------------------------------------------------------------
    # 1) Data
    # ------------------------------------------------------------------
    data_root = Path("data")
    train_ds, test_ds = prepare_datasets(data_root)

    val_split = cfg.get("val_split", 0.1)
    val_len = int(len(train_ds) * val_split)
    train_len = len(train_ds) - val_len
    train_ds, val_ds = random_split(train_ds, [train_len, val_len])

    dl_train = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    dl_val = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    # ------------------------------------------------------------------
    # 2) Model, loss, optimiser
    # ------------------------------------------------------------------
    model = MLP().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # ------------------------------------------------------------------
    # 3) Training loop
    # ------------------------------------------------------------------
    train_losses, val_losses = [], []
    epochs = cfg["epochs"]
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad(set_to_none=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimiser.step()
            ep_loss += loss.item() * xb.size(0)
        ep_loss /= train_len
        train_losses.append(ep_loss)

        # ––– validation –––
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb in dl_val:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)
            val_loss /= val_len
        val_losses.append(val_loss)

        print(json.dumps({"epoch": ep, "train_loss": ep_loss, "val_loss": val_loss}))

    # save model
    models_dir = Path("models"); models_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models_dir / "mnist_mlp.pt")

    return model, train_losses, val_losses
