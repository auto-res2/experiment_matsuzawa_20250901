"""
train.py – Training utilities for a toy regression task
The goal is to keep the code base minimal but still demonstrate the
complete research-style pipeline requested in the Instructions.
The model learns y = 2x + ε from synthetic data generated in
preprocess.py.  Training artefacts are stored under ./models.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

# relative import (module lives in the same package – src)
from .utils import set_seed

# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------
class SimpleRegressor(nn.Module):
    """A two-layer perceptron for 1-D regression."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401 (plain docstring)
        """Forward pass."""
        return self.net(x)


# -----------------------------------------------------------------------------
# Training routine
# -----------------------------------------------------------------------------

def train_model(
    train_xy: Tuple[Tensor, Tensor],
    val_xy: Tuple[Tensor, Tensor],
    cfg: Dict,
    save_path: Path | None = None,
) -> Tuple[SimpleRegressor, Dict]:
    """Train the model and return the trained instance + history.

    Parameters
    ----------
    train_xy / val_xy: Tuple containing (x, y) tensors.
    cfg              : Dict with hyper-parameters.
    save_path        : If given, serialises the trained weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", 0))

    x_tr, y_tr = train_xy
    x_va, y_va = val_xy

    net = SimpleRegressor(cfg["hidden_dim"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    dl = DataLoader(TensorDataset(x_tr, y_tr), batch_size=cfg["batch_size"], shuffle=True)

    history = {"train_loss": [], "val_loss": []}
    t0 = time.time()
    for epoch in range(cfg["epochs"]):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = net(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        # log
        net.eval()
        with torch.no_grad():
            tr_loss = loss_fn(net(x_tr.to(device)), y_tr.to(device)).item()
            va_loss = loss_fn(net(x_va.to(device)), y_va.to(device)).item()
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        if (epoch + 1) % cfg["print_every"] == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:03d}/{cfg['epochs']} – "
                f"train MSE: {tr_loss:.4f} – val MSE: {va_loss:.4f}"
            )

    dur = time.time() - t0
    print(f"Training finished in {dur:.1f} s. Best val MSE: {min(history['val_loss']):.4f}")

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), save_path)
        print(f"Model weights saved → {save_path}")

    return net, history
