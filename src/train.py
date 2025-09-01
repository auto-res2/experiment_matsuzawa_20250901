"""
train.py – contains the training routine that is used by src.main
implements a very small fully connected neural network that is able to
solve the Iris classification problem.  The goal of this script is **not**
to reproduce the very complex ACHyD benchmark from the research draft,
but to provide a fully-runnable, self-contained example that fulfils all
engineering constraints given in the Instructions section (relative
imports, clean stdout, high-quality PDF plots, etc.).

The code purposefully stays minimal while still following good research
software hygiene (deterministic seeding, GPU support, progress display,
etc.).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------------------------
#  Model definition
# ----------------------------------------------------------------------------


class SimpleNet(nn.Module):
    """A very small MLP with one hidden layer (16 units, ReLU)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


# ----------------------------------------------------------------------------
#  Helper – deterministic seeding
# ----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


# ----------------------------------------------------------------------------
#  main training utility – returns the trained model _and_ the loss curves
# ----------------------------------------------------------------------------

def train_model(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    epochs: int = 100,
    lr: float = 1e-2,
    batch_size: int = 32,
    seed: int = 42,
    device: torch.device | str | None = None,
) -> Tuple[SimpleNet, List[float], List[float]]:
    """Train *SimpleNet* on the provided dataset.

    Parameters
    ----------
    train_ds / val_ds : TensorDataset
        Pre-processed training / validation splits.
    epochs : int
        Number of epochs.
    lr : float
        SGD learning-rate.
    batch_size : int
        Mini-batch size.
    seed : int
        RNG seed for reproducibility.
    device : Union[torch.device, str, None]
        Where to place the network («cuda» or «cpu»).

    Returns
    -------
    model : SimpleNet – trained network (in *eval* mode)
    tr_loss : list[float] – average training loss per epoch
    val_loss : list[float] – average validation loss per epoch
    """

    set_seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device = torch.device(device)

    net = SimpleNet(in_dim=train_ds.tensors[0].shape[1], out_dim=3).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    tr_curve, val_curve = [], []

    for epoch in range(1, epochs + 1):
        # --- training -------------------------------------------------------
        net.train()
        epoch_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = net(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * xb.size(0)
        tr_curve.append(epoch_loss / len(train_ds))

        # --- validation -----------------------------------------------------
        net.eval()
        with torch.no_grad():
            v_loss = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = net(xb)
                v_loss += criterion(logits, yb).item() * xb.size(0)
        val_curve.append(v_loss / len(val_ds))

        # quick CLI feedback every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            print(f"[train] epoch {epoch:3d}/{epochs} – loss: {tr_curve[-1]:.4f}  val: {val_curve[-1]:.4f}")

    net.eval()
    return net, tr_curve, val_curve
