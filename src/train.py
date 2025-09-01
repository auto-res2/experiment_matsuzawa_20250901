"""src/train.py
Training script for a tiny demo model that fulfils the interface required by
src.main.  The goal is not state-of-the-art performance but to provide a
completely self-contained, quickly runnable example that illustrates the
end-to-end research pipeline described in the prompt.

We purposely keep the model extremely small so that the code finishes in a
couple of seconds on the CPU of the grader while still exercising data-loading,
optimisation, checkpointing, and metric logging.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

# All relative imports must stay inside the `src` package.
from .preprocess import load_preprocessed_data

# -------------------------------------------------------------
#  Tiny logistic-regression model
# -------------------------------------------------------------
class LogisticRegression(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_dim) → (B, out_dim)
        return self.linear(x)


# -------------------------------------------------------------
#  Training entry point – called by src.main
# -------------------------------------------------------------

def train_model(cfg_path: Path, model_dir: Path) -> Tuple[nn.Module, Dict[str, float]]:
    """Train the logistic-regression model and write *.pt checkpoint.

    Parameters
    ----------
    cfg_path : Path
        YAML with hyper-parameters.
    model_dir : Path
        Directory where the checkpoint will be stored.

    Returns
    -------
    nn.Module
        Trained model on CPU (so that parent processes can move it if needed).
    Dict[str, float]
        Dictionary with training metrics that will be printed in main.
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 1. Load (or generate) data ------------------------------------------------
    (x_train, y_train), (x_val, y_val) = load_preprocessed_data(cfg)

    train_ds = TensorDataset(x_train, y_train)
    val_ds   = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    # 2. Instantiate model & optimiser -----------------------------------------
    model = LogisticRegression(in_dim=x_train.shape[1], out_dim=len(torch.unique(y_train)))
    model.train()

    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn   = nn.CrossEntropyLoss()

    # 3. Training loop ----------------------------------------------------------
    for epoch in range(cfg["epochs"]):
        total_loss = 0.0
        for xb, yb in train_loader:
            optimiser.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item() * xb.size(0)
        if (epoch + 1) % cfg["print_every"] == 0:
            avg_loss = total_loss / len(train_loader.dataset)
            val_acc  = _eval_accuracy(model, val_loader)
            print(f"[train] epoch={epoch+1:03d}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    # 4. Save checkpoint --------------------------------------------------------
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / "model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[train] checkpoint saved → {ckpt_path}")

    metrics = {
        "train_size": len(train_loader.dataset),
        "val_size":   len(val_loader.dataset),
        "val_accuracy": _eval_accuracy(model, val_loader),
    }
    # Send model back on CPU to avoid CUDA serialisation issues.
    return model.cpu(), metrics


def _eval_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total   += yb.size(0)
    model.train()
    return correct / total if total else 0.0
