"""src/train.py
Training utilities for the toy DHAC-placeholder experiment.
The module intentionally stays lightweight so that it can run on a
CPU-only laptop in < 2 minutes while preserving the project structure
requested in the task description.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from tqdm import tqdm


class SimpleCNN(nn.Module):
    """A very small CNN (≈ 0.1 M params) good enough for MNIST/CIFAR-10."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        # For 28×28 inputs (MNIST after padding) we end up with 7×7 feature maps
        self.head = nn.Linear(64 * 7 * 7, n_classes)

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        return self.head(self.body(x))


def _create_model(num_classes: int = 10, device: torch.device | str = "cpu") -> nn.Module:
    model = SimpleCNN(num_classes)
    return model.to(device)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict,
    save_dir: Path,
) -> Tuple[nn.Module, List[float], List[float]]:
    """Train the model for *cfg["epochs"]* epochs.

    Returns
    -------
    model : nn.Module
        The trained model (on CPU).
    train_loss_hist : list[float]
        Per-epoch training loss.
    val_acc_hist : list[float]
        Per-epoch validation accuracy in %.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _create_model(cfg.get("num_classes", 10), device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    train_loss_hist: List[float] = []
    val_acc_hist: List[float] = []

    num_epochs = cfg.get("epochs", 5)
    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)
        train_loss_hist.append(train_loss)

        # ---------------- validation -----------------
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
        acc = 100.0 * correct / len(val_loader.dataset)
        val_acc_hist.append(acc)
        print(f"Epoch {epoch}: train-loss={train_loss:.4f}  val-acc={acc:.2f}%")

    # Persist model (always on CPU for portability)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.to("cpu").state_dict(), save_dir / "model.pt")

    return model.to("cpu"), train_loss_hist, val_acc_hist
