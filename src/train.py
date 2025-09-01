# src/train.py
"""Training script for a tiny CIFAR-10 classifier.
The goal is not state-of-the-art accuracy but to supply a *runnable* example
that fulfils the project requirements:
  * uses the dataset returned by ``src.preprocess``.
  * stores the trained weights under ``models/``.
  * produces publication-quality training curves (loss & accuracy) as PDF
    under ``.research/iteration1/images``.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------
# Simple CNN – intentionally lightweight so that it trains fast
# -------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


# -------------------------------------------------------------
# Training routine
# -------------------------------------------------------------

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 3,
    lr: float = 1e-3,
    device: str | torch.device = "cuda",
    model_save_path: Path = Path("models/simple_cnn.pt"),
) -> Tuple[SimpleCNN, List[float], List[float]]:
    """Train *epochs* epochs, save best checkpoint, return model & history."""

    device = torch.device(device)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    history_loss: List[float] = []
    history_acc: List[float] = []

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimiser.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimiser.step()

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # quick validation pass
        val_acc = _eval_acc(model, val_loader, device)

        history_loss.append(train_loss)
        history_acc.append(val_acc)

        print(
            f"[Epoch {ep:02d}/{epochs}] loss={train_loss:.4f}  "
            f"train_acc={train_acc*100:.2f}%  val_acc={val_acc*100:.2f}%  "
            f"t={time.time()-t0:.1f}s",
            flush=True,
        )

        # checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_save_path)

    _plot_training_curves(history_loss, history_acc)
    return model, history_loss, history_acc


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def _eval_acc(model: nn.Module, loader: DataLoader, device="cuda") -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def _plot_training_curves(losses: List[float], accs: List[float]):
    """Save loss/accuracy PDF to .research/iteration1/images."""
    out_dir = Path(".research/iteration1/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(losses) + 1)
    sns.set_style("whitegrid")

    # Loss curve
    plt.figure(figsize=(4, 3))
    sns.lineplot(x=epochs, y=losses, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Cross-Entropy Loss"); plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.pdf", format="pdf")
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(4, 3))
    sns.lineplot(x=epochs, y=[a * 100 for a in accs], marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Validation Accuracy (%)"); plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.pdf", format="pdf")
    plt.close()
