"""
train.py
~~~~~~~~
Training script for a toy image–classification experiment (MNIST).
The model is a very small CNN that achieves >98 % accuracy after a
few epochs on a single GPU/CPU.

The *trained* model is stored under ``models/mnist_cnn.pt``.
All code is written with purely *relative imports* so that the project
can be executed via
    python -m src.main
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Constants & Directories
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RESEARCH_IMG_DIR = PROJECT_ROOT / ".research" / "iteration1" / "images"
RESEARCH_IMG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "mnist_cnn.pt"


# -----------------------------------------------------------------------------
# Model definition
# -----------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):  # type: ignore[override]
        x = self.features(x)
        x = self.classifier(x)
        return x


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------

def _get_dataloaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(root=PROJECT_ROOT / "data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=PROJECT_ROOT / "data", train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_dl, test_dl


def train_model(epochs: int = 3, lr: float = 1e-3, device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu") -> Path:
    """Train model – return path to trained weights."""
    train_dl, test_dl = _get_dataloaders()
    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        with tqdm(total=len(train_dl), desc=f"[Train] Epoch {epoch}/{epochs}") as pbar:
            for imgs, labels in train_dl:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * imgs.size(0)
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
        # quick validation accuracy for information only
        acc = _evaluate_once(model, test_dl, device)
        print(f"Epoch {epoch:02d}  val-acc={acc:.4f}  avg-loss={epoch_loss/len(train_dl.dataset):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved trained weights to {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    return MODEL_PATH


def _evaluate_once(model: nn.Module, dl: DataLoader, device: torch.device | str) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in dl:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(imgs)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
    acc = correct / len(dl.dataset)
    model.train()
    return acc


# -----------------------------------------------------------------------------
# CLI (useful for quick standalone execution)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MNIST CNN")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_model(epochs=args.epochs, lr=args.lr)
