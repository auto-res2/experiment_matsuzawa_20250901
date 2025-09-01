
"""
src/evaluate.py
----------------
Standalone evaluation utilities.  They are used by src.main but can also be run
independently via `python -m src.evaluate --ckpt models/cnn.pt`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .train import SmallCNN  # re-use same architecture
from .utils import set_seed


@torch.no_grad()
def evaluate(ckpt: Path, batch_size: int = 256, seed: int = 0) -> float:
    """Loads a model checkpoint and returns test accuracy."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    test_ds = datasets.FashionMNIST(root=Path(ckpt).parent, train=False, download=True, transform=tfm)
    loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = SmallCNN()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
    acc = correct / len(loader.dataset)
    return acc


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    acc = evaluate(Path(args.ckpt), args.batch_size, args.seed)
    print(f"Accuracy: {acc:.4f}")
