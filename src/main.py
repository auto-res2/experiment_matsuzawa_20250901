# src/main.py
"""Project entry point.
Runs: 1) data preparation, 2) training, 3) evaluation.
Example usage (from project root):
    python -m src.main --epochs 3 --batch_size 128
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .preprocess import get_dataloaders
from .train import train_model
from .evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Tiny DHAC demo pipeline (CIFAR-10)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("===== Stage 1: Data preparation =====")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    print("\n===== Stage 2: Training =====")
    model_path = Path("models/simple_cnn.pt")
    train_model(
        train_loader,
        val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        model_save_path=model_path,
    )

    print("\n===== Stage 3: Evaluation =====")
    acc, _ = evaluate_model(model_path, test_loader, device=args.device)
    print(f"Test accuracy: {acc*100:.2f}% (checkpoint: {model_path})")


if __name__ == "__main__":
    main()
