"""src/evaluate.py
Loads the model written by `train.py` and reports test-set accuracy.  This file
is deliberately lightweight so that it can be executed quickly even on CPU-
only environments used for automatic assessment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import MNIST

from .train import SimpleCNN, MODEL_FILE, DATA_DIR


@torch.inference_mode()
def evaluate_model(model_path: Path | str | None = None) -> Dict[str, Any]:
    model_path = Path(model_path or MODEL_FILE)
    if not model_path.exists():
        raise FileNotFoundError(
            "Model file not found.  Run `python -m src.main pipeline` first to "
            "train the network."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleCNN().to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    tf = T.Compose([T.ToTensor()])
    test_loader = DataLoader(
        MNIST(root=str(DATA_DIR), train=False, download=True, transform=tf),
        batch_size=256,
        shuffle=False,
        num_workers=2,
    )

    correct, n = 0, 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        n += y.size(0)
    acc = correct / n
    print(f"Test-set accuracy : {acc:.3%}  (evaluated on {n} images)")
    return {"accuracy": acc}


if __name__ == "__main__":
    evaluate_model()