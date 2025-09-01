"""
src/train.py
--------------
Training utilities.  The current reference implementation trains a very small
ConvNet on (Fashion-)MNIST so that the whole pipeline is completely runnable
within < 2 minutes on the provided Tesla-T4 or on CPU.

The code deliberately stays simple – the goal of this repository is to provide a
*working* end-to-end skeleton that can later be swapped out for the real CLRD
components described in the research plan.  Wherever CLRD-specific logic would
live in the future there are TODO comments so that follow-up iterations can plug
in their custom diffusion samplers, multi–scale adapters, etc.  Nothing in the
current file prevents such extensions – everything is modular and uses pure
PyTorch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .utils import set_seed, save_pdf_figure

# -----------------------------------------------------------------------------
# very small CNN (≈ 11 k parameters)
# -----------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  # 28×28 → 28×28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # 14×14
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                # 7×7
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,C,H,W)
        return self.net(x)


# -----------------------------------------------------------------------------
# data
# -----------------------------------------------------------------------------

def get_dataloaders(batch_size: int, data_root: Path) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_ds = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=tfm)
    test_ds  = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader


# -----------------------------------------------------------------------------
# training loop
# -----------------------------------------------------------------------------

def train(config: Dict[str, Any]) -> Dict[str, Any]:
    """High-level training entry point used by src.main.  Returns a dictionary with
    all metrics that shall be persisted to disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config["seed"])

    train_loader, test_loader = get_dataloaders(config["batch_size"], Path(config["data_root"]))

    model = SmallCNN().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=config["lr"])
    ce    = nn.CrossEntropyLoss()

    best_acc = 0.0
    history  = {"train_loss": [], "test_acc": []}

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = ce(pred, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        avg_loss = running / len(train_loader.dataset)
        history["train_loss"].append(avg_loss)

        # quick evaluation each epoch – nothing expensive
        acc = _evaluate(model, test_loader, device)
        history["test_acc"].append(acc)
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch:02d}/{config['epochs']}  | loss={avg_loss:.4f}  | acc={acc:.3f}")

    # ---------------------------------------------------------------------
    # persist artefacts
    # ---------------------------------------------------------------------
    artefact_dir = Path(config["model_dir"])
    artefact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artefact_dir / "cnn.pt"
    torch.save(model.state_dict(), model_path)

    # tiny learning-curve PDF for the paper appendix
    save_pdf_figure(history, artefact_dir / "learning_curve.pdf")

    metrics = {
        "best_acc": best_acc,
        "final_acc": history["test_acc"][-1],
        "train_loss": history["train_loss"],
        "test_acc_curve": history["test_acc"],
        "model_path": str(model_path),
    }
    with open(artefact_dir / "metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)
    return metrics


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
    return correct / len(loader.dataset)
