"""src/train.py
Light-weight training routine that fits a toy convolutional network on a
synthetic image dataset generated during the preprocessing step.  The goal is
not to obtain a useful model but to make the complete pipeline runnable and to
produce artefacts (trained weights + loss curve) that downstream scripts can
consume.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image, make_grid  # noqa: F401 – keep for parity with original code

# -------------------------------------------------------------
#  Simple CNN that roughly mimics an image-to-image network so
#  that we have something to save / load during evaluation.
# -------------------------------------------------------------


class TinyCNN(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


# ------------------------------------------------------------------
#  Public API that will be invoked from src.main
# ------------------------------------------------------------------


def run(cfg: Dict):
    """Train the dummy CNN for a few epochs and save artefacts.

    Args:
        cfg: dictionary with at least the following fields
              - "data_root": directory that contains the synthetic tensors
              - "model_dir": where to save the trained weights
    """
    rng = np.random.RandomState(0)
    torch.manual_seed(0)

    data_root = Path(cfg["data_root"])
    model_dir = Path(cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    imgs = torch.load(data_root / "train.pt")  # shape B×C×H×W, values in [0,1]

    ds = TensorDataset(imgs, imgs)  # identity mapping → auto-encoder style
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyCNN().to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    losses: List[float] = []
    n_epochs = 3  # short & sweet – keeps CI fast
    for epoch in range(n_epochs):
        running = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        epoch_loss = running / len(loader.dataset)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{n_epochs} – loss={epoch_loss:.4f}")

    # ------------------------------------------------------------------
    # save model + loss curve
    # ------------------------------------------------------------------
    torch.save(model.state_dict(), model_dir / "tinycnn.pth")
    with open(model_dir / "train_metrics.json", "w") as f:
        json.dump({"loss_curve": losses}, f, indent=2)

    # nice PDF for the paper
    import matplotlib.pyplot as plt

    plt.figure(figsize=(4, 3))
    plt.plot(range(1, n_epochs + 1), losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training curve – TinyCNN")
    plt.tight_layout()
    # ------------------------------------------------------------------
    #  All experiment images must live under .research/iteration3/images
    # ------------------------------------------------------------------
    img_dir = Path(".research/iteration3/images")
    img_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_dir / "training_loss.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    return losses
