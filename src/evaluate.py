"""
evaluate.py
Evaluation and plotting helpers.
"""
from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
#  Figure directory – all images are stored under `.research/iteration3/images`
# -----------------------------------------------------------------------------
FIG_DIR = Path(".research/iteration3/images")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
#  Accuracy evaluation
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train()
    return 100.0 * correct / total

# -----------------------------------------------------------------------------
#  Plotting
# -----------------------------------------------------------------------------

def plot_accuracy_curve(stats: List[Dict[str, Any]], title: str, filename: str):
    tasks = [s['task'] for s in stats]
    accs  = [s['accuracy'] for s in stats]
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    plt.plot(tasks, accs, marker='o', label='HydraMemory')
    for t, a in zip(tasks, accs):
        plt.text(t, a + 0.3, f"{a:.1f}", fontsize=8)
    plt.xlabel('Task'); plt.ylabel('Accuracy (%)'); plt.title(title)
    plt.legend(); plt.tight_layout()
    pdf_path = FIG_DIR / f"{filename}.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure: {pdf_path}")
    plt.close()
