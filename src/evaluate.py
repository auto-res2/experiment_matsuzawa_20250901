"""src/evaluate.py
Evaluation helpers (metrics & plotting).
"""
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.cuda.amp import autocast

# ------------------------------------------------------------------
#  Save directory for all generated figures
# ------------------------------------------------------------------
# All images must be stored under .research/iteration8/images
FIG_DIR = Path(".research/iteration8/images")
# Ensure the directory hierarchy exists
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
#  Accuracy utility
# ------------------------------------------------------------------

def accuracy(model, loader: Iterable, device="cuda", is_nlp: bool = False):
    """Compute top-1 accuracy on `loader`. Works for vision & NLP variants."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            if not is_nlp:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                with autocast():
                    logits = model(x)
            else:
                ids, mask, y = batch
                ids = ids.to(device)
                mask = mask.to(device)
                y = y.to(device)
                with autocast():
                    logits = model(ids, mask)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    model.train()
    return 100 * correct / total

# ------------------------------------------------------------------
#  Figure utility – line plot with value annotations
# ------------------------------------------------------------------

def save_line(stats, title: str, fname: str):
    tasks = [s["task"] for s in stats]
    acc = [s["acc"] for s in stats]
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    plt.plot(tasks, acc, marker="o")
    for t, a in zip(tasks, acc):
        plt.text(t, a + 0.4, f"{a:.1f}", fontsize=8)
    plt.xlabel("Task")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.tight_layout()
    path = FIG_DIR / fname
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {path}")
