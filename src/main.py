"""src/main.py
--------------
Entry-point for the experiment – invoked via `python -m src.main`.
It orchestrates the whole pipeline:
  1. Loads hyper-parameters from `config/config.yaml` (creates a reasonable
     default if the file is missing).
  2. Calls `src.train.train_model`.
  3. Evaluates the trained model via `src.evaluate.evaluate`.
  4. Generates a PDF figure with training/validation curves and stores it in
     `.research/iteration19/images` as mandated.
  5. Writes a small summary of the results to *stdout*.

The code purposefully avoids any external dependencies beyond the ones listed
in `requirements.txt` and PyTorch/torchvision.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # headless back-ends are safer on servers/CI
import matplotlib.pyplot as plt
import yaml

from .train import train_model
from .evaluate import evaluate

# -----------------------------------------------------------------------------
# 1) Configuration handling
# -----------------------------------------------------------------------------
CONFIG_DIR = Path("config"); CONFIG_DIR.mkdir(exist_ok=True)
CFG_FILE = CONFIG_DIR / "config.yaml"

def _default_cfg() -> Dict:
    return {
        "batch_size": 128,
        "epochs": 5,
        "lr": 1e-3,
        "val_split": 0.1,
    }

if CFG_FILE.exists():
    cfg = yaml.safe_load(CFG_FILE.read_text())
else:
    cfg = _default_cfg()
    CFG_FILE.write_text(yaml.safe_dump(cfg))
    print("[INFO] Default config.yaml created – feel free to edit and re-run.")

print("[INFO] Config:", json.dumps(cfg, indent=2))

# -----------------------------------------------------------------------------
# 2) Run the pipeline
# -----------------------------------------------------------------------------
model, train_losses, val_losses = train_model(cfg)
acc = evaluate(model, batch_size=cfg["batch_size"])
print(json.dumps({"test_accuracy": acc}))

# -----------------------------------------------------------------------------
# 3) Visualisation – save as PDF for publication quality
# -----------------------------------------------------------------------------
# All images must be stored under `.research/iteration19/images` as required.
img_dir = Path(".research/iteration19/images")
img_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(6, 3))
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.legend()
plt.tight_layout()

fig_path = img_dir / "mnist_loss_curves.pdf"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")

# The previous implementation attempted to use Path.relative_to with a mixture
# of absolute/relative paths, which raises `ValueError`.  Instead, we simply
# print the (relative) POSIX path that we know lives inside the project tree.
print(f"[INFO] Training curves saved → {fig_path.as_posix()}")
