"""src/main.py
Run the complete experiment via
    python -m src.main
"""
from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .preprocess import make_dataloaders
from .train import train
from .evaluate import evaluate

# ----------------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------------
IMAGES_DIR = Path(".research/iteration6/images")
MODELS_DIR = Path("models")


def _plot_training_curve(loss_hist, acc_hist, pdf_path: Path):
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(loss_hist, label="train-loss", color="tab:red")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(acc_hist, label="val-acc", color="tab:blue")
    ax2.set_ylabel("accuracy (%)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    fig.tight_layout()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main(cfg_path: Path):
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f)

    train_loader, val_loader, test_loader = make_dataloaders(cfg)

    print("[INFO] Starting training …")
    model, loss_hist, val_acc_hist = train(train_loader, val_loader, cfg, MODELS_DIR)

    print("[INFO] Evaluating …")
    test_acc = evaluate(model, test_loader, IMAGES_DIR)

    print("========== Final Results ==========")
    print(f"Test accuracy : {test_acc:.2f}%")

    # -------- training curve figure --------
    _plot_training_curve(loss_hist, val_acc_hist, IMAGES_DIR / "training_curve.pdf")
    print(f"All figures saved to {IMAGES_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Toy DHAC experiment")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        type=Path,
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()
    main(args.config)
