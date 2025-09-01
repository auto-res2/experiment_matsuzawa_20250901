"""
evaluate.py – Evaluation & visualisation
The script loads the trained model, evaluates on the held-out test set
and produces a PDF figure suitable for academic publication.
Figures are stored under ./.research/iteration16/images (as per updated
specification).
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")  # head-less backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402

# -----------------------------------------------------------------------------
# Re-usable utility – falls back to local definition if src.utils is unavailable
# -----------------------------------------------------------------------------
try:
    from .utils import set_seed  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – local fallback for robustness

    def set_seed(seed: int | None = None) -> None:  # noqa: D401
        """Set random seeds for reproducibility (torch / numpy / python)."""
        if seed is None:
            return
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

from .train import SimpleRegressor

# Directory mandated by the updated instructions
IMG_DIR = Path(".research/iteration16/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def evaluate_model(
    model: SimpleRegressor,
    test_xy: Tuple[Tensor, Tensor],
    cfg: Dict,
    model_name: str = "simple_regressor",
) -> float:
    """Return test MSE and create a prediction vs. ground truth plot."""
    set_seed(cfg.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    x_te, y_te = test_xy
    x_te, y_te = x_te.to(device), y_te.to(device)
    preds = model(x_te).cpu()
    mse = torch.mean((preds - y_te.cpu()) ** 2).item()

    # ---------------- plot ----------------
    sns.set_theme(style="white", font_scale=1.1)
    plt.figure(figsize=(4, 4))
    plt.scatter(x_te.cpu().numpy(), y_te.cpu().numpy(), s=10, label="ground truth")
    plt.scatter(x_te.cpu().numpy(), preds.numpy(), s=10, label="prediction", alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(frameon=False)
    plt.tight_layout()
    out_f = IMG_DIR / f"{model_name}_pred_vs_gt.pdf"
    plt.savefig(out_f, dpi=300, bbox_inches="tight")
    print(f"Prediction figure saved → {out_f}")

    return mse
