"""src/main.py
Main entry-point for running the demo experiment.  This script ties
together the pre-processing, training, evaluation, and visualisation
steps so that users can simply execute
    python -m src.main
from the project root.

All plots are saved as vector-graphics PDF in
    .research/iteration7/images/
per the project instructions.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")  # headless / servers
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from .preprocess import load_config
from .train import train
from .evaluate import evaluate


# -------------------------  Directories  ------------------------- #
IMG_DIR = Path(".research/iteration7/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------  MAIN  ----------------------------- #

def run_experiment(cfg: Dict[str, Any]):
    print("\n==========  EXPERIMENT CONFIG  ==========")
    for k, v in cfg.items():
        print(f"{k:>15s}: {v}")
    print("========================================\n")

    # --------------------  TRAIN  -------------------- #
    policy, stats = train(cfg)

    # ------------------  EVALUATE  ------------------- #
    mean_r, std_r = evaluate(
        policy,
        cfg["env_name"],
        cfg["eval_episodes"],
        cfg["max_steps"],
        device=cfg["device"],  # ensure tensors are on the correct device
    )
    print(
        f"Evaluation over {cfg['eval_episodes']} episodes – mean reward = {mean_r:.2f} ± {std_r:.2f}\n"
    )

    # ------------------  PLOT CURVE  ------------------ #
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(6, 4))
    plt.plot(stats["episode"], stats["reward"], label="Episode reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training curve – CartPole (REINFORCE)")
    plt.tight_layout()
    pdf_path = IMG_DIR / "training_curve.pdf"
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"Saved training curve → {pdf_path.relative_to(Path.cwd())}\n")

    # Note: in a full research setting you might return additional
    # artefacts (trained model path, raw CSV, etc.).  For brevity we end
    # the demo here.


def main():
    cfg_path = os.environ.get("CONFIG", None)  # optional env-var override
    cfg = load_config(cfg_path)
    # if GPU is available, move model computations there – the demo is
    # tiny so this mainly shows how to respect the GPU in the code.
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    run_experiment(cfg)


if __name__ == "__main__":
    main()
