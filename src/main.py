"""src/main.py
Entry-point called via `python -m src.main`.
It wires together preprocessing, training, evaluation and
stores artefacts / figures in the required locations.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from . import preprocess, train, evaluate

# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

# Save images to the requested directory
IMG_DIR = ROOT / ".research" / "iteration23" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = ROOT / "models"

# Robustly find the configuration file – fall back if the preferred name is missing
CONFIG_DIR = ROOT / "config"
if (CONFIG_DIR / "cartpole.yaml").exists():
    CONFIG_FILE = CONFIG_DIR / "cartpole.yaml"
else:
    CONFIG_FILE = CONFIG_DIR / "config.yaml"


def main():
    # ------------------- load config -------------------
    cfg = yaml.safe_load(CONFIG_FILE.read_text())
    print("[MAIN] Loaded config:\n" + json.dumps(cfg, indent=2))

    # ------------------- preprocessing -----------------
    env = preprocess.make_env(cfg)

    # ------------------- training ----------------------
    ckpt_path, reward_history = train.train(env, cfg, MODELS_DIR)

    # save reward curve
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(6, 3))
    plt.plot(reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Training Curve – CartPole-v1")
    plt.tight_layout()
    out_fig = IMG_DIR / "training_curve.pdf"
    plt.savefig(out_fig, bbox_inches="tight")
    print(f"[MAIN] Training curve saved → {out_fig.relative_to(ROOT)}")

    # ------------------- evaluation --------------------
    eval_stats = evaluate.evaluate(env, ckpt_path, episodes=cfg["eval_episodes"])
    print("[MAIN] Evaluation result:", json.dumps(eval_stats, indent=2))

    # persist evaluation as json for reproducibility
    eval_path = ROOT / "data" / "eval_stats.json"
    eval_path.parent.mkdir(exist_ok=True)
    eval_path.write_text(json.dumps(eval_stats, indent=2))
    print(f"[MAIN] Evaluation stats saved → {eval_path.relative_to(ROOT)}")

    env.close()


if __name__ == "__main__":
    main()
