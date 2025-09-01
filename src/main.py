"""
main.py – Entry point executed via `python -m src.main`.
Implements the end-to-end pipeline: preprocessing → training → evaluation.
Detailed results are printed to stdout as required.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from . import evaluate as ev
from . import preprocess as pp
from . import train as tr

CFG_DIR = Path("config")
DEFAULT_CFG = CFG_DIR / "config.yaml"  # adjusted to match the provided file name


def load_cfg(path: Path | str | None) -> Dict[str, Any]:
    """Load YAML config, falling back to the default configuration."""
    if path is None:
        path = DEFAULT_CFG
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    print(f"Using configuration: {path}")
    return yaml.safe_load(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end ACHyD prototype experiment")
    parser.add_argument("--cfg", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)

    # 1) preprocessing
    train_xy, val_xy, test_xy = pp.make_dataset(cfg["data"])
    print("Data statistics →", {k: v[0].shape[0] for k, v in zip(["train", "val", "test"], (train_xy, val_xy, test_xy))})

    # 2) training
    model_save = Path("models/simple_regressor.pt")
    model, history = tr.train_model(train_xy, val_xy, cfg["train"], save_path=model_save)

    # 3) evaluation
    mse = ev.evaluate_model(model, test_xy, cfg["eval"])
    print(f"\n===== FINAL RESULTS =====\nTest MSE: {mse:.4f}\n=========================")


if __name__ == "__main__":
    main()
