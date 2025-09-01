"""
main.py
~~~~~~~
Experiment driver that wires the *preprocess*, *train*, and *evaluate*
modules together.  All stdout logging is *detailed* so that a researcher
can follow the full experimental trace when calling

    python -m src.main
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import uuid

from . import preprocess, train, evaluate
from .train import PROJECT_ROOT


# -----------------------------------------------------------------------------
# Helper for rich console printing (no extra dependency if unavailable)
# -----------------------------------------------------------------------------
try:
    from rich import print as rprint
except ImportError:  # pragma: no cover – fallback if rich not installed
    def rprint(*args, **kwargs):  # type: ignore[override]
        print(*args, **kwargs)


def _save_metadata(cfg: dict) -> None:
    meta = {
        "run_id": cfg["run_id"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg,
    }
    out_dir = PROJECT_ROOT / "runs"
    out_dir.mkdir(exist_ok=True, parents=True)
    path = out_dir / f"summary_{cfg['run_id']}.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    rprint(f"[bold green]Saved run-metadata to[/] {path.relative_to(PROJECT_ROOT)}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:  # pylint: disable=too-many-statements
    parser = argparse.ArgumentParser(description="End-to-end MNIST experiment")
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    args = parser.parse_args()

    cfg: dict = {
        "epochs": args.epochs,
        "lr": args.lr,
        "run_id": uuid.uuid4().hex[:8],
    }

    rprint("[bold cyan]Step 1/3:[/] Pre-processing dataset …")
    preprocess.prepare_datasets()

    rprint("[bold cyan]Step 2/3:[/] Training model …")
    train.train_model(epochs=cfg["epochs"], lr=cfg["lr"])

    rprint("[bold cyan]Step 3/3:[/] Evaluating model …")
    metrics = evaluate.evaluate_model()
    rprint(f"[bold magenta]Final test accuracy = {metrics['accuracy']:.4f}")

    _save_metadata(cfg | metrics)


if __name__ == "__main__":
    main()
