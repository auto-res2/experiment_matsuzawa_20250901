"""
src/main.py
-----------
Entry-point for the whole experiment suite.  It wires together the preprocessing
step, the training routine and the evaluation so that a single command

    python -m src.main

runs everything end-to-end and prints a concise report to stdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from rich import print as rprint
from rich.table import Table

from . import preprocess, train, evaluate
from .utils import set_seed


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "epochs": 3,            # keep tiny for demonstration purposes
    "batch_size": 128,
    "lr": 1e-3,
    "data_root": "data/raw",
    "model_dir": "models",
}


def pretty_table(title: str, rows: Dict[str, Any]):
    tbl = Table(title=title, show_header=True, header_style="bold cyan")
    tbl.add_column("Key", style="bold")
    tbl.add_column("Value")
    for k, v in rows.items():
        tbl.add_row(str(k), str(v))
    rprint(tbl)


def run(cfg: Dict[str, Any]):
    # ------------------------------------------------------------------
    # 1. preprocessing (no-op for now)
    # ------------------------------------------------------------------
    preprocess.prepare_data(cfg["data_root"])

    # ------------------------------------------------------------------
    # 2. training
    # ------------------------------------------------------------------
    rprint("[yellow]Starting training …")
    metrics = train.train(cfg)

    # ------------------------------------------------------------------
    # 3. evaluation (double-check that loading works)
    # ------------------------------------------------------------------
    acc = evaluate.evaluate(Path(metrics["model_path"]))
    metrics["eval_reloaded_acc"] = acc

    # ------------------------------------------------------------------
    # 4. persist & pretty-print summary
    # ------------------------------------------------------------------
    summary_path = Path(cfg["model_dir"]) / "summary.json"
    with open(summary_path, "w") as fp:
        json.dump(metrics, fp, indent=2)

    pretty_table("Experiment Summary", metrics)
    rprint(f"[green]Summary written to {summary_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal CLRD placeholder experiment")
    parser.add_argument("--config", type=str, help="Path to custom JSON config", default=None)
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.config is not None:
        with open(args.config) as fp:
            cfg.update(json.load(fp))
    set_seed(cfg["seed"])
    run(cfg)
