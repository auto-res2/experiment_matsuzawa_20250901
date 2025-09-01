"""src/main.py
Single entry-point for the whole project – executed via `python -m src.main`.
The script wires together preprocessing, (dummy) training and evaluation so
that the experiment runs end-to-end with one command and produces artefacts
that satisfy the requirements.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from rich import print as rprint

from . import preprocess as _pre
from . import train as _train
from . import evaluate as _eval


# -----------------------------------------------------------------------------
#  Parse CLI arguments first so that the other modules can use the flags.
# -----------------------------------------------------------------------------

def _cli() -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(description="End-to-end DHAC toy pipeline")
    parser.add_argument("--exp", type=int, default=1, choices=[1, 2, 3], help="Which experiment to run (subset implemented)")
    return parser.parse_args()


# -----------------------------------------------------------------------------
#  Main orchestrator
# -----------------------------------------------------------------------------


def main() -> None:  # noqa: D401
    args = _cli()

    # 1. Pre-processing – create tiny synthetic datasets
    rprint("[bold cyan]\n▶ Pre-processing synthetic datasets…")
    dataset_roots: Dict[str, Path] = _pre.run()

    # 2. Training – fit a toy CNN so that we have weights on disk
    rprint("[bold cyan]\n▶ Training dummy model…")
    _train.run({"data_root": dataset_roots["imagenet64"], "model_dir": "models"})

    # 3. Evaluation / experiments – lightweight reproduction of EXP-1
    rprint("[bold cyan]\n▶ Running evaluation / experiments…")
    _eval.run(args.exp, dataset_roots)

    rprint("\n[bold green]Finished – artefacts are stored in ./outputs and ./.research/iteration3/images")


# -----------------------------------------------------------------------------
#  Python module entry-point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
