"""
main.py – orchestration entry-point
This module is executed via  `python -m src.main`  from the project root.
It simply runs (1) preprocessing, (2) training, (3) evaluation, printing all
relevant information to standard output so the user can inspect the results.
"""
from __future__ import annotations

from pathlib import Path
import time

from .preprocess import preprocess
from .train import train
from .evaluate import evaluate

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    t0 = time.time()
    print("\n====================  ACHyD ‑ Minimal Demo Pipeline  ====================\n")

    print("[1/3] Pre-processing data …")
    preprocess()

    print("\n[2/3] Training model …")
    model_path = train()

    print("\n[3/3] Evaluating model …")
    evaluate(model_path)

    dt = time.time() - t0
    print(f"\nPipeline finished in {dt:.1f} s – all artefacts written to disk.\n")


if __name__ == "__main__":
    main()
