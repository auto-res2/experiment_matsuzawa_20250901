"""src/main.py
Project entry-point.  Run via
    python -m src.main
The script orchestrates preprocessing, training, evaluation, and logs results
with academic-quality PDF figures saved under .research/iteration10/images.
"""
from __future__ import annotations

import time
from pathlib import Path

from .preprocess import maybe_prepare_data
from .train import train
from .evaluate import evaluate

IMAGES_DIR = Path(".research/iteration10/images")  # ← updated path
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    t0 = time.time()
    # 1. Pre-processing (idempotent)
    maybe_prepare_data()

    # 2. Train the model
    acc_train, model_path = train()

    # 3. Evaluate
    acc_test, _ = evaluate(model_path)

    # 4. Print summary
    print("\n================= SUMMARY =================")
    print(f"Train accuracy (after final epoch): {acc_train * 100:.2f}%")
    print(f"Test  accuracy: {acc_test  * 100:.2f}%")
    print(f"Figures saved in: {IMAGES_DIR.resolve()}")
    print(f"Total wall-clock time: {time.time() - t0:.1f} s")
    print("===========================================")


if __name__ == "__main__":
    main()
