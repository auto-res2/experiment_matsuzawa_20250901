"""src/main.py
Entry point that orchestrates the whole experimental workflow.  Run with

    python -m src.main
"""

from pathlib import Path
import torch

from src.evaluate import experiment1, experiment2, experiment3


def main():
    torch.set_float32_matmul_precision("high")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Ensure the image output directory exists ----------------------------------
    Path(".research/iteration3/images").mkdir(parents=True, exist_ok=True)

    # Run a lightweight version of the experiments. The loops have been shortened
    # to keep the execution time reasonable inside the testing sandbox.
    experiment1(device)
    experiment2(device)
    experiment3(device)


if __name__ == "__main__":
    main()
