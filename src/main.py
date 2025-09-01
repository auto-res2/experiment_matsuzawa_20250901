"""src/main.py
Entry point that orchestrates the whole experimental workflow.  Run with

    python -m src.main
"""

import torch

from src.evaluate import experiment1, experiment2, experiment3


def main():
    torch.set_float32_matmul_precision("high")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Uncomment experiments as desired
    experiment1(device)
    experiment2(device)
    experiment3(device)


if __name__ == "__main__":
    main()
