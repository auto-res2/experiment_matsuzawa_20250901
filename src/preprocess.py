"""
preprocess.py – data preparation
Loads the iris data set via scikit-learn, splits it into train/val/test and
stores the resulting tensors for fast re-use.  Also plots a simple pair plot
(which can be useful for quick sanity checks).
"""
from __future__ import annotations

from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
IMG_DIR = ROOT / ".research" / "iteration11" / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def preprocess() -> None:
    """Pre-process the Iris data set and save it as tensors."""
    iris = load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)

    # standardise features
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # split – 60 % train, 20 % val, 20 % test
    x_train, x_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
    x_val, x_test, y_val, y_test = train_test_split(x_tmp, y_tmp, test_size=0.5, random_state=0, stratify=y_tmp)

    tensors = {
        "x_train": torch.from_numpy(x_train),
        "y_train": torch.from_numpy(y_train),
        "x_val": torch.from_numpy(x_val),
        "y_val": torch.from_numpy(y_val),
        "x_test": torch.from_numpy(x_test),
        "y_test": torch.from_numpy(y_test),
    }
    torch.save(tensors, DATA_DIR / "dataset.pt")
    print(f"Pre-processed data saved → {(DATA_DIR / 'dataset.pt').relative_to(ROOT)}")

    # quick pair-plot – useful for inspection
    df = pd.DataFrame(X, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(y, iris.target_names)
    sns.pairplot(df, hue="species", corner=True)
    plt.tight_layout()
    fig_path = IMG_DIR / "iris_pairplot.pdf"
    plt.savefig(fig_path)
    print(f"Pair plot saved → {fig_path.relative_to(ROOT)}")
