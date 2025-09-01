"""src/preprocess.py
Download / load the Iris dataset, perform a simple train-test split, standardise
features, and persist them under ./data/ as compressed .npz files.
The helper function `maybe_prepare_data()` is idempotent and safe to call
multiple times.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

TARGET_NAMES = ["setosa", "versicolor", "virginica"]

TRAIN_NPZ = DATA_DIR / "iris_train.npz"
TEST_NPZ = DATA_DIR / "iris_test.npz"


def _prepare() -> Tuple[dict, dict]:
    iris = datasets.load_iris()
    X = iris.data.astype(np.float32)
    y = iris.target.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    np.savez_compressed(TRAIN_NPZ, x=X_train, y=y_train)
    np.savez_compressed(TEST_NPZ, x=X_test, y=y_test)
    print("[preprocess] prepared Iris dataset → ./data/")

    return {"x": X_train, "y": y_train}, {"x": X_test, "y": y_test}


def maybe_prepare_data() -> Tuple[dict, dict]:
    """Ensures pre-processed dataset files exist and returns them."""
    if not (TRAIN_NPZ.exists() and TEST_NPZ.exists()):
        return _prepare()
    train = dict(np.load(TRAIN_NPZ))
    test = dict(np.load(TEST_NPZ))
    return train, test
