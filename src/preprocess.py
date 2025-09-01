"""
preprocess.py – takes care of loading the raw Iris dataset, performing a
train/val/test split, feature standardisation, and persisting the result
under ./data/iris_preprocessed.npz so that subsequent runs skip the heavy
lifting.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
#  Paths & constants
# ----------------------------------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
FILE = DATA_DIR / "iris_preprocessed.npz"


# ----------------------------------------------------------------------------
#  Main entry – called from src.main
# ----------------------------------------------------------------------------

def preprocess(force: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_train, y_train, x_val, y_val, x_test, y_test).

    If a cached .npz file already exists we reuse it unless *force* is
    True.  The .npz contains six arrays exactly in that order.
    """

    if FILE.exists() and not force:
        cached = np.load(FILE)
        return tuple(cached[f"arr_{i}"] for i in range(6))  # type: ignore[return-value]

    iris = load_iris()
    x, y = iris.data.astype(np.float32), iris.target.astype(np.int64)

    # first split train+val vs. test
    x_tv, x_test, y_tv, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    # split train vs. val
    x_train, x_val, y_train, y_val = train_test_split(x_tv, y_tv, test_size=0.2, random_state=42, stratify=y_tv)

    # feature standardisation ------------------------------------------------
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    # persist to disk for future runs ---------------------------------------
    np.savez(FILE, x_train, y_train, x_val, y_val, x_test, y_test)
    meta = {"train": len(x_train), "val": len(x_val), "test": len(x_test)}
    (DATA_DIR / "iris_meta.json").write_text(json.dumps(meta, indent=2))
    return x_train, y_train, x_val, y_val, x_test, y_test
