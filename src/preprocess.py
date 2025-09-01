
"""
src/preprocess.py
-----------------
For this minimal runnable example there is no complex preprocessing.  A helper
is still provided to illustrate where such logic would live (e.g. wavelet
transforms, multi-scale latent extraction for CLRD, etc.).  The current function
merely makes sure the data directory exists and can be expanded later.
"""
from pathlib import Path


def prepare_data(data_root: str | Path = "data/raw") -> None:
    """Creates the data directory if it does not exist."""
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    # Future work: download custom datasets, convert to LMDB, create wavelet
    # decompositions, etc.
    print(f"Data directory ready: {root.resolve()}")
