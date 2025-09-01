"""src/preprocess.py
Data-loading utilities.  A real diffusion-policy project would use D4RL
or other offline RL datasets.  For demonstration purposes we fall back
to a *random* dataset if the environment does not expose `.get_dataset()`.
"""
from __future__ import annotations

import numpy as np
from typing import Dict


def load_dataset(env) -> Dict[str, np.ndarray]:
    """Return a dictionary with keys observations / actions / rewards / terminals.
    If the env already supplies a dataset (D4RL) we simply forward it.
    """
    if hasattr(env, "get_dataset"):
        return env.get_dataset()

    # ----------------  fallback: create a random dataset  ----------------
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    N = 10_000  # small dummy dataset
    return {
        "observations": np.random.randn(N, obs_dim).astype(np.float32),
        "actions": np.random.randn(N, act_dim).astype(np.float32),
        "rewards": np.random.randn(N, 1).astype(np.float32),
        "terminals": np.random.randint(0, 2, size=(N, 1)).astype(np.float32),
    }
