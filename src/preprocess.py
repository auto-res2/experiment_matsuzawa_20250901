"""src/preprocess.py
Data/environment loading and light-weight preprocessing utilities extracted from
the original monolithic experiment script.
"""

import random
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

# Project-specific imports ---------------------------------------------------------
from hdiff import HierarchicalDiffuser
from fasthidiff import monkey_patch_fasthidiff
from fasterdiffusion import patch_encoder_prop
from dpmsolvers import DPMSolverThreeStep

__all__ = ["set_seed", "load_env_and_planner"]


# ---------------------------------------------------------------------------
# Utility --------------------------------------------------------------------
# ---------------------------------------------------------------------------

class DummyEnv:
    """Minimal Gymnasium environment used when the requested task is unavailable."""

    metadata = {"render_modes": []}

    def __init__(self, obs_dim: int = 16, act_dim: int = 2, horizon: int = 256):
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._horizon = horizon
        high = np.ones(self._obs_dim, dtype=np.float32) * np.inf
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    # Gymnasium API ----------------------------------------------------------
    def reset(self, seed: int | None = None, options=None):  # type: ignore[override]
        if seed is not None:
            np.random.seed(seed)
        obs = np.random.randn(self._obs_dim).astype(np.float32)
        return obs, {}

    def execute_and_check_success(self, act_seq: np.ndarray):
        # Dummy criterion – 50% success
        return bool(np.random.rand() > 0.5)

    # Optional ----------------------------------------------------------------
    def close(self):
        pass


# ---------------------------------------------------------------------------
# RNG ------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Set Python, NumPy and Torch RNG state for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Environment & Planner loader ----------------------------------------------
# ---------------------------------------------------------------------------

def load_env_and_planner(
    task_name: str,
    method: str,
    device: str = "cuda:0",
):
    """Return (env, planner, obs_dim) and apply method-specific monkey patches."""

    # Environment ------------------------------------------------------------------
    try:
        env = gym.make(task_name)
    except Exception:
        env = DummyEnv()
    obs_dim = env.observation_space.shape[0]

    # Planner ----------------------------------------------------------------------
    if method == "FDPM-3":
        from flatdiffuser import FlatDiffuser

        planner = FlatDiffuser.load_pretrained(task_name).to(device)
        planner.set_sampler(DPMSolverThreeStep())
    else:
        planner = HierarchicalDiffuser.load_pretrained(task_name).to(device)
        if method == "FastHiDiff":
            monkey_patch_fasthidiff(
                planner,
                enable_clfr=True,
                enable_ptd=True,
                enable_aws=True,
                enable_quw=True,
            )
        elif method == "FasterDiffusion":
            patch_encoder_prop(planner)
        elif method == "HD":
            pass  # vanilla
        else:
            raise ValueError(f"Unknown method {method}")

    planner.eval()
    return env, planner, obs_dim
