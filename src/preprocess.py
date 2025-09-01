"""src/preprocess.py
Data/environment loading and light-weight preprocessing utilities extracted from
the original monolithic experiment script.
"""

import random
import os
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

# Project-specific imports ---------------------------------------------------------
from hdiff import HierarchicalDiffuser
from fasthidiff import monkey_patch_fasthidiff
from fasterdiffusion import patch_encoder_prop
from dpmsolvers import DPMSolverThreeStep

__all__ = ["set_seed", "load_env_and_planner"]


def set_seed(seed: int):
    """Set Python, NumPy and Torch RNG state for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_env_and_planner(
    task_name: str,
    method: str,
    device: str = "cuda:0",
):
    """Return (env, planner, obs_dim) and apply method-specific monkey patches."""

    # Environment ------------------------------------------------------------------
    env = gym.make(task_name)
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
