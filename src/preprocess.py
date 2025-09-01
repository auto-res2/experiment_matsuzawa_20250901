"""src/preprocess.py
For this simple RL experiment preprocessing is minimal –
we just create the environment and set seeds.
"""
from __future__ import annotations

from typing import Dict
import gymnasium as gym

from .train import set_seed


def make_env(cfg: Dict) -> gym.Env:
    set_seed(cfg.get("seed", None))
    env = gym.make(cfg["env_name"])
    return env
