"""src/preprocess.py
For the simple CartPole example we do not require any heavy
pre-processing, but we keep the module so that the pipeline matches the
spec provided in the instructions.
"""
from __future__ import annotations

from typing import Dict, Any
import yaml


DEFAULT_CONFIG = {
    "seed": 42,
    "env_name": "CartPole-v1",
    "total_episodes": 500,
    "max_steps": 200,
    "learning_rate": 1e-2,
    "gamma": 0.99,
    "hidden_size": 128,
    "eval_episodes": 20,
}


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load YAML configuration; fall back to defaults if not provided."""
    if path is None:
        return DEFAULT_CONFIG.copy()
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # fill in missing keys with defaults
    full_cfg = DEFAULT_CONFIG.copy()
    full_cfg.update(cfg or {})
    return full_cfg
