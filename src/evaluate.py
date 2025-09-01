"""src/evaluate.py
Evaluation utilities – importable by src.main.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import gymnasium as gym

from .train import PolicyNet  # re-use same architecture


def evaluate(env: gym.Env, ckpt_path: Path, episodes: int = 20) -> Dict[str, float]:
    """Run policy for a few episodes and report mean / std return."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    policy = PolicyNet(obs_dim, act_dim).to(device)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.eval()

    rets: List[float] = []
    with torch.no_grad():
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_ret = 0.0
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                logits = policy(obs_t)
                action = torch.argmax(logits).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_ret += reward
            rets.append(ep_ret)
    return {
        "mean_return": float(np.mean(rets)),
        "std_return": float(np.std(rets)),
    }
