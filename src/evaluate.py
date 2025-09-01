"""src/evaluate.py
Simple evaluation helper that runs a trained policy for several
episodes and returns the mean reward.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import gymnasium as gym


def evaluate(policy: torch.nn.Module, env_name: str, episodes: int = 20, max_steps: int = 200,
             device: str = "cpu") -> Tuple[float, float]:
    """Run a trained policy for `episodes` episodes and compute metrics.

    Returns
    -------
    mean_reward : float
    std_reward  : float
    """
    env = gym.make(env_name)
    policy.eval()
    rewards = []
    with torch.no_grad():
        for ep in range(episodes):
            obs, _ = env.reset(seed=ep)
            ep_reward = 0.0
            for _ in range(max_steps):
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                logits = policy(obs_t)
                action = torch.argmax(logits, dim=-1).item()
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    break
            rewards.append(ep_reward)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))
