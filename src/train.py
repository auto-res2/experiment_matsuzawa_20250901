"""src/train.py
Training utilities – implements a tiny REINFORCE agent so that the
whole research pipeline specified in `src.main` can actually run on a
single GPU / CPU machine without the (very large) third-party
hierarchical–diffusion code base.

The goal is NOT to reproduce ACHyD (which is far beyond the scope of
this template) but to provide a fully runnable placeholder that
respects all interface requirements:
  • `train()` returns a trained PyTorch `nn.Module` and a Pandas
    DataFrame with per-episode statistics so that `src.main` can
    generate publication-quality PDF plots.
  • The policy network is deliberately simple (2-layer MLP) so that it
    trains quickly even on CPU while still demonstrating the full data
    flow (pre-process → train → evaluate → visualise).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym


# ------------------------  Model definition  ------------------------ #
class PolicyNet(nn.Module):
    """Small 2-layer MLP with a categorical action head."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # logits
        return self.net(x)


# ----------------------  REINFORCE algorithm  ----------------------- #
@dataclass
class TrainConfig:
    env_name: str = "CartPole-v1"
    total_episodes: int = 500
    max_steps: int = 200
    learning_rate: float = 1e-2
    gamma: float = 0.99
    hidden_size: int = 128
    seed: int = 42
    device: str = "cpu"


class ReinforceAgent:
    """Lightweight REINFORCE implementation."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: TrainConfig):
        self.policy = PolicyNet(obs_dim, act_dim, cfg.hidden_size).to(cfg.device)
        self.opt = optim.Adam(self.policy.parameters(), lr=cfg.learning_rate)
        self.cfg = cfg
        self.device = cfg.device

    def select_action(self, obs: np.ndarray) -> Tuple[int, torch.Tensor]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy(obs_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    def update(self, log_probs: List[torch.Tensor], returns: List[float]):
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        loss = -torch.sum(torch.stack(log_probs) * returns_t)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


# ----------------------------  Train  ------------------------------- #

def train(cfg_dict: Dict[str, Any]) -> Tuple[nn.Module, pd.DataFrame]:
    """Main training loop.

    Parameters
    ----------
    cfg_dict : Dict[str, Any]
        Parsed YAML configuration.

    Returns
    -------
    model : torch.nn.Module
        The trained policy network.
    stats : pd.DataFrame
        Per-episode reward statistics for plotting.
    """
    cfg = TrainConfig(**cfg_dict)  # type: ignore[arg-type]

    # --- Seed everything for reproducibility --- #
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    env = gym.make(cfg.env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n  # only discrete envs supported in this tiny demo

    agent = ReinforceAgent(obs_dim, act_dim, cfg)

    episode_rewards = []

    for ep in range(cfg.total_episodes):
        obs, _ = env.reset(seed=cfg.seed + ep)
        log_probs, rewards = [], []
        ep_reward = 0.0
        for step in range(cfg.max_steps):
            action, log_prob = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            ep_reward += reward
            obs = next_obs
            if terminated or truncated:
                break
        # compute discounted returns (future rewards)
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + cfg.gamma * G
            returns.insert(0, G)
        # normalise returns for stability
        returns = (np.array(returns) - np.mean(returns)) / (np.std(returns) + 1e-8)
        agent.update(log_probs, list(returns))
        episode_rewards.append(ep_reward)
        if (ep + 1) % 50 == 0:
            print(f"[TRAIN] Episode {ep + 1:4d} | reward = {ep_reward:6.2f}")

    env.close()
    stats = pd.DataFrame({"episode": np.arange(1, cfg.total_episodes + 1),
                          "reward": episode_rewards})
    return agent.policy, stats
