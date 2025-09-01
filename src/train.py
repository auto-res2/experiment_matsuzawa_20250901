"""src/train.py
Train module: implements a very small REINFORCE agent for CartPole-v1.
The function `train` is the public entry-point used by src.main.
All heavy lifting (optimisation, logging, checkpointing) happens here.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gymnasium as gym

# deterministic helper -----------------------------------------------------------------

def set_seed(seed: int | None = None) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)


# small policy network ------------------------------------------------------------------

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # logits
        return self.net(x)


# REINFORCE -----------------------------------------------------------------------------

def _collect_episode(env: gym.Env, policy: PolicyNet, device: torch.device) -> Tuple[List[torch.Tensor], List[torch.Tensor], float]:
    obs, _ = env.reset()
    done = False
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    ep_ret = 0.0
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        logits = policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        rewards.append(reward)
        ep_ret += reward
    returns: List[float] = []
    g = 0.0
    for r in reversed(rewards):
        g = r + 0.99 * g
        returns.insert(0, g)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
    returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-6)
    return log_probs, returns_t, ep_ret


def train(env: gym.Env, cfg: Dict, models_dir: Path) -> Tuple[Path, List[float]]:
    """Train policy; returns (path_to_checkpoint, reward_history)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy = PolicyNet(obs_dim, act_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])

    reward_history: List[float] = []
    start = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        log_probs, returns_t, ep_ret = _collect_episode(env, policy, device)
        loss = -torch.stack([lp * G for lp, G in zip(log_probs, returns_t)]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reward_history.append(ep_ret)
        if ep % cfg["log_every"] == 0:
            elapsed = time.time() - start
            print(f"[TRAIN] Episode {ep:4d}/{cfg['epochs']}  |  return = {ep_ret:6.1f}  |  elapsed {elapsed:5.1f}s", flush=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / "policy_cartpole.pt"
    torch.save(policy.state_dict(), ckpt_path)
    print(f"[TRAIN] finished – checkpoint saved to {ckpt_path.resolve()}")
    return ckpt_path, reward_history
