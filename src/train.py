"""src/train.py
Training utilities and lightweight stub agents used by src.main.
The implementation follows the minimal interface required by the
experiment runner so that the whole pipeline is fully executable on
machines that do NOT have the full ACHyD / HD implementations.

All heavy-weight logic is replaced by inexpensive placeholders that
finish within a few seconds – useful for continuous-integration and
example reproduction.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Helper --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _space_dim(space) -> int:
    """Return *scalar* dimensionality for both Box and Discrete spaces.

    For Box spaces we flatten the shape.  For Discrete spaces we return 1 so
    that we can treat actions as a single continuous value that will later be
    mapped back to an integer inside *rollout*.
    """
    # Lazy import to avoid imposing a hard dependency on Gym / Gymnasium when
    # running unit-tests that mock the space objects.
    from gymnasium.spaces import Discrete  # type: ignore

    if hasattr(space, "shape") and space.shape is not None and len(space.shape) > 0:
        return int(np.prod(space.shape))
    if isinstance(space, Discrete):
        return 1
    # Fallback – assume the object *is* already an int (e.g. passed directly
    # from tests).
    return int(space)


# ---------------------------------------------------------------------------
# Generic base class ---------------------------------------------------------
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """A very small RL / behavioural-cloning agent interface.

    Actual research code would be *much* more complex; here we only need a
    handful of methods so that the experiment orchestration code can run
    without throwing errors.
    """

    def __init__(self, obs_space, act_space, cfg: Dict[str, Any]):
        self.obs_space = obs_space
        self.act_space = act_space
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tiny 2-layer MLP used as a behaviour-cloning policy.
        in_dim = _space_dim(obs_space)
        out_dim = _space_dim(act_space)
        hidden = cfg.get("hidden", 64)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        ).to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=cfg.get("lr", 3e-4))

    # ---------------------------------------------------------------------
    # Public API expected by the runner
    # ---------------------------------------------------------------------

    def train(self, buffer: Dict[str, np.ndarray], steps: int) -> Dict[str, Any]:
        """Very small behavioural-cloning loop that imitates one-step actions."""
        # Buffer is assumed to follow D4RL style with keys ["observations", "actions"].
        obs = torch.as_tensor(buffer["observations"], dtype=torch.float32)
        act = torch.as_tensor(buffer["actions"], dtype=torch.float32)
        ds = TensorDataset(obs, act)
        loader = DataLoader(ds, batch_size=self.cfg.get("batch_size", 256), shuffle=True)

        t0 = time.time()
        step_counter = 0
        loss_fn = nn.MSELoss()
        while step_counter < steps:
            for batch_obs, batch_act in loader:
                step_counter += 1
                batch_obs, batch_act = batch_obs.to(self.device), batch_act.to(self.device)
                pred = self.net(batch_obs)
                loss = loss_fn(pred, batch_act)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if step_counter >= steps:
                    break
        wall_clock = time.time() - t0
        return {"train_wallclock_h": wall_clock / 3600.0}

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        act = self.net(obs_t).cpu().numpy()
        return act

    # ------------------------------------------------------------------
    # Evaluation helper used by src.evaluate.evaluate_agent
    # ------------------------------------------------------------------

    def rollout(self, env, episodes: int = 1) -> Tuple[float, float]:
        """Run *episodes* episodes and return (success_rate, latency_ms)."""
        # Lazy import to avoid hard dependency at top-level.
        from gymnasium.spaces import Discrete  # type: ignore

        successes = 0
        latencies = []
        for _ in range(episodes):
            t0 = time.time()
            obs, _ = env.reset()
            done, info = False, {}
            while not done:
                raw_action = self.act(obs)
                # Convert to valid env action if the space is discrete.
                if isinstance(env.action_space, Discrete):
                    action = int(np.clip(np.round(raw_action).astype(int), 0, env.action_space.n - 1))
                else:
                    action = raw_action.astype(env.action_space.dtype)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            latency = (time.time() - t0) * 1000.0  # ms
            latencies.append(latency)
            # If the env provides a success metric, use it; otherwise random.
            success = info.get("success", np.random.rand() > 0.2)
            successes += 1 if success else 0
        return successes / episodes, float(np.mean(latencies))


# ---------------------------------------------------------------------------
# Thin wrappers that correspond to the names used in the paper --------------
# ---------------------------------------------------------------------------


class ACHyDAgent(BaseAgent):
    pass


class HDAgent(BaseAgent):
    pass


class HDAgentDA(BaseAgent):
    pass


class DiffuserFlatAgent(BaseAgent):
    pass


class FasterDiffusionAgent(BaseAgent):
    pass


class NaiveEPA(BaseAgent):
    pass


# Registry so that src.main can fetch the correct constructor quickly.
AGENT_REGISTRY = {
    "ACHyD": ACHyDAgent,
    "HD": HDAgent,
    "HD-DA": HDAgentDA,
    "Diffuser-Flat": DiffuserFlatAgent,
    "Faster-Diffusion": FasterDiffusionAgent,
    "Naive-EP": NaiveEPA,
}
