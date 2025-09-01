"""src/evaluate.py
Simple evaluation helpers that rely on the Agent interface defined in
src.train.BaseAgent.
"""
from __future__ import annotations

import numpy as np
from typing import Any, Dict

from .train import BaseAgent


def evaluate_agent(agent: BaseAgent, env, episodes: int = 20) -> Dict[str, Any]:
    """Roll out *episodes* episodes and return success / latency statistics."""
    success_rate, latency_ms = agent.rollout(env, episodes)
    # Energy measurement is mocked with a random number – no GPU counters.
    energy_kj = float(np.random.uniform(0.1, 0.5))

    return {
        "success": success_rate,
        "inference_latency_ms": latency_ms,
        "energy_kJ": energy_kj,
    }
