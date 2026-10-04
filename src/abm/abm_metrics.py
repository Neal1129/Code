"""Shared first-passage and transport metrics for ABM/RWR comparisons."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def summarize_transport(
    first_passage_times: Any,
    *,
    n_agents: int,
    penetration_count: int,
    final_x: Any,
    initial_x: Any,
    final_theta: Any,
    trap_events: int = 0,
    duration_s: float,
) -> Dict[str, Any]:
    """Compute common dimensionless/rate metrics from target first passages.

    First-passage times use seconds; x coordinates use micrometres; angles use
    radians; duration uses seconds. Non-hits must be represented by NaN.
    """
    if n_agents <= 0 or duration_s <= 0.0:
        raise ValueError("n_agents and duration_s must be positive")
    times = np.asarray(first_passage_times, dtype=float)
    if times.shape != (n_agents,):
        raise ValueError("first_passage_times must contain one value per agent")
    hit = np.isfinite(times)
    n_hit = int(hit.sum())
    conditional_mfpt = float(np.mean(times[hit])) if n_hit else float("nan")
    hit_rate = n_hit / n_agents
    efficiency = 1.0 / conditional_mfpt if n_hit and conditional_mfpt > 0.0 else float("nan")
    final_x_arr = np.asarray(final_x, dtype=float)
    initial_x_arr = np.asarray(initial_x, dtype=float)
    theta = np.asarray(final_theta, dtype=float)
    if final_x_arr.shape != (n_agents,) or initial_x_arr.shape != (n_agents,) or theta.shape != (n_agents,):
        raise ValueError("position and orientation arrays must contain one value per agent")
    return {
        "first_passage_times": [float(v) if np.isfinite(v) else None for v in times],
        "conditional_MFPT": conditional_mfpt,
        "hit_rate": float(hit_rate),
        "penetration_efficiency": efficiency,
        "target_arrival_rate": float(hit_rate),
        "penetration_rate": float(penetration_count / n_agents),
        "mean_x_displacement": float(np.mean(final_x_arr - initial_x_arr)),
        "directionality": float(np.mean(np.cos(theta))),
        "trap_rate": float(trap_events / (duration_s * n_agents)),
    }

