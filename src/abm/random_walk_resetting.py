"""Independent two-dimensional random walk with resetting baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Tuple

import numpy as np

from abm_metrics import summarize_transport


@dataclass
class RWRConfig:
    """RWR inputs; positions are um, time is s, speed is um/s, rates are s^-1."""

    n_agents: int = 500
    duration_s: float = 300.0
    dt_s: float = 0.05
    speed_um_s: float = 20.0
    response_factor_f: float = 0.0
    anisotropy_mode: Literal["isotropic", "axis_weighted"] = "isotropic"
    reset_rate_per_s: float = 0.0
    reset_mode: Literal["position", "orientation"] = "position"
    reset_position_um: Tuple[float, float] = (100.0, -4000.0)
    x_min_um: float = 0.0
    x_max_um: float = 600.0
    y_min_um: float = -8000.0
    y_max_um: float = 0.0
    target_x_um: float = 500.0
    periodic_y: bool = True
    reflect_left: bool = True
    trajectory_sample_count: int = 0
    spatial_bins_x: int = 65
    spatial_bins_y: int = 80
    collect_spatial_occupancy: bool = True
    seed: int = 7


def reset_probability(reset_rate_per_s: float, dt_s: float) -> float:
    """Return Poisson reset probability ``1-exp(-r*dt)``."""
    if reset_rate_per_s < 0.0 or dt_s <= 0.0:
        raise ValueError("reset_rate_per_s must be >= 0 and dt_s must be > 0")
    return float(1.0 - np.exp(-reset_rate_per_s * dt_s))


def axis_direction_probabilities(response_factor_f: float) -> np.ndarray:
    """Return grid-step probabilities for ``(+x, -x, +y, -y)``.

    ``f`` is the normalized peptide blue-light response in [0, 1].  At
    ``f=0`` the grid walk is isotropic.  Increasing ``f`` transfers probability
    mass from the backward direction to the forward direction while leaving
    the two transverse directions unchanged::

        P(+x)=(1+f)/4, P(-x)=(1-f)/4, P(+y)=P(-y)=1/4.

    This is a deliberately minimal anisotropic baseline; ``f`` is not the RWR
    reset rate ``r`` and does not alter the resetting process.
    """
    f = float(response_factor_f)
    if not np.isfinite(f) or not 0.0 <= f <= 1.0:
        raise ValueError("response_factor_f must be finite and lie in [0, 1]")
    return np.array([(1.0 + f) / 4.0, (1.0 - f) / 4.0, 0.25, 0.25])


def run_random_walk_resetting(config: RWRConfig) -> Dict[str, Any]:
    """Simulate fixed-step 2-D RWR and return shared transport metrics.

    A fresh uniform heading is drawn at every non-position-reset step. Position
    resetting moves an agent exactly to ``reset_position_um``; orientation
    resetting only redraws its heading and never changes its position directly.
    Target/exit first passages are absorbing statistics, matching the ABM rule.
    """
    cfg = config
    if cfg.n_agents <= 0 or cfg.duration_s <= 0.0 or cfg.dt_s <= 0.0 or cfg.speed_um_s <= 0.0:
        raise ValueError("agent count, duration, dt, and speed must be positive")
    if cfg.reset_mode not in {"position", "orientation"}:
        raise ValueError("reset_mode must be 'position' or 'orientation'")
    if cfg.anisotropy_mode not in {"isotropic", "axis_weighted"}:
        raise ValueError("anisotropy_mode must be 'isotropic' or 'axis_weighted'")
    direction_probabilities = axis_direction_probabilities(cfg.response_factor_f)
    if not (cfg.x_min_um <= cfg.reset_position_um[0] <= cfg.x_max_um):
        raise ValueError("reset x position must lie in the simulation domain")
    if not (cfg.y_min_um <= cfg.reset_position_um[1] <= cfg.y_max_um):
        raise ValueError("reset y position must lie in the simulation domain")
    if not (cfg.x_min_um < cfg.target_x_um <= cfg.x_max_um):
        raise ValueError("target_x_um must lie in (x_min_um, x_max_um]")

    rng = np.random.default_rng(cfg.seed)
    x = np.full(cfg.n_agents, cfg.reset_position_um[0], dtype=float)
    y = np.full(cfg.n_agents, cfg.reset_position_um[1], dtype=float)
    x0 = x.copy()
    theta = rng.uniform(-np.pi, np.pi, cfg.n_agents)
    first_passage = np.full(cfg.n_agents, np.nan)
    exited = np.zeros(cfg.n_agents, dtype=bool)
    total_resets = 0
    p_reset = reset_probability(cfg.reset_rate_per_s, cfg.dt_s)
    n_steps = int(np.floor(cfg.duration_s / cfg.dt_s))
    sample_count = min(max(int(cfg.trajectory_sample_count), 0), cfg.n_agents)
    if cfg.spatial_bins_x <= 0 or cfg.spatial_bins_y <= 0:
        raise ValueError("spatial bin counts must be positive")
    spatial_x_edges = np.linspace(cfg.x_min_um, cfg.x_max_um, cfg.spatial_bins_x + 1)
    spatial_y_edges = np.linspace(cfg.y_min_um, cfg.y_max_um, cfg.spatial_bins_y + 1)
    spatial_occupancy = np.zeros((cfg.spatial_bins_y, cfg.spatial_bins_x), dtype=np.float64)
    trajectory_x = np.empty((sample_count, n_steps + 1), dtype=np.float32)
    trajectory_y = np.empty((sample_count, n_steps + 1), dtype=np.float32)
    if sample_count:
        trajectory_x[:, 0] = x[:sample_count]
        trajectory_y[:, 0] = y[:sample_count]

    for step in range(n_steps):
        active = ~exited
        reset = active & (rng.random(cfg.n_agents) < p_reset)
        total_resets += int(reset.sum())
        if cfg.reset_mode == "position" and np.any(reset):
            x[reset] = cfg.reset_position_um[0]
            y[reset] = cfg.reset_position_um[1]

        moving = active & (~reset if cfg.reset_mode == "position" else np.ones(cfg.n_agents, dtype=bool))
        if cfg.anisotropy_mode == "axis_weighted":
            grid_headings = np.array([0.0, np.pi, 0.5 * np.pi, -0.5 * np.pi])
            theta[moving] = rng.choice(
                grid_headings,
                size=int(moving.sum()),
                p=direction_probabilities,
            )
        else:
            theta[moving] = rng.uniform(-np.pi, np.pi, int(moving.sum()))
        if cfg.reset_mode == "orientation" and np.any(reset):
            if cfg.anisotropy_mode == "axis_weighted":
                theta[reset] = rng.choice(
                    np.array([0.0, np.pi, 0.5 * np.pi, -0.5 * np.pi]),
                    size=int(reset.sum()),
                    p=direction_probabilities,
                )
            else:
                theta[reset] = rng.uniform(-np.pi, np.pi, int(reset.sum()))

        x[moving] += cfg.speed_um_s * cfg.dt_s * np.cos(theta[moving])
        y[moving] += cfg.speed_um_s * cfg.dt_s * np.sin(theta[moving])
        if cfg.periodic_y:
            span = cfg.y_max_um - cfg.y_min_um
            y[moving] = ((y[moving] - cfg.y_min_um) % span) + cfg.y_min_um
        else:
            y[moving] = np.clip(y[moving], cfg.y_min_um, cfg.y_max_um)
        under = active & (x < cfg.x_min_um)
        if np.any(under):
            x[under] = 2.0 * cfg.x_min_um - x[under] if cfg.reflect_left else cfg.x_min_um
            if cfg.reflect_left:
                theta[under] = np.pi - theta[under]

        t = (step + 1) * cfg.dt_s
        new_hit = np.isnan(first_passage) & (x >= cfg.target_x_um)
        first_passage[new_hit] = t
        new_exit = active & (x >= cfg.x_max_um)
        exited[new_exit] = True
        x[new_exit] = cfg.x_max_um
        present = ~exited
        if cfg.collect_spatial_occupancy and np.any(present):
            counts, _, _ = np.histogram2d(
                y[present], x[present], bins=[spatial_y_edges, spatial_x_edges]
            )
            spatial_occupancy += counts
        if sample_count:
            trajectory_x[:, step + 1] = x[:sample_count]
            trajectory_y[:, step + 1] = y[:sample_count]

    metrics = summarize_transport(
        first_passage,
        n_agents=cfg.n_agents,
        penetration_count=int(exited.sum()),
        final_x=x,
        initial_x=x0,
        final_theta=theta,
        duration_s=cfg.duration_s,
    )
    metrics.update({
        "model": "RWR" if cfg.reset_rate_per_s > 0.0 else "ordinary_random_walk",
        "reset_rate": cfg.reset_rate_per_s,
        "response_factor_f": cfg.response_factor_f,
        "anisotropy_mode": cfg.anisotropy_mode,
        "direction_probabilities": direction_probabilities.copy(),
        "reset_mode": cfg.reset_mode,
        "reset_events": total_resets,
        "speed": cfg.speed_um_s,
        "n_agents": cfg.n_agents,
        "seed": cfg.seed,
        "config": asdict(cfg),
        "final_x": x.copy(),
        "final_y": y.copy(),
        "final_theta": theta.copy(),
        "trajectory_x": trajectory_x,
        "trajectory_y": trajectory_y,
        "spatial_occupancy": spatial_occupancy,
        "spatial_x_edges": spatial_x_edges,
        "spatial_y_edges": spatial_y_edges,
    })
    return metrics
