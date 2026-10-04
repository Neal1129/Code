"""Experimental fluorescence-to-ABM light-response mapping utilities."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def _as_numeric_array(value: Any, name: str) -> np.ndarray:
    """Convert scalar/array/Series-like input to a finite float array."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or non-finite values")
    return array


def _restore_container(template: Any, values: np.ndarray) -> Any:
    """Restore scalar and pandas Series inputs without requiring pandas."""
    if np.ndim(template) == 0:
        return float(np.asarray(values))
    if template.__class__.__module__.startswith("pandas") and hasattr(template, "index"):
        return template.__class__(values, index=template.index, name=getattr(template, "name", None))
    return np.asarray(values)


def compute_experimental_response(F_dark: Any, F_light: Any) -> Any:
    """Return ``D_exp = F_light/F_dark`` for fluorescence values.

    Parameters are fluorescence intensities in the same arbitrary unit. Scalars,
    NumPy arrays, and pandas Series are supported. All values must be finite and
    every dark fluorescence value must be strictly positive.
    """
    dark = _as_numeric_array(F_dark, "F_dark")
    light = _as_numeric_array(F_light, "F_light")
    try:
        dark, light = np.broadcast_arrays(dark, light)
    except ValueError as exc:
        raise ValueError("F_dark and F_light are not broadcast-compatible") from exc
    if np.any(dark <= 0.0):
        raise ValueError("F_dark must be strictly positive")
    if np.any(light < 0.0):
        warnings.warn("F_light contains negative fluorescence values", RuntimeWarning, stacklevel=2)
    result = light / dark
    template = F_light if np.ndim(F_light) > 0 else F_dark
    return _restore_container(template, result)


def normalize_sequence_response(
    D_exp: Any,
    D_ref: float,
    D_positive: float,
    *,
    return_unclipped: bool = False,
) -> Any:
    """Map experimental response to ``D_tilde`` using fixed anchors.

    ``D_ref`` and ``D_positive`` are dimensionless experimental light/dark
    ratios, and ``D_positive`` must exceed ``D_ref``. Values outside the anchor
    interval are clipped to [0, 1]. Set ``return_unclipped=True`` to receive
    ``(clipped, unclipped)`` for audit logging.
    """
    response = _as_numeric_array(D_exp, "D_exp")
    ref = float(_as_numeric_array(D_ref, "D_ref"))
    positive = float(_as_numeric_array(D_positive, "D_positive"))
    if positive <= ref:
        raise ValueError("D_positive must be greater than D_ref")
    raw = (response - ref) / (positive - ref)
    clipped = np.clip(raw, 0.0, 1.0)
    clipped_out = _restore_container(D_exp, clipped)
    raw_out = _restore_container(D_exp, raw)
    return (clipped_out, raw_out) if return_unclipped else clipped_out


def effective_light_input(D_tilde: Any, I_light: Any, light_gate: Any) -> Any:
    """Return ``u_eff = D_tilde * I_light * light_gate`` (dimensionless).

    Every input must be finite and lie in [0, 1]. Broadcasting is supported and
    the output is guaranteed to lie in [0, 1] up to floating-point roundoff.
    """
    response = _as_numeric_array(D_tilde, "D_tilde")
    intensity = _as_numeric_array(I_light, "I_light")
    gate = _as_numeric_array(light_gate, "light_gate")
    for name, value in (("D_tilde", response), ("I_light", intensity), ("light_gate", gate)):
        if np.any((value < 0.0) | (value > 1.0)):
            raise ValueError(f"{name} must lie in [0, 1]")
    try:
        result = response * intensity * gate
    except ValueError as exc:
        raise ValueError("D_tilde, I_light, and light_gate are not broadcast-compatible") from exc
    template = next((v for v in (D_tilde, I_light, light_gate) if np.ndim(v) > 0), D_tilde)
    return _restore_container(template, np.clip(result, 0.0, 1.0))

