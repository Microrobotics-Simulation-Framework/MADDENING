"""
Adaptive timestepping via Richardson extrapolation.

Uses step-doubling: compares a full step at ``dt`` with two half-steps
at ``dt/2`` to estimate the local truncation error without modifying
individual node update functions.

A PI step-size controller adjusts ``dt`` each step to keep the error
within user-specified absolute and relative tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class AdaptiveConfig:
    """Configuration for adaptive timestepping.

    Parameters
    ----------
    dt_initial : float
        Starting timestep.
    atol : float
        Absolute error tolerance.
    rtol : float
        Relative error tolerance.
    dt_min : float
        Minimum allowed timestep.
    dt_max : float
        Maximum allowed timestep.
    safety : float
        Safety factor for step-size controller (< 1).
    max_factor : float
        Maximum factor by which dt can grow per step.
    min_factor : float
        Minimum factor by which dt can shrink per step.
    order : int
        Order of the base method (Euler = 1).  Used by the PI controller.
    """
    dt_initial: float = 0.01
    atol: float = 1e-6
    rtol: float = 1e-3
    dt_min: float = 1e-8
    dt_max: float = 0.1
    safety: float = 0.9
    max_factor: float = 5.0
    min_factor: float = 0.2
    order: int = 1


def _tree_error_norm(state_fine, state_coarse, atol, rtol):
    """Compute the mixed absolute/relative error norm.

    Uses the formula:
        err_i = |fine_i - coarse_i| / (atol + rtol * max(|fine_i|, |coarse_i|))
    Returns the RMS norm over all elements.
    """
    sum_sq = jnp.array(0.0)
    count = jnp.array(0, dtype=jnp.int32)

    def _accumulate(fine, coarse):
        nonlocal sum_sq, count
        diff = jnp.abs(fine - coarse)
        scale = atol + rtol * jnp.maximum(jnp.abs(fine), jnp.abs(coarse))
        scaled = diff / scale
        sum_sq += jnp.sum(scaled ** 2)
        count += scaled.size

    jax.tree.map(_accumulate, state_fine, state_coarse)
    return jnp.sqrt(sum_sq / jnp.maximum(count, 1))


def build_adaptive_step(
    raw_step_fn: Callable,
    config: AdaptiveConfig,
    node_names: list[str],
) -> Callable:
    """Build an adaptive step function using Richardson extrapolation.

    The returned function has signature::

        (state, dt, external_inputs) -> (new_state, dt_next, error, accepted)

    The caller decides whether to accept or retry.

    Parameters
    ----------
    raw_step_fn : callable
        The unjitted graph step function with signature
        ``(full_state, external_inputs) -> full_state``.
        This function uses the *node's own timestep* internally.
        For adaptive stepping we need a dt-parameterised version,
        so we use a wrapper that scales boundary computations.
    config : AdaptiveConfig
        Adaptive stepping parameters.
    node_names : list of str
        Node names (used for error norm computation).
    """
    atol = config.atol
    rtol = config.rtol
    safety = config.safety
    max_factor = config.max_factor
    min_factor = config.min_factor
    order = config.order

    def adaptive_step(state, dt, external_inputs, dt_step_fn):
        """Take one adaptive step.

        Parameters
        ----------
        state : dict
            Full state dict.
        dt : jax array (scalar)
            Current timestep.
        external_inputs : dict
            External inputs.
        dt_step_fn : callable
            ``(state, ext, dt) -> new_state`` parameterised by dt.

        Returns
        -------
        new_state, dt_next, error_norm, accepted
        """
        # Full step at dt
        state_full = dt_step_fn(state, external_inputs, dt)

        # Two half-steps at dt/2
        half_dt = dt / 2.0
        state_half = dt_step_fn(state, external_inputs, half_dt)
        state_half = dt_step_fn(state_half, external_inputs, half_dt)

        # Error estimate: difference between the two approaches
        # Only compare user state (not _meta)
        user_full = {k: v for k, v in state_full.items() if k != "_meta"}
        user_half = {k: v for k, v in state_half.items() if k != "_meta"}
        error_norm = _tree_error_norm(user_half, user_full, atol, rtol)

        # PI controller: dt_new = dt * safety * (1/error)^(1/(order+1))
        # Clamp to [min_factor, max_factor]
        factor = safety * jnp.where(
            error_norm > 0,
            jnp.power(1.0 / error_norm, 1.0 / (order + 1)),
            max_factor,
        )
        factor = jnp.clip(factor, min_factor, max_factor)
        dt_next = jnp.clip(dt * factor, config.dt_min, config.dt_max)

        # Accept if error <= 1 (scaled error)
        accepted = error_norm <= 1.0

        # Use the more accurate result (half-step) when accepted
        new_state = state_half

        return new_state, dt_next, error_norm, accepted

    return adaptive_step
