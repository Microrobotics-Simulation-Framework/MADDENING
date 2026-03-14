"""
Coupling acceleration methods and convergence utilities.

Provides residual norms, state flattening/unflattening, and
acceleration strategies (Aitken, fixed relaxation, IQN-ILS)
for iterative coupling.

All functions are JAX-traceable pure functions suitable for use
inside ``jax.lax.fori_loop``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


# ------------------------------------------------------------------
# Convergence norms
# ------------------------------------------------------------------

def coupling_residual_l2(
    s_new: dict[str, dict],
    s_old: dict[str, dict],
    node_names: list[str],
) -> jnp.ndarray:
    """Compute the L2 norm of state change between iterations.

    Parameters
    ----------
    s_new : dict
        New iteration state.
    s_old : dict
        Previous iteration state.
    node_names : list of str
        Node names to include in the norm.

    Returns
    -------
    jnp.ndarray
        Scalar L2 norm.
    """
    total = jnp.array(0.0)
    for nn in node_names:
        for field_name in s_new[nn]:
            diff = s_new[nn][field_name] - s_old[nn][field_name]
            total = total + jnp.sum(diff ** 2)
    return jnp.sqrt(total)


def coupling_residual_mixed(
    s_new: dict[str, dict],
    s_old: dict[str, dict],
    node_names: list[str],
    atol: float,
    rtol: float,
) -> jnp.ndarray:
    """Compute a mixed absolute/relative convergence norm.

    Uses the formula::

        err_i = |new_i - old_i| / (atol + rtol * max(|new_i|, |old_i|))

    Returns the RMS norm.  Convergence is achieved when the result
    is <= 1.0.  This is the same pattern used by the adaptive
    timestepping error estimator.

    Parameters
    ----------
    s_new : dict
        New iteration state.
    s_old : dict
        Previous iteration state.
    node_names : list of str
        Node names to include in the norm.
    atol : float
        Absolute tolerance.
    rtol : float
        Relative tolerance.

    Returns
    -------
    jnp.ndarray
        Scalar RMS error norm.  Converged when <= 1.0.
    """
    sum_sq = jnp.array(0.0)
    count = jnp.array(0, dtype=jnp.int32)
    for nn in node_names:
        for field_name in s_new[nn]:
            new_val = s_new[nn][field_name]
            old_val = s_old[nn][field_name]
            diff = jnp.abs(new_val - old_val)
            scale = atol + rtol * jnp.maximum(
                jnp.abs(new_val), jnp.abs(old_val)
            )
            scaled = diff / scale
            sum_sq = sum_sq + jnp.sum(scaled ** 2)
            count = count + scaled.size
    return jnp.sqrt(sum_sq / jnp.maximum(count, 1))


# ------------------------------------------------------------------
# State flattening / unflattening
# ------------------------------------------------------------------

def flatten_coupled_state(
    state: dict[str, dict],
    node_names: list[str],
) -> jnp.ndarray:
    """Flatten coupled nodes' state fields into a single 1D vector.

    Fields are iterated in sorted order for determinism.

    Parameters
    ----------
    state : dict
        Nested state dict ``{node_name: {field: array}}``.
    node_names : list of str
        Which nodes to include.

    Returns
    -------
    jnp.ndarray
        1D vector of all field values concatenated.
    """
    parts = []
    for nn in node_names:
        for field in sorted(state[nn].keys()):
            parts.append(jnp.ravel(state[nn][field]))
    return jnp.concatenate(parts)


def unflatten_coupled_state(
    flat: jnp.ndarray,
    template: dict[str, dict],
    node_names: list[str],
) -> dict[str, dict]:
    """Unflatten a 1D vector back into the nested state dict structure.

    Parameters
    ----------
    flat : jnp.ndarray
        1D vector produced by :func:`flatten_coupled_state`.
    template : dict
        State dict with the correct shapes (used as a template).
    node_names : list of str
        Which nodes were included in the flat vector.

    Returns
    -------
    dict
        Nested state dict with restored shapes.
    """
    result: dict[str, dict[str, Any]] = {}
    offset = 0
    for nn in node_names:
        result[nn] = {}
        for field in sorted(template[nn].keys()):
            shape = template[nn][field].shape
            size = 1
            for s in shape:
                size *= s
            result[nn][field] = flat[offset:offset + size].reshape(shape)
            offset += size
    return result


# ------------------------------------------------------------------
# Acceleration methods
# ------------------------------------------------------------------

def aitken_relaxation(
    x_old_flat: jnp.ndarray,
    x_raw_flat: jnp.ndarray,
    prev_residual_flat: jnp.ndarray,
    omega: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute Aitken delta-squared accelerated update.

    Aitken's method computes an optimal relaxation factor from
    two successive residuals::

        omega_{k+1} = -omega_k * (r_k . (r_{k+1} - r_k))
                      / ||r_{k+1} - r_k||^2

    Parameters
    ----------
    x_old_flat : jnp.ndarray
        Previous iteration state (flattened).
    x_raw_flat : jnp.ndarray
        Raw fixed-point result (flattened).
    prev_residual_flat : jnp.ndarray
        Residual from the previous iteration.
    omega : jnp.ndarray
        Current relaxation factor.

    Returns
    -------
    x_relaxed : jnp.ndarray
        Relaxed state vector.
    new_omega : jnp.ndarray
        Updated relaxation factor.
    residual : jnp.ndarray
        Current residual (for next iteration).
    """
    residual = x_raw_flat - x_old_flat
    delta_r = residual - prev_residual_flat
    denom = jnp.sum(delta_r ** 2)
    # Guard against zero denominator (first iteration or identical residuals)
    safe_denom = jnp.where(denom > 1e-30, denom, jnp.array(1.0))
    new_omega = -omega * jnp.sum(prev_residual_flat * delta_r) / safe_denom
    new_omega = jnp.clip(new_omega, 0.01, 2.0)
    # Fall back to current omega when denominator is too small
    new_omega = jnp.where(denom > 1e-30, new_omega, omega)

    x_relaxed = x_old_flat + new_omega * residual
    return x_relaxed, new_omega, residual


def fixed_relaxation(
    x_old_flat: jnp.ndarray,
    x_raw_flat: jnp.ndarray,
    omega: float,
) -> jnp.ndarray:
    """Apply fixed (constant) under-relaxation.

    Parameters
    ----------
    x_old_flat : jnp.ndarray
        Previous iteration state (flattened).
    x_raw_flat : jnp.ndarray
        Raw fixed-point result (flattened).
    omega : float
        Relaxation factor.  ``omega=1.0`` is no relaxation,
        ``0 < omega < 1`` is under-relaxation,
        ``1 < omega < 2`` is over-relaxation.

    Returns
    -------
    jnp.ndarray
        Relaxed state vector.
    """
    return x_old_flat + omega * (x_raw_flat - x_old_flat)
