"""
Coupling acceleration methods and convergence utilities.

Provides residual norms, state flattening/unflattening, and
acceleration strategies (Aitken, fixed relaxation, IQN-ILS, IQN-IMVJ)
for iterative coupling.

All functions are JAX-traceable pure functions suitable for use
inside ``jax.lax.fori_loop``.
"""

from __future__ import annotations

from typing import Any, Optional

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


def coupling_residual_interface(
    s_new: dict[str, dict],
    s_old: dict[str, dict],
    interface_edges: list,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> jnp.ndarray:
    """Check interface consistency rather than iterate change.

    Computes the difference in interface values (edge source fields)
    between two successive iterations.  Only the fields that appear
    on intra-group edges are compared.

    Parameters
    ----------
    s_new : dict
        New iteration state.
    s_old : dict
        Previous iteration state.
    interface_edges : list of EdgeSpec
        Edges internal to the coupling group.
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
    for edge in interface_edges:
        new_val = s_new[edge.source_node][edge.source_field]
        old_val = s_old[edge.source_node][edge.source_field]
        if edge.transform is not None:
            new_val = edge.transform(new_val)
            old_val = edge.transform(old_val)
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
    fields: Optional[dict[str, tuple[str, ...]]] = None,
) -> jnp.ndarray:
    """Flatten coupled nodes' state fields into a single 1D vector.

    Fields are iterated in sorted order for determinism.

    Parameters
    ----------
    state : dict
        Nested state dict ``{node_name: {field: array}}``.
    node_names : list of str
        Which nodes to include.
    fields : dict or None
        If provided, only include the specified fields per node.
        ``{node_name: (field1, field2, ...)}``
        If None, include all fields.

    Returns
    -------
    jnp.ndarray
        1D vector of all field values concatenated.
    """
    parts = []
    for nn in node_names:
        if fields is not None:
            if nn not in fields:
                continue  # Skip nodes not in the fields dict
            field_list = sorted(fields[nn])
        else:
            field_list = sorted(state[nn].keys())
        for field in field_list:
            parts.append(jnp.ravel(state[nn][field]))
    return jnp.concatenate(parts)


def unflatten_coupled_state(
    flat: jnp.ndarray,
    template: dict[str, dict],
    node_names: list[str],
    fields: Optional[dict[str, tuple[str, ...]]] = None,
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
    fields : dict or None
        If provided, only these fields per node are in the flat vector.

    Returns
    -------
    dict
        Nested state dict with restored shapes.
    """
    result: dict[str, dict[str, Any]] = {}
    offset = 0
    for nn in node_names:
        if fields is not None:
            if nn not in fields:
                continue  # Skip nodes not in the fields dict
            field_list = sorted(fields[nn])
        else:
            field_list = sorted(template[nn].keys())
        result[nn] = {}
        for field in field_list:
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


def iqn_ils_update(
    x_raw_flat: jnp.ndarray,
    x_old_flat: jnp.ndarray,
    prev_residual: jnp.ndarray,
    prev_state: jnp.ndarray,
    V_mat: jnp.ndarray,
    W_mat: jnp.ndarray,
    n_cols: jnp.ndarray,
    omega: jnp.ndarray,
    prev_r_aitken: jnp.ndarray,
) -> tuple:
    """IQN-ILS quasi-Newton update with Aitken fallback.

    Builds a low-rank approximation of the inverse Jacobian from
    residual and state differences across iterations.  Falls back
    to Aitken relaxation when the quasi-Newton update is invalid
    (first iteration, NaN, or excessively large step).

    Parameters
    ----------
    x_raw_flat : jnp.ndarray
        Raw fixed-point result (flattened), shape ``(n_dof,)``.
    x_old_flat : jnp.ndarray
        Previous iteration state (flattened), shape ``(n_dof,)``.
    prev_residual : jnp.ndarray
        Residual from the previous iteration, shape ``(n_dof,)``.
    prev_state : jnp.ndarray
        State from the previous iteration, shape ``(n_dof,)``.
    V_mat : jnp.ndarray
        Pre-allocated residual difference matrix, shape
        ``(n_dof, max_cols)``.
    W_mat : jnp.ndarray
        Pre-allocated state difference matrix, shape
        ``(n_dof, max_cols)``.
    n_cols : jnp.ndarray
        Number of active columns (int32 scalar).
    omega : jnp.ndarray
        Aitken relaxation factor (for fallback).
    prev_r_aitken : jnp.ndarray
        Previous Aitken residual (for fallback), shape ``(n_dof,)``.

    Returns
    -------
    x_new : jnp.ndarray
        Updated state vector.
    V_mat : jnp.ndarray
        Updated V matrix.
    W_mat : jnp.ndarray
        Updated W matrix.
    n_cols : jnp.ndarray
        Updated active column count.
    residual : jnp.ndarray
        Current residual.
    x_old_flat : jnp.ndarray
        Current state (becomes prev_state next iteration).
    new_omega : jnp.ndarray
        Updated Aitken omega.
    cur_r_aitken : jnp.ndarray
        Current Aitken residual.
    """
    residual = x_raw_flat - x_old_flat
    is_first = n_cols == 0

    # Compute differences for V and W
    delta_r = residual - prev_residual
    delta_x = x_old_flat - prev_state

    # Shift existing columns right, add new column at position 0
    max_cols = V_mat.shape[1]
    new_V = jnp.where(
        is_first, V_mat,
        jnp.roll(V_mat, shift=1, axis=1).at[:, 0].set(delta_r),
    )
    new_W = jnp.where(
        is_first, W_mat,
        jnp.roll(W_mat, shift=1, axis=1).at[:, 0].set(delta_x),
    )
    new_n_cols = jnp.where(
        is_first, jnp.int32(0),
        jnp.minimum(n_cols + 1, max_cols),
    )

    # Mask inactive columns to zero
    col_mask = jnp.arange(max_cols) < new_n_cols
    V_masked = new_V * col_mask[None, :]
    W_masked = new_W * col_mask[None, :]

    # Solve V^T V c = -V^T r via regularized normal equations
    VtV = V_masked.T @ V_masked + 1e-10 * jnp.eye(max_cols)
    Vtr = V_masked.T @ (-residual)
    c = jnp.linalg.solve(VtV, Vtr)

    # QN correction
    correction = W_masked @ c + residual
    x_qn = x_old_flat + correction

    # Aitken fallback
    x_aitken, new_omega, cur_r_aitken = aitken_relaxation(
        x_old_flat, x_raw_flat, prev_r_aitken, omega
    )

    # Validate QN result: must be finite and not excessively large
    correction_norm = jnp.sqrt(jnp.sum(correction ** 2))
    residual_norm = jnp.sqrt(jnp.sum(residual ** 2))
    is_valid = (
        jnp.all(jnp.isfinite(x_qn))
        & (correction_norm < 10.0 * jnp.maximum(residual_norm, 1e-12))
        & ~is_first
    )
    x_new = jnp.where(is_valid, x_qn, x_aitken)

    return (
        x_new, new_V, new_W, new_n_cols,
        residual, x_old_flat,
        new_omega, cur_r_aitken,
    )


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
