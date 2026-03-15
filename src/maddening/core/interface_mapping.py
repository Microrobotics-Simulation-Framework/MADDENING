"""
Spatial interpolation factories for coupling interfaces.

Each factory takes grid/mesh coordinates at construction time,
pre-computes interpolation weights, and returns a JAX-traceable pure
function suitable for use as an ``EdgeSpec.transform``.

All returned transforms have the signature::

    transform(source_values) -> target_values

where ``source_values`` is an array on the source grid and
``target_values`` is the interpolated result on the target grid.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp


def nearest_neighbor_1d(
    source_x: jnp.ndarray,
    target_x: jnp.ndarray,
) -> Callable:
    """Create a nearest-neighbor interpolation from source to target grid.

    Parameters
    ----------
    source_x : array, shape ``(N_src,)``
        Source grid coordinates.
    target_x : array, shape ``(N_tgt,)``
        Target grid coordinates.

    Returns
    -------
    callable
        ``(source_values,) -> target_values`` where
        ``source_values`` has shape ``(..., N_src)`` and
        ``target_values`` has shape ``(..., N_tgt)``.
    """
    source_x = jnp.asarray(source_x)
    target_x = jnp.asarray(target_x)
    # Pre-compute index mapping: for each target point, find nearest source
    indices = jnp.argmin(
        jnp.abs(source_x[:, None] - target_x[None, :]), axis=0
    )

    def transform(source_values):
        return source_values[..., indices]

    return transform


def linear_interpolation_1d(
    source_x: jnp.ndarray,
    target_x: jnp.ndarray,
) -> Callable:
    """Create a linear interpolation from source to target grid.

    Parameters
    ----------
    source_x : array, shape ``(N_src,)``
        Source grid coordinates (must be sorted ascending).
    target_x : array, shape ``(N_tgt,)``
        Target grid coordinates.  Values outside the source range
        are clamped to the boundary values.

    Returns
    -------
    callable
        ``(source_values,) -> target_values``
    """
    source_x = jnp.asarray(source_x)
    target_x = jnp.asarray(target_x)
    n_src = source_x.shape[0]

    # Pre-compute interpolation weights
    idx = jnp.searchsorted(source_x, target_x) - 1
    idx = jnp.clip(idx, 0, n_src - 2)
    dx = source_x[idx + 1] - source_x[idx]
    alpha = (target_x - source_x[idx]) / jnp.where(dx > 0, dx, 1.0)
    alpha = jnp.clip(alpha, 0.0, 1.0)

    def transform(source_values):
        v0 = source_values[..., idx]
        v1 = source_values[..., idx + 1]
        return v0 + alpha * (v1 - v0)

    return transform


def rbf_interpolation(
    source_points: jnp.ndarray,
    target_points: jnp.ndarray,
    epsilon: float = 1.0,
    kernel: str = "gaussian",
) -> Callable:
    """Create an RBF interpolation from source to target points.

    Pre-computes the interpolation matrix ``H`` such that the returned
    transform is simply ``H @ source_values``.

    Parameters
    ----------
    source_points : array, shape ``(N_src, D)``
        Source point coordinates.
    target_points : array, shape ``(N_tgt, D)``
        Target point coordinates.
    epsilon : float
        Shape parameter for the RBF kernel.
    kernel : str
        One of ``"gaussian"``, ``"multiquadric"``,
        ``"inverse_multiquadric"``, ``"thin_plate_spline"``.

    Returns
    -------
    callable
        ``(source_values,) -> target_values`` where
        ``source_values`` has shape ``(N_src,)`` or ``(N_src, C)``
        and ``target_values`` has the corresponding target shape.
    """
    source_points = jnp.asarray(source_points)
    target_points = jnp.asarray(target_points)

    # Pairwise distances
    r_ss = jnp.sqrt(jnp.sum(
        (source_points[:, None, :] - source_points[None, :, :]) ** 2,
        axis=-1,
    ))
    r_ts = jnp.sqrt(jnp.sum(
        (target_points[:, None, :] - source_points[None, :, :]) ** 2,
        axis=-1,
    ))

    def _kernel(r, eps, name):
        if name == "gaussian":
            return jnp.exp(-(eps * r) ** 2)
        elif name == "multiquadric":
            return jnp.sqrt(1.0 + (eps * r) ** 2)
        elif name == "inverse_multiquadric":
            return 1.0 / jnp.sqrt(1.0 + (eps * r) ** 2)
        elif name == "thin_plate_spline":
            # r^2 * log(r), with 0*log(0) = 0
            r_safe = jnp.where(r > 0, r, 1.0)
            return jnp.where(r > 0, r ** 2 * jnp.log(r_safe), 0.0)
        else:
            raise ValueError(f"Unknown kernel: {name}")

    Phi_ss = _kernel(r_ss, epsilon, kernel)
    Phi_ts = _kernel(r_ts, epsilon, kernel)

    # Pre-compute interpolation matrix: H = Phi_ts @ inv(Phi_ss + eps*I)
    Phi_ss_reg = Phi_ss + 1e-8 * jnp.eye(source_points.shape[0])
    H = Phi_ts @ jnp.linalg.inv(Phi_ss_reg)

    def transform(source_values):
        return H @ source_values

    return transform


def conservative_projection_1d(
    source_boundaries: jnp.ndarray,
    target_boundaries: jnp.ndarray,
) -> Callable:
    """Create a conservative (integral-preserving) 1D projection.

    Maps cell-averaged values from a source grid to a target grid
    such that the integral of the field is preserved.

    Parameters
    ----------
    source_boundaries : array, shape ``(N_src + 1,)``
        Cell boundary coordinates of the source grid.
    target_boundaries : array, shape ``(N_tgt + 1,)``
        Cell boundary coordinates of the target grid.

    Returns
    -------
    callable
        ``(source_values,) -> target_values`` where
        ``source_values`` has shape ``(N_src,)`` and
        ``target_values`` has shape ``(N_tgt,)``.
    """
    source_boundaries = jnp.asarray(source_boundaries)
    target_boundaries = jnp.asarray(target_boundaries)
    n_src = source_boundaries.shape[0] - 1
    n_tgt = target_boundaries.shape[0] - 1

    # Build projection matrix P: P[i, j] = overlap(target_i, source_j) / target_width_i
    rows = []
    for i in range(n_tgt):
        tgt_lo = target_boundaries[i]
        tgt_hi = target_boundaries[i + 1]
        tgt_width = tgt_hi - tgt_lo
        row = []
        for j in range(n_src):
            src_lo = source_boundaries[j]
            src_hi = source_boundaries[j + 1]
            overlap_lo = jnp.maximum(tgt_lo, src_lo)
            overlap_hi = jnp.minimum(tgt_hi, src_hi)
            overlap = jnp.maximum(0.0, overlap_hi - overlap_lo)
            row.append(overlap / jnp.where(tgt_width > 0, tgt_width, 1.0))
        rows.append(jnp.stack(row))
    P = jnp.stack(rows)

    def transform(source_values):
        return P @ source_values

    return transform


# ------------------------------------------------------------------
# 2D mapping functions
# ------------------------------------------------------------------

def nearest_neighbor_2d(
    source_points: jnp.ndarray,
    target_points: jnp.ndarray,
) -> Callable:
    """Create a nearest-neighbor interpolation for 2D point clouds.

    Parameters
    ----------
    source_points : array, shape ``(N_src, 2)``
        Source point coordinates in 2D.
    target_points : array, shape ``(N_tgt, 2)``
        Target point coordinates in 2D.

    Returns
    -------
    callable
        ``(source_values,) -> target_values`` where
        ``source_values`` has shape ``(N_src,)`` or ``(N_src, C)``
        and ``target_values`` has the corresponding target shape.
    """
    source_points = jnp.asarray(source_points)
    target_points = jnp.asarray(target_points)

    # Pre-compute index mapping: for each target point, find nearest source
    # Squared distances: (N_tgt, N_src)
    diff = target_points[:, None, :] - source_points[None, :, :]  # (N_tgt, N_src, 2)
    dist_sq = jnp.sum(diff ** 2, axis=-1)  # (N_tgt, N_src)
    indices = jnp.argmin(dist_sq, axis=1)  # (N_tgt,)

    def transform(source_values):
        return source_values[indices]

    return transform


def rbf_interpolation_2d(
    source_points: jnp.ndarray,
    target_points: jnp.ndarray,
    epsilon: float = 1.0,
    kernel: str = "gaussian",
) -> Callable:
    """Create an RBF interpolation for 2D point clouds.

    This is a convenience wrapper around :func:`rbf_interpolation`
    that ensures the input points are treated as 2D.

    Parameters
    ----------
    source_points : array, shape ``(N_src, 2)``
        Source point coordinates in 2D.
    target_points : array, shape ``(N_tgt, 2)``
        Target point coordinates in 2D.
    epsilon : float
        Shape parameter for the RBF kernel.
    kernel : str
        One of ``"gaussian"``, ``"multiquadric"``,
        ``"inverse_multiquadric"``, ``"thin_plate_spline"``.

    Returns
    -------
    callable
        ``(source_values,) -> target_values`` where
        ``source_values`` has shape ``(N_src,)`` or ``(N_src, C)``
        and ``target_values`` has the corresponding target shape.
    """
    source_points = jnp.asarray(source_points)
    target_points = jnp.asarray(target_points)

    if source_points.ndim == 1:
        source_points = source_points.reshape(-1, 1)
    if target_points.ndim == 1:
        target_points = target_points.reshape(-1, 1)

    # Reuse the general RBF interpolation
    return rbf_interpolation(
        source_points, target_points, epsilon=epsilon, kernel=kernel
    )
