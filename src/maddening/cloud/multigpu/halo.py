"""Halo exchange primitive for sharded stencil computations.

Provides :func:`halo_exchange` -- a collective op that, **when called
inside a** :func:`jax.experimental.shard_map.shard_map` **with a named
mesh axis matching the shard layout**, returns a padded copy of the
local array with ghost cells from neighbouring shards filled in along
one or more spatial axes.

This is the building block ``ShardedStencilNode`` uses to feed
:meth:`SimulationNode.update_padded` a halo-aware view of the state.

Boundary modes
--------------
- ``"periodic"`` -- wrap; ghost on shard 0 comes from shard P-1.
- ``"edge"``     -- replicate own edge values (the default; BCs apply
                    in ``update_padded`` after the exchange).
- ``"zero"``     -- zero-fill ghosts at the global boundary.

Differentiability
-----------------
``lax.ppermute`` is differentiable; gradients flow back through halo
exchange.  See ``tests/cloud/multigpu/test_halo.py`` for the audit.
"""

from __future__ import annotations

from typing import Iterable, Union

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh

_BOUNDARY_MODES = ("periodic", "edge", "zero")


def _exchange_axis(
    local: jax.Array,
    *,
    mesh: Mesh,
    mesh_axis: str,
    spatial_axis: int,
    halo: int,
    boundary: str,
) -> jax.Array:
    """Halo exchange on a single (mesh_axis, spatial_axis) pair."""
    if halo == 0:
        return local
    if boundary not in _BOUNDARY_MODES:
        raise ValueError(
            f"halo_exchange: unknown boundary mode {boundary!r}; "
            f"expected one of {_BOUNDARY_MODES}"
        )

    p_size = int(mesh.shape[mesh_axis])

    # Slices we send. ``slice_in_dim`` uses static indices (halo is a
    # Python int), so this is JIT-friendly.
    n_local = local.shape[spatial_axis]
    left_slice = lax.slice_in_dim(local, 0, halo, axis=spatial_axis)
    right_slice = lax.slice_in_dim(
        local, n_local - halo, n_local, axis=spatial_axis
    )

    # ppermute(x, axis_name, perm) -- pair (src, dst) means rank `dst`
    # receives `x` from rank `src`. Forward shift puts rank r-1's right
    # edge into rank r's left halo slot.
    perm_forward = [(s, (s + 1) % p_size) for s in range(p_size)]
    perm_backward = [(s, (s - 1) % p_size) for s in range(p_size)]

    left_halo = lax.ppermute(right_slice, mesh_axis, perm_forward)
    right_halo = lax.ppermute(left_slice, mesh_axis, perm_backward)

    if boundary != "periodic" and p_size > 1:
        rank = lax.axis_index(mesh_axis)
        on_left_global = rank == 0
        on_right_global = rank == p_size - 1

        if boundary == "edge":
            # Replicate own edge at the global boundary.
            left_halo = jnp.where(on_left_global, left_slice, left_halo)
            right_halo = jnp.where(on_right_global, right_slice, right_halo)
        else:  # "zero"
            zeros = jnp.zeros_like(left_halo)
            left_halo = jnp.where(on_left_global, zeros, left_halo)
            right_halo = jnp.where(on_right_global, zeros, right_halo)

    return jnp.concatenate(
        [left_halo, local, right_halo], axis=spatial_axis
    )


def halo_exchange(
    local: jax.Array,
    *,
    mesh: Mesh,
    axes: Iterable[tuple[str, int, int]] | None = None,
    mesh_axis: str | None = None,
    spatial_axis: int | None = None,
    halo: int | None = None,
    boundary: Union[str, dict[str, str]] = "edge",
) -> jax.Array:
    """Exchange halo cells across one or more mesh axes.

    **Must be called inside** :func:`shard_map` whose mesh has every
    ``mesh_axis`` requested here as a named axis.

    Two calling styles
    ------------------
    *Single axis*::

        halo_exchange(local, mesh=mesh,
                       mesh_axis="spatial_y", spatial_axis=1, halo=1)

    *Multiple axes* (pencil)::

        halo_exchange(local, mesh=mesh,
                       axes=[("spatial_y", 1, 1), ("spatial_z", 2, 1)])

    Parameters
    ----------
    local : jax.Array
        Per-shard local array (the value seen inside ``shard_map``).
    mesh : Mesh
        The device mesh (used to read the size of each named axis
        statically; ``mesh`` is a Python object so this is JIT-safe).
    axes : iterable of (mesh_axis, spatial_axis, halo)
        Triples to exchange.  Pencil decomposition supplies one triple
        per sharded spatial axis.
    mesh_axis, spatial_axis, halo : optional single-axis shortcut
        Equivalent to ``axes=[(mesh_axis, spatial_axis, halo)]``.
    boundary : str or dict[str, str]
        Either a single mode applied to all axes, or a per-mesh-axis
        dict.  Modes: ``"periodic"``, ``"edge"`` (default), ``"zero"``.

    Returns
    -------
    jax.Array
        Padded copy of ``local`` with ``halo`` ghost cells prepended and
        appended along each requested spatial axis.
    """
    if axes is None:
        if mesh_axis is None or spatial_axis is None or halo is None:
            raise ValueError(
                "halo_exchange: provide either `axes=` or all of "
                "`mesh_axis=`, `spatial_axis=`, `halo=`."
            )
        axes = [(mesh_axis, int(spatial_axis), int(halo))]
    axes = list(axes)

    if isinstance(boundary, str):
        boundary_map: dict[str, str] = {ma: boundary for ma, _, _ in axes}
    else:
        boundary_map = dict(boundary)

    out = local
    for ma, sa, h in axes:
        out = _exchange_axis(
            out,
            mesh=mesh,
            mesh_axis=ma,
            spatial_axis=sa,
            halo=int(h),
            boundary=boundary_map.get(ma, "edge"),
        )
    return out
