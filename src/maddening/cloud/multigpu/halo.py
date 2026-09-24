"""Halo exchange primitive for sharded stencil computations.

Provides :func:`halo_exchange` -- a collective op that, **when called
inside a** :func:`jax.shard_map.shard_map` **with a named
mesh axis matching the shard layout**, returns a padded copy of the
local array with ghost cells from neighbouring shards filled in along
one or more spatial axes.

This is the building block ``ShardedStencilNode`` uses to feed
:meth:`SimulationNode.update_padded` a halo-aware view of the state.

Boundary modes
--------------
Halos between two shards always hold the neighbouring shard's cells.
The mode decides only the halos at the two edges of the *global* grid,
for a halo ``h`` cells wide and a global row ``r0, r1, ..., r(n-1)``:

- ``"periodic"`` -- wrap: the left halo is ``r(n-h), ..., r(n-1)`` and
                    the right halo ``r0, ..., r(h-1)``.
- ``"edge"``     -- replicate the outermost cell into every halo cell:
                    ``r0`` repeated ``h`` times on the left, ``r(n-1)``
                    repeated ``h`` times on the right (``numpy.pad``'s
                    ``mode="edge"``).  The default; a node applies its
                    physical boundary conditions in ``update_padded``
                    after the exchange.
- ``"zero"``     -- zero-fill ghosts at the global boundary.

These hold for every mesh-axis size, one device included: a mesh axis
of size 1 has a single shard that owns both global edges.

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


def _global_edge_halos(
    left_slice: jax.Array,
    right_slice: jax.Array,
    *,
    spatial_axis: int,
    halo: int,
    boundary: str,
) -> tuple[jax.Array, jax.Array]:
    """The ``(left, right)`` halos at the edges of the global grid.

    ``left_slice`` and ``right_slice`` are a block's first and last
    ``halo`` cells along ``spatial_axis`` -- the block being the shard
    that owns that global edge (or, for an axis that is not sharded, the
    whole axis).  Shared by :func:`halo_exchange` and the local padding
    of unsharded halo axes in
    :class:`~maddening.cloud.multigpu.sharded_node.ShardedStencilNode`,
    so the two cannot fill differently.

    ``"periodic"`` wraps the block onto itself, which is right only when
    the block is the whole axis; :func:`halo_exchange` fetches periodic
    halos from the opposite shard instead of calling this.
    """
    if boundary == "periodic":
        return right_slice, left_slice
    if boundary == "zero":
        zeros = jnp.zeros_like(left_slice)
        return zeros, zeros
    if boundary != "edge":
        raise ValueError(
            f"halo_exchange: unknown boundary mode {boundary!r}; "
            f"expected one of {_BOUNDARY_MODES}"
        )
    # "edge": every halo cell is the outermost cell (numpy.pad "edge").
    # At halo == 1 the slices *are* the outermost cells.  Before 0.4.0
    # the slices were used at every width, which put ``r0, r1`` before
    # ``r0`` -- a copy of the block, not a replication of its edge.
    if halo == 1:
        return left_slice, right_slice
    first = lax.slice_in_dim(left_slice, 0, 1, axis=spatial_axis)
    last = lax.slice_in_dim(right_slice, halo - 1, halo, axis=spatial_axis)
    return (
        jnp.broadcast_to(first, left_slice.shape),
        jnp.broadcast_to(last, right_slice.shape),
    )


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

    if boundary != "periodic" and p_size == 1:
        # One shard on this mesh axis: it is both the left and the right
        # global boundary, so both halos are the boundary fill and there
        # is nothing to exchange.  Before 0.4.0 this case went through
        # the ppermute below, which on one device hands the shard its
        # own opposite edge -- periodic halos whatever ``boundary`` said.
        left_halo, right_halo = _global_edge_halos(
            left_slice, right_slice,
            spatial_axis=spatial_axis, halo=halo, boundary=boundary,
        )
        return jnp.concatenate(
            [left_halo, local, right_halo], axis=spatial_axis
        )

    # ppermute(x, axis_name, perm) -- pair (src, dst) means rank `dst`
    # receives `x` from rank `src`. Forward shift puts rank r-1's right
    # edge into rank r's left halo slot.
    perm_forward = [(s, (s + 1) % p_size) for s in range(p_size)]
    perm_backward = [(s, (s - 1) % p_size) for s in range(p_size)]

    left_halo = lax.ppermute(right_slice, mesh_axis, perm_forward)
    right_halo = lax.ppermute(left_slice, mesh_axis, perm_backward)

    if boundary != "periodic":
        rank = lax.axis_index(mesh_axis)
        on_left_global = rank == 0
        on_right_global = rank == p_size - 1
        left_fill, right_fill = _global_edge_halos(
            left_slice, right_slice,
            spatial_axis=spatial_axis, halo=halo, boundary=boundary,
        )
        left_halo = jnp.where(on_left_global, left_fill, left_halo)
        right_halo = jnp.where(on_right_global, right_fill, right_halo)

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
        dict (an axis it does not name gets ``"edge"``).  The mode fills
        only the halos at the two edges of the global grid; a halo
        between two shards always holds the neighbouring shard's cells.
        For a halo ``h`` cells wide along a global row ``r0, ...,
        r(n-1)``:

        - ``"periodic"``: left halo ``r(n-h), ..., r(n-1)``, right halo
          ``r0, ..., r(h-1)``.
        - ``"edge"`` (default): left halo ``r0`` repeated ``h`` times,
          right halo ``r(n-1)`` repeated ``h`` times -- ``numpy.pad``'s
          ``mode="edge"``.  At ``h == 1`` this is also the mirror image
          of the edge cell.
        - ``"zero"``: ``h`` zeros on each side.

        Every mode means the same on a mesh axis of any size; with one
        device on an axis, that device's halos along it are both global
        halos.

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
