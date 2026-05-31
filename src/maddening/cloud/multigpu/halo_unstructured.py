"""Sparse halo exchange for unstructured / graph-partitioned sharding.

v0.3.0 §A6 substrate.  Sibling of
:mod:`maddening.cloud.multigpu.halo` (Cartesian, stencil-based) —
they share the same job (deliver neighbour-cell values inside a
``shard_map`` call) but the data structures are different:

* Cartesian: ``halo_width: dict[int, int]`` per spatial axis, ghost
  cells obtained from the two neighbour shards along the axis via
  ``lax.ppermute`` of fixed-size slices.
* Unstructured: an arbitrary graph connectivity, partitioned by
  PyMetis (or equivalent) at setup time.  Each shard owns a contiguous
  block of local cells; ghost cells are the *exterior* of that block —
  cells the shard's local stencil reads but the shard doesn't own.
  The set of ghost cells is in general not the same on each shard.

This module provides:

* :func:`build_unstructured_partition` — given a global connectivity
  table and a partition assignment, compute the per-shard local cell
  count, ghost-cell count, and the index tables needed to dispatch
  the halo exchange.  Called **once at experiment setup time**, on
  the host, outside any jit.  Its result is a small Python dataclass
  (:class:`UnstructuredPartitionLayout`) that gets cached on the
  ``ShardedUnstructuredNode``.
* :func:`exchange_unstructured` — the runtime collective.  Called
  inside ``shard_map``.  Implements ghost-cell fetch via
  ``lax.all_to_all`` on a fixed-shape send/recv index table.

Differentiability: ``lax.all_to_all`` is differentiable; gradients
flow through.

Scope (v0.3.0): toy-mesh sized (up to ~10⁴ cells).  Production-grade
performance (batched send/recv, NCCL fast-path, real-mesh sized) is
v0.4.0 work — see ``plans/MADDENING_v0.3.0_PLAN.md`` §A6 v0.4.0
commitment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


@dataclass(frozen=True)
class UnstructuredPartitionLayout:
    """Per-shard index tables for graph-partitioned sharding.

    All fields are numpy arrays computed **once at experiment setup time**.
    They never get traced through JIT — only the cell-data arrays they
    index into do.

    Attributes
    ----------
    partition_assignment : np.ndarray, shape ``(n_global_cells,)``, int
        ``partition_assignment[g] = device_index``.  By convention the
        partitioned axis is axis 0 of any partitioned ``StaticArray``
        (the "global cell" axis).
    n_devices : int
        Number of shards (= mesh size).
    local_global_ids : list[np.ndarray]
        ``local_global_ids[d]`` is the array of global cell IDs owned
        by device ``d``.  ``local_global_ids[d][l]`` is the global ID
        of device ``d``'s local cell ``l``.
    n_local : list[int]
        ``n_local[d] = len(local_global_ids[d])`` cached for convenience.
    n_local_max : int
        ``max(n_local)``.  All per-shard local-cell arrays are padded
        to this length so ``shard_map`` sees uniform per-shard shape.
    ghost_global_ids : list[np.ndarray]
        ``ghost_global_ids[d]`` is the sorted unique array of global
        cell IDs that device ``d`` needs as ghost values (owned by
        other devices, but read by device ``d``'s stencil).
    n_ghost : list[int]
        ``len(ghost_global_ids[d])``.
    n_ghost_max : int
        ``max(n_ghost)``.  Ghost arrays are padded to this length.
    send_indices : np.ndarray, shape ``(n_devices, n_devices, n_ghost_max)``, int
        ``send_indices[src, dst, k]`` = local index on ``src`` of the
        kth cell ``src`` needs to send to ``dst`` (or 0 if k >=
        actual send count — see ``send_counts`` for the valid range).
        Used inside ``all_to_all`` to pack the outgoing payload.
    recv_local_index : np.ndarray, shape ``(n_devices, n_ghost_max)``, int
        ``recv_local_index[d, g]`` = the local index in device ``d``'s
        ``ghost_global_ids`` of the gth slot it'll receive.
        Effectively a permutation: ``all_to_all`` gives us a
        ``(n_devices, n_per_device)`` payload and we need to scatter
        it into the per-device ghost buffer.
    send_counts : np.ndarray, shape ``(n_devices, n_devices)``, int
        ``send_counts[src, dst]`` = how many real (non-padding) cells
        ``src`` sends to ``dst``.  Used for the per-shard masking
        when ``n_ghost_max`` overshoots a particular shard's actual
        ghost count.

    Notes
    -----
    The contract here is what v0.4.0's production-grade implementation
    will harden.  Index dtypes are int32 (sufficient for up to ~2.1B
    global cells — well beyond any v1.0 use case) and named in the
    same way; v0.4.0 may add a streaming send/recv buffer alongside,
    but cannot rename or restructure these.
    """
    partition_assignment: np.ndarray
    n_devices: int
    local_global_ids: tuple[np.ndarray, ...]
    n_local: tuple[int, ...]
    n_local_max: int
    ghost_global_ids: tuple[np.ndarray, ...]
    n_ghost: tuple[int, ...]
    n_ghost_max: int
    send_indices: np.ndarray
    recv_local_index: np.ndarray
    send_counts: np.ndarray

    def local_index_of(self, device: int, global_id: int) -> int:
        """Look up where global cell ``global_id`` lives on ``device``.

        Useful for the partition-assignment handoff contract documented
        in ``plans/MADDENING_v0.3.0_PLAN.md`` §A6 — the experiment-setup
        code can use this to translate global cell IDs into the local
        layout the node sees.

        Returns
        -------
        int
            Local index (>= 0) if owned; -1 if not on this device.
        """
        ids = self.local_global_ids[device]
        matches = np.where(ids == global_id)[0]
        if len(matches) == 0:
            return -1
        return int(matches[0])


def build_unstructured_partition(
    *,
    partition_assignment: np.ndarray,
    edges: np.ndarray,
    n_devices: int,
) -> UnstructuredPartitionLayout:
    """Compute the per-shard index tables for a graph partition.

    Pure-host, called once at experiment setup.  Does not call into
    JAX (consumes / produces numpy arrays).

    Parameters
    ----------
    partition_assignment : np.ndarray, shape ``(n_global_cells,)``, int
        Mapping from global cell index → device index.  Output of
        PyMetis or an equivalent partitioner.
    edges : np.ndarray, shape ``(n_edges, 2)``, int
        Connectivity table: each row ``(u, v)`` is an undirected edge
        between global cells ``u`` and ``v``.  Used to compute ghost
        cells (cells on one shard that another shard's stencil reads).
        Self-loops are ignored.
    n_devices : int
        Number of shards.  Must be >= max(partition_assignment) + 1.

    Returns
    -------
    UnstructuredPartitionLayout
        Frozen dataclass with all per-shard index tables.

    Raises
    ------
    ValueError
        If inputs are inconsistent (out-of-range partition values,
        edges referencing missing cells, etc.).
    """
    pa = np.asarray(partition_assignment, dtype=np.int32)
    if pa.ndim != 1:
        raise ValueError(
            f"partition_assignment must be 1-D (got shape {pa.shape})"
        )
    if pa.size == 0:
        raise ValueError("partition_assignment must have at least one cell")
    n_global = pa.size
    if int(pa.max()) >= n_devices:
        raise ValueError(
            f"partition_assignment.max()={int(pa.max())} but n_devices="
            f"{n_devices}: partition values must be in [0, n_devices)."
        )
    if int(pa.min()) < 0:
        raise ValueError(
            f"partition_assignment.min()={int(pa.min())}: partition "
            "values must be non-negative."
        )

    edges = np.asarray(edges, dtype=np.int32)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(
            f"edges must have shape (n_edges, 2) (got shape {edges.shape})"
        )
    if edges.size and (edges.max() >= n_global or edges.min() < 0):
        raise ValueError(
            f"edges reference cells outside [0, {n_global}); max="
            f"{int(edges.max())}, min={int(edges.min())}."
        )

    # Build per-device local→global ID lists.
    local_global_ids = []
    for d in range(n_devices):
        ids = np.where(pa == d)[0].astype(np.int32)
        local_global_ids.append(ids)
    n_local = tuple(len(ids) for ids in local_global_ids)
    n_local_max = max(n_local)
    # Build global→(device, local) reverse map.
    global_to_local = np.full(n_global, -1, dtype=np.int32)
    for d, ids in enumerate(local_global_ids):
        global_to_local[ids] = np.arange(len(ids), dtype=np.int32)

    # Compute ghost cells per device.
    ghost_sets: list[set[int]] = [set() for _ in range(n_devices)]
    for u, v in edges:
        if u == v:
            continue
        du = pa[u]
        dv = pa[v]
        if du != dv:
            # u is a ghost on dv's side; v is a ghost on du's side.
            ghost_sets[dv].add(int(u))
            ghost_sets[du].add(int(v))

    ghost_global_ids = tuple(
        np.asarray(sorted(g), dtype=np.int32) for g in ghost_sets
    )
    n_ghost = tuple(len(g) for g in ghost_global_ids)
    n_ghost_max = max(n_ghost) if any(n_ghost) else 0

    # Build send_indices, recv_local_index, send_counts.
    #
    # send_indices[src, dst, k]: local index on `src` of the kth cell
    #   src needs to send to dst.  Reading ghost_global_ids[dst]
    #   restricted to cells owned by src.
    # recv_local_index[d, g]: where in d's ghost buffer the gth
    #   incoming cell from device d's a2a slot lands.
    # send_counts[src, dst]: how many of those k indices are real
    #   (non-padding).
    send_indices = np.zeros((n_devices, n_devices, max(n_ghost_max, 1)),
                            dtype=np.int32)
    recv_local_index = np.zeros((n_devices, max(n_ghost_max, 1)),
                                dtype=np.int32)
    send_counts = np.zeros((n_devices, n_devices), dtype=np.int32)

    # For each receiving device d, scan its ghost list and bucket
    # entries by the source device.
    for d in range(n_devices):
        ghosts_d = ghost_global_ids[d]
        for g_slot, g_id in enumerate(ghosts_d):
            src = int(pa[g_id])
            local_idx_on_src = int(global_to_local[g_id])
            k = send_counts[src, d]
            send_indices[src, d, k] = local_idx_on_src
            recv_local_index[d, g_slot] = g_slot
            send_counts[src, d] = k + 1

    return UnstructuredPartitionLayout(
        partition_assignment=pa.copy(),
        n_devices=n_devices,
        local_global_ids=tuple(local_global_ids),
        n_local=n_local,
        n_local_max=n_local_max,
        ghost_global_ids=ghost_global_ids,
        n_ghost=n_ghost,
        n_ghost_max=n_ghost_max,
        send_indices=send_indices,
        recv_local_index=recv_local_index,
        send_counts=send_counts,
    )


def exchange_unstructured(
    local: jax.Array,
    *,
    layout: UnstructuredPartitionLayout,
    mesh_axis: str,
) -> jax.Array:
    """Sparse halo exchange — fetch each shard's ghost cells via all_to_all.

    Must be called INSIDE ``shard_map`` with ``mesh_axis`` matching the
    sharded axis name.  ``local`` is the per-shard slab of length
    ``layout.n_local_max`` (first axis), with trailing dimensions
    arbitrary.

    Parameters
    ----------
    local : jax.Array, shape ``(n_local_max, ...)``
        The per-shard slab of cell values.
    layout : UnstructuredPartitionLayout
        Pre-computed index tables (built once via
        :func:`build_unstructured_partition`).
    mesh_axis : str
        Name of the mesh axis ``all_to_all`` operates on.

    Returns
    -------
    jax.Array, shape ``(n_local_max + n_ghost_max, ...)``
        The local slab concatenated with the ghost slab (in
        ``layout.ghost_global_ids[<this device>]`` order).  Trailing
        unused ghost slots are zero-filled.
    """
    n_devices = layout.n_devices
    n_ghost_max = layout.n_ghost_max
    # Convert tracing-safe numpy arrays to traced jnp arrays.
    send_indices = jnp.asarray(layout.send_indices)   # (D, D, n_ghost_max)
    send_counts = jnp.asarray(layout.send_counts)     # (D, D)

    src = lax.axis_index(mesh_axis)
    # Pack outgoing payload: for each dst, gather indices send_indices[src, dst, :]
    # from `local`.  The dst dimension becomes the all_to_all axis.
    # send_payload shape: (D, n_ghost_max, *trailing)
    def gather_for_dst(dst):
        idxs = send_indices[src, dst]
        # Mask out the padding entries.
        valid = jnp.arange(n_ghost_max) < send_counts[src, dst]
        gathered = jnp.take(local, idxs, axis=0)
        return jnp.where(
            valid.reshape((-1,) + (1,) * (gathered.ndim - 1)),
            gathered,
            jnp.zeros_like(gathered),
        )

    send_payload = jax.vmap(gather_for_dst)(jnp.arange(n_devices))
    # Now all_to_all over the leading device axis.
    recv = lax.all_to_all(
        send_payload, mesh_axis,
        split_axis=0, concat_axis=0,
    )
    # ``recv`` has shape (D, n_ghost_max, *trailing): recv[src, k, ...]
    # is what `src` sent to us in slot k.  Flatten and arrange so
    # ghost slot g on this device gets the cell from src=pa[ghost_global_ids[g]].
    # Since recv_local_index[d, g] = g (by construction), and the
    # all_to_all returns the same buffer order, we just need to know
    # which (src, k) corresponds to each ghost slot g.
    #
    # The ghost slot order on this device is layout.ghost_global_ids[this].
    # Their owners (the src of the all_to_all transfer) and the per-src
    # slot index are encoded in the layout.  We rebuild the (src, slot)
    # pair per ghost.  This is what makes the all_to_all work even when
    # different devices need different ghost counts.
    #
    # We need a (n_ghost_max, 2) table per-device giving (src_of_slot, k_in_src)
    # so we can gather recv[src_of_slot, k_in_src].
    src_of_slot, k_in_src = _build_ghost_source_table(layout)  # (D, n_ghost_max), (D, n_ghost_max)
    src_of_slot_j = jnp.asarray(src_of_slot)[src]              # (n_ghost_max,)
    k_in_src_j = jnp.asarray(k_in_src)[src]                    # (n_ghost_max,)

    # Gather the ghost values.
    def pick_one(s, k):
        return recv[s, k]

    if recv.ndim == 1:
        # Shouldn't happen — local is (n_local_max, ...) — but guard.
        ghost = jax.vmap(pick_one)(src_of_slot_j, k_in_src_j)
    else:
        # recv is (D, n_ghost_max, *trailing); gather (s,k) along (0,1).
        # Use a fancy index via jnp.take.
        D, K = recv.shape[0], recv.shape[1]
        flat = recv.reshape((D * K,) + recv.shape[2:])
        flat_idx = src_of_slot_j * K + k_in_src_j
        ghost = jnp.take(flat, flat_idx, axis=0)

    # Mask out padding ghost slots (slots beyond this shard's actual ghost count).
    n_ghost_per_dev = jnp.asarray(layout.n_ghost, dtype=jnp.int32)
    my_n_ghost = n_ghost_per_dev[src]
    valid = jnp.arange(n_ghost_max) < my_n_ghost
    valid = valid.reshape((-1,) + (1,) * (ghost.ndim - 1))
    ghost = jnp.where(valid, ghost, jnp.zeros_like(ghost))

    return jnp.concatenate([local, ghost], axis=0)


def _build_ghost_source_table(
    layout: UnstructuredPartitionLayout,
) -> tuple[np.ndarray, np.ndarray]:
    """For each (device, ghost slot), compute (src_device, slot_on_src).

    Result tables are shape ``(n_devices, n_ghost_max)``, int32.
    ``src_of_slot[d, g]`` is the device that sends device ``d``'s gth
    ghost cell; ``k_in_src[d, g]`` is the position within
    ``send_indices[src_of_slot[d, g], d, :]`` corresponding to that
    ghost cell.

    This is a derived table — we cache it as a hidden field on the
    layout via ``object.__setattr__`` to bypass dataclass(frozen=True),
    since it depends only on the layout's other fields.
    """
    cached = getattr(layout, "_ghost_source_table_cache", None)
    if cached is not None:
        return cached

    D = layout.n_devices
    K = max(layout.n_ghost_max, 1)
    pa = layout.partition_assignment
    src_of_slot = np.zeros((D, K), dtype=np.int32)
    k_in_src = np.zeros((D, K), dtype=np.int32)

    # For each device d, we already have the ghost_global_ids in
    # sorted order.  We need to know, for slot g, which (src, k_in_src)
    # pair populated it.  Walk the same construction order used in
    # build_unstructured_partition: for each ghost cell g_id of device
    # d, src=pa[g_id], k_in_src is the per-src running counter.
    per_src_counter = np.zeros((D, D), dtype=np.int32)
    for d in range(D):
        for g_slot, g_id in enumerate(layout.ghost_global_ids[d]):
            src = int(pa[g_id])
            src_of_slot[d, g_slot] = src
            k_in_src[d, g_slot] = per_src_counter[src, d]
            per_src_counter[src, d] += 1

    object.__setattr__(layout, "_ghost_source_table_cache",
                       (src_of_slot, k_in_src))
    return (src_of_slot, k_in_src)


def partition_value(
    *,
    value: np.ndarray,
    layout: UnstructuredPartitionLayout,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Slice a global array into a (n_devices, n_local_max, *) per-shard tensor.

    Pure-host helper for materialising state / static arrays onto a
    sharded mesh.  Trailing slots (per-shard) are padded with
    ``pad_value`` so all shards see uniform per-shard shape.

    Parameters
    ----------
    value : np.ndarray
        First axis is the global cell axis.
    layout : UnstructuredPartitionLayout
        Pre-computed partition layout.
    pad_value : scalar
        Fill value for padding slots beyond a shard's actual cell count.
    """
    value = np.asarray(value)
    trailing = value.shape[1:]
    out = np.full(
        (layout.n_devices, layout.n_local_max) + trailing,
        pad_value, dtype=value.dtype,
    )
    for d, ids in enumerate(layout.local_global_ids):
        out[d, : len(ids)] = value[ids]
    return out


def gather_value(
    *,
    per_shard: np.ndarray,
    layout: UnstructuredPartitionLayout,
) -> np.ndarray:
    """Inverse of :func:`partition_value` — flatten per-shard slab back to global.

    Useful for tests and for the experiment driver to gather sharded
    state for serialization or end-of-step output.

    Parameters
    ----------
    per_shard : np.ndarray, shape ``(n_devices, n_local_max, *)``
        Per-shard slab (e.g. after ``jax.device_get``).
    layout : UnstructuredPartitionLayout
        The same layout passed to :func:`partition_value`.
    """
    per_shard = np.asarray(per_shard)
    trailing = per_shard.shape[2:]
    n_global = layout.partition_assignment.size
    out = np.zeros((n_global,) + trailing, dtype=per_shard.dtype)
    for d, ids in enumerate(layout.local_global_ids):
        out[ids] = per_shard[d, : len(ids)]
    return out


__all__ = [
    "UnstructuredPartitionLayout",
    "build_unstructured_partition",
    "exchange_unstructured",
    "partition_value",
    "gather_value",
]
