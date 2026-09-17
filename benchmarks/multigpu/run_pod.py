#!/usr/bin/env python
"""Pod-side runner for the multi-GPU hardware session.

Runs *on the machine that has the GPUs* (or, with ``--dry-run``, on CPU
virtual devices) and writes one JSON file per goal under ``--out``.  It
never talks to a cloud provider: launching, copying results back and
tearing the pod down are the human's job (see ``README.md`` next to this
file).

Goals (``--goal``)::

    exchange   NCCL ranking of the two unstructured halo-exchange
               transports, ``all_to_all`` vs ``ppermute``, at 1e5-1e6
               cells -- the measurement that decides whether ``ppermute``
               becomes the default ``exchange=`` of
               ``ShardedUnstructuredNode``.       -> exchange.json
    forward    1e6-cell forward run of a ``ShardedUnstructuredNode`` on a
               real unstructured mesh (``--mesh``) or a synthetic one,
               both transports, checked against the unsharded node.
                                                  -> forward.json
    gradient   the real-GPU half of gradient parity: ``jax.grad`` through
               a sharded rollout (both transports) and through the
               Jacobi-preconditioned ``sharded_cg`` against the unsharded
               references.                        -> gradient.json
    all        the three above, in that order.

``--summarise DIR`` reads the JSON files back and prints the ranking
table and the ``ppermute`` vs ``all_to_all`` recommendation.  It does not
import JAX, so it works on a laptop without a usable jaxlib.

Every timed callable receives inputs that were placed on the device mesh
once, with the ``NamedSharding`` the compiled executable expects, outside
the timed region (:func:`place_on_mesh`); the runner refuses to time
anything else, so no per-call reshard from device 0 is charged to a
transport.  Compile time is reported separately (``compile_s``) from the
steady-state timings, on both the sharded and the unsharded side.

The recommendation only counts rows from a real accelerator run (not
``--dry-run``) on at least ``MIN_DECIDING_DEVICES`` (4) devices; with
``--allow-fewer-devices`` (recorded in the JSON) a 2- or 3-device run may
decide.  A single device performs no exchange and never decides.

Sizes default to the hardware: on GPUs the cell counts are
``1e5, 3e5, 1e6``; on CPU (or with ``--dry-run``) they are a few hundred
cells so the whole script proves itself in well under a minute.
``--dry-run`` additionally pins JAX to the CPU backend with four virtual
host devices when no accelerator backend was requested, so::

    python benchmarks/multigpu/run_pod.py --goal all --dry-run --out /tmp/mg
    python benchmarks/multigpu/run_pod.py --summarise /tmp/mg

works on any laptop.  On the pod::

    python benchmarks/multigpu/run_pod.py --goal all --out results/
    python benchmarks/multigpu/run_pod.py --goal forward --mesh helix.npz --out results/

``--mesh`` accepts an ``.npz`` with an ``edges`` array of shape
``(n_edges, 2)`` (cell adjacency, global ids) and an optional
``partition`` array (``(n_cells,)`` device index, e.g. from PyMetis), or
a ``.npy`` holding just the edges.  Without ``partition`` the cells are
partitioned with PyMetis when importable, else by reverse-Cuthill-McKee
ordering (SciPy) cut into contiguous blocks, else by cell id.  A supplied
partition must use exactly ``--n-devices`` non-empty parts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _pre_import_setup(argv: list[str]) -> None:
    """Environment that must be in place before ``import jax``.

    ``--dry-run`` with no accelerator backend requested (``JAX_PLATFORMS``
    unset or ``cpu``) pins the CPU backend and gives it four virtual
    devices unless ``XLA_FLAGS`` already sets a device count.
    """
    if "--dry-run" not in argv:
        return
    platforms = os.environ.get("JAX_PLATFORMS", "").strip().lower()
    if platforms not in ("", "cpu"):
        return
    os.environ["JAX_PLATFORMS"] = "cpu"
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_force_host_platform_device_count" not in flags:
        os.environ["XLA_FLAGS"] = (flags + " --xla_force_host_platform_device_count=4").strip()


_pre_import_setup(sys.argv)

SCHEMA_VERSION = 2
METHODS = ("all_to_all", "ppermute")
MESH_AXIS = "devices"
GPU_CELLS = (100_000, 300_000, 1_000_000)
DRY_RUN_CELLS = (256, 1024)
MIN_DECIDING_DEVICES = 4        # the session's question is "on 4 GPUs"
MIN_DEVICES_WITH_ESCAPE = 2     # --allow-fewer-devices: still needs a real exchange

# JAX and the sharding helpers are imported on first use by
# ``_load_backend()`` (goal runners and ``environment()``), never at
# module scope: ``--summarise`` reads JSON only and ``recommend()`` is
# unit-tested without JAX.  The names are declared here so the rest of
# the module refers to ordinary module globals.
jax = jnp = lax = shard_map = P = NamedSharding = None
create_device_mesh = build_unstructured_partition = exchange_traffic = None
exchange_unstructured = gather_value = partition_value = None
jacobi_preconditioner = sharded_cg = ShardedUnstructuredNode = None
SimulationNode = StaticArray = NeighbourMeanNode = None


def _load_backend() -> None:
    """Import JAX and the MADDENING sharding helpers (idempotent)."""
    global jax, jnp, lax, shard_map, P, NamedSharding
    global create_device_mesh, build_unstructured_partition, exchange_traffic
    global exchange_unstructured, gather_value, partition_value
    global jacobi_preconditioner, sharded_cg, ShardedUnstructuredNode
    global SimulationNode, StaticArray, NeighbourMeanNode
    if jax is not None:
        return
    # Benchmarks share the GPU with nothing else; not preallocating keeps
    # the unsharded reference (device 0) and the sharded run from fighting
    # over memory at 1e6 cells.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax as _jax
    import jax.numpy as _jnp
    from jax import lax as _lax
    from jax import shard_map as _shard_map
    from jax.sharding import NamedSharding as _NamedSharding
    from jax.sharding import PartitionSpec as _P

    from maddening.cloud.multigpu import device_mesh as _dm
    from maddening.cloud.multigpu import halo_unstructured as _hu
    from maddening.cloud.multigpu import iterative_solver as _its
    from maddening.cloud.multigpu import sharded_unstructured as _su
    from maddening.core import node as _node
    from maddening.core import static_data as _sd

    jax, jnp, lax, shard_map, P, NamedSharding = _jax, _jnp, _lax, _shard_map, _P, _NamedSharding
    create_device_mesh = _dm.create_device_mesh
    build_unstructured_partition = _hu.build_unstructured_partition
    exchange_traffic = _hu.exchange_traffic
    exchange_unstructured = _hu.exchange_unstructured
    gather_value = _hu.gather_value
    partition_value = _hu.partition_value
    jacobi_preconditioner = _its.jacobi_preconditioner
    sharded_cg = _its.sharded_cg
    ShardedUnstructuredNode = _su.ShardedUnstructuredNode
    SimulationNode = _node.SimulationNode
    StaticArray = _sd.StaticArray
    NeighbourMeanNode = _make_node_class(SimulationNode, StaticArray, jnp)


# ---------------------------------------------------------------------------
# Environment record
# ---------------------------------------------------------------------------


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _nvidia_smi() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def environment(*, dry_run: bool = False) -> dict:
    """What the numbers were measured on.

    A dry run never shells out to ``nvidia-smi``: its numbers come from
    CPU virtual devices and a GPU listed next to ``platform: cpu`` would
    only mislead the reader.
    """
    _load_backend()
    devices = jax.devices()
    import jaxlib  # noqa: PLC0415

    return {
        "hostname": socket.gethostname(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "platform": devices[0].platform,
        "devices": [str(d) for d in devices],
        "device_kinds": sorted({d.device_kind for d in devices}),
        "n_devices_visible": len(devices),
        "nvidia_smi": "skipped (dry run)" if dry_run else _nvidia_smi(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
        "git_commit": _git_commit(),
    }


def on_gpu() -> bool:
    _load_backend()
    return jax.devices()[0].platform == "gpu"


# ---------------------------------------------------------------------------
# Meshes, neighbour tables, partitions (NumPy only)
# ---------------------------------------------------------------------------


class MeshPartitionError(ValueError):
    """A ``--mesh`` partition that cannot be laid out on ``--n-devices``."""


def ring_edges(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.int32)
    return np.stack([i, (i + 1) % n], axis=1)


def grid_edges(n: int) -> tuple[int, np.ndarray]:
    """4-neighbour lattice with about ``n`` cells (rounded to a square)."""
    side = max(int(round(np.sqrt(n))), 2)
    ids = np.arange(side * side, dtype=np.int32).reshape(side, side)
    horiz = np.stack([ids[:, :-1].ravel(), ids[:, 1:].ravel()], axis=1)
    vert = np.stack([ids[:-1, :].ravel(), ids[1:, :].ravel()], axis=1)
    return side * side, np.concatenate([horiz, vert]).astype(np.int32)


def synthetic_mesh(kind: str, n: int) -> tuple[int, np.ndarray]:
    if kind == "ring":
        return n, ring_edges(n)
    if kind == "grid":
        return grid_edges(n)
    raise ValueError(f"unknown synthetic mesh {kind!r} (ring|grid)")


def load_mesh(path: str) -> tuple[int, np.ndarray, np.ndarray | None]:
    """``(n_cells, edges, partition-or-None)`` from ``.npz`` / ``.npy``."""
    p = Path(path)
    partition = None
    if p.suffix == ".npz":
        with np.load(p) as z:
            edges = np.asarray(z["edges"])
            if "partition" in z.files:
                partition = np.asarray(z["partition"], dtype=np.int32)
    else:
        edges = np.load(p)
    edges = np.asarray(edges, dtype=np.int32)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError(f"{path}: edges must be (n_edges, 2), got {edges.shape}")
    n = int(edges.max()) + 1 if partition is None else int(partition.size)
    return n, edges, partition


def check_file_partition(pa: np.ndarray, n_devices: int) -> np.ndarray:
    """A supplied partition must fill every one of ``n_devices`` shards.

    Fewer non-empty parts than devices would run silently with empty
    shards (and time nothing on them); more parts cannot be placed.
    """
    pa = np.asarray(pa, dtype=np.int32)
    if pa.size and (int(pa.min()) < 0 or int(pa.max()) >= n_devices):
        raise MeshPartitionError(
            f"--mesh partition uses part ids {int(pa.min())}..{int(pa.max())} but "
            f"--n-devices is {n_devices} (valid ids are 0..{n_devices - 1})")
    counts = np.bincount(pa, minlength=n_devices)
    empty = np.flatnonzero(counts == 0).tolist()
    if empty:
        non_empty = int((counts > 0).sum())
        raise MeshPartitionError(
            f"--mesh partition has {non_empty} non-empty part(s) but --n-devices is "
            f"{n_devices}: device(s) {empty} would own no cells; re-partition the mesh "
            f"for {n_devices} devices or pass --n-devices {non_empty}")
    return pa


def neighbour_table(n: int, edges: np.ndarray) -> np.ndarray:
    """``(n, max_degree)`` neighbour ids, rows padded with the cell itself."""
    u = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.int64)
    v = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.int64)
    keep = u != v
    u, v = u[keep], v[keep]
    order = np.argsort(u, kind="stable")
    u, v = u[order], v[order]
    deg = np.bincount(u, minlength=n)
    max_deg = int(deg.max()) if deg.size else 1
    start = np.zeros(n, dtype=np.int64)
    start[1:] = np.cumsum(deg)[:-1]
    pos = np.arange(u.size) - start[u]
    tbl = np.repeat(np.arange(n, dtype=np.int32)[:, None], max(max_deg, 1), axis=1)
    tbl[u, pos] = v
    return tbl


def partition_cells(n: int, edges: np.ndarray, n_devices: int, how: str) -> tuple[np.ndarray, str]:
    """Global cell -> device index.  Returns ``(assignment, method_used)``."""
    if n_devices == 1:
        return np.zeros(n, dtype=np.int32), "single"
    if how in ("auto", "metis"):
        try:
            import pymetis  # noqa: PLC0415

            tbl = neighbour_table(n, edges)
            adjacency = [row[row != i] for i, row in enumerate(tbl)]
            _, parts = pymetis.part_graph(n_devices, adjacency=adjacency)
            return np.asarray(parts, dtype=np.int32), "metis"
        except ImportError:
            if how == "metis":
                raise
    if how in ("auto", "rcm"):
        try:
            from scipy.sparse import coo_matrix  # noqa: PLC0415
            from scipy.sparse.csgraph import reverse_cuthill_mckee  # noqa: PLC0415

            u, v = edges[:, 0], edges[:, 1]
            a = coo_matrix((np.ones(2 * u.size, np.float32),
                            (np.concatenate([u, v]), np.concatenate([v, u]))), shape=(n, n)).tocsr()
            order = reverse_cuthill_mckee(a, symmetric_mode=True)
            pa = np.empty(n, dtype=np.int32)
            pa[order] = (np.arange(n) * n_devices // n).astype(np.int32)
            return pa, "rcm"
        except ImportError:
            if how == "rcm":
                raise
    return (np.arange(n) * n_devices // n).astype(np.int32), "contiguous"


def slab_table(layout, tbl_global: np.ndarray) -> np.ndarray:
    """Per-cell neighbour table in *slab* indices of the owning shard.

    Row ``g`` holds, for each neighbour of global cell ``g``, its position
    in the ``(n_local_max + n_ghost_max)`` slab that ``pa[g]`` sees after
    the ghost exchange.  Vectorised: a ``(n_devices, n_cells)`` lookup is
    16 MB at 1e6 cells on 4 shards.
    """
    D = layout.n_devices
    n = tbl_global.shape[0]
    slab = np.full((D, n), -1, dtype=np.int32)
    for d in range(D):
        slab[d, layout.local_global_ids[d]] = np.arange(layout.n_local[d], dtype=np.int32)
        slab[d, layout.ghost_global_ids[d]] = (
            layout.n_local_max + np.arange(layout.n_ghost[d], dtype=np.int32))
    out = slab[layout.partition_assignment[:, None], tbl_global]
    if (out < 0).any():
        raise RuntimeError("neighbour not owned or ghosted -- layout/edges mismatch")
    return out


# The file mesh is loaded, and every (mesh, n_devices, method) partition
# computed, once per process: ``--goal all --mesh big.npz`` used to load
# and PyMetis-partition the same 1e6-cell mesh once per goal *and* per
# ``--cells`` entry.
_MESH_CACHE: dict = {}


def _mesh_source(args, n_hint: int):
    """``(n, edges, partition-or-None, source_label)`` for one size."""
    if args.mesh:
        key = ("file", str(args.mesh))
        if key not in _MESH_CACHE:
            _MESH_CACHE[key] = load_mesh(args.mesh)
        n, edges, pa = _MESH_CACHE[key]
        return n, edges, pa, f"file:{args.mesh}"
    n, edges = synthetic_mesh(args.synthetic, n_hint)
    return n, edges, None, args.synthetic


def _sizes(args) -> list[int]:
    """A file mesh has one size; the ``--cells`` list only sizes synthetic meshes."""
    return [args.cells[-1]] if args.mesh else list(args.cells)


def _partition_for(args, source: str, n: int, edges: np.ndarray,
                   pa: np.ndarray | None, n_devices: int) -> tuple[np.ndarray, str]:
    """Validated file partition, or a cached computed one."""
    if pa is not None:
        return check_file_partition(pa, n_devices), "file"
    key = ("partition", source, n, n_devices, args.partition)
    if key not in _MESH_CACHE:
        _MESH_CACHE[key] = partition_cells(n, edges, n_devices, args.partition)
    return _MESH_CACHE[key]


# ---------------------------------------------------------------------------
# The node under test
# ---------------------------------------------------------------------------


def _make_node_class(SimulationNode, StaticArray, jnp):  # noqa: N803 -- class factory
    class NeighbourMeanNode(SimulationNode):
        """``x <- (1 - w) x + w * mean_j x[nbr_j]`` over an arbitrary graph.

        The unsharded ``update`` gathers from a global neighbour table; the
        sharded ``update_padded`` gathers from the same table rewritten in
        slab indices (``slab_table``).  Padded neighbour slots point at the
        cell itself, so both paths compute exactly the same arithmetic and
        the sharded result is bit-comparable to the unsharded one.  ``total``
        (sum of ``x`` over owned cells) exercises the ``psum`` path.
        """

        def __init__(self, name: str, n: int, tbl: np.ndarray, *,
                     partition_assignment: np.ndarray | None = None,
                     weight: float = 0.5, timestep: float = 1.0) -> None:
            super().__init__(name=name, timestep=timestep)
            self._n = int(n)
            self._tbl = np.asarray(tbl, dtype=np.int32)
            self._pa = partition_assignment
            self._w = float(weight)

        def state_fields(self) -> list[str]:
            return ["x"]

        def domain_integral_fields(self) -> set[str]:
            return {"total"}

        @property
        def static_data(self) -> dict:
            if self._pa is None:
                return {}
            return {
                "nbr": StaticArray(self._tbl, replication="partition",
                                   partition_assignment=self._pa),
                "own": StaticArray(np.ones(self._n, np.float32), replication="partition",
                                   partition_assignment=self._pa),
            }

        def initial_state(self) -> dict:
            i = np.arange(self._n, dtype=np.float32)
            return {"x": jnp.asarray(np.sin(0.01 * i) + 0.1 * np.cos(0.37 * i))}

        def _relax(self, x, tbl):
            g = jnp.take(x, tbl, axis=0).mean(axis=-1)
            return (1.0 - self._w) * x[: tbl.shape[0]] + self._w * g

        def update(self, state, boundary_inputs, dt):
            new = self._relax(state["x"], jnp.asarray(self._tbl))
            return {"x": new, "total": jnp.sum(new)}

        def update_padded(self, state_padded, boundary_inputs, dt, *, static_padded=None,
                          shard_info=None):
            n_local = shard_info[0][1]
            new = self._relax(state_padded["x"], static_padded["nbr"][:n_local])
            own = static_padded["own"][:n_local]
            return {"x": new, "total": jnp.sum(new * own)}

    return NeighbourMeanNode


def build_pair(n: int, edges: np.ndarray, mesh, pa: np.ndarray, exchange: str):
    """Unsharded reference node, its sharded twin, and the layout."""
    _load_backend()
    layout = build_unstructured_partition(partition_assignment=pa, edges=edges,
                                          n_devices=int(mesh.shape[MESH_AXIS]))
    tbl = neighbour_table(n, edges)
    ref = NeighbourMeanNode("mesh", n, tbl)
    inner = NeighbourMeanNode("mesh", n, slab_table(layout, tbl), partition_assignment=pa)
    return ref, ShardedUnstructuredNode(inner, mesh, layout, exchange=exchange), layout


# ---------------------------------------------------------------------------
# Placement and timing helpers
# ---------------------------------------------------------------------------


def mesh_sharding(mesh):
    """The ``NamedSharding`` every ``shard_map`` here compiles its inputs to."""
    return NamedSharding(mesh, P(MESH_AXIS))


def layout_slab(value, layout) -> np.ndarray:
    """Global-order ``value`` -> ``(D * n_local_max, ...)`` partition-layout host array."""
    per = partition_value(value=np.asarray(value), layout=layout)
    return per.reshape((layout.n_devices * layout.n_local_max,) + per.shape[2:])


def place_on_mesh(value, mesh):
    """One ``device_put`` with the mesh sharding -- done outside every timed region.

    An uncommitted array (``jnp.asarray`` of host data) lives on device 0
    and would be scattered to the mesh on *every* call of a compiled
    ``shard_map``, inside the timed window; that scatter is not exchange.
    """
    return jax.device_put(jnp.asarray(value), mesh_sharding(mesh))


def require_presharded(arrays, mesh, what: str, compiled=None) -> bool:
    """Refuse to time inputs that would be resharded on each call.

    Every array must already carry the mesh sharding; when the
    ``compiled`` executable is given (positional, non-pytree arguments)
    its input shardings must agree too.  Returns ``True`` for the JSON.
    """
    expected = mesh_sharding(mesh)
    for i, a in enumerate(arrays):
        if not a.sharding.is_equivalent_to(expected, a.ndim):
            raise RuntimeError(
                f"{what}: timed input {i} is placed as {a.sharding}, not {expected}; "
                "a timed call would reshard it from device 0 -- place it once with "
                "place_on_mesh() outside the timed region")
    if compiled is not None:
        for i, (a, s) in enumerate(zip(arrays, compiled.input_shardings[0])):
            if not s.is_equivalent_to(a.sharding, a.ndim):
                raise RuntimeError(
                    f"{what}: compiled input sharding {s} differs from the placed "
                    f"input {i} ({a.sharding})")
    return True


def timed(fn, *, warmup: int, repeats: int) -> dict:
    """``fn()`` must return something ``block_until_ready`` can wait on."""
    for _ in range(warmup):
        jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append((time.perf_counter() - t0) * 1e3)
    return {
        "warmup": warmup,
        "repeats": repeats,
        "ms": samples,
        "min_ms": min(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
    }


def compile_seconds(jitted, *args) -> tuple[float, object]:
    """Ahead-of-time compile of ``jitted`` for ``args``; primes the jit cache.

    Returns ``(seconds, compiled)``; the following ``jitted(*args)`` calls
    hit the cache, so compile time never leaks into the steady-state
    timings and never includes an execution.
    """
    t0 = time.perf_counter()
    compiled = jitted.lower(*args).compile()
    return time.perf_counter() - t0, compiled


def _diff(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    max_abs = float(np.max(np.abs(a - b))) if a.size else 0.0
    return {
        "max_abs": max_abs,
        "max_rel": (max_abs / scale) if scale > 0 else max_abs,
        "reference_scale": scale,
        "finite": bool(np.all(np.isfinite(a))),
    }


def _mesh_for(n_devices: int):
    _load_backend()
    return create_device_mesh(shape=(n_devices,))


def _placed_statics(sharded, layout, mesh) -> dict:
    """The wrapper's partitioned static arrays, placed on the mesh once."""
    return {k: place_on_mesh(layout_slab(np.asarray(sa.value), layout), mesh)
            for k, sa in sharded._sharded_static.items()}


# ---------------------------------------------------------------------------
# Goal a: exchange ranking
# ---------------------------------------------------------------------------


def exchange_input(values: np.ndarray, layout, mesh):
    """The exchange benchmark's timed input: partition layout, placed on the mesh."""
    return place_on_mesh(layout_slab(values, layout), mesh)


def run_exchange(args, out: dict) -> dict:
    _load_backend()
    mesh = _mesh_for(args.n_devices)
    D = args.n_devices
    trailing = (args.fields,) if args.fields > 1 else ()
    results = []
    for n in args.cells:
        n, edges = synthetic_mesh(args.synthetic, n)
        pa, how = _partition_for(args, args.synthetic, n, edges, None, D)
        t0 = time.perf_counter()
        layout = build_unstructured_partition(partition_assignment=pa, edges=edges, n_devices=D)
        layout_s = time.perf_counter() - t0
        traffic = exchange_traffic(layout)
        itemsize = 4 * args.fields
        rng = np.random.default_rng(0)
        values = rng.standard_normal((n,) + trailing).astype(np.float32)
        slab = exchange_input(values, layout, mesh)
        entry = {
            "cells": n, "mesh": args.synthetic, "partition": how,
            "n_devices": D, "fields_per_cell": args.fields,
            "n_local_max": layout.n_local_max, "n_ghost_max": layout.n_ghost_max,
            "layout_build_s": layout_s, "traffic_cells_per_shard": traffic,
            "input_sharding": str(slab.sharding), "input_presharded": False,
            "methods": {},
        }
        outputs = {}
        for method in METHODS:
            def local(x, _m=method):
                return exchange_unstructured(x, layout=layout, mesh_axis=MESH_AXIS, method=_m)

            fn = jax.jit(shard_map(local, mesh=mesh, in_specs=P(MESH_AXIS), out_specs=P(MESH_AXIS)))
            compile_s, compiled = compile_seconds(fn, slab)
            entry["input_presharded"] = require_presharded([slab], mesh, f"exchange/{method}",
                                                           compiled=compiled)
            stats = timed(lambda: fn(slab), warmup=args.warmup, repeats=args.repeats)
            outputs[method] = np.asarray(jax.device_get(fn(slab)))
            per_shard_bytes = int(traffic[method]) * itemsize
            entry["methods"][method] = {
                **stats,
                "compile_s": compile_s,
                "bytes_per_shard": per_shard_bytes,
                "bytes_total": per_shard_bytes * D,
                "messages": traffic["ppermute_messages"] if method == "ppermute" else 1,
                "bandwidth_GBps": (per_shard_bytes * D / 1e9) / (stats["min_ms"] / 1e3)
                if stats["min_ms"] > 0 else None,
            }
        entry["bit_identical"] = bool(np.array_equal(outputs["all_to_all"], outputs["ppermute"]))
        a, p = entry["methods"]["all_to_all"], entry["methods"]["ppermute"]
        entry["ppermute_speedup_median"] = (a["median_ms"] / p["median_ms"]) if p["median_ms"] else None
        entry["ppermute_speedup_min"] = (a["min_ms"] / p["min_ms"]) if p["min_ms"] else None
        results.append(entry)
        speedup = entry["ppermute_speedup_median"]
        speedup_txt = f"x{speedup:.2f}" if speedup is not None else "x n/a (zero median)"
        print(f"[exchange] cells={n:>8} {how:<10} a2a {a['median_ms']:8.3f} ms  "
              f"ppermute {p['median_ms']:8.3f} ms  {speedup_txt}  "
              f"identical={entry['bit_identical']}")
    out["results"] = results
    return out


# ---------------------------------------------------------------------------
# Goal b: forward run on a real (or synthetic) mesh
# ---------------------------------------------------------------------------


def run_forward(args, out: dict) -> dict:
    _load_backend()
    mesh = _mesh_for(args.n_devices)
    D = args.n_devices
    results = []
    for n_hint in _sizes(args):
        n, edges, pa, source = _mesh_source(args, n_hint)
        pa, how = _partition_for(args, source, n, edges, pa, D)
        entry = {"cells": n, "mesh": source, "partition": how, "n_devices": D,
                 "steps": args.steps, "methods": {}}
        ref_node = None
        ref_state = None
        for method in METHODS:
            ref_node, sharded, layout = build_pair(n, edges, mesh, pa, method)
            entry["n_local_max"] = layout.n_local_max
            entry["n_ghost_max"] = layout.n_ghost_max
            entry["traffic_cells_per_shard"] = exchange_traffic(layout)
            state = sharded.initial_state()       # placed on the mesh by the wrapper
            dt = 1.0
            dt_j = jnp.asarray(dt)

            # The compiled step alone, with pre-placed statics: this is the
            # number that compares across transports.
            fn = sharded._get_sharded_fn(state, {}, None)
            statics = _placed_statics(sharded, layout, mesh)
            presharded = require_presharded([state["x"], *statics.values()], mesh,
                                            f"forward/{method}")
            compile_s, _ = compile_seconds(fn, state, {}, dt_j, statics, {})

            def device_block(state=state, fn=fn, statics=statics, dt_j=dt_j):
                for _ in range(args.steps):
                    state = {"x": fn(state, {}, dt_j, statics, {})["x"]}
                return state["x"]

            device = timed(device_block, warmup=args.warmup, repeats=args.repeats)

            # The public ``update()`` re-partitions the static arrays on the
            # host every call; its first call is host work only (the
            # executable is already compiled).
            t0 = time.perf_counter()
            jax.block_until_ready(sharded.update(state, {}, dt)["x"])
            wrapper_first_call_s = time.perf_counter() - t0

            def step_block(state=state, sharded=sharded, dt=dt):
                for _ in range(args.steps):
                    state = {"x": sharded.update(state, {}, dt)["x"]}
                return state["x"]

            wrapper = timed(step_block, warmup=args.warmup, repeats=args.repeats)
            # parity against the unsharded node after ``steps`` steps
            final = state
            for _ in range(args.steps):
                final = sharded.update({"x": final["x"]}, {}, dt)
            got = sharded.gather_global(final)
            if ref_state is None:
                ref_state = ref_node.initial_state()
                for _ in range(args.steps):
                    ref_state = ref_node.update({"x": ref_state["x"]}, {}, dt)
            m = {
                "compile_s": compile_s,
                "wrapper_first_call_s": wrapper_first_call_s,
                "input_presharded": presharded,
                "wrapper_step": {**wrapper, "ms_per_step": wrapper["median_ms"] / args.steps},
                "device_step": {**device, "ms_per_step": device["median_ms"] / args.steps},
                "parity_x": _diff(got["x"], np.asarray(jax.device_get(ref_state["x"]))),
                "parity_total": _diff(np.asarray(got["total"]).reshape(-1),
                                      np.asarray(jax.device_get(ref_state["total"])).reshape(-1)),
            }
            entry["methods"][method] = m
            print(f"[forward] cells={n:>8} {method:<10} wrapper {m['wrapper_step']['ms_per_step']:8.3f}"
                  f" ms/step  device {m['device_step']['ms_per_step']:8.3f} ms/step"
                  f"  compile {compile_s:6.2f} s  max|dx|={m['parity_x']['max_abs']:.2e}")
        results.append(entry)
    out["results"] = results
    return out


# ---------------------------------------------------------------------------
# Goal c: gradient parity
# ---------------------------------------------------------------------------


def _laplacian_matvec_unsharded():
    def matvec(x):
        left = jnp.concatenate([jnp.zeros((1,), dtype=x.dtype), x[:-1]])
        right = jnp.concatenate([x[1:], jnp.zeros((1,), dtype=x.dtype)])
        return 2 * x - left - right
    return matvec


def _laplacian_matvec_sharded(mesh):
    D = int(mesh.shape[MESH_AXIS])

    def shard_matvec(x):
        left_ghost = lax.ppermute(x[-1], MESH_AXIS, [(i, (i + 1) % D) for i in range(D)])
        right_ghost = lax.ppermute(x[0], MESH_AXIS, [(i, (i - 1) % D) for i in range(D)])
        idx = lax.axis_index(MESH_AXIS)
        left_ghost = jnp.where(idx == 0, 0.0, left_ghost)
        right_ghost = jnp.where(idx == D - 1, 0.0, right_ghost)
        left = jnp.concatenate([jnp.asarray([left_ghost], dtype=x.dtype), x[:-1]])
        right = jnp.concatenate([x[1:], jnp.asarray([right_ghost], dtype=x.dtype)])
        return 2 * x - left - right

    def matvec(x):
        return shard_map(shard_matvec, mesh=mesh, in_specs=(P(MESH_AXIS),),
                         out_specs=P(MESH_AXIS))(x)
    return matvec


def run_gradient(args, out: dict) -> dict:
    _load_backend()
    mesh = _mesh_for(args.n_devices)
    D = args.n_devices
    results = []
    for n_hint in _sizes(args):
        n, edges, pa, source = _mesh_source(args, n_hint)
        pa, how = _partition_for(args, source, n, edges, pa, D)
        rng = np.random.default_rng(1)
        w_global = rng.standard_normal(n).astype(np.float32)
        entry = {"cells": n, "mesh": source, "partition": how, "n_devices": D,
                 "grad_steps": args.grad_steps, "rollout": {}, "sharded_cg": None}

        # (i) reverse mode through a rollout: jitted grad on both sides,
        # compile time apart, statics placed once (like-for-like).
        ref_node, _, _ = build_pair(n, edges, mesh, pa, "all_to_all")
        w_j = jnp.asarray(w_global)

        def loss_ref(x0, ref_node=ref_node, w_j=w_j):
            st = {"x": x0}
            for _ in range(args.grad_steps):
                st = {"x": ref_node.update(st, {}, 1.0)["x"]}
            return jnp.sum(st["x"] * w_j)

        x0_global = ref_node.initial_state()["x"]
        g_ref_fn = jax.jit(jax.grad(loss_ref))
        ref_compile_s, _ = compile_seconds(g_ref_fn, x0_global)
        g_ref = np.asarray(jax.device_get(g_ref_fn(x0_global)))
        ref_t = timed(lambda: g_ref_fn(x0_global), warmup=args.warmup, repeats=args.repeats)
        entry["rollout"]["unsharded"] = {"grad": ref_t, "compile_s": ref_compile_s}
        for method in METHODS:
            _, sharded, layout = build_pair(n, edges, mesh, pa, method)
            state0 = sharded.initial_state()
            x0 = state0["x"]
            w_layout = place_on_mesh(layout_slab(w_global, layout), mesh)
            step_fn = sharded._get_sharded_fn(state0, {}, None)
            statics = _placed_statics(sharded, layout, mesh)
            dt_j = jnp.asarray(1.0)
            presharded = require_presharded([x0, w_layout, *statics.values()], mesh,
                                            f"gradient/rollout/{method}")

            def loss_sh(x0, step_fn=step_fn, statics=statics, dt_j=dt_j, w_layout=w_layout):
                st = {"x": x0}
                for _ in range(args.grad_steps):
                    st = {"x": step_fn(st, {}, dt_j, statics, {})["x"]}
                return jnp.sum(st["x"] * w_layout)

            g_fn = jax.jit(jax.grad(loss_sh))
            compile_s, _ = compile_seconds(g_fn, x0)
            g_layout = np.asarray(jax.device_get(g_fn(x0)))
            g_sh = gather_value(per_shard=g_layout.reshape(D, layout.n_local_max), layout=layout)
            stats = timed(lambda: g_fn(x0), warmup=args.warmup, repeats=args.repeats)
            entry["rollout"][method] = {"grad": stats, "compile_s": compile_s,
                                        "input_presharded": presharded,
                                        "parity": _diff(g_sh, g_ref)}
            print(f"[gradient] cells={n:>8} rollout {method:<10} grad {stats['median_ms']:8.2f} ms"
                  f" (unsharded {ref_t['median_ms']:8.2f})  compile {compile_s:6.2f} s"
                  f"  max|dg|={entry['rollout'][method]['parity']['max_abs']:.2e}"
                  f"  rel={entry['rollout'][method]['parity']['max_rel']:.2e}")

        # (ii) reverse + forward mode through the Jacobi-preconditioned sharded CG
        n_cg = (n // D) * D
        matvec_ref = _laplacian_matvec_unsharded()
        matvec_sh = _laplacian_matvec_sharded(mesh)
        pc = jacobi_preconditioner(jnp.full((n_cg,), 2.0, jnp.float32))
        kw = dict(max_iters=args.cg_max_iters, rtol=1e-6, atol=1e-8, backend="loop",
                  preconditioner=pc, differentiable=True)
        b_host = np.sin(np.linspace(0.0, 6.0, n_cg, dtype=np.float32)) + np.float32(0.1)
        b = jnp.asarray(b_host)
        b_sh = place_on_mesh(b_host, mesh)

        def cg_loss_sh(bb):
            return jnp.sum(sharded_cg(matvec_sh, bb, mesh=mesh, in_specs=P(MESH_AXIS), **kw).value ** 2)

        def cg_loss_ref(bb):
            return jnp.sum(sharded_cg(matvec_ref, bb, **kw).value ** 2)

        g_sh_fn, g_ref_fn = jax.jit(jax.grad(cg_loss_sh)), jax.jit(jax.grad(cg_loss_ref))
        cg_presharded = require_presharded([b_sh], mesh, "gradient/sharded_cg")
        sh_compile_s, _ = compile_seconds(g_sh_fn, b_sh)
        ref_compile_s, _ = compile_seconds(g_ref_fn, b)
        g_sh = np.asarray(jax.device_get(g_sh_fn(b_sh)))
        g_ref = np.asarray(jax.device_get(g_ref_fn(b)))
        v = jnp.ones_like(b)
        _, t_sh = jax.jvp(lambda bb: sharded_cg(matvec_sh, bb, mesh=mesh, in_specs=P(MESH_AXIS),
                                                **kw).value, (b,), (v,))
        _, t_ref = jax.jvp(lambda bb: sharded_cg(matvec_ref, bb, **kw).value, (b,), (v,))
        entry["sharded_cg"] = {
            "dof": n_cg,
            "input_presharded": cg_presharded,
            "grad_sharded": timed(lambda: g_sh_fn(b_sh), warmup=args.warmup, repeats=args.repeats),
            "grad_unsharded": timed(lambda: g_ref_fn(b), warmup=args.warmup, repeats=args.repeats),
            "compile_s": {"sharded": sh_compile_s, "unsharded": ref_compile_s},
            "grad_parity": _diff(g_sh, g_ref),
            "jvp_parity": _diff(np.asarray(jax.device_get(t_sh)), np.asarray(jax.device_get(t_ref))),
        }
        cg = entry["sharded_cg"]
        print(f"[gradient] dof={n_cg:>8} sharded_cg grad {cg['grad_sharded']['median_ms']:8.2f} ms "
              f"(unsharded {cg['grad_unsharded']['median_ms']:8.2f})  "
              f"rel grad={cg['grad_parity']['max_rel']:.2e} jvp={cg['jvp_parity']['max_rel']:.2e}")
        results.append(entry)
    out["results"] = results
    return out


# ---------------------------------------------------------------------------
# Summary / recommendation (no JAX)
# ---------------------------------------------------------------------------


def _load_results(directory: Path, goal: str) -> list[dict]:
    docs = []
    for path in sorted(directory.glob(f"{goal}*.json")):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("goal") == goal:
            docs.append(doc)
    return docs


def _excluded_because(row: dict, *, min_cells: int, min_devices: int) -> str | None:
    """Why a row does not decide the transport, or ``None`` if it does."""
    if not row["hardware"]:
        return "dry-run / CPU row (does not rank NCCL transports)"
    if row["cells"] < min_cells:
        return f"{row['cells']} cells < {min_cells}"
    if row["n_devices"] < MIN_DEVICES_WITH_ESCAPE:
        return f"n_devices={row['n_devices']}: no exchange happens on one device"
    if row["n_devices"] < min_devices and not row["allow_fewer_devices"]:
        return (f"n_devices={row['n_devices']} < {min_devices} "
                "(re-run with --allow-fewer-devices to let it decide)")
    if row["speedup_median"] is None:
        return "ppermute median is 0 ms (below timer resolution); speedup undefined"
    return None


def recommend(exchange_docs: list[dict], *, min_cells: int = 100_000,
              margin: float = 1.05, min_devices: int = MIN_DECIDING_DEVICES) -> dict:
    """Which transport should be the default.

    A row *decides* only when it comes from a real accelerator run (not
    ``--dry-run``), has ``cells >= min_cells``, ran on ``n_devices >=
    min_devices`` (or ``>= 2`` when that run recorded
    ``allow_fewer_devices``), and has a finite speedup.  ``ppermute`` wins
    when its median exchange time beats ``all_to_all`` by at least
    ``margin`` at every deciding row; ``all_to_all`` keeps the default
    when it is faster at any such row; anything in between is a tie.
    Without a deciding row the result is ``undecided`` and ``reason``
    says what each row lacked.
    """
    rows = []
    for doc in exchange_docs:
        env = doc["environment"]
        hw = env["platform"] == "gpu" and not doc.get("dry_run", False)
        allow_fewer = bool(doc.get("allow_fewer_devices",
                                   doc.get("config", {}).get("allow_fewer_devices", False)))
        for r in doc["results"]:
            a, p = r["methods"]["all_to_all"], r["methods"]["ppermute"]
            row = {
                "cells": r["cells"], "hardware": hw, "n_devices": r["n_devices"],
                "allow_fewer_devices": allow_fewer,
                "device_kinds": env["device_kinds"],
                "a2a_median_ms": a["median_ms"], "ppermute_median_ms": p["median_ms"],
                "a2a_min_ms": a["min_ms"], "ppermute_min_ms": p["min_ms"],
                "a2a_bytes_total": a["bytes_total"], "ppermute_bytes_total": p["bytes_total"],
                "speedup_median": r["ppermute_speedup_median"],
                "bit_identical": r["bit_identical"],
            }
            row["excluded_because"] = _excluded_because(row, min_cells=min_cells,
                                                        min_devices=min_devices)
            row["deciding"] = row["excluded_because"] is None
            rows.append(row)
    deciding = [r for r in rows if r["deciding"]]
    if not deciding:
        reasons = sorted({r["excluded_because"] for r in rows})
        reason = (f"no deciding measurement: needs a real-GPU run (not --dry-run) at "
                  f">= {min_cells} cells on >= {min_devices} devices "
                  f"(>= {MIN_DEVICES_WITH_ESCAPE} with --allow-fewer-devices)")
        if reasons:
            reason += "; rows excluded because: " + "; ".join(reasons)
        return {"rows": rows, "decision": "undecided", "reason": reason}
    speedups = [r["speedup_median"] for r in deciding]
    if all(s >= margin for s in speedups):
        decision = "ppermute"
        reason = (f"ppermute is >= {margin:.2f}x faster (median) at every deciding point "
                  f"(>= {min_cells} cells, real GPUs); make it the default exchange")
    elif any(s < 1.0 for s in speedups):
        decision = "all_to_all"
        reason = "all_to_all is faster at at least one deciding point; keep it the default"
    else:
        decision = "tie"
        reason = (f"ppermute is faster but by less than {margin:.2f}x somewhere; keep "
                  "all_to_all the default (fewer collectives) unless the byte savings matter")
    return {"rows": rows, "decision": decision, "reason": reason,
            "min_speedup": min(speedups), "max_speedup": max(speedups),
            "deciding_rows": len(deciding)}


def _fmt_speedup(s) -> str:
    return f"{s:8.2f}" if s is not None else f"{'n/a':>8}"


def summarise(directory: Path) -> int:
    exchange_docs = _load_results(directory, "exchange")
    forward_docs = _load_results(directory, "forward")
    gradient_docs = _load_results(directory, "gradient")
    if not (exchange_docs or forward_docs or gradient_docs):
        print(f"no exchange/forward/gradient JSON under {directory}")
        return 1
    rec = recommend(exchange_docs)
    if exchange_docs:
        print("Exchange ranking (all_to_all vs ppermute), median / min ms per exchange")
        print(f"{'cells':>9} {'dev':>4} {'hw':>3} {'a2a med':>9} {'ppm med':>9} {'a2a min':>9} "
              f"{'ppm min':>9} {'speedup':>8} {'a2a MB':>8} {'ppm MB':>8} {'same':>5} {'decides':>7}")
        for r in rec["rows"]:
            print(f"{r['cells']:>9} {r['n_devices']:>4} {'gpu' if r['hardware'] else 'cpu':>3} "
                  f"{r['a2a_median_ms']:9.3f} {r['ppermute_median_ms']:9.3f} "
                  f"{r['a2a_min_ms']:9.3f} {r['ppermute_min_ms']:9.3f} "
                  f"{_fmt_speedup(r['speedup_median'])} {r['a2a_bytes_total'] / 1e6:8.2f} "
                  f"{r['ppermute_bytes_total'] / 1e6:8.2f} {'yes' if r['bit_identical'] else 'NO':>5} "
                  f"{'yes' if r['deciding'] else 'no':>7}")
        print(f"Recommendation: {rec['decision']} -- {rec['reason']}")
    if forward_docs:
        print("\nForward run (ms per step; wrapper = public update(), device = compiled step)")
        for doc in forward_docs:
            for r in doc["results"]:
                for method, m in r["methods"].items():
                    print(f"{r['cells']:>9} cells {r['mesh']:<24} {method:<10} "
                          f"wrapper {m['wrapper_step']['ms_per_step']:9.3f}  "
                          f"device {m['device_step']['ms_per_step']:9.3f}  "
                          f"compile {m['compile_s']:6.2f} s  "
                          f"max|dx| {m['parity_x']['max_abs']:.2e}  "
                          f"finite={m['parity_x']['finite']}")
    if gradient_docs:
        print("\nGradient parity (sharded vs unsharded, max-abs / max-rel; jitted grads, compile apart)")
        for doc in gradient_docs:
            for r in doc["results"]:
                ref = r["rollout"]["unsharded"]
                print(f"{r['cells']:>9} cells rollout   {'unsharded':<10} "
                      f"{'-':>8}   {'-':>8}  grad {ref['grad']['median_ms']:8.2f} ms  "
                      f"compile {ref['compile_s']:6.2f} s")
                for method in METHODS:
                    m = r["rollout"][method]
                    p = m["parity"]
                    print(f"{r['cells']:>9} cells rollout   {method:<10} "
                          f"{p['max_abs']:.2e} / {p['max_rel']:.2e}  "
                          f"grad {m['grad']['median_ms']:8.2f} ms  compile {m['compile_s']:6.2f} s")
                cg = r["sharded_cg"]
                print(f"{cg['dof']:>9} dof   sharded_cg grad     {cg['grad_parity']['max_abs']:.2e} / "
                      f"{cg['grad_parity']['max_rel']:.2e}  jvp {cg['jvp_parity']['max_abs']:.2e} / "
                      f"{cg['jvp_parity']['max_rel']:.2e}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goal", choices=("exchange", "forward", "gradient", "all"))
    ap.add_argument("--out", type=Path, help="directory for the JSON results")
    ap.add_argument("--summarise", type=Path, metavar="DIR",
                    help="print the ranking table + recommendation from DIR and exit (no JAX needed)")
    ap.add_argument("--dry-run", action="store_true",
                    help="small sizes on CPU virtual devices; proves the script, ranks nothing")
    ap.add_argument("--cells", type=int, nargs="+",
                    help=f"cell counts (default {GPU_CELLS} on GPU, {DRY_RUN_CELLS} otherwise)")
    ap.add_argument("--n-devices", type=int, help="mesh size (default: min(4, visible devices))")
    ap.add_argument("--allow-fewer-devices", action="store_true",
                    help=f"let a real-GPU exchange run on 2..{MIN_DECIDING_DEVICES - 1} devices "
                         f"decide the transport (default: only >= {MIN_DECIDING_DEVICES} decide); "
                         "recorded in the JSON")
    ap.add_argument("--warmup", type=int, help="untimed calls before timing (default 5; dry-run 1)")
    ap.add_argument("--repeats", type=int, help="timed calls (default 20; dry-run 3)")
    ap.add_argument("--steps", type=int, help="steps per timed block for --goal forward (default 20; dry-run 3)")
    ap.add_argument("--grad-steps", type=int, default=5, help="rollout length differentiated through")
    ap.add_argument("--cg-max-iters", type=int, help="sharded_cg iteration cap (default 3000; dry-run 300)")
    ap.add_argument("--fields", type=int, default=1, help="float32 fields per cell in the exchange payload")
    ap.add_argument("--synthetic", choices=("ring", "grid"), default="grid",
                    help="synthetic mesh when no --mesh is given")
    ap.add_argument("--mesh", help=".npz (edges[, partition]) or .npy edges of a real mesh")
    ap.add_argument("--partition", choices=("auto", "metis", "rcm", "contiguous"), default="auto")
    args = ap.parse_args(argv)
    if args.summarise is None and (args.goal is None or args.out is None):
        ap.error("--goal and --out are required unless --summarise is given")
    return args


def check_exchange_device_count(n_devices: int, *, dry_run: bool,
                                allow_fewer: bool) -> None:
    """Refuse an ``exchange`` run whose result could never decide.

    One device performs no exchange at all; 2-3 devices decide only with
    ``--allow-fewer-devices``.  A dry run may use any count (it never
    decides anyway).
    """
    if dry_run:
        return
    if n_devices < MIN_DEVICES_WITH_ESCAPE:
        raise SystemExit(f"--goal exchange on {n_devices} device performs no exchange and "
                         "cannot rank transports; use --dry-run to prove the script")
    if n_devices < MIN_DECIDING_DEVICES and not allow_fewer:
        raise SystemExit(f"--goal exchange on {n_devices} devices will not decide the "
                         f"transport (the session's question is on {MIN_DECIDING_DEVICES}); "
                         "pass --allow-fewer-devices to record it as deciding anyway")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summarise is not None:
        return summarise(args.summarise)

    _load_backend()
    gpu = on_gpu()
    visible = len(jax.devices())
    if args.n_devices is None:
        args.n_devices = min(MIN_DECIDING_DEVICES, visible)
    if args.n_devices > visible:
        raise SystemExit(f"--n-devices {args.n_devices} but only {visible} device(s) visible")
    small = args.dry_run or not gpu
    if args.cells is None:
        args.cells = list(DRY_RUN_CELLS if small else GPU_CELLS)
    args.warmup = args.warmup if args.warmup is not None else (1 if small else 5)
    args.repeats = args.repeats if args.repeats is not None else (3 if small else 20)
    args.steps = args.steps if args.steps is not None else (3 if small else 20)
    args.cg_max_iters = args.cg_max_iters if args.cg_max_iters is not None else (300 if small else 3000)
    goals = ("exchange", "forward", "gradient") if args.goal == "all" else (args.goal,)
    if "exchange" in goals:
        check_exchange_device_count(args.n_devices, dry_run=args.dry_run,
                                    allow_fewer=args.allow_fewer_devices)

    env = environment(dry_run=args.dry_run)
    print(f"devices: {env['devices']}  jax {env['jax']}  platform {env['platform']}"
          f"{'  [DRY RUN]' if args.dry_run else ''}")
    if not gpu and not args.dry_run:
        print("WARNING: no GPU backend -- these numbers do not rank NCCL transports")

    args.out.mkdir(parents=True, exist_ok=True)
    runners = {"exchange": run_exchange, "forward": run_forward, "gradient": run_gradient}
    for goal in goals:
        doc = {
            "schema_version": SCHEMA_VERSION,
            "goal": goal,
            "dry_run": bool(args.dry_run),
            "allow_fewer_devices": bool(args.allow_fewer_devices),
            "environment": env,
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        }
        t0 = time.perf_counter()
        try:
            runners[goal](args, doc)
        except MeshPartitionError as e:
            raise SystemExit(str(e)) from e
        doc["wall_s"] = time.perf_counter() - t0
        path = args.out / f"{goal}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        print(f"wrote {path} ({doc['wall_s']:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
