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
table and the ``ppermute`` vs ``all_to_all`` recommendation.

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
ordering (SciPy) cut into contiguous blocks, else by cell id.
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

# Benchmarks share the GPU with nothing else; not preallocating keeps
# the unsharded reference (device 0) and the sharded run from fighting
# over memory at 1e6 cells.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax, shard_map  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from maddening.cloud.multigpu.device_mesh import create_device_mesh  # noqa: E402
from maddening.cloud.multigpu.halo_unstructured import (  # noqa: E402
    build_unstructured_partition,
    exchange_traffic,
    exchange_unstructured,
    gather_value,
    partition_value,
)
from maddening.cloud.multigpu.iterative_solver import (  # noqa: E402
    jacobi_preconditioner,
    sharded_cg,
)
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode  # noqa: E402
from maddening.core.node import SimulationNode  # noqa: E402
from maddening.core.static_data import StaticArray  # noqa: E402

SCHEMA_VERSION = 1
METHODS = ("all_to_all", "ppermute")
GPU_CELLS = (100_000, 300_000, 1_000_000)
DRY_RUN_CELLS = (256, 1024)


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


def environment() -> dict:
    """What the numbers were measured on."""
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
        "nvidia_smi": _nvidia_smi(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
        "git_commit": _git_commit(),
    }


def on_gpu() -> bool:
    return jax.devices()[0].platform == "gpu"


# ---------------------------------------------------------------------------
# Meshes, neighbour tables, partitions
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# The node under test
# ---------------------------------------------------------------------------


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


def build_pair(n: int, edges: np.ndarray, mesh, pa: np.ndarray, exchange: str):
    """Unsharded reference node, its sharded twin, and the layout."""
    layout = build_unstructured_partition(partition_assignment=pa, edges=edges,
                                          n_devices=int(mesh.shape["devices"]))
    tbl = neighbour_table(n, edges)
    ref = NeighbourMeanNode("mesh", n, tbl)
    inner = NeighbourMeanNode("mesh", n, slab_table(layout, tbl), partition_assignment=pa)
    return ref, ShardedUnstructuredNode(inner, mesh, layout, exchange=exchange), layout


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


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
    return create_device_mesh(shape=(n_devices,))


# ---------------------------------------------------------------------------
# Goal a: exchange ranking
# ---------------------------------------------------------------------------


def run_exchange(args, out: dict) -> dict:
    mesh = _mesh_for(args.n_devices)
    D = args.n_devices
    trailing = (args.fields,) if args.fields > 1 else ()
    results = []
    for n in args.cells:
        n, edges = synthetic_mesh(args.synthetic, n)
        pa, how = partition_cells(n, edges, D, args.partition)
        t0 = time.perf_counter()
        layout = build_unstructured_partition(partition_assignment=pa, edges=edges, n_devices=D)
        layout_s = time.perf_counter() - t0
        traffic = exchange_traffic(layout)
        itemsize = 4 * args.fields
        rng = np.random.default_rng(0)
        values = rng.standard_normal((n,) + trailing).astype(np.float32)
        slab = jnp.asarray(partition_value(value=values, layout=layout)
                           .reshape((D * layout.n_local_max,) + trailing))
        entry = {
            "cells": n, "mesh": args.synthetic, "partition": how,
            "n_devices": D, "fields_per_cell": args.fields,
            "n_local_max": layout.n_local_max, "n_ghost_max": layout.n_ghost_max,
            "layout_build_s": layout_s, "traffic_cells_per_shard": traffic,
            "methods": {},
        }
        outputs = {}
        for method in METHODS:
            def local(x, _m=method):
                return exchange_unstructured(x, layout=layout, mesh_axis="devices", method=_m)

            fn = jax.jit(shard_map(local, mesh=mesh, in_specs=P("devices"), out_specs=P("devices")))
            t0 = time.perf_counter()
            jax.block_until_ready(fn(slab))
            compile_s = time.perf_counter() - t0
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
        print(f"[exchange] cells={n:>8} {how:<10} a2a {a['median_ms']:8.3f} ms  "
              f"ppermute {p['median_ms']:8.3f} ms  x{entry['ppermute_speedup_median']:.2f}  "
              f"identical={entry['bit_identical']}")
    out["results"] = results
    return out


# ---------------------------------------------------------------------------
# Goal b: forward run on a real (or synthetic) mesh
# ---------------------------------------------------------------------------


def _mesh_source(args, n_hint: int):
    if args.mesh:
        n, edges, pa = load_mesh(args.mesh)
        return n, edges, pa, f"file:{args.mesh}"
    n, edges = synthetic_mesh(args.synthetic, n_hint)
    return n, edges, None, args.synthetic


def run_forward(args, out: dict) -> dict:
    mesh = _mesh_for(args.n_devices)
    D = args.n_devices
    results = []
    sizes = [args.cells[-1]] if args.mesh else args.cells
    for n_hint in sizes:
        n, edges, pa, source = _mesh_source(args, n_hint)
        how = "file"
        if pa is None:
            pa, how = partition_cells(n, edges, D, args.partition)
        elif int(pa.max()) >= D:
            raise SystemExit(f"--mesh partition uses {int(pa.max()) + 1} parts but "
                             f"--n-devices is {D}")
        entry = {"cells": n, "mesh": source, "partition": how, "n_devices": D,
                 "steps": args.steps, "methods": {}}
        ref_node = None
        ref_state = None
        for method in METHODS:
            ref_node, sharded, layout = build_pair(n, edges, mesh, pa, method)
            entry["n_local_max"] = layout.n_local_max
            entry["n_ghost_max"] = layout.n_ghost_max
            entry["traffic_cells_per_shard"] = exchange_traffic(layout)
            state = sharded.initial_state()
            dt = 1.0

            def step_block(state=state, sharded=sharded, dt=dt):
                for _ in range(args.steps):
                    state = {"x": sharded.update(state, {}, dt)["x"]}
                return state["x"]

            t0 = time.perf_counter()
            jax.block_until_ready(sharded.update(state, {}, dt)["x"])
            compile_s = time.perf_counter() - t0
            wrapper = timed(step_block, warmup=args.warmup, repeats=args.repeats)
            # The compiled step alone (the wrapper's public ``update`` also
            # re-partitions the static arrays on the host every call).
            device = None
            getter = getattr(sharded, "_get_sharded_fn", None)
            if getter is not None:
                fn = getter(state, {}, None)
                statics = {}
                for k, sa in sharded._sharded_static.items():
                    per = partition_value(value=np.asarray(sa.value), layout=layout)
                    statics[k] = jax.device_put(
                        jnp.asarray(per.reshape((D * layout.n_local_max,) + per.shape[2:])),
                        jax.sharding.NamedSharding(mesh, P("devices")))
                dt_j = jnp.asarray(dt)

                def device_block(state=state):
                    for _ in range(args.steps):
                        state = {"x": fn(state, {}, dt_j, statics, {})["x"]}
                    return state["x"]

                device = timed(device_block, warmup=args.warmup, repeats=args.repeats)
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
                "wrapper_step": {**wrapper, "ms_per_step": wrapper["median_ms"] / args.steps},
                "device_step": ({**device, "ms_per_step": device["median_ms"] / args.steps}
                                if device else None),
                "parity_x": _diff(got["x"], np.asarray(jax.device_get(ref_state["x"]))),
                "parity_total": _diff(np.asarray(got["total"]).reshape(-1),
                                      np.asarray(jax.device_get(ref_state["total"])).reshape(-1)),
            }
            entry["methods"][method] = m
            dev = m["device_step"]["ms_per_step"] if m["device_step"] else float("nan")
            print(f"[forward] cells={n:>8} {method:<10} wrapper {m['wrapper_step']['ms_per_step']:8.3f}"
                  f" ms/step  device {dev:8.3f} ms/step  max|dx|={m['parity_x']['max_abs']:.2e}")
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
    D = int(mesh.shape["devices"])

    def shard_matvec(x):
        left_ghost = lax.ppermute(x[-1], "devices", [(i, (i + 1) % D) for i in range(D)])
        right_ghost = lax.ppermute(x[0], "devices", [(i, (i - 1) % D) for i in range(D)])
        idx = lax.axis_index("devices")
        left_ghost = jnp.where(idx == 0, 0.0, left_ghost)
        right_ghost = jnp.where(idx == D - 1, 0.0, right_ghost)
        left = jnp.concatenate([jnp.asarray([left_ghost], dtype=x.dtype), x[:-1]])
        right = jnp.concatenate([x[1:], jnp.asarray([right_ghost], dtype=x.dtype)])
        return 2 * x - left - right

    def matvec(x):
        return shard_map(shard_matvec, mesh=mesh, in_specs=(P("devices"),),
                         out_specs=P("devices"))(x)
    return matvec


def run_gradient(args, out: dict) -> dict:
    mesh = _mesh_for(args.n_devices)
    D = args.n_devices
    results = []
    for n_hint in args.cells:
        n, edges, pa, source = _mesh_source(args, n_hint)
        how = "file"
        if pa is None:
            pa, how = partition_cells(n, edges, D, args.partition)
        rng = np.random.default_rng(1)
        w_global = rng.standard_normal(n).astype(np.float32)
        entry = {"cells": n, "mesh": source, "partition": how, "n_devices": D,
                 "grad_steps": args.grad_steps, "rollout": {}, "sharded_cg": None}

        # (i) reverse mode through a sharded rollout, both transports
        ref_node, _, _ = build_pair(n, edges, mesh, pa, "all_to_all")
        w_j = jnp.asarray(w_global)

        def loss_ref(x0):
            st = {"x": x0}
            for _ in range(args.grad_steps):
                st = {"x": ref_node.update(st, {}, 1.0)["x"]}
            return jnp.sum(st["x"] * w_j)

        x0_global = ref_node.initial_state()["x"]
        g_ref_fn = jax.jit(jax.grad(loss_ref))
        g_ref = np.asarray(jax.device_get(g_ref_fn(x0_global)))
        ref_t = timed(lambda: g_ref_fn(x0_global), warmup=args.warmup, repeats=args.repeats)
        entry["rollout"]["unsharded"] = {"grad": ref_t}
        for method in METHODS:
            _, sharded, layout = build_pair(n, edges, mesh, pa, method)
            w_layout = jnp.asarray(partition_value(value=w_global, layout=layout).reshape(-1))
            x0 = sharded.initial_state()["x"]

            def loss_sh(x0, sharded=sharded, w_layout=w_layout):
                st = {"x": x0}
                for _ in range(args.grad_steps):
                    st = {"x": sharded.update(st, {}, 1.0)["x"]}
                return jnp.sum(st["x"] * w_layout)

            g_fn = jax.grad(loss_sh)
            g_layout = np.asarray(jax.device_get(g_fn(x0)))
            g_sh = gather_value(per_shard=g_layout.reshape(D, layout.n_local_max), layout=layout)
            stats = timed(lambda: g_fn(x0), warmup=args.warmup, repeats=args.repeats)
            entry["rollout"][method] = {"grad": stats, "parity": _diff(g_sh, g_ref)}
            print(f"[gradient] cells={n:>8} rollout {method:<10} grad {stats['median_ms']:8.2f} ms"
                  f"  max|dg|={entry['rollout'][method]['parity']['max_abs']:.2e}"
                  f"  rel={entry['rollout'][method]['parity']['max_rel']:.2e}")

        # (ii) reverse + forward mode through the Jacobi-preconditioned sharded CG
        n_cg = (n // D) * D
        matvec_ref = _laplacian_matvec_unsharded()
        matvec_sh = _laplacian_matvec_sharded(mesh)
        pc = jacobi_preconditioner(jnp.full((n_cg,), 2.0, jnp.float32))
        kw = dict(max_iters=args.cg_max_iters, rtol=1e-6, atol=1e-8, backend="loop",
                  preconditioner=pc, differentiable=True)
        b = jnp.sin(jnp.linspace(0.0, 6.0, n_cg, dtype=jnp.float32)) + 0.1

        def cg_loss_sh(bb):
            return jnp.sum(sharded_cg(matvec_sh, bb, mesh=mesh, in_specs=P("devices"), **kw).value ** 2)

        def cg_loss_ref(bb):
            return jnp.sum(sharded_cg(matvec_ref, bb, **kw).value ** 2)

        g_sh_fn, g_ref_fn = jax.jit(jax.grad(cg_loss_sh)), jax.jit(jax.grad(cg_loss_ref))
        g_sh = np.asarray(jax.device_get(g_sh_fn(b)))
        g_ref = np.asarray(jax.device_get(g_ref_fn(b)))
        v = jnp.ones_like(b)
        _, t_sh = jax.jvp(lambda bb: sharded_cg(matvec_sh, bb, mesh=mesh, in_specs=P("devices"),
                                                **kw).value, (b,), (v,))
        _, t_ref = jax.jvp(lambda bb: sharded_cg(matvec_ref, bb, **kw).value, (b,), (v,))
        entry["sharded_cg"] = {
            "dof": n_cg,
            "grad_sharded": timed(lambda: g_sh_fn(b), warmup=args.warmup, repeats=args.repeats),
            "grad_unsharded": timed(lambda: g_ref_fn(b), warmup=args.warmup, repeats=args.repeats),
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
# Summary / recommendation
# ---------------------------------------------------------------------------


def _load_results(directory: Path, goal: str) -> list[dict]:
    docs = []
    for path in sorted(directory.glob(f"{goal}*.json")):
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("goal") == goal:
            docs.append(doc)
    return docs


def recommend(exchange_docs: list[dict], *, min_cells: int = 100_000,
              margin: float = 1.05) -> dict:
    """Which transport should be the default.

    ``ppermute`` wins when its median exchange time beats ``all_to_all``
    by at least ``margin`` at every hardware-sized point (``cells >=
    min_cells``); ``all_to_all`` keeps the default when it is faster at
    any such point; anything in between is a tie.  Dry-run or CPU
    results never decide -- they only prove the script.
    """
    rows = []
    for doc in exchange_docs:
        hw = doc["environment"]["platform"] == "gpu" and not doc.get("dry_run", False)
        for r in doc["results"]:
            a, p = r["methods"]["all_to_all"], r["methods"]["ppermute"]
            rows.append({
                "cells": r["cells"], "hardware": hw, "n_devices": r["n_devices"],
                "device_kinds": doc["environment"]["device_kinds"],
                "a2a_median_ms": a["median_ms"], "ppermute_median_ms": p["median_ms"],
                "a2a_min_ms": a["min_ms"], "ppermute_min_ms": p["min_ms"],
                "a2a_bytes_total": a["bytes_total"], "ppermute_bytes_total": p["bytes_total"],
                "speedup_median": r["ppermute_speedup_median"],
                "bit_identical": r["bit_identical"],
            })
    deciding = [r for r in rows if r["hardware"] and r["cells"] >= min_cells]
    if not deciding:
        return {"rows": rows, "decision": "undecided",
                "reason": f"no real-GPU measurement at >= {min_cells} cells "
                          "(dry-run / CPU results do not rank NCCL transports)"}
    speedups = [r["speedup_median"] for r in deciding]
    if all(s >= margin for s in speedups):
        decision = "ppermute"
        reason = (f"ppermute is >= {margin:.2f}x faster (median) at every measured size "
                  f">= {min_cells} cells on real GPUs; make it the default exchange")
    elif any(s < 1.0 for s in speedups):
        decision = "all_to_all"
        reason = "all_to_all is faster at at least one hardware-sized point; keep it the default"
    else:
        decision = "tie"
        reason = (f"ppermute is faster but by less than {margin:.2f}x somewhere; keep "
                  "all_to_all the default (fewer collectives) unless the byte savings matter")
    return {"rows": rows, "decision": decision, "reason": reason,
            "min_speedup": min(speedups), "max_speedup": max(speedups)}


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
              f"{'ppm min':>9} {'speedup':>8} {'a2a MB':>8} {'ppm MB':>8} {'same':>5}")
        for r in rec["rows"]:
            print(f"{r['cells']:>9} {r['n_devices']:>4} {'gpu' if r['hardware'] else 'cpu':>3} "
                  f"{r['a2a_median_ms']:9.3f} {r['ppermute_median_ms']:9.3f} "
                  f"{r['a2a_min_ms']:9.3f} {r['ppermute_min_ms']:9.3f} "
                  f"{r['speedup_median']:8.2f} {r['a2a_bytes_total'] / 1e6:8.2f} "
                  f"{r['ppermute_bytes_total'] / 1e6:8.2f} {'yes' if r['bit_identical'] else 'NO':>5}")
        print(f"Recommendation: {rec['decision']} -- {rec['reason']}")
    if forward_docs:
        print("\nForward run (ms per step; wrapper = public update(), device = compiled step)")
        for doc in forward_docs:
            for r in doc["results"]:
                for method, m in r["methods"].items():
                    dev = m["device_step"]["ms_per_step"] if m["device_step"] else float("nan")
                    print(f"{r['cells']:>9} cells {r['mesh']:<24} {method:<10} "
                          f"wrapper {m['wrapper_step']['ms_per_step']:9.3f}  device {dev:9.3f}  "
                          f"max|dx| {m['parity_x']['max_abs']:.2e}  "
                          f"finite={m['parity_x']['finite']}")
    if gradient_docs:
        print("\nGradient parity (sharded vs unsharded, max-abs / max-rel)")
        for doc in gradient_docs:
            for r in doc["results"]:
                for method in METHODS:
                    p = r["rollout"][method]["parity"]
                    print(f"{r['cells']:>9} cells rollout   {method:<10} "
                          f"{p['max_abs']:.2e} / {p['max_rel']:.2e}  "
                          f"grad {r['rollout'][method]['grad']['median_ms']:8.2f} ms")
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
                    help="print the ranking table + recommendation from DIR and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="small sizes on CPU virtual devices; proves the script, ranks nothing")
    ap.add_argument("--cells", type=int, nargs="+",
                    help=f"cell counts (default {GPU_CELLS} on GPU, {DRY_RUN_CELLS} otherwise)")
    ap.add_argument("--n-devices", type=int, help="mesh size (default: min(4, visible devices))")
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summarise is not None:
        return summarise(args.summarise)

    gpu = on_gpu()
    visible = len(jax.devices())
    if args.n_devices is None:
        args.n_devices = min(4, visible)
    if args.n_devices > visible:
        raise SystemExit(f"--n-devices {args.n_devices} but only {visible} device(s) visible")
    small = args.dry_run or not gpu
    if args.cells is None:
        args.cells = list(DRY_RUN_CELLS if small else GPU_CELLS)
    args.warmup = args.warmup if args.warmup is not None else (1 if small else 5)
    args.repeats = args.repeats if args.repeats is not None else (3 if small else 20)
    args.steps = args.steps if args.steps is not None else (3 if small else 20)
    args.cg_max_iters = args.cg_max_iters if args.cg_max_iters is not None else (300 if small else 3000)

    env = environment()
    print(f"devices: {env['devices']}  jax {env['jax']}  platform {env['platform']}"
          f"{'  [DRY RUN]' if args.dry_run else ''}")
    if not gpu and not args.dry_run:
        print("WARNING: no GPU backend -- these numbers do not rank NCCL transports")

    args.out.mkdir(parents=True, exist_ok=True)
    goals = ("exchange", "forward", "gradient") if args.goal == "all" else (args.goal,)
    runners = {"exchange": run_exchange, "forward": run_forward, "gradient": run_gradient}
    for goal in goals:
        doc = {
            "schema_version": SCHEMA_VERSION,
            "goal": goal,
            "dry_run": bool(args.dry_run),
            "environment": env,
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        }
        t0 = time.perf_counter()
        runners[goal](args, doc)
        doc["wall_s"] = time.perf_counter() - t0
        path = args.out / f"{goal}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        print(f"wrote {path} ({doc['wall_s']:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
