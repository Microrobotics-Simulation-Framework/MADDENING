#!/usr/bin/env python3
"""Shard an unstructured (graph-partitioned) node across four emulated CPU devices.

``ShardedUnstructuredNode`` is the sharding wrapper for a node whose
cells are the vertices of an arbitrary graph -- the cells of an
unstructured mesh -- rather than a Cartesian grid.  The cells are
assigned to devices by a partitioner on the host (PyMetis in practice;
here a coordinate bisection, so the example needs no extra package),
``build_unstructured_partition`` turns that assignment into a layout
(which cells each device owns, which other cells it reads -- its ghost
cells -- and who sends them), and the wrapper runs the node with one
block of cells per device, exchanging ghost cells before every step.

This example builds a 2-D point cloud joined to its nearest neighbours,
runs a diffusion node on it, and shows:

1. the partition: cells owned and ghost cells per device;
2. ``exchange_traffic(layout)``: the cells each device sends per exchange
   under the dense ``all_to_all`` transport and the per-neighbour
   ``ppermute`` one, for the bisection and for a random assignment of the
   same cells;
3. ``ShardedUnstructuredNode(exchange="ppermute")`` in a ``GraphManager``
   agrees with the same node unsharded to float32 rounding, and the two
   transports agree bit for bit;
4. why the node's ``static_data`` returns the same array objects on every
   access, with the step time of a node that rebuilds them for contrast;
5. ``sharded_cg(differentiable=True)`` with a Jacobi preconditioner,
   solving ``(I + tau L) u = b`` on the same partition (``L`` the graph
   Laplacian), and its gradient with respect to ``b`` and ``tau`` checked
   against a dense, unsharded solve.

**This is CPU emulation, not GPU performance.**  The four devices are
virtual CPU devices created with
``XLA_FLAGS=--xla_force_host_platform_device_count=4``, all on this one
machine and sharing its cores, so no timing here says anything about
real devices; the layout, the traffic counts and the agreement carry
over.  On a multi-GPU machine, build the mesh over the GPUs instead.

JAX fixes its device count when its backend starts, so this script sets
``XLA_FLAGS`` itself, before JAX is imported, replacing any device count
already in the variable and keeping every other flag (as
``sharding_demo.py`` does).  Run it as its own process
(``python -m ...``), not from a session that has already used JAX.

Usage
-----
    python -m maddening.examples.advanced.unstructured_sharding_demo
    python -m maddening.examples.advanced.unstructured_sharding_demo --n-cells 256 --steps 20
"""

from __future__ import annotations

import os
import sys

DEVICES = 4
_FLAG = "--xla_force_host_platform_device_count"


def _pin_device_count(xla_flags: str, count: int) -> str:
    """*xla_flags* with the host device count forced to *count*: any
    inherited setting of the flag (``--flag=N`` or ``--flag N``) is
    dropped, every other flag is kept in order."""
    tokens, kept, i = xla_flags.split(), [], 0
    while i < len(tokens):
        if tokens[i] == _FLAG:
            i += 2
        elif tokens[i].startswith(_FLAG + "="):
            i += 1
        else:
            kept.append(tokens[i])
            i += 1
    return " ".join([*kept, f"{_FLAG}={count}"])


# Before anything imports JAX.
os.environ["XLA_FLAGS"] = _pin_device_count(os.environ.get("XLA_FLAGS", ""), DEVICES)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import shard_map  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402

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
from maddening.core.graph_manager import GraphManager  # noqa: E402
from maddening.core.node import SimulationNode  # noqa: E402
from maddening.core.static_data import StaticArray  # noqa: E402

NEIGHBOURS = 6        # each point is joined to its 6 nearest (then symmetrised)
DIFFUSIVITY = 1.0     # [1/s] rate per unit edge weight
DT = 0.02             # [s]; explicit Euler is stable below 1 / (2 * max degree)
TAU = 0.5             # the solve's (I + tau L): condition number at most 1 + 2 tau max degree


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# The mesh and its partition (host side, once, at experiment setup)
# ---------------------------------------------------------------------------

def point_cloud_graph(n_cells: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random points in the unit square, each joined to its nearest
    neighbours: an irregular graph like the cell adjacency of an
    unstructured mesh.  Returns ``(positions (n, 2), edges (m, 2))`` with
    every undirected edge listed once."""
    rng = np.random.default_rng(seed)
    xy = rng.random((n_cells, 2))
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1)
    np.fill_diagonal(d2, np.inf)
    nearest = np.argpartition(d2, NEIGHBOURS, axis=1)[:, :NEIGHBOURS]
    pairs = np.sort(np.stack([np.repeat(np.arange(n_cells), NEIGHBOURS),
                              nearest.ravel()], axis=1), axis=1)
    return xy, np.unique(pairs, axis=0).astype(np.int32)


def coordinate_bisection(xy: np.ndarray) -> np.ndarray:
    """Four equal, compact parts: halve by x, then each half by y.  A
    stand-in for PyMetis, which a real mesh would use."""
    n = len(xy)
    assignment = np.empty(n, dtype=np.int32)
    by_x = np.argsort(xy[:, 0], kind="stable")
    for base, half in ((0, by_x[: n // 2]), (2, by_x[n // 2:])):
        by_y = half[np.argsort(xy[half, 1], kind="stable")]
        assignment[by_y[: len(by_y) // 2]] = base
        assignment[by_y[len(by_y) // 2:]] = base + 1
    return assignment


def neighbour_table(n_cells: int, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(nbr, weight)``, both ``(n_cells, max_degree)``: cell ``i``'s
    neighbours by global id, padded with ``i`` itself at weight 0."""
    lists: list[list[int]] = [[] for _ in range(n_cells)]
    for u, v in edges:
        lists[u].append(int(v))
        lists[v].append(int(u))
    width = max(len(row) for row in lists)
    nbr = np.repeat(np.arange(n_cells, dtype=np.int32)[:, None], width, axis=1)
    weight = np.zeros((n_cells, width), dtype=np.float32)
    for i, row in enumerate(lists):
        nbr[i, : len(row)] = sorted(row)
        weight[i, : len(row)] = 1.0
    return nbr, weight


def slab_table(nbr: np.ndarray, layout) -> np.ndarray:
    """``nbr`` rewritten in the layout's slab positions: entry ``(g, k)`` is
    where neighbour ``nbr[g, k]`` sits in the slab of the device that owns
    ``g`` -- ``[its own cells | its ghost cells]``, the order
    ``update_padded`` receives."""
    position = []
    for d in range(layout.n_devices):
        where = {int(g): i for i, g in enumerate(layout.local_global_ids[d])}
        where.update({int(g): layout.n_local_max + j
                      for j, g in enumerate(layout.ghost_global_ids[d])})
        position.append(where)
    owner = layout.partition_assignment
    return np.array([[position[owner[g]][int(j)] for j in row]
                     for g, row in enumerate(nbr)], dtype=np.int32)


# ---------------------------------------------------------------------------
# The node: diffusion on the graph, written for the partition layout
# ---------------------------------------------------------------------------

class GraphDiffusionNode(SimulationNode):
    """``du_i/dt = D * sum_j w_ij (u_j - u_i)`` on a graph, explicit Euler.

    ``update`` steps the whole graph (unsharded).  ``update_padded`` is
    what ``ShardedUnstructuredNode`` calls on each device: ``u`` arrives
    as ``[owned cells | ghost cells]`` and the node reads its neighbours
    through a slab-position table held as a partitioned ``StaticArray``.
    ``halo_width()`` stays the default ``{}``: the halo is the layout's
    ghost cells, not an axis-aligned width.
    """

    def __init__(self, name: str, timestep: float, *, nbr: np.ndarray,
                 weight: np.ndarray, u0: np.ndarray, layout=None,
                 diffusivity: float = DIFFUSIVITY):
        super().__init__(name, timestep, diffusivity=diffusivity)
        self._nbr = jnp.asarray(nbr)
        self._weight = jnp.asarray(weight)
        self._u0 = np.asarray(u0, dtype=np.float32)
        # The static data, built ONCE, here, and returned as the same
        # objects on every access (the property below).  The sharded
        # wrapper caches its per-device copy of each partitioned
        # StaticArray keyed on the identity of the array: a static_data
        # that built new arrays on each access would make every eager
        # update() re-partition them (device to host, a gather through the
        # layout, host to device).  Measured on 4 GPUs at 1e7 cells, a node
        # that rebuilt them stepped in 242 ms against 4.45 ms.  Section 4
        # shows the same effect on this CPU.
        self._static = {} if layout is None else {
            "nbr_slab": StaticArray(slab_table(nbr, layout), replication="partition",
                                    partition_assignment=layout.partition_assignment),
            "weight": StaticArray(weight, replication="partition",
                                  partition_assignment=layout.partition_assignment),
        }

    @property
    def static_data(self) -> dict:
        return self._static          # the same dict and arrays every time

    def state_fields(self) -> list[str]:
        return ["u"]

    def initial_state(self) -> dict:
        return {"u": jnp.asarray(self._u0)}

    def update(self, state, boundary_inputs, dt):
        u = state["u"]
        rate = self.params["diffusivity"]
        return {"u": u + dt * rate * jnp.sum(self._weight * (u[self._nbr] - u[:, None]), axis=1)}

    def update_padded(self, state_padded, boundary_inputs, dt, *, static_padded=None,
                      shard_info=None):
        u = state_padded["u"]                    # [owned | ghosts]
        n_own = shard_info[0][1]                 # the owned block's length
        own = u[:n_own]
        idx = static_padded["nbr_slab"][:n_own]
        w = static_padded["weight"][:n_own]      # 0 on padding rows: they stay put
        rate = self.params["diffusivity"]
        return {"u": own + dt * rate * jnp.sum(w * (jnp.take(u, idx, axis=0) - own[:, None]),
                                               axis=1)}


class RebuildingDiffusionNode(GraphDiffusionNode):
    """The pattern NOT to copy: a ``static_data`` that builds new arrays
    on every access.  The values are the same; their identity is not."""

    @property
    def static_data(self) -> dict:
        return {k: StaticArray(np.array(v.value), replication=v.replication,
                               partition_assignment=v.partition_assignment)
                for k, v in self._static.items()}


def graph_of(node: SimulationNode) -> GraphManager:
    gm = GraphManager()
    gm.add_node(node)
    gm.compile()
    return gm


def mean_step_ms(sharded: ShardedUnstructuredNode, steps: int) -> float:
    state = sharded.initial_state()
    jax.block_until_ready(sharded.update(state, {}, DT))      # compile
    start = time.perf_counter()
    for _ in range(steps):
        state = sharded.update(state, {}, DT)
    jax.block_until_ready(state)
    return 1e3 * (time.perf_counter() - start) / steps


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-cells", type=int, default=2048,
                        help=f"Cells in the point cloud; a multiple of {DEVICES} (default 2048)")
    parser.add_argument("--steps", type=int, default=200,
                        help="Diffusion steps run by each graph (default 200)")
    args = parser.parse_args(argv)
    if args.n_cells % DEVICES or args.n_cells < 8 * DEVICES:
        parser.error(f"--n-cells must be a multiple of {DEVICES}, at least {8 * DEVICES}")

    if jax.device_count() != DEVICES:
        print(f"JAX started with {jax.device_count()} devices, not {DEVICES}: it was "
              f"initialised before this script set XLA_FLAGS.  Run it as its own "
              f"process: python -m maddening.examples.advanced.unstructured_sharding_demo")
        return 1
    print(f"JAX devices: {jax.devices()}  (virtual CPU devices: emulation, "
          f"not a performance measurement)")
    mesh = create_device_mesh(shape=(DEVICES,))

    # 1 ------------------------------------------------------------------
    section(f"1. A {args.n_cells}-cell point cloud, partitioned four ways")
    xy, edges = point_cloud_graph(args.n_cells)
    nbr, weight = neighbour_table(args.n_cells, edges)
    max_degree = int(weight.sum(axis=1).max())
    assignment = coordinate_bisection(xy)
    layout = build_unstructured_partition(partition_assignment=assignment, edges=edges,
                                          n_devices=DEVICES)
    print(f"  {len(edges)} edges, degree {int(weight.sum(axis=1).min())} to {max_degree}")
    print(f"  cells owned per device: {list(layout.n_local)}")
    print(f"  ghost cells per device: {list(layout.n_ghost)}  (read from other devices)")
    assert sum(layout.n_local) == args.n_cells and layout.n_local_max == args.n_cells // DEVICES
    assert all(0 < g < args.n_cells // DEVICES for g in layout.n_ghost)
    assert DT * DIFFUSIVITY * 2 * max_degree < 1.0, "explicit Euler would be unstable"

    # 2 ------------------------------------------------------------------
    section("2. exchange_traffic(layout): cells sent per device per exchange")
    rng = np.random.default_rng(1)
    random_layout = build_unstructured_partition(
        partition_assignment=rng.permutation(np.arange(args.n_cells) % DEVICES).astype(np.int32),
        edges=edges, n_devices=DEVICES)
    traffic = {"coordinate bisection": exchange_traffic(layout),
               "random assignment": exchange_traffic(random_layout)}
    print(f"  {'partition':<22}{'all_to_all':>11}{'ppermute':>10}{'useful':>8}{'messages':>10}")
    for name, t in traffic.items():
        print(f"  {name:<22}{t['all_to_all']:>11}{t['ppermute']:>10}{t['useful']:>8}"
              f"{t['ppermute_messages']:>10}")
        assert t["useful"] <= t["ppermute"] <= t["all_to_all"], t
        assert t["ppermute_messages"] <= DEVICES - 1, t
    compact, scattered = traffic["coordinate bisection"], traffic["random assignment"]
    assert compact["useful"] < scattered["useful"], (compact, scattered)
    print("  all_to_all sends n_devices * n_ghost_max cells whoever needs them;")
    print("  ppermute sends one message per device offset that carries any, sized")
    print("  to its largest.  A compact partition needs far fewer ghosts.")

    # 3 ------------------------------------------------------------------
    section(f"3. ShardedUnstructuredNode(exchange='ppermute'): {args.steps} steps")
    u0 = np.exp(-((xy - 0.3) ** 2).sum(axis=1) / 0.01).astype(np.float32)  # a hot spot
    plain = graph_of(GraphDiffusionNode("heat", DT, nbr=nbr, weight=weight, u0=u0))
    wrapped = {}
    for exchange in ("ppermute", "all_to_all"):
        node = GraphDiffusionNode("heat", DT, nbr=nbr, weight=weight, u0=u0, layout=layout)
        wrapped[exchange] = ShardedUnstructuredNode(node, mesh, layout, exchange=exchange)
    sharded = graph_of(wrapped["ppermute"])
    sharded.print_graph()

    u_plain = np.asarray(plain.run_scan(args.steps)["heat"]["u"])
    u_layout = sharded.run_scan(args.steps)["heat"]["u"]      # partition layout, on the mesh
    shards = sorted(u_layout.addressable_shards, key=lambda s: s.index[0].start)
    print(f"  the result: {u_layout.shape[0]} rows in partition layout, "
          f"{[s.data.shape[0] for s in shards]} on the devices {[s.device.id for s in shards]}")
    assert len({s.device for s in shards}) == DEVICES
    u_sharded = wrapped["ppermute"].gather_global({"u": u_layout})["u"]   # global cell order
    diff = float(np.abs(u_sharded - u_plain).max())
    print(f"  heat left: {u_plain.sum():.4f} (unsharded), {u_sharded.sum():.4f} (sharded)")
    print(f"  largest difference sharded vs unsharded: {diff:.1e}"
          + ("  (bitwise identical)" if diff == 0.0 else ""))
    assert diff <= 1e-6 * float(np.abs(u_plain).max()), diff
    assert u_plain.max() < 0.9 * u0.max(), "the hot spot should have spread"

    dense_route = graph_of(wrapped["all_to_all"])
    u_a2a = wrapped["all_to_all"].gather_global(
        {"u": dense_route.run_scan(args.steps)["heat"]["u"]})["u"]
    print(f"  ppermute and all_to_all bit-identical: {np.array_equal(u_a2a, u_sharded)}")
    assert np.array_equal(u_a2a, u_sharded)

    # 4 ------------------------------------------------------------------
    section("4. static_data must return the same objects on every access")
    good = GraphDiffusionNode("heat", DT, nbr=nbr, weight=weight, u0=u0, layout=layout)
    bad = RebuildingDiffusionNode("heat", DT, nbr=nbr, weight=weight, u0=u0, layout=layout)
    assert good.static_data["nbr_slab"].value is good.static_data["nbr_slab"].value
    assert bad.static_data["nbr_slab"].value is not bad.static_data["nbr_slab"].value
    ms_good = mean_step_ms(ShardedUnstructuredNode(good, mesh, layout, exchange="ppermute"), 50)
    ms_bad = mean_step_ms(ShardedUnstructuredNode(bad, mesh, layout, exchange="ppermute"), 50)
    print(f"  eager update(), the same arrays each time: {ms_good:.2f} ms/step")
    print(f"  eager update(), new arrays on each access: {ms_bad:.2f} ms/step"
          f"  (every step re-partitions the statics)")
    print("  Timings on a shared CPU are indicative only and not checked; on 4 GPUs at")
    print("  1e7 cells the same mistake cost 242 ms/step against 4.45.  Build static")
    print("  data once in __init__ and return it (or call invalidate_static_cache()")
    print("  after rewriting a buffer in place).")

    # 5 ------------------------------------------------------------------
    section("5. sharded_cg(differentiable=True) with a Jacobi preconditioner")
    rows = layout.n_devices * layout.n_local_max
    on_mesh = NamedSharding(mesh, P("devices"))

    def to_layout(value: np.ndarray) -> jax.Array:
        per_shard = partition_value(value=value, layout=layout)
        return jax.device_put(jnp.asarray(per_shard.reshape((rows,) + per_shard.shape[2:])),
                              on_mesh)

    nbr_slab, w_slab = to_layout(slab_table(nbr, layout)), to_layout(weight)

    def local_matvec(x, tau, idx, w):
        """``(I + tau L) x`` for one device's block; its ghosts come by ppermute."""
        padded = exchange_unstructured(x, layout=layout, mesh_axis="devices",
                                       method="ppermute")
        return x + tau * jnp.sum(w * (x[:, None] - jnp.take(padded, idx, axis=0)), axis=1)

    matvec_on_mesh = shard_map(local_matvec, mesh=mesh,
                               in_specs=(P("devices"), P(), P("devices"), P("devices")),
                               out_specs=P("devices"))
    # The preconditioner is built from the nominal tau and carries no
    # derivative; the solution does not depend on it.
    diagonal = to_layout(1.0 + TAU * weight.sum(axis=1))
    precondition = jacobi_preconditioner(diagonal)
    kw = dict(mesh=mesh, in_specs=P("devices"), rtol=1e-5, max_iters=500,
              backend="loop", preconditioner=precondition)

    def solve_sharded(b, tau, **extra):
        return sharded_cg(lambda x: matvec_on_mesh(x, tau, nbr_slab, w_slab), b, **kw, **extra)

    target = np.sin(4 * xy[:, 0]).astype(np.float32)
    target_layout = to_layout(target)
    b_global = np.cos(3 * xy[:, 1]).astype(np.float32)
    b0 = to_layout(b_global)

    def loss_sharded(b, tau):
        result = solve_sharded(b, tau, differentiable=True)
        loss = 0.5 * jnp.sum((result.value - target_layout) ** 2)
        return loss, (result.value, result.converged, result.residual_norm)

    # Called under jax.jit, as it would be inside a compiled step.  Called
    # eagerly, every operation around the CG loop is compiled on its own:
    # one solve took about ten times as long here.
    grad_sharded = jax.jit(jax.grad(loss_sharded, argnums=(0, 1), has_aux=True))
    (g_b, g_tau), (u_layout, converged, residual) = grad_sharded(b0, jnp.float32(TAU))
    print(f"  converged: {bool(converged)}, ||b - A u|| = {float(residual):.2e}"
          f" (condition number at most {1 + 2 * TAU * max_degree:.0f})")
    assert bool(converged)

    def to_global(a: jax.Array) -> np.ndarray:
        """A partition-layout vector back in global cell order."""
        per_shard = np.asarray(jax.device_get(a)).reshape(DEVICES, layout.n_local_max)
        return gather_value(per_shard=per_shard, layout=layout)

    # The unsharded reference: a dense solve in float64 on the host, and its
    # gradient by the adjoint formula (A is symmetric): dL/db = A^-1 (u - t),
    # dL/dtau = -(dL/db) . (L u).
    laplacian = np.zeros((args.n_cells, args.n_cells))
    laplacian[edges[:, 0], edges[:, 1]] = laplacian[edges[:, 1], edges[:, 0]] = -1.0
    laplacian[np.diag_indices(args.n_cells)] = weight.sum(axis=1)
    a_dense = np.eye(args.n_cells) + TAU * laplacian
    u_dense = np.linalg.solve(a_dense, b_global.astype(np.float64))
    d_b = np.linalg.solve(a_dense, u_dense - target)
    d_tau = -float(d_b @ (laplacian @ u_dense))

    err_u = float(np.abs(to_global(u_layout) - u_dense).max() / np.abs(u_dense).max())
    err_b = float(np.abs(to_global(g_b) - d_b).max() / np.abs(d_b).max())
    err_tau = abs(float(g_tau) - d_tau) / abs(d_tau)
    print(f"  u:              largest difference from the dense solve {err_u:.1e} (relative)")
    print(f"  d loss / d b:   largest difference from the dense solve {err_b:.1e} (relative)")
    print(f"  d loss / d tau: {float(g_tau):.6f} sharded, {d_tau:.6f} dense "
          f"({err_tau:.1e} relative)")
    assert err_u < 1e-3 and err_b < 1e-3 and err_tau < 1e-3, (err_u, err_b, err_tau)

    try:
        jax.jit(jax.grad(lambda b: jnp.sum(solve_sharded(b, TAU).value)))(b0)
    except ValueError as exc:
        print(f"  without differentiable=True, jax.grad refuses: {str(exc)[:90]}...")
    else:
        raise AssertionError("jax.grad through sharded_cg without differentiable=True ran")

    print()
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
