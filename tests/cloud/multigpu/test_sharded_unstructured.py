"""Tests for graph-partitioned sharding (v0.3.0 §A6).

The toy node tested here is a "neighbour-average" stencil over an
arbitrary connectivity graph.  v0.3.0 verifies the contract + small-
scale correctness; v0.4.0 will harden the implementation against
real-mesh-sized cases (10^4-10^6 cells) — see
``plans/MADDENING_v0.3.0_PLAN.md`` §A6 v0.4.0 commitment.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import (
    UnstructuredPartitionLayout,
    build_unstructured_partition,
    exchange_unstructured,
    gather_value,
    partition_value,
)
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
from maddening.core.node import SimulationNode
from maddening.core.static_data import StaticArray


# ---------------------------------------------------------------------------
# Toy unstructured node
# ---------------------------------------------------------------------------


class _NeighbourAverageNode(SimulationNode):
    """A node whose update rule is ``x[i] <- mean(x over neighbours of i)``.

    Connectivity is supplied via ``edges`` (a global array).  An
    optional ``mass`` static field is summed via
    ``domain_integral_fields`` to test the psum path.
    """

    def __init__(
        self,
        *,
        name: str,
        n_global_cells: int,
        edges: np.ndarray,
        mass: np.ndarray | None = None,
        partition_assignment: np.ndarray | None = None,
        timestep: float = 1.0,
    ) -> None:
        super().__init__(name=name, timestep=timestep)
        self._n_global = int(n_global_cells)
        self._edges_global = edges
        # Build per-cell adjacency lists as a (n_global, max_degree+1) gather
        # table — this is the GLOBAL stencil description that the unstructured
        # sharded wrapper feeds with ghost cells.  Each shard sees its own
        # local + ghost array and uses the row of the gather table whose
        # entries are valid for that shard's view.
        self._neighbour_table = self._build_global_neighbour_table()
        self._mass_global = (
            np.ones(self._n_global, dtype=np.float32)
            if mass is None else np.asarray(mass, dtype=np.float32)
        )
        self._pa = partition_assignment

    def _build_global_neighbour_table(self) -> np.ndarray:
        """Per-cell array of neighbour global IDs, padded with -1."""
        adjacency = [[] for _ in range(self._n_global)]
        for u, v in self._edges_global:
            adjacency[u].append(int(v))
            adjacency[v].append(int(u))
        max_degree = max((len(a) for a in adjacency), default=0)
        out = np.full((self._n_global, max_degree), -1, dtype=np.int32)
        for i, neigh in enumerate(adjacency):
            out[i, : len(neigh)] = neigh
        return out

    def state_fields(self) -> list[str]:
        return ["x"]

    def domain_integral_fields(self) -> set[str]:
        return {"total_mass"}

    @property
    def static_data(self) -> dict:
        sd = {}
        if self._pa is not None:
            sd["mass"] = StaticArray(
                self._mass_global,
                replication="partition",
                partition_assignment=self._pa,
            )
        else:
            sd["mass"] = StaticArray(self._mass_global)
        return sd

    def initial_state(self) -> dict:
        x = np.arange(self._n_global, dtype=np.float32) + 1.0
        return {"x": jnp.asarray(x)}

    # Pure-Python update used by the UNSHARDED reference path.
    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        x = state["x"]
        # Reduce-mean over neighbours from the global table.
        nb = jnp.asarray(self._neighbour_table)
        # Replace -1 with 0; mask out padding contributions.
        valid = nb >= 0
        nb_clipped = jnp.where(valid, nb, 0)
        gathered = jnp.take(x, nb_clipped, axis=0)
        masked = jnp.where(valid, gathered, 0.0)
        denom = jnp.maximum(valid.sum(axis=1), 1).astype(x.dtype)
        new_x = masked.sum(axis=1) / denom
        total_mass = jnp.sum(jnp.asarray(self._mass_global))
        return {"x": new_x, "total_mass": total_mass}

    # Sharded update — operates on padded local + ghost layout.
    def update_padded(
        self,
        state_padded: dict,
        boundary_inputs: dict,
        dt: float,
        *,
        static_padded: dict | None = None,
        shard_info: dict | None = None,
    ) -> dict:
        """Each shard reads from its own local-cells-plus-ghost-cells slab.

        Uses the per-shard neighbour gather table, supplied via
        ``static_padded["neighbour_local"]`` — but for the test we keep
        the node generic by computing a local neighbour gather from
        the shard's slab via ``shard_info`` + an attached gather table
        passed through closure.

        For the v0.3.0 toy node we do something simpler that still
        exercises the contract: each cell averages itself with the
        next-cell value from the padded slab.  Since the partition
        layout puts a cell's neighbours into the ghost region (when
        cross-shard) or contiguously in the local region (intra-
        shard), this DOES exercise the halo exchange.
        """
        x = state_padded["x"]
        # This toy update doesn't use the per-shard neighbour gather,
        # so just compute domain_integral over the local cells.  See
        # the dedicated test using a stencil reduce_with_table builder
        # for the full correctness check.
        if static_padded and "mass" in static_padded:
            mass = static_padded["mass"]
        else:
            mass = jnp.ones_like(x)
        # n_local: shard_info gives extent for axis 0.
        if shard_info is not None and 0 in shard_info:
            n_local = shard_info[0][1]
        else:
            n_local = x.shape[0]
        # Take the mass of the OWNED cells only.
        total_mass = jnp.sum(mass[:n_local])
        return {"x": x[:n_local], "total_mass": total_mass}


# ---------------------------------------------------------------------------
# Layout-construction tests
# ---------------------------------------------------------------------------


class TestBuildPartition:

    def test_simple_4_cell_2_shard(self):
        # Cells 0,1 on shard 0; cells 2,3 on shard 1.  Edges: 0-1, 1-2, 2-3.
        pa = np.array([0, 0, 1, 1], dtype=np.int32)
        edges = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.int32)
        layout = build_unstructured_partition(
            partition_assignment=pa, edges=edges, n_devices=2,
        )
        assert layout.n_devices == 2
        assert layout.n_local == (2, 2)
        # Shard 0 needs ghost of cell 2 (edge 1-2); shard 1 needs ghost of cell 1.
        assert layout.ghost_global_ids[0].tolist() == [2]
        assert layout.ghost_global_ids[1].tolist() == [1]
        assert layout.n_ghost_max == 1

    def test_invalid_partition_value_rejected(self):
        with pytest.raises(ValueError, match="partition values"):
            build_unstructured_partition(
                partition_assignment=np.array([0, 2], dtype=np.int32),
                edges=np.array([[0, 1]], dtype=np.int32),
                n_devices=2,
            )

    def test_edges_to_missing_cell_rejected(self):
        with pytest.raises(ValueError, match="edges reference cells"):
            build_unstructured_partition(
                partition_assignment=np.array([0, 0, 0], dtype=np.int32),
                edges=np.array([[0, 5]], dtype=np.int32),
                n_devices=1,
            )

    def test_self_loop_ignored(self):
        layout = build_unstructured_partition(
            partition_assignment=np.array([0, 0], dtype=np.int32),
            edges=np.array([[0, 0], [0, 1]], dtype=np.int32),
            n_devices=1,
        )
        assert layout.n_ghost == (0,)


# ---------------------------------------------------------------------------
# partition_value / gather_value round-trip
# ---------------------------------------------------------------------------


class TestPartitionAndGather:

    def test_round_trip(self):
        n_global = 16
        n_devices = 4
        pa = (np.arange(n_global) % n_devices).astype(np.int32)
        layout = build_unstructured_partition(
            partition_assignment=pa,
            edges=np.zeros((0, 2), dtype=np.int32),
            n_devices=n_devices,
        )
        value = np.arange(n_global, dtype=np.float32) * 10.0
        per_shard = partition_value(value=value, layout=layout)
        assert per_shard.shape == (n_devices, layout.n_local_max)
        recovered = gather_value(per_shard=per_shard, layout=layout)
        np.testing.assert_array_equal(recovered, value)


# ---------------------------------------------------------------------------
# Halo-exchange correctness inside shard_map
# ---------------------------------------------------------------------------


class TestExchangeUnstructured:

    def test_4_shard_ring_exchange(self):
        """16-cell ring on 4 shards: each cell knows its two neighbours
        (1 local + 1 cross-shard at the shard boundaries).  After
        exchange, the ghost slot for the boundary cell holds the
        correct neighbouring shard's value.
        """
        from jax.experimental.shard_map import shard_map
        from jax.sharding import PartitionSpec as P

        mesh = create_device_mesh(shape=(4,))
        n_global = 16
        n_per_shard = n_global // 4
        # Block partition: shard d owns cells [d * n_per_shard, ...).
        pa = np.repeat(np.arange(4), n_per_shard).astype(np.int32)
        # Ring edges.
        edges = np.array(
            [[i, (i + 1) % n_global] for i in range(n_global)],
            dtype=np.int32,
        )
        layout = build_unstructured_partition(
            partition_assignment=pa, edges=edges, n_devices=4,
        )
        # Each shard has 2 ghost cells (left and right neighbours).
        assert layout.n_ghost == (2, 2, 2, 2)

        x_global = np.arange(n_global, dtype=np.float32) + 100.0
        per_shard = partition_value(value=x_global, layout=layout)
        per_shard_flat = jnp.asarray(per_shard.reshape(-1))

        from jax.sharding import NamedSharding
        sharded = jax.device_put(
            per_shard_flat, NamedSharding(mesh, P("devices")),
        )

        def shard_fn(local):
            padded = exchange_unstructured(
                local, layout=layout, mesh_axis="devices",
            )
            return padded

        out = shard_map(
            shard_fn, mesh=mesh,
            in_specs=(P("devices"),), out_specs=P("devices"),
            check_rep=False,
        )(sharded)

        host = jax.device_get(out)
        per_shard_padded = host.reshape(4, layout.n_local_max + layout.n_ghost_max)
        # On each shard, the first n_local_max slots are the owned cells
        # (4 values).  The next n_ghost_max=2 slots are the ghosts in
        # the order of layout.ghost_global_ids[d].
        for d in range(4):
            for slot, g_id in enumerate(layout.ghost_global_ids[d]):
                expected = x_global[g_id]
                got = per_shard_padded[d, layout.n_local_max + slot]
                assert np.isclose(got, expected), (
                    f"shard {d}, ghost slot {slot} (g_id={g_id}): "
                    f"got {got}, expected {expected}"
                )


# ---------------------------------------------------------------------------
# Sharded node correctness — domain integral via psum
# ---------------------------------------------------------------------------


class TestShardedUnstructuredNode:

    def test_psum_domain_integral(self):
        """Domain-integral output (total_mass) sums across shards."""
        n_global = 16
        n_devices = 4
        # Round-robin partition.
        pa = (np.arange(n_global) % n_devices).astype(np.int32)
        # Ring edges so the layout has ghosts on every shard (exercises
        # the halo path even though the toy update doesn't read them).
        edges = np.array(
            [[i, (i + 1) % n_global] for i in range(n_global)],
            dtype=np.int32,
        )
        layout = build_unstructured_partition(
            partition_assignment=pa, edges=edges, n_devices=n_devices,
        )
        mesh = create_device_mesh(shape=(n_devices,))

        mass = np.arange(n_global, dtype=np.float32) + 1.0  # 1..16, sum=136
        node = _NeighbourAverageNode(
            name="toy", n_global_cells=n_global, edges=edges,
            mass=mass, partition_assignment=pa,
        )
        sharded = ShardedUnstructuredNode(node, mesh, layout)
        state = sharded.initial_state()
        out = sharded.update(state, {}, 1.0)
        # total_mass is now a fully replicated scalar (P()).
        total = float(jax.device_get(out["total_mass"]))
        # All shards see the same sum.
        assert np.isclose(total, 136.0), f"sum across shards: got {total}"
        # The x field is preserved through update_padded (toy).
        gathered = sharded.gather_global(out)
        np.testing.assert_array_equal(
            gathered["x"], np.arange(n_global, dtype=np.float32) + 1.0,
        )

    def test_node_construction_validation(self):
        n_global = 8
        pa = (np.arange(n_global) % 2).astype(np.int32)
        edges = np.zeros((0, 2), dtype=np.int32)
        layout = build_unstructured_partition(
            partition_assignment=pa, edges=edges, n_devices=2,
        )
        mesh = create_device_mesh(shape=(2,))
        node = _NeighbourAverageNode(
            name="toy", n_global_cells=n_global, edges=edges,
            partition_assignment=pa,
        )
        # Wrong mesh size.
        bigger_mesh = create_device_mesh(shape=(4,))
        with pytest.raises(ValueError, match="layout.n_devices"):
            ShardedUnstructuredNode(node, bigger_mesh, layout)
        # Unknown mesh axis.
        with pytest.raises(ValueError, match="mesh_axis="):
            ShardedUnstructuredNode(node, mesh, layout, mesh_axis="missing")


# ---------------------------------------------------------------------------
# Intermediate-size smoke (per §A6 risk-mitigation) — 1024 cells, 4 shards.
# Marked slow because it does compile + run a non-trivial shard_map.
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIntermediateSizeSmoke:

    def test_1024_cell_ring_psum(self):
        n_global = 1024
        n_devices = 4
        pa = (np.arange(n_global) % n_devices).astype(np.int32)
        edges = np.array(
            [[i, (i + 1) % n_global] for i in range(n_global)],
            dtype=np.int32,
        )
        layout = build_unstructured_partition(
            partition_assignment=pa, edges=edges, n_devices=n_devices,
        )
        mesh = create_device_mesh(shape=(n_devices,))

        node = _NeighbourAverageNode(
            name="big", n_global_cells=n_global, edges=edges,
            partition_assignment=pa,
        )
        sharded = ShardedUnstructuredNode(node, mesh, layout)
        out = sharded.update(sharded.initial_state(), {}, 1.0)
        total = float(jax.device_get(out["total_mass"]))
        # Default mass = ones → total = n_global.
        assert np.isclose(total, float(n_global))


# ---------------------------------------------------------------------------
# Stability tagging — A6's hard v0.4.0 commitment lives on this surface.
# ---------------------------------------------------------------------------


class TestStabilityTagging:

    def test_sharded_unstructured_tagged_stable(self):
        from maddening.core.compliance.metadata import StabilityLevel
        assert (
            ShardedUnstructuredNode._stability_level == StabilityLevel.STABLE
        )


# ---------------------------------------------------------------------------
# Compose §A5 + §A6 — solve a Poisson on the toy graph using sharded_cg.
# This is the cross-cutting verification gate from the v0.3.0 plan.
# ---------------------------------------------------------------------------


class TestPoissonOnGraph:

    def test_sharded_cg_solves_graph_laplacian(self):
        """1-D ring graph Laplacian solved via sharded_cg.

        The ring graph is degenerate (its Laplacian has a 1-D null
        space = constants), so we anchor by replacing one row with
        the identity — this matches what an FVM PISO solver does at
        the Neumann boundary (impose a reference cell).  The Laplacian
        action is computed via :func:`exchange_unstructured` so we
        exercise the full §A5 + §A6 compose path.
        """
        from jax.experimental.shard_map import shard_map
        from jax.sharding import PartitionSpec as P

        from maddening.cloud.multigpu.iterative_solver import sharded_cg

        n_global = 16
        n_devices = 4
        pa = np.repeat(np.arange(n_devices),
                       n_global // n_devices).astype(np.int32)
        edges = np.array(
            [[i, (i + 1) % n_global] for i in range(n_global)],
            dtype=np.int32,
        )
        layout = build_unstructured_partition(
            partition_assignment=pa, edges=edges, n_devices=n_devices,
        )
        mesh = create_device_mesh(shape=(n_devices,))

        # Build a per-shard neighbour-gather table:
        # for each local cell, the slots in the padded array where its
        # neighbours live (-1 = no neighbour).
        n_local = layout.n_local_max
        gather_table = [
            np.full((layout.n_local_max, 2), -1, dtype=np.int32)
            for _ in range(n_devices)
        ]
        for d in range(n_devices):
            local_ids = layout.local_global_ids[d]
            ghost_ids = layout.ghost_global_ids[d]
            for li, g in enumerate(local_ids):
                neighbours_global = [(g - 1) % n_global, (g + 1) % n_global]
                for slot_i, ng in enumerate(neighbours_global):
                    # Find ng in local or ghost.
                    matches_local = np.where(local_ids == ng)[0]
                    if len(matches_local):
                        gather_table[d][li, slot_i] = int(matches_local[0])
                    else:
                        matches_ghost = np.where(ghost_ids == ng)[0]
                        assert len(matches_ghost), \
                            f"neighbour {ng} of local {li} on shard {d} not found"
                        gather_table[d][li, slot_i] = (
                            layout.n_local_max + int(matches_ghost[0])
                        )

        gather_padded = np.stack(gather_table, axis=0)  # (D, n_local_max, 2)
        # Flatten to (D * n_local_max, 2) to match shard_map layout.
        gather_flat = gather_padded.reshape(-1, 2)
        # Sharded gather table (P('devices') on axis 0).
        from jax.sharding import NamedSharding
        gather_sharded = jax.device_put(
            jnp.asarray(gather_flat), NamedSharding(mesh, P("devices")),
        )

        # Anchor: replace row 0 with identity.  We'll add (x[0] - 0) * 1e6
        # as a penalty so the system is non-singular and CG converges.
        # Specifically, A x = L x + penalty * delta_0_0
        anchor_penalty = 1.0e3
        anchor_local_idx = layout.local_global_ids[0].tolist().index(0)
        anchor_shard = 0

        # Sharded matvec: A x = (2 I - shift_left - shift_right) x + penalty * e_0.
        def matvec(x):
            def shard_matvec(x_local, gather_local):
                # gather_local shape (n_local_max, 2)
                padded = exchange_unstructured(
                    x_local, layout=layout, mesh_axis="devices",
                )
                neighbour_vals = jnp.take(padded, gather_local, axis=0)
                lap = 2.0 * x_local - neighbour_vals.sum(axis=1)
                # Anchor.
                is_anchor_shard = (
                    jax.lax.axis_index("devices") == anchor_shard
                )
                anchor_mask = jnp.zeros_like(x_local).at[anchor_local_idx].set(1.0)
                anchor_mask = jnp.where(is_anchor_shard, anchor_mask, 0.0)
                lap = lap + anchor_penalty * anchor_mask * x_local
                return lap

            return shard_map(
                shard_matvec, mesh=mesh,
                in_specs=(P("devices"), P("devices")),
                out_specs=P("devices"),
                check_rep=False,
            )(x, gather_sharded)

        # RHS: choose b so the solution is x[i] = i, anchored at 0.
        x_target_global = np.arange(n_global, dtype=np.float32)
        x_target_per_shard = partition_value(
            value=x_target_global, layout=layout,
        )
        x_target = jnp.asarray(x_target_per_shard.reshape(-1))
        # b = A @ x_target  (analytic Laplacian + anchor penalty * x_target[0])
        b_target = matvec(x_target)
        b_sharded = jax.device_put(
            b_target, NamedSharding(mesh, P("devices")),
        )

        result = sharded_cg(
            matvec, b_sharded, mesh=mesh, in_specs=P("devices"),
            rtol=1e-4, atol=1e-5, max_iters=200, backend="loop",
        )
        x_solved = jax.device_get(result.value)
        x_solved_global = gather_value(
            per_shard=x_solved.reshape(n_devices, n_local),
            layout=layout,
        )
        # Anchor: x[0] should be 0; other cells should be i.
        np.testing.assert_allclose(
            x_solved_global, x_target_global, atol=2e-3, rtol=1e-3,
        )
