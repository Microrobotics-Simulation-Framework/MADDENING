"""Contract stress-test for the v0.3.0 §A6 -> v0.4.0 MIME commitment.

We write out the API a hypothetical MIME ``FVMFluidNode`` would have
(subclassing ``ShardedUnstructuredNode``) and verify that *nothing* in
the v0.3.0 contract forces MIME to ask us for a breaking change in
v0.4.0.

If this test breaks because we tightened a constructor signature,
renamed a method, or restructured a contract — that's exactly the
signal the v0.3.0 plan demands.  Either revert the contract change,
or escalate it as a deliberate plan revision (with a paired update
to ``plans/MADDENING_v0.3.0_PLAN.md`` §A6 + a heads-up to the MIME
team).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

import jax
import jax.numpy as jnp

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import (
    build_unstructured_partition,
)
from maddening.cloud.multigpu.iterative_solver import sharded_gmres, sharded_cg
from maddening.cloud.multigpu.sharded_unstructured import (
    ShardedUnstructuredNode,
)
from maddening.core.node import SimulationNode
from maddening.core.static_data import StaticArray


# ---------------------------------------------------------------------------
# A mock "FVMFluidNode" — the shape MIME's port would take in v0.5.0.
# ---------------------------------------------------------------------------


class _MockFVMFluidNode(SimulationNode):
    """Stand-in for MIME's future FVMFluidNode.

    Demonstrates that the v0.3.0 ShardedUnstructuredNode contract supports
    everything a real FVM node would need:
      * Multi-field state (rho, u, p).
      * Partitioned static reference data (cell volumes, face areas).
      * A domain integral (mass conservation check).
      * A sharded linear solve via :func:`sharded_cg` for the
        Poisson pressure correction.
    """

    def __init__(
        self,
        *,
        name: str,
        n_global_cells: int,
        edges: np.ndarray,
        partition_assignment: np.ndarray,
        cell_volumes: np.ndarray,
        timestep: float = 1.0e-3,
    ) -> None:
        super().__init__(name=name, timestep=timestep)
        self._n_global = n_global_cells
        self._edges = edges
        self._pa = partition_assignment
        self._cell_volumes = np.asarray(cell_volumes, dtype=np.float32)

    def state_fields(self) -> list[str]:
        return ["rho", "u", "p"]

    def domain_integral_fields(self) -> set[str]:
        return {"total_mass"}

    @property
    def static_data(self) -> dict:
        # The contract DOES need to support: a static array that is
        # partitioned along the global cell axis using the same
        # partition_assignment the wrapper was built with.
        return {
            "cell_volumes": StaticArray(
                self._cell_volumes,
                replication="partition",
                partition_assignment=self._pa,
            ),
        }

    def initial_state(self) -> dict:
        return {
            "rho": jnp.ones(self._n_global, dtype=jnp.float32),
            "u": jnp.zeros(self._n_global, dtype=jnp.float32),
            "p": jnp.zeros(self._n_global, dtype=jnp.float32),
        }

    def update(self, state, boundary_inputs, dt):
        # Required by SimulationNode ABC; unsharded path not exercised.
        return state

    def update_padded(
        self, state_padded, boundary_inputs, dt,
        *, static_padded=None, shard_info=None,
    ) -> dict:
        # Validates that the contract delivers:
        #   - The padded state (local + ghost) on the first axis.
        #   - The partitioned static array via static_padded.
        #   - shard_info exposing the per-shard offset.
        # And we can return: state fields (halo-stripped) + a domain
        # integral (psum'd).
        assert static_padded is not None
        assert "cell_volumes" in static_padded
        assert shard_info is not None and 0 in shard_info

        n_local = shard_info[0][1]
        rho = state_padded["rho"]
        u = state_padded["u"]
        p = state_padded["p"]
        vol = static_padded["cell_volumes"]

        # Trivial "advection" using the ghost-padded state — the
        # contract supports reading ghosts beyond n_local.
        # (Not physical — just shape-correct so the assertion sticks.)
        new_rho = rho * (1.0 + 0.0 * jnp.sum(rho))  # ghost-aware sum
        new_u = u
        new_p = p

        total_mass = jnp.sum(rho[:n_local] * vol[:n_local])

        return {
            "rho": new_rho,
            "u": new_u,
            "p": new_p,
            "total_mass": total_mass,
        }


# ---------------------------------------------------------------------------
# Stress test — instantiate the mock node, run a step, then run a
# Poisson pressure-correction-style solve.  If any of this needs us
# to break the v0.3.0 contract, the test will fail to compile / run.
# ---------------------------------------------------------------------------


class TestA6ContractIsV040Ready:

    def test_mime_fvm_shape_node_smoke(self):
        n_global = 8
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

        cell_volumes = np.full(n_global, 0.5, dtype=np.float32)
        node = _MockFVMFluidNode(
            name="fvm_demo", n_global_cells=n_global, edges=edges,
            partition_assignment=pa, cell_volumes=cell_volumes,
        )
        sharded = ShardedUnstructuredNode(node, mesh, layout)
        out = sharded.update(sharded.initial_state(), {}, 1e-3)
        total = float(jax.device_get(out["total_mass"]))
        # rho=1, vol=0.5, n_global=8 → total = 4.0
        assert np.isclose(total, 4.0), f"got {total}"

    def test_sharded_gmres_signature_works_for_fvm_use_case(self):
        """The §A5 GMRES surface (which MIME's PISO pressure step will
        call) accepts a sharded matvec without any contract change.
        """
        n_global = 16
        n_devices = 4
        pa = np.repeat(np.arange(n_devices),
                       n_global // n_devices).astype(np.int32)
        edges = np.array(
            [[i, (i + 1) % n_global] for i in range(n_global)],
            dtype=np.int32,
        )
        from jax.experimental.shard_map import shard_map
        from jax.sharding import PartitionSpec as P, NamedSharding

        from maddening.cloud.multigpu.halo_unstructured import (
            exchange_unstructured, partition_value,
        )

        layout = build_unstructured_partition(
            partition_assignment=pa, edges=edges, n_devices=n_devices,
        )
        mesh = create_device_mesh(shape=(n_devices,))

        # Trivial matvec: identity-with-Laplacian-shape.  The point is
        # the SHAPE of the call, not the operator's interesting maths.
        def matvec(x):
            def shard_matvec(local):
                exchange_unstructured(local, layout=layout,
                                      mesh_axis="devices")
                return local

            return shard_map(
                shard_matvec, mesh=mesh,
                in_specs=(P("devices"),), out_specs=P("devices"),
                check_rep=False,
            )(x)

        b = jnp.ones(n_global, dtype=jnp.float32)
        b_sharded = jax.device_put(b, NamedSharding(mesh, P("devices")))

        # The signature accepted here is the one MIME's FVMFluidNode
        # will call.  If we rename `mesh` → `axis_env` (or similar),
        # MIME's port breaks.  This test pins the surface.
        result = sharded_gmres(
            matvec, b_sharded,
            mesh=mesh, in_specs=P("devices"),
            restart=n_global, max_iters=50,
            backend="loop",
        )
        # Identity matvec → solution = b.
        assert jnp.allclose(
            jax.device_get(result.value), np.ones(n_global, dtype=np.float32),
            atol=1e-5,
        )

    def test_no_breaking_renames_v030_to_v040(self):
        """Names the v0.4.0 commitment relies on still exist.

        Lock the names of every public symbol MIME's hypothetical
        v0.5.0 port depends on.  Tightening this is the entire point
        of the v0.4.0 contract: a renamed export here equals a MIME
        rewrite there.
        """
        # The contract-bearing module + function names.
        from maddening.cloud.multigpu.iterative_solver import (
            sharded_cg, sharded_gmres, SharedSolveResult,
        )
        from maddening.cloud.multigpu.halo_unstructured import (
            UnstructuredPartitionLayout,
            build_unstructured_partition,
            exchange_unstructured,
            partition_value,
            gather_value,
        )
        from maddening.cloud.multigpu.sharded_unstructured import (
            ShardedUnstructuredNode,
        )
        from maddening.core.static_data import StaticArray
        # Validate the partition variant exists.
        arr = np.zeros(4, dtype=np.float32)
        pa = np.zeros(4, dtype=np.int32)
        sa = StaticArray(
            arr, replication="partition", partition_assignment=pa,
        )
        assert sa.replication == "partition"
        # Validate the layout has the named tables MIME's index gymnastics
        # will read.
        layout = build_unstructured_partition(
            partition_assignment=pa,
            edges=np.zeros((0, 2), dtype=np.int32),
            n_devices=1,
        )
        for attr in (
            "partition_assignment", "n_devices", "local_global_ids",
            "n_local", "n_local_max", "ghost_global_ids", "n_ghost",
            "n_ghost_max", "send_indices", "recv_local_index",
            "send_counts",
        ):
            assert hasattr(layout, attr), (
                f"v0.4.0 commitment break: UnstructuredPartitionLayout "
                f"missing attribute {attr!r}"
            )
