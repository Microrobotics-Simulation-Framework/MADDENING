"""Tests for ShardedNode data-parallel wrapper."""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax
import jax.numpy as jnp
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedNode
from maddening.core.node import SimulationNode

_HAS_2_DEVICES = len(jax.devices()) >= 2
_SKIP_MSG = "Requires >=2 JAX devices"


# -- Test nodes -----------------------------------------------------------

class PointwiseNode(SimulationNode):
    """A node where each element is independent (no neighbor access)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def requires_halo(self) -> bool:
        return False

    def initial_state(self):
        n = self.params.get("n_elements", 100)
        return {
            "values": jnp.zeros(n),
            "velocities": jnp.ones(n) * 0.5,
        }

    def update(self, state, boundary_inputs, dt):
        force = boundary_inputs.get("force", jnp.zeros_like(state["values"]))
        new_vel = state["velocities"] + force * dt
        new_val = state["values"] + new_vel * dt
        return {"values": new_val, "velocities": new_vel}


class StencilNode(SimulationNode):
    """A node that accesses spatial neighbors (requires halo)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def requires_halo(self) -> bool:
        return True

    def initial_state(self):
        return {"field": jnp.zeros(50)}

    def update(self, state, boundary_inputs, dt):
        # Uses neighbor access (would be incorrect if sharded without halos)
        f = state["field"]
        laplacian = jnp.roll(f, 1) - 2 * f + jnp.roll(f, -1)
        return {"field": f + 0.1 * laplacian * dt}


# -- Tests ----------------------------------------------------------------

class TestShardedNodeConstruction:
    def test_rejects_stencil_node(self):
        mesh = create_device_mesh(n_devices=1)
        node = StencilNode(name="stencil", timestep=0.01)
        with pytest.raises(ValueError, match="requires halo exchange"):
            ShardedNode(node, mesh)

    def test_accepts_pointwise_node(self):
        mesh = create_device_mesh(n_devices=1)
        node = PointwiseNode(name="pw", timestep=0.01, n_elements=100)
        sharded = ShardedNode(node, mesh)
        assert sharded.name == "pw"
        assert sharded.requires_halo is False

    def test_shard_axes_int_converted_to_tuple(self):
        mesh = create_device_mesh(n_devices=1)
        node = PointwiseNode(name="pw", timestep=0.01, n_elements=100)
        sharded = ShardedNode(node, mesh, shard_axes=0)
        assert sharded._shard_axes == (0,)

    def test_shard_axes_tuple_accepted(self):
        mesh = create_device_mesh(n_devices=1)
        node = PointwiseNode(name="pw", timestep=0.01, n_elements=100)
        sharded = ShardedNode(node, mesh, shard_axes=(0,))
        assert sharded._shard_axes == (0,)

    def test_multi_axis_raises_not_implemented(self):
        mesh = create_device_mesh(n_devices=1)
        node = PointwiseNode(name="pw", timestep=0.01, n_elements=100)
        with pytest.raises(NotImplementedError, match="Multi-axis"):
            ShardedNode(node, mesh, shard_axes=(0, 1))


class TestShardedNodeRequiresHaloOnRealNodes:
    def test_heat_node_requires_halo(self):
        from maddening.nodes.heat import HeatNode
        node = HeatNode("h", timestep=0.01, n_cells=20)
        assert node.requires_halo is True

    def test_ball_node_no_halo(self):
        from maddening.nodes.ball import BallNode
        node = BallNode("b", timestep=0.01)
        assert node.requires_halo is False

    def test_spring_node_no_halo(self):
        from maddening.nodes.spring import SpringDamperNode
        node = SpringDamperNode("s", timestep=0.01, stiffness=100.0)
        assert node.requires_halo is False


@pytest.mark.skipif(not _HAS_2_DEVICES, reason=_SKIP_MSG)
class TestShardedNodeCorrectness:
    def test_update_matches_unsharded(self):
        mesh = create_device_mesh(n_devices=2)
        node = PointwiseNode(name="pw", timestep=0.01, n_elements=100)
        sharded = ShardedNode(node, mesh)

        # Unsharded reference
        state = node.initial_state()
        bi = {"force": jnp.ones(100) * 0.1}
        ref_result = node.update(state, bi, 0.01)

        # Sharded
        s_state = sharded.initial_state()
        s_result = sharded.update(s_state, bi, 0.01)

        for field in ref_result:
            assert jnp.allclose(
                ref_result[field], s_result[field], atol=1e-6,
            ), f"Mismatch in {field}"

    def test_initial_state_sharded(self):
        mesh = create_device_mesh(n_devices=2)
        node = PointwiseNode(name="pw", timestep=0.01, n_elements=100)
        sharded = ShardedNode(node, mesh)

        state = sharded.initial_state()
        assert state["values"].shape == (100,)
        assert state["velocities"].shape == (100,)
