"""Tests for ShardedPointwiseNode data-parallel wrapper."""

import jax
import jax.numpy as jnp
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode
from maddening.core.node import SimulationNode

_HAS_2_DEVICES = len(jax.devices()) >= 2
_SKIP_MSG = "Requires >=2 JAX devices"


# -- Test nodes -----------------------------------------------------------

class PointwiseNode(SimulationNode):
    """A node where each element is independent (no neighbor access)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def halo_width(self) -> dict[int, int]:
        return {}

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

    def halo_width(self) -> dict[int, int]:
        return {0: 1}

    def initial_state(self):
        return {"field": jnp.zeros(50)}

    def update(self, state, boundary_inputs, dt):
        # Uses neighbor access (would be incorrect if sharded without halos)
        f = state["field"]
        laplacian = jnp.roll(f, 1) - 2 * f + jnp.roll(f, -1)
        return {"field": f + 0.1 * laplacian * dt}


# -- Tests ----------------------------------------------------------------

class TestShardedPointwiseNodeConstruction:
    def test_rejects_stencil_node(self):
        mesh = create_device_mesh(n_devices=1)
        node = StencilNode(name="stencil", timestep=0.01)
        with pytest.raises(ValueError, match="ShardedStencilNode"):
            ShardedPointwiseNode(node, mesh)

    def test_accepts_pointwise_node(self):
        mesh = create_device_mesh(n_devices=1)
        node = PointwiseNode(name="pw", timestep=0.01, n_elements=100)
        sharded = ShardedPointwiseNode(node, mesh)
        assert sharded.name == "pw"
        # Pointwise nodes report empty halo_width().
        assert sharded.halo_width() == {}

    def test_shard_axes_int_converted_to_tuple(self):
        mesh = create_device_mesh(n_devices=1)
        node = PointwiseNode(name="pw", timestep=0.01, n_elements=100)
        sharded = ShardedPointwiseNode(node, mesh, shard_axes=0)
        assert sharded._shard_axes == (0,)

    def test_shard_axes_tuple_accepted(self):
        mesh = create_device_mesh(n_devices=1)
        node = PointwiseNode(name="pw", timestep=0.01, n_elements=100)
        sharded = ShardedPointwiseNode(node, mesh, shard_axes=(0,))
        assert sharded._shard_axes == (0,)

    def test_multi_axis_raises_not_implemented(self):
        mesh = create_device_mesh(n_devices=1)
        node = PointwiseNode(name="pw", timestep=0.01, n_elements=100)
        with pytest.raises(NotImplementedError, match="Multi-axis"):
            ShardedPointwiseNode(node, mesh, shard_axes=(0, 1))

    @pytest.mark.parametrize("shard_axis", [0, 2])
    def test_a_node_with_nothing_to_shard_is_refused_naming_its_fields(self, shard_axis):
        """A field too short to have the shard axis is replicated; a node
        none of whose fields has it was accepted and ran whole on every
        device, sharding nothing and saying nothing (a spring's 0-d state;
        ``shard_axes=(2,)`` on 1-D fields).  Refused at construction."""
        from maddening.nodes.spring import SpringDamperNode

        mesh = create_device_mesh(n_devices=1)
        spring = SpringDamperNode("s", timestep=0.01, stiffness=30.0)
        with pytest.raises(ValueError) as excinfo:
            ShardedPointwiseNode(spring, mesh, shard_axes=(shard_axis,))
        message = str(excinfo.value)
        assert f"cannot shard SpringDamperNode 's' along array axis {shard_axis}" in message
        assert "'position' (0-d), 'velocity' (0-d)" in message
        assert "nothing would be sharded" in message
        vector = PointwiseNode(name="pw", timestep=0.01, n_elements=8)
        with pytest.raises(ValueError, match=r"'values' \(1-d\), 'velocities' \(1-d\)"):
            ShardedPointwiseNode(vector, mesh, shard_axes=(2,))
        ShardedPointwiseNode(vector, mesh, shard_axes=(0,))    # the axis they have

    def test_one_field_with_the_axis_is_enough(self):
        """A 0-d field beside a sharded one is replicated, as documented."""

        class Mixed(PointwiseNode):
            def initial_state(self):
                return {**super().initial_state(), "clock": jnp.float32(0.0)}

            def update(self, state, boundary_inputs, dt):
                return {**super().update(state, boundary_inputs, dt),
                        "clock": state["clock"] + dt}

        sharded = ShardedPointwiseNode(Mixed(name="m", timestep=0.01, n_elements=8),
                                       create_device_mesh(n_devices=1))
        assert sharded.initial_state()["clock"].shape == ()


class TestShardedPointwiseNodeHaloWidthOnRealNodes:
    def test_heat_node_has_halo(self):
        from maddening.nodes.heat import HeatNode
        node = HeatNode("h", timestep=0.01, n_cells=20)
        assert node.halo_width() == {0: 1}

    def test_ball_node_no_halo(self):
        from maddening.nodes.ball import BallNode
        node = BallNode("b", timestep=0.01)
        assert node.halo_width() == {}

    def test_spring_node_no_halo(self):
        from maddening.nodes.spring import SpringDamperNode
        node = SpringDamperNode("s", timestep=0.01, stiffness=100.0)
        assert node.halo_width() == {}


@pytest.mark.skipif(not _HAS_2_DEVICES, reason=_SKIP_MSG)
class TestShardedPointwiseNodeCorrectness:
    def test_update_matches_unsharded(self):
        mesh = create_device_mesh(n_devices=2)
        node = PointwiseNode(name="pw", timestep=0.01, n_elements=100)
        sharded = ShardedPointwiseNode(node, mesh)

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
        sharded = ShardedPointwiseNode(node, mesh)

        state = sharded.initial_state()
        assert state["values"].shape == (100,)
        assert state["velocities"].shape == (100,)
