"""Two sharding refusals that nothing pinned.

A mutation audit disabled each guard in the sharded wrappers and ran the
sharding tests: these two stayed green with their guard removed.

* ``ShardedPointwiseNode`` shards over a mesh axis named ``"devices"``;
  without the check, a mesh without one failed later inside JAX
  ("Resource axis: devices ... is not found in mesh"), naming neither the
  wrapper nor the way out.
* ``ShardedStencilNode`` reads the node's halo whenever it rebuilds its
  step (at construction and on every ``compile()``); a node whose
  ``halo_width()`` no longer lists a sharded axis failed with a bare
  ``KeyError: 0`` without the check.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode, ShardedStencilNode
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.heat import HeatNode

_HAS_4 = len(jax.devices()) >= 4
pytestmark = pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")


class _Decay(SimulationNode):
    """Pointwise ``x <- x - rate * x * dt`` on 8 cells."""

    def __init__(self):
        super().__init__(name="decay", timestep=0.1, rate=0.5)

    def initial_state(self):
        return {"x": jnp.ones(8, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] * (1.0 - self.params["rate"] * dt)}


def test_a_pointwise_wrapper_refuses_a_mesh_without_a_devices_axis():
    mesh = create_device_mesh(shape=(2, 2))
    assert "devices" not in mesh.axis_names
    with pytest.raises(ValueError, match=r"mesh axis named 'devices'") as exc:
        ShardedPointwiseNode(_Decay(), mesh)
    assert str(tuple(mesh.axis_names)) in str(exc.value)
    assert "create_device_mesh(shape=(n,))" in str(exc.value)


def test_a_pointwise_wrapper_takes_a_mesh_with_a_devices_axis():
    wrapped = ShardedPointwiseNode(_Decay(), create_device_mesh(shape=(4,)))
    out = wrapped.update(wrapped.initial_state(), {}, 0.1)
    assert float(out["x"][0]) == pytest.approx(0.95)


def _rod():
    return HeatNode("rod", 1e-4, n_cells=16, thermal_diffusivity=0.1)


def test_a_halo_that_drops_the_sharded_axis_is_refused_on_rebuild():
    rod = _rod()
    wrapped = ShardedStencilNode(rod, create_device_mesh(shape=(2,)), {"devices": 0})
    rod.halo_width = lambda: {1: 1}
    with pytest.raises(ValueError, match="no longer lists") as exc:
        wrapped.invalidate_static_cache()
    assert "axes [0]" in str(exc.value) and "Rebuild the wrapper" in str(exc.value)


def test_a_halo_that_drops_the_sharded_axis_is_refused_by_a_recompile():
    rod = _rod()
    gm = GraphManager()
    gm.add_node(ShardedStencilNode(rod, create_device_mesh(shape=(2,)), {"devices": 0}))
    gm.compile()
    rod.halo_width = lambda: {}
    with pytest.raises(ValueError, match="no longer lists"):
        gm.compile()
