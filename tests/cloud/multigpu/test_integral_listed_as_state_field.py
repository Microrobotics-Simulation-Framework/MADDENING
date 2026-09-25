"""A domain integral that ``state_fields()`` also lists is still an integral.

The default ``state_fields()`` lists every ``initial_state`` key, so a node
that declares its integral's initial value lists the integral as a state
field unless it overrides the method.  Both sharded wrappers classified a
step's outputs as state fields first: ``ShardedStencilNode`` stripped the
integral like a grid field and never summed it, and ``shard_map`` refused
the unreplicated result with an error about ``out_specs``;
``ShardedUnstructuredNode`` sliced a 0-d value.  Both now classify a
declared integral first.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
from maddening.core.node import SimulationNode

_HAS_2 = len(jax.devices()) >= 2
pytestmark = pytest.mark.skipif(not _HAS_2, reason="needs 2 CPU-virtual devices")


class _Cells(SimulationNode):
    """``x <- x + dt`` on 8 cells with ``total = sum(x)`` in its state, and
    no ``state_fields()`` override: the default lists ``total`` too."""

    def __init__(self, halo):
        super().__init__(name="cells", timestep=0.1)
        self._halo = halo

    def halo_width(self):
        return {0: 1} if self._halo else {}

    def initial_state(self):
        return {"x": jnp.arange(1.0, 9.0, dtype=jnp.float32), "total": jnp.float32(0.0)}

    def domain_integral_fields(self):
        return {"total"}

    def update(self, state, boundary_inputs, dt):
        x = state["x"] + dt
        return {"x": x, "total": jnp.sum(x)}

    def update_padded(self, state_padded, boundary_inputs, dt, *,
                      static_padded=None, shard_info=None):
        if self._halo:
            x = state_padded["x"] + dt
            return {"x": x, "total": jnp.sum(x[1:-1])}
        _, n_local_max = shard_info[0]
        x = state_padded["x"][:n_local_max] + dt
        owned = jnp.where(jnp.arange(n_local_max) < shard_info["n_local"], x, 0.0)
        return {"x": x, "total": jnp.sum(owned)}


def _layout():
    pa = (np.arange(8) * 2 // 8).astype(np.int32)
    edges = np.stack([np.arange(7), np.arange(1, 8)], axis=1).astype(np.int32)
    return build_unstructured_partition(partition_assignment=pa, edges=edges, n_devices=2)


@pytest.mark.parametrize("kind", ["stencil", "unstructured"])
def test_an_integral_listed_in_state_fields_is_summed_over_the_shards(kind):
    node = _Cells(halo=kind == "stencil")
    assert "total" in node.state_fields()
    mesh = create_device_mesh(shape=(2,))
    if kind == "stencil":
        wrapped = ShardedStencilNode(node, mesh, {"devices": 0}, boundary="periodic")
    else:
        wrapped = ShardedUnstructuredNode(node, mesh, _layout())
    state = wrapped.initial_state()
    for _ in range(2):
        state = wrapped.update(state, {}, 0.1)
    assert float(np.asarray(state["total"])) == pytest.approx(36.0 + 8 * 0.2, rel=1e-6)
