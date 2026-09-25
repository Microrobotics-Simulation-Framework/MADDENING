"""A domain integral carried in ``ShardedStencilNode``'s state is not a
grid field.

A node that declares its integral's initial value -- which a graph's scan
carry needs -- had that value placed like a grid field: split along the
sharded spatial axis and halo-padded.  A scalar escaped by luck (no axis
to split); a 3-vector drag on two devices was refused as "3 cells" (and
failed with ``IndivisibleError`` from ``device_put`` before 0.4.0), and a
per-shard (unreduced) integral fed back by a graph was refused as a grid
"changed by a parameter".  The integral is now placed as the step returns
it -- replicated once reduced, stacked along its unreduced mesh axes
otherwise -- and reaches ``update_padded`` unpadded.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode

_HAS_4 = len(jax.devices()) >= 4
pytestmark = pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")


class _Rod(SimulationNode):
    """``x <- x + dt`` on 8 cells, halo 1, with a domain integral declared in
    ``initial_state``: a 3-vector ``force = sum(x) * [1, 2, 3]`` or, with
    ``stacked``, a per-shard scalar ``total``."""

    def __init__(self, stacked=False):
        super().__init__(name="rod", timestep=0.1)
        self._stacked = stacked
        #: Shapes of the integral ``update_padded`` was handed (trace time).
        self.seen = []

    def halo_width(self):
        return {0: 1}

    def halo_boundary(self):
        return "periodic"

    def _key(self):
        return "total" if self._stacked else "force"

    def _integral(self, total):
        return total if self._stacked else total * jnp.arange(1.0, 4.0, dtype=jnp.float32)

    def initial_state(self):
        zero = jnp.float32(0.0) if self._stacked else jnp.zeros(3, jnp.float32)
        return {"x": jnp.arange(1.0, 9.0, dtype=jnp.float32), self._key(): zero}

    def state_fields(self):
        return ["x"]

    def domain_integral_fields(self):
        return {self._key()}

    def domain_integral_axes(self):
        return {"total": ()} if self._stacked else {}

    def update(self, state, boundary_inputs, dt):
        x = state["x"] + dt
        return {"x": x, self._key(): self._integral(jnp.sum(x))}

    def update_padded(self, state_padded, boundary_inputs, dt, *,
                      static_padded=None, shard_info=None):
        if self._key() in state_padded:
            self.seen.append(tuple(state_padded[self._key()].shape))
        x = state_padded["x"] + dt
        return {"x": x, self._key(): self._integral(jnp.sum(x[1:-1]))}


def _graph(node):
    gm = GraphManager()
    gm.add_node(node)
    gm.compile()
    return gm


@pytest.mark.parametrize("n_devices", [2, 4])
def test_a_vector_domain_integral_in_the_state_runs_in_a_graph_like_the_unsharded_node(n_devices):
    """A 3-vector on 2 and 4 devices: refused as "3 cells" before this."""
    ref = _graph(_Rod())
    ref.run_scan(3)
    inner = _Rod()
    gm = _graph(ShardedStencilNode(inner, create_device_mesh(shape=(n_devices,)),
                                   {"devices": 0}, boundary="periodic"))
    gm.run_scan(3)
    # Handed to update_padded as it is, not split and halo-padded (5,).
    assert inner.seen and set(inner.seen) == {(3,)}
    np.testing.assert_allclose(np.asarray(gm.get_node_state("rod")["force"]),
                               np.asarray(ref.get_node_state("rod")["force"]), rtol=1e-6)


def test_a_per_shard_domain_integral_fed_back_by_a_graph_steps_again():
    """Stacked along the mesh axis, as the step returns it; before this the
    second step compared its (2,) shape with the node's () and refused it
    as a grid changed by a parameter."""
    inner = _Rod(stacked=True)
    gm = _graph(ShardedStencilNode(inner, create_device_mesh(shape=(2,)),
                                   {"devices": 0}, boundary="periodic"))
    for _ in range(3):
        gm.step()
    # This shard's slice of the stacked value, unpadded.
    assert inner.seen and set(inner.seen) == {(1,)}
    # blocks [1..4] and [5..8], each cell + 0.3
    np.testing.assert_allclose(np.asarray(gm.get_node_state("rod")["total"]),
                               [10.0 + 1.2, 26.0 + 1.2], rtol=1e-6)
    gm.run_scan(2)
    np.testing.assert_allclose(np.asarray(gm.get_node_state("rod")["total"]),
                               [10.0 + 2.0, 26.0 + 2.0], rtol=1e-6)


def test_the_initial_value_of_an_integral_is_placed_as_the_step_returns_it():
    """Replicated when reduced; stacked, one row per shard, when not."""
    mesh = create_device_mesh(shape=(2,))
    reduced = ShardedStencilNode(_Rod(), mesh, {"devices": 0}, boundary="periodic")
    stacked = ShardedStencilNode(_Rod(stacked=True), mesh, {"devices": 0},
                                 boundary="periodic")
    force = reduced.initial_state()["force"]
    assert force.shape == (3,) and force.sharding.is_fully_replicated
    total = stacked.initial_state()["total"]
    assert total.shape == (2,) and not total.sharding.is_fully_replicated
