"""A ``ShardedStencilNode`` wrapping another keeps the node's parameters.

Nesting is supported (the constructor checks the two halo fills agree),
and the outer wrapper decides whether the node takes ``params`` by asking
the inner wrapper about ``update_padded``.  The inner wrapper used to
answer from its own ``update_padded`` signature, which had no ``params``
keyword, so a ``HeatNode`` wrapped twice reported ``params_pytree() ==
{}``: ``gm.params`` had no entry for it, a ``run_scan(params=)`` write of
``thermal_diffusivity`` moved nothing (0.0 K, against 30.9 K for the
singly wrapped rod), and d(loss)/d(alpha) had nothing to differentiate.
The inner wrapper now answers for its node and forwards ``params``.
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
from maddening.core.simulation.hybrid_node import HybridNode
from maddening.nodes.heat import HeatNode

_HAS_2 = len(jax.devices()) >= 2
pytestmark = pytest.mark.skipif(not _HAS_2, reason="needs 2 CPU-virtual devices")

N_CELLS = 16
DT = 0.2 * (1.0 / N_CELLS) ** 2 / 0.01
T0 = (300.0 + 40.0 * np.sin(np.linspace(0.0, 3.0, N_CELLS))).tolist()
LEAVES = {"initial_temperature", "length", "thermal_diffusivity"}


def _rod():
    return HeatNode("rod", DT, n_cells=N_CELLS, initial_temperature=T0)


def _mesh():
    return create_device_mesh(shape=(2,))


BUILDS = {
    "unwrapped": lambda: _rod(),
    "wrapped": lambda: ShardedStencilNode(_rod(), _mesh(), {"devices": 0}),
    "wrapped-twice": lambda: ShardedStencilNode(
        ShardedStencilNode(_rod(), _mesh(), {"devices": 0}), _mesh(), {"devices": 0}),
    "hybrid-of-wrapped": lambda: HybridNode(
        ShardedStencilNode(_rod(), _mesh(), {"devices": 0}), lambda s, b, d: {}),
    "hybrid-of-wrapped-twice": lambda: HybridNode(
        ShardedStencilNode(ShardedStencilNode(_rod(), _mesh(), {"devices": 0}),
                           _mesh(), {"devices": 0}),
        lambda s, b, d: {}),
}


def _graph(build):
    gm = GraphManager()
    gm.add_node(build())
    gm.add_external_input("rod", "left_temperature")
    gm.compile()
    return gm


def _final_temperature(build, alpha_scale):
    gm = _graph(build)            # a fresh graph: run_scan moves the state
    params = jax.tree.map(lambda x: x, gm.params)
    leaf = params["nodes"]["rod"]
    leaf["thermal_diffusivity"] = leaf["thermal_diffusivity"] * alpha_scale
    gm.run_scan(20, params=params)
    return np.asarray(gm.get_node_state("rod")["temperature"])


@pytest.fixture(scope="module")
def reference():
    """The unwrapped rod at the constructed and the scaled diffusivity."""
    return {s: _final_temperature(BUILDS["unwrapped"], s) for s in (1.0, 1.5)}


@pytest.mark.parametrize("name", [n for n in BUILDS if n != "unwrapped"])
def test_every_wrapping_exposes_the_rods_parameters(name):
    gm = _graph(BUILDS[name])
    assert set(gm.params["nodes"]["rod"]) == LEAVES


@pytest.mark.parametrize("name", [n for n in BUILDS if n != "unwrapped"])
def test_a_params_write_moves_every_wrapping_as_it_moves_the_rod(name, reference):
    moved = {s: _final_temperature(BUILDS[name], s) for s in (1.0, 1.5)}
    for scale in (1.0, 1.5):
        np.testing.assert_allclose(moved[scale], reference[scale], rtol=0, atol=1e-3)
    # The write is live: 1.5x the diffusivity is a different rod.
    assert np.max(np.abs(moved[1.5] - moved[1.0])) > 1.0


def test_the_inner_wrapper_answers_for_its_node_about_update_padded():
    inner = ShardedStencilNode(_rod(), _mesh(), {"devices": 0})
    assert inner.accepts_params(method="update_padded") is True
    assert inner.accepts_params() is True


def test_a_gradient_reaches_the_rod_through_two_wrappers():
    gm = _graph(BUILDS["wrapped-twice"])
    step = gm._build_step_fn()
    ext = gm._default_external_inputs()

    def loss(alpha):
        p = jax.tree.map(lambda x: x, gm.params)
        p["nodes"]["rod"]["thermal_diffusivity"] = alpha
        final, _ = jax.lax.scan(lambda s, _: (step(s, ext, p), None), gm._state,
                                None, length=5)
        return jnp.sum(final["rod"]["temperature"] ** 2)

    g = jax.grad(loss)(jnp.float32(0.01))
    assert bool(jnp.isfinite(g)) and float(g) != 0.0


class _Legacy(SimulationNode):
    """A stencil node on the three-argument contract: no ``params`` anywhere."""

    def __init__(self):
        super().__init__(name="legacy", timestep=0.1, rate=2.0)

    def halo_width(self):
        return {0: 1}

    def initial_state(self):
        return {"x": jnp.zeros(8, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] + self.params["rate"] * dt}

    def update_padded(self, state_padded, boundary_inputs, dt, *,
                      static_padded=None, shard_info=None):
        return {"x": state_padded["x"] + self.params["rate"] * dt}


def test_a_node_without_params_stays_without_them_wrapped_twice():
    wrapped = ShardedStencilNode(
        ShardedStencilNode(_Legacy(), _mesh(), {"devices": 0}), _mesh(), {"devices": 0})
    assert wrapped.accepts_params() is False
    assert wrapped.params_pytree() == {}
    assert wrapped.param_specs() == {}
    out = wrapped.update(wrapped.initial_state(), {}, 0.1)
    np.testing.assert_allclose(np.asarray(out["x"]), 0.2, rtol=1e-6)
