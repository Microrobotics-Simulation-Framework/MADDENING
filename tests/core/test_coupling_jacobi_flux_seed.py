"""A Jacobi coupling pass seeds producers' fluxes the way Gauss-Seidel does.

``one_pass_jacobi`` computes every flux producer's flux from the previous
iterate before the pass.  It resolved each producer's inputs strictly and
with no flux dictionary, so a producer whose own input is another
producer's flux -- two slabs exchanging boundary fluxes -- raised
``KeyError`` naming the flux at trace.  ``one_pass_gs`` has seeded in two
sweeps (the first tolerating a missing flux) since before 0.4.0; the Jacobi
pass now does the same, but only where a producer reads a flux, so every
other Jacobi group compiles to the program it always did.
"""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.property.coupled_graphs import live_knobs


class Slab(SimulationNode):
    """``x = g * u + b`` and the flux ``q = 2 x`` (a function of state only).

    Counts its own ``update`` and ``compute_boundary_fluxes`` calls at
    trace time, which is how the number of flux sweeps per pass is read.
    """

    def __init__(self, name, timestep, *, g, b, x0=0.0):
        super().__init__(name, timestep, g=g, b=b)
        self._x0 = x0
        self.updates = 0
        self.fluxes = 0

    def initial_state(self):
        return {"x": jnp.asarray(self._x0, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.asarray(0.0, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        self.updates += 1
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs.get("u", jnp.asarray(0.0, jnp.float32))
        return {"x": (p["g"] * u + p["b"]).astype(jnp.float32)}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        self.fluxes += 1
        return {"q": jnp.float32(2.0) * state["x"]}


class Reader(Slab):
    """A :class:`Slab` that produces no flux."""

    compute_boundary_fluxes = SimulationNode.compute_boundary_fluxes


#: ``x0 = 0.2 u0 + 1`` and ``x1 = -0.3 u1 + 0.5``, coupled three ways.  A
#: flux edge carries ``q = 2 x``; ``g1`` is a :class:`Reader` (no flux) in
#: the last, so there no producer reads a flux.
_TOPOLOGIES = {
    # both producers read the other's flux
    "mutual": (Slab, ("g0", "g1", "q"), ("g1", "g0", "q")),
    # a producer reads a flux one way (this raised too: any producer
    # reading any flux did, whatever the order)
    "one-way": (Slab, ("g0", "g1", "q"), ("g1", "g0", "x")),
    # the flux's consumer produces none; the producer reads state
    "consumer-produces-none": (Reader, ("g0", "g1", "q"), ("g1", "g0", "x")),
}


def _graph(mode, *, topology="mutual", solver="ift", **group):
    second, e01, e10 = _TOPOLOGIES[topology]
    gm = GraphManager()
    gm.add_node(Slab("g0", 1.0, g=0.2, b=1.0, x0=0.3))
    gm.add_node(second("g1", 1.0, g=-0.3, b=0.5, x0=-0.4))
    gm.add_edge(e01[0], e01[1], e01[2], "u")
    gm.add_edge(e10[0], e10[1], e10[2], "u")
    cfg = dict(max_iterations=40, tolerance=1e-6, iteration_mode=mode, solver=solver,
               diagnostics=True, **group)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "CouplingGroup solver='fori' is deprecated",
                                DeprecationWarning)
        gm.add_coupling_group(["g0", "g1"], **live_knobs(cfg))
    gm.compile()
    return gm


def _exact(topology="mutual"):
    """``x0 = a x1 + 1``, ``x1 = c x0 + 0.5`` with the flux's factor 2 folded in."""
    _second, e01, e10 = _TOPOLOGIES[topology]
    c = -0.3 * (2.0 if e01[2] == "q" else 1.0)
    a = 0.2 * (2.0 if e10[2] == "q" else 1.0)
    x0 = (1.0 + a * 0.5) / (1.0 - a * c)
    return np.array([x0, c * x0 + 0.5])


@pytest.mark.parametrize("topology", ["mutual", "one-way"])
@pytest.mark.parametrize("accel", ["none", "aitken", "iqn-ils"])
@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_flux_reading_producers_step_under_jacobi_to_the_gauss_seidel_fixed_point(
        solver, accel, topology):
    states = {}
    for mode in ("jacobi", "gauss-seidel"):
        gm = _graph(mode, topology=topology, solver=solver, acceleration=accel)
        gm.step()
        assert gm.coupling_diagnostics()["g0+g1"]["converged"], mode
        states[mode] = np.array([float(gm.get_node_state(n)["x"]) for n in ("g0", "g1")])
    np.testing.assert_allclose(states["jacobi"], _exact(topology), rtol=1e-5)
    np.testing.assert_allclose(states["gauss-seidel"], _exact(topology), rtol=1e-5)


@pytest.mark.parametrize("topology, sweeps", [
    ("mutual", 2), ("one-way", 2), ("consumer-produces-none", 1)])
def test_a_jacobi_pass_seeds_in_two_sweeps_only_where_a_producer_reads_a_flux(topology, sweeps):
    """Per traced pass, each producer's flux is computed once per seed sweep.

    The Jacobi pass computes fluxes only in its seed (never after an
    update), so ``fluxes == sweeps * updates`` for every producer.  One
    sweep where no producer reads a flux is the program the Jacobi pass
    always compiled to; two where one does is the Gauss-Seidel seed.
    """
    gm = _graph("jacobi", topology=topology, solver="fori")
    producers = [n for n in ("g0", "g1") if type(gm.get_node(n)) is Slab]
    for name in producers:      # ``compile()``'s own probes are not passes
        gm.get_node(name).updates = gm.get_node(name).fluxes = 0
    gm.step()
    np.testing.assert_allclose(
        [float(gm.get_node_state(n)["x"]) for n in ("g0", "g1")], _exact(topology), rtol=1e-5)
    for name in producers:
        node = gm.get_node(name)
        assert node.updates > 0
        assert node.fluxes == sweeps * node.updates, (name, node.fluxes, node.updates)


def test_a_flux_reading_producer_beside_a_non_producer_steps_under_jacobi():
    """A three-node chain: ``r -> g0`` (state), ``g0.q -> g1``, ``g1.q -> g0``."""
    gm = GraphManager()
    gm.add_node(Reader("r", 1.0, g=0.5, b=0.1))
    gm.add_node(Slab("g0", 1.0, g=0.2, b=1.0))
    gm.add_node(Slab("g1", 1.0, g=-0.3, b=0.5))
    gm.add_edge("g1", "r", "x", "u")
    gm.add_edge("g0", "g1", "q", "u")
    gm.add_edge("g1", "g0", "q", "u")
    gm.add_coupling_group(["r", "g0", "g1"], max_iterations=40, tolerance=1e-6,
                          iteration_mode="jacobi", diagnostics=True)
    gm.compile()
    gm.step()
    assert gm.coupling_diagnostics()["g0+g1+r"]["converged"]
    np.testing.assert_allclose(
        [float(gm.get_node_state(n)["x"]) for n in ("g0", "g1")], _exact(), rtol=1e-5)
