"""The graph's coupling hooks read the one params signature rule.

"Takes ``params``" means an explicit ``params`` keyword *or* a
``**kwargs`` that would forward it
(:func:`maddening.core.node._signature_takes_params`).  ``accepts_params``,
the integrators, the implicit solver and the verification battery read
that rule; the graph's probes for ``compute_interface_correction`` and
``compute_boundary_fluxes``, and ``HybridNode``'s forwarding to its physics
node, still accepted only the explicit keyword.  So a hook written as
``def compute_boundary_fluxes(self, *args, **kwargs): return
super().compute_boundary_fluxes(*args, **kwargs)`` was called *without*
``params``: it corrected the interface cells, or delivered the flux, from
the constructor's constants while ``update`` used the calibrated ones --
silently, with the verification battery calling the same node compliant.

Measured before the fix, through a real graph: a pair of Dirichlet-coupled
rods with a forwarding ``compute_interface_correction`` and ``length``
calibrated from 1.0 to 2.0 put their interface cells **11.39 K** from rods
built with ``length=2.0`` (both solvers, bare and inside ``HybridNode``); a
forwarding spring with ``stiffness`` calibrated from 30 to 300 delivered
**-23.1** on its flux edge instead of **-285.0**.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core import graph_manager as gm_mod
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode, _signature_takes_params
from maddening.core.simulation import hybrid_node as hybrid_mod
from maddening.core.simulation.hybrid_node import HybridNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode
from maddening.testing import verification as verification_mod


# ------------------------------------------------------------------
# Three spellings of each hook
# ------------------------------------------------------------------

class ForwardingHeat(HeatNode):
    """``compute_interface_correction`` behind ``**kwargs``, forwarded."""

    def compute_interface_correction(self, *args, **kwargs):  # noqa: D102
        return super().compute_interface_correction(*args, **kwargs)


class LegacyHeat(HeatNode):
    """``compute_interface_correction`` with neither spelling."""

    def compute_interface_correction(self, pre_state, boundary_inputs, dt):  # noqa: D102
        return super().compute_interface_correction(pre_state, boundary_inputs, dt)


class ForwardingSpring(SpringDamperNode):
    """``compute_boundary_fluxes`` behind ``**kwargs``, forwarded."""

    def compute_boundary_fluxes(self, *args, **kwargs):  # noqa: D102
        return super().compute_boundary_fluxes(*args, **kwargs)


class LegacySpring(SpringDamperNode):
    """``compute_boundary_fluxes`` with neither spelling."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):  # noqa: D102
        return super().compute_boundary_fluxes(state, boundary_inputs, dt)


class Sink(SimulationNode):
    """``x += force * dt`` for a scalar flux input."""

    def initial_state(self):
        return {"x": jnp.array(0.0, jnp.float32)}

    def update(self, s, bi, dt, *, params=None):
        return {"x": s["x"] + bi.get("force", jnp.array(0.0, jnp.float32)) * dt}

    def boundary_input_spec(self):
        return {"force": BoundaryInputSpec(shape=(), description="force")}


def _wrap(node, wrapped: bool):
    return HybridNode(node, lambda s, bi, dt: {}) if wrapped else node


WRAPS = [pytest.param(False, id="bare"), pytest.param(True, id="in-HybridNode")]


# ------------------------------------------------------------------
# compute_interface_correction, through a coupled graph step
# ------------------------------------------------------------------

def _rods(cls, length, *, wrapped, solver):
    """A fresh pair of Dirichlet-coupled rods.  ``run_scan`` advances the
    graph's own state, so every measurement builds its own."""
    gm = GraphManager()
    for name, t0 in (("rod_a", 100.0), ("rod_b", 0.0)):
        gm.add_node(_wrap(cls(name=name, timestep=0.001, n_cells=10,
                              thermal_diffusivity=0.5, length=length,
                              initial_temperature=t0), wrapped))
    gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                transform=lambda T: T[-1])
    gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                transform=lambda T: T[0])
    gm.add_coupling_group(["rod_a", "rod_b"], solver=solver, max_iterations=10)
    gm.compile()
    return gm


def _calibrated_rods(cls, *, wrapped, solver):
    gm = _rods(cls, 1.0, wrapped=wrapped, solver=solver)
    for n in ("rod_a", "rod_b"):
        gm.params["nodes"][n]["length"] = jnp.asarray(2.0, jnp.float32)
    return gm.run_scan(2)


@pytest.mark.parametrize("wrapped", WRAPS)
@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_a_var_keyword_interface_correction_receives_the_calibrated_params(solver, wrapped):
    """The forwarding override puts the interface cells where rods *built*
    with the calibrated length put them, and is bit-identical to the
    explicit-keyword node it forwards to.  Before the fix the two
    interface cells sat 11.39 K off with every interior cell agreeing."""
    built = _rods(HeatNode, 2.0, wrapped=wrapped, solver=solver).run_scan(2)
    explicit = _calibrated_rods(HeatNode, wrapped=wrapped, solver=solver)
    forwarding = _calibrated_rods(ForwardingHeat, wrapped=wrapped, solver=solver)
    uncalibrated = _rods(ForwardingHeat, 1.0, wrapped=wrapped, solver=solver).run_scan(2)
    for n in ("rod_a", "rod_b"):
        # The fixture can express the defect: the calibration moves the
        # rods by far more than the tolerance below.
        assert float(jnp.max(jnp.abs(built[n]["temperature"]
                                     - uncalibrated[n]["temperature"]))) > 1.0
        np.testing.assert_allclose(np.asarray(forwarding[n]["temperature"]),
                                   np.asarray(built[n]["temperature"]),
                                   rtol=1e-6, atol=5e-7)
        np.testing.assert_array_equal(np.asarray(forwarding[n]["temperature"]),
                                      np.asarray(explicit[n]["temperature"]))


# ------------------------------------------------------------------
# compute_boundary_fluxes, through a graph step
# ------------------------------------------------------------------

K_CONSTRUCTED, K_CALIBRATED = 30.0, 300.0


def _flux_graph(cls, *, wrapped, coupled):
    gm = GraphManager()
    gm.add_node(_wrap(cls("s", 0.01, stiffness=K_CONSTRUCTED, damping=2.0,
                          initial_position=2.0), wrapped))    # stretched: F != 0
    gm.add_node(Sink("sink", 0.01))
    gm.add_edge("s", "sink", "spring_force", "force")
    if coupled:
        gm.add_edge("sink", "s", "x", "anchor_position")
        gm.add_coupling_group(["s", "sink"], max_iterations=4)
    gm.compile()
    return gm


def _step_with(gm, stiffness):
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["s"]["stiffness"] = jnp.asarray(stiffness, jnp.float32)
    return gm._compiled_step(gm._state, gm._default_external_inputs(), p)


def _spring_force(k, s):
    return float(-k * (s["position"] - 1.0) - 2.0 * s["velocity"])


@pytest.mark.parametrize("wrapped", WRAPS)
@pytest.mark.parametrize("coupled", [pytest.param(False, id="plain-edge"),
                                     pytest.param(True, id="in-coupling-group")])
def test_a_var_keyword_flux_producer_delivers_the_calibrated_flux(coupled, wrapped):
    """The flux edge carries the calibrated stiffness's force, bit-identical
    to the explicit-keyword spring.  Before the fix it carried the
    constructor's: -23.1 against -285.0 on the plain edge."""
    forwarding = _step_with(_flux_graph(ForwardingSpring, wrapped=wrapped, coupled=coupled),
                            K_CALIBRATED)
    explicit = _step_with(_flux_graph(SpringDamperNode, wrapped=wrapped, coupled=coupled),
                          K_CALIBRATED)
    constructed = _step_with(_flux_graph(ForwardingSpring, wrapped=wrapped, coupled=coupled),
                             K_CONSTRUCTED)
    assert not np.isclose(float(forwarding["sink"]["x"]), float(constructed["sink"]["x"]))
    for n in ("s", "sink"):
        for f in explicit[n]:
            np.testing.assert_array_equal(np.asarray(forwarding[n][f]),
                                          np.asarray(explicit[n][f]))
    if not coupled:
        delivered = float(forwarding["sink"]["x"]) / 0.01
        assert delivered == pytest.approx(_spring_force(K_CALIBRATED, forwarding["s"]), rel=1e-6)
        assert delivered != pytest.approx(_spring_force(K_CONSTRUCTED, forwarding["s"]), rel=1e-3)


# ------------------------------------------------------------------
# One rule: every probe answers the same on every spelling
# ------------------------------------------------------------------

_HEATS = {"explicit": HeatNode, "var_keyword": ForwardingHeat, "legacy": LegacyHeat}
_SPRINGS = {"explicit": SpringDamperNode, "var_keyword": ForwardingSpring, "legacy": LegacySpring}


@pytest.mark.parametrize("spelling, expected", [
    pytest.param("explicit", True, id="explicit-keyword"),
    pytest.param("var_keyword", True, id="**kwargs"),
    pytest.param("legacy", False, id="neither"),
])
def test_every_hook_probe_reads_the_shared_signature_rule(spelling, expected):
    """The graph's two probes, ``HybridNode``'s, and the verification
    battery's flux probe agree with ``_signature_takes_params`` on all
    three spellings.  The battery already read the shared rule; the graph
    did not, so the battery certified a ``**kwargs`` flux producer the
    graph never passed ``params`` to."""
    heat = _HEATS[spelling]("h", 0.001, n_cells=10)
    spring = _SPRINGS[spelling]("s", 0.01)
    assert _signature_takes_params(heat.compute_interface_correction) is expected
    assert _signature_takes_params(spring.compute_boundary_fluxes) is expected
    assert gm_mod._correction_accepts_params(heat) is expected
    assert gm_mod._flux_accepts_params(spring) is expected
    assert hybrid_mod._accepts_params(heat.compute_interface_correction) is expected
    assert hybrid_mod._accepts_params(spring.compute_boundary_fluxes) is expected
    assert verification_mod._flux_accepts_params(spring) is expected


@pytest.mark.parametrize("wrapped", WRAPS)
def test_a_hook_with_neither_spelling_is_still_called_the_old_way(wrapped):
    """The widened rule must not reach an override that cannot take the
    keyword: it is called with three arguments, as before, and steps.
    Passing ``params=`` to it would be a ``TypeError`` at trace time."""
    rods = _rods(LegacyHeat, 1.0, wrapped=wrapped, solver="ift")
    for n in ("rod_a", "rod_b"):
        rods.params["nodes"][n]["length"] = jnp.asarray(2.0, jnp.float32)
    out = rods.run_scan(2)
    assert all(bool(jnp.all(jnp.isfinite(out[n]["temperature"]))) for n in ("rod_a", "rod_b"))
    s = _step_with(_flux_graph(LegacySpring, wrapped=wrapped, coupled=False), K_CALIBRATED)
    assert np.isfinite(float(s["sink"]["x"]))
