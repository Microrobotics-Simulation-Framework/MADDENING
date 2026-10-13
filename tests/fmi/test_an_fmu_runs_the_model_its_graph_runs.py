"""An FMU runs the model its graph runs: no export over a change the compiled
step has not taken in.

The FMU's sidecar runs the graph's *compiled* step
(``SidecarConfig(step_fn=gm._compiled_step)``), and the step bakes in every
structural value it reads when it is traced.  A structural ``node.params``
write after ``compile()`` -- ``HeatNode``'s ``stencil_order`` 2 -> 4 -- is
taken into the graph only as "dirty": every graph entry point recompiles and
runs order 4.  ``build_model_description`` and the guide's sidecar wiring never
looked, so the FMU ran order 2 if the step had been traced before the write
(6.35 K off the graph after 50 steps) and order 4 if it had not.  Now the
description refuses a graph that has changed since its compile, a sidecar
refuses a graph's step once that graph has changed or been compiled again, and
a bridge refuses a description or a step whose graph has (claims FMU-040,
SYS-115).
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi import FmuTcpBridge, build_model_description
from maddening.fmi.model_description import FMIVariable, ModelDescription
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import values_of
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode

DT = 0.001
N_STEPS = 50
CHANGED = "has changed since its last compile"
RECOMPILED = "has been compiled again since"


def _rod(order: int, *, pretrace: bool = False) -> GraphManager:
    gm = GraphManager()
    gm.add_node(HeatNode("rod", DT, n_cells=12, stencil_order=order))
    gm.add_external_input("rod", "left_temperature")
    gm.compile()
    if pretrace:
        # a graph that has run is a graph whose compiled step has been traced
        gm.step(external_inputs={"rod": {"left_temperature": jnp.float32(0.0)}})
        gm.reset_state()
    return gm


def _sidecar(gm: GraphManager, md: ModelDescription) -> FmuSidecar:
    """The guide's wiring, verbatim."""
    return FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,   # noqa: SLF001
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),  # noqa: SLF001
        fixed_params=md.fixed_parameters,
        input_resolver=gm._resolve_external_inputs,                    # noqa: SLF001
    ))


def _fmu_temperature(gm: GraphManager) -> np.ndarray:
    """``rod.temperature`` after ``N_STEPS`` graph steps through the bridge,
    the rod's left end driven at 100."""
    md = build_model_description(gm, model_name="Rod")
    bridge = FmuTcpBridge(_sidecar(gm, md), md, master_dt=gm.timestep)
    try:
        vr = {v.name: v.value_reference for v in md.variables}
        assert bridge.handle({"op": "set", "type": "Float32", "vr": [vr["rod.left_temperature"]],
                              "values": [100.0]})["ok"]
        reply = bridge.handle({"op": "step", "t": 0.0, "dt": N_STEPS * gm.timestep})
        assert reply["ok"], reply
        return values_of(bridge.handle({"op": "get", "type": "Float32",
                                        "vr": [vr["rod.temperature"]]}))
    finally:
        bridge.stop()


@pytest.fixture(scope="module")
def stencil_4_reference() -> np.ndarray:
    gm = _rod(4)
    gm.run_scan(N_STEPS, external_inputs={"rod": {"left_temperature": jnp.float32(100.0)}})
    return np.asarray(gm.get_node_state("rod")["temperature"], np.float64)


@pytest.mark.parametrize("pretrace", [False, True], ids=["untraced", "traced"])
def test_a_description_over_a_pending_structural_write_is_refused(pretrace):
    """The reproducer's graph, traced before the write or not: refused either
    way, with nothing exported and the way out named."""
    gm = _rod(2, pretrace=pretrace)
    gm.get_node("rod").params["stencil_order"] = 4
    with pytest.raises(ValueError, match=CHANGED) as err:
        build_model_description(gm, model_name="Rod")
    assert "compile()" in str(err.value) and "stencil_order" in str(err.value)


@pytest.mark.parametrize("pretrace", [False, True], ids=["untraced", "traced"])
def test_after_compile_the_fmu_runs_the_written_model(pretrace, stencil_4_reference):
    """The way out works: compile, and the FMU built as the guide shows runs
    order 4, the model every graph entry point runs (it used to run order 2,
    6.35 K off, when the step had been traced before the write)."""
    gm = _rod(2, pretrace=pretrace)
    gm.get_node("rod").params["stencil_order"] = 4
    gm.compile()
    fmu = _fmu_temperature(gm)
    np.testing.assert_allclose(fmu, stencil_4_reference, rtol=0, atol=1e-4)
    # and the graph itself agrees
    gm.run_scan(N_STEPS, external_inputs={"rod": {"left_temperature": jnp.float32(100.0)}})
    np.testing.assert_allclose(fmu, np.asarray(gm.get_node_state("rod")["temperature"]),
                               rtol=0, atol=1e-4)


def test_every_change_that_makes_the_graph_recompile_is_refused():
    """A structural write is one change of several the graph recompiles for;
    each makes the compiled step the old model, and each is refused.  A
    graph never compiled has no step at all."""
    def spring():
        gm = GraphManager()
        gm.add_node(SpringDamperNode("s", 0.01, initial_position=0.5))
        return gm

    never = spring()
    with pytest.raises(ValueError, match="has not been compiled"):
        build_model_description(never, model_name="S")

    added = spring()
    added.compile()
    added.add_node(SpringDamperNode("t", 0.01))
    with pytest.raises(ValueError, match=CHANGED):
        build_model_description(added, model_name="S")

    wired = spring()
    wired.compile()
    wired.add_external_input("s", "anchor_position")
    with pytest.raises(ValueError, match=CHANGED):
        build_model_description(wired, model_name="S")


def test_a_leaf_write_is_taken_in_and_not_refused():
    """The neighbour that must keep working: a write to a leaf ``gm.params``
    carries needs no recompile, so it is copied in and exported (FMU-040's
    first edition), not refused."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, initial_position=0.5, stiffness=30.0))
    gm.compile()
    gm.get_node("s").params["stiffness"] = 60.0
    md = build_model_description(gm, model_name="S")
    assert {v.name: v.start for v in md.variables}["s.params.stiffness"] == "60.0"
    FmuTcpBridge(_sidecar(gm, md), md, master_dt=gm.timestep).stop()


def test_a_sidecar_refuses_a_step_its_graph_has_left():
    """The guide builds the sidecar after the description.  A structural
    write in between -- or a recompile, which leaves the old step behind --
    is refused there too, and evaluating ``params=gm.params`` is what takes
    the write in."""
    gm = _rod(2, pretrace=True)
    md = build_model_description(gm, model_name="Rod")
    gm.get_node("rod").params["stencil_order"] = 4
    with pytest.raises(ValueError, match=CHANGED) as err:
        _sidecar(gm, md)
    assert "SidecarConfig.step_fn" in str(err.value)

    gm2 = _rod(2)
    old_step = gm2._compiled_step                                   # noqa: SLF001
    gm2.compile()
    with pytest.raises(ValueError, match=RECOMPILED):
        FmuSidecar(SidecarConfig(schema_token="x", step_fn=old_step,
                                 initial_state=gm2._state, params=gm2.params))  # noqa: SLF001
    # the current step is fine
    FmuSidecar(SidecarConfig(schema_token="x", step_fn=gm2._compiled_step,  # noqa: SLF001
                             initial_state=gm2._state, params=gm2.params))  # noqa: SLF001


def test_a_bridge_refuses_a_description_or_a_step_its_graph_has_left():
    """The last door: the description and the sidecar were each current when
    built, and the graph changed before the bridge was.  The bridge looks at
    both: a description of a graph compiled again since, a sidecar whose
    step's graph took a structural write."""
    gm = _rod(2, pretrace=True)
    md = build_model_description(gm, model_name="Rod")
    sidecar = _sidecar(gm, md)
    gm.get_node("rod").params["stencil_order"] = 4
    with pytest.raises(ValueError, match=CHANGED):
        FmuTcpBridge(sidecar, md, master_dt=gm.timestep)

    gm.compile()                       # the graph is current again, the md is not
    with pytest.raises(ValueError, match=RECOMPILED) as err:
        FmuTcpBridge(_sidecar(gm, md), md, master_dt=gm.timestep)
    assert "model description was built from compile" in str(err.value)

    fresh = build_model_description(gm, model_name="Rod")   # a current description...
    with pytest.raises(ValueError, match=RECOMPILED) as err:  # ...and a stale sidecar
        FmuTcpBridge(sidecar, fresh, master_dt=gm.timestep)
    assert "sidecar's step_fn" in str(err.value)

    FmuTcpBridge(_sidecar(gm, fresh), fresh, master_dt=gm.timestep).stop()


def test_what_carries_no_graph_is_not_judged():
    """A hand-built description and a step that is not a graph's are served as
    before: there is no compile to disagree with."""
    md = ModelDescription(model_name="hand", instantiation_token="t",
                          default_step_size=0.01, variables=[FMIVariable(
                              name="time", value_reference=1, dtype="float64",
                              causality="independent", variability="continuous")])
    assert md._graph is None and md._graph_generation is None       # noqa: SLF001
    sidecar = FmuSidecar(SidecarConfig(schema_token="t", step_fn=lambda s, e: s,
                                       initial_state={}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        FmuTcpBridge(sidecar, md, master_dt=0.01).stop()
