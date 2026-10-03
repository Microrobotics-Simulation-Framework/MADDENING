"""The FMU's stability filter is by node, and covers the node's inputs too.

``build_model_description`` exports a node's outputs and parameters only if
the node is STABLE (or EVOLVING / PROVISIONAL with ``include_evolving=True``)
-- "only STABLE-tagged sources/sinks contribute" -- but its inputs loop never
asked: an external input of an EXPERIMENTAL ``HeartPumpNode`` entered the FMU
as a settable variable, even with ``include_evolving=True``, while every
output and parameter of that node was left out (claim FMU-003).  Such an
input is now held at zero, like an input ``selected_inputs`` leaves out, and
listed in ``held_inputs``; naming it in ``selected_inputs`` is refused.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import _STABILITY_REGISTRY
from maddening.core.graph_manager import GraphManager
from maddening.fmi import FmuTcpBridge, build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import values_of
from maddening.nodes.heart_pump import HeartPumpNode
from maddening.nodes.spring import SpringDamperNode

PUMP = f"{HeartPumpNode.__module__}.{HeartPumpNode.__name__}"
HELD = "feed node.s. this FMU does not export"


def _graph() -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", 0.01, initial_position=0.5))
    gm.add_node(HeartPumpNode("pump", 0.01))
    gm.add_external_input("spring", "anchor_position")
    gm.add_external_input("pump", "backpressure")
    gm.compile()
    return gm


@pytest.fixture(scope="module")
def gm() -> GraphManager:
    return _graph()


def _names(md) -> dict[str, str]:
    return {v.name: v.causality for v in md.variables}


@pytest.mark.parametrize("include_evolving", [False, True])
def test_an_experimental_nodes_input_is_held_at_zero_not_exported(gm, include_evolving):
    assert _STABILITY_REGISTRY.get(PUMP) is StabilityLevel.EXPERIMENTAL
    with pytest.warns(UserWarning, match=HELD) as caught:
        md = build_model_description(gm, model_name="M", include_evolving=include_evolving)
    names = _names(md)
    assert not any(n.startswith("pump.") for n in names), names     # no variable of the node
    assert names["spring.anchor_position"] == "input"               # the STABLE one's stays
    assert md.held_inputs["pump.backpressure"][:2] == ("pump", "backpressure")
    assert "spring.anchor_position" not in md.held_inputs
    assert "pump.backpressure" in str(caught[0].message)
    # the node's parameters were already left out; now all three agree
    assert any(k.startswith("pump.params.") for k in md.fixed_parameters)


def test_an_evolving_nodes_input_follows_include_evolving(gm, monkeypatch):
    """The same rule an output follows: excluded by default, exported with
    ``include_evolving=True`` (and then no warning)."""
    monkeypatch.setitem(_STABILITY_REGISTRY, PUMP, StabilityLevel.EVOLVING)
    with pytest.warns(UserWarning, match=HELD):
        default = build_model_description(gm, model_name="M")
    assert "pump.backpressure" not in _names(default)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        opted = build_model_description(gm, model_name="M", include_evolving=True)
    assert _names(opted)["pump.backpressure"] == "input"
    assert any(n.startswith("pump.") and _names(opted)[n] == "output" for n in _names(opted))
    assert opted.held_inputs == {}


def test_naming_such_an_input_is_refused(gm):
    with pytest.raises(ValueError, match=r"pump\.backpressure.*does not export"):
        build_model_description(gm, model_name="M",
                                selected_inputs=["spring.anchor_position", "pump.backpressure"])


def test_a_multi_clock_description_tags_no_input_of_such_a_node(gm):
    """The clocks are built for the exported nodes only; the held input used
    to be exported and carry no clock at all."""
    with pytest.warns(UserWarning, match=HELD):
        md = build_model_description(gm, model_name="M", multi_clock=True)
    assert "pump.backpressure" not in _names(md)
    inputs = [v for v in md.variables if v.causality == "input" and not v.is_clock]
    assert [v.name for v in inputs] == ["spring.anchor_position"]
    assert all(v.clocks for v in inputs)


def test_the_bridge_holds_it_at_zero_as_gm_step_does(gm):
    """What the FMU computes with the input held: the graph stepped with that
    input omitted, which ``GraphManager.step`` fills with zero."""
    with pytest.warns(UserWarning, match=HELD):
        md = build_model_description(gm, model_name="M")
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,   # noqa: SLF001
        initial_state={n: dict(f) for n, f in gm._state.items()},        # noqa: SLF001
        params=gm.params))
    bridge = FmuTcpBridge(sidecar, md, master_dt=gm.timestep)
    try:
        vr = {v.name: v.value_reference for v in md.variables}
        assert bridge.handle({"op": "set", "type": "Float32", "vr": [vr["spring.anchor_position"]],
                              "values": [0.25]})["ok"]
        assert bridge.handle({"op": "step", "t": 0.0, "dt": 5 * gm.timestep})["ok"]
        fmu = {name: values_of(bridge.handle({"op": "get", "vr": [vr[name]]}))
               for name in ("spring.position",)}
    finally:
        bridge.stop()
    ref = _graph()
    for _ in range(5):
        ref.step(external_inputs={"spring": {"anchor_position": jnp.float32(0.25)}})
    np.testing.assert_allclose(fmu["spring.position"],
                               np.asarray(ref.get_node_state("spring")["position"]).ravel(),
                               rtol=1e-6)
