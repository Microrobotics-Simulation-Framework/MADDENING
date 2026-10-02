"""An external input the FMU does not export is held at zero, as the graph holds it.

``GraphManager.step`` completes a partial ``external_inputs`` with zeros and
refuses a name the graph does not declare (``_resolve_external_inputs``,
whose docstring cites a 98 N error from leaving one out).  The FMU paths
did neither: the bridge passed only the inputs its model description
exports, and ``FmuSidecar.step`` called the compiled step with whatever it
was given.  A ``BallNode`` whose ``table_position`` the description left
out (``selected_inputs``) took its own "no table" branch and fell through
the floor -- ``ball.position`` -2.178 after 0.8 s where the graph bounces
at +0.4635 -- and a misspelt node name did nothing at all.

Now ``build_model_description`` lists the inputs it leaves out
(``held_inputs``, with a warning), the bridge builds a resolver from its
description when its sidecar has none, and ``SidecarConfig.input_resolver``
takes the graph's own.
"""

import base64
import io
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi import MODEL_IDENTIFIER, build_model_description
from maddening.fmi.package import find_c_compiler
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, state_of
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode

DT = 1e-2
N = 80
ANCHOR = {"spring": {"anchor_position": jnp.asarray(0.25, jnp.float32)}}


def _graph():
    gm = GraphManager()
    gm.add_node(BallNode(name="ball", timestep=DT, initial_position=1.0, elasticity=0.7))
    gm.add_node(SpringDamperNode(name="spring", timestep=DT, stiffness=30.0, damping=2.0,
                                 initial_position=0.5))
    gm.add_external_input("ball", "table_position")
    gm.add_external_input("spring", "anchor_position")
    gm.compile()
    return gm


@pytest.fixture(scope="module")
def gm():
    return _graph()


@pytest.fixture(scope="module")
def graph_answer():
    """The graph's own run: the anchor supplied, the table omitted (zero)."""
    g = _graph()
    out = g.run_scan(N, external_inputs=ANCHOR)
    return float(out["ball"]["position"])


def _partial_description(gm):
    with pytest.warns(UserWarning, match=r"\['ball.table_position'\], which will be held at zero"):
        return build_model_description(gm, model_name="Plant", model_identifier=MODEL_IDENTIFIER,
                                       selected_inputs=["spring.anchor_position"])


def _sidecar(gm, md, **kw):
    return FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state={n: dict(f) for n, f in gm._state.items()}, params=gm.params,
        param_specs=gm.param_specs(), **kw))


def _vr(md, name):
    return next(v.value_reference for v in md.variables if v.name == name)


def test_the_description_lists_the_inputs_it_holds_at_zero(gm):
    md = _partial_description(gm)
    assert [v.name for v in md.variables if v.causality == "input"] == ["spring.anchor_position"]
    assert md.held_inputs == {"ball.table_position": ("ball", "table_position", (), "float32")}
    # not part of the XML: an importer cannot set what it cannot name
    assert "table_position" not in md.to_xml()
    # every input exported: nothing held, and no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        full = build_model_description(gm, model_name="Plant")
    assert full.held_inputs == {}


def test_build_model_description_refuses_an_input_the_graph_does_not_declare(gm):
    with pytest.raises(ValueError, match=r"selected_inputs names \['sprnig.anchor_position'\]"):
        build_model_description(gm, model_name="Plant", selected_inputs=["sprnig.anchor_position"])
    with pytest.raises(ValueError, match="not the string"):
        build_model_description(gm, model_name="Plant", selected_inputs="spring.anchor_position")


@pytest.mark.parametrize("resolver", ["the bridge's, from its description", "the graph's own"])
def test_the_bridge_holds_an_unexported_input_at_zero_like_the_graph(gm, graph_answer, resolver):
    md = _partial_description(gm)
    kw = {} if resolver.startswith("the bridge") else {"input_resolver": gm._resolve_external_inputs}
    bridge = FmuTcpBridge(_sidecar(gm, md, **kw), md, master_dt=DT)
    try:
        assert bridge.handle({"op": "set", "vr": [_vr(md, "spring.anchor_position")],
                              "values": [0.25]}) == {"ok": True}
        assert bridge.handle({"op": "step", "t": 0.0, "dt": N * DT})["ok"]
        got = bridge.handle({"op": "get", "vr": [_vr(md, "ball.position")]})["values"][0]
    finally:
        bridge.stop()
    assert got == pytest.approx(graph_answer, rel=1e-6)
    assert graph_answer > 0.0                       # bounced on a table at zero


def test_the_sidecar_with_the_graphs_resolver_steps_like_the_graph(gm, graph_answer):
    md = _partial_description(gm)
    sidecar = _sidecar(gm, md, input_resolver=gm._resolve_external_inputs)
    for _ in range(N):
        sidecar.step(ANCHOR)                         # the table input omitted
    assert float(sidecar.state["ball"]["position"]) == pytest.approx(graph_answer, rel=1e-6)


def test_the_sidecar_refuses_a_misspelt_input_like_the_graph(gm):
    """``GraphManager.step`` refuses a name it does not declare; so does a
    sidecar given the graph's resolver, with the graph's own message, and
    so does a sidecar behind a bridge, through the bridge's resolver.
    Nothing is advanced either way."""
    md = _partial_description(gm)
    typo = {"sprnig": {"anchor_position": jnp.asarray(0.25, jnp.float32)}}
    g = _graph()
    with pytest.raises(ValueError, match="does not declare") as graph_refusal:
        g.step(external_inputs=typo)
    own = _sidecar(gm, md, input_resolver=gm._resolve_external_inputs)
    before = own.state
    with pytest.raises(ValueError) as sidecar_refusal:
        own.step(typo)
    assert str(sidecar_refusal.value) == str(graph_refusal.value)
    assert own.state is before
    behind = _sidecar(gm, md)
    assert behind.input_resolver is None
    bridge = FmuTcpBridge(behind, md, master_dt=DT)
    try:
        assert behind.input_resolver is not None     # installed from the description
        with pytest.raises(ValueError, match=r"external_inputs names \['sprnig.anchor_position'\]"):
            behind.step(typo)
    finally:
        bridge.stop()


def test_a_configured_resolver_is_kept_by_the_bridge(gm):
    md = _partial_description(gm)
    sidecar = _sidecar(gm, md, input_resolver=gm._resolve_external_inputs)
    bridge = FmuTcpBridge(sidecar, md, master_dt=DT)
    bridge.stop()
    assert sidecar.input_resolver == gm._resolve_external_inputs


def test_a_sidecar_without_a_resolver_steps_with_what_it_is_given(gm):
    """Documented, not hidden: without ``input_resolver`` (and not behind a
    bridge) the sidecar hands its inputs to the step as they are."""
    md = _partial_description(gm)
    sidecar = _sidecar(gm, md)
    assert sidecar.input_resolver is None
    for _ in range(N):
        sidecar.step(ANCHOR)
    assert float(sidecar.state["ball"]["position"]) < 0.0      # the node's "no table" branch


def test_held_inputs_are_not_in_the_fmu_state_and_a_restore_keeps_them_at_zero(gm, graph_answer):
    """Neighbouring reader of the same inputs: the FMU-state archive.  It
    carries the inputs an importer can set and nothing else, so a snapshot
    restores, and the held input is still zero after it."""
    md = _partial_description(gm)
    bridge = FmuTcpBridge(_sidecar(gm, md), md, master_dt=DT)
    try:
        anchor = _vr(md, "spring.anchor_position")
        assert bridge.handle({"op": "set", "vr": [anchor], "values": [0.25]}) == {"ok": True}
        blob = state_of(bridge.handle({"op": "get_state"}))
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            inputs = sorted(k for k in data.files if k.startswith("i/"))
        assert inputs == ["i/spring/anchor_position"]
        assert bridge.handle({"op": "step", "t": 0.0, "dt": 5 * DT})["ok"]
        assert bridge.handle({"op": "set_state",
                              "state": base64.b64encode(blob).decode("ascii")}) == {"ok": True}
        assert bridge.handle({"op": "step", "t": 0.0, "dt": N * DT})["ok"]
        got = bridge.handle({"op": "get", "vr": [_vr(md, "ball.position")]})["values"][0]
    finally:
        bridge.stop()
    assert got == pytest.approx(graph_answer, rel=1e-6)


@pytest.mark.skipif(find_c_compiler() is None, reason="no C compiler")
def test_an_importer_driving_the_partial_fmu_reproduces_the_graph(gm, graph_answer, tmp_path):
    """End to end, through the compiled wrapper and FMPy."""
    fmpy = pytest.importorskip("fmpy")
    from maddening.fmi.package import build_fmu_binary, write_fmu

    md = _partial_description(gm)
    so = build_fmu_binary(tmp_path)
    sig = np.array([(0.0, 0.25), (1.0, 0.25)],
                   dtype=[("time", np.float64), ("spring.anchor_position", np.float64)])
    with FmuTcpBridge(_sidecar(gm, md), md, master_dt=DT) as bridge:
        fmu = write_fmu(md, tmp_path / "partial.fmu", binary=so, endpoint=bridge.endpoint)
        res = fmpy.simulate_fmu(str(fmu), start_time=0.0, stop_time=N * DT, step_size=DT,
                                output_interval=DT, input=sig, output=["ball.position"])
    assert res["ball.position"][-1] == pytest.approx(graph_answer, rel=1e-5)
