"""Edge tests for rows of ``docs/validation/sysid_fmu_claims.yaml`` (FMU-NNN).

Each test is cited by a row of the inventory and sits at an edge of the
conditions its claim was stated for: timesteps that do not divide each
other, a value below a float32's range, a graph written to after its last
compile, the description an earlier release wrote.  Where the tree does
not meet a claim the test is a strict xfail whose reason starts with the
row's id; ``tests/compliance/test_claims_inventories.py`` holds the two
together.
"""

from __future__ import annotations

import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
from maddening.fmi import tcp_bridge
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, values_of
from maddening.nodes.spring import SpringDamperNode


def _springs(*dts, external=False):
    gm = GraphManager()
    for i, dt in enumerate(dts):
        gm.add_node(SpringDamperNode(f"s{i}", dt, stiffness=30.0, damping=2.0, mass=1.0,
                                     rest_length=1.0, initial_position=0.5 + 0.25 * i))
    for i in range(1, len(dts)):
        gm.add_edge(f"s{i - 1}", f"s{i}", "position", "anchor_position")
    if external:
        gm.add_external_input("s0", "anchor_position")
    gm.compile()
    return gm


def _bridge(gm, md, **config):
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),
        input_resolver=gm._resolve_external_inputs, **config))
    return FmuTcpBridge(sidecar, md, master_dt=gm.timestep)


def _vr(md):
    return {v.name: v.value_reference for v in md.variables}


# ---------------------------------------------------------------------------
# The model description
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dts, fastest_is_the_step", [((0.01, 0.05), True),
                                                       ((0.02, 0.03), False)])
def test_the_fastest_clock_is_the_default_step_only_where_the_timesteps_divide(
        dts, fastest_is_the_step):
    """FMU-012: every clock is a whole number of default steps, and the fastest equals the
    step "unless the timesteps do not divide each other" -- 0.02 and 0.03 run on 0.01."""
    gm = _springs(*dts)
    md = build_model_description(gm, model_name="m", multi_clock=True)
    intervals = sorted(c.interval_decimal for c in md.clocks())
    assert intervals == sorted(dts)
    for interval in intervals:
        ratio = interval / md.default_step_size
        assert abs(ratio - round(ratio)) < 1e-9 and round(ratio) >= 1
    assert (intervals[0] == pytest.approx(md.default_step_size)) is fastest_is_the_step


#: What ``build_model_description(gm, model_name="m")`` wrote at v0.3.0 for the
#: one-spring graph below (``git archive v0.3.0``, run on 2026-10-02).
_V030_XML = """<?xml version='1.0' encoding='utf-8'?>
<fmiModelDescription fmiVersion="3.0" modelName="m" instantiationToken="f16fda5a-e56e-2e1e-a930-e8486dbb71f5" generationTool="maddening.fmi">
  <UnitDefinitions>
    <Unit name="s" />
  </UnitDefinitions>
  <DefaultExperiment startTime="0.0" stopTime="1.0" tolerance="1e-06" stepSize="0.001" />
  <ModelVariables>
    <Float64 name="time" valueReference="1" causality="independent" variability="continuous" description="Simulation time (independent variable)." unit="s" />
    <Float32 name="s.position" valueReference="2" causality="output" variability="continuous" description="State field 'position' of node 's'" />
    <Float32 name="s.velocity" valueReference="3" causality="output" variability="continuous" description="State field 'velocity' of node 's'" />
  </ModelVariables>
  <ModelStructure>
    <Output valueReference="2" />
    <Output valueReference="3" />
    <InitialUnknown valueReference="2" />
    <InitialUnknown valueReference="3" />
  </ModelStructure>
</fmiModelDescription>"""


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "FMU-013: a default (single-clock) export is not byte-for-byte what v0.3.0 produced: "
    "0.4.0 adds parameters, units and the graph's step, and the token covers more; "
    "pending docs fix"))
def test_a_single_clock_export_is_byte_for_byte_what_v0_3_0_produced():
    """FMU-013: "Clocks are off by default, so a single-clock FMU is byte-for-byte what
    v0.3.0 produced"."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.0,
                                 rest_length=1.0, initial_position=0.5))
    gm.compile()
    xml = build_model_description(gm, model_name="m").to_xml()
    assert "<Clock" not in xml and "clocks=" not in xml       # what does hold
    assert xml.rstrip() == _V030_XML


def test_the_token_covers_value_references_and_shapes():
    """FMU-010: the token hashes every variable's value reference and shape, besides its
    name: the same names under other value references, or another shape, do not
    instantiate against each other."""
    from maddening.nodes.heat import HeatNode

    def two(order):
        gm = GraphManager()
        for name in order:
            gm.add_node(SpringDamperNode(name, 0.01, stiffness=30.0, damping=2.0,
                                         rest_length=1.0, initial_position=0.5))
        gm.compile()
        return build_model_description(gm, model_name="m")

    ab, ba = two(("a", "b")), two(("b", "a"))
    assert {v.name for v in ab.variables} == {v.name for v in ba.variables}
    assert _vr(ab) != _vr(ba)
    assert ab.instantiation_token != ba.instantiation_token

    def rod(n):
        gm = GraphManager()
        gm.add_node(HeatNode("rod", 1e-3, n_cells=n, length=1.0, thermal_diffusivity=1e-4))
        gm.compile()
        return build_model_description(gm, model_name="m")

    eight, nine = rod(8), rod(9)
    assert {v.name for v in eight.variables} == {v.name for v in nine.variables}
    assert eight.instantiation_token != nine.instantiation_token


def test_the_description_offers_no_directional_derivative():
    """FMU-017: the FMU binary does not provide directional derivatives -- its
    ``fmi3GetDirectionalDerivative`` returns ``fmi3Error`` -- and the description says so by
    not claiming ``providesDirectionalDerivative``; the sidecar's ``get_dd`` is a Python API."""
    gm = _springs(0.01)
    md = build_model_description(gm, model_name="m", model_identifier="maddening_fmu")
    xml = md.to_xml()
    assert "<CoSimulation" in xml
    assert 'providesDirectionalDerivative="true"' not in xml
    assert 'providesAdjointDerivatives="true"' not in xml


# ---------------------------------------------------------------------------
# The sidecar and the bridge against the graph
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, raises=TypeError, reason=(
    "FMU-030: SidecarConfig.step_fn is documented as 'typically GraphManager.step bound to a "
    "particular graph', which takes no state argument; pending docs fix"))
def test_a_sidecar_built_on_gm_step_as_its_docstring_suggests_steps():
    """FMU-030: ``step_fn(state, external_inputs) -> new_state``, "Typically
    :meth:`GraphManager.step` bound to a particular graph"."""
    gm = _springs(0.01)
    md = build_model_description(gm, model_name="m")
    sidecar = FmuSidecar(SidecarConfig(schema_token=md.instantiation_token,
                                       step_fn=gm.step, initial_state=gm._state))
    sidecar.step({})


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "FMU-040: an FMU exported after a node.params write and before the next compile "
    "advertises and runs the earlier value, while gm.run runs the written one; pending fix"))
def test_an_fmu_exported_after_a_node_params_write_runs_what_the_graph_runs():
    """FMU-040: an FMU reproduces the graph it was exported from, and a ``node.params``
    write after ``compile()`` "takes effect at the next run of any entry point ... so they
    all run the same model"."""
    gm = _springs(0.01)
    gm.get_node("s0").params["stiffness"] = 60.0
    md = build_model_description(gm, model_name="m")
    bridge = _bridge(gm, md)
    start = {v.name: v.start for v in md.variables}["s0.params.stiffness"]
    bridge.handle({"op": "step", "t": 0.0, "dt": 0.05})
    fmu = float(values_of(bridge.handle({"op": "get", "vr": [_vr(md)["s0.position"]]}))[0])
    gm.run(5)
    graph = float(gm.get_node_state("s0")["position"])
    assert float(start) == 60.0, start
    assert fmu == pytest.approx(graph, rel=1e-6), (fmu, graph)


def test_a_float32_value_beyond_its_range_is_refused_and_one_below_it_rounds_to_zero():
    """FMU-024: values are checked "in the variable's own type": a float32 input past
    ``FLT_MAX`` is refused, and one below the smallest subnormal is accepted and reads back
    as 0.0 -- representable means in range, not exact."""
    gm = _springs(0.01, external=True)
    md = build_model_description(gm, model_name="m")
    bridge = _bridge(gm, md)
    vr = _vr(md)["s0.anchor_position"]

    def put(value):
        return bridge.handle({"op": "set", "type": "Float32", "vr": [vr], "values": [value]})

    def get():
        return float(values_of(bridge.handle({"op": "get", "type": "Float32", "vr": [vr]}))[0])

    assert put(3.4e38)["ok"] and get() == pytest.approx(3.4e38, rel=1e-7)
    refused = put(3.5e38)
    assert not refused["ok"] and "does not fit its type float32" in refused["error"]
    assert get() == pytest.approx(3.4e38, rel=1e-7)          # nothing written
    assert put(1e-50)["ok"] and get() == 0.0
    assert put(1e-40)["ok"] and 0.0 < get() < 1.2e-38        # a subnormal is kept


def test_the_bridges_waits_and_limits_are_the_documented_numbers():
    """FMU-050: ten seconds to begin the first frame, five minutes of silence between
    frames, two minutes to finish an announced frame, sixteen live connection threads,
    64 MiB a frame, 100000 graph steps a request, about five seconds for ``stop()``, and a
    communication-point tolerance of a millionth of a master step."""
    assert tcp_bridge._HANDSHAKE_TIMEOUT == 10.0
    assert tcp_bridge._IDLE_TIMEOUT == 300.0
    assert tcp_bridge._FRAME_TIMEOUT == 120.0
    assert tcp_bridge._MAX_CONNECTIONS == 16
    assert tcp_bridge._MAX_MESSAGE == 64 * 1024 * 1024
    assert tcp_bridge.MAX_STEPS_PER_REQUEST == 100_000
    assert tcp_bridge._STOP_JOIN_TIMEOUT == 5.0
    assert tcp_bridge._COMM_POINT_TOLERANCE == 1e-6
