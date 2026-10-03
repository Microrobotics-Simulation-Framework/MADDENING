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

import numpy as np
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


def test_a_single_clock_export_carries_no_clock():
    """FMU-013: "Clocks are off by default: a single-clock FMU has no <Clock> variable and
    no clocks= attribute, and every output is continuous" -- on a multi-rate graph, where
    multi_clock=True would emit two."""
    gm = _springs(0.01, 0.05)
    md = build_model_description(gm, model_name="m")
    xml = md.to_xml()
    assert "<Clock" not in xml and "clocks=" not in xml and not md.clocks()
    outputs = [v for v in md.variables if v.causality == "output"]
    assert outputs and all(v.variability == "continuous" for v in outputs)
    assert len(build_model_description(gm, model_name="m", multi_clock=True).clocks()) == 2


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


def test_a_sidecar_on_the_compiled_step_steps_as_the_graph_does():
    """FMU-030: ``step_fn(state, external_inputs, params)`` when ``params`` is given -- "the
    graph's compiled step, GraphManager._compiled_step"; ``GraphManager.step`` "cannot
    serve -- it takes no state"."""
    gm = _springs(0.01)
    md = build_model_description(gm, model_name="m")
    sidecar = FmuSidecar(SidecarConfig(schema_token=md.instantiation_token,
                                       step_fn=gm._compiled_step, initial_state=gm._state,
                                       params=gm.params))
    state = sidecar.step({})
    gm.step()
    assert float(state["s0"]["position"]) == float(gm.get_node_state("s0")["position"])
    with pytest.raises(TypeError):
        FmuSidecar(SidecarConfig(schema_token=md.instantiation_token, step_fn=gm.step,
                                 initial_state=gm._state)).step({})


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


def test_a_float32_value_beyond_its_range_at_either_end_is_refused():
    """FMU-024: values are checked "in the variable's own type": a float32 input past
    ``FLT_MAX`` is refused, and so is a non-zero one the type flushes to 0 (below its
    smallest subnormal), with nothing written; one that rounds to a subnormal keeps its sign
    and magnitude -- representable means in range, not exact.  (The underflow read back as
    0.0 until the maintainer ruled it refused: MADD-ANO-137.)"""
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
    for tiny in (1e-50, -1e-50):
        refused = put(tiny)
        assert not refused["ok"] and "does not fit its type float32" in refused["error"]
        assert get() == pytest.approx(3.4e38, rel=1e-7)      # nothing written
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


def test_a_parameter_cannot_be_set_below_its_advertised_min_by_a_subnormal():
    """FMU-039: "Setting a parameter is held to the min / max the description advertises".
    ``damping`` advertises ``min="0.0"``; a negative float32 subnormal is below it.  (It was
    accepted while ``ParamSpec.check`` compared through ``jnp``: MADD-ANO-136.)"""
    gm = _springs(0.01)
    md = build_model_description(gm, model_name="m")
    bridge = _bridge(gm, md)
    var = {v.name: v for v in md.variables}["s0.params.damping"]
    assert float(var.min) == 0.0
    tiny = bridge.handle({"op": "set", "type": "Float32", "vr": [var.value_reference],
                          "values": [-np.finfo(np.float32).tiny]})
    assert not tiny["ok"]                                   # a normal one is refused
    reply = bridge.handle({"op": "set", "type": "Float32", "vr": [var.value_reference],
                           "values": [-1e-40]})
    read = float(values_of(bridge.handle({"op": "get", "type": "Float32",
                                          "vr": [var.value_reference]}))[0])
    assert not reply["ok"], (reply, read)
    assert read >= float(var.min)


@pytest.mark.parametrize("dts, subcycled", [((0.02, 0.03), False), ((0.01, 0.02), True)],
                         ids=["multirate-not-dividing", "subcycled"])
def test_the_compiled_fmu_is_the_graph_on_a_multi_rate_and_a_sub_cycled_graph(
        tmp_path, dts, subcycled):
    """FMU-002 at the edge of its conditions: FMPy drives the compiled wrapper through the
    bridge, and the outputs are the graph's after the same simulated time."""
    fmpy = pytest.importorskip("fmpy")
    from maddening.fmi.package import build_fmu_binary, find_c_compiler, write_fmu

    if find_c_compiler() is None:
        pytest.skip("no C compiler")

    def build():
        gm = GraphManager()
        for i, dt in enumerate(dts):
            gm.add_node(SpringDamperNode(f"s{i}", dt, stiffness=30.0, damping=0.5,
                                         rest_length=1.0 - 2.0 * i, initial_position=0.25 * i))
        gm.add_edge("s0", "s1", "position", "anchor_position")
        gm.add_edge("s1", "s0", "position", "anchor_position")
        if subcycled:
            gm.add_coupling_group(["s0", "s1"], max_iterations=20, tolerance=1e-6,
                                  subcycling=True)
        gm.compile()
        return gm

    gm = build()
    md = build_model_description(gm, model_name="m", model_identifier="maddening_fmu")
    bridge = _bridge(gm, md)
    so = build_fmu_binary(tmp_path)
    stop = 10 * gm.timestep
    with bridge:
        fmu = write_fmu(md, tmp_path / "plant.fmu", binary=so, endpoint=bridge.endpoint)
        res = fmpy.simulate_fmu(str(fmu), start_time=0.0, stop_time=stop,
                                step_size=gm.timestep, output_interval=gm.timestep,
                                output=["s0.position", "s1.position"])
    ref = build()
    ref.run(10)
    assert res["time"][-1] == pytest.approx(stop)
    for name in ("s0", "s1"):
        assert res[f"{name}.position"][-1] == pytest.approx(
            float(ref.get_node_state(name)["position"]), rel=1e-5), name
