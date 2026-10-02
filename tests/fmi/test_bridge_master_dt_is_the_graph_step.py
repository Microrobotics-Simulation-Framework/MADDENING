"""One sidecar step is one graph step: the bridge refuses a ``master_dt``
the model description contradicts.

``FmuTcpBridge`` used to store whatever ``master_dt`` its caller passed.
A ``step`` of ``h`` then ran ``h / master_dt`` graph steps and reported
``t + h``, so a uniform 0.01 s graph served with ``master_dt=0.005`` ran
two graph steps per 0.01 s ``doStep`` and labelled a state at 0.10 s with
the time 0.05 s.  ``build_model_description`` now records the graph's step
(``ModelDescription.graph_timestep``, ``GraphManager.timestep``) whatever
``default_step_size`` advertises, and the bridge refuses a ``master_dt``
that is not it, or that does not divide the advertised step into a whole
number of graph steps.

Every case is checked end to end where it is accepted: the state after a
run of ``doStep`` calls is the graph's state after the same simulated
time, and the ``time`` variable says that time.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
from maddening.fmi.model_description import FMIVariable, ModelDescription
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, values_of
from maddening.nodes.spring import SpringDamperNode


def _springs(*dts, subcycled=()):
    """Springs ``s0, s1, ...`` in a ring (each anchored to the next), with
    rest lengths off their start so a step moves them; *subcycled* indexes
    one sub-cycling coupling group."""
    gm = GraphManager()
    for i, dt in enumerate(dts):
        gm.add_node(SpringDamperNode(f"s{i}", dt, stiffness=30.0, damping=0.5,
                                     rest_length=1.0 - 2.0 * (i % 2),
                                     initial_position=0.25 * i))
    if len(dts) > 1:
        for i in range(len(dts)):
            gm.add_edge(f"s{i}", f"s{(i + 1) % len(dts)}", "position", "anchor_position")
    if subcycled:
        gm.add_coupling_group([f"s{i}" for i in subcycled], max_iterations=20,
                              tolerance=1e-6, subcycling=True)
    gm.compile()
    return gm


def _sidecar(gm, md):
    return FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs()))


def _served(build, master_dt, **md_kw):
    gm = build()
    md = build_model_description(gm, model_name="m", **md_kw)
    return md, FmuTcpBridge(_sidecar(gm, md), md, master_dt=master_dt)


def _positions_and_time(md, bridge):
    names = [v.name for v in md.variables if v.name.endswith(".position")]
    vr = {v.name: v.value_reference for v in md.variables}
    got = values_of(bridge.handle({"op": "get", "vr": [vr[n] for n in names] + [vr["time"]]}))
    return dict(zip(names, got[:-1])), float(got[-1])


def _graph_positions(build, n_steps):
    gm = build()
    gm.run(n_steps)
    return {f"{name}.position": float(gm.get_node_state(name)["position"])
            for name in gm._nodes}


#: name -> (graph, its step, a master_dt that is not its step)
CASES = {
    # the audit's uniform case: half the step ran the physics at 2x
    "uniform": (lambda: _springs(0.01), 0.01, 0.005),
    # the audit's multi-rate case: 0.02 and 0.03 step by 0.01, and the
    # smallest node timestep (the old default) is not the step
    "multirate_not_dividing": (lambda: _springs(0.02, 0.03), 0.01, 0.02),
    "multirate_dividing": (lambda: _springs(0.01, 0.05), 0.01, 0.05),
    # a sub-cycling group advances its largest member timestep per step
    "subcycled": (lambda: _springs(0.01, 0.02, subcycled=(0, 1)), 0.02, 0.01),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_master_dt_other_than_the_graph_step_is_refused(case):
    build, step, wrong = CASES[case]
    gm = build()
    md = build_model_description(gm, model_name="m")
    assert md.graph_timestep == pytest.approx(step, rel=1e-12) == gm.timestep
    with pytest.raises(ValueError, match="is not the step of the graph"):
        FmuTcpBridge(_sidecar(gm, md), md, master_dt=wrong)


@pytest.mark.parametrize("case", sorted(CASES))
def test_at_the_graph_step_the_fmu_is_the_graph_at_the_time_it_reports(case):
    build, step, _ = CASES[case]
    md, bridge = _served(build, step)
    try:
        t, n_steps = 0.0, 6
        for _ in range(n_steps):
            reply = bridge.handle({"op": "step", "t": t, "dt": md.default_step_size})
            assert reply["ok"], reply
            t = reply["t"]
        positions, reported = _positions_and_time(md, bridge)
    finally:
        bridge.stop()
    assert reported == pytest.approx(n_steps * step, rel=1e-9)
    want = _graph_positions(build, n_steps)
    for name, value in want.items():
        assert positions[name] == pytest.approx(value, rel=1e-6, abs=1e-7), name


def test_the_spelling_of_the_step_does_not_matter():
    """``0.01`` is the step of a graph whose GCD came out as
    ``0.009999999999999998``; a relative 1e-9 apart is the same step."""
    gm = _springs(0.02, 0.03)
    assert gm.timestep != 0.01 and gm.timestep == pytest.approx(0.01, rel=1e-12)
    md = build_model_description(gm, model_name="m")
    FmuTcpBridge(_sidecar(gm, md), md, master_dt=0.01).stop()
    with pytest.raises(ValueError, match="is not the step"):
        FmuTcpBridge(_sidecar(gm, md), md, master_dt=0.01 * (1 + 1e-7))


def test_an_advertised_step_of_several_graph_steps_runs_that_many():
    """``default_step_size`` may advertise a coarser communication step; the
    bridge still runs graph steps, and ``master_dt`` is still the graph's
    step (``graph_timestep``), not the advertised one.  Passing the
    advertised step as ``master_dt`` -- the natural mistake once the two
    differ -- is the audit's defect again, and refused."""
    build = lambda: _springs(0.01)          # noqa: E731
    md, bridge = _served(build, 0.01, default_step_size=0.05)
    assert md.default_step_size == 0.05 and md.graph_timestep == 0.01
    try:
        reply = bridge.handle({"op": "step", "t": 0.0, "dt": 0.05})
        assert reply == {"ok": True, "t": pytest.approx(0.05)}
        positions, reported = _positions_and_time(md, bridge)
    finally:
        bridge.stop()
    assert reported == pytest.approx(0.05)
    assert positions["s0.position"] == pytest.approx(_graph_positions(build, 5)["s0.position"],
                                                     rel=1e-6)
    with pytest.raises(ValueError, match="graph_timestep = 0.01"):
        _served(build, 0.05, default_step_size=0.05)


def test_an_advertised_step_the_bridge_would_refuse_is_refused_up_front():
    """An importer steps at the advertised size (the FMU cannot handle a
    variable step), so a ``default_step_size`` that is not a whole number
    of graph steps -- or more of them than one request may take -- would
    make every ``doStep`` fail; the bridge says so when it is built."""
    build = lambda: _springs(0.01)          # noqa: E731
    with pytest.raises(ValueError, match="not a whole multiple"):
        _served(build, 0.01, default_step_size=0.015)
    gm = build()
    md = build_model_description(gm, model_name="m", default_step_size=0.1)
    with pytest.raises(ValueError, match="more than the 5 one request may take"):
        FmuTcpBridge(_sidecar(gm, md), md, master_dt=0.01, max_steps_per_request=5)
    FmuTcpBridge(_sidecar(gm, md), md, master_dt=0.01, max_steps_per_request=10).stop()


def test_a_multi_clock_description_is_served_at_the_graph_step():
    """A neighbouring reader of the same step: with ``multi_clock=True``
    every clock interval is a whole number of graph steps, and the bridge
    serves the description at ``gm.timestep``."""
    gm = _springs(0.02, 0.03)
    md = build_model_description(gm, model_name="m", multi_clock=True)
    for clock in md.clocks():
        k = clock.interval_decimal / md.graph_timestep
        assert k == pytest.approx(round(k), abs=1e-9)
    FmuTcpBridge(_sidecar(gm, md), md, master_dt=gm.timestep).stop()


def _hand_built(step):
    return ModelDescription(model_name="m", instantiation_token="tok", variables=[FMIVariable(
        name="time", value_reference=1, dtype="float64", causality="independent",
        variability="continuous")], default_step_size=step)


def test_a_hand_built_description_is_held_to_its_default_step_size():
    """Without ``graph_timestep`` (a description built by hand) the step the
    description records is ``default_step_size``."""
    state = {"n": {"x": np.float32(1.0)}}
    sidecar = lambda: FmuSidecar(SidecarConfig(          # noqa: E731
        schema_token="tok", step_fn=lambda s, e: s, initial_state=state))
    assert _hand_built(0.01).graph_timestep is None
    FmuTcpBridge(sidecar(), _hand_built(0.01), master_dt=0.01).stop()
    with pytest.raises(ValueError, match=r"default_step_size = 0.001"):
        FmuTcpBridge(sidecar(), _hand_built(1e-3), master_dt=0.01)


@pytest.mark.parametrize("bad", [0.0, -0.01, math.nan, math.inf, "0.01", True, None])
def test_master_dt_must_be_a_positive_finite_number(bad):
    gm = _springs(0.01)
    md = build_model_description(gm, model_name="m")
    with pytest.raises(ValueError, match="master_dt"):
        FmuTcpBridge(_sidecar(gm, md), md, master_dt=bad)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "10"])
def test_max_steps_per_request_must_be_a_positive_integer(bad):
    gm = _springs(0.01)
    md = build_model_description(gm, model_name="m")
    with pytest.raises(ValueError, match="max_steps_per_request"):
        FmuTcpBridge(_sidecar(gm, md), md, master_dt=0.01, max_steps_per_request=bad)
