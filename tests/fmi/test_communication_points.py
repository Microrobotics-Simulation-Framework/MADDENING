"""A step starts where the FMU is, and the start time is the FMU's time.

FMI 3.0 has every ``fmi3DoStep`` begin at the previous communication point
plus the previous step size, or -- first -- at the start time
``fmi3EnterInitializationMode`` was given.  The bridge trusted whatever
point it was sent: a jump from 0.01 to 100 was accepted, the physics
advanced one master step, and ``time`` read 100.01, a state labelled with
a time it never reached.  And the C wrapper kept the start time to itself,
so ``time`` read 0.0 until the first step.

The policy now: a point within a millionth of a master step (plus a few
ulps of the time) of the FMU's clock is **adopted** -- the step ends at
``t + n * master_dt`` on the importer's clock -- and one outside it is
refused with nothing advanced.  ``initialize`` sets the start time, until
the instance's first step.
"""

import numpy as np
import pytest

from maddening.fmi.package import find_c_compiler
from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr


@pytest.fixture(scope="module")
def gm():
    return _graph()


@pytest.fixture
def served(gm):
    md, bridge = _bridge(gm)
    yield md, bridge
    bridge.stop()


def _time(md, bridge):
    return bridge.handle({"op": "get", "vr": [_vr(md, "time")]})["values"][0]


def _state(md, bridge):
    names = ("time", "spring.position", "spring.velocity", "ball.position")
    return bridge.handle({"op": "get", "vr": [_vr(md, n) for n in names]})["values"]


def test_a_discontinuous_communication_point_is_refused_and_advances_nothing(served):
    md, bridge = served
    assert bridge.handle({"op": "step", "t": 0.0, "dt": DT})["ok"]
    before = _state(md, bridge)
    reply = bridge.handle({"op": "step", "t": 100.0, "dt": DT})
    assert reply["ok"] is False and "is not the FMU's current time" in reply["error"], reply
    assert _state(md, bridge) == before
    # backwards too, and by exactly one master step either way
    for t in (0.0, 2 * DT):
        assert bridge.handle({"op": "step", "t": t, "dt": DT})["ok"] is False, t
    assert _state(md, bridge) == before
    # the right point still works
    assert bridge.handle({"op": "step", "t": DT, "dt": DT}) == {"ok": True, "t": 2 * DT}


def test_a_point_within_the_tolerance_is_adopted(served):
    """The importer's clock wins inside the tolerance: the step ends at the
    point it sent plus the master steps, not at the bridge's own sum, so
    the two clocks cannot drift apart by accumulated rounding."""
    md, bridge = served
    assert bridge.handle({"op": "step", "t": 0.0, "dt": DT})["ok"]
    nudged = DT + 0.5e-6 * DT                        # half the tolerance off
    reply = bridge.handle({"op": "step", "t": nudged, "dt": DT})
    assert reply == {"ok": True, "t": nudged + DT}
    assert _time(md, bridge) == nudged + DT
    # just outside it is refused
    off = nudged + DT + 2e-6 * DT
    assert bridge.handle({"op": "step", "t": off, "dt": DT})["ok"] is False


@pytest.mark.parametrize("arithmetic", ["start + k * h", "running sum"])
def test_an_importers_rounded_points_are_accepted_over_a_long_run(gm, arithmetic):
    """FMPy computes each point as ``start + k * h``; other masters keep a
    running sum.  Neither equals the bridge's clock exactly after a few
    steps (0.01 + 0.02 is not 3 * 0.01), and both are accepted for the
    whole run."""
    md, bridge = _bridge(gm)
    try:
        start, t = 1.0, 1.0
        assert bridge.handle({"op": "initialize", "t": start})["ok"]
        for k in range(400):
            point = start + k * DT if arithmetic == "start + k * h" else t
            reply = bridge.handle({"op": "step", "t": point, "dt": DT})
            assert reply["ok"], (k, point, reply)
            t = t + DT
        assert _time(md, bridge) == pytest.approx(start + 400 * DT, abs=1e-12)
    finally:
        bridge.stop()


def test_initialize_sets_the_start_time_until_the_first_step(served):
    md, bridge = served
    assert bridge.handle({"op": "initialize", "t": 5.0}) == {"ok": True, "t": 5.0}
    assert _time(md, bridge) == 5.0
    # the first step is at the start time, not at zero
    assert bridge.handle({"op": "step", "t": 0.0, "dt": DT})["ok"] is False
    assert bridge.handle({"op": "step", "t": 5.0, "dt": DT}) == {"ok": True, "t": 5.0 + DT}
    # once stepped, the instance has left initialization for good
    refused = bridge.handle({"op": "initialize", "t": 0.0})
    assert refused["ok"] is False and "has stepped" in refused["error"]
    assert _time(md, bridge) == 5.0 + DT
    # reset returns it to the instantiated state, where it may initialize again
    assert bridge.handle({"op": "reset"}) == {"ok": True}
    assert _time(md, bridge) == 0.0
    assert bridge.handle({"op": "initialize", "t": 2.5}) == {"ok": True, "t": 2.5}


@pytest.mark.parametrize("bad", [None, "5", True, float("nan"), float("inf")])
def test_initialize_takes_a_finite_number(served, bad):
    md, bridge = served
    request = {"op": "initialize"} if bad is None else {"op": "initialize", "t": bad}
    reply = bridge.handle(request)
    assert reply["ok"] is False and "start time" in reply["error"], reply
    assert _time(md, bridge) == 0.0


def test_a_step_without_a_point_runs_on_the_bridges_clock(served):
    """Unchanged: ``t`` is optional, and its absence means "where you are"."""
    md, bridge = served
    assert bridge.handle({"op": "initialize", "t": 3.0})["ok"]
    assert bridge.handle({"op": "step", "dt": DT}) == {"ok": True, "t": 3.0 + DT}


def test_restoring_an_fmu_state_moves_the_clock_to_the_snapshot(served):
    """FMI's way back in time: after ``set_state`` the next step starts at
    the snapshot's time (and a point at the time it was restored from is
    a discontinuity)."""
    md, bridge = served
    assert bridge.handle({"op": "step", "t": 0.0, "dt": 2 * DT})["ok"]
    snap = bridge.handle({"op": "get_state"})["state"]
    assert bridge.handle({"op": "step", "t": 2 * DT, "dt": 3 * DT})["ok"]
    assert bridge.handle({"op": "set_state", "state": snap}) == {"ok": True}
    assert _time(md, bridge) == 2 * DT
    assert bridge.handle({"op": "step", "t": 5 * DT, "dt": DT})["ok"] is False
    assert bridge.handle({"op": "step", "t": 2 * DT, "dt": DT})["ok"]


@pytest.mark.skipif(find_c_compiler() is None, reason="no C compiler")
def test_the_wrapper_reports_the_start_time_and_refuses_a_jump(gm, tmp_path):
    """Through the compiled wrapper and FMPy's FMI 3 binding: ``time`` is
    the start time straight after ``fmi3EnterInitializationMode``, and a
    ``fmi3DoStep`` that jumps is ``fmi3Error`` with nothing advanced."""
    fmpy = pytest.importorskip("fmpy")
    from fmpy.fmi1 import FMICallException
    from fmpy.fmi3 import FMU3Slave

    from maddening.fmi.package import build_fmu_binary, write_fmu

    md, bridge = _bridge(gm)
    so = build_fmu_binary(tmp_path)
    with bridge:
        fmu = write_fmu(md, tmp_path / "plant.fmu", binary=so, endpoint=bridge.endpoint)
        unz = fmpy.extract(str(fmu))
        desc = fmpy.read_model_description(unz)
        inst = FMU3Slave(guid=desc.guid, unzipDirectory=unz,
                         modelIdentifier=desc.coSimulation.modelIdentifier, instanceName="i")
        inst.instantiate()
        try:
            T, POS = _vr(md, "time"), _vr(md, "spring.position")
            inst.enterInitializationMode(startTime=5.0)
            assert inst.getFloat64([T])[0] == 5.0
            inst.exitInitializationMode()
            inst.doStep(currentCommunicationPoint=5.0, communicationStepSize=DT)
            before = inst.getFloat64([T, POS])
            with pytest.raises(FMICallException):
                inst.doStep(currentCommunicationPoint=100.0, communicationStepSize=DT)
            assert inst.getFloat64([T, POS]) == before
            inst.doStep(currentCommunicationPoint=5.0 + DT, communicationStepSize=DT)
            assert inst.getFloat64([T])[0] == pytest.approx(5.0 + 2 * DT)
            inst.terminate()
        finally:
            inst.freeInstance()
    assert np.isfinite(before).all()
