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
    # the reply carries the restored time, which the C wrapper keeps as its
    # own clock (it reports it as lastSuccessfulTime when a doStep fails)
    assert bridge.handle({"op": "set_state", "state": snap}) == {"ok": True, "t": 2 * DT}
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
            # time is a Float64, the spring's position a Float32: each is read
            # through the getter of its own type (FMI 3.0; the bridge refuses
            # any other)
            before = inst.getFloat64([T]) + inst.getFloat32([POS])
            with pytest.raises(FMICallException):
                inst.doStep(currentCommunicationPoint=100.0, communicationStepSize=DT)
            assert inst.getFloat64([T]) + inst.getFloat32([POS]) == before
            inst.doStep(currentCommunicationPoint=5.0 + DT, communicationStepSize=DT)
            assert inst.getFloat64([T])[0] == pytest.approx(5.0 + 2 * DT)
            inst.terminate()
        finally:
            inst.freeInstance()
    assert np.isfinite(before).all()


# ------------------------------------------- one tolerance for both checks
#
# The step size used to be allowed a millionth of a master step *per master
# step it covered* off a whole number of them, the communication point a
# millionth of one.  A step of 3 * DT * (1 + 5e-7) was accepted, the
# bridge's clock advanced 3 * DT, and the next doStep at t + h -- the point
# FMI requires -- was refused (the audit's C5).  Both now use the same
# absolute tolerance, a millionth of a master step, so a step the bridge
# accepts always leaves the next legal point inside it.

TOL = 1e-6 * DT


@pytest.mark.parametrize("n", [1, 3, 1000])
def test_a_step_inside_the_tolerance_leaves_the_next_legal_point_accepted(served, n):
    """The next legal point, ``t + h``, is accepted after a step whose size
    was inside the tolerance -- and steps from there that do not add to the
    drift (exact ones, or ones off the other way) go on being accepted."""
    md, bridge = served
    h = n * DT + 0.9 * TOL
    reply = bridge.handle({"op": "step", "t": 0.0, "dt": h})
    assert reply["ok"], reply
    t = h                                             # where the importer goes next
    for size in (n * DT, n * DT - 0.9 * TOL, n * DT + 0.9 * TOL):
        reply = bridge.handle({"op": "step", "t": t, "dt": size})
        assert reply["ok"], (t, size, reply)
        t = t + size
    assert _time(md, bridge) == pytest.approx(4 * n * DT, abs=TOL)


@pytest.mark.parametrize("n", [1, 3, 1000])
@pytest.mark.parametrize("sign", [1, -1])
def test_a_step_outside_the_tolerance_is_refused_and_advances_nothing(served, n, sign):
    md, bridge = served
    before = _state(md, bridge)
    reply = bridge.handle({"op": "step", "t": 0.0, "dt": n * DT + sign * 1.1 * TOL})
    assert reply["ok"] is False and "is not a whole multiple" in reply["error"], reply
    assert _state(md, bridge) == before


def test_the_audits_step_size_is_refused_instead_of_breaking_the_next_step(served):
    md, bridge = served
    h = 3 * DT * (1 + 5e-7)                           # 1.5e-8 off, TOL is 1e-8
    reply = bridge.handle({"op": "step", "t": 0.0, "dt": h})
    assert reply["ok"] is False and "is not a whole multiple" in reply["error"], reply
    assert bridge.handle({"op": "step", "t": 0.0, "dt": 3 * DT})["ok"]
    assert bridge.handle({"op": "step", "t": 3 * DT, "dt": 3 * DT})["ok"]


def test_a_biased_importer_is_stopped_before_its_drift_passes_the_tolerance(served):
    """Every step size a third of the tolerance long, at the importer's own
    running sum: each one alone is inside the tolerance, and each point is
    adopted, but the reported time used to move ahead of the simulated time
    by that much per step, without bound (1.6 master steps after 2000 steps
    at a 1e-12 s step).  The step that would carry the drift past the
    tolerance is refused, with nothing advanced, and the reported time never
    left the simulated time by more than the tolerance."""
    md, bridge = served
    h, t = DT + TOL / 3, 0.0
    for k in range(3):
        reply = bridge.handle({"op": "step", "t": t, "dt": h})
        assert reply["ok"], (k, reply)
        t += h
    before = _state(md, bridge)
    refused = bridge.handle({"op": "step", "t": t, "dt": h})
    assert refused["ok"] is False and "from the time the FMU has simulated" in refused["error"]
    assert _state(md, bridge) == before
    assert abs(_time(md, bridge) - 3 * DT) <= TOL     # reported vs simulated
    # the importer can still go on at the master step
    assert bridge.handle({"op": "step", "t": t, "dt": DT})["ok"]


def test_a_point_that_has_drifted_is_refused_even_if_the_step_would_end_back_inside(served):
    """The point and the step's end are each held to the simulated time.  A
    point 0.9 tolerance ahead of the FMU's clock is adopted; a second one
    another 0.9 ahead, with a step 0.9 short, would end back inside the
    tolerance -- but the step would start, and the reported time with it,
    1.8 tolerances from the physics.  Refused, nothing advanced."""
    md, bridge = served
    assert bridge.handle({"op": "step", "t": 0.0, "dt": DT})["ok"]
    assert bridge.handle({"op": "step", "t": DT + 0.9 * TOL, "dt": DT})["ok"]
    before = _state(md, bridge)
    reply = bridge.handle({"op": "step", "t": 2 * DT + 1.8 * TOL, "dt": DT - 0.9 * TOL})
    assert reply["ok"] is False, reply
    assert "the communication point" in reply["error"]
    assert "from the time the FMU has simulated" in reply["error"], reply
    assert _state(md, bridge) == before


@pytest.mark.parametrize("dt", [1e-2, 1e-9, 1e-12])
def test_the_tolerance_is_a_millionth_of_the_master_step_at_any_step(dt):
    """The rounding slack used to be in ulps of ``max(|t|, 1)``, an absolute
    ~3.6e-15 s: below a master step of about 1e-9 s that floor was the
    tolerance, and 8.9e-4 of a 1e-12 s step was accepted.  Now a step size
    1.1 millionths off is refused at every master step, 0.9 millionths is
    accepted, and so is a communication point 0.9 millionths off."""
    bridge = _spring_bridge(dt)
    try:
        assert bridge.handle({"op": "step", "t": 0.0, "dt": dt * (1 + 0.9e-6)})["ok"]
        bridge.handle({"op": "reset"})
        reply = bridge.handle({"op": "step", "t": 0.0, "dt": dt * (1 + 1.1e-6)})
        assert reply["ok"] is False and "is not a whole multiple" in reply["error"], reply
        assert bridge.handle({"op": "step", "t": 0.0, "dt": dt})["ok"]
        assert bridge.handle({"op": "step", "t": dt * (1 + 0.9e-6), "dt": dt})["ok"]
        reply = bridge.handle({"op": "step", "t": dt * (3 + 1.1e-6), "dt": dt})
        assert reply["ok"] is False and "not the FMU's current time" in reply["error"], reply
    finally:
        bridge.stop()


def _spring_bridge(dt):
    from maddening.core.graph_manager import GraphManager
    from maddening.fmi import FmuTcpBridge, build_model_description
    from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
    from maddening.nodes.spring import SpringDamperNode

    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", dt, initial_position=0.5))
    gm.compile()
    md = build_model_description(gm, model_name="M")
    return FmuTcpBridge(FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,      # noqa: SLF001
        initial_state=gm._state, params=gm.params)), md, master_dt=gm.timestep)  # noqa: SLF001


@pytest.mark.parametrize("start, dt", [(0.0, 0.01), (1000.0, 0.1), (1.0, 1e-9)])
def test_an_honest_importers_clock_stays_inside_the_drift_tolerance(start, dt):
    """The drift bound grows by a few ulps of the time per step, the rounding
    a running sum of step sizes gathers, up to a tenth of a master step, so
    an importer that keeps one -- or computes ``start + k * h`` -- is not
    refused over a million steps from these starts (the running sum from
    1 s at 1e-9 s gathers 0.08 of a step by then; past the cap it is
    refused: the next test).  A million steps of both clocks, judged by the
    bridge's own tolerance at every thousandth step and at the hundred worst
    ones (no physics is run: the drift check is a function of the times
    alone)."""
    bridge = _spring_bridge(dt)
    try:
        bridge._t_ref, bridge._n_ref = start, 0                           # noqa: SLF001
        k = np.arange(1, 1_000_001)
        # sequential sums from the start, as an importer keeps them (``start +
        # cumsum`` would round once, not once a step)
        running = np.add.accumulate(np.concatenate([[start], np.full(k.size, dt)]))[1:]
        clocks = {"running sum": running, "start + k * h": start + k * dt}
        simulated = start + k * dt
        for name, clock in clocks.items():
            err = np.abs(clock - simulated)
            picks = np.unique(np.concatenate([np.arange(999, k.size, 1000),
                                              np.argsort(err / k)[-100:]]))
            for i in picks:
                tol = bridge._drift_tolerance(float(clock[i]), int(k[i]))  # noqa: SLF001
                assert err[i] <= tol, (name, int(k[i]), float(err[i]), tol)
    finally:
        bridge.stop()


# ----------------------------------- a time the clock cannot resolve
#
# The rounding slacks are in ulps of the times involved, and were uncapped:
# at 1000 s one ulp is 0.11 of a 1e-12 s master step, so the communication
# point's 16 ulps were 1.8 steps and the drift slack grew by 0.45 of a step
# per step -- a doStep one whole master step ahead was adopted, and an
# importer whose clock ran 40% fast was never refused (the B1 round-8
# audit's M4; the same at 1.7e9 s with 1e-6 s steps).  Each slack is now at
# most a tenth of a master step, and a time whose 16 ulps pass that is
# refused outright.

SLACK = 0.1                                           # of a master step


def _resolution_limit(dt):
    """The smallest power of two whose 16 ulps pass a tenth of ``dt``: every
    time below it is resolved, none at or above it (an independent search,
    not the bridge's formula)."""
    k = -1074
    while 16 * float(np.spacing(2.0 ** k)) <= SLACK * dt:
        k += 1
    return 2.0 ** k


@pytest.mark.parametrize("start, dt", [(1000.0, 1e-12), (1.7e9, 1e-6)])
def test_a_start_time_the_clock_cannot_resolve_against_the_master_step_is_refused(start, dt):
    """The audit's two configurations: refused at ``initialize``, naming the
    ratio, with the time left at 0; a step from that start is refused too,
    and nothing advances."""
    bridge = _spring_bridge(dt)
    try:
        reply = bridge.handle({"op": "initialize", "t": start})
        assert reply["ok"] is False, reply
        assert "is too large for the master step" in reply["error"], reply
        assert "of a master step" in reply["error"] and "resolves times below" in reply["error"]
        assert bridge._time == 0.0 and bridge._n_ref == 0                     # noqa: SLF001
        refused = bridge.handle({"op": "step", "t": start, "dt": dt})
        assert refused["ok"] is False and bridge._n_ref == 0, refused         # noqa: SLF001
    finally:
        bridge.stop()


@pytest.mark.parametrize("dt", [1e-12, 1e-9, 1e-6, 1e-3])
def test_the_resolution_limit_is_where_sixteen_ulps_pass_a_tenth_of_the_master_step(dt):
    """Just below the limit a start is accepted, at it refused; and the limit
    the message names is that one."""
    limit = _resolution_limit(dt)
    bridge = _spring_bridge(dt)
    try:
        below = float(np.nextafter(limit, 0.0))
        assert bridge.handle({"op": "initialize", "t": below}) == {"ok": True, "t": below}
        bridge.handle({"op": "reset"})
        reply = bridge.handle({"op": "initialize", "t": limit})
        assert reply["ok"] is False and f"below about {limit:.3g} s" in reply["error"], reply
        reply = bridge.handle({"op": "initialize", "t": -limit})
        assert reply["ok"] is False, reply
    finally:
        bridge.stop()


def test_a_step_that_would_end_past_the_resolution_limit_is_refused():
    """From an accepted start, a step whose end ``t + h`` is a time the clock
    cannot resolve is refused, with nothing advanced; one that ends below
    the limit is taken."""
    dt = 1e-12
    limit = _resolution_limit(dt)
    bridge = _spring_bridge(dt)
    try:
        start = limit - 5e-8
        assert bridge.handle({"op": "initialize", "t": start})["ok"]
        reply = bridge.handle({"op": "step", "t": start, "dt": 100_000 * dt})
        assert reply["ok"] is False and "end of the step (t + h)" in reply["error"], reply
        assert "is too large for the master step" in reply["error"], reply
        assert bridge._n_ref == 0 and bridge._time == start                   # noqa: SLF001
        assert bridge.handle({"op": "step", "t": start, "dt": dt})["ok"]
    finally:
        bridge.stop()


def test_a_restored_time_the_clock_cannot_resolve_is_refused():
    """An archive is a door into the time too: one carrying 1000 s for a
    1e-12 s master step is refused with nothing restored, while the FMU's
    own snapshot restores."""
    import base64
    import io

    from maddening.fmi.tcp_bridge import state_of

    bridge = _spring_bridge(1e-12)
    try:
        assert bridge.handle({"op": "step", "t": 0.0, "dt": 1e-12})["ok"]
        blob = state_of(bridge.handle({"op": "get_state"}))
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            members = {k: data[k] for k in data.files}
        members["_time"] = np.asarray(1000.0)
        buf = io.BytesIO()
        np.savez(buf, **members)
        reply = bridge.handle({"op": "set_state",
                               "state": base64.b64encode(buf.getvalue()).decode("ascii")})
        assert reply["ok"] is False and "FMU state's time" in reply["error"], reply
        assert bridge._time == 1e-12                                          # noqa: SLF001
        assert bridge.handle({"op": "set_state", "state": base64.b64encode(blob).decode()})["ok"]
    finally:
        bridge.stop()


def test_at_a_large_accepted_ratio_a_biased_clock_is_stopped_within_a_tenth_of_a_step():
    """At 16 s with a 1e-12 s master step (an ulp is 0.0036 of a step, inside
    the resolution limit), every point a hundredth of a step late: each one
    passes the communication-point check, but the drift slack used to grow
    by 4 ulps -- 0.014 of a step -- per step, faster than this bias, so the
    importer was never refused and the reported time ran ten master steps
    ahead of the physics in a thousand steps.  Now it is refused once the
    drift passes a tenth of a step, and the reported time never left the
    simulated time by more than that; a whole-step skip is refused too."""
    dt, start = 1e-12, 16.0
    bridge = _spring_bridge(dt)
    try:
        assert bridge.handle({"op": "initialize", "t": start})["ok"]
        t, refused_at = start, None
        for k in range(1000):
            reply = bridge.handle({"op": "step", "t": t, "dt": dt})
            if not reply["ok"]:
                refused_at = k
                assert "from the time the FMU has simulated" in reply["error"], reply
                break
            ahead = (bridge._time - (start + bridge._n_ref * dt)) / dt          # noqa: SLF001
            assert abs(ahead) <= SLACK + 1e-6, (k, ahead)
            t = t + dt + 0.01 * dt
        assert refused_at is not None and refused_at < 20, refused_at
        point = bridge._time                                                  # noqa: SLF001
        assert bridge.handle({"op": "step", "t": point + dt, "dt": dt})["ok"] is False
    finally:
        bridge.stop()


def test_an_honest_running_sum_is_refused_once_its_rounding_passes_a_tenth_of_a_step():
    """The cap's cost, stated: from 1 s at a 1e-9 s step a running sum of step
    sizes gathers 0.37 ulp of the time per step (systematically, within a
    binade), passes a tenth of a master step after about 1.2 million steps,
    and is refused from there, while ``start + k * h`` is not.  The
    tolerance itself never exceeds a millionth plus a tenth of a step."""
    dt, start = 1e-9, 1.0
    bridge = _spring_bridge(dt)
    try:
        bridge._t_ref, bridge._n_ref = start, 0                           # noqa: SLF001
        k = np.arange(1, 2_000_001)
        running = np.add.accumulate(np.concatenate([[start], np.full(k.size, dt)]))[1:]
        simulated = start + k * dt
        tolerance = bridge._drift_tolerance(float(running[-1]), int(k[-1]))  # noqa: SLF001
        assert tolerance == pytest.approx((1e-6 + SLACK) * dt, rel=1e-12)
        err = np.abs(running - simulated)
        first = int(np.argmax(err > tolerance))
        assert err[first] > tolerance and 1_000_000 < first < 1_500_000, first
        stepped = np.abs(start + k * dt - simulated)
        assert float(stepped.max()) <= tolerance
    finally:
        bridge.stop()
