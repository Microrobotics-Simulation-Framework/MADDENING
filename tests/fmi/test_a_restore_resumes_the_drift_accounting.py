"""Restoring an FMU state resumes the drift accounting; it does not start it
again.

The bridge holds the time it reports to the time it has simulated: the
start time plus the master steps taken since (``_t_ref``, ``_n_ref``), to a
millionth of a master step plus a rounding slack of at most a tenth of one.
``set_state`` used to set that reference to the restored *time* and the
count to zero.  The restored time is the importer's own clock, with
whatever it has gathered, so every restore forgave the drift so far: with a
``get_state`` / ``set_state`` pair after every step -- what a rollback
master does -- step sizes 0.45 millionths of a step too long were never
refused (third step without restores), and an honest running sum at 30000 s
with a 1e-9 s step drifted 1.78 master steps in 4000 (refused after 225
without restores).

The reference is now part of the archive, and a restore resumes it.  These
tests drive one bridge in process; the same importers through the TCP
bridge, the sidecar, the graph and the compiled wrapper are in
``tests/property/test_differential_fmu.py`` (the clock patterns).
"""

from __future__ import annotations

import base64
import io
from fractions import Fraction

import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi import FmuTcpBridge, build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import state_of
from maddening.nodes.spring import SpringDamperNode

#: The claimed bound on |reported - simulated|, in master steps: a
#: millionth plus a tenth (FMU-023), to the clock's own float64 rounding
#: (three ulps of the largest time involved: ``_bound``).
BOUND = Fraction(1, 10) + Fraction(1, 10 ** 6)


def _bound(dt, start, steps, reported) -> Fraction:
    simulated = float(Fraction(start) + steps * Fraction(dt))
    biggest = max(abs(reported), abs(simulated), abs(start), steps * dt)
    return BOUND + 3 * Fraction(float(np.spacing(biggest))) / Fraction(dt)


def _bridge(dt):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", dt, stiffness=30.0, damping=2.0,
                                 initial_position=0.5))
    gm.compile()
    md = build_model_description(gm, model_name="Plant")
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,      # noqa: SLF001
        initial_state=gm._state, params=gm.params))                          # noqa: SLF001
    bridge = FmuTcpBridge(sidecar, md, master_dt=gm.timestep)
    tvr = next(v.value_reference for v in md.variables if v.name == "time")
    return bridge, tvr


def _clock(bridge):
    """The reported time and the reference it is held to."""
    return bridge._time, bridge._t_ref, bridge._n_ref                      # noqa: SLF001


# The importers of the audit's reproducer.  Each is ``(master step, start,
# clock)`` with ``clock(k, t_fmu, t_own) -> (t, h)`` for step ``k``: the
# FMU's reported time and the importer's own running clock in, the
# communication point and step size out.
IMPORTERS = {
    # every step size 0.45 millionths of a step long, on its own running sum
    "long steps": (1e-2, 0.0, lambda k, t_fmu, t_own: (t_own, 1e-2 * (1 + 0.45e-6))),
    # each point 0.05 of a step past the FMU's time, which the point
    # tolerance (16 ulps of 30000 s = 0.058 of a 1e-9 s step) adopts
    "late points": (1e-9, 30000.0, lambda k, t_fmu, t_own: (t_fmu + 0.05e-9, 1e-9)),
    # an honest importer keeping a running sum t += h in float64
    "running sum": (1e-9, 30000.0, lambda k, t_fmu, t_own: (t_own, 1e-9)),
    # FMPy's arithmetic: one rounding per point, never refused
    "start + k * h": (1e-9, 30000.0, lambda k, t_fmu, t_own: (30000.0 + k * 1e-9, 1e-9)),
}

#: Steps each importer is accepted for without a restore, of this many tried
#: (``None``: never refused).
ACCEPTED = {"long steps": (2, 12), "late points": (1, 12), "running sum": (225, 300),
            "start + k * h": (None, 300)}


class _Run:
    """One importer against a fresh bridge, recording every reply."""

    def __init__(self, name):
        self.dt, self.start, self.clock = IMPORTERS[name]
        self.bridge, self.tvr = _bridge(self.dt)
        assert self.bridge.handle({"op": "hello"})["ok"]
        assert self.bridge.handle({"op": "initialize", "t": self.start})["ok"]
        self.t_own, self.k, self.done = self.start, 0, 0
        self.replies: dict[int, list] = {}

    def reported(self):
        return self.bridge.handle({"op": "get", "vr": [self.tvr]})["values"][0]

    def drift(self) -> Fraction:
        """(reported - simulated) in master steps, in exact arithmetic."""
        simulated = Fraction(self.start) + self.done * Fraction(self.dt)
        return (Fraction(self.reported()) - simulated) / Fraction(self.dt)

    def step(self) -> bool:
        t, h = self.clock(self.k, self.reported(), self.t_own)
        before, state = _clock(self.bridge), self.bridge._sidecar.state      # noqa: SLF001
        reply = self.bridge.handle({"op": "step", "t": t, "dt": h})
        self.replies.setdefault(self.k, []).append(reply)
        if not reply["ok"]:
            # nothing advanced: the clock, its reference and the state object
            assert _clock(self.bridge) == before
            assert self.bridge._sidecar.state is state                     # noqa: SLF001
            return False
        self.k, self.done, self.t_own = self.k + 1, self.done + 1, t + h
        assert abs(self.drift()) <= _bound(self.dt, self.start, self.done, self.reported()), (
            self.k, float(self.drift()))
        return True

    def save(self):
        return (self.bridge.handle({"op": "get_state"})["state"], self.k, self.done, self.t_own)

    def restore(self, saved):
        blob, self.k, self.done, self.t_own = saved
        reply = self.bridge.handle({"op": "set_state", "state": blob})
        assert reply["ok"], reply
        assert reply["t"] == self.reported()


def _run(name, restore):
    """The importer's run under ``restore``: ``"none"``, ``"every step"`` (a
    get_state / set_state pair after every accepted step) or ``"rollback"``
    (a state saved part-way, two more steps or a refusal, back to it, and
    on).  Returns the replies per step index and the steps accepted."""
    _, tried = ACCEPTED[name]
    run = _Run(name)
    saved, rolled_back = None, False
    midway = max(1, (ACCEPTED[name][0] or tried) // 2)
    try:
        while run.k < tried:
            if not run.step():
                if restore == "rollback" and saved is not None and not rolled_back:
                    run.restore(saved)
                    rolled_back = True
                    continue
                break
            if restore == "every step":
                run.restore(run.save())
            elif restore == "rollback":
                if saved is None and run.k == midway:
                    saved = run.save()
                elif saved is not None and not rolled_back and run.k == midway + 2:
                    run.restore(saved)
                    rolled_back = True
        if restore == "rollback":
            assert rolled_back, "the run never went back"
        return run.replies, run.done
    finally:
        run.bridge.stop()


@pytest.mark.parametrize("name", sorted(IMPORTERS))
def test_an_importer_is_accepted_for_as_many_steps_as_the_bound_allows(name):
    """Without a restore: the audit's figures (the long steps refused at the
    third, the late points at the second, the running sum after 225), and
    ``start + k * h`` never."""
    accepted, tried = ACCEPTED[name]
    replies, done = _run(name, "none")
    assert done == (tried if accepted is None else accepted)
    if accepted is not None:
        refusal = replies[accepted][-1]
        assert refusal["ok"] is False
        assert "from the time the FMU has simulated" in refusal["error"], refusal


@pytest.mark.parametrize("restore", ["every step", "rollback"])
@pytest.mark.parametrize("name", sorted(IMPORTERS))
def test_a_restore_changes_no_verdict_and_no_reported_time(name, restore):
    """With a get_state / set_state pair after every step, and with one
    rollback part-way, every step gets the reply it gets without a restore
    -- the same reported time to the bit, the same refusal at the same step
    with the same words -- including the steps taken again after going
    back.  The restore used to forgive the drift so far, so the long steps
    and the late points were never refused and the running sum went on past
    225."""
    plain, plain_done = _run(name, "none")
    replies, done = _run(name, restore)
    assert done == plain_done
    assert sorted(replies) == sorted(plain)
    for k, seen in replies.items():
        for reply in seen:                          # a step taken again after a rollback too
            assert reply == plain[k][0], (k, reply, plain[k][0])
    if restore == "rollback":
        assert any(len(seen) > 1 for seen in replies.values())


# ---------------------------------------------------------------- the archive

def _members(blob_b64):
    with np.load(io.BytesIO(base64.b64decode(blob_b64)), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _archive(members):
    buf = io.BytesIO()
    np.savez(buf, **members)
    return base64.b64encode(buf.getvalue()).decode("ascii")


DT = 1e-2


@pytest.fixture
def stepped():
    """A bridge started at 5 s and stepped three times, its archive, and a
    fourth step taken since (so a restore has something to undo)."""
    bridge, tvr = _bridge(DT)
    assert bridge.handle({"op": "initialize", "t": 5.0})["ok"]
    t = 5.0
    for _ in range(3):
        t = bridge.handle({"op": "step", "t": t, "dt": DT})["t"]
    blob = bridge.handle({"op": "get_state"})["state"]
    assert bridge.handle({"op": "step", "t": t, "dt": DT})["ok"]
    yield bridge, tvr, blob, t
    bridge.stop()


def test_the_archive_carries_the_start_time_and_the_steps_taken_since(stepped):
    bridge, tvr, blob, t = stepped
    members = _members(blob)
    assert members["_t_ref"].dtype == np.float64 and members["_t_ref"].shape == ()
    assert members["_n_ref"].dtype == np.int64 and members["_n_ref"].shape == ()
    assert (float(members["_time"]), float(members["_t_ref"]), int(members["_n_ref"])) == (t, 5.0, 3)
    assert _clock(bridge) == (t + DT, 5.0, 4)
    assert bridge.handle({"op": "set_state", "state": blob}) == {"ok": True, "t": t}
    assert _clock(bridge) == (t, 5.0, 3)
    # a reset starts a new reference, and the archive brings its own back
    assert bridge.handle({"op": "reset"}) == {"ok": True}
    assert _clock(bridge) == (0.0, 0.0, 0)
    assert bridge.handle({"op": "set_state", "state": blob})["ok"]
    assert _clock(bridge) == (t, 5.0, 3)
    assert bridge.handle({"op": "step", "t": t, "dt": DT}) == {"ok": True, "t": t + DT}
    assert _clock(bridge) == (t + DT, 5.0, 4)


def test_a_start_time_given_after_a_restore_is_a_new_reference(stepped):
    """``initialize`` (``fmi3EnterInitializationMode``) names the start time,
    and is accepted until the instance's first step -- after a restore into
    a reset instance too.  The FMU takes the importer's word for the time
    there, as at any start: the reference is that time, with no steps."""
    bridge, tvr, blob, t = stepped
    assert bridge.handle({"op": "reset"}) == {"ok": True}
    assert bridge.handle({"op": "set_state", "state": blob})["ok"]
    assert bridge.handle({"op": "initialize", "t": 7.0}) == {"ok": True, "t": 7.0}
    assert _clock(bridge) == (7.0, 7.0, 0)


@pytest.mark.parametrize("dropped", [("_t_ref",), ("_n_ref",), ("_t_ref", "_n_ref")])
def test_an_archive_without_the_reference_is_refused(stepped, dropped):
    """What a bridge before this fix wrote.  It is refused, naming the
    members, rather than restored with the count started again at its
    time; nothing is written."""
    bridge, tvr, blob, t = stepped
    members = _members(blob)
    for key in dropped:
        members.pop(key)
    before, state = _clock(bridge), bridge._sidecar.state                  # noqa: SLF001
    reply = bridge.handle({"op": "set_state", "state": _archive(members)})
    assert reply["ok"] is False, reply
    assert "carries no drift reference" in reply["error"] and str(sorted(dropped)) in reply["error"]
    assert "before 0.4.0's fix" in reply["error"]
    assert _clock(bridge) == before and bridge._sidecar.state is state     # noqa: SLF001
    assert bridge.handle({"op": "set_state", "state": blob})["ok"]         # its own still does


BAD_REFERENCES = {
    "start as a string": ({"_t_ref": np.array("5.0")}, "must be a real scalar"),
    "start as an array": ({"_t_ref": np.array([5.0])}, "must be a real scalar"),
    "start as a boolean": ({"_t_ref": np.array(True)}, "must be a real scalar"),
    "start not a number": ({"_t_ref": np.array(np.nan)}, "non-finite start time"),
    "start infinite": ({"_t_ref": np.array(np.inf)}, "non-finite start time"),
    "start the clock cannot resolve": ({"_t_ref": np.array(1e300)},
                                       "FMU state's start time 1e+300 is too large"),
    "count as a float": ({"_n_ref": np.array(3.0)}, "must be an integer scalar"),
    "count as a string": ({"_n_ref": np.array("3")}, "must be an integer scalar"),
    "count as a boolean": ({"_n_ref": np.array(True)}, "must be an integer scalar"),
    "count as an array": ({"_n_ref": np.array([3])}, "must be an integer scalar"),
    "count negative": ({"_n_ref": np.array(-1)}, "negative step count"),
    "count one more": ({"_n_ref": np.array(4)}, "from the time its own drift reference"),
    "count one fewer": ({"_n_ref": np.array(2)}, "from the time its own drift reference"),
    "count enormous": ({"_n_ref": np.array(2 ** 63 - 1)}, "from the time its own drift reference"),
    "count enormous and unsigned": ({"_n_ref": np.array(2 ** 64 - 1, np.uint64)},
                                    "from the time its own drift reference"),
    "start a step later": ({"_t_ref": np.array(5.0 + DT)}, "from the time its own drift reference"),
    "start at the restored time": ({"_t_ref": np.array(5.0 + 3 * DT)},
                                   "from the time its own drift reference"),
    "time past half a step on": ({"_time": np.array(5.0 + 3.51 * DT)},
                                 "from the time its own drift reference"),
    "time past half a step back": ({"_time": np.array(5.0 + 2.49 * DT)},
                                   "from the time its own drift reference"),
}


@pytest.mark.parametrize("edit", sorted(BAD_REFERENCES))
def test_a_reference_the_bridge_never_writes_is_refused(stepped, edit):
    """The archive is the importer's to hand back, so its reference is
    checked like its time: a finite real scalar the clock resolves, a
    non-negative integer scalar, and a time within half a master step of
    the simulated time the pair gives.  Refused with nothing written."""
    bridge, tvr, blob, t = stepped
    changes, message = BAD_REFERENCES[edit]
    members = {**_members(blob), **changes}
    before, state = _clock(bridge), bridge._sidecar.state                  # noqa: SLF001
    reply = bridge.handle({"op": "set_state", "state": _archive(members)})
    assert reply["ok"] is False and message in reply["error"], reply
    assert _clock(bridge) == before and bridge._sidecar.state is state     # noqa: SLF001


def test_a_reference_in_another_integer_type_restores(stepped):
    """Not a refusal: the count as an unsigned or a narrower integer, and
    the start time as an integer or a float32, are the same numbers."""
    bridge, tvr, blob, t = stepped
    for changes in ({"_n_ref": np.array(3, np.uint64)}, {"_n_ref": np.array(3, np.int8)},
                    {"_t_ref": np.array(5, np.int32)}, {"_t_ref": np.array(5.0, np.float32)}):
        members = {**_members(blob), **changes}
        assert bridge.handle({"op": "set_state", "state": _archive(members)})["ok"], changes
        assert _clock(bridge) == (t, 5.0, 3)


def test_an_edited_time_inside_half_a_step_restores_and_is_then_held_to_the_drift_bound(stepped):
    """The half-step rule is a sanity check on the archive, not the drift
    tolerance: a time 0.3 of a step from its reference restores (an archive
    is the importer's word for its time), and the step from there is then
    refused by the drift tolerance, with nothing advanced, where a
    re-based reference would have accepted it.  Either side of the half
    step: 0.49 restores, 0.51 does not (the parametrised refusals above)."""
    bridge, tvr, blob, t = stepped
    for inside in (3.49, 2.51):
        members = {**_members(blob), "_time": np.array(5.0 + inside * DT)}
        assert bridge.handle({"op": "set_state", "state": _archive(members)})["ok"], inside
    late = 5.0 + 3.3 * DT
    members = {**_members(blob), "_time": np.array(late)}
    assert bridge.handle({"op": "set_state", "state": _archive(members)}) == {"ok": True, "t": late}
    assert _clock(bridge) == (late, 5.0, 3)
    reply = bridge.handle({"op": "step", "t": late, "dt": DT})
    assert reply["ok"] is False and "from the time the FMU has simulated" in reply["error"], reply
    assert _clock(bridge) == (late, 5.0, 3)


def test_every_archive_a_bridge_writes_is_far_inside_the_half_step_rule():
    """The rule must never refuse the FMU's own state.  Over the importers
    above, at every step they are accepted for, the archive's time is within
    the drift bound (a millionth plus a tenth of a step, and the clock's
    rounding) of its own reference -- under a quarter of the half step the
    rule allows -- and restores."""
    for name in sorted(IMPORTERS):
        run = _Run(name)
        try:
            worst = Fraction(0)
            while run.k < ACCEPTED[name][1] and run.step():
                blob = run.bridge.handle({"op": "get_state"})["state"]
                m = _members(blob)
                simulated = Fraction(float(m["_t_ref"])) + int(m["_n_ref"]) * Fraction(run.dt)
                offset = abs(Fraction(float(m["_time"])) - simulated) / Fraction(run.dt)
                worst = max(worst, offset)
                assert offset <= _bound(run.dt, run.start, run.done, float(m["_time"]))
                assert int(m["_n_ref"]) == run.done and float(m["_t_ref"]) == run.start
                assert run.bridge.handle({"op": "set_state", "state": blob})["ok"]
            assert worst < Fraction(1, 8), (name, float(worst))    # a quarter of the half step
        finally:
            run.bridge.stop()
