"""``coupling_diagnostics()`` counts every sweep of ``waveform_iterations > 1``.

(The sweeps are restarts of one fixed-point solve, not waveform
relaxation: MADD-ANO-027.)

A group with ``subcycling=True`` over nodes at different timesteps runs
``waveform_iterations`` sweeps per step.  Each sweep is a fixed-point
solve with a budget of ``max_iterations`` passes of its own, and each
iterates the same one-pass map from where the sweep before it stopped.
The report used to keep only the last sweep's pass count, so a first
sweep that exhausted its budget read ``iterations=1`` beside a last
sweep that converged in one pass (MADD-ANO-026).  The invariants pinned
here:

* ``iterations`` is the largest sweep's count, so
  ``iterations >= max_iterations`` holds exactly when some sweep
  exhausted its budget;
* ``total_iterations`` is the sum over the sweeps: the work done;
* ``converged`` and the other keys are the last sweep's, which produced
  the returned state;
* a group that runs one sweep reports, carries and returns exactly what
  it did before, and no group's returned state moves.

Every graph here is compiled once per module (``runs`` below) and
stepped once from its initial state; each test reads the recorded
result.
"""

import inspect
import warnings

import jax
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

KEY = "fast+slow"
TOTAL_SLOT = f"coupling_{KEY}_total_iterations"


def _springs(dt_fast=0.001, dt_slow=0.01):
    """The sub-cycled spring pair of ``test_phases_5_7_8``: coupled both ways."""
    gm = GraphManager()
    for name, dt, x0 in (("fast", dt_fast, 0.0), ("slow", dt_slow, 3.0)):
        gm.add_node(SpringDamperNode(name=name, timestep=dt, stiffness=50.0,
                                     damping=1.0, mass=1.0, rest_length=1.0,
                                     initial_position=x0))
    gm.add_edge("fast", "slow", "position", "anchor_position")
    gm.add_edge("slow", "fast", "position", "anchor_position")
    return gm


def _leaves(tree):
    flat, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {jax.tree_util.keystr(p): np.asarray(v) for p, v in flat}


def _hex_state(run):
    """``float.hex`` of every node-state leaf a recorded run returned."""
    return {k: float(v).hex() for k, v in run["state"].items()}


def _build(max_iterations=10, waveform_iterations=1, solver="ift",
           diagnostics=False, rates="mixed", subcycling=True,
           boundary_interpolation="linear"):
    """Compile, step once, and record what the report and the state say."""
    gm = _springs(dt_fast=0.001 if rates == "mixed" else 0.01)
    kw = dict(max_iterations=max_iterations, tolerance=1e-8,
              subcycling=subcycling, waveform_iterations=waveform_iterations,
              solver=solver, diagnostics=diagnostics)
    if boundary_interpolation != "linear":
        kw["boundary_interpolation"] = boundary_interpolation
    if waveform_iterations != 1 and (not subcycling or rates == "same"):
        # A group that does not sub-cycle does not read the knob, and says
        # so: at registration without ``subcycling``, at ``compile()``
        # when ``subcycling`` is demoted over one shared timestep.
        with pytest.warns(UserWarning, match="waveform_iterations"):
            gm.add_coupling_group(["fast", "slow"], **kw)
            gm.compile()
    else:
        gm.add_coupling_group(["fast", "slow"], **kw)
        gm.compile()
    seeds = _leaves(gm._state.get("_meta", {}))
    gm.step()
    report = gm.coupling_diagnostics()
    return {
        # ``solver="fori"`` reports only with ``diagnostics=True``.
        "report": dict(report[KEY]) if KEY in report else None,
        "state": _leaves({k: v for k, v in gm._state.items() if k != "_meta"}),
        "meta": _leaves(gm._state.get("_meta", {})),
        "seeds": seeds,
        "gm": gm,
    }


@pytest.fixture(scope="module")
def runs():
    """``runs(**config)``: each configuration compiled and stepped once per module."""
    cache: dict = {}

    defaults = {name: param.default
                for name, param in inspect.signature(_build).parameters.items()}

    def get(**config):
        # Keyed on the full configuration, so a default spelled out and a
        # default left out are one graph, compiled once.
        key = tuple(sorted({**defaults, **config}.items()))
        if key not in cache:
            with warnings.catch_warnings():
                # ``solver="fori"`` is deprecated; this file compares
                # against it on purpose.
                warnings.filterwarnings("ignore", "CouplingGroup solver='fori'",
                                        DeprecationWarning)
                cache[key] = _build(**config)
        return cache[key]

    return get


# ---------------------------------------------------------------------------
# The counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("waveform_iterations,iterations,total", [
    (1, 3, 3),
    (2, 3, 4),
    (3, 3, 5),
])
def test_iterations_is_the_largest_sweep_and_total_iterations_the_sum(
        runs, waveform_iterations, iterations, total):
    """The first sweep takes three passes to converge; each later one, one.

    Before the fix this read ``iterations`` 3 / 1 / 1 -- the last sweep
    only -- and had no total: the second and third sweeps' single pass
    was reported as the step's whole work.
    """
    d = runs(waveform_iterations=waveform_iterations)["report"]
    assert (d["iterations"], d["total_iterations"]) == (iterations, total), d
    assert d["converged"] is True, d


def test_the_cap_check_sees_a_first_sweep_that_exhausted_its_budget(runs):
    """``max_iterations=2``: the first sweep stops at the cap, the last converges.

    One sweep is the same computation as the first of three, and it
    stops at the cap unconverged.  With three sweeps the last converges
    in one pass, and the report used to read ``iterations=1`` -- the cap
    check ``iterations >= max_iterations`` said no sweep had hit it.
    ``converged`` is the last sweep's verdict, on the returned state.
    """
    one = runs(max_iterations=2, waveform_iterations=1)["report"]
    assert one["iterations"] == 2 and one["converged"] is False, one

    three = runs(max_iterations=2, waveform_iterations=3)["report"]
    assert three["iterations"] >= 2, three
    assert three["iterations"] == 2 and three["total_iterations"] == 4, three
    assert three["converged"] is True, three


def test_the_cap_check_stays_quiet_when_no_sweep_hits_the_cap(runs):
    """``max_iterations=4``: sweeps of 3, 1 and 1 passes, none at the cap.

    Their sum, 5, is over the cap, so a report whose ``iterations`` were
    the total would claim a budget was exhausted when none was.
    """
    d = runs(max_iterations=4, waveform_iterations=3)["report"]
    assert d["iterations"] < 4, d
    assert d["total_iterations"] >= 4, d
    assert (d["iterations"], d["total_iterations"]) == (3, 5), d


def test_both_solvers_report_the_same_counts_and_verdict(runs):
    """``"fori"`` runs the same sweeps and reports them by the same rule."""
    ift = runs(max_iterations=2, waveform_iterations=3)["report"]
    fori = runs(max_iterations=2, waveform_iterations=3, solver="fori",
                diagnostics=True)["report"]
    for key in ("iterations", "total_iterations", "converged", "ratio_usable"):
        assert fori[key] == ift[key], (key, fori, ift)


# ---------------------------------------------------------------------------
# A group that runs one sweep, and the returned state
# ---------------------------------------------------------------------------

#: The ``_meta`` slots a one-sweep group owned before the fix.
_ONE_SWEEP_SLOTS = {
    f"coupling_{KEY}_iterations",
    f"coupling_{KEY}_residual",
    f"coupling_{KEY}_amplification",
}


@pytest.mark.parametrize("config", [
    dict(waveform_iterations=1),
    # ``subcycling=True`` over nodes that share a timestep does not
    # sub-cycle, so ``waveform_iterations`` is not read.
    dict(waveform_iterations=3, rates="same"),
    dict(waveform_iterations=3, rates="same", subcycling=False),
], ids=["waveform-1", "same-rate", "not-subcycled"])
def test_a_group_that_runs_one_sweep_owns_no_sum_slot(runs, config):
    """Nothing is added to a one-sweep group's carry, and its sum is its count."""
    run = runs(**config)
    group_slots = {k.strip("[]'") for k in run["meta"] if KEY in k}
    assert group_slots == _ONE_SWEEP_SLOTS, group_slots
    assert {k.strip("[]'") for k in run["seeds"] if KEY in k} == _ONE_SWEEP_SLOTS
    d = run["report"]
    assert d["total_iterations"] == d["iterations"], d


#: One step of the spring pair, recorded on the tree before the fix
#: (741a624) and identical there on jaxlib 0.10.2, 0.11.0 and 0.11.2:
#: ``float.hex`` of every state leaf, and the report of the one-sweep
#: group.  A one-sweep group must reproduce both exactly, and a group of
#: any sweep count must reproduce the state: this is a reporting fix, and
#: the solve does not change.  Three of the four runs end on the same
#: float32 fixed point; one sweep capped at two passes stops one ulp short
#: of it in ``fast.velocity``, which the later sweeps then close.
_FIXED_POINT = {
    "['fast']['position']": "0x1.6650680000000p-7",
    "['fast']['velocity']": "0x1.fc03e80000000p+0",
    "['slow']['position']": "0x1.7eba1c0000000p+1",
    "['slow']['velocity']": "-0x1.fd33600000000p-1",
}
_STATE_BEFORE_THE_FIX = {
    # (max_iterations, waveform_iterations) -> {leaf: float.hex}
    (10, 1): _FIXED_POINT,
    (10, 3): _FIXED_POINT,
    (2, 1): {**_FIXED_POINT, "['fast']['velocity']": "0x1.fc03ea0000000p+0"},
    (2, 3): _FIXED_POINT,
}
_ONE_SWEEP_REPORT_BEFORE_THE_FIX = {
    "amplification": 1.0,
    "converged": True,
    "error_estimate": 0.0,
    "gradient_bound_usable": False,
    "gradient_error_estimate": 0.0,
    "gradient_relative_error_bound": float("nan"),
    "iterations": 3,
    "precision_limited": True,
    "ratio_usable": True,
    "residual": 0.0,
    "rho_spectral": float("nan"),
    "spectral_error_bound": float("nan"),
    "spectral_usable": False,
}


@pytest.mark.parametrize("max_iterations,waveform_iterations",
                         sorted(_STATE_BEFORE_THE_FIX))
def test_the_returned_state_is_the_state_before_the_fix(
        runs, max_iterations, waveform_iterations):
    run = runs(max_iterations=max_iterations,
               waveform_iterations=waveform_iterations)
    assert _hex_state(run) == _STATE_BEFORE_THE_FIX[(max_iterations, waveform_iterations)]


def test_one_sweep_reports_exactly_what_it_did_before_the_fix(runs):
    """Every key the report had, value for value; the new key equals ``iterations``."""
    d = runs(waveform_iterations=1)["report"]
    before = dict(_ONE_SWEEP_REPORT_BEFORE_THE_FIX)
    assert set(d) == set(before) | {"total_iterations"}
    for key, want in before.items():
        got = d[key]
        if isinstance(want, float) and np.isnan(want):
            assert isinstance(got, float) and np.isnan(got), (key, got)
        else:
            assert type(got) is type(want) and got == want, (key, got, want)


def test_the_report_does_not_move_the_state(runs):
    """``"fori"`` with ``diagnostics=False`` computes no counts at all.

    With ``diagnostics=True`` it computes and reduces every sweep's; the
    two must hand back the same state to the last bit.
    """
    reported = runs(max_iterations=2, waveform_iterations=3, solver="fori",
                    diagnostics=True)["state"]
    silent = runs(max_iterations=2, waveform_iterations=3, solver="fori",
                  diagnostics=False)["state"]
    assert set(reported) == set(silent)
    for leaf, value in silent.items():
        np.testing.assert_array_equal(reported[leaf], value, err_msg=leaf)


# ---------------------------------------------------------------------------
# What the sweeps do today: restarts, not waveform relaxation (MADD-ANO-027)
# ---------------------------------------------------------------------------
#
# These pin the behaviour MADD-ANO-027 records, so that the entry is
# revisited when it changes: 0.5.0 plans real waveform relaxation, and
# then every assertion below is expected to fail.


def test_a_converged_first_sweep_leaves_the_later_sweeps_nothing_to_change(runs):
    """Every sweep solves the same fixed point, so a converged first one ends it.

    Waveform relaxation would hand each sweep the previous sweep's
    boundary waveform over the sub-step window and could move the state;
    a restart from a converged state cannot.
    """
    one = runs(max_iterations=10, waveform_iterations=1)
    three = runs(max_iterations=10, waveform_iterations=3)
    assert one["report"]["converged"] is True
    assert _hex_state(three) == _hex_state(one)


def test_a_capped_first_sweep_is_continued_by_the_later_ones(runs):
    """At ``max_iterations=2`` the extra sweeps are extra passes toward the same point.

    One sweep stops short of the fixed point; three sweeps land on the
    state ten passes of one sweep reach.
    """
    capped = runs(max_iterations=2, waveform_iterations=1)
    swept = runs(max_iterations=2, waveform_iterations=3)
    converged = runs(max_iterations=10, waveform_iterations=1)
    assert capped["report"]["converged"] is False
    assert _hex_state(capped) != _hex_state(converged)
    assert _hex_state(swept) == _hex_state(converged)


def test_the_interpolation_modes_coincide_when_the_source_is_scheduled_after(runs):
    """With the sub-cycled node's source scheduled after it, the modes are one.

    Both ends of the interpolation are estimates of the end-of-step value:
    the pass's incoming iterate and the in-pass state.  They differ only
    by the in-pass change of a source scheduled *before* the sub-cycled
    node, so the modes coincide under Jacobi, for a source scheduled after
    the sub-cycled node, and at exact stationarity, and nowhere else
    (MADD-ANO-027) -- not merely "at a converged step": with the slow node
    scheduled first they differ by about the tolerance per step.  This
    graph is the second case: ``fast`` (sub-cycled) is scheduled before
    ``slow``, its only source, under Gauss-Seidel, so ``"constant"``,
    ``"linear"`` and ``"quadratic"`` (which is never given its third value,
    and is ``"linear"`` everywhere) return the same state to the last bit.
    """
    gm = runs()["gm"]
    # The precondition the claim rests on, checked rather than assumed.
    assert gm.schedule == ["fast", "slow"], gm.schedule
    assert gm._coupling_groups[0].iteration_mode == "gauss-seidel"
    states = [_hex_state(runs(boundary_interpolation=mode))
              for mode in ("constant", "linear", "quadratic")]
    assert runs()["report"]["converged"] is True
    assert states[0] == states[1] == states[2]


# ---------------------------------------------------------------------------
# The slot the sum lives in
# ---------------------------------------------------------------------------


def test_the_sum_is_seeded_carried_through_a_scan_and_reset(runs):
    """``compile()`` seeds the slot, a scan carries it, ``reset_state()`` restores it.

    Without the seed the first step would add a key to ``_meta`` and a
    ``lax.scan`` carry would change structure; without the reset seed the
    slot would keep the last run's sum.
    """
    run = runs(max_iterations=2, waveform_iterations=3)
    seed = run["seeds"][f"['{TOTAL_SLOT}']"]
    assert seed.dtype == np.int32 and int(seed) == 0
    gm = run["gm"]  # the last test to use this graph: it is advanced here
    gm.run_scan(3)
    d = dict(gm.coupling_diagnostics()[KEY])
    assert d["total_iterations"] >= d["iterations"] >= 1, d
    assert int(gm._state["_meta"][TOTAL_SLOT]) == d["total_iterations"]
    gm.reset_state()
    assert gm.coupling_diagnostics() == {}
    after = np.asarray(gm._state["_meta"][TOTAL_SLOT])
    assert after.dtype == np.int32 and int(after) == 0


def test_a_checkpoint_without_the_sum_reads_it_as_the_largest_sweep(runs, tmp_path):
    """A checkpoint written before the slot existed carries no sum.

    Restored into a waveform group it leaves the freshly seeded 0 beside
    a non-zero count; the report reads the sum as that count rather than
    as fewer passes than one sweep took.
    """
    gm = runs(max_iterations=4, waveform_iterations=3)["gm"]  # left as found
    assert dict(gm.coupling_diagnostics()[KEY])["total_iterations"] == 5
    path = gm.save_state(tmp_path / "ck.npz")
    with np.load(path, allow_pickle=False) as data:
        names = list(data.files)
        kept = {k: data[k] for k in names if not k.endswith("_total_iterations")}
    assert len(kept) == len(names) - 1, names
    old = tmp_path / "old.npz"
    np.savez(old, **kept)
    gm.reset_state()
    gm.load_state(old)
    d = dict(gm.coupling_diagnostics()[KEY])
    assert d["iterations"] == 3 and d["total_iterations"] == 3, d
    gm.load_state(path)
    assert dict(gm.coupling_diagnostics()[KEY])["total_iterations"] == 5
