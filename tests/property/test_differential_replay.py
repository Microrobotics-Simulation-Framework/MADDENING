"""Differential oracle: a replay from recorded state is the recorded run.

A record of a graph's own trajectory (``run_scan_with_history`` through
``observations_from_history``) is a sequence of states the graph passed
through.  Every path that restarts the graph from those states must land
on the next ones exactly:

* **teacher forcing** -- ``windowed_loss`` at the generating parameters is
  ``0.0``, for every window that tiles the record, every ``sample_every``,
  and a multi-rate record started at any base step (``start_step``);
* **multiple shooting** -- ``windowed_loss`` with ``window_states`` seeded
  from the record (``init_window_states``) and any continuity weight is
  ``0.0`` too: each window's end is the next window's start;
* **a fit started at the truth stays there** -- ``fit`` on the
  teacher-forced loss and ``fit_multiple_shooting``, started at the
  generating parameters, return them bit for bit (the loss and its
  gradient are exactly zero there);
* **a checkpoint taken mid ``run_adaptive``** resumes it: the restored
  graph and the one that never stopped take the same steps (``dt_history``)
  to the same state, ``_meta`` included.  (``run`` and ``step`` resumes are
  ``test_differential_checkpoint.py``'s.)

Drawn over coupled groups with no predictor and with ``linear`` and
``quadratic`` predictors, IQN-ILS and IQN-IMVJ with Jacobian reuse (warm
starts carried in ``_meta``), a group short of convergence (so the carried
history matters), multi-rate graphs, and a sub-cycling group.

Tolerance: none.  The replay evaluates the step the record was made with,
from the same state, in the same order; any non-zero loss is a state that
differs.  (Measured exactly ``0.0`` for every family without carried
history on jaxlib 0.11.0, as ``tests/core/test_sysid_claims_edges.py``
already asserts for a single spring on every CI jaxlib.)

**Known failing: H2** -- ``windowed_loss`` restarts every window with
``_meta`` zeroed (``_state_from_obs``), so a group whose next step reads
carried history (a predictor, IMVJ warm starts) is replayed from another
state: the loss at the truth is not zero and a fit started there walks off.

What it cannot see: a defect the record and the replay share (both call the
compiled step), and anything outside ``gm._state``.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes import BallNode, HeatNode, SpringDamperNode
from maddening.sysid import (
    fit,
    fit_multiple_shooting,
    init_window_states,
    observations_from_history,
    windowed_loss,
)

from tests.conftest import EXAMPLES_COSTLY
from tests.property.differential import (
    assert_trees_identical,
    checkpoint_path,
    full_state,
    note,
    tmp_dir,
)


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

def _springs(max_iterations: int = 6, **group):
    """Two springs anchored to each other, as strongly coupled as the
    ``_meta`` slots need: an IQN-IMVJ group stores a secant history every
    step, and at ``max_iterations=1`` a predictor's extrapolation *is* the
    coupled value."""
    def build():
        gm = GraphManager()
        gm.add_node(SpringDamperNode("a", 0.01, stiffness=100.0, rest_length=0.5,
                                     initial_position=0.5, damping=0.5))
        gm.add_node(SpringDamperNode("b", 0.01, stiffness=80.0, rest_length=0.3,
                                     initial_position=0.2, damping=0.1))
        gm.add_edge("a", "b", "position", "anchor_position")
        gm.add_edge("b", "a", "position", "anchor_position")
        gm.add_coupling_group(["a", "b"], max_iterations=max_iterations, tolerance=1e-14,
                              **group)
        with warnings.catch_warnings():
            # Anchored on each other (MADD-ANO-098's pattern): compile() warns.
            warnings.simplefilter("ignore", UserWarning)
            gm.compile()
        return gm
    return build


def _ball_on_spring(**group):
    def build():
        gm = GraphManager()
        gm.add_node(BallNode("ball", 0.01, initial_position=1.0, gravity=-3.0))
        gm.add_node(SpringDamperNode("spring", 0.01, stiffness=20.0, rest_length=0.5,
                                     initial_position=0.2, damping=0.5))
        gm.add_edge("ball", "spring", "position", "anchor_position")
        gm.add_edge("spring", "ball", "position", "table_position")
        gm.add_coupling_group(["ball", "spring"], max_iterations=5, tolerance=1e-7, **group)
        gm.compile()
        return gm
    return build


def _rods(subcycling: bool, max_iterations: int = 4, **group):
    """Two rods end to end, the right one at twice the left's timestep:
    multi-rate without a group, sub-cycled with one."""
    def build():
        gm = GraphManager()
        gm.add_node(HeatNode("left", 0.01, n_cells=6, thermal_diffusivity=0.01,
                             initial_temperature=np.linspace(1.0, 2.0, 6).tolist()))
        gm.add_node(HeatNode("right", 0.02, n_cells=6, thermal_diffusivity=0.01,
                             initial_temperature=0.5))
        gm.add_edge("left", "right", "temperature", "left_temperature",
                    transform="extract_last")
        gm.add_edge("right", "left", "temperature", "right_temperature",
                    transform="extract_first")
        if subcycling:
            gm.add_coupling_group(["left", "right"], subcycling=True,
                                  max_iterations=max_iterations, tolerance=1e-14, **group)
        gm.compile()
        return gm
    return build


def _imvj_rods(jacobian_reuse: int = 8):
    """Two rods at one rate, coupled end to end under IQN-IMVJ: the secant
    columns carried in ``_meta`` save an iteration on the next step (3
    passes cold, 2 warm), so the converged state depends on them in its
    last bits.  (At 8 cells, or at a tolerance of 1e-12, both starts land
    on the same bits: the warm start then changes nothing to replay.)"""
    def build():
        gm = GraphManager()
        gm.add_node(HeatNode("left", 0.01, n_cells=16, thermal_diffusivity=0.05,
                             initial_temperature=np.linspace(1.0, 2.0, 16).tolist()))
        gm.add_node(HeatNode("right", 0.01, n_cells=16, thermal_diffusivity=0.05,
                             initial_temperature=0.5))
        gm.add_edge("left", "right", "temperature", "left_temperature",
                    transform="extract_last")
        gm.add_edge("right", "left", "temperature", "right_temperature",
                    transform="extract_first")
        gm.add_coupling_group(["left", "right"], max_iterations=10, tolerance=1e-6,
                              acceleration="iqn-imvj", jacobian_reuse=jacobian_reuse)
        gm.compile()
        return gm
    return build


def _chain():
    """A spring at 0.01 driven by a ball at 0.04: the record's phase on the
    ball's schedule is part of the state."""
    def build():
        gm = GraphManager()
        gm.add_node(BallNode("ball", 0.04, initial_position=2.0, gravity=-5.0))
        gm.add_node(SpringDamperNode("spring", 0.01, stiffness=10.0, rest_length=0.3))
        gm.add_edge("ball", "spring", "position", "anchor_position")
        gm.compile()
        return gm
    return build


#: Families whose next step reads nothing carried in ``_meta`` but the
#: multi-rate step counter, which ``start_step`` reconstructs.
PLAIN = {
    "no-predictor": _springs(),
    "aitken": _ball_on_spring(predictor="none", acceleration="aitken"),
    "iqn-ils": _ball_on_spring(predictor="none", acceleration="iqn-ils"),
    "multirate-chain": _chain(),
    "multirate-rods": _rods(False),
    "subcycled-rods": _rods(True),
}
#: Families whose next step reads carried history: a predictor short of
#: convergence, IMVJ warm starts, a sub-cycled group with a predictor.
HISTORY = {
    "linear-predictor": _springs(max_iterations=2, predictor="linear"),
    "quadratic-predictor": _springs(max_iterations=1, predictor="quadratic"),
    "iqn-imvj-warm-start": _imvj_rods(),
    "subcycled-quadratic": _rods(True, max_iterations=1, predictor="quadratic"),
}
FAMILIES = {**PLAIN, **HISTORY}
MULTIRATE = {"multirate-chain", "multirate-rods"}
#: ``run_adaptive`` advances every node by one ``dt``: no multi-rate graph.
ADAPTIVE = sorted(set(FAMILIES) - MULTIRATE)

_H2_REASON = ("H2: windowed_loss zeroes the carried _meta (predictor history, IQN warm "
              "starts) at every window start (sysid.py _state_from_obs), so the replay "
              "starts from another state; pending fix")

_BUILT: dict[str, GraphManager] = {}


def family(name: str) -> GraphManager:
    """One compiled graph per family and module, reset before every use."""
    if name not in _BUILT:
        _BUILT[name] = FAMILIES[name]()
    gm = _BUILT[name]
    gm.reset_state()
    return gm


def record(gm: GraphManager, n_steps: int, *, start: int = 0) -> tuple[dict, int]:
    """``n_steps`` of the graph's own trajectory after ``start`` steps, as
    observations, and the base step it starts at."""
    gm.reset_state()
    if start:
        gm.run(start)
    init = {n: gm.get_node_state(n) for n in gm.node_names}
    _, hist = gm.run_scan_with_history(n_steps)
    gm.reset_state()
    return observations_from_history(init, hist), start


def every_field(states: dict) -> dict:
    """``obs_fn`` comparing every user-state field."""
    return {n: dict(f) for n, f in states.items()}


def subsample(obs: dict, every: int) -> dict:
    return jax.tree.map(lambda x: x[::every], obs)


# ---------------------------------------------------------------------------
# The oracles
# ---------------------------------------------------------------------------

def check_windowed_replay(name: str, *, window: int, sample_every: int, start: int,
                          multiple_shooting: bool, continuity_weight: float = 1.0) -> None:
    gm = family(name)
    n_samples = 12 // sample_every
    obs, start = record(gm, n_samples * sample_every, start=start)
    obs = subsample(obs, sample_every)
    kwargs: dict = dict(obs_fn=every_field, window=window, sample_every=sample_every,
                        start_step=start)
    if multiple_shooting:
        kwargs.update(window_states=init_window_states(obs, window),
                      continuity_weight=continuity_weight)
    loss = float(windowed_loss(gm, gm.params, obs, **kwargs))
    note(f"{name}: window={window} sample_every={sample_every} start={start} "
         f"multiple_shooting={multiple_shooting} -> {loss!r}")
    assert loss == 0.0, (
        f"{name}: the replay of the graph's own record is {loss!r} away from it "
        f"(window={window}, sample_every={sample_every}, start_step={start}, "
        f"multiple shooting={multiple_shooting})")


def check_fit_stays_at_the_truth(name: str, fitter: str) -> None:
    """A fit of two stiffnesses, started at the generating values.

    In identity coordinates: under a ``log`` spec the fitters work in,
    ``constrain(unconstrain(100.0))`` is ``100.00000763`` -- one float32 ulp
    off -- so the loss the fitter evaluates at its start is that of a
    neighbouring point (1.3e-13 here), and Adam, which normalises the
    gradient, takes a full-``lr`` step from it.  That is the transform's
    rounding, not a replay defect, so the oracle states the claim where no
    transform rounds: there the loss at the start is exactly zero, so is
    its gradient, and so is every Adam step."""
    gm = FAMILIES[name]()
    stiff = [(n, "stiffness") for n in gm.node_names
             if "stiffness" in (gm.params["nodes"].get(n) or {})]
    assert stiff, f"{name} has no stiffness to fit"
    for node, key in stiff:
        gm.set_param_spec(node, key, ParamSpec(bounds=(None, None)))
    mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
    for node, key in stiff:
        mask["nodes"][node][key] = True
    obs, _ = record(gm, 12)
    truth = jax.tree.map(lambda x: x, gm.params)
    with warnings.catch_warnings():
        # A guard declining or holding over a zero gradient says so.
        warnings.simplefilter("ignore", RuntimeWarning)
        if fitter == "fit":
            res = fit(gm, lambda p: windowed_loss(gm, p, obs, obs_fn=every_field, window=4,
                                                  start_step=0),
                      params=truth, mask=mask, n_iter=5, lr=0.05)
        else:
            res, _ = fit_multiple_shooting(gm, obs, obs_fn=every_field, window=4,
                                           params=truth, mask=mask, n_iter=5, lr=0.05,
                                           start_step=0)
    note(f"{name} {fitter}: losses {res.losses}")
    assert all(float(x) == 0.0 for x in res.losses), (
        f"{name}: {fitter} started at the truth saw losses {res.losses}")
    for path, (want, got) in zip(
            jax.tree_util.tree_flatten_with_path(truth)[0],
            zip(jax.tree.leaves(truth), jax.tree.leaves(res.params))):
        assert np.asarray(want).tobytes() == np.asarray(got).tobytes(), (
            f"{name}: {fitter} started at the truth moved {jax.tree_util.keystr(path[0])} "
            f"{np.asarray(want)!r} -> {np.asarray(got)!r} (losses {res.losses})")


def check_adaptive_checkpoint(name: str, *, first: float, then: float, ahead: int) -> None:
    """``run_adaptive(first)``, a checkpoint, and ``run_adaptive(then)`` on
    the graph that never stopped and on a second graph -- built and run
    ahead -- that loaded the checkpoint."""
    original = FAMILIES[name]()
    resumed = FAMILIES[name]()
    kwargs = dict(dt_initial=0.01, dt_max=0.02, atol=1e-6, rtol=1e-4)
    original.run_adaptive(first, **kwargs)
    with tmp_dir() as tmp:
        path = original.save_state(checkpoint_path(tmp))
        if ahead:
            resumed.run(ahead)
        resumed.load_state(path)
    assert_trees_identical(full_state(original), full_state(resumed), what="restored state")
    _, info_a = original.run_adaptive(then, **kwargs)
    _, info_b = resumed.run_adaptive(then, **kwargs)
    assert info_a["dt_history"] == info_b["dt_history"], (info_a, info_b)
    assert_trees_identical(full_state(original), full_state(resumed),
                           what="the adaptive run after the checkpoint")


# ---------------------------------------------------------------------------
# Per push
# ---------------------------------------------------------------------------

#: One configuration per family per push that exercises every axis at
#: once: windows shorter than the record, a sample spacing of two, and a
#: multi-rate record started on an odd base step.
_PER_PUSH = dict(window=3, sample_every=2)


def _start_for(name: str) -> int:
    return 3 if name in MULTIRATE else 0


@pytest.mark.parametrize("multiple_shooting", [False, True], ids=["teacher-forced",
                                                                   "multiple-shooting"])
@pytest.mark.parametrize("name", sorted(PLAIN))
def test_a_replay_of_a_graphs_own_record_is_exact(name, multiple_shooting):
    check_windowed_replay(name, start=_start_for(name), multiple_shooting=multiple_shooting,
                          **_PER_PUSH)


@pytest.mark.parametrize("multiple_shooting", [False, True], ids=["teacher-forced",
                                                                   "multiple-shooting"])
@pytest.mark.parametrize("name", sorted(HISTORY))
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_H2_REASON)
def test_a_replay_of_a_record_with_carried_history_is_exact(name, multiple_shooting):
    check_windowed_replay(name, start=0, multiple_shooting=multiple_shooting, **_PER_PUSH)


@pytest.mark.parametrize("name", sorted(HISTORY))
def test_every_history_family_carries_history_its_next_step_reads(name):
    """The fixtures can express H2: in each history family, a step from a
    state whose ``_meta`` is zeroed lands somewhere else than the step from
    the recorded state."""
    gm = family(name)
    gm.run(4)
    warm = {n: dict(f) for n, f in gm._state.items()}                     # noqa: SLF001
    cold = {n: dict(f) for n, f in warm.items()}
    cold["_meta"] = {k: (v if k == "step_count" else jnp.zeros_like(v))
                     for k, v in warm["_meta"].items()}
    step = gm._build_step_fn()                                            # noqa: SLF001
    ext = gm._resolve_external_inputs(None)                               # noqa: SLF001
    a = step(warm, ext, gm.params)
    b = step(cold, ext, gm.params)
    moved = [f"{n}.{k}" for n in gm.node_names for k in a[n]
             if np.asarray(a[n][k]).tobytes() != np.asarray(b[n][k]).tobytes()]
    assert moved, f"{name}: zeroing _meta changes nothing the next step computes"


@pytest.mark.parametrize("fitter", ["fit", "fit_multiple_shooting"])
def test_a_fit_started_at_the_truth_stays_there(fitter):
    check_fit_stays_at_the_truth("no-predictor", fitter)


@pytest.mark.parametrize("fitter", ["fit", "fit_multiple_shooting"])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=_H2_REASON)
def test_a_fit_started_at_the_truth_stays_there_with_a_predictor(fitter):
    check_fit_stays_at_the_truth("quadratic-predictor", fitter)


@pytest.mark.parametrize("name", ADAPTIVE)
def test_a_checkpoint_mid_adaptive_run_resumes_it(name):
    check_adaptive_checkpoint(name, first=0.05, then=0.04, ahead=2)


# ---------------------------------------------------------------------------
# Slow lane: drawn configurations
# ---------------------------------------------------------------------------

@st.composite
def replays(draw, names):
    name = draw(st.sampled_from(sorted(names)), label="family")
    sample_every = draw(st.sampled_from([1, 2, 3]), label="sample_every")
    n_samples = 12 // sample_every
    window = draw(st.sampled_from([w for w in range(1, n_samples + 1) if n_samples % w == 0]),
                  label="window")
    start = draw(st.integers(0, 5), label="start") if name in MULTIRATE else 0
    multiple_shooting = draw(st.booleans(), label="multiple shooting")
    weight = draw(st.sampled_from([0.0, 1.0, 1e3]), label="continuity weight")
    return dict(name=name, window=window, sample_every=sample_every, start=start,
                multiple_shooting=multiple_shooting, continuity_weight=weight)


# Per push: tests/property/test_differential_replay.py::test_a_replay_of_a_graphs_own_record_is_exact
@pytest.mark.slow  # a windowed loss traced and compiled per drawn window shape
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(case=replays(PLAIN))
def test_a_replay_of_a_graphs_own_record_is_exact_in_every_configuration(case):
    check_windowed_replay(**case)


# Per push: tests/property/test_differential_replay.py::test_a_checkpoint_mid_adaptive_run_resumes_it
@pytest.mark.slow  # two graphs compiled and two adaptive runs per example
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(name=st.sampled_from(ADAPTIVE), first=st.sampled_from([0.01, 0.03, 0.07]),
       then=st.sampled_from([0.02, 0.05]), ahead=st.integers(0, 3))
def test_a_checkpoint_mid_adaptive_run_resumes_it_anywhere(name, first, then, ahead):
    check_adaptive_checkpoint(name, first=first, then=then, ahead=ahead)
