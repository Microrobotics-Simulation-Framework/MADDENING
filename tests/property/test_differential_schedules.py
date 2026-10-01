"""Differential oracles: multi-rate, sub-cycling and adaptive stepping.

Three schedules that each have a second, simpler path to the same answer,
stated over generated coupled graphs of synthetic nodes
(:mod:`tests.property.coupled_graphs`):

**Multi-rate == hand-unrolled.**  A multi-rate graph gates every block on
``step_count % divider`` inside one compiled program, with the coupling
group's solve under a ``lax.cond``.
:class:`~tests.property.coupled_graphs.HandUnrolledMultirate` writes the
same schedule out as a Python loop: a block fires or is not touched; the
group is a *uniform-rate* graph of its members, stepped only when it fires,
so its ``_meta`` (report, predictor history, IMVJ warm start) can only ever
come from applied solves.  On top of that, within the library's own run, a
step on which a block does not fire must leave its state and its ``_meta``
slots bit-identical.

**Sub-cycled == uniform-rate.**  Under ``boundary_interpolation="constant"``
a sub-cycled member takes ``d`` sub-steps of its own timestep per pass,
each reading the in-pass state of its sources.  That is
:class:`~tests.property.coupled_graphs.SubStepped`'s update at the macro
timestep, written by hand, in a group that does not sub-cycle: the same
map, so the same passes, the same state to round-off, and **exactly** the
same clocks.  ``"linear"`` interpolation is documented to coincide with
``"constant"`` bit for bit under Jacobi and whenever every source of the
sub-cycled node is scheduled after it; ``"quadratic"`` is ``"linear"``
(MADD-ANO-027, decided); each later waveform sweep adds at least a pass
(decided).

**Adaptive at a fixed accepted dt == run_scan.**  An accepted adaptive step
is two half steps of the dt-parameterised step, so ``run_adaptive`` and
``run_adaptive_scan`` pinned to one ``dt`` are ``run_scan`` of the same
graph at half the timestep.  Replaying ``run_adaptive``'s accepted
``dt_history`` through that step reproduces its state, and every node's
clock reads the time the stepper reports (MADD-ANO-061).

Tolerances.  Two separately compiled programs evaluate the same arithmetic,
so they may round differently by an ulp per pass (as the two solvers do,
``test_differential_coupling_solvers.py``): states agree to
``_ULPS_PER_PASS`` per pass, counted over every pass of the run.  The
multi-rate gate and the replay, which compare a program with *itself*, are
bitwise; clocks are dyadic sums and are compared exactly.

What these cannot see: the shared one-pass map and the shared coupling
solve -- a fault inside ``_run_coupled_block_impl`` that every path calls
the same way is invisible here; that is the solver and fixed-point
oracles' business.
"""

from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from tests.conftest import EXAMPLES_COSTLY
from tests.core.test_coupling_solver_equivalence import residual_noise_floor
from tests.property import coupled_graphs as cg

_ULPS_PER_PASS = 4


def _key(group: dict) -> tuple:
    return tuple(sorted(group.items()))


def _leaves_at(gdef, state, updates: dict):
    for nd in gdef.nodes:
        for field, want in cg.expected_leaves(nd, updates[nd.name]).items():
            got = np.asarray(state[nd.name][field])
            assert got.dtype == want.dtype and got.tobytes() == want.tobytes(), (
                f"{nd.name}.{field} = {got!r} after {updates[nd.name]} updates, "
                f"want {want!r}")


# ---------------------------------------------------------------------------
# Multi-rate == hand-unrolled
# ---------------------------------------------------------------------------


def _multirate_structure(group_dt: float, sink_dt: float) -> cg.GraphDef:
    """Driver (1) -> pair (group_dt) -> sink (sink_dt) -> driver, a back edge.

    The back edge makes the driver read the sink's state from the start of
    the base step, so the reference has to get the forward/back split
    right as well as the gating.
    """
    leaves = cg.LEAF_POOL
    nodes = (
        cg.NodeDef("drv", 1, timestep=1.0, alpha=1.0, beta=1.0, leaves=leaves),
        cg.NodeDef("g0", 2, timestep=group_dt, alpha=0.5, beta=1.0, leaves=leaves),
        cg.NodeDef("g1", 1, timestep=group_dt, alpha=-0.25, beta=-0.5, leaves=leaves),
        cg.NodeDef("sink", 1, timestep=sink_dt, alpha=0.5, leaves=leaves),
    )
    edges = (cg.EdgeDef("g1", "g0", 0), cg.EdgeDef("g0", "g1", 0),
             cg.EdgeDef("drv", "g0", 1), cg.EdgeDef("g1", "sink", 0),
             cg.EdgeDef("sink", "drv", 0))
    return cg.GraphDef(n=2, nodes=nodes, edges=edges, group_nodes=("g0", "g1"))


@functools.lru_cache(maxsize=None)
def _multirate_pair(group_dt, sink_dt, group_items):
    gdef = _multirate_structure(group_dt, sink_dt)
    group = dict(group_items)
    gm = cg.build_graph(gdef, group)
    ref = cg.HandUnrolledMultirate(gdef, group, gm.schedule, base_dt=1.0)
    return gdef, gm, ref


def assert_multirate_matches_hand_unrolled(gdef, gm, ref, values, n_steps):
    cg.set_initial(gm, values)
    params = cg.params_for(gm, values)
    rtraj, rmetas = ref.run(values, n_steps)
    dividers = ref.divider
    passes = 0
    fired = {nd.name: 0 for nd in gdef.nodes}
    for k in range(n_steps):
        before, bmeta = cg.snapshot(gm), cg.group_meta(gm, gdef.key)
        gm.step(params=params)
        after, ameta = cg.snapshot(gm), cg.group_meta(gm, gdef.key)
        for nd in gdef.nodes:
            if k % dividers[nd.name] == 0:
                fired[nd.name] += 1
            else:
                moved = cg.bitwise_differences(before[nd.name], after[nd.name])
                assert not moved, f"base step {k}: {nd.name} does not fire but moved {moved}"
        if k % dividers[gdef.group_nodes[0]] != 0:
            changed = [s for s in bmeta if bmeta[s].tobytes() != ameta[s].tobytes()]
            assert not changed, (
                f"base step {k}: the group does not fire but its _meta slots "
                f"{changed} changed")
        else:
            passes += int(ameta.get("iterations", 1)) if "iterations" in ameta else 1
        _leaves_at(gdef, after, fired)
        # Against the hand-unrolled loop: the report and warm-start slots
        # the two share, and the states to round-off.
        shared = sorted(set(ameta) & set(rmetas[k]))
        if "iterations" in shared:
            assert int(ameta["iterations"]) == int(rmetas[k]["iterations"]), (
                f"base step {k}: {int(ameta['iterations'])} passes against the "
                f"hand-unrolled loop's {int(rmetas[k]['iterations'])}")
        bound = _ULPS_PER_PASS * cg.EPS32 * max(passes, 1)
        gap = cg.relative_gap(after, rtraj[k])
        assert gap <= bound, (
            f"base step {k}: {gap:.3e} from the hand-unrolled loop (bound {bound:.3e})")
        for slot in shared:
            if slot in ("V", "W") or slot.startswith("pred_"):
                g = float(np.max(np.abs(ameta[slot].astype(np.float64)
                                        - rmetas[k][slot].astype(np.float64))))
                scale = max(float(np.max(np.abs(rmetas[k][slot]))), 1e-30) \
                    if rmetas[k][slot].size else 1.0
                assert g <= bound * max(scale, 1.0), (
                    f"base step {k}: warm-start slot {slot} differs by {g:.3e}")


_MULTIRATE_CASES = {
    "ift-iqn-imvj-linear-2": (2.0, 1.0, dict(
        acceleration="iqn-imvj", jacobian_reuse=2, tolerance=1e-4,
        max_iterations=12, predictor="linear")),
    "fori-aitken-diagnostics-3": (3.0, 2.0, dict(
        solver="fori", diagnostics=True, acceleration="aitken",
        convergence_norm="mixed", rtol=1e-3, max_iterations=12)),
    "ift-jacobi-quadratic-2-6": (2.0, 6.0, dict(
        iteration_mode="jacobi", tolerance=1e-4, max_iterations=30,
        predictor="quadratic")),
}


@pytest.mark.parametrize("case", sorted(_MULTIRATE_CASES))
# Costly tier: an example is a 7-10 step rollout of the multi-rate graph
# and of the hand-unrolled loop, which steps its sub-graph and jitted node
# updates one call at a time.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_multirate_graph_steps_like_its_hand_unrolled_schedule(case, data):
    """Multi-rate == hand-unrolled over drawn spectra (per push).

    Slow sibling: :func:`test_generated_multirate_graphs_step_like_their_schedule`.
    """
    group_dt, sink_dt, group = _MULTIRATE_CASES[case]
    gdef, gm, ref = _multirate_pair(group_dt, sink_dt, _key(group))
    values = data.draw(cg.drawn_values(gdef, rhos=(0.3, 0.9, 0.99)))
    assert_multirate_matches_hand_unrolled(gdef, gm, ref, values, 3 * int(group_dt) + 1)


# Slow: structure and configuration are drawn, so every example builds and
# compiles graphs of its own (seconds each on CI).
# Per push: tests/property/test_differential_schedules.py::test_a_multirate_graph_steps_like_its_hand_unrolled_schedule
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_generated_multirate_graphs_step_like_their_schedule(data):
    """Multi-rate == hand-unrolled with the dividers and configuration drawn.

    Per-push sibling: :func:`test_a_multirate_graph_steps_like_its_hand_unrolled_schedule`.
    """
    group_dt = float(data.draw(st.sampled_from([2, 3, 4])))
    sink_dt = float(data.draw(st.sampled_from([1, 2, 3, 6])))
    gdef = _multirate_structure(group_dt, sink_dt)
    group = data.draw(cg.group_configs(gdef, caps=(1, 2, 5, 12, 30)))
    group = dict(group, solver=data.draw(st.sampled_from(["ift", "fori"])),
                 diagnostics=data.draw(st.booleans()))
    gdef = cg.steer_leaves_around_known_crashes(gdef, group)
    note(f"group_dt={group_dt} sink_dt={sink_dt} {group}")
    gm = cg.build_graph(gdef, group)
    ref = cg.HandUnrolledMultirate(gdef, group, gm.schedule, base_dt=1.0)
    values = data.draw(cg.drawn_values(gdef, rhos=(0.3, 0.9, 0.99)))
    assert_multirate_matches_hand_unrolled(gdef, gm, ref, values, 3 * int(group_dt) + 1)


# ---------------------------------------------------------------------------
# Sub-cycled == uniform-rate reference
# ---------------------------------------------------------------------------

#: The macro timestep: dyadic, so ``T / d`` and every clock sum are exact.
_MACRO = 0.5


def _subcycled(m: int, d: int, fast: str, *, chords=()) -> cg.GraphDef:
    gdef = cg._cycle(m, 1, chords=chords, leaves=cg.LEAF_POOL)
    ts = {nd.name: _MACRO for nd in gdef.nodes}
    ts[fast] = _MACRO / d
    return gdef.with_timesteps(ts)


@functools.lru_cache(maxsize=None)
def _subcycled_pair(m, d, fast, chords, group_items, interp):
    gdef = _subcycled(m, d, fast, chords=chords)
    group = dict(group_items)
    gm = cg.build_graph(gdef, dict(group, subcycling=True, boundary_interpolation=interp))
    ref = cg.build_graph(gdef, group, substep={fast: d})
    return gdef, gm, ref


def assert_subcycled_matches_reference(gdef, group, d, fast, sub, ref, steps=4):
    passes = 0
    for k, ((s, _m, r), (s_ref, _mr, r_ref)) in enumerate(zip(sub, ref), start=1):
        updates = {nd.name: k * (d if nd.name == fast else 1) for nd in gdef.nodes}
        _leaves_at(gdef, s, updates)
        _leaves_at(gdef, s_ref, updates)
        for nm in gdef.group_nodes:
            # Exact: every clock is a sum of dyadic timesteps.
            assert float(s[nm]["clock"]) == k * _MACRO, (
                f"step {k}: {nm}'s clock reads {float(s[nm]['clock'])!r}, the graph's "
                f"time is {k * _MACRO!r}")
            assert s[nm]["clock"].tobytes() == s_ref[nm]["clock"].tobytes()
        if r is not None and r_ref is not None:
            assert r["iterations"] == r_ref["iterations"], (
                f"step {k}: the sub-cycled group took {r['iterations']} passes, "
                f"its uniform-rate reference {r_ref['iterations']}")
            assert r["converged"] == r_ref["converged"]
            passes += r["iterations"] * d
        else:
            passes += int(group.get("max_iterations", 10)) * d
        bound = _ULPS_PER_PASS * cg.EPS32 * max(passes, 1)
        gap = cg.relative_gap(s, s_ref)
        assert gap <= bound, (
            f"step {k}: sub-cycled state {gap:.3e} from the uniform-rate "
            f"reference (bound {bound:.3e})")


_SUBCYCLING_CASES = {
    "gs-fast-last-aitken": (3, 4, "g2", ((0, 2),), dict(
        acceleration="aitken", tolerance=1e-4, max_iterations=20)),
    "jacobi-fast-first-iqn": (2, 2, "g0", (), dict(
        acceleration="iqn-ils", iteration_mode="jacobi", tolerance=1e-4,
        max_iterations=20, predictor="linear")),
    "gs-fast-middle-imvj-interface": (3, 4, "g1", (), dict(
        acceleration="iqn-imvj", jacobian_reuse=2, convergence_norm="interface",
        rtol=1e-3, max_iterations=20)),
}


@pytest.mark.parametrize("case", sorted(_SUBCYCLING_CASES))
# Costly tier: two four-step rollouts of compiled sub-cycling graphs.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_subcycled_group_steps_like_its_uniform_rate_reference(case, data):
    """Sub-cycled (constant) == hand-written sub-stepping node (per push).

    Slow sibling: :func:`test_generated_subcycled_groups_step_like_their_reference`.
    """
    m, d, fast, chords, group = _SUBCYCLING_CASES[case]
    gdef, gm, ref = _subcycled_pair(m, d, fast, chords, _key(group), "constant")
    values = data.draw(cg.drawn_values(gdef, rhos=(0.3, 0.9, 0.99)))
    sub = cg.trajectory(gm, gdef, values, 4)
    reference = cg.trajectory(ref, gdef, values, 4)
    assert_subcycled_matches_reference(gdef, group, d, fast, sub, reference)


# Slow: structure and configuration are drawn, so every example builds and
# compiles graphs of its own (seconds each on CI).
# Per push: tests/property/test_differential_schedules.py::test_a_subcycled_group_steps_like_its_uniform_rate_reference
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_generated_subcycled_groups_step_like_their_reference(data):
    """Sub-cycled == uniform-rate with the group, divider and config drawn.

    Per-push sibling: :func:`test_a_subcycled_group_steps_like_its_uniform_rate_reference`.
    """
    m = data.draw(st.integers(2, 4))
    d = data.draw(st.sampled_from([2, 3, 4]))
    fast = f"g{data.draw(st.integers(0, m - 1))}"
    gdef = _subcycled(m, d, fast)
    group = data.draw(cg.group_configs(gdef, caps=(1, 2, 5, 20)))
    group = dict(group, solver=data.draw(st.sampled_from(["ift", "fori"])),
                 diagnostics=True)
    gdef = cg.steer_leaves_around_known_crashes(gdef, group)
    note(f"m={m} d={d} fast={fast} {group}")
    gm = cg.build_graph(gdef, dict(group, subcycling=True, boundary_interpolation="constant"))
    ref = cg.build_graph(gdef, group, substep={fast: d})
    values = data.draw(cg.drawn_values(gdef, rhos=(0.3, 0.9, 0.99)))
    assert_subcycled_matches_reference(
        gdef, group, d, fast, cg.trajectory(gm, gdef, values, 4),
        cg.trajectory(ref, gdef, values, 4))


def _interp_runs(m, d, fast, group, values, interps, steps=3):
    out = {}
    for interp in interps:
        gdef, gm, _ref = _subcycled_pair(m, d, fast, (), _key(group), interp)
        out[interp] = cg.trajectory(gm, gdef, values, steps)
    return out


@pytest.mark.parametrize("mode,fast", [("jacobi", "g1"), ("gauss-seidel", "g0")])
# Costly tier: three three-step rollouts of compiled sub-cycling graphs.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_linear_interpolation_is_constant_where_documented_and_quadratic_is_linear(
        mode, fast, data):
    """``"quadratic"`` == ``"linear"`` bitwise; ``"linear"`` vs ``"constant"`` as documented.

    ``"quadratic"`` never receives its third value and *is* ``"linear"``:
    the same traced program, so bitwise (MADD-ANO-027, decided; pinned,
    not reported).  ``boundary_interpolation`` documents that the two ends
    ``"linear"`` interpolates between are both estimates of the
    end-of-step value, so it coincides with ``"constant"`` under Jacobi
    (both ends are the incoming iterate) and under Gauss-Seidel when every
    source of the sub-cycled node is scheduled after it (``g0`` is first
    in this group's sweep): the interpolated value is ``a + alpha (a - a)
    = a`` exactly.  The docstring says "bit-identical"; the two settings
    are different compiled programs, and like the two solvers they round
    an ulp apart now and then (measured: 9.2e-08 relative at the second
    step of a two-dimensional Jacobi group), so the claim held here is
    ``_ULPS_PER_PASS`` per pass with equal pass counts.  Otherwise -- a
    fast node whose sources move earlier in the pass -- the docstring says
    they differ by "about the tolerance per step", which is not a bound
    anything derives, so nothing is asserted there.
    """
    group = dict(iteration_mode=mode, tolerance=1e-4, max_iterations=40, diagnostics=True)
    gdef = _subcycled(3, 4, fast)
    values = data.draw(cg.drawn_values(gdef, rhos=(0.3, 0.6)))
    runs = _interp_runs(3, 4, fast, group, values, ("constant", "linear", "quadratic"))
    coincide = mode == "jacobi" or fast == "g0"
    passes = 0
    for k in range(3):
        (s_c, _mc, r_c), (s_l, _ml, r_l), (s_q, _mq, _rq) = (
            runs["constant"][k], runs["linear"][k], runs["quadratic"][k])
        assert not cg.bitwise_differences(s_l, s_q), (
            f"step {k + 1}: quadratic interpolation is not linear (MADD-ANO-027)")
        gap = cg.relative_gap(s_l, s_c)
        if coincide:  # always, in the two cases drawn here
            if r_l["iterations"] != r_c["iterations"]:
                # An ulp apart, the two may stop on adjacent passes only
                # where the estimate is within its own rounding of the
                # threshold; after that the trajectories have parted.
                floor = residual_noise_floor("l2", 1e-6, 3 * 3)
                assert not (cg.criterion_is_resolved(r_l, floor)
                            and cg.criterion_is_resolved(r_c, floor)), (
                    f"step {k + 1}: linear took {r_l['iterations']} passes where "
                    f"constant took {r_c['iterations']}, with both verdicts resolved")
                return
            passes += r_l["iterations"] * 4
            bound = _ULPS_PER_PASS * cg.EPS32 * passes
            assert gap <= bound, (
                f"step {k + 1}: linear and constant interpolation {gap:.3e} apart where "
                f"documented to coincide ({mode}, fast node {fast}; bound {bound:.3e})")


def test_each_later_waveform_sweep_adds_at_least_one_pass():
    """Decided behaviour, pinned: a sweep is a restart that starts with a pass.

    ``waveform_iterations=w`` runs the same fixed-point solve ``w`` times,
    each from where the last stopped, and every sweep begins with one pass
    -- so ``total_iterations`` is at least the single-sweep count plus
    ``w - 1``, and a converged single sweep stays converged.
    """
    gdef = _subcycled(2, 2, "g1")
    values = cg.draw_values(np.random.default_rng(7), gdef, 0.5)
    base = dict(tolerance=1e-4, max_iterations=60, diagnostics=True)
    one = cg.trajectory(cg.build_graph(gdef, dict(base, subcycling=True)), gdef, values, 2)
    for w in (2, 3):
        many = cg.trajectory(cg.build_graph(
            gdef, dict(base, subcycling=True, waveform_iterations=w)), gdef, values, 1)
        r1, rw = one[0][2], many[0][2]
        assert r1["converged"] and rw["converged"]
        assert rw["total_iterations"] >= r1["total_iterations"] + (w - 1)


# ---------------------------------------------------------------------------
# Adaptive == run_scan at a fixed accepted dt; replay; clocks
# ---------------------------------------------------------------------------

#: The pinned adaptive step and its half: dyadic, so every clock is exact.
_H = 0.25
_N = 4


#: Leaves the adaptive oracles leave out, pending the strict xfails at the
#: end of this module: the adaptive error norm subtracts every leaf, so a
#: ``bool`` raises and a ``uint32`` wraps -- an unread tag reads an error of
#: order one even at ``atol=1e9`` and the "pinned" stepper rejects.
_ADAPTIVE_DROP = ("flag", "tag")

#: Rates for the explicit-Euler nodes: the coupling gain a pass sees is
#: ``dt * G``, so at ``dt = 0.25`` these are rates of about 0.3 and 0.9.
_ODE_RHOS = (1.2, 3.6)


def _adaptive_structure(structure):
    return cg.STRUCTURES[structure].as_ode(-1.0).without_leaves(_ADAPTIVE_DROP)


@functools.lru_cache(maxsize=None)
def _adaptive_pair(structure, group_items, subcycle_fast=None, d=2):
    gdef = _adaptive_structure(structure)
    group = dict(group_items)
    if subcycle_fast is None:
        full = gdef.with_timesteps({nd.name: _H for nd in gdef.nodes})
        half = gdef.with_timesteps({nd.name: _H / 2 for nd in gdef.nodes})
        gm = cg.build_graph(full, group)
        ref = cg.build_graph(half, group)
    else:
        full = gdef.with_timesteps({nd.name: (_H / d if nd.name == subcycle_fast else _H)
                                    for nd in gdef.nodes})
        half = gdef.with_timesteps({nd.name: (_H / 2 / d if nd.name == subcycle_fast else _H / 2)
                                    for nd in gdef.nodes})
        gm = cg.build_graph(full, dict(group, subcycling=True, boundary_interpolation="constant"))
        ref = cg.build_graph(half, dict(group, subcycling=True, boundary_interpolation="constant"))
    return full, gm, ref


def _pinned_adaptive(gm, values, scan: bool):
    cg.set_initial(gm, values)
    params = cg.params_for(gm, values)
    kw = dict(dt_initial=_H, atol=1e9, rtol=0.0, dt_min=_H, dt_max=_H, params=params)
    if scan:
        _s, _hist, info = gm.run_adaptive_scan(_N * _H, max_steps=_N + 2, **kw)
        return cg.snapshot(gm), int(info["n_steps"]), float(info["final_t"])
    _s, info = gm.run_adaptive(_N * _H, **kw)
    return cg.snapshot(gm), info["n_steps"], info["t_history"][-1]


def assert_pinned_adaptive_is_run_scan(gdef, gm, ref, values, fast=None, d=1):
    cg.set_initial(ref, values)
    ref.run_scan(2 * _N, params=cg.params_for(ref, values))
    want = cg.snapshot(ref)
    want_meta = cg.group_meta(ref, gdef.key)
    for scan in (False, True):
        got, n, t = _pinned_adaptive(gm, values, scan)
        label = "run_adaptive_scan" if scan else "run_adaptive"
        assert n == _N and t == _N * _H, f"{label}: {n} steps to t={t}"
        got_meta = cg.group_meta(gm, gdef.key)
        for nm in gdef.group_nodes:
            assert float(got[nm]["clock"]) == t, (
                f"{label}: {nm}'s clock {float(got[nm]['clock'])!r} against the "
                f"stepper's time {t!r}")
        updates = {nd.name: 2 * _N * (d if nd.name == fast else 1) for nd in gdef.nodes}
        _leaves_at(gdef, got, updates)
        passes = 2 * _N * int(gm._coupling_groups[0].max_iterations) * d  # noqa: SLF001
        bound = _ULPS_PER_PASS * cg.EPS32 * passes
        gap = cg.relative_gap(got, want)
        assert gap <= bound, (
            f"{label} pinned at dt={_H}: {gap:.3e} from run_scan at {_H / 2} (bound {bound:.3e})")
        if "iterations" in got_meta and "iterations" in want_meta:
            assert int(got_meta["iterations"]) == int(want_meta["iterations"]), label


_ADAPTIVE_CASES = {
    "triangle-iqn-predictor": ("triangle", dict(
        acceleration="iqn-ils", tolerance=1e-5, max_iterations=20, predictor="linear"), None),
    "flux-fori-aitken": ("flux-pair", dict(
        solver="fori", diagnostics=True, acceleration="aitken", tolerance=1e-5,
        max_iterations=20), None),
    "triangle-subcycled": ("triangle", dict(tolerance=1e-5, max_iterations=20), "g1"),
}


@pytest.mark.parametrize("case", sorted(_ADAPTIVE_CASES))
# Costly tier: per example, two adaptive runs and one scan of compiled graphs.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_adaptive_stepping_at_a_pinned_dt_is_run_scan_at_half_the_timestep(case, data):
    """``run_adaptive`` / ``run_adaptive_scan`` pinned == ``run_scan`` (per push).

    ``dt_min = dt_max = dt_initial`` and an ``atol`` no error reaches, so
    every attempt is accepted at ``dt``: each accepted step is two half
    steps, i.e. ``run_scan`` of the same graph built at ``dt / 2``.  The
    sub-cycled case also checks that a fast member's sub-steps cover the
    step (MADD-ANO-046's adaptive half): its clock reads the stepper's
    time exactly.  Slow sibling:
    :func:`test_generated_adaptive_runs_replay_and_keep_their_clocks`.
    """
    structure, group, fast = _ADAPTIVE_CASES[case]
    gdef, gm, ref = _adaptive_pair(structure, _key(group), fast)
    values = data.draw(cg.drawn_values(gdef, rhos=_ODE_RHOS))
    assert_pinned_adaptive_is_run_scan(gdef, gm, ref, values, fast=fast,
                                       d=2 if fast else 1)


_REPLAY_STEPS: dict = {}


def _replay(gm, values, dt_history):
    """``run_adaptive``'s accepted steps, replayed: two half steps each.

    Through the dt-parameterised step ``run_adaptive`` itself jits, built
    once per graph -- the replay checks the stepper's bookkeeping (which
    attempt it kept, which ``dt`` it recorded), not the step.
    """
    cg.set_initial(gm, values)
    params = cg.params_for(gm, values)
    if id(gm) not in _REPLAY_STEPS:
        _REPLAY_STEPS[id(gm)] = (gm, jax.jit(gm._build_dt_step_fn(collect_strict=True)))  # noqa: SLF001
    step = _REPLAY_STEPS[id(gm)][1]
    ext = gm._resolve_external_inputs(None)  # noqa: SLF001
    state = gm._state  # noqa: SLF001
    for dt in dt_history:
        half = jnp.array(dt) / 2.0
        state, _ = step(state, ext, half, params)
        state, _ = step(state, ext, half, params)
    return {nm: {f: np.asarray(v) for f, v in state[nm].items()} for nm in gm.node_names}


def assert_adaptive_replays_and_keeps_its_clock(gdef, gm, values, *, atol, dt_min):
    cg.set_initial(gm, values)
    params = cg.params_for(gm, values)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Adaptive stepper hit dt_min")
        _s, info = gm.run_adaptive(1.0, dt_initial=0.25, atol=atol, rtol=atol,
                                   dt_min=dt_min, dt_max=0.25, params=params)
    got = cg.snapshot(gm)
    t = info["t_history"][-1]
    n = info["n_steps"]
    cg.note(f"n_steps={n} rejected={info['n_rejected']} dt={info['dt_history']}")
    # The clock is a float32 sum of 2n half steps; the stepper's time a
    # float64 sum of n steps.  Each float32 addition rounds by at most
    # half an ulp of the running total.
    tol = 2 * n * cg.EPS32 * max(abs(t), 1.0)
    for nm in gdef.group_nodes:
        assert abs(float(got[nm]["clock"]) - t) <= tol, (
            f"{nm}'s clock {float(got[nm]['clock'])!r} against the stepper's time {t!r} "
            f"after {n} accepted steps (MADD-ANO-061)")
    updates = {nd.name: 2 * n for nd in gdef.nodes}
    _leaves_at(gdef, got, updates)
    replayed = _replay(gm, values, info["dt_history"])
    moved = cg.bitwise_differences(got, replayed)
    assert not moved, f"replaying dt_history moved {moved}"


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("atol,dt_min", [(1e-2, 1e-3), (1e-7, 0.1)])
def test_adaptive_runs_replay_their_accepted_steps_and_keep_their_clocks(atol, dt_min, seed):
    """Replaying ``dt_history`` reproduces ``run_adaptive`` bitwise (per push).

    The ``(1e-7, 0.1)`` case drives the controller to ``dt_min`` and
    through the accept-with-warning path that MADD-ANO-061 fixed: the
    accepted attempt covered more than ``dt_min``, and the clock must say
    so.  Fixed draws rather than a property: ``run_adaptive`` jits its
    step afresh on every call, a compile per run.  Slow sibling (drawn):
    :func:`test_generated_adaptive_runs_replay_and_keep_their_clocks`.
    """
    group = dict(acceleration="aitken", tolerance=1e-5, max_iterations=20)
    gdef, gm, _ref = _adaptive_pair("triangle", _key(group))
    values = cg.draw_values(np.random.default_rng(seed), gdef, _ODE_RHOS[seed % 2],
                            nonnormal=bool(seed % 2))
    assert_adaptive_replays_and_keeps_its_clock(gdef, gm, values, atol=atol, dt_min=dt_min)


# Slow: structure and configuration are drawn, so every example builds and
# compiles graphs of its own (seconds each on CI).
# Per push: tests/property/test_differential_schedules.py::test_adaptive_stepping_at_a_pinned_dt_is_run_scan_at_half_the_timestep
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_generated_adaptive_runs_replay_and_keep_their_clocks(data):
    """The adaptive oracles with the structure and configuration drawn.

    Per-push siblings:
    :func:`test_adaptive_stepping_at_a_pinned_dt_is_run_scan_at_half_the_timestep`
    and :func:`test_adaptive_runs_replay_their_accepted_steps_and_keep_their_clocks`.
    """
    gdef = data.draw(cg.graph_defs(allow_flux=False))
    gdef = gdef.with_leaves(cg.LEAF_POOL, names=[nd.name for nd in gdef.nodes])
    gdef = gdef.as_ode(-1.0).without_leaves(_ADAPTIVE_DROP)
    group = data.draw(cg.group_configs(gdef, caps=(1, 2, 5, 20)))
    group = dict(group, solver=data.draw(st.sampled_from(["ift", "fori"])))
    gdef = cg.steer_leaves_around_known_crashes(gdef, group)
    note(f"{gdef}\n{group}")
    full = gdef.with_timesteps({nd.name: _H for nd in gdef.nodes})
    half = gdef.with_timesteps({nd.name: _H / 2 for nd in gdef.nodes})
    gm = cg.build_graph(full, group)
    ref = cg.build_graph(half, group)
    values = data.draw(cg.drawn_values(gdef, rhos=_ODE_RHOS))
    assert_pinned_adaptive_is_run_scan(full, gm, ref, values)
    atol, dt_min = data.draw(st.sampled_from([(1e-2, 1e-3), (1e-7, 0.1), (1e-4, 1e-2)]))
    assert_adaptive_replays_and_keeps_its_clock(full, gm, values, atol=atol, dt_min=dt_min)


# ---------------------------------------------------------------------------
# Disagreements: the adaptive error norm reads non-float leaves
# ---------------------------------------------------------------------------


def _adaptive_dt_history(leaves, scan=False):
    """The accepted steps of an explicit-Euler pair carrying *leaves*.

    ``x' = -x + G u + b`` on a two-node cycle: a consistent time
    discretisation, so the step-doubling error estimate shrinks with
    ``dt`` and the controller settles.
    """
    base = cg._cycle(2, 1, outside=False, leaves=()).as_ode(-1.0)
    gdef = base.with_leaves(leaves).with_timesteps({"g0": 0.25, "g1": 0.25})
    gm = _leaf_graph(gdef)
    values = cg.draw_values(np.random.default_rng(0), base, 2.0)
    cg.set_initial(gm, values)
    kw = dict(dt_initial=0.25, atol=1e-4, rtol=1e-4, dt_min=1e-3, dt_max=0.25,
              params=cg.params_for(gm, values))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Adaptive stepper hit dt_min")
        if scan:
            _s, hist, info = gm.run_adaptive_scan(2.0, max_steps=400, **kw)
            return int(info["n_steps"]), np.asarray(hist["g0"]["x"])
        _s, info = gm.run_adaptive(2.0, **kw)
    return info["n_steps"], info["dt_history"]


@functools.lru_cache(maxsize=None)
def _leaf_graph(gdef):
    return cg.build_graph(gdef, dict(tolerance=1e-6, max_iterations=30))


@pytest.mark.parametrize("scan", [False, True], ids=["run_adaptive", "run_adaptive_scan"])
@pytest.mark.parametrize("leaf", [
    pytest.param("count", marks=pytest.mark.xfail(strict=True, reason=(
        "differential: the adaptive error norm counts an int32 leaf (a counter "
        "nothing reads changes the RMS and so the accepted dt sequence); "
        "pending fix"))),
    pytest.param("tag", marks=pytest.mark.xfail(strict=True, reason=(
        "differential: the adaptive error norm subtracts a uint32 leaf, which "
        "wraps, so an unread tag dominates the error and the controller "
        "rejects nearly every step; pending fix"))),
])
def test_a_non_float_leaf_nothing_reads_does_not_change_the_adaptive_steps(leaf, scan):
    """Adaptive stepping with and without an integer leaf no edge reads.

    ``_tree_error_norm`` maps over every leaf of the user state and
    computes ``|fine - coarse| / (atol + rtol max(|fine|, |coarse|))``,
    then the RMS over every element.  A counter is ``k + 2`` after two
    half steps and ``k + 1`` after one full step *by construction*, so an
    ``int32`` counter adds a term and an element: from ``2**24 + 1`` the
    term is small and the extra element dilutes the RMS (77 accepted steps
    become 65, and the state moves 0.7%); a counter from ``0`` would be
    ``1 / (atol + rtol k)`` and dominate.  A ``uint32`` tag's ``fine -
    coarse`` wraps modulo ``2**32`` (199,981 rejections in 199,991
    attempts; ``run_adaptive_scan`` ends at ``t = 0.0019`` of ``2.0`` when
    ``max_steps`` runs out).  A ``bool`` cannot be subtracted at all (the
    next test).  A *float* field that is the same in both estimates would
    also dilute the RMS -- that is the RMS convention every adaptive ODE
    solver uses, and not what this checks.
    """
    base_n, base_steps = _adaptive_dt_history((), scan)
    n, steps = _adaptive_dt_history((leaf,), scan)
    assert n == base_n
    assert np.array_equal(np.asarray(steps), np.asarray(base_steps))


@pytest.mark.xfail(strict=True, raises=TypeError, reason=(
    "differential: run_adaptive / run_adaptive_scan raise TypeError on any "
    "graph with a bool state leaf (the error norm subtracts it); pending fix"))
@pytest.mark.parametrize("scan", [False, True], ids=["run_adaptive", "run_adaptive_scan"])
def test_adaptive_stepping_runs_a_graph_with_a_boolean_leaf(scan):
    """``step`` / ``run_scan`` run this graph; the adaptive steppers raise."""
    n, _steps = _adaptive_dt_history(("flag",), scan)
    assert n > 0
