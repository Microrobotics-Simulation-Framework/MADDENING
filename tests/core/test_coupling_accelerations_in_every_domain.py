"""The acceleration mechanics' claims, in every numeric domain.

Seven rows of ``docs/validation/coupling_claims.yaml`` describe what an
acceleration does inside the coupling loop, each verified in float32 on a
single-rate graph:

* CPL-049: under ``acceleration="fixed"`` the error estimate sums the
  steps the relaxed iterate takes, so it is the distance past the first
  pass whatever the relaxation;
* CPL-050: on the first loop pass an under-relaxed group translates the
  rate it read into the relaxed iteration's, so wherever it stops the
  estimate is not below the distance;
* CPL-061: Aitken stops only on two consecutive passes at or below the
  threshold (the predecessor held to its raw residual, the pass to its
  estimate), and IQN on its first;
* CPL-062: the guard's cost against Aitken's own exit (the claim of at
  most one pass fails: MADD-ANO-163);
* CPL-073: a positive eigenvalue above one is reported unconverged under
  every fixed-point acceleration, under-relaxation included, and IQN
  converges the same group;
* CPL-074: IQN on a single accelerated scalar (a rank-deficient
  least-squares) agrees with plain iteration;
* CPL-076: a Jacobi pass seeds flux-reading producers as the Gauss-Seidel
  pass does, so both step to the same fixed point.

Each is stated once here, over the memoryless pair of
:mod:`tests.core.coupling_domains`, and run in every domain: float64 and
a float32 member beside a float64 one under x64, bfloat16 and float16, a
``jax.vmap`` of the step, a multi-rate graph, a sub-cycled group, a
predictor (with IQN-IMVJ's Jacobian reuse where the acceleration is IQN),
a checkpoint restart and (slow) ``run_adaptive``.  The members are
memoryless, so every domain solves the same fixed point, and the oracle
is its closed form in float64 from the parameters as stored.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core import graph_manager as gm_mod
from maddening.core.graph_manager import _analysis_dtype, _fixed_point_while
from tests.core import coupling_domains as cd

SEQ = 5        # steps of a sequence run (predictor, restart)


def _sixteen(domain) -> bool:
    return jnp.dtype(domain.coarsest).itemsize == 2


def _eps(domain) -> float:
    return float(cd.finfo(domain.coarsest).eps)


def _sequenced(domain) -> bool:
    return domain.predictor or domain.restart


_GRAPHS: dict = {}


def _graph(label, kind, **kw):
    """The compiled pair for *label* and *kind*, built once per module."""
    if (label, kind) not in _GRAPHS:
        d = cd.DOMAINS[label]
        with cd.entered(d):
            _GRAPHS[(label, kind)] = cd.pair(d, **kw)
    return _GRAPHS[(label, kind)]


def _solves(domain, gm, scenarios, *, x0=None):
    """One list of solves per scenario: a sequence (predictor, restart: every
    step of one run) or the first entry of it (one solve, the batch domain
    all scenarios in one ``jax.vmap`` of the step)."""
    if _sequenced(domain):
        out = [cd.run_sequence(domain, gm, seq, x0=x0) for seq in scenarios]
    else:
        out = [[s] for s in cd.run(domain, gm, [seq[0] for seq in scenarios], x0=x0)]
    cd.assert_in_domain(domain, gm, [s for ss in out for s in ss])
    return out


def _relative_distance(s, x_star) -> float:
    """The group's L2 norm of ``x - x*``: each member's change over its
    own magnitude, the units ``tolerance`` is in."""
    total = 0.0
    for name, want in zip(("a", "b"), x_star):
        got = np.asarray(s.x(name), np.float64)
        want = np.broadcast_to(np.asarray(want, np.float64), got.shape)
        ref = max(float(np.max(np.abs(got))), float(np.max(np.abs(want))))
        if ref > 0.0:
            total += float(np.sum(((got - want) / ref) ** 2))
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# CPL-049: the estimate is the distance under any relaxation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label", cd.EVERY)
def test_the_estimate_is_the_distance_for_fixed_relaxation(label):
    """CPL-049: ``x_a <- g x_b + c``, ``x_b <- g x_a`` under
    ``acceleration="fixed"`` at relaxations either side of one: past the
    first loop pass the reported estimate over the true distance is within
    [0.9, 1.15], whatever omega -- a ratio near ``1/omega`` would be the
    series summing residuals rather than the steps the iterate takes.

    ``g = 0.95`` (rate 0.9025) at ``tolerance=1e-3``, as in float32.  A
    16-bit group cannot resolve that rate's residual ratio: its iterate
    stalls within a few ulps and the ratio is rejected (the raw residual
    test stands in, and ``ratio_usable`` says so -- CPL-054's stall).  So
    there ``g = 0.7`` at ``tolerance=0.2``, where the residual is many
    ulps, and the bracket widens by ``4 eps / tolerance``, the estimate's
    own rounding.
    """
    d = cd.DOMAINS[label]
    gain, tol = (0.7, 0.2) if _sixteen(d) else (0.95, 1e-3)
    slack = 4 * _eps(d) / tol if _sixteen(d) else 0.0
    checked = 0
    with cd.entered(d):
        for omega in (0.5, 1.3, 1.9):
            gm = _graph(label, f"relax-{omega}", g=(gain, gain), acceleration="fixed",
                        relaxation=omega, tolerance=tol, max_iterations=400)
            seq = cd.moving(gm, SEQ, g=(gain, gain))
            for s in _solves(d, gm, [seq])[0]:
                r = s.report
                if r["iterations"] <= 2:
                    continue            # the first loop pass is CPL-050's
                checked += 1
                dist = _relative_distance(s, cd.fixed_point(s))
                assert r["ratio_usable"] and r["converged"], (label, omega, r)
                ratio = r["error_estimate"] / dist
                assert 0.9 - slack <= ratio <= 1.15 + slack, (
                    f"{label} omega={omega}: estimate {r['error_estimate']:.4e} for a "
                    f"distance of {dist:.4e} (ratio {ratio:.4f})")
    assert checked, "no solve ran past its first loop pass"


# ---------------------------------------------------------------------------
# CPL-050: the first pass reads the relaxed rate
# ---------------------------------------------------------------------------
_RHO = 0.5


@pytest.mark.parametrize("label", cd.EVERY)
def test_an_under_relaxed_first_pass_estimate_is_not_below_the_distance(label):
    """CPL-050: ``x_a <- 0.5 x_b + c``, ``x_b <- x_a`` (one mode, rate 0.5),
    started off its fixed point by 0.2, 1.2 and 4 thresholds under
    ``acceleration="fixed"`` at ``omega < 1``: wherever it stops -- on the
    first loop pass or later -- it is converged, within the threshold of
    the fixed point, and its estimate is not below the distance, up to the
    estimate's own rounding (``floor amp + 2 omega floor amp**2``,
    ``floor`` eight ulps of the coarsest member per entry).  In the
    predictor and restart domains the forcing moves by a tenth of the
    threshold a step, so later steps start near their fixed point too.

    A 16-bit group asks for ``tolerance=0.2`` and relaxations it resolves
    (an ``omega = 0.1`` iteration crawls at 0.95 and stalls in a few
    ulps, CPL-054's stall); where its ratio is rejected the raw residual
    test stands in, which this claim does not describe, and at least one
    solve per relaxation must have a usable ratio.
    """
    d = cd.DOMAINS[label]
    sixteen = _sixteen(d)
    tol = 0.2 if sixteen else 1e-4
    omegas = (0.3, 0.6) if sixteen else (0.1, 0.3, 0.6)
    floor = 8 * _eps(d) * math.sqrt(2)
    first = 0
    with cd.entered(d):
        for omega in omegas:
            gm = _graph(label, f"under-{omega}", g=(_RHO, 1.0), acceleration="fixed",
                        relaxation=omega, tolerance=tol, max_iterations=60)
            usable = 0
            for delta in (0.2, 1.2, 4.0):
                seq = cd.moving(gm, SEQ, g=(_RHO, 1.0), c0=(1.0, 0.0), dc=(0.1 * tol, 0.0))
                x_star0 = cd.fixed_point(cd.Solve({}, {}, {}, seq[0]))
                x0 = tuple(np.asarray(v) * (1.0 + delta * tol) for v in x_star0)
                for s in _solves(d, gm, [seq], x0=x0)[0]:
                    r = s.report
                    if sixteen and not r["ratio_usable"]:
                        continue
                    usable += 1
                    first += r["iterations"] == 1
                    dist = _relative_distance(s, cd.fixed_point(s))
                    amp = r["amplification"]
                    allow = floor * amp + 2.0 * omega * floor * amp ** 2
                    assert r["converged"] and r["ratio_usable"], (label, omega, delta, r)
                    assert dist <= tol + allow, (label, omega, delta, dist / tol, r)
                    assert r["error_estimate"] >= dist - allow, (
                        f"{label} omega={omega}: estimate {r['error_estimate']:.4e} under "
                        f"the distance {dist:.4e}")
            assert usable, f"{label} omega={omega}: no solve had a usable ratio"
    assert first, "fixture premise: some solve stopped on its first loop pass"


# ---------------------------------------------------------------------------
# CPL-061 / CPL-062: the two-pass guard
# ---------------------------------------------------------------------------
#: Two plateaus above the threshold, each ending in one pass two decades
#: below it, then a genuine pair: a single-pass criterion stops on pass 3,
#: the guard on pass 7 (``test_coupling_convergence_reporting.py``).
_DIP_SCHEDULE = (1.0, 1.0, 1e-3, 1.0, 1.0, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3)


def _scripted_loop(acceleration, residuals, dtype, *, threshold=1e-2, max_iter=30,
                   batched=False):
    """``_fixed_point_while`` against a scripted residual sequence, in *dtype*.

    ``x[1]`` counts passes, so the residual the loop tests is dictated by
    the schedule (see the float32 original).  With *batched* the loop runs
    under ``jax.vmap`` over three copies of the schedule, scaled by 1, 2
    and 1/2 together with the threshold's place in it unchanged.
    """
    def one(scale):
        x0 = jnp.asarray([1.0, 0.0], dtype)
        schedule = jnp.asarray(residuals, dtype) * scale
        last = schedule.shape[0] - 1

        def step_pure(x):
            k = jnp.clip(x[1].astype(jnp.int32), 0, last)
            return jnp.stack([x[0] * jnp.asarray(0.5, dtype), x[1] + 1]), schedule[k]

        # The secant matrices in the dtype the graph seeds them in
        # (float32 for a 16-bit group: ``_analysis_dtype``).
        empty = jnp.zeros((1, 4), _analysis_dtype(jnp.dtype(dtype)))
        init = (empty, empty) if acceleration.startswith("iqn") else ()
        _x, n, res, _amp, _vw = _fixed_point_while(
            step_pure, x0, (), init, jnp.asarray(jnp.inf, dtype), threshold * scale,
            max_iter, acceleration, 1.0, 0, (0,))
        return n, res

    if not batched:
        n, res = one(1.0)
        return [(int(n), float(res))]
    scales = (1.0, 2.0, 0.5)
    ns, rs = jax.vmap(lambda s: one(s))(jnp.asarray(scales, dtype))
    return [(int(n), float(r) / s) for n, r, s in zip(ns, rs, scales)]


@pytest.mark.parametrize("acc,want", [("aitken", 7), ("none", 3), ("fixed", 3),
                                      ("iqn-ils", 3), ("iqn-imvj", 3)])
@pytest.mark.parametrize("label", cd.DTYPES + ("vmap",))
def test_aitken_needs_two_consecutive_passes_in_every_dtype(label, acc, want):
    """CPL-061 in the loop itself: on the dip schedule Aitken stops on pass 7
    (the first genuine pair) and every other acceleration on pass 3 (its
    first sub-threshold pass); an Aitken cap landing on a lone dip reports
    the measurement that springs back, not the dip.  In each member dtype,
    and under ``jax.vmap`` over three scaled copies."""
    d = cd.DOMAINS[label]
    with cd.entered(d):
        for dtype in dict.fromkeys(d.dtypes):
            for n, res in _scripted_loop(acc, _DIP_SCHEDULE, dtype, batched=d.vmap):
                assert n == want, (label, jnp.dtype(dtype).name, acc, n)
                assert res == pytest.approx(1e-3, rel=4 * float(cd.finfo(dtype).eps))
            if acc == "aitken":
                for n, res in _scripted_loop("aitken", (1.0, 1e-3) * 8, dtype, max_iter=9,
                                             batched=d.vmap):
                    assert n == 9 and res > 1e-2, (label, n, res)


#: Three independent modes, ``x_a <- g_a x_b + c``, ``x_b <- s x_a`` entry by
#: entry with ``s = +-1``: Aitken's single factor cannot fit all three, so
#: its estimate is not monotone near the exit.  The first is MADD-ANO-163's
#: reproducer: the guard holds Aitken two passes past its own exit.
_MODES = (
    ((-0.02, 0.92, 0.53), (-1.0, -1.0, 1.0), (1.82, 1.27, 1.02)),
    ((0.47, 0.68, -0.7), (1.0, 1.0, 1.0), (1.77, 1.14, 1.97)),
    ((0.72, -0.05, 0.8), (1.0, 1.0, -1.0), (0.61, 0.61, 1.8)),
    ((-0.32, -0.68, -0.1), (1.0, -1.0, -1.0), (1.11, 0.8, 0.64)),
)


def _exits(domain, label, acceleration, guard, monkeypatch, *, tol):
    """The iterations of every mode draw, on a graph traced with ``_TWO_PASS_EXIT = guard``.

    The list is read when the step is traced, so the graph is built,
    and its step (and the batch domain's vmap of it) traced, under the
    patch, which is then undone.
    """
    monkeypatch.setattr(gm_mod, "_TWO_PASS_EXIT", guard)
    try:
        gm = cd.pair(domain, n=3, g=(np.ones(3), np.ones(3)), c=(np.ones(3), np.zeros(3)),
                     acceleration=acceleration, max_iterations=60, tolerance=tol)
        scen = []
        for ga, gb, c in _MODES:
            if _sequenced(domain):
                scen.append(cd.moving(gm, 4, g=(ga, gb), c0=(c, np.zeros(3)),
                                      dc=(np.ones(3), np.zeros(3))))
            else:
                scen.append([cd.params_with(gm, g=(ga, gb), c=(c, np.zeros(3)))])
        runs = _solves(domain, gm, scen)
    finally:
        monkeypatch.undo()
    return [[s.report for s in run] for run in runs]


@pytest.mark.parametrize("label", cd.EVERY)
def test_the_guard_holds_aitken_past_its_own_exit(label, monkeypatch):
    """CPL-061 on a graph: the guard is live for Aitken.

    Aitken with its guard never stops before Aitken compiled without it
    (``_TWO_PASS_EXIT`` emptied), and on some draw stops later: the guard
    is doing something.  Every converged exit stays converged.
    """
    d = cd.DOMAINS[label]
    tol = 0.05 if _sixteen(d) else 1e-4
    with cd.entered(d):
        guarded = _exits(d, label, "aitken", ("aitken",), monkeypatch, tol=tol)
        bare = _exits(d, label, "aitken", (), monkeypatch, tol=tol)
    later = 0
    for g_run, b_run in zip(guarded, bare):
        for g, b in zip(g_run, b_run):
            if not b["converged"]:
                continue
            assert g["iterations"] >= b["iterations"], (label, g, b)
            assert g["converged"], (label, g)
            later += g["iterations"] > b["iterations"]
    assert later, f"{label}: the guard never held Aitken past its own exit"


@pytest.mark.parametrize("label", cd.EVERY)
def test_iqn_stops_on_its_first_sub_threshold_pass(label, monkeypatch):
    """CPL-061 on a graph: IQN is not on the guard's list.

    IQN-ILS stops where IQN-ILS with the guard forced on
    (``_TWO_PASS_EXIT`` extended) would not: never later, and on some draw
    sooner -- it stops on its first sub-threshold pass.
    """
    d = cd.DOMAINS[label]
    tol = 0.05 if _sixteen(d) else 1e-4
    with cd.entered(d):
        iqn = _exits(d, label, "iqn-ils", ("aitken",), monkeypatch, tol=tol)
        held = _exits(d, label, "iqn-ils", ("aitken", "iqn-ils"), monkeypatch, tol=tol)
    sooner = 0
    for i_run, h_run in zip(iqn, held):
        for i, h in zip(i_run, h_run):
            assert i["iterations"] <= h["iterations"], (label, i, h)
            sooner += i["iterations"] < h["iterations"]
    assert sooner, f"{label}: IQN behaved as if it were on the guard's list"


#: The domains MADD-ANO-163's draws reproduce in (bfloat16, float16 and the
#: batch members, at their own criteria, land within one pass on these draws).
_ANO_163 = ("f32", "f64", "mixed_dtype", "vmap", "multi_rate", "sub_cycled",
            "predictors_warm_starts", "checkpoint_restart")


@pytest.mark.parametrize("label", _ANO_163)
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "CPL-062: MADD-ANO-163: the two-pass guard can hold Aitken two passes past its own "
    "exit, not at most one: the predecessor is held to its raw residual and the pass to "
    "its estimate, which can rise back above the threshold; pending 0.5.0"))
def test_the_guard_adds_at_most_one_pass_to_aitkens_exit(label, monkeypatch):
    """CPL-062: "The guard adds at most one pass to Aitken's own exit".

    On the first mode draw, f32: Aitken without the guard stops at pass 9
    (estimate 6.9e-5 against a threshold of 1e-4); with it, pass 8's raw
    residual (9.2e-4) is above the threshold, pass 10's estimate rises back
    to 1.9e-4 over a raw residual of 4.6e-5, and the guard stops at pass 11
    -- two passes late, on raw residuals that fall every pass.
    """
    d = cd.DOMAINS[label]
    with cd.entered(d):
        guarded = _exits(d, label, "aitken", ("aitken",), monkeypatch, tol=1e-4)
        bare = _exits(d, label, "aitken", (), monkeypatch, tol=1e-4)
    extra = [g["iterations"] - b["iterations"] for g_run, b_run in zip(guarded, bare)
             for g, b in zip(g_run, b_run) if b["converged"]]
    assert extra and all(0 <= k <= 1 for k in extra), (label, extra)


# ---------------------------------------------------------------------------
# CPL-073: a positive eigenvalue above one
# ---------------------------------------------------------------------------
_STIFF = dict(g=(1.2, 1.2), c=(1.0, 0.5))


@pytest.mark.parametrize("label", cd.EVERY)
def test_a_contraction_above_one_is_reported_unconverged(label):
    """CPL-073: ``x_a <- 1.2 x_b + c_a``, ``x_b <- 1.2 x_a + c_b``: the
    Gauss-Seidel eigenvalue is +1.44 and the Jacobi pair +-1.2.  Every
    fixed-point acceleration -- none, Aitken, under-relaxation at 0.5 and
    0.8, and under-relaxed Jacobi (which maps -1.2 inside the circle and
    leaves +1.2 outside it) -- reports itself unconverged at its cap, or on
    a non-finite state, on every step.
    """
    d = cd.DOMAINS[label]
    tol = 0.05 if _sixteen(d) else 1e-5
    configs = [("none", {}), ("aitken", {}), ("fixed", dict(relaxation=0.5)),
               ("fixed", dict(relaxation=0.8)),
               ("fixed", dict(relaxation=0.5, iteration_mode="jacobi"))]
    with cd.entered(d):
        for i, (acc, extra) in enumerate(configs):
            gm = _graph(label, f"stiff-{i}", **_STIFF, acceleration=acc, tolerance=tol,
                        max_iterations=40, **extra)
            seq = cd.moving(gm, SEQ, g=_STIFF["g"], c0=_STIFF["c"])
            for s in _solves(d, gm, [seq])[0]:
                r = s.report
                assert r["converged"] is False, (label, acc, extra, r)
                assert r["iterations"] == 40 or not math.isfinite(r["residual"]), (
                    label, acc, extra, r)


@pytest.mark.parametrize("label", cd.EVERY)
def test_iqn_converges_a_contraction_above_one(label):
    """CPL-073: IQN-ILS and IQN-IMVJ converge the same group, on every step,
    onto its exact fixed point ``(I - M)^{-1} c``: within ``amp * tol`` with
    ``amp = 1 / |1 - 1.44|``, plus the dtype's rounding of ``x*``."""
    d = cd.DOMAINS[label]
    tol = 0.05 if _sixteen(d) else 1e-5
    with cd.entered(d):
        for acc in ("iqn-ils", "iqn-imvj"):
            gm = _graph(label, f"stiff-{acc}", **_STIFF, acceleration=acc, tolerance=tol,
                        max_iterations=40)
            seq = cd.moving(gm, SEQ, g=_STIFF["g"], c0=_STIFF["c"])
            for s in _solves(d, gm, [seq])[0]:
                r = s.report
                assert r["converged"] is True, (label, acc, r)
                dist = _relative_distance(s, cd.fixed_point(s))
                assert dist <= 4.0 * tol + 64 * _eps(d), (label, acc, dist, r)


# ---------------------------------------------------------------------------
# CPL-074: IQN on one accelerated scalar
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label", cd.EVERY)
def test_iqn_on_a_single_accelerated_scalar_agrees_with_plain_iteration(label):
    """CPL-074: ``accelerated_fields={"a": ("x",)}`` on a scalar pair makes the
    quasi-Newton least-squares one-dimensional (rank-deficient); IQN-ILS and
    IQN-IMVJ still converge every step onto plain iteration's answer -- no
    NaN swallowed, no other fixed point -- and take more than one pass, so
    the degenerate least-squares is entered.
    """
    d = cd.DOMAINS[label]
    tol = 0.05 if _sixteen(d) else 1e-5
    with cd.entered(d):
        plain = _graph(label, "one-dof-none", g=(0.8, 0.9), acceleration="none",
                       tolerance=tol, max_iterations=80)
        want = _solves(d, plain, [cd.moving(plain, SEQ, g=(0.8, 0.9))])[0]
        for acc in ("iqn-ils", "iqn-imvj"):
            gm = _graph(label, f"one-dof-{acc}", g=(0.8, 0.9), acceleration=acc,
                        accelerated_fields={"a": ("x",)}, tolerance=tol, max_iterations=80)
            got = _solves(d, gm, [cd.moving(gm, SEQ, g=(0.8, 0.9))])[0]
            for w, s in zip(want, got):
                assert s.report["converged"] is True, (label, acc, s.report)
                for n in ("a", "b"):
                    x, ref = (np.asarray(v.x(n), np.float64) for v in (s, w))
                    assert np.all(np.isfinite(x)), (label, acc)
                    # Both within amp * tol of x* (amp = 1 / (1 - 0.72)).
                    assert np.max(np.abs(x - ref)) <= 8.0 * tol * np.max(np.abs(ref)), (
                        label, acc, n, x, ref)
            assert max(s.report["iterations"] for s in got) > 1, (
                f"{label}/{acc}: the rank-deficient least-squares was never entered")


# ---------------------------------------------------------------------------
# CPL-076: a Jacobi pass seeds flux-reading producers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("label", cd.EVERY)
def test_flux_reading_producers_step_under_jacobi_to_the_gauss_seidel_fixed_point(label):
    """CPL-076: both members produce the flux ``q = 2 x`` and read the other's
    (``x_a <- g_a q_b + c_a``, ``x_b <- g_b q_a + c_b``): a Jacobi pass seeds
    each producer's flux from the previous iterate in two sweeps, as the
    Gauss-Seidel pass does, so under every iterator both modes converge to
    the one fixed point -- the closed form with the flux's factor two
    folded into the gains.
    """
    d = cd.DOMAINS[label]
    tol = 0.05 if _sixteen(d) else 1e-5
    g = (0.2, -0.3)
    with cd.entered(d):
        for mode in ("jacobi", "gauss-seidel"):
            for acc in ("none", "iqn-ils"):
                gm = _graph(label, f"flux-{mode}-{acc}", g=g, c=(1.0, 0.5), node=cd.Flux,
                            fields=("q", "q"), iteration_mode=mode, acceleration=acc,
                            tolerance=tol, max_iterations=60)
                seq = cd.moving(gm, SEQ, g=g, c0=(1.0, 0.5))
                for s in _solves(d, gm, [seq])[0]:
                    assert s.report["converged"] is True, (label, mode, acc, s.report)
                    ga, gb = s.gains()
                    ca, cb = s.forcing()
                    # q = 2 x: the gains double.
                    xa = (ca + 2 * ga * cb) / (1.0 - 4 * ga * gb)
                    xb = 2 * gb * xa + cb
                    dist = _relative_distance(s, (xa, xb))
                    assert dist <= 4.0 * tol + 64 * _eps(d), (label, mode, acc, dist)
