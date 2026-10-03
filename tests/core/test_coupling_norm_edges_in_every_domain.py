"""The norm's dtype-edge claims, held at each dtype's own edges and in every domain.

Four rows of ``docs/validation/coupling_claims.yaml`` are claims *about*
``finfo``: where the dead band starts (CPL-011), where a field becomes
unevaluable (CPL-043), how a field below the normal range is measured
(CPL-044), and what the spectral bound keeps of a dead-banded field and a
broken-down Krylov space (CPL-090).  Their own tests run float32.  Here
each claim is stated once and run in every domain of
:mod:`tests.core.coupling_domains` -- float64 under x64, a float32 member
beside a float64 one under x64, bfloat16 and float16, a ``jax.vmap`` of
the step, a multi-rate graph, a sub-cycled group, a predictor, a
checkpoint restart and (slow) ``run_adaptive`` -- **at the edges of the
dtype in hand**: its own ``tiny``, ``eps`` and ``maxexp``, not
float32's.

* CPL-011: ``atol`` is compared after rounding to each field's dtype, so
  a field at exactly ``atol`` leaves and one ulp above it stays; with
  the default ``0.0`` a field at exactly zero leaves and one at the
  dtype's smallest normal stays; a group whose every moving field is
  dead-banded reports ``residual=0.0, converged=True`` after one pass
  however far it is from its fixed point.
* CPL-043: a field whose reference is non-finite, or (under the L2
  norm's ``rtol=1``) above ``1 / tiny = 2**(maxexp - 2)``, reads ``inf``
  under every norm built on it; at exactly ``2**(maxexp - 2)`` it is
  still measured, and the mixed and interface norms measure it far
  above.  So a field within a factor of four of its dtype's overflow
  never converges under L2, in every dtype.
* CPL-044: the group scaled by ``2**-shift`` to where a one-ulp change of
  its fields is subnormal -- and further, to a few ``tiny`` -- takes the
  control's passes and reports its verdict and residual to the bit, its
  state the control's times ``2**-shift`` exactly.
* CPL-090: a repeated eigenvalue does not leave the residual outside the
  Krylov space the bound's resolvent was measured on, and a dead-banded
  field on the loop keeps its weight in the spectrum.

The members are memoryless (``x <- g u + c``), so every domain solves the
same fixed point and the oracle is the closed form in float64 from the
parameters as each member's dtype stores them.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import (
    _scaled_change,
    coupling_residual_interface,
    coupling_residual_l2,
    coupling_residual_mixed,
)
from maddening.core.edge import EdgeSpec
from tests.core import coupling_domains as cd

NORMS = ("l2", "mixed", "interface")


def _sixteen(domain) -> bool:
    return jnp.dtype(domain.coarsest).itemsize == 2


def _criterion(domain, norm, value=None) -> dict:
    """The group's criterion: ``value`` (default a resolvable one for the dtype)."""
    if value is None:
        value = 0.05 if _sixteen(domain) else 1e-5
    return dict(tolerance=value) if norm == "l2" else dict(rtol=value)


_GRAPHS: dict = {}


def _graph(label, kind, **kw):
    """The compiled pair for *label* and *kind*, built once per module."""
    if (label, kind) not in _GRAPHS:
        d = cd.DOMAINS[label]
        with cd.entered(d):
            _GRAPHS[(label, kind)] = cd.pair(d, **kw)
    return _GRAPHS[(label, kind)]


def _solves(domain, gm, scenarios, *, steps=4, x0=None):
    """One list of solves per scenario (a ``params`` pytree, or a sequence of them).

    The predictor and restart domains run each scenario as a sequence
    (the predictor reads the run's history, the restart saves mid-run)
    and check every step; the other domains take one solve per scenario,
    the batch domain all of them in one ``jax.vmap`` of the step.
    """
    seqs = [s if isinstance(s, list) else [s] * (steps if (domain.predictor or domain.restart)
                                                 else 1) for s in scenarios]
    if domain.predictor or domain.restart:
        out = [cd.run_sequence(domain, gm, seq, x0=x0) for seq in seqs]
    else:
        flat = cd.run(domain, gm, [seq[0] for seq in seqs], x0=x0)
        out = [[s] for s in flat]
    cd.assert_in_domain(domain, gm, [s for ss in out for s in ss])
    return out


# ---------------------------------------------------------------------------
# CPL-011: the dead band
# ---------------------------------------------------------------------------
def _below(value, dtype):
    """The largest number of *dtype* below *value*."""
    v = jnp.asarray(value, dtype)
    return jnp.nextafter(v, jnp.asarray(-jnp.inf, dtype))


@pytest.mark.parametrize("label", cd.DTYPES + ("vmap",))
def test_the_dead_band_is_drawn_at_each_fields_own_resolution(label):
    """CPL-011 at each dtype's edges, every norm.

    ``atol`` is compared after rounding to the field's dtype: a field whose
    magnitude is exactly ``atol`` leaves the norm, one whose magnitude is
    one ulp above it stays.  At the default ``atol=0.0`` a field at exactly
    zero leaves (no scale at all) and a field at the dtype's smallest
    normal, ``finfo.tiny``, stays and is measured.  In the batch domain the
    four fields are one ``jax.vmap`` of the helper.
    """
    d = cd.DOMAINS[label]
    with cd.entered(d):
        for dtype in dict.fromkeys(d.dtypes):
            info = cd.finfo(dtype)
            m = float(_below(1.0, dtype))             # a full mantissa below 1
            up = float(jnp.nextafter(jnp.asarray(m, dtype), jnp.asarray(2.0, dtype)))
            tiny = float(info.tiny)
            # (new, old, atol, active?, the L2 quotient of the first entry)
            cases = [
                ((m, 0.5 * m), (0.5 * m, 0.25 * m), m, False, 0.0),
                ((up, 0.5 * up), (0.5 * up, 0.25 * up), m, True, 0.5),
                ((0.0, 0.0), (0.0, 0.0), 0.0, False, 0.0),
                ((2 * tiny, tiny), (tiny, tiny), 0.0, True, 0.5),
            ]
            new = jnp.asarray([c[0] for c in cases], dtype)
            old = jnp.asarray([c[1] for c in cases], dtype)
            for rtol in (1.0, 1e-2):
                for i, (_n, _o, atol, active, quotient) in enumerate(cases):
                    if d.vmap:
                        scaled, act = jax.vmap(lambda a, b, atol=atol: _scaled_change(
                            a, b, atol, rtol))(new, old)
                        scaled, act = scaled[i], act[i]
                    else:
                        scaled, act = _scaled_change(new[i], old[i], atol, rtol)
                    where = f"{jnp.dtype(dtype).name} case {i} rtol={rtol}"
                    assert bool(act) is active, where
                    if active:
                        got = float(np.asarray(scaled, np.float64)[0])
                        assert got == pytest.approx(quotient / rtol, rel=4 * float(info.eps)), where
                    else:
                        assert not np.any(np.asarray(scaled, np.float64)), where
            # Every norm: the dead-banded field contributes nothing, the kept one all.
            keep = jnp.asarray([2.0, 1.0], dtype)
            keep_old = jnp.asarray([1.0, 1.0], dtype)
            band_new = jnp.asarray([m, 0.0], dtype)
            band_old = jnp.asarray([0.5 * m, 0.0], dtype)
            s_new = {"n": {"k": keep, "z": band_new}}
            s_old = {"n": {"k": keep_old, "z": band_old}}
            alone_new, alone_old = {"n": {"k": keep}}, {"n": {"k": keep_old}}
            edges = [EdgeSpec("n", "n", "k", "u"), EdgeSpec("n", "n", "z", "v")]
            assert float(coupling_residual_l2(s_new, s_old, ["n"], m)) == float(
                coupling_residual_l2(alone_new, alone_old, ["n"], m))
            assert float(coupling_residual_mixed(s_new, s_old, ["n"], m, 1e-2)) == float(
                coupling_residual_mixed(alone_new, alone_old, ["n"], m, 1e-2))
            assert float(coupling_residual_interface(s_new, s_old, edges, m, 1e-2)) == float(
                coupling_residual_interface(alone_new, alone_old, edges[:1], m, 1e-2))


@pytest.mark.parametrize("label", cd.EVERY)
def test_a_group_whose_moving_fields_are_all_dead_banded_stops_after_one_pass(label):
    """CPL-011: "reports residual=0.0, converged=True after one pass however far
    it is from its fixed point" -- under all three norms -- where the same group
    with the default dead band takes more passes; and the default excludes a
    field with no scale at all without putting a NaN or an inf in the norm.

    ``x_a <- 0.9 x_b + c``, ``x_b <- x_a``: ``x* = 10 c``, one pass from rest
    leaves ``x = c``, 90% from it.  ``atol = 100`` is above every magnitude
    the run reaches.
    """
    d = cd.DOMAINS[label]
    with cd.entered(d):
        for norm in NORMS:
            crit = _criterion(d, norm)
            banded = _graph(label, f"band-{norm}", g=(0.9, 1.0), convergence_norm=norm,
                            max_iterations=60, atol=100.0, **crit)
            plain = _graph(label, f"noband-{norm}", g=(0.9, 1.0), convergence_norm=norm,
                           max_iterations=60, **crit)
            seq = cd.moving(banded, 4, g=(0.9, 1.0))
            far = 0.0
            for s in _solves(d, banded, [seq if (d.predictor or d.restart) else seq[0]])[0]:
                r = s.report
                assert (r["iterations"], r["residual"], r["converged"]) == (1, 0.0, True), (
                    f"{label}/{norm}: every moving field dead-banded, yet {r}")
                xa_star, _ = cd.fixed_point(s)
                far = max(far, float(abs(s.x("a")[0] - xa_star[0]) / abs(xa_star[0])))
            assert far > 0.5, f"{label}/{norm}: fixture premise: one pass is far from x*"
            for s in _solves(d, plain, [seq if (d.predictor or d.restart) else seq[0]])[0]:
                assert s.report["iterations"] > 1, (
                    f"{label}/{norm}: the undeclared band let the group stop: {s.report}")
            # A member pinned at exactly zero has no scale and leaves the norm.
            zero = cd.params_with(plain, g=(0.0, 0.0), c=(1.0, 0.0))
            for s in _solves(d, plain, [zero])[0]:
                assert not np.any(np.asarray(s.x("b"), np.float64)), "fixture premise"
                r = s.report
                assert math.isfinite(r["residual"]) and r["converged"] is True, (label, norm, r)


# ---------------------------------------------------------------------------
# CPL-043: a field the criterion cannot evaluate fails it
# ---------------------------------------------------------------------------
def _overflow_edge(dtype) -> tuple[float, float]:
    """``(2**(maxexp - 2), the next number of dtype above it)``: ``1 / tiny``,
    where the L2 norm's scale stops having a normal reciprocal."""
    info = cd.finfo(dtype)
    edge = 2.0 ** (int(info.maxexp) - 2)
    assert edge * float(info.tiny) == 1.0, "1 / tiny is 2**(maxexp - 2)"
    above = float(jnp.nextafter(jnp.asarray(edge, dtype), jnp.asarray(jnp.inf, dtype)))
    return edge, above


@pytest.mark.parametrize("label", cd.DTYPES + ("vmap",))
def test_every_norm_reads_a_field_it_cannot_evaluate_as_inf_at_each_dtypes_edges(label):
    """CPL-043 at each dtype's ``maxexp``, every norm.

    A non-finite entry makes the field active and every entry ``inf``.  A
    constant field at ``2**(maxexp - 2)`` (``1 / tiny``) is still measured
    by the L2 norm (``rtol = 1``) and reads zero; one ulp above it reads
    ``inf`` there, and only there: the mixed and interface scales,
    ``rtol * max|v|``, stay normal.  ``2**(maxexp - 2)`` is a quarter of the
    dtype's overflow, in every dtype -- the claim's "within a factor of
    four of overflow".
    """
    d = cd.DOMAINS[label]
    with cd.entered(d):
        for dtype in dict.fromkeys(d.dtypes):
            edge, above = _overflow_edge(dtype)
            assert 3.9 < float(cd.finfo(dtype).max) / edge <= 4.0
            for bad in (jnp.nan, jnp.inf, -jnp.inf):
                new = jnp.asarray([bad, 1.0], dtype)
                old = jnp.asarray([1.0, 1.0], dtype)
                for rtol in (1.0, 1e-2):
                    if d.vmap:
                        scaled, act = jax.vmap(lambda a, b: _scaled_change(a, b, 0.0, rtol))(
                            jnp.stack([new, old]), jnp.stack([old, old]))
                        assert bool(act[0]) and bool(jnp.all(jnp.isposinf(scaled[0])))
                        assert bool(act[1]) and not bool(jnp.any(scaled[1]))
                    else:
                        scaled, act = _scaled_change(new, old, 0.0, rtol)
                        assert bool(act) and bool(jnp.all(jnp.isposinf(scaled))), (dtype, bad)
                s_new, s_old = {"n": {"x": new}}, {"n": {"x": old}}
                e = [EdgeSpec("n", "n", "x", "u")]
                assert math.isinf(float(coupling_residual_l2(s_new, s_old, ["n"])))
                assert math.isinf(float(coupling_residual_mixed(s_new, s_old, ["n"], 0.0, 1e-2)))
                assert math.isinf(float(coupling_residual_interface(s_new, s_old, e, 0.0, 1e-2)))
            for value, l2_inf in ((edge, False), (above, True)):
                flat = {"n": {"x": jnp.asarray([value, -0.5 * value], dtype)}}
                e = [EdgeSpec("n", "n", "x", "u")]
                l2 = float(coupling_residual_l2(flat, flat, ["n"]))
                assert math.isinf(l2) is l2_inf, (jnp.dtype(dtype).name, value, l2)
                if not l2_inf:
                    assert l2 == 0.0
                assert float(coupling_residual_mixed(flat, flat, ["n"], 0.0, 1e-2)) == 0.0
                assert float(coupling_residual_interface(flat, flat, e, 0.0, 1e-2)) == 0.0


@pytest.mark.parametrize("label", cd.EVERY)
def test_a_group_at_its_dtypes_overflow_edge_converges_only_where_it_is_measured(label):
    """CPL-043 on a graph: an exact fixed point at the edge, one ulp above it, NaN.

    ``x_a <- c``, ``x_b <- x_a``: from rest the first pass lands on the
    fixed point.  At ``c = 2**(maxexp - 2)`` every norm measures it and the
    group converges on that pass; one ulp above, the L2 norm reads ``inf``
    on every pass, the group runs to its cap and reports ``converged=False``
    -- a field within a factor of four of overflow never converges under
    L2 -- while the mixed and interface norms still converge it.  A NaN
    forcing reads ``inf`` and unconverged under every norm.
    """
    d = cd.DOMAINS[label]
    with cd.entered(d):
        edge, above = _overflow_edge(d.coarsest)
        for norm in NORMS:
            gm = _graph(label, f"flat-{norm}", g=(0.0, 1.0), convergence_norm=norm,
                        max_iterations=6, **_criterion(d, norm, 1e-3 if norm == "l2" else 1e-2))
            at = cd.params_with(gm, g=(0.0, 1.0), c=(edge, 0.0))
            past = cd.params_with(gm, g=(0.0, 1.0), c=(above, 0.0))
            nan = cd.params_with(gm, g=(0.5, 0.5), c=(float("nan"), 0.0))
            at_s, past_s, nan_s = _solves(d, gm, [at, past, nan])
            for s in at_s:
                assert s.report["converged"] is True and s.report["residual"] == 0.0, (norm, s.report)
            for s in past_s:
                if norm == "l2":
                    assert (s.report["converged"], s.report["iterations"]) == (False, 6), s.report
                    assert math.isinf(s.report["residual"]), s.report
                else:
                    assert s.report["converged"] is True, (norm, s.report)
            for s in nan_s:
                assert s.report["converged"] is False and math.isinf(s.report["residual"]), (
                    norm, s.report)


# ---------------------------------------------------------------------------
# CPL-044: below the normal range, the same verdict to the bit
# ---------------------------------------------------------------------------
def _shifts(dtype, x_star=20.0) -> tuple[int, ...]:
    """The power-of-two shifts the group is scaled by: the least one putting a
    one-ulp change of ``x*`` below the normal range, and the deepest one
    leaving the forcing four ``tiny`` above zero (so every value the node
    computes, ``0.9 x_b`` included, stays normal and scales exactly).  For
    bfloat16, whose ``eps`` is coarse, they are one shift."""
    info = cd.finfo(dtype)
    edge = math.ceil(math.log2(x_star * float(info.eps) / float(info.tiny))) + 1
    deep = -int(info.minexp) - 2
    assert x_star * 2.0 ** -edge * float(info.eps) < float(info.tiny), "subnormal ulp"
    assert 0.9 * 2.0 ** -max(edge, deep) >= float(info.tiny), "every value stays normal"
    return tuple(sorted({edge, max(edge, deep)}))


@pytest.mark.parametrize("label", cd.EVERY)
def test_the_verdict_does_not_change_below_the_dtypes_normal_range(label):
    """CPL-044: every norm, ``acceleration="none"``, the group scaled into the
    underflow of the dtype in hand.

    ``x_a <- 0.9 x_b + c``, ``x_b <- x_a`` (``rho = 0.9``, ``x* = 10 c``), run to
    convergence at ``c`` near one and at ``c * 2**-shift``, where ``shift``
    puts a one-ulp change of every field below ``finfo.tiny`` (the ``edge``)
    or the forcing at four ``tiny`` (``deep``).  A power of two scales every
    normal-range operation of the node exactly and the norm is a ratio, so
    the scaled group must take the control's passes and report its verdict
    and residual to the bit, with its state the control's times
    ``2**-shift``.  The predictor and restart domains compare every step of
    a run whose forcing moves.  Both solvers: every norm under ``"ift"``,
    the L2 norm under the deprecated ``"fori"``.
    """
    d = cd.DOMAINS[label]
    with cd.entered(d):
        shifts = _shifts(d.coarsest)
        # Every norm under the default solver, and the deprecated "fori" (which
        # reports only with diagnostics) under the L2 norm: "both solvers".
        for norm, solver in [(n, "ift") for n in NORMS] + [("l2", "fori")]:
            extra = dict(solver="fori", diagnostics=True) if solver == "fori" else {}
            gm = _graph(label, f"slow-{norm}-{solver}", g=(0.9, 1.0), convergence_norm=norm,
                        max_iterations=400, **_criterion(d, norm), **extra)
            if d.predictor or d.restart:
                # ``c_b`` stays exactly zero: a moving ``c_b`` near zero would be a
                # subnormal parameter at the deepest shift, which no node scales.
                scen = [cd.moving(gm, 5, g=(0.9, 1.0), dc=(1.0, 0.0), scale=2.0 ** -s)
                        for s in (0,) + shifts]
            else:
                scen = [cd.params_with(gm, g=(0.9, 1.0), c=(2.0 ** -s, 0.0))
                        for s in (0,) + shifts]
            control, *scaled = _solves(d, gm, scen)
            assert all(c.report["converged"] and c.report["iterations"] > 3 for c in control), (
                f"{label}/{norm}/{solver}: fixture premise: the control iterates to convergence: "
                f"{[c.report for c in control]}")
            for shift, run in zip(shifts, scaled):
                for k, (c, s) in enumerate(zip(control, run)):
                    want = (c.report["iterations"], c.report["converged"], c.report["residual"])
                    got = (s.report["iterations"], s.report["converged"], s.report["residual"])
                    assert got == want, f"{label}/{norm}/{solver} 2**-{shift} step {k}: {got} != {want}"
                    for n in ("a", "b"):
                        exact = np.asarray(c.x(n), np.float64) * 2.0 ** -shift
                        assert np.array_equal(np.asarray(s.x(n), np.float64), exact), (
                            f"{label}/{norm}/{solver} 2**-{shift} step {k}: {n}.x did not scale exactly")


# ---------------------------------------------------------------------------
# CPL-090: what the spectral bound keeps
# ---------------------------------------------------------------------------
_ALPHA, _BETA = 23.6584, 0.0024
_C4 = ((-1.434, 0.824), (-0.107, 0.129))


def _group_l2(pairs) -> float:
    total = 0.0
    for got, want in pairs:
        ref = max(float(np.max(np.abs(got))), float(np.max(np.abs(want))))
        if ref > 0.0:
            total += float(np.sum(((got - want) / ref) ** 2))
    return math.sqrt(total)


def _scaled_forcing(gm, factors):
    return [cd.params_with(gm, g=(_ALPHA, _BETA),
                           c=(np.asarray(_C4[0]) * f, np.asarray(_C4[1]) * f)) for f in factors]


@pytest.mark.parametrize("label", cd.EVERY)
def test_a_repeated_eigenvalue_does_not_leave_the_residual_outside_the_space(label):
    """CPL-090: Jacobi on ``x_a <- 23.66 x_b + c_a``, ``x_b <- 0.0024 x_a + c_b`` in R^2.

    ``A = B (x) I_2`` has a repeated eigenvalue, so one start vector's
    Krylov space breaks down at dimension two of four; continued from the
    residual, the bound is usable and covers the true distance to the
    exact fixed point (float64, from the gains and forcing as stored).
    """
    d = cd.DOMAINS[label]
    with cd.entered(d):
        gm = _graph(label, "repeated", n=2, g=(_ALPHA, _BETA), c=_C4, iteration_mode="jacobi",
                    max_iterations=4, tolerance=1e-9, diagnostics=True)
        scen = _scaled_forcing(gm, (1.0, 0.7, 1.3, 0.9))
        runs = _solves(d, gm, [scen if (d.predictor or d.restart) else scen[0]])[0]
        for s in runs:
            r = s.report
            ga, gb = s.gains()
            ca, cb = s.forcing()
            a = np.kron(np.array([[0.0, ga], [gb, 0.0]]), np.eye(2))
            x_star = np.linalg.solve(np.eye(4) - a, np.concatenate([ca, cb]))
            dist = _group_l2([(np.asarray(s.x("a"), np.float64), x_star[:2]),
                              (np.asarray(s.x("b"), np.float64), x_star[2:])])
            assert dist > 1e-3, f"{label}: fixture premise: far above the dtype's floor"
            assert r["rho_spectral"] == pytest.approx(math.sqrt(ga * gb), abs=4e-3), r
            assert r["spectral_usable"] is True, (label, r)
            assert r["spectral_error_bound"] >= dist, (
                f"{label}: bound {r['spectral_error_bound']:.4e} under the distance {dist:.4e}")


def _dead_band_loop(domain) -> tuple[float, float, float]:
    """``(G, h, atol)``: loop gain ``G h = 0.9`` with the relay field ``h x_a``
    inside the band.  float16 cannot hold ``1e9``, so its relay sits at
    ``2**-12`` of ``x_a`` (normal, ~2e-3) under a band of ``1e-2``."""
    if domain.coarsest == jnp.float16:
        return 0.9 * 2.0 ** 12, 2.0 ** -12, 1e-2
    return 0.9e9, 1e-9, 1e-7


@pytest.mark.parametrize("label", cd.EVERY)
def test_a_dead_banded_field_on_the_loop_stays_in_the_spectrum(label):
    """CPL-090: "a dead-banded field keeps a positive weight in the spectrum".

    ``x_a <- G x_b + c``, ``x_b <- h x_a`` with ``x_b`` inside the dead band:
    the spectrum still sees the loop (``rho_spectral`` the loop gain, to the
    group dtype's resolution), the bound is usable, and it covers the kept
    field's true distance to its fixed point.
    """
    d = cd.DOMAINS[label]
    big, h, atol = _dead_band_loop(d)
    with cd.entered(d):
        gm = _graph(label, "deadband", g=(big, h), max_iterations=3, tolerance=1e-6,
                    atol=atol, diagnostics=True)
        scen = [cd.params_with(gm, g=(big, h), c=(f, 0.0)) for f in (1.0, 0.7, 1.3, 0.9)]
        runs = _solves(d, gm, [scen if (d.predictor or d.restart) else scen[0]])[0]
        for s in runs:
            r = s.report
            ga, gb = s.gains()
            ca, _cb = s.forcing()
            assert float(np.max(np.abs(np.asarray(s.x("b"), np.float64)))) <= atol, (
                "fixture premise: the relay field is inside the dead band")
            x_star = float(ca[0]) / (1.0 - ga * gb)
            xa = float(np.asarray(s.x("a"), np.float64)[0])
            dist = abs(xa - x_star) / max(abs(xa), abs(x_star))
            resolution = 4 * float(cd.finfo(d.coarsest).eps) if _sixteen(d) else 1e-3
            assert r["rho_spectral"] == pytest.approx(ga * gb, abs=resolution), (label, r)
            assert r["spectral_usable"] is True, (label, r)
            assert r["spectral_error_bound"] >= dist, (
                f"{label}: bound {r['spectral_error_bound']:.4e} under the kept field's "
                f"distance {dist:.4e}")
