"""``fim``'s verdict is read from the singular values of ``J``, not from ``F``.

``F = J.T @ J`` sums ``m`` rounded products into each entry.  Where the
sensitivities are constant or piecewise constant (a steady state, a step)
every term of the sum is the same number, the roundings have one sign, and
the entries come back off by hundreds of ``eps`` (475 at ``m = 4000`` in
float32) where the rank cutoff ``max(n, sqrt(m)) * eps`` allowed for 63.
Read from that matrix's eigenvalues, two parameters that enter a residual
only through their sum -- exactly unidentifiable, for any data -- reported
``rank = 2 of 2`` and a finite Cramer-Rao bound, with no warning, at either
precision (MADD-ANO-273).  The tree's own test of the cutoff drew random
Jacobians only, and a random Jacobian does not align anything.

So the populations here are built, not only drawn:

* an **exactly degenerate pair**, ``r_i = (k1 + k2) * u_i - y_i``, with
  constant, step, ramp and noise inputs at 1e3 to 1e5 rows, in float32 and
  with float64 leaves under x64: rank 1 and an infinite bound for both
  parameters on **every** draw of a seeded population (the constant and
  step inputs are the ones the old route got wrong; the others are the
  control that says so);
* its **nearly degenerate neighbours**, where the verdict has to change,
  and change at the cutoff the documents state;
* a seeded fault in the Gram product, which the verdict must not see.

``fim_core`` is held to the same populations on the device.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import contextlib
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening import sysid
from maddening.sysid import _resolve_rank_rtol, fim, fim_core
from maddening.warnings import PrecisionLimitWarning

KINDS = ("const", "step", "ramp", "noise")

#: One seed for every population in this module.  A population test
#: asserts every draw, so the seed is part of the claim.
SEED = 0


@contextlib.contextmanager
def _x64():
    """``jax_enable_x64`` for the duration of the block (process-global,
    so it is put back)."""
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


def _draws(kind, m, count, seed=SEED):
    """``count`` draws of ``(k1, k2, u)``: the audit's sweep, to the draw.

    ``k1``, ``k2`` uniform in ``[0.5, 20]``; ``u`` a constant level in
    ``[0.5, 3]``, a step between two such levels at the middle of the
    record, a ramp from a level to 1.2 times it, or unit noise.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, m)
    for _ in range(count):
        k1, k2 = rng.uniform(0.5, 20.0), rng.uniform(0.5, 20.0)
        if kind == "const":
            u = np.full(m, rng.uniform(0.5, 3.0))
        elif kind == "step":
            u = np.where(t < 0.5, rng.uniform(0.5, 3.0), rng.uniform(0.5, 3.0))
        elif kind == "ramp":
            u = rng.uniform(0.5, 3.0) * (1.0 + 0.2 * t)
        else:
            u = rng.standard_normal(m)
        yield k1, k2, u


def _pair(k1, k2, u, dtype):
    """The exactly degenerate pair: only ``k1 + k2`` is in the residual."""
    u = jnp.asarray(u, dtype=dtype)
    params = {"k1": jnp.asarray(k1, dtype=dtype), "k2": jnp.asarray(k2, dtype=dtype)}
    return (lambda p: (p["k1"] + p["k2"]) * u), params


def _assert_rank_one_everywhere(kind, sizes, count, dtype):
    """Every draw: rank 1, both bounds infinite, nothing warned.

    Warnings are errors in this suite, so a ``PrecisionLimitWarning`` (the
    old route gave one on a further 7% to 17% of the draws, naming a re-run
    under x64 that reported the same full rank) fails the draw too.
    """
    for m in sizes:
        for i, (k1, k2, u) in enumerate(_draws(kind, m, count)):
            residual, params = _pair(k1, k2, u, dtype)
            report = fim(residual, params)
            where = f"{kind}, m={m}, draw {i} of seed {SEED}: k1={k1!r}, k2={k2!r}"
            assert report.rank == 1, f"rank {report.rank} of 2 ({where})"
            assert np.isinf(np.asarray(report.crb)).all(), (report.crb, where)
            assert report.eigvals.dtype == np.dtype(dtype)
            ratio = float(report.eigvals[0]) / float(report.eigvals[-1])
            assert ratio <= _resolve_rank_rtol(dtype, 2, None, n_residual=m), where


# ---------------------------------------------------------------------------
# The exactly degenerate pair: rank 1 on every draw
# ---------------------------------------------------------------------------

PER_PUSH_SIZES, PER_PUSH_DRAWS = (1000, 4000, 20000), 8
SWEEP_SIZES, SWEEP_DRAWS = (1000, 4000, 20000, 100000), 150


@pytest.mark.parametrize("kind", KINDS)
def test_a_pair_only_its_sum_identifies_is_rank_one_in_float32(kind):
    """Eight draws per size and input; the constant and step inputs fail
    on the route that read ``F``'s eigenvalues (jax 0.10.2 and 0.11.0)."""
    _assert_rank_one_everywhere(kind, PER_PUSH_SIZES, PER_PUSH_DRAWS, jnp.float32)


@pytest.mark.parametrize("kind", KINDS)
def test_a_pair_only_its_sum_identifies_is_rank_one_with_float64_leaves(kind):
    """The same population under x64 with float64 leaves: the remedy the
    float32 warning names has to give the same answer, and on the old
    route it did not (the rounding of the sum aligns at any precision)."""
    with _x64():
        _assert_rank_one_everywhere(kind, PER_PUSH_SIZES, PER_PUSH_DRAWS, jnp.float64)


# Per push: tests/core/test_sysid_fim_reads_the_jacobian.py::test_a_pair_only_its_sum_identifies_is_rank_one_in_float32
@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
def test_the_whole_float32_sweep_is_rank_one(kind):
    """The audit's sweep as it was run: 150 draws per cell, 1e3 to 1e5
    rows.  On the old route 0 / 10 / 28 / 12 of the constant-input draws
    and 5 / 17 / 23 / 7 of the step-input ones were full rank in silence."""
    _assert_rank_one_everywhere(kind, SWEEP_SIZES, SWEEP_DRAWS, jnp.float32)


# Per push: tests/core/test_sysid_fim_reads_the_jacobian.py::test_a_pair_only_its_sum_identifies_is_rank_one_with_float64_leaves
@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
def test_the_whole_float64_sweep_is_rank_one(kind):
    """The same sweep with float64 leaves under x64 (old route: up to 17
    of 150 full rank in silence)."""
    with _x64():
        _assert_rank_one_everywhere(kind, SWEEP_SIZES, SWEEP_DRAWS, jnp.float64)


def test_the_x64_rerun_agrees_with_float32_on_the_audited_record():
    """The one record the finding was reported on: 4000 samples of a
    steady state.  Float32 read rank 2 with ``crb = [0.01167, 0.00802]``;
    both precisions now read rank 1, and neither warns."""
    k1, k2, u0, m = 12.090304374694824, 14.586374282836914, 0.8686811327934265, 4000
    f32 = fim(*_pair(k1, k2, np.full(m, u0), jnp.float32))
    with _x64():
        f64 = fim(*_pair(k1, k2, np.full(m, u0), jnp.float64))
    for report in (f32, f64):
        assert report.rank == 1 and np.isinf(np.asarray(report.crb)).all()
        assert "rank 1 of 2 parameters (1 undetermined direction)" in str(report)


# ---------------------------------------------------------------------------
# fim_core: the same populations on the device
# ---------------------------------------------------------------------------


def _core_runner():
    """One jitted ``fim_core`` of the degenerate pair, ``u`` an argument
    (one compilation per record length)."""
    return jax.jit(lambda q, u: fim_core(lambda p: (p["k1"] + p["k2"]) * u, q))


def _assert_core_rank_one_everywhere(kind, sizes, count):
    run = _core_runner()
    for m in sizes:
        for i, (k1, k2, u) in enumerate(_draws(kind, m, count)):
            core = run({"k1": jnp.float32(k1), "k2": jnp.float32(k2)},
                       jnp.asarray(u, jnp.float32))
            where = f"{kind}, m={m}, draw {i} of seed {SEED}: k1={k1!r}, k2={k2!r}"
            assert int(core.rank) == 1, f"rank {int(core.rank)} of 2 ({where})"
            assert np.isinf(np.asarray(core.crb)).all(), where
            assert not bool(core.precision_limited), where
            assert bool(core.finite), where


@pytest.mark.parametrize("kind", ["const", "step"])
def test_the_jitted_core_reads_rank_one_for_the_degenerate_pair(kind):
    """Old route, constant input at 20,000 rows: rank 2 in 51 of 150
    draws, ``precision_limited`` in only 8 of them."""
    _assert_core_rank_one_everywhere(kind, (4000, 20000), 12)


# Per push: tests/core/test_sysid_fim_reads_the_jacobian.py::test_the_jitted_core_reads_rank_one_for_the_degenerate_pair
@pytest.mark.slow
@pytest.mark.parametrize("kind", KINDS)
def test_the_jitted_core_reads_rank_one_over_the_whole_sweep(kind):
    _assert_core_rank_one_everywhere(kind, SWEEP_SIZES, SWEEP_DRAWS)


def _t(r):
    """The right-hand side of ``_resolve_rank_rtol``'s sufficient
    condition, ``sqrt(r) / (1 + sqrt(r))``."""
    return np.sqrt(r) / (1.0 + np.sqrt(r))


def test_the_core_cutoff_covers_a_decomposition_at_the_jacobians_precision():
    """``fim_core`` computes the singular values at ``J``'s own precision,
    whose conventional bound is ``max(m, n) * eps`` of the largest
    (``numpy.linalg.matrix_rank``'s tolerance).  Its default cutoff meets
    the docstring's sufficient condition as written: what it leaves for the
    sensitivities after that bound, ``t(r_dev) - max(m, n) * eps``, is at
    least half of what ``fim``'s leaves, ``t(r_h) / 2``, at every size --
    and is that exactly wherever the device term is in force.  The term is
    in force only past about 16,300 rows in float32 at ``n = 2``; from 4.1
    million rows no cutoff under 1 meets the condition and nothing is
    resolved."""
    for dtype in (np.float32, np.float64):
        eps = float(np.finfo(dtype).eps)
        for n in (1, 2, 5, 25, 200):
            for m in np.unique(np.geomspace(1, 1e8, 97).astype(int)):
                m = int(m)
                host = _resolve_rank_rtol(dtype, n, None, n_residual=m)
                device = _resolve_rank_rtol(dtype, n, None, n_residual=m,
                                            decomposed_at=dtype)
                assert host == max(n, np.sqrt(m)) * eps
                assert device >= host, (dtype, n, m)
                if device == 1.0:
                    # Nothing under 1 meets it: half of fim's allowance
                    # plus the bound reaches t(1) = 1/2.
                    assert _t(host) / 2.0 + max(m, n) * eps >= 0.5, (dtype, n, m)
                    continue
                margin = _t(device) - max(m, n) * eps
                assert margin >= _t(host) / 2.0 * (1.0 - 1e-9), (dtype, n, m)
                if device > host:
                    assert margin == pytest.approx(_t(host) / 2.0, rel=1e-9)
    eps = float(np.finfo(np.float32).eps)
    for n, m in ((2, 3), (2, 4000), (5, 16000), (25, 16000)):
        assert _resolve_rank_rtol(np.float32, n, None, n_residual=m,
                                  decomposed_at=np.float32) == \
            _resolve_rank_rtol(np.float32, n, None, n_residual=m), (n, m)
    for n, m, times in ((2, 41286, 2.28), (2, 100000, 6.13), (7, 1000000, 170.0)):
        host = _resolve_rank_rtol(np.float32, n, None, n_residual=m)
        device = _resolve_rank_rtol(np.float32, n, None, n_residual=m,
                                    decomposed_at=np.float32)
        assert device / host == pytest.approx(times, rel=1e-2), (n, m)
    assert _resolve_rank_rtol(np.float32, 2, None, n_residual=4_200_000,
                              decomposed_at=np.float32) == 1.0
    # A stated cutoff is the caller's, on either route.
    assert _resolve_rank_rtol(np.float32, 2, 1e-9, n_residual=100000,
                              decomposed_at=np.float32) == 1e-9
    m = 100000
    core = fim_core(*_pair(3.0, 5.0, np.linspace(1.0, 2.0, m), jnp.float32))
    assert core.rank_rtol == _resolve_rank_rtol(np.float32, 2, None, n_residual=m,
                                                decomposed_at=np.float32)
    assert int(core.rank) == 1 and not bool(core.precision_limited)


def _two_columns(m, ratio, dtype):
    """``[c * 1, c * 1 + d * w]`` with ``w`` alternating ``+-1`` (exactly
    orthogonal to the constant column), ``d`` chosen in closed form so
    that the squared singular value ratio of the float64 matrix is
    ``ratio``; returned at ``dtype`` with its measured float64 ratio."""
    c = 1.3
    # Gram matrix m * [[a, a], [a, a + x]], a = c**2, x = d**2: the
    # eigenvalue ratio is (s - r) / (s + r), s = 2a + x, r = sqrt(x**2 +
    # 4a**2).  With k = r / s, r**2 = k**2 s**2 is a quadratic in x whose
    # smaller root, rationalised, is 2a (1 - k**2) / (k**2 + sqrt(2k**2 - 1)).
    a = c * c
    k = (1.0 - ratio) / (1.0 + ratio)
    one_minus_k2 = 4.0 * ratio / (1.0 + ratio) ** 2
    x = 2.0 * a * one_minus_k2 / (k * k + np.sqrt(2.0 * k * k - 1.0))
    w = np.where(np.arange(m) % 2 == 0, 1.0, -1.0)
    A = np.stack([np.full(m, c), c + np.sqrt(x) * w], axis=1).astype(dtype)
    s = np.linalg.svd(A.astype(np.float64), compute_uv=False)
    return A, float((s[-1] / s[0]) ** 2)


def _linear_core():
    """``fim_core`` of ``r = A @ theta`` under ``scale=None``, jitted with
    ``A`` as its argument: one compilation per shape."""
    return jax.jit(lambda A: fim_core(lambda p: A @ p["theta"],
                                      {"theta": jnp.ones(A.shape[1], A.dtype)},
                                      scale=None))


def _linear_fim(A, **kw):
    A = jnp.asarray(A)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = fim(lambda p: A @ p["theta"],
                     {"theta": jnp.ones(A.shape[1], A.dtype)}, scale=None, **kw)
    return report, [c for c in caught if issubclass(c.category, PrecisionLimitWarning)]


#: Where a second direction is put at 100,000 float32 rows, as
#: ``(multiple, of which cutoff)``: under, inside and above the band
#: between half of ``fim``'s cutoff and twice ``fim_core``'s, each at
#: least 10% from an edge.  ``fim_core``'s cutoff is 6.1 times ``fim``'s.
BAND_PLACEMENTS = (
    (0.4, "host"), (0.6, "host"), (1.5, "host"), (3.0, "host"),
    (0.9, "device"), (1.2, "device"), (1.8, "device"), (2.5, "device"),
)


def test_every_direction_between_the_two_cutoffs_is_flagged_by_the_core():
    """Past 16,300 float32 rows ``fim_core``'s default cutoff is the larger,
    and a direction between the two is one ``fim`` counts (or warns about)
    and ``fim_core`` does not.  Every such direction sets
    ``precision_limited``, with ``deciding_ratio`` naming it; outside the
    band the two routes agree on ``rank`` and nothing is flagged.  Before,
    a direction at 1.5 to 3 times ``fim``'s cutoff (and at a million rows,
    one at 7.6% of the strongest) came back unresolved and unflagged."""
    m, dtype = 100000, np.float32
    host = _resolve_rank_rtol(dtype, 2, None, n_residual=m)
    device = _resolve_rank_rtol(dtype, 2, None, n_residual=m, decomposed_at=dtype)
    assert device / host == pytest.approx(6.13, rel=1e-2)
    run = _linear_core()
    for multiple, which in BAND_PLACEMENTS:
        aim = multiple * (host if which == "host" else device)
        A, ratio = _two_columns(m, aim, dtype)
        assert ratio == pytest.approx(aim, rel=2e-2), (multiple, which)
        core = run(jnp.asarray(A))
        report, warned = _linear_fim(A)
        where = f"{multiple} x {which} cutoff (ratio {ratio:.4g})"
        assert core.rank_rtol == device
        assert report.rank == (2 if ratio > host else 1), where
        assert int(core.rank) == (2 if ratio > device else 1), where
        inside = host / 2.0 <= ratio <= 2.0 * device
        assert bool(core.precision_limited) == inside, where
        if inside:
            assert float(core.deciding_ratio) == pytest.approx(ratio, rel=1e-3), where
        else:
            assert float(core.deciding_ratio) == 0.0, where
            assert int(core.rank) == report.rank, where
        # fim's own band is around its own cutoff only.
        assert bool(warned) == (host / 2.0 <= ratio <= 2.0 * host), where


def test_a_direction_at_eight_percent_of_the_strongest_is_flagged_at_a_million_rows():
    """The audit's widest case: at 1e6 float32 rows ``fim_core``'s cutoff is
    0.020, so a direction whose singular value is 7.6% of the strongest
    (eigenvalue ratio 5.7e-3) is unresolved there -- and flagged, where
    ``fim`` resolves it with no warning."""
    m, dtype = 1000000, np.float32
    A, ratio = _two_columns(m, 5.7e-3, dtype)
    core = _linear_core()(jnp.asarray(A))
    report, warned = _linear_fim(A)
    assert core.rank_rtol == pytest.approx(0.0203, rel=1e-2)
    assert report.rank == 2 and not warned
    assert int(core.rank) == 1
    assert bool(core.precision_limited)
    assert float(core.deciding_ratio) == pytest.approx(ratio, rel=1e-3)


def _orthogonal_pair(m, small, dtype):
    """``[1, small * w]``, ``w`` alternating ``+-1``: exactly orthogonal
    columns, so the singular values are the column norms and the squared
    ratio is ``small**2`` as ``dtype`` stores ``small`` -- accurately
    computed by any backward-stable decomposition."""
    w = np.where(np.arange(m) % 2 == 0, 1.0, -1.0)
    A = np.stack([np.ones(m), small * w], axis=1).astype(dtype)
    return A, float(np.float64(A[0, 1]) ** 2)


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
def test_the_snap_never_reaches_above_the_callers_cutoff(x64):
    """A singular value at 1e-13 of the largest (float32; 1e-14 with
    float64 leaves) is under the float64 decomposition's bound at 1,000
    rows (4.4e-13), so the default reports its eigenvalue as ``0.0``.
    ``rank_rtol=0.0`` is documented to resolve every positive eigenvalue,
    and a cutoff under that eigenvalue resolves it too: both read rank 2
    with the exact eigenvalue, as ``fim_core`` does.  Before, the snap
    zeroed it under any cutoff (rank 1, ``cond`` inf)."""
    m = 1000
    with (_x64() if x64 else contextlib.nullcontext()):
        dtype = np.float64 if x64 else np.float32
        A, exact = _orthogonal_pair(m, 1e-14 if x64 else 1e-13, dtype)
        default, _ = _linear_fim(A)
        assert default.rank == 1 and float(default.eigvals[0]) == 0.0
        assert default.cond == float("inf")
        for rtol in (0.0, 1e-30, exact / 10.0):
            report, warned = _linear_fim(A, rank_rtol=rtol)
            got = float(report.eigvals[0]) / float(report.eigvals[-1])
            assert report.rank == 2, rtol
            assert got == pytest.approx(exact, rel=1e-5), rtol
            assert report.cond == pytest.approx(1.0 / exact, rel=1e-5), rtol
            assert np.isfinite(np.asarray(report.crb)).all(), rtol
            assert not warned
            core = fim_core(lambda p: jnp.asarray(A) @ p["theta"],
                            {"theta": jnp.ones(2, dtype)}, scale=None, rank_rtol=rtol)
            assert int(core.rank) == 2, rtol
        # A cutoff above the eigenvalue: unresolved, and now snapped too.
        report, _ = _linear_fim(A, rank_rtol=exact * 10.0)
        assert report.rank == 1 and float(report.eigvals[0]) == 0.0


def test_the_snap_reaches_the_decompositions_bound_and_no_further():
    """The snap pinned from both sides at 1,000 float32 rows, under the
    default cutoff (bound ``max(m, n) * n * eps64 = 4.4e-13``): a real
    singular value at a tenth of the bound reads ``0.0``; one at 22 times
    the bound is reported as computed, ``cond`` finite at ``1e22``.  Both
    are far under the cutoff, so ``rank`` is 1 either way: the snap moves
    ``eigvals`` and ``cond`` only."""
    m, dtype = 1000, np.float32
    bound = m * 2 * float(np.finfo(np.float64).eps)
    assert bound == pytest.approx(4.44e-13, rel=1e-2)
    under, _ = _orthogonal_pair(m, bound / 10.0, dtype)
    report, _ = _linear_fim(under)
    assert float(report.eigvals[0]) == 0.0 and report.cond == float("inf")
    assert report.rank == 1
    over, exact = _orthogonal_pair(m, 1e-11, dtype)
    report, _ = _linear_fim(over)
    got = float(report.eigvals[0]) / float(report.eigvals[-1])
    assert got == pytest.approx(exact, rel=1e-5)
    assert report.cond == pytest.approx(1.0 / exact, rel=1e-5)
    assert report.rank == 1 and np.isinf(np.asarray(report.crb)[1])


# ---------------------------------------------------------------------------
# The nearly degenerate neighbours: the verdict changes at the stated cutoff
# ---------------------------------------------------------------------------


def _neighbour(k1, k2, u, w, tau, dtype):
    """``r = (k1 + k2) * u + k2 * tau * w``: at ``tau = 0`` the degenerate
    pair, and a second direction of size ``tau`` beside it."""
    u, w = jnp.asarray(u, dtype=dtype), jnp.asarray(w, dtype=dtype)
    tau = jnp.asarray(tau, dtype=dtype)
    params = {"k1": jnp.asarray(k1, dtype=dtype), "k2": jnp.asarray(k2, dtype=dtype)}
    return (lambda p: (p["k1"] + p["k2"]) * u + p["k2"] * tau * w), params


def _reference_ratio(residual, params):
    """``lambda_min / lambda_max`` of the Jacobian ``fim`` decomposes --
    the working-precision ``J`` with its relative column scale -- from its
    singular values in float64."""
    J = jax.jacfwd(lambda q: residual({"k1": q[0], "k2": q[1]}))(
        jnp.stack([params["k1"], params["k2"]]))
    J = np.asarray(J * jnp.stack([params["k1"], params["k2"]])[None, :], np.float64)
    s = np.linalg.svd(J, compute_uv=False)
    return float((s[-1] / s[0]) ** 2)


def _tau_for(target, k1, k2, u, w):
    """The ``tau`` that puts the squared singular value ratio of
    ``[k1 * u, k2 * (u + tau * w)]`` at ``target``, by bisection in log
    space on the float64 matrix (the ratio grows as ``tau ** 2``).  Where
    the working precision then lands it is measured, not assumed
    (:func:`_reference_ratio`)."""
    def ratio(tau):
        s = np.linalg.svd(np.stack([k1 * u, k2 * (u + tau * w)], axis=1),
                          compute_uv=False)
        return (s[-1] / s[0]) ** 2

    lo, hi = 1e-14, 1.0
    for _ in range(100):
        mid = float(np.sqrt(lo * hi))
        lo, hi = (mid, hi) if ratio(mid) < target else (lo, mid)
    return float(np.sqrt(lo * hi))


#: Where a neighbour is put, as a multiple of the cutoff: two outside the
#: warning's factor-2 band, two inside it on either side of the cutoff.
PLACEMENTS = (0.25, 0.9, 1.1, 4.0)


def _assert_the_verdict_changes_at_the_cutoff(kind, dtype, m=4000, count=3):
    cutoff = _resolve_rank_rtol(dtype, 2, None, n_residual=m)
    assert cutoff == max(2.0, np.sqrt(m)) * float(np.finfo(dtype).eps)
    rng = np.random.default_rng(SEED + 1)
    for i, (k1, k2, u) in enumerate(_draws(kind, m, count)):
        w = rng.choice([-1.0, 1.0], size=m)
        for placement in PLACEMENTS:
            tau = _tau_for(placement * cutoff, k1, k2, u, w)
            residual, params = _neighbour(k1, k2, u, w, tau, dtype)
            ratio = _reference_ratio(residual, params)
            # The bisection and the float32 storage of ``tau`` leave the
            # ratio within a percent of where it was aimed.
            assert abs(ratio / (placement * cutoff) - 1.0) < 1e-2, (ratio, placement)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                report = fim(residual, params)
            where = f"{kind}, draw {i}, {placement} x cutoff (ratio {ratio:.6g})"
            # The verdict is the reference's, exactly: resolved if and
            # only if the squared singular value ratio is above the
            # cutoff.  (The reported eigenvalues are rounded once to the
            # report's precision, 1e-7 relative, against placements 10%
            # off the cutoff.)
            assert report.rank == (2 if ratio > cutoff else 1), where
            assert report.rank == (2 if placement > 1.0 else 1), where
            assert np.isfinite(np.asarray(report.crb)).all() == (report.rank == 2), where
            got = float(report.eigvals[0]) / float(report.eigvals[-1])
            assert abs(got / ratio - 1.0) < 1e-5, (got, ratio, where)
            warned = [c for c in caught if issubclass(c.category, PrecisionLimitWarning)]
            assert len(caught) == len(warned), [str(c.message) for c in caught]
            assert bool(warned) == (0.5 <= placement <= 2.0), where


@pytest.mark.parametrize("kind", KINDS)
def test_the_float32_verdict_changes_at_the_stated_cutoff(kind):
    _assert_the_verdict_changes_at_the_cutoff(kind, jnp.float32)


@pytest.mark.parametrize("kind", KINDS)
def test_the_float64_verdict_changes_at_the_stated_cutoff(kind):
    with _x64():
        _assert_the_verdict_changes_at_the_cutoff(kind, jnp.float64)


@pytest.mark.parametrize("kind", ["const", "noise"])
def test_the_core_verdict_changes_across_the_stated_cutoff(kind):
    """Outside the band the device verdict is the host's and is not
    flagged; inside it ``precision_limited`` says the two precisions are
    entitled to differ."""
    m, dtype = 4000, jnp.float32
    cutoff = _resolve_rank_rtol(dtype, 2, None, n_residual=m)
    rng = np.random.default_rng(SEED + 1)
    for k1, k2, u in _draws(kind, m, 3):
        w = rng.choice([-1.0, 1.0], size=m)
        for placement in PLACEMENTS:
            tau = _tau_for(placement * cutoff, k1, k2, u, w)
            core = fim_core(*_neighbour(k1, k2, u, w, tau, dtype))
            assert core.rank_rtol == cutoff
            if 0.5 <= placement <= 2.0:
                assert bool(core.precision_limited), (kind, placement)
            else:
                assert not bool(core.precision_limited), (kind, placement)
                assert int(core.rank) == (2 if placement > 1.0 else 1), (kind, placement)


# ---------------------------------------------------------------------------
# The verdict does not read the Gram product
# ---------------------------------------------------------------------------


def _a_gram_product_off_by_thousands_of_eps(monkeypatch):
    """Seed the fault itself, whatever the backend's summation does: the
    second diagonal entry of ``F`` up by 2048 ``eps`` relative, which
    moves ``F``'s smallest eigenvalue as far past the cutoff (six times)
    as the aligned rounding of the audited record did."""
    original = sysid._fim_gram
    seen = []

    def faulty(J):
        F, finite, nonzero = original(J)
        F = F.at[1, 1].multiply(1.0 + 2048.0 * float(np.finfo(F.dtype).eps))
        seen.append(F)
        return F, finite, nonzero

    monkeypatch.setattr(sysid, "_fim_gram", faulty)
    return seen


def test_the_report_carries_the_gram_product_and_decides_nothing_from_it(monkeypatch):
    k1, k2, m = 3.0, 5.0, 4000
    residual, params = _pair(k1, k2, np.full(m, 1.5), jnp.float32)
    clean = fim(residual, params)
    seen = _a_gram_product_off_by_thousands_of_eps(monkeypatch)
    faulted = fim(residual, params)
    assert len(seen) == 1, "the patch did not take effect"
    # The fault is one the old route could not survive: read from this
    # matrix, the pair is full rank by a wide margin.
    ev = np.linalg.eigvalsh(np.asarray(seen[0], np.float64))
    assert ev[0] / ev[-1] > 4 * _resolve_rank_rtol(np.float32, 2, None, n_residual=m)
    assert np.array_equal(np.asarray(faulted.fim), np.asarray(seen[0]))
    assert not np.array_equal(np.asarray(faulted.fim), np.asarray(clean.fim))
    assert faulted.rank == clean.rank == 1
    for field in ("eigvals", "eigvecs", "crb"):
        assert np.array_equal(np.asarray(getattr(faulted, field)),
                              np.asarray(getattr(clean, field))), field
    assert faulted.cond == clean.cond

    core = fim_core(residual, params)
    assert len(seen) == 2, "the patch did not reach fim_core"
    assert int(core.rank) == 1 and np.isinf(np.asarray(core.crb)).all()
    assert not bool(core.precision_limited)


def test_the_reported_eigenpairs_are_those_of_the_exact_gram_matrix():
    """A well-conditioned problem: the report's eigenpairs satisfy
    ``F v = lambda v`` against ``JᵀJ`` formed in float64 from the float32
    ``J``, to the rounding of one float32 storage; and ``rank``, ``crb``
    and ``cond`` follow from them by the documented rules."""
    rng = np.random.default_rng(SEED)
    A = jnp.asarray(rng.standard_normal((60, 3)), jnp.float32)
    params = {"p": jnp.asarray([1.5, -0.75, 2.0], jnp.float32)}
    report = fim(lambda p: A @ p["p"], params, scale=None)
    exact = np.asarray(A, np.float64).T @ np.asarray(A, np.float64)
    lam = np.asarray(report.eigvals, np.float64)
    V = np.asarray(report.eigvecs, np.float64)
    eps = float(np.finfo(np.float32).eps)
    assert np.all(np.diff(lam) >= 0.0)
    assert np.abs(exact @ V - V * lam).max() <= 8 * eps * lam[-1]
    assert np.abs(V.T @ V - np.eye(3)).max() <= 8 * eps
    assert report.rank == 3
    assert report.cond == float(report.eigvals[-1]) / float(report.eigvals[0])
    np.testing.assert_allclose(np.asarray(report.crb, np.float64),
                               np.diag(np.linalg.inv(exact)), rtol=64 * eps)
    # The stored matrix is the float32 Gram product, to its own rounding.
    np.testing.assert_allclose(np.asarray(report.fim, np.float64), exact,
                               rtol=0, atol=60 * eps * np.abs(exact).max())


def test_fewer_rows_than_parameters_leaves_the_missing_directions_at_zero():
    """A ``J`` with ``m < n`` rows has ``n - m`` directions no row reads:
    their eigenvalue is ``0`` and every parameter with support there has
    no bound."""
    A = jnp.asarray([[1.0, 2.0, 0.0]], jnp.float32)
    params = {"p": jnp.asarray([1.0, 1.0, 1.0], jnp.float32)}
    report = fim(lambda p: A @ p["p"], params, scale=None)
    assert report.rank == 1
    np.testing.assert_allclose(np.asarray(report.eigvals), [0.0, 0.0, 5.0], atol=1e-12)
    assert np.isinf(np.asarray(report.crb)).tolist() == [True, True, True]
    core = fim_core(lambda p: A @ p["p"], params, scale=None)
    assert int(core.rank) == 1 and np.isinf(np.asarray(core.crb)).all()
    np.testing.assert_allclose(np.asarray(core.eigvals), [0.0, 0.0, 5.0], atol=1e-5)


# ---------------------------------------------------------------------------
# The fitters' rank guard reads the same cutoff, from the Jacobian
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["const", "step"])
def test_fit_lms_rank_guard_reads_rank_one_for_the_degenerate_pair(kind):
    """``fit_lm``'s identifiability guard (``_curvature_rank``) decides by
    the cutoff ``fim`` uses, and always took it against the singular
    values of ``J`` in float64: the aligned rounding of a Gram product
    never reached it.  Held here on the population ``fim`` got wrong, so
    the two cannot drift apart again: the guard's docstring promises that
    it holds a direction only if ``fim`` would call it unresolved."""
    for m in PER_PUSH_SIZES:
        for k1, k2, u in _draws(kind, m, PER_PUSH_DRAWS):
            J = np.stack([u, u], axis=1).astype(np.float32)
            scale = np.array([1.0 / k1, 1.0 / k2])      # the guard's relative coordinates
            assert sysid._curvature_rank(J, scale) == 1, (kind, m, k1, k2)
            residual, params = _pair(k1, k2, u, jnp.float32)
            assert fim(residual, params).rank == 1


# ---------------------------------------------------------------------------
# The sign of the relative scale
# ---------------------------------------------------------------------------


def test_a_negative_parameter_keeps_its_sign_in_the_relative_scale():
    """``scale="relative"`` multiplies column ``j`` by ``theta_j``, sign
    and all: against an exact ``F`` (small integers, exact in float32), a
    scale by ``|theta_j|`` flips every off-diagonal entry that pairs a
    negative parameter with a positive one, and with it the sign of that
    parameter's component in every eigenvector."""
    A = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    theta = np.array([2.0, -3.0, 4.0])
    params = {"p": jnp.asarray(theta, jnp.float32)}
    A32 = jnp.asarray(A, jnp.float32)
    report = fim(lambda p: A32 @ p["p"], params)
    scaled = A * theta[None, :]
    exact = scaled.T @ scaled
    assert np.array_equal(np.asarray(report.fim, np.float64), exact)
    assert exact[0, 1] < 0.0 and exact[1, 2] < 0.0 and exact[0, 2] > 0.0
    lam, V = np.linalg.eigh(exact)
    got = np.asarray(report.eigvecs, np.float64)
    for j in range(3):
        # An eigenvector is defined up to one overall sign, not one per
        # component: fix it on the largest component of the reference.
        k = int(np.argmax(np.abs(V[:, j])))
        sign = np.sign(V[k, j]) * np.sign(got[k, j])
        np.testing.assert_allclose(sign * got[:, j], V[:, j], atol=1e-5)
    np.testing.assert_allclose(np.asarray(report.eigvals, np.float64), lam, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(report.crb, np.float64),
                               np.diag(np.linalg.inv(exact)), rtol=1e-4)
    core = fim_core(lambda p: A32 @ p["p"], params)
    assert np.array_equal(np.asarray(core.fim, np.float64), exact)
