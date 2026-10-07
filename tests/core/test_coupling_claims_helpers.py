"""The coupling claims inventory's helper-level rows, each at the edge of its domain.

``docs/validation/coupling_claims.yaml`` lists every documented claim about
coupling with a test that can fail.  The rows below had none, or none at the
edge of what the documentation states; each test here names the row it
pins.  They call the coupling helpers directly -- the norms, the rate
estimate, the accelerators, the bound helpers, the adjoint's linear solve --
so each runs in milliseconds and needs no graph.
"""

from __future__ import annotations

import math
import os
import re

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import (
    PRECISION_FLOOR_ULPS,
    aitken_relaxation,
    arnoldi_spectral_radius,
    coupling_residual_interface,
    coupling_residual_l2,
    coupling_residual_mixed,
    error_amplification,
    estimated_error,
    ift_gradient_error_bound,
    residual_precision_floor,
)
from maddening.core.edge import EdgeSpec
from maddening.core.coupling._bounds import (
    _gradient_error_bound_at,
    _interface_spectral_rate_at,
    _spectral_rate_at,
)
from maddening.core.coupling._ift import (
    _ift_linear_solve,
    _ift_solve,
)

F32 = jnp.float32
EPS32 = float(np.finfo(np.float32).eps)


def _state(**fields):
    return {"n": {k: jnp.asarray(v, F32) for k, v in fields.items()}}


# ---------------------------------------------------------------------------
# CPL-008: tolerance is relative; absolute only at unit magnitude
# ---------------------------------------------------------------------------


def test_the_l2_tolerance_is_absolute_only_at_unit_magnitude():
    """CPL-008: the L2 norm is ``||dx|| / max|v|`` per field, so ``||dx||`` only at ``max|v| = 1``."""
    unit = coupling_residual_l2(_state(x=[1.0, 0.5]), _state(x=[0.9, 0.5]), ["n"])
    assert float(unit) == pytest.approx(0.1, rel=1e-6)            # = ||dx||
    big = coupling_residual_l2(_state(x=[8.0, 4.0]), _state(x=[7.2, 4.0]), ["n"])
    assert float(big) == pytest.approx(0.8 / 8.0, rel=1e-6)       # = ||dx|| / 8, not 0.8
    # The reference is the larger of the two iterates' max|v|.
    swapped = coupling_residual_l2(_state(x=[7.2, 4.0]), _state(x=[8.0, 4.0]), ["n"])
    assert float(swapped) == float(big)


def test_the_l2_tolerance_is_one_threshold_for_the_whole_group():
    """CPL-008: the root-sum-square over every entry of every field, not a per-field threshold.

    The docstring once said "an absolute threshold of tolerance * max|v| per
    field"; two fields each inside that fail the group's test, and a field
    of ``n`` entries each moving by ``d`` of its magnitude reads ``d sqrt(n)``.
    """
    tol = 1e-3
    one = jnp.ones((1,), F32)
    # Two fields, each moving by 0.8 tol of its own magnitude.
    two = float(coupling_residual_l2(
        {"A": {"x": one * (1 + 0.8 * tol)}, "B": {"y": one * 50.0 * (1 + 0.8 * tol)}},
        {"A": {"x": one}, "B": {"y": one * 50.0}}, ["A", "B"]))
    each = 0.8 * tol / (1 + 0.8 * tol)
    assert two == pytest.approx(math.sqrt(2.0) * each, rel=1e-4) and two > tol
    # One 100-entry field and ten one-entry members, every entry moving by d.
    d = 0.5 * tol
    many = float(coupling_residual_l2({"n": {"x": jnp.full((100,), 1 + d, F32)}},
                                      {"n": {"x": jnp.ones((100,), F32)}}, ["n"]))
    assert many == pytest.approx(10.0 * d / (1 + d), rel=1e-4) and many > tol
    members = [f"N{i}" for i in range(10)]
    ten = float(coupling_residual_l2({m: {"x": one * (1 + d)} for m in members},
                                     {m: {"x": one} for m in members}, members))
    assert ten == pytest.approx(math.sqrt(10.0) * d / (1 + d), rel=1e-4) and ten > tol


# ---------------------------------------------------------------------------
# CPL-013: what the diagnostics cost, counted
# ---------------------------------------------------------------------------


def test_arnoldi_costs_n_steps_plus_one_jacobian_vector_products():
    """CPL-013: ``n_steps + 1`` products, with or without ``v_extra``; the spectrum costs nine.

    The one more is the product the estimate checks itself against.
    """
    A = jnp.asarray(np.diag([0.9, -0.3, 0.1]), F32)
    calls = []

    def matvec(v):
        calls.append(1)
        return A @ v

    with jax.disable_jit():
        arnoldi_spectral_radius(matvec, jnp.ones(3), n_steps=8)
        assert len(calls) == 9
        calls.clear()
        arnoldi_spectral_radius(matvec, jnp.ones(3), n_steps=8,
                                v_extra=jnp.asarray([1.0, 0.0, 0.0]))
        assert len(calls) == 9
        calls.clear()
        arnoldi_spectral_radius(matvec, jnp.ones(3), n_steps=5)
        assert len(calls) == 6

    # The group-level spectrum: nine Jacobian-vector products of the
    # one-pass map, plus one evaluation of the map itself for the residual.
    counts = _count_map_evaluations(
        lambda step, x, consts, w: _spectral_rate_at(step, x, consts, w))
    assert counts == {"jvp": 9, "primal": 1}, counts



def test_the_spectrum_on_a_transformed_reading_costs_nine_more_products():
    """CPL-013: "18 under the interface norm with a transform on an internal edge".

    The report of such a group takes a second spectrum, on the norm's
    reading (``_interface_spectral_rate_at``): nine more Jacobian-vector
    products of the one-pass map and one evaluation of it, beside the
    nine of the state's own spectrum.  The reading's products are of the
    transforms alone.
    """
    def reading(x):
        return jnp.stack([x[0] + 273.15, x[2]])     # an offset on one edge, a selection

    def on_the_reading(step, x, consts, w):
        ones = jnp.ones(2, F32)
        return _interface_spectral_rate_at(step, x, consts, w, reading, ones, ones,
                                           jnp.zeros(2, F32), lambda xx: jnp.abs(reading(xx)))

    counts = _count_map_evaluations(on_the_reading)
    assert counts == {"jvp": 9, "primal": 1}, counts


def _loop_vmap(f, in_axes=0, out_axes=0):
    """``jax.vmap`` as a Python loop over the leading axis, for counting calls."""
    assert in_axes == 0 and out_axes == 0

    def run(*args):
        n = jax.tree.leaves(args)[0].shape[0]
        outs = [f(*jax.tree.map(lambda a, i=i: a[i], args)) for i in range(n)]
        return jax.tree.map(lambda *xs: jnp.stack(xs), *outs)

    return run


def _count_map_evaluations(analysis):
    """``{"jvp": n, "primal": m}``: how often *analysis* evaluates a one-pass map.

    The map is ``F(x) = A x + c1 + c2 * x[0]`` on three entries with two
    floating constants (``k = 3`` range vectors, ``n_c = 2`` probes),
    evaluated eagerly with ``vmap`` unrolled, so every Jacobian-vector
    product is one call with a JVP tracer among its arguments.
    """
    A = jnp.asarray([[0.5, 0.1, 0.0], [0.0, 0.4, 0.2], [0.1, 0.0, 0.3]], F32)
    counts = {"jvp": 0, "primal": 0}

    def step_pure(x, c1, c2):
        leaves = jax.tree.leaves((x, c1, c2))
        traced = any(isinstance(leaf, jax.core.Tracer) for leaf in leaves)
        counts["jvp" if traced else "primal"] += 1
        fx = A @ x + c1 + c2 * x[0]
        return fx, jnp.linalg.norm(fx - x)

    x = jnp.asarray([0.3, -0.2, 0.7], F32)
    consts = (jnp.asarray([1.0, 2.0, 3.0], F32), jnp.asarray(0.25, F32))
    weights = jnp.ones(3, F32)
    with jax.disable_jit(), pytest.MonkeyPatch.context() as mp:
        mp.setattr(jax, "vmap", _loop_vmap)
        analysis(step_pure, x, consts, weights)
    return counts


def test_the_gradient_bound_costs_eleven_plus_four_k_plus_five_n_p_plus_two_k_n_p_products():
    """CPL-013: ``11 + 4 k + 5 n_p + 2 k n_p`` Jacobian-vector products beside the spectrum's eight.

    ``n_p`` is the number of probes: every entry of a constant of at most
    ``GRADIENT_PROBE_ENTRY_LIMIT`` entries, one for a larger constant --
    here a three-entry and a scalar constant, four probes.  Five per probe
    since 0.4.0's round-5 fix (the exact resolvent applied to each secant;
    it was four, and the probes one per constant).  ``3 k`` since round 6:
    the Kantorovich check takes the affine-covariant constant as an operator
    on the Jacobian's ``k`` row-space directions -- its change at both
    points, ``2 k``, and the resolvent applied to each, ``k``.  A state of
    at most ``k`` entries needs no reverse-mode product for the resolvent
    norm.  ``2 k n_p`` since the undirected part of the distance takes an
    operator norm (MADD-ANO-226): per probe and row-space direction, one
    forward-over-forward product and the resolvent applied to it.
    """
    def bound(step, x, consts, w):
        return _gradient_error_bound_at(step, x, consts, w, jnp.asarray(0.5, F32),
                                        jnp.asarray(0.0, F32), jnp.asarray(2.0, F32))

    counts = _count_map_evaluations(bound)
    k, n_p = 3, 4
    assert counts["jvp"] == 11 + 4 * k + 5 * n_p + 2 * k * n_p, counts
    assert counts["jvp"] <= 43 + 21 * n_p


# ---------------------------------------------------------------------------
# CPL-036 / CPL-037: the dense adjoint
# ---------------------------------------------------------------------------


def _dense_matvec(n):
    A = jnp.asarray(np.random.default_rng(0).normal(size=(n, n)) * 0.01, F32)
    x = jnp.ones(n, F32)

    def F(y):
        return jnp.tanh(A @ y) + 0.1 * y

    def matvec(v):
        return v - jax.jvp(F, (x,), (v,))[1]

    return matvec


def test_the_dense_adjoint_peaks_at_two_jacobians_in_reverse_mode():
    """CPL-036 (reverse half): ``jax.grad`` through ``"dense"`` peaks at ``2 N**2`` floats.

    XLA's compiled-module memory analysis of the transposed (adjoint) dense
    solve, for a coupling Jacobian with no structure XLA can exploit.
    """
    n = 512
    matvec = _dense_matvec(n)

    def loss(b):
        return jnp.sum(_ift_linear_solve(matvec, b, "dense"))

    compiled = jax.jit(jax.grad(loss)).lower(jnp.ones(n, F32)).compile()
    peak = compiled.memory_analysis().temp_size_in_bytes
    assert peak <= 2.05 * n * n * 4, f"{peak / (n * n * 4):.3f} N**2 floats"


def test_the_dense_tangent_solve_peaks_at_three_jacobians_in_forward_mode():
    """CPL-036 (forward half): ``jax.jvp`` through ``"dense"`` keeps ``I - J`` beside the basis.

    The documented forward-mode figure is ``3 N**2`` floats; held as an
    upper bound, so a jaxlib that fuses more cannot fail it.
    """
    n = 512
    matvec = _dense_matvec(n)
    compiled = jax.jit(lambda b: _ift_linear_solve(matvec, b, "dense")).lower(
        jnp.ones(n, F32)).compile()
    peak = compiled.memory_analysis().temp_size_in_bytes
    assert peak <= 3.05 * n * n * 4, f"{peak / (n * n * 4):.3f} N**2 floats"


def _primitives(fn, *args):
    return set(re.findall(r"= (\w+)\[", str(jax.make_jaxpr(fn)(*args))))


def test_the_dense_env_var_overrides_the_configured_solver(monkeypatch):
    """CPL-037: ``MADDENING_IFT_DENSE_SOLVE=1`` replaces GMRES whatever was configured."""
    matvec = _dense_matvec(4)
    b = jnp.ones(4, F32)
    monkeypatch.delenv("MADDENING_IFT_DENSE_SOLVE", raising=False)
    krylov = _primitives(lambda bb: _ift_linear_solve(matvec, bb, "gmres"), b)
    assert "linear_solve" in krylov                 # lineax's GMRES
    monkeypatch.setenv("MADDENING_IFT_DENSE_SOLVE", "1")
    forced = _primitives(lambda bb: _ift_linear_solve(matvec, bb, "gmres"), b)
    assert "linear_solve" not in forced and "triangular_solve" in forced
    # And the two answers agree.
    monkeypatch.delenv("MADDENING_IFT_DENSE_SOLVE")
    got = _ift_linear_solve(matvec, b, "gmres")
    monkeypatch.setenv("MADDENING_IFT_DENSE_SOLVE", "1")
    dense = _ift_linear_solve(matvec, b, "gmres")
    np.testing.assert_allclose(np.asarray(got), np.asarray(dense), rtol=1e-4)


# ---------------------------------------------------------------------------
# CPL-041 .. CPL-046: the norms at their edges
# ---------------------------------------------------------------------------


def test_the_interface_norm_measures_the_transformed_edge_values():
    """CPL-041: only the edges' source fields, through each edge's transform, RMS-scaled."""
    def transform(v):
        return 3.0 * v + 1.0

    edges = [EdgeSpec("a", "b", "x", "u", transform=transform),
             EdgeSpec("b", "a", "y", "u")]
    new = {"a": {"x": jnp.asarray([1.0, 2.0], F32), "z": jnp.asarray(5.0, F32)},
           "b": {"y": jnp.asarray(4.0, F32)}}
    old = {"a": {"x": jnp.asarray([0.9, 2.0], F32), "z": jnp.asarray(-5.0, F32)},
           "b": {"y": jnp.asarray(3.0, F32)}}
    rtol = 1e-3
    got = float(coupling_residual_interface(new, old, edges, atol=0.0, rtol=rtol))

    tn, to = 3.0 * np.array([1.0, 2.0]) + 1.0, 3.0 * np.array([0.9, 2.0]) + 1.0
    ref_x = max(np.max(np.abs(tn)), np.max(np.abs(to)))
    terms = list(np.abs(tn - to) / (rtol * ref_x)) + [abs(4.0 - 3.0) / (rtol * 4.0)]
    expected = math.sqrt(sum(t * t for t in terms) / len(terms))
    assert got == pytest.approx(expected, rel=1e-5)
    # ``z`` moved by 200% and is on no edge: it is not in the norm.
    new["a"]["z"] = jnp.asarray(1e6, F32)
    assert float(coupling_residual_interface(new, old, edges, atol=0.0, rtol=rtol)) == got


def test_the_old_and_new_mixed_scales_agree_only_at_the_fields_largest_entry():
    """CPL-042: within ``1 + atol/(rtol |v|)`` on a scalar; not on a small entry of an array."""
    atol, rtol = 1e-8, 1e-3

    def old_elementwise(new, old):
        new, old = np.asarray(new, np.float64), np.asarray(old, np.float64)
        return np.abs(new - old) / (atol + rtol * np.abs(new))

    # A scalar field well above atol / rtol (values whose float32 difference is exact).
    v_new, v_old = 2.0, 1.75
    new_norm = float(coupling_residual_mixed(_state(x=v_new), _state(x=v_old), ["n"], atol, rtol))
    old_norm = float(old_elementwise(v_new, v_old))
    bound = 1.0 + atol / (rtol * abs(v_new))
    assert max(new_norm / old_norm, old_norm / new_norm) <= bound * (1.0 + 1e-6)

    # An array field: its small entry is measured against the field's scale.
    a_new, a_old = np.array([1.0, 1e-3]), np.array([1.0, 0.999e-3])
    new_entry = float(coupling_residual_mixed(
        {"n": {"x": jnp.asarray([1e-3], F32)}}, {"n": {"x": jnp.asarray([0.999e-3], F32)}},
        ["n"], atol, rtol))
    whole = float(coupling_residual_mixed(_state(x=a_new), _state(x=a_old), ["n"], atol, rtol))
    old_entry = float(old_elementwise(a_new, a_old)[1])
    # Alone, the small field is its own scale: within the claim's factor.
    small_bound = 1.0 + atol / (rtol * 1e-3)
    assert max(new_entry / old_entry, old_entry / new_entry) <= small_bound * (1.0 + 1e-5)
    # Inside the array the entry contributes (rtol * 1.0) / (rtol * 1e-3) = 1000x less.
    assert whole * math.sqrt(2) < old_entry / 100.0


def test_only_the_l2_norm_fails_a_field_near_overflow():
    """CPL-043: a constant field above ``1 / tiny`` (8.5e37) is ``inf`` under l2 only."""
    near = _state(x=[1e38, -2e38])
    assert math.isinf(float(coupling_residual_l2(near, near, ["n"])))
    assert float(coupling_residual_mixed(near, near, ["n"], 0.0, 1e-6)) == 0.0
    inside = _state(x=[3e37, -6e37])
    assert float(coupling_residual_l2(inside, inside, ["n"])) == 0.0


def test_a_subnormal_field_reads_as_zero_and_leaves_the_norm():
    """CPL-045: a field at 1e-40 doubling reads as no change, under every norm."""
    new, old = _state(x=[2e-40], y=[1.0]), _state(x=[1e-40], y=[1.0])
    assert float(coupling_residual_l2(new, old, ["n"])) == 0.0
    assert float(coupling_residual_mixed(new, old, ["n"], 0.0, 1e-6)) == 0.0
    edges = [EdgeSpec("n", "n", "x", "u")]
    assert float(coupling_residual_interface(new, old, edges, 0.0, 1e-6)) == 0.0
    # The control: the same relative change one exponent range up is measured.
    assert float(coupling_residual_l2(_state(x=[2e-30]), _state(x=[1e-30]), ["n"])) \
        == pytest.approx(0.5, rel=1e-6)


# ---------------------------------------------------------------------------
# CPL-047 / CPL-048: the rate estimate and the error estimate
# ---------------------------------------------------------------------------


def _amp(r, rp, rpp=None):
    rpp = None if rpp is None else jnp.asarray(rpp, F32)
    return float(error_amplification(jnp.asarray(r, F32), jnp.asarray(rp, F32), rpp))


def test_error_amplification_follows_its_formula_at_every_edge():
    """CPL-047: ``1/(1 - max(r/r1, sqrt(r/r2)))``, ``0.0`` at every documented rejection."""
    # prev2 defaults to prev: the two-step term is sqrt of the one-step one.
    assert _amp(0.5, 1.0) == pytest.approx(1.0 / (1.0 - math.sqrt(0.5)), rel=1e-6)
    assert _amp(0.25, 0.5, 1.0) == pytest.approx(2.0, rel=1e-6)
    # The worse of the two rates: an alternating sequence's good half.
    assert _amp(0.25, 2.5, 0.5) == pytest.approx(1.0 / (1.0 - math.sqrt(0.5)), rel=1e-6)
    # A residual that reached zero contracts instantly: amplification 1, valid.
    assert _amp(0.0, 1.0, 1.0) == 1.0
    # Rejections: rho >= 1, a zero or non-finite predecessor, a non-finite residual.
    for args in [(1.0, 1.0), (2.0, 1.0), (0.5, 0.0), (0.5, 1.0, 0.0), (math.nan, 1.0),
                 (0.5, math.inf), (0.5, 1.0, math.nan), (math.inf, 1.0), (0.0, 0.0)]:
        assert _amp(*args) == 0.0, args
    # A valid amplification is never below one.
    rng = np.random.default_rng(0)
    for _ in range(200):
        r1, r2 = rng.uniform(1e-3, 1.0, size=2)
        r = rng.uniform(0.0, 1.0) * min(r1, r2)
        a = _amp(r, r1, r2)
        assert a == 0.0 or a >= 1.0, (r, r1, r2, a)
    assert error_amplification(jnp.asarray(0.5, F32), jnp.asarray(1.0, F32)).dtype == F32


def test_estimated_error_is_never_below_the_residual():
    """CPL-048: ``residual * max(step_scale * amplification, 1)``, rejected ratio included."""
    r = jnp.asarray(1e-3, F32)
    assert float(estimated_error(r, jnp.asarray(0.0, F32))) == float(r)          # rejected
    assert float(estimated_error(r, jnp.asarray(1.5, F32), 0.5)) == float(r)     # floored
    assert float(estimated_error(r, jnp.asarray(3.0, F32), 1.3)) == pytest.approx(3.9e-3, rel=1e-6)
    rng = np.random.default_rng(1)
    for _ in range(300):
        res = jnp.asarray(rng.uniform(0.0, 10.0), F32)
        amp = jnp.asarray(rng.choice([0.0, rng.uniform(0.0, 1e4)]), F32)
        scale = float(rng.uniform(0.01, 2.0))
        assert float(estimated_error(res, amp, scale)) >= float(res)


# ---------------------------------------------------------------------------
# CPL-064: Aitken's factor
# ---------------------------------------------------------------------------


def test_aitken_omega_is_clipped_and_falls_back_as_documented():
    """CPL-064: the zero sentinel keeps omega; ``[0.01, 2.0]``; a degenerate denominator keeps it."""
    x_old = jnp.asarray([0.0], F32)
    one = jnp.asarray(1.0, F32)

    def omega_after(prev_res, res, omega=one):
        x_raw = x_old + jnp.asarray(res, F32)
        x_rel, new_omega, cur = aitken_relaxation(x_old, x_raw, jnp.asarray(prev_res, F32), omega)
        np.testing.assert_array_equal(np.asarray(cur), np.asarray(res, np.float32))
        np.testing.assert_allclose(np.asarray(x_rel), np.asarray(new_omega * jnp.asarray(res, F32)))
        return float(new_omega)

    seeded = jnp.asarray(0.7, F32)
    assert omega_after([0.0], [0.3], seeded) == pytest.approx(0.7)   # first pass: no previous
    assert omega_after([1.0], [0.9]) == 2.0       # raw -1*(1*-0.1)/0.01 = 10, clipped
    assert omega_after([1.0], [2.0]) == float(np.float32(0.01))   # raw -1, clipped
    assert omega_after([1.0], [1.0], seeded) == pytest.approx(0.7)   # delta_r == 0
    assert omega_after([1.0], [math.inf], seeded) == pytest.approx(0.7)  # non-finite
    # Inside the interval the formula is applied as written.
    assert omega_after([1.0], [0.5]) == pytest.approx(-1.0 * (1.0 * -0.5) / 0.25)


# ---------------------------------------------------------------------------
# CPL-095: the gradient bound helper's conventions
# ---------------------------------------------------------------------------


def test_the_gradient_bound_helper_never_reports_a_number_where_none_was_computed():
    """CPL-095: NaN where nothing was computed; 0.0 only at a zero distance; inf where nothing contracts."""
    def b(*args):
        return float(ift_gradient_error_bound(*[jnp.asarray(a, F32) for a in args]))

    assert b(2.0, 0.01, 3e-3, 0.01, 1.0) == pytest.approx(2.0 * 0.01 * 3e-3 / (0.01 * 1.0))
    assert math.isnan(b(math.nan, 0.01, 3e-3, 0.01, 1.0))      # amplification not computed
    assert math.isnan(b(2.0, math.nan, 3e-3, 0.01, 1.0))       # distance not computed
    assert math.isnan(b(2.0, 0.01, 3e-3, 0.01, 0.0))           # no response: no relative error
    assert math.isnan(b(2.0, 0.01, math.inf, 0.01, 1.0))       # a non-finite secant
    assert math.isnan(b(2.0, 0.01, 3e-3, 0.0, 1.0))            # no curvature off a zero step
    assert b(2.0, 0.0, 0.0, 0.0, 1.0) == 0.0                   # a norm that reads nothing
    assert math.isinf(b(math.inf, math.inf, 3e-3, 0.01, 1.0))
    assert math.isinf(b(math.inf, 0.01, 3e-3, 0.01, 1.0))


# ---------------------------------------------------------------------------
# CPL-100: the floor
# ---------------------------------------------------------------------------


def test_the_floor_traces_under_jit_and_reads_nothing_inside_the_dead_band():
    """CPL-100: traceable; 0.0 when every field is dead-banded; each field at its own eps."""
    s = {"n": {"x": jnp.ones(4, F32), "h": jnp.ones(9, jnp.float16), "k": jnp.int32(3)}}
    eager = residual_precision_floor(s, ["n"], "l2")
    jitted = jax.jit(lambda st: residual_precision_floor(st, ["n"], "l2"))(s)
    assert float(eager) == float(jitted)
    eps16 = float(np.finfo(np.float16).eps)
    expected = PRECISION_FLOOR_ULPS * math.sqrt(4 * EPS32 ** 2 + 9 * eps16 ** 2)
    assert float(eager) == pytest.approx(expected, rel=1e-6)
    assert float(residual_precision_floor(s, ["n"], "l2", atol=2.0)) == 0.0
    assert float(residual_precision_floor(s, ["n"], "mixed", atol=2.0, rtol=1e-3)) == 0.0
    assert float(residual_precision_floor(s, ["n"], "interface", interface_edges=[])) == 0.0


# ---------------------------------------------------------------------------
# CPL-142: the IFT rule ignores the initial guess and the stopping bookkeeping
# ---------------------------------------------------------------------------


def _affine_step(x, c):
    fx = jnp.float32(0.5) * x + c
    return fx, jnp.linalg.norm(fx - x)


def test_the_ift_solve_gives_the_initial_guess_a_zero_derivative():
    """CPL-142: tangents on ``x0`` and ``first_res`` reach nothing; one on ``c`` is ``2 c_dot``."""
    x0 = jnp.asarray([0.3, -0.4], F32)
    c = jnp.asarray([1.0, 2.0], F32)
    first_res = jnp.asarray(1.0, F32)

    def solve(x0_, c_, first_res_):
        x_star, _aux = _ift_solve(_affine_step, x0_, (c_,), (), first_res_, 1e-6, 60,
                                  "none", 1.0, 0, None, "gmres")
        return x_star

    x_star, t_guess = jax.jvp(solve, (x0, c, first_res),
                              (jnp.ones(2, F32), jnp.zeros(2, F32), jnp.asarray(1.0, F32)))
    np.testing.assert_allclose(np.asarray(x_star), 2.0 * np.asarray(c), rtol=1e-5)
    np.testing.assert_array_equal(np.asarray(t_guess), np.zeros(2, np.float32))
    _, t_c = jax.jvp(solve, (x0, c, first_res),
                     (jnp.zeros(2, F32), jnp.asarray([1.0, -3.0], F32), jnp.asarray(0.0, F32)))
    np.testing.assert_allclose(np.asarray(t_c), [2.0, -6.0], rtol=1e-5)
