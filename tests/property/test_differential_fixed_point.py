"""Differential oracle: a coupled step against its exact float64 fixed point.

On a graph of linear synthetic nodes the coupling group's fixed point is a
float64 linear solve (:func:`~tests.property.coupled_graphs.exact_fixed_point`)
of the map the graph *evaluates* (gains rounded to float32 first).  The
distance from the returned state to it is measured in the group's own
norm, restated in NumPy (:func:`~tests.property.coupled_graphs.group_distance`),
so it is in the units the group's threshold is quoted in.  Two claims are
held against it:

**Converged => within the threshold**, where the design says the error
estimate is exact.  ``converged=True`` tests ``omega * r / (1 - rho)``
against the threshold; ``_fixed_point_while`` lists the conditions under
which that is the distance, and names the ones under which it is not
(a measure that is not a metric, a rate read off a mode that is not the
slowest, Aitken's clipped and IQN's quasi-Newton step -- documented, not
re-reported here).  Where none of those applies -- one coupling mode, the
iterate on it, ``acceleration`` ``"none"`` or ``"fixed"`` -- the estimate
*is* the distance, and ``converged`` must mean it.  The strategy builds
that regime: a pure cycle in Gauss-Seidel order whose cycle product has
rank one (one eigenvalue ``rho``, negative allowed, non-normal allowed),
started on that eigenvector a drawn multiple of the threshold away.

**A usable ``spectral_error_bound`` >= the true distance**, everywhere.
For a linear map the error of *any* iterate is ``(A - I)^{-1}`` of its
residual, so the bound holds "whatever the step sequence, the relaxation
or the accelerator did".  Here over general non-normal spectra near 1,
every acceleration, both modes and all three norms, from far away and
from a start already at the float floor.

Tolerances.  The threshold claim allows the float32 resolution of the
estimate, derived from its formula: a residual is a cancellation, so its
last ``8 eps`` (:func:`residual_noise_floor`) is rounding; that moves the
distance by ``floor * amp`` and the rate ``r_k / r_{k-1}`` by ``2 floor /
r``, which the amplification turns into ``2 omega floor amp**2`` on the
estimate.  At ``rho = 0.99`` the second term exceeds any threshold float32
residuals can resolve -- the honest statement is that such a group's
distance cannot be certified from its residuals in float32, and the bound
says so by being loose there.  A group the report calls
``precision_limited`` (a stalled iterate, documented) and one whose ratio
was rejected (``ratio_usable=False``: documented to fall back to the raw
residual test, which is held instead) are outside the claim.  The spectral claim is bare: the key carries its own float
floor.

What this cannot see: a non-linear group (no closed form), and any fault
shared by the reference and the library -- the restated norm is
independent, the definition of "fixed point of the evaluated map" is not.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from tests.conftest import EXAMPLES_COSTLY
from tests.core.test_coupling_solver_equivalence import residual_noise_floor
from tests.property import coupled_graphs as cg


def _threshold(group):
    return float(group["tolerance"]) if group.get("convergence_norm", "l2") == "l2" else 1.0


def _key(group):
    return tuple(sorted(group.items()))


# ---------------------------------------------------------------------------
# Converged => within the threshold, in the regime where the estimate is exact
# ---------------------------------------------------------------------------


def _single_mode_structure(m: int, n: int, flux: bool) -> cg.GraphDef:
    """A pure cycle ``g0 -> ... -> g0``, swept in its own order; no ``alpha``.

    ``alpha = 0`` keeps the pre-step state out of the constant, so the
    fixed point does not move with the start the strategy chooses.
    """
    return cg._cycle(m, n, flux_edge=0 if flux else None, outside=False, leaves=(),
                     alpha=0.0, beta=1.0)


@functools.lru_cache(maxsize=None)
def _single_mode_graph(m, n, flux, group_items):
    gdef = _single_mode_structure(m, n, flux)
    gm = cg.build_graph(gdef, dict(group_items))
    assert [nm for nm in gm.schedule if nm in gdef.group_nodes] == list(gdef.group_nodes), (
        "fixture premise: the group is swept in its cycle's order")
    return gdef, gm


def _start_on_the_mode(gdef, values, exact, group, head_start, sign):
    """``x* + s w``: *head_start* thresholds from ``x*`` along the mode ``w``.

    ``w`` is the eigenvector of the float64 Gauss-Seidel error map with the
    eigenvalue of largest modulus.  The step is sized in the group's own
    norm, so "30 thresholds away" means the same under every norm.
    """
    J = cg.gauss_seidel_matrix(gdef, values)
    lam, vec = np.linalg.eig(J)
    w = np.real(vec[:, int(np.argmax(np.abs(lam)))])
    n = gdef.n
    names = list(gdef.group_nodes)
    unit = {nm: {"x": exact[nm] + w[i * n:(i + 1) * n]} for i, nm in enumerate(names)}
    d1 = cg.group_distance(None, gdef, group, unit, exact)
    s = sign * head_start * _threshold(group) / d1 if d1 > 0 else 0.0
    start = {k: dict(v) for k, v in values.items()}
    for i, nm in enumerate(names):
        start[nm]["x0"] = np.asarray(exact[nm] + s * w[i * n:(i + 1) * n], np.float32)
    return start, float(np.real(lam[int(np.argmax(np.abs(lam)))]))


def assert_converged_means_within_threshold(gdef, gm, group, values, head_start, sign):
    pre = {nm: {"x": np.zeros(gdef.n)} for nm in gdef.group_nodes}  # alpha = 0
    exact = cg.exact_fixed_point(gdef, values, pre, {}, dt=1.0)
    start, lam = _start_on_the_mode(gdef, values, exact, group, head_start, sign)
    state, _meta, _r = cg.trajectory(gm, gdef, start, 1)[0]
    d = gm.coupling_diagnostics()[gdef.key]
    dist = cg.group_distance(gm, gdef, group, state, exact)
    thr = _threshold(group)
    n_float = gdef.n * len(gdef.group_nodes)
    floor = residual_noise_floor(group.get("convergence_norm", "l2"),
                                 group.get("rtol", 1e-6), n_float)
    # The estimate is ``omega r / (1 - rho_hat)`` with ``rho_hat`` a ratio of
    # two residuals, each carrying up to ``floor`` of rounding: the
    # residual's own error moves the distance by ``floor * amp``, and the
    # ratio's by ``2 floor / r`` -- which ``amp**2`` turns into
    # ``2 omega floor amp**2`` on the estimate, whatever ``r`` is.  At a
    # rate of 0.99 that is the whole threshold and more: float32 cannot
    # certify such a group's distance from its residuals.
    amp = 1.0 / (1.0 - abs(lam)) if abs(lam) < 1.0 else float("inf")
    omega = float(group.get("relaxation", 1.0)) if group.get("acceleration") == "fixed" else 1.0
    slack = floor * amp + 2.0 * omega * floor * amp ** 2
    note(f"lambda={lam:.6f} head_start={head_start} dist={dist:.4e} thr={thr:.4e} "
         f"{dict(d)}")
    if not d["converged"] or d["precision_limited"]:
        return
    if not d["ratio_usable"]:
        # Documented: a rejected ratio degrades ``converged`` to the raw
        # residual test, and says so.  That fallback is held below, not
        # the distance (see the module docstring).
        assert d["residual"] <= thr and d["error_estimate"] == d["residual"]
        return
    assert dist <= thr + slack, (
        f"converged=True at {dist / thr:.3f} thresholds from the exact fixed point "
        f"(one coupling mode, lambda={lam:.6f}, error_estimate={d['error_estimate']:.4e}, "
        f"iterations={d['iterations']})")


_SINGLE_MODE_CASES = {
    "none-l2": (3, 2, False, dict(tolerance=1e-4, max_iterations=200)),
    "none-l2-loose": (3, 1, False, dict(tolerance=1e-2, max_iterations=40)),
    "fixed-0.8-mixed": (3, 2, False, dict(acceleration="fixed", relaxation=0.8,
                                          convergence_norm="mixed", rtol=1e-4,
                                          max_iterations=200)),
    "fixed-1.3-mixed-flux": (2, 3, True, dict(acceleration="fixed", relaxation=1.3,
                                              convergence_norm="mixed", rtol=1e-3,
                                              max_iterations=60)),
    "fixed-0.3-l2": (2, 2, False, dict(acceleration="fixed", relaxation=0.3,
                                       tolerance=1e-4, max_iterations=400)),
    "none-interface-cap": (4, 1, False, dict(convergence_norm="interface", rtol=1e-3,
                                             max_iterations=12)),
}

#: How far from the fixed point the iterate starts, in thresholds.  Just
#: outside it (the exit then comes on the first loop pass or two), a few
#: thresholds, and far.
_HEAD_STARTS = (1.05, 1.5, 3.0, 30.0, 1000.0)


@pytest.mark.parametrize("case", sorted(_SINGLE_MODE_CASES))
# Costly tier: an eigen-decomposition and a solve per example besides the step.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_converged_means_within_the_threshold_of_the_exact_fixed_point(case, data):
    """Per push; slow sibling :func:`test_converged_means_within_the_threshold_on_generated_cycles`."""
    m, n, flux, group = _SINGLE_MODE_CASES[case]
    gdef, gm = _single_mode_graph(m, n, flux, _key(group))
    values = data.draw(cg.drawn_values(gdef, rhos=(0.5, 0.9, 0.99, -0.9), rank_one=True))
    head_start = data.draw(st.sampled_from(_HEAD_STARTS))
    sign = data.draw(st.sampled_from([1.0, -1.0]))
    assert_converged_means_within_threshold(gdef, gm, group, values, head_start, sign)


# Slow: structure and configuration are drawn, so every example builds and
# compiles graphs of its own (seconds each on CI).
# Per push: tests/property/test_differential_fixed_point.py::test_converged_means_within_the_threshold_of_the_exact_fixed_point
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_converged_means_within_the_threshold_on_generated_cycles(data):
    """The same claim with the cycle, the field size and the knobs drawn.

    Per-push sibling: :func:`test_converged_means_within_the_threshold_of_the_exact_fixed_point`.
    """
    m = data.draw(st.integers(2, 4))
    n = data.draw(st.integers(1, 3))
    flux = data.draw(st.booleans())
    norm = data.draw(st.sampled_from(["l2", "mixed"] + ([] if flux else ["interface"])))
    accel = data.draw(st.sampled_from(["none", "fixed"]))
    group = dict(convergence_norm=norm, acceleration=accel,
                 max_iterations=data.draw(st.sampled_from([6, 20, 200])))
    thr = data.draw(st.sampled_from([1e-5, 1e-4, 1e-3, 1e-2]))
    group.update({"tolerance": thr} if norm == "l2" else {"rtol": thr})
    if accel == "fixed":
        group["relaxation"] = data.draw(st.sampled_from([0.3, 0.6, 0.9, 1.3, 1.7]))
    gdef = _single_mode_structure(m, n, flux)
    gm = cg.build_graph(gdef, group)
    values = data.draw(cg.drawn_values(gdef, rhos=(0.3, 0.9, 0.99, 0.999, -0.5, -0.99),
                                       rank_one=True))
    assert_converged_means_within_threshold(
        gdef, gm, group, values, data.draw(st.sampled_from(_HEAD_STARTS)),
        data.draw(st.sampled_from([1.0, -1.0])))


# ---------------------------------------------------------------------------
# A characterised limitation: a rate float32 cannot tell from 1
# ---------------------------------------------------------------------------

#: ``(m, norm, threshold, seed, head start)``: one coupling mode at rate
#: 0.995, started on the mode 100 thresholds out, each measured to stop on a
#: noise-rejected ratio 99.5-99.7 thresholds from the fixed point on jaxlib
#: 0.10.2, 0.11.0 and 0.11.2 (so did every seed 0-5 tried, and the same
#: graph under the mixed norm at 22-30 thresholds).  One graph, one compile.
_NOISE_REJECTED = [(2, "l2", 1e-5, 0, 100.0), (2, "l2", 1e-5, 2, 100.0),
                   (2, "l2", 1e-5, 3, 100.0)]


def test_a_rate_float32_cannot_tell_from_one_falls_back_to_the_raw_residual_test():
    """A *contracting* sequence's ratio can read ``>= 1``: then the raw test decides.

    ``error_amplification`` rejects a ratio ``r_k / r_{k-1} >= 1``, and the
    criterion falls back to the raw residual test with
    ``ratio_usable=False`` -- documented, and meant for a sequence that is
    not contracting.  But each residual carries about ``floor`` of float32
    rounding, so the ratio carries about ``2 floor / r``: at rate 0.995 it
    is noise once the residual is within ~400 floors of its floor, and a
    monotone single-mode contraction reads ``>= 1`` often.  The group then
    reports ``converged=True`` on the raw residual, here about 100 thresholds
    from its fixed point (the estimate would have been ``r / (1 - 0.995)``,
    200x the residual), and ``precision_limited`` -- which reads the
    residual against its floor, not the ratio -- stays False.  This pins
    the documented fallback on every case and the reach of the limitation
    on at least one (MADD-ANO-005's residual risk); a criterion that stopped
    treating such a rejection as a pass (see the anomaly) would fail the
    second assertion, and the docs would change with it.
    """
    far = []
    for m, norm, thr, seed, head in _NOISE_REJECTED:
        group = dict(acceleration="none", max_iterations=60, convergence_norm=norm,
                     diagnostics=True)
        group.update({"tolerance": thr} if norm == "l2" else {"rtol": thr})
        gdef, gm = _single_mode_graph(m, 1, False, _key(cg.live_knobs(group)))
        values = cg.draw_values(np.random.default_rng(seed), gdef, 0.995, rank_one=True,
                                nonnormal=bool(seed % 2))
        pre = {nm: {"x": np.zeros(gdef.n)} for nm in gdef.group_nodes}
        exact = cg.exact_fixed_point(gdef, values, pre, {}, dt=1.0)
        start, lam = _start_on_the_mode(gdef, values, exact, group, head, 1.0)
        assert 0.99 < lam < 1.0, "fixture premise: a monotone contraction near 1"
        state, _m, _r = cg.trajectory(gm, gdef, start, 1)[0]
        d = gm.coupling_diagnostics()[gdef.key]
        dist = cg.group_distance(gm, gdef, group, state, exact)
        if d["converged"] and not d["ratio_usable"]:
            # The documented fallback, exactly: the raw residual decided.
            assert d["error_estimate"] == d["residual"] <= _threshold(group)
            if dist > 10 * _threshold(group) and not d["precision_limited"]:
                far.append(dist / _threshold(group))
    assert far, "no case stopped on a noise-rejected ratio far from its fixed point"


# ---------------------------------------------------------------------------
# A usable spectral bound bounds the true distance, whatever the spectrum
# ---------------------------------------------------------------------------


def assert_spectral_bound_holds(gdef, gm, group, values, near: float | None):
    """One step from ``x0`` (or from ``x* (1 + near)``); the bound against the truth."""
    if near is not None:
        # Start a relative *near* from this step's fixed point.  With a
        # driver, its output this step is a function of its own state
        # only, so it is known before the step: compute it the way the
        # node does, in float32.
        inputs = _outside_inputs(gdef, values)
        pre = {nm: {"x": values[nm]["x0"]} for nm in gdef.group_nodes}
        exact0 = cg.exact_fixed_point(gdef, values, pre, inputs, dt=1.0)
        if any(gdef.node(nm).alpha != 0.0 for nm in gdef.group_nodes):
            # The constant depends on the pre-step state; iterate the start
            # onto the fixed point of the map it defines.
            for _ in range(3):
                for nm in gdef.group_nodes:
                    values[nm]["x0"] = np.asarray(exact0[nm], np.float32)
                pre = {nm: {"x": values[nm]["x0"]} for nm in gdef.group_nodes}
                exact0 = cg.exact_fixed_point(gdef, values, pre, inputs, dt=1.0)
        for nm in gdef.group_nodes:
            values[nm]["x0"] = np.asarray(exact0[nm] * (1.0 + near), np.float32)
    pre = {nm: {"x": np.asarray(values[nm]["x0"], np.float64)} for nm in gdef.group_nodes}
    state, _meta, _r = cg.trajectory(gm, gdef, values, 1)[0]
    inputs = {(e.dst, e.port): state[e.src]["x"] for e in gdef.edges
              if e.dst in gdef.group_nodes and e.src not in gdef.group_nodes}
    exact = cg.exact_fixed_point(gdef, values, pre, inputs, dt=1.0)
    dist = cg.group_distance(gm, gdef, group, state, exact)
    d = gm.coupling_diagnostics()[gdef.key]
    note(f"dist={dist:.4e} near={near} {dict(d)}")
    if d["spectral_usable"]:
        assert d["spectral_error_bound"] >= dist, (
            f"spectral_error_bound {d['spectral_error_bound']:.4e} below the true "
            f"distance {dist:.4e} (rho_spectral={d['rho_spectral']}, "
            f"iterations={d['iterations']}, precision_limited={d['precision_limited']})")


def _outside_inputs(gdef, values):
    """The driver's output this step, computed as the node computes it (float32)."""
    out = {}
    for e in gdef.edges:
        if e.dst in gdef.group_nodes and e.src not in gdef.group_nodes:
            nd = gdef.node(e.src)
            v = values[e.src]
            x = (np.float32(nd.alpha) * np.asarray(v["x0"], np.float32) + v["b"]
                 + np.float32(nd.beta) * np.float32(1.0))
            out[(e.dst, e.port)] = np.asarray(x, np.float32)
    return out


_LINEAR_TRIANGLE = cg._cycle(3, 2, chords=((0, 2),), leaves=())
_LINEAR_RING = cg._cycle(4, 1, chords=((1, 3),), leaves=())
_SPECTRAL_STRUCTURES = {"triangle": _LINEAR_TRIANGLE, "ring": _LINEAR_RING}

_SPECTRAL_CASES = {
    "iqn-imvj-jacobi-interface": ("triangle", dict(
        acceleration="iqn-imvj", jacobian_reuse=2, iteration_mode="jacobi",
        convergence_norm="interface", rtol=1e-4, max_iterations=30)),
    "aitken-gs-mixed": ("ring", dict(
        acceleration="aitken", convergence_norm="mixed", rtol=1e-4, max_iterations=30)),
    "none-l2-capped": ("triangle", dict(tolerance=1e-6, max_iterations=8)),
}


@functools.lru_cache(maxsize=None)
def _spectral_graph(structure, group_items):
    gdef = _SPECTRAL_STRUCTURES[structure]
    return gdef, cg.build_graph(gdef, dict(group_items, diagnostics=True))


@pytest.mark.parametrize("case", sorted(_SPECTRAL_CASES))
# Costly tier: diagnostics=True takes Jacobian-vector products per step.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_usable_spectral_bound_is_never_below_the_true_distance(case, data):
    """Per push; slow sibling :func:`test_a_usable_spectral_bound_holds_on_generated_graphs`."""
    structure, group = _SPECTRAL_CASES[case]
    gdef, gm = _spectral_graph(structure, _key(group))
    values = data.draw(cg.drawn_values(gdef, rhos=(0.5, 0.9, 0.99, 0.999)))
    near = data.draw(st.sampled_from([None, None, 1e-3, 1e-6]))
    assert_spectral_bound_holds(gdef, gm, group, values, near)


# Slow: structure and configuration are drawn, so every example builds and
# compiles graphs of its own (seconds each on CI).
# Per push: tests/property/test_differential_fixed_point.py::test_a_usable_spectral_bound_is_never_below_the_true_distance
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_usable_spectral_bound_holds_on_generated_graphs(data):
    """The spectral claim with structure and configuration drawn.

    Per-push sibling: :func:`test_a_usable_spectral_bound_is_never_below_the_true_distance`.
    """
    gdef = data.draw(cg.graph_defs(allow_nonlinear=False, leaves=()))
    group = data.draw(cg.group_configs(gdef, predictors=("none",), caps=(2, 5, 30, 200),
                                       thresholds=(1e-6, 1e-4, 1e-2)))
    note(f"{gdef}\n{group}")
    gm = cg.build_graph(gdef, dict(group, diagnostics=True))
    values = data.draw(cg.drawn_values(gdef, rhos=(0.5, 0.9, 0.99, 0.999)))
    assert_spectral_bound_holds(gdef, gm, group, values,
                                data.draw(st.sampled_from([None, 1e-3, 1e-6])))


# ---------------------------------------------------------------------------
# Long Gauss-Seidel chains at the float32 floor
# ---------------------------------------------------------------------------
#
# The spectral bound adds the residual's float floor, a model of how far a
# computed residual can sit from the exact map's: ``PRECISION_FLOOR_ULPS``
# units per evaluation the pass rounds like.  Under Gauss-Seidel a member
# reads every member scheduled before it from the same pass, already
# rounded, so the rounding at the end of a chain of such reads is the sum
# along it; the floor counted the worst *node*, and on a 32-relay ring
# stalled at float32 the bound read 0.51x the true distance with the flag
# set (64 relays: 0.30x).  The structures and examples above are too short
# to see it (the floor carries 2.6x headroom per evaluation, the stall
# distance grows with the chain), so this draws long rings, at loop gains
# near one, started a hair from the fixed point with a tolerance only the
# bitwise stall meets: the residual is the floor, and the floor is the bound.


def _long_ring(m):
    """A pure cycle of *m* scalar relays, swept in its own order; no outside nodes."""
    return cg._cycle(m, 1, outside=False, leaves=(), alpha=0.0, beta=1.0)


@functools.lru_cache(maxsize=None)
def _long_ring_graph(m, mode):
    gdef = _long_ring(m)
    group = dict(tolerance=1e-12, max_iterations=600, iteration_mode=mode, diagnostics=True)
    gm = cg.build_graph(gdef, group)
    assert [nm for nm in gm.schedule if nm in gdef.group_nodes] == list(gdef.group_nodes), (
        "fixture premise: the ring is swept in its own order, a same-pass chain of m")
    return gdef, gm, group


def _uniform_ring_values(gdef, rho, scale):
    """Every link's gain ``rho ** (1/m)``, biases ``scale * (1 - g)``: the worst case.

    Each relay reads its predecessor with a gain just below one, so a
    rounding at the start of the chain reaches its end almost undamped,
    and every fixed-point entry is ``scale`` -- no cancellation anywhere,
    which is the regime the floor's model claims.  This is the ring the
    finding measured (0.51x at 32 relays); drawn rank-one gains damp the
    chain and read 1.0-1.7x on the old floor instead.
    """
    m = len(gdef.group_nodes)
    g = np.float32(rho ** (1.0 / m))
    return {nm: {"G": [np.array([[g]], np.float32)],
                 "b": np.array([scale * (1.0 - float(g))], np.float32),
                 "x0": np.zeros(1, np.float32)}
            for nm in gdef.group_nodes}


@st.composite
def _long_ring_values(draw, m):
    """The worst-case uniform ring, or drawn rank-one gains, near one either way."""
    gdef = _long_ring(m)
    if draw(st.booleans()):
        return _uniform_ring_values(gdef, draw(st.sampled_from([0.99, 0.995])),
                                    draw(st.sampled_from([1.0, 0.75, 1e-3, 3e3])))
    return draw(cg.drawn_values(gdef, rhos=(0.99, 0.995, 0.999), rank_one=True,
                                bias_scales=(1.0,)))


def assert_the_floor_bound_holds_on_a_long_ring(m, mode, values, near):
    gdef, gm, group = _long_ring_graph(m, mode)
    assert_spectral_bound_holds(gdef, gm, group, values, near)


@pytest.mark.parametrize("m", [8])
@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
# Costly tier: one compile per (m, mode); the examples change values only.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_usable_spectral_bound_holds_on_a_stalled_ring(m, mode, data):
    """Per push, at a length the old floor still covered; slow sibling at 24 and 32."""
    assert_the_floor_bound_holds_on_a_long_ring(
        m, mode, data.draw(_long_ring_values(m)), data.draw(st.sampled_from([1e-3, 1e-4])))


# Slow: a 24- or 32-relay group with diagnostics=True compiles the spectral
# and gradient-bound machinery over every relay's constants (~15-30 s each).
# Per push: tests/property/test_differential_fixed_point.py::test_a_usable_spectral_bound_holds_on_a_stalled_ring
@pytest.mark.slow
@pytest.mark.parametrize("m", [24, 32])
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_usable_spectral_bound_holds_on_a_long_gauss_seidel_chain(m, data):
    """A usable bound against the exact float64 fixed point on a 24- or 32-deep chain.

    Per-push sibling: :func:`test_a_usable_spectral_bound_holds_on_a_stalled_ring`.
    """
    assert_the_floor_bound_holds_on_a_long_ring(
        m, "gauss-seidel", data.draw(_long_ring_values(m)),
        data.draw(st.sampled_from([1e-3, 3e-4])))


# ---------------------------------------------------------------------------
# A disagreement: under-relaxation and an exit on the first loop pass
# ---------------------------------------------------------------------------


def _relaxed_pair(solver, relaxation, delta, tolerance=1e-4):
    """``a: x = 0.5 u + 1``, ``b: x = u``: one mode, rate 0.5, fixed point 2.

    Both nodes start at ``2 (1 + delta)``, on the mode, so the true
    distance is known exactly: ``sqrt(2) * delta`` in the relative L2
    norm after ``k`` passes is ``sqrt(2) * delta * 0.5**k``.
    """
    gdef = cg.GraphDef(n=1, nodes=(cg.NodeDef("a", 1), cg.NodeDef("b", 1)),
                       edges=(cg.EdgeDef("b", "a", 0), cg.EdgeDef("a", "b", 0)),
                       group_nodes=("a", "b"))
    start = np.array([2.0 * (1.0 + delta)], np.float32)
    values = {"a": {"G": [np.array([[0.5]], np.float32)], "b": np.array([1.0], np.float32),
                    "x0": start},
              "b": {"G": [np.array([[1.0]], np.float32)], "b": np.array([0.0], np.float32),
                    "x0": start}}
    group = dict(acceleration="fixed", relaxation=relaxation, tolerance=tolerance,
                 max_iterations=50, solver=solver, diagnostics=True)
    gm = _relaxed_graph(gdef, _key(group))
    state, _m, _r = cg.trajectory(gm, gdef, values, 1)[0]
    d = gm.coupling_diagnostics()["a+b"]
    exact = {"a": np.array([2.0]), "b": np.array([2.0])}
    return d, cg.group_distance(gm, gdef, group, state, exact)


@functools.lru_cache(maxsize=None)
def _relaxed_graph(gdef, group_items):
    return cg.build_graph(gdef, dict(group_items))


@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_the_estimate_is_the_distance_for_fixed_relaxation_past_the_first_pass(solver):
    """The control: with the exit past the first loop pass, est == distance.

    Relaxation 0.7 exits on its third pass here, and the estimate equals
    the true distance to three digits -- ``omega`` *is* exact for
    ``acceleration="fixed"`` once the ratio comes from relaxed passes.
    """
    d, dist = _relaxed_pair(solver, 0.7, 2.5e-4)
    assert d["iterations"] > 1 and d["converged"] and d["ratio_usable"]
    assert d["error_estimate"] == pytest.approx(dist, rel=1e-2)


@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_converged_on_the_first_pass_under_under_relaxation_is_within_the_threshold(solver):
    """Converged => within the threshold, in the regime the docs call exact.

    ``_fixed_point_while`` documents ``omega`` as exact for
    ``acceleration="fixed"``: the iterate's steps are ``omega`` times the
    residual and contract at ``mu = 1 - omega (1 - rho)``, so the series
    ``omega r / (1 - mu)`` is ``r / (1 - rho)``, the distance.  But the
    first loop pass seeds its ratio with ``first_res`` -- the residual of
    the *unrelaxed* pass ``_run_coupling_inner`` ran before the loop --
    so on that pass the rate read is ``rho`` (as ``sqrt(rho)``, the seed
    filling both slots), not ``mu``, and ``omega r / (1 - sqrt(rho))``
    fell short of ``r / (1 - rho)`` whenever ``omega < 1 / (1 + sqrt(rho))``.
    Here (``rho = 0.5``, ``omega = 0.3``, two thresholds away) the group
    stopped on that pass with ``ratio_usable=True``, ``error_estimate``
    0.72 thresholds and ``converged=True``, 1.41 thresholds from the
    fixed point.  The first pass now reads the relaxed rate the ratio
    implies (``first_pass_relaxed_amplification``), the estimate there is
    ``r / (1 - sqrt(rho))`` = 2.4 thresholds, and the group goes on to a
    later pass where the estimate is the distance.  A first-pass exit
    closer in, and the neighbouring relaxations:
    ``tests/core/test_coupling_first_pass_relaxed_rate.py``.
    """
    d, dist = _relaxed_pair(solver, 0.3, 2.0e-4)
    assert d["ratio_usable"] and d["converged"]
    assert d["iterations"] > 1, "the first pass's estimate (2.4 thresholds) let it stop"
    assert dist <= 1e-4, (
        f"converged=True at {dist / 1e-4:.2f} thresholds "
        f"(error_estimate {d['error_estimate'] / 1e-4:.2f}, {d['iterations']} passes)")
    # Past the first pass the estimate is the distance, to its own float32
    # resolution (``floor * amp + 2 omega floor amp**2``; the module docstring).
    floor = residual_noise_floor("l2", 1e-6, 2)
    amp = d["amplification"]
    assert abs(d["error_estimate"] - dist) <= floor * amp + 2 * 0.3 * floor * amp ** 2
