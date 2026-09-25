"""How wrong the IFT gradient is when the coupled forward stops early.

Under ``solver="ift"`` the adjoint solves ``(I - dF/dx)^T lambda = dL/dx``
at the iterate the forward *returned*, not at the fixed point, so a
forward that stops early hands back the derivative of a fixed point
linearised in the wrong place.  ``coupling_diagnostics()`` reports
``gradient_relative_error_bound`` for it under ``diagnostics=True``: the
distance to the fixed point (``spectral_error_bound``), times the factor
the adjoint's resolvent applies, times the change in the one-pass map's
linearisation per unit distance -- the curvature.

What is pinned here, each on a fresh graph through the public
``gm.step()``, with the "true" gradient of the fixed point taken from a
tight unaccelerated ``l2`` solve and **confirmed** by two independent
arms (``ift`` and ``fori``) agreeing, never assumed:

* on two non-linear maps of opposite curvature sign the bound is at
  least the true relative error at every point of a ``max_iterations``
  sweep that stops the forward early by construction, and the ratio is
  the recorded size (two-sided, like the spectral bound's own pins);
* on an affine map it reads ~0 -- truthfully, the gradient is exact --
  while the forward value is far from the fixed point: the bound is a
  statement about the gradient, not the solve;
* an affine map is not exempt when a parameter multiplies the state;
* the distance it uses is the spectral bound's, which is what holds on
  a hidden slow mode where ``error_estimate`` reads 100x short;
* where nothing was computed it is NaN and not usable, never a number.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _Curved(SimulationNode):
    """``x <- a + g * phi(u)`` with ``a`` and ``g`` parameters.

    ``phi = log(1 + |u|)`` is concave and ``phi = u**2`` convex, so the
    two fixtures bend the linearisation in opposite directions as the
    iterate approaches the fixed point.  ``phi = u`` is affine: its
    gradient with respect to ``a`` (which enters additively) is exact
    wherever the forward stops, and with respect to ``g`` (which
    multiplies the state) is not.
    """

    def __init__(self, name, kind, a, g):
        super().__init__(name=name, timestep=1.0, a=a, g=g)
        self._kind = kind

    def initial_state(self):
        return {"x": jnp.asarray(0.0, jnp.float32)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs["u"]
        if self._kind == "log":
            phi = jnp.log1p(jnp.abs(u))
        elif self._kind == "square":
            phi = u * u
        else:
            phi = u
        return {"x": jnp.asarray(p["a"]) + jnp.asarray(p["g"]) * phi}


class _Relay(SimulationNode):
    """``x <- u``: closes the loop, so one pass is ``u <- a + g phi(u)``."""

    def __init__(self, name, shape=()):
        super().__init__(name=name, timestep=1.0)
        self._shape = shape

    def initial_state(self):
        return {"x": jnp.zeros(self._shape, jnp.float32)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=self._shape, dtype=jnp.float32,
                                       default=jnp.zeros(self._shape, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": boundary_inputs["u"]}


#: ``(a, g)`` per curvature.  Both contract at the fixed point with a
#: positive slope -- ``F'(x*) = 0.368`` (log) and ``0.553`` (square) --
#: and are approached monotonically from below from ``x = 0``.  The
#: affine map's slope is 0.9, slow enough for a long sweep.
_CURVED = {"log": (1.0, 2.5), "square": (1.0, 0.2), "affine": (1.0, 0.9)}

#: Unreachable in float32 within the sweep's caps, so every point of it
#: exhausts ``max_iterations``: the early exit is by construction, and
#: the test asserts it rather than assuming it.
_UNREACHABLE = 1e-7

_PHI = {
    "log": (lambda u: math.log1p(abs(u)), lambda u: 1.0 / (1.0 + u)),
    "square": (lambda u: u * u, lambda u: 2.0 * u),
    "affine": (lambda u: u, lambda u: 1.0),
}


def _curved_graph(kind, **group_kw):
    a, g = _CURVED[kind]
    gm = GraphManager()
    gm.add_node(_Curved("a", kind, a, g))
    gm.add_node(_Relay("b"))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    kw = dict(diagnostics=True, max_iterations=60, tolerance=_UNREACHABLE)
    kw.update(group_kw)
    gm.add_coupling_group(["a", "b"], **kw)
    gm.compile()
    return gm


def _gradients(build, **group_kw):
    """``d x_a / d params['a']`` through one step, on a fresh graph each call."""
    base = build(**group_kw).params

    def loss(p):
        return jnp.ravel(build(**group_kw).run_scan(1, params=p)["a"]["x"])[0]

    return jax.grad(loss)(base)["nodes"]["a"]


def _analytic_curved(kind):
    """``u*`` and ``d u*/d(a, g)`` in float64, by fixed-point iteration."""
    a, g = _CURVED[kind]
    phi, dphi = _PHI[kind]
    u = 0.0
    for _ in range(100_000):
        u = a + g * phi(u)
    den = 1.0 - g * dphi(u)
    return u, {"a": 1.0 / den, "g": phi(u) / den}


@functools.lru_cache(maxsize=None)
def _curved_reference(kind):
    """The fixed point's gradient, from two independent tight arms.

    ``ift`` differentiates the fixed point implicitly; ``fori`` unrolls
    four hundred passes and differentiates straight through them.  They
    share nothing but the map, so their agreement is what makes either
    a reference -- and both are checked against the float64 fixed point
    as well, because a reference that is merely self-consistent is the
    failure this suite exists to catch elsewhere.
    """
    tight = dict(max_iterations=400, tolerance=_UNREACHABLE)
    ift = _gradients(functools.partial(_curved_graph, kind), **tight)
    fori = _gradients(functools.partial(_curved_graph, kind), solver="fori", **tight)
    _, exact = _analytic_curved(kind)
    ref = {}
    for p in ("a", "g"):
        gi, gf = float(ift[p]), float(fori[p])
        assert gi == pytest.approx(gf, rel=1e-5), (kind, p, gi, gf)
        assert gi == pytest.approx(exact[p], rel=1e-5), (kind, p, gi, exact[p])
        ref[p] = gi
    return ref


# ---------------------------------------------------------------------------
# The bound holds on curved maps, across a sweep that stops early
# ---------------------------------------------------------------------------

#: ``gradient_relative_error_bound / true relative error``, measured on
#: jaxlib 0.11.0 (CPU, float32) at ``max_iterations`` 3-8, one entry per
#: (map, parameter).  Why each is the size it is:
#:
#: * two conservative factors enter every ratio, neither of them slack
#:   in the curvature: the distance is ``spectral_error_bound``, 1.1x
#:   the true distance in the group's norm here, and the resolvent
#:   factor is the one that bound applies, the ``a -> b -> a`` relay's
#:   1.22x over ``1/(1 - F')``.  So the parameter whose relative error
#:   is the larger reads near their product -- log's ``a`` 1.21-1.66,
#:   square's ``g`` 1.17-1.35 -- higher at the earliest caps, where the
#:   curvature is not constant over the distance (3.5 and 1.03 at a cap
#:   of two, outside the sweep);
#: * the reported value is the *largest* over one probe per constant, so
#:   the other parameter reads its gap to that one as well: log's ``g``
#:   has a ninth of ``a``'s relative error (6.9-11.4), square's ``a``
#:   2.7x less than ``g``'s (3.46-3.55).
_RECORDED_RATIO = {
    ("log", "a"): 1.3,
    ("log", "g"): 9.0,
    ("square", "a"): 3.5,
    ("square", "g"): 1.3,
}

#: A factor of 2.5 each way, matching the spectral bound's own pins --
#: wide enough for a float32 reshuffle and far too narrow for an order of
#: magnitude -- with the lower edge never below 1, which is the bound
#: itself.
_BAND = 2.5

#: Passes at which the forward is stopped: the curved maps are 10-26%
#: from their fixed points after three and 0.2-0.3% after eight.
_SWEEP = (3, 4, 6, 8)


# Slow-marked (still run by slow-tests.yml), like the four tests after it:
# each point of the sweep is a forward graph and a gradient graph compiled
# for its own static cap, plus two tight reference gradients -- 20-23 s on
# the CI runner.  The bound's defining claim, "it holds where it is usable",
# stays on every push in
# ``test_the_gradient_bound_is_unusable_where_kantorovich_fails_and_holds_where_it_passes``
# and the NaN-where-not-computed tests.
@pytest.mark.slow
@pytest.mark.parametrize("kind", ["log", "square"])
def test_the_gradient_bound_holds_across_an_early_exit_sweep(kind):
    """``|g_k - g*| <= bound * |g_k|`` at every cap, and the ratio is recorded.

    ``F''`` is negative on the log map and positive on the square one,
    so a curvature term that only worked for one sign -- or an absolute
    value dropped somewhere -- fails one of the two.  Every point
    exhausts its cap (asserted), so the forward is early by
    construction rather than by accident, and each point is a fresh
    graph: ``step()`` and ``run_scan`` both leave the graph at the state
    they reached.
    """
    ref = _curved_reference(kind)
    u_star, _ = _analytic_curved(kind)
    for m in _SWEEP:
        gm = _curved_graph(kind, max_iterations=m)
        gm.step()
        d = gm.coupling_diagnostics()["a+b"]
        assert d["iterations"] == m and d["converged"] is False, (
            f"{kind} m={m}: fixture premise, the forward must stop early", d)
        assert d["gradient_bound_usable"] is True, (kind, m, d)
        x = float(gm.get_node_state("a")["x"])
        assert abs(x - u_star) / u_star > 1e-3, "fixture premise: visibly early"
        g_k = _gradients(functools.partial(_curved_graph, kind), max_iterations=m)
        bound = d["gradient_relative_error_bound"]
        for p in ("a", "g"):
            gk = float(g_k[p])
            true = abs(gk - ref[p]) / abs(gk)
            assert true <= bound, (
                f"{kind} m={m} d/d{p}: the returned gradient is {true:.3e} "
                f"(relative) from the fixed point's, above the reported bound "
                f"{bound:.3e}"
            )
            rec = _RECORDED_RATIO[(kind, p)]
            ratio = bound / true
            assert max(1.0, rec / _BAND) <= ratio <= rec * _BAND, (
                f"{kind} m={m} d/d{p}: bound / true = {ratio:.3f}, not the "
                f"~{rec} recorded (band x{_BAND}).  Below it the bound lost a "
                f"factor it is built from; above it one grew."
            )


class _SwitchedOff(SimulationNode):
    """``x <- a + g u**2 + h u**5`` with ``h = 0``: a correction term switched off.

    ``h`` does not change the dynamics at all, and the gradient with
    respect to it is still a question a calibration asks -- whether to
    switch the term on.  ``F_h = u**5`` moves five times as fast as
    ``u`` does in relative terms, against ``F_g = u**2``'s two, so its
    relative error is the largest of the three and its probe is the one
    that decides the bound.  A zero-valued constant has no magnitude to
    perturb it by; it is probed at its fallback scale.
    """

    def __init__(self, name):
        super().__init__(name=name, timestep=1.0, a=1.0, g=0.2, h=0.0)

    def initial_state(self):
        return {"x": jnp.asarray(0.0, jnp.float32)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs["u"]
        return {"x": jnp.asarray(p["a"]) + jnp.asarray(p["g"]) * u * u
                + jnp.asarray(p["h"]) * u ** 5}


def _switched_off_graph(**group_kw):
    gm = GraphManager()
    gm.add_node(_SwitchedOff("a"))
    gm.add_node(_Relay("b"))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    kw = dict(diagnostics=True, max_iterations=60, tolerance=_UNREACHABLE)
    kw.update(group_kw)
    gm.add_coupling_group(["a", "b"], **kw)
    gm.compile()
    return gm


# Slow-marked: a tight reference gradient and two capped forward/gradient
# pairs, 11-14 s on the CI runner (see the sweep above).
@pytest.mark.slow
def test_a_parameter_that_sits_at_zero_is_still_probed():
    """``h = 0`` still gets a probe, and the bound covers its gradient.

    The probe perturbs each constant by its own magnitude, so a constant
    at zero would contribute a zero direction, a zero tangent, and drop
    out of the maximum -- silently, since the other probes still give a
    number.  Measured when it did: 0.54-0.70 of ``d/dh``'s true error.
    Probed at the fallback scale it reads 1.17-1.35x, the same two
    conservative factors as the parameter that decides the bound on the
    other curved maps (jaxlib 0.11.0).
    """
    u_star, _ = _analytic_curved("square")
    exact_h = u_star ** 5 / (1.0 - 0.4 * u_star)
    g_star = _gradients(_switched_off_graph, max_iterations=400)
    assert float(g_star["h"]) == pytest.approx(exact_h, rel=1e-5), "fixture premise"
    for m in (3, 6):
        gm = _switched_off_graph(max_iterations=m)
        gm.step()
        d = gm.coupling_diagnostics()["a+b"]
        assert d["iterations"] == m and d["gradient_bound_usable"] is True, d
        g_k = _gradients(_switched_off_graph, max_iterations=m)
        true = {p: abs(float(g_k[p]) - float(g_star[p])) / abs(float(g_k[p]))
                for p in ("a", "g", "h")}
        assert true["h"] == max(true.values()), (
            "fixture premise: the switched-off term has the largest error", true)
        ratio = d["gradient_relative_error_bound"] / true["h"]
        assert 1.0 <= ratio <= 1.35 * _BAND, (
            f"m={m}: bound / true for d/dh = {ratio:.3f}; a zero-valued "
            f"constant that is not probed reads ~0.6 here")


# ---------------------------------------------------------------------------
# The bound is about the gradient, not the solve
# ---------------------------------------------------------------------------


class _TwoMode(SimulationNode):
    """``x <- (c_slow + r_s u0 + q u0**2, c_fast + r_f u1)``; ``c_*`` are parameters.

    With ``q = 0`` the map is affine and both parameters enter
    additively, so the IFT gradient is exact from any iterate.  The slow
    mode carries ``c_slow`` per pass but is amplified ``1/(1 - r_s)``;
    the fast mode carries 1.0 and dominates the step until long after
    the criterion is met -- the case where ``error_estimate`` reads the
    fast rate and understates the distance by ~100x.
    """

    def __init__(self, name, q):
        super().__init__(name=name, timestep=1.0, c_slow=1e-5, c_fast=1.0)
        self._q = float(q)

    def initial_state(self):
        return {"x": jnp.zeros(2, jnp.float32)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(2,), dtype=jnp.float32,
                                       default=jnp.zeros(2, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs["u"]
        slow = jnp.asarray(p["c_slow"]) + 0.999 * u[0] + self._q * u[0] * u[0]
        fast = jnp.asarray(p["c_fast"]) + 0.2 * u[1]
        return {"x": jnp.stack([slow, fast])}


def _two_mode_graph(q, **group_kw):
    gm = GraphManager()
    gm.add_node(_TwoMode("a", q))
    gm.add_node(_Relay("b", shape=(2,)))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    kw = dict(diagnostics=True, max_iterations=60, tolerance=1e-4)
    kw.update(group_kw)
    gm.add_coupling_group(["a", "b"], **kw)
    gm.compile()
    return gm


def _two_mode_fixed_point(q):
    u = 0.0
    for _ in range(400_000):
        u = 1e-5 + 0.999 * u + q * u * u
    return (u, 1.0 / (1.0 - 0.2)), 1.0 / (1.0 - (0.999 + 2.0 * q * u))


def _two_mode_distance(gm, q):
    """Distance to the float64 fixed point, in the group's own L2 norm."""
    exact, _ = _two_mode_fixed_point(q)
    total = 0.0
    for node in ("a", "b"):
        got = [float(v) for v in gm.get_node_state(node)["x"]]
        ref = max(max(abs(v) for v in got), max(abs(e) for e in exact))
        total += sum(((v - e) / ref) ** 2 for v, e in zip(got, exact))
    return total ** 0.5


@functools.lru_cache(maxsize=None)
def _two_mode_reference(q):
    """``d x_a[0] / d c_slow`` at the fixed point, two tight arms agreeing.

    The slow mode contracts at ~0.999 per pass, so "tight" means twenty
    thousand passes: the unrolled ``fori`` derivative is a truncated
    geometric series and is 5% short after three thousand.  At twenty
    thousand the arms agree to 1e-4 and the ``ift`` one matches the
    float64 fixed point to 1e-5.
    """
    tight = dict(max_iterations=20_000, tolerance=1e-9)
    build = functools.partial(_two_mode_graph, q)
    gi = float(_gradients(build, **tight)["c_slow"])
    gf = float(_gradients(build, solver="fori", **tight)["c_slow"])
    _, exact = _two_mode_fixed_point(q)
    assert gi == pytest.approx(gf, rel=2e-4), (q, gi, gf)
    assert gi == pytest.approx(exact, rel=1e-4), (q, gi, exact)
    return gi


# Slow-marked: its reference is two gradients through 20 000 passes, 10-12 s
# on the CI runner (see the sweep above).
@pytest.mark.slow
def test_on_an_affine_map_the_gradient_bound_reads_zero_while_the_forward_is_far_off():
    """The docstring caveat, as a test: a tiny bound is not a healthy solve.

    The two-mode map is affine and its parameters enter additively, so
    ``G(x) = J(x) t + F_c(x) c_dot`` is the same at every point and the
    IFT gradient is the fixed point's wherever the forward stopped.  The
    bound therefore reads zero -- exactly, because the curvature is a
    difference of two evaluations of the same Jacobian-vector product --
    and it is *right*.  On the same graph the forward reports
    ``converged=True`` while sitting over a hundred tolerances from the
    fixed point, and ``spectral_error_bound`` is the key that says so.
    The returned ``(value, gradient)`` pair is mutually inconsistent --
    the gradient is ``d(fixed point)/dc`` and the value is not the fixed
    point -- and nothing in ``gradient_relative_error_bound`` can say
    so, which is what its docstring warns.
    """
    tol = 1e-4
    gm = _two_mode_graph(0.0, tolerance=tol)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    distance = _two_mode_distance(gm, 0.0)
    assert d["converged"] is True, d
    assert distance > 100 * tol, (
        f"fixture premise: the forward is {distance:.3e} from the fixed point")
    assert d["spectral_error_bound"] >= distance, (
        "the solve's health is spectral_error_bound's to report", d, distance)
    assert d["gradient_bound_usable"] is True, d
    assert d["gradient_relative_error_bound"] <= 1e-6, (
        f"an affine map with additive parameters has an exact IFT gradient "
        f"from any iterate, but the bound reads "
        f"{d['gradient_relative_error_bound']:.3e}.  A float32-sized value "
        f"here is the resolvent amplifying rounding in the secant (measured "
        f"2.5e-4 when the norm weights were applied before the difference)."
    )
    g_k = float(_gradients(functools.partial(_two_mode_graph, 0.0), tolerance=tol)["c_slow"])
    g_star = _two_mode_reference(0.0)
    assert abs(g_k - g_star) <= 1e-6 * abs(g_star), (
        f"fixture premise: the early-exit gradient {g_k} is the fixed "
        f"point's {g_star}")


# Slow-marked: two capped forward/gradient pairs, 8-9 s on the CI runner
# (see the sweep above).
@pytest.mark.slow
def test_an_affine_map_with_a_multiplicative_parameter_is_not_exempt():
    """Affine in the state is not enough: ``g`` multiplies it.

    ``x <- a + g u`` is affine, and its gradient in ``a`` is exact at any
    iterate (``F_a = 1`` everywhere).  Its gradient in ``g`` is not:
    ``F_g = u`` moves with the state, so the adjoint's right-hand side
    at the returned iterate is not the fixed point's.  The bound's
    per-constant probe sees that where a curvature of ``F`` in the state
    alone would read zero.  On this map the secant is exact (the change
    is linear in the distance), so the ratio is the two conservative
    factors alone and does not move with the cap: measured 1.814 at
    every cap from 2 to 14 (jaxlib 0.11.0).
    """
    _, exact = _analytic_curved("affine")
    for m in (3, 6):
        gm = _curved_graph("affine", max_iterations=m)
        gm.step()
        d = gm.coupling_diagnostics()["a+b"]
        assert d["iterations"] == m and d["converged"] is False, d
        g_k = _gradients(functools.partial(_curved_graph, "affine"), max_iterations=m)
        assert float(g_k["a"]) == pytest.approx(exact["a"], rel=1e-5), (
            "fixture premise: the additive parameter's gradient is exact")
        true = abs(float(g_k["g"]) - exact["g"]) / abs(float(g_k["g"]))
        assert true > 0.5, "fixture premise: the multiplicative one is far off"
        ratio = d["gradient_relative_error_bound"] / true
        assert 1.0 <= ratio <= 1.814 * 1.1, (
            f"m={m}: bound / true = {ratio:.4f}; recorded 1.814")


# ---------------------------------------------------------------------------
# The distance is the spectral bound's
# ---------------------------------------------------------------------------

#: The hidden-slow-mode map with a concave slow mode: ``F'(x*) = 0.99866``.
_HIDDEN_Q = -0.02


# Slow-marked: its reference is two gradients through 20 000 passes, 10-12 s
# on the CI runner (see the sweep above).
@pytest.mark.slow
def test_the_gradient_bound_takes_its_distance_from_the_spectral_bound():
    """On a hidden slow mode the distance is what decides the bound.

    ``error_estimate`` reads the fast mode's rate off the residual
    sequence and puts the forward ~100x closer to the fixed point than
    it is (``converged=True`` all the same).  The slow mode is where the
    curvature is, so the gradient is ~25% off.  The bound holds because
    its distance is ``spectral_error_bound``, which sees the slow mode;
    the same arithmetic with ``error_estimate`` as the distance -- the
    bound is linear in it -- would read two orders of magnitude short.
    That counterfactual is computed here so the fixture's premise is on
    the record, not implied.
    """
    q = _HIDDEN_Q
    gm = _two_mode_graph(q)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    distance = _two_mode_distance(gm, q)
    assert d["converged"] is True and d["gradient_bound_usable"] is True, d
    assert distance > 50 * d["error_estimate"], (
        "fixture premise: error_estimate understates the distance", d, distance)
    g_k = float(_gradients(functools.partial(_two_mode_graph, q))["c_slow"])
    g_star = _two_mode_reference(q)
    true = abs(g_k - g_star) / abs(g_k)
    bound = d["gradient_relative_error_bound"]
    assert true <= bound, (
        f"the gradient is {true:.3e} off and the bound reads {bound:.3e}")
    with_estimate = bound * d["error_estimate"] / d["spectral_error_bound"]
    assert with_estimate < true / 10, (
        f"fixture premise: with error_estimate as its distance the bound "
        f"would read {with_estimate:.3e} against a true {true:.3e}")


# ---------------------------------------------------------------------------
# Outside the leading-order regime: the Kantorovich check
# ---------------------------------------------------------------------------


class _StartedSquare(SimulationNode):
    """``x <- a + g u**2`` (or the relay ``x <- u``) started at ``x0``."""

    def __init__(self, name, a, g, x0, relay=False):
        super().__init__(name=name, timestep=1.0, a=jnp.float32(a), g=jnp.float32(g))
        self._x0 = x0
        self._relay = relay

    def initial_state(self):
        return {"x": jnp.float32(self._x0)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs["u"]
        return {"x": u if self._relay else p["a"] + p["g"] * u * u}


#: ``F'(x*) = 0.99``: the gap ``1 - F'`` is 0.01 at the fixed point and
#: 0.055 at 4.5% short of it, so the resolvent the adjoint uses there is
#: 5.5x smaller than the fixed point's -- the leading-order term is not
#: the whole error, and the uncorrected bound read 0.20-0.96x it.
_STIFF_SLOPE, _STIFF_G = 0.99, 0.25
_STIFF_A = (1.0 - (1.0 - _STIFF_SLOPE) ** 2) / (4.0 * _STIFF_G)


def _stiff_graph(x0, cap):
    gm = GraphManager()
    gm.add_node(_StartedSquare("a", _STIFF_A, _STIFF_G, x0))
    gm.add_node(_StartedSquare("b", _STIFF_A, _STIFF_G, x0, relay=True))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(["a", "b"], diagnostics=True, max_iterations=cap, tolerance=1e-9)
    gm.compile()
    return gm


def _stiff_fixed_point():
    a32, g32 = float(jnp.float32(_STIFF_A)), float(jnp.float32(_STIFF_G))
    x = (1.0 - math.sqrt(1.0 - 4.0 * g32 * a32)) / (2.0 * g32)
    for _ in range(50):
        x -= (a32 + g32 * x * x - x) / (2.0 * g32 * x - 1.0)
    den = 1.0 - 2.0 * g32 * x
    return x, {"a": 1.0 / den, "g": x * x / den}


@pytest.mark.parametrize("start,cap,certified", [
    (-0.05, 3, False),      # 4.5% short: h ~ 0.58
    (-0.01, 3, False),      # 1.0% short: h ~ 0.55
    (-0.05, 100, True),     # 0.70% short: h ~ 0.49
    (-0.01, 30, True),      # 0.65% short: h ~ 0.48
])
def test_the_gradient_bound_is_unusable_where_kantorovich_fails_and_holds_where_it_passes(
    start, cap, certified,
):
    """``h = amp * L * ||delta|| < 1/2``, or no bound.

    Newton-Kantorovich is the check that the linearisation at the
    returned iterate says anything about the fixed point: below one half
    it bounds the resolvent there (``amp / sqrt(1 - 2h)``) and the
    distance, above it nothing measured at ``x_k`` does.  So the bound
    is ``inf`` and unusable where it fails, and where it passes it holds
    -- on the four points where the uncorrected bound read 0.20-0.96x
    the true relative error with the flag ``True``.
    """
    x_star, exact = _stiff_fixed_point()
    x0 = float(jnp.float32(x_star * (1.0 + start)))
    gm = _stiff_graph(x0, cap)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    if not certified:
        assert math.isinf(d["gradient_relative_error_bound"]), d
        assert d["gradient_bound_usable"] is False, d
        return
    assert d["gradient_bound_usable"] is True, d
    g_k = _gradients(functools.partial(_stiff_graph, x0, cap))
    true = max(abs(float(g_k[p]) - exact[p]) / abs(float(g_k[p])) for p in ("a", "g"))
    assert true > 0.1, "fixture premise: the gradient is visibly off"
    assert d["gradient_relative_error_bound"] >= true, (
        f"bound {d['gradient_relative_error_bound']:.3f} under the true "
        f"relative error {true:.3f}"
    )


# ---------------------------------------------------------------------------
# Where nothing was computed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("label,group_kw", [
    ("fori", dict(solver="fori")),
    ("ift without diagnostics", dict(diagnostics=False)),
    ("max_iterations=1", dict(max_iterations=1, tolerance=1e3)),
])
def test_the_gradient_bound_reads_nan_and_unusable_where_nothing_was_computed(
    label, group_kw,
):
    """NaN with ``gradient_bound_usable=False``, never a plausible number.

    ``"fori"`` differentiates through its iterates and has no IFT
    linearisation to be wrong about; ``diagnostics=False`` is not charged
    the Jacobian-vector products; a cap of one solves no fixed point.
    ``0.0`` in the float slot would read as "the gradient is exact",
    which is the one thing none of these can know.
    """
    gm = _curved_graph("log", **group_kw)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert math.isnan(d["gradient_relative_error_bound"]), (label, d)
    assert d["gradient_bound_usable"] is False, (label, d)


def test_before_a_step_and_after_reset_state_the_gradient_bound_is_nan():
    """Seeded NaN by ``compile()``, and put back to NaN, not 0.0, by a reset.

    Neither is reported -- a group that has not stepped has no entry --
    so the slot is read where the next step's carry will find it.
    """
    gm = _curved_graph("log", max_iterations=4)
    slot = "coupling_a+b_gradient_relative_error_bound"
    assert "a+b" not in gm.coupling_diagnostics()
    assert math.isnan(float(gm._state["_meta"][slot]))
    gm.step()
    assert gm.coupling_diagnostics()["a+b"]["gradient_bound_usable"] is True
    gm.reset_state()
    assert "a+b" not in gm.coupling_diagnostics()
    assert math.isnan(float(gm._state["_meta"][slot]))


# MADD-ANO-005 quotes the key set ``coupling_diagnostics()`` reports.  That
# quotation is read back and compared with a step of ``_curved_graph`` in
# tests/compliance/test_registry_quotes_the_diagnostics_keys.py: the
# registry is under docs/, and a docs-only change runs the compliance job
# alone, so a test that reads it has to live there.
