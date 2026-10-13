"""Each factor of the numbers ``coupling_diagnostics()`` reports is carried, one test each.

The targeted search (``tests/property/test_coupling_targeted_search.py``)
scores the reported numbers against exact answers, and a fault it cannot
see is one whose factor the drawn examples leave headroom for: the float
floor has 8 to 25 times what those relays' rounding reaches, either half
of the bound's ``max`` carries it alone, and so on.  This module holds
the factors to their documented values directly:

* the floor's unit, its evaluation count (in the report and in the step)
  and the gradient bound's use of it;
* the bound's rate ``1 / (1 - rho_safe)`` where the resolvent factor is
  the smaller;
* the part of a residual outside an invariant Krylov space;
* the products' rounding in a group whose fields differ in dtype;
* how many interface scalars a settled spectrum can have;
* the two circles of the rounding certificate;
* the tangent of a sub-cycled member's interpolated boundary value.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _coupled_block
from maddening.core.coupling.acceleration import (
    SPECTRAL_KRYLOV_STEPS,
    SPECTRAL_SETTLED_FRACTION,
    _rounding_keeps_the_radius,
    arnoldi_spectral_radius,
    residual_precision_floor,
    spectral_error_bound,
    spectral_rate_settled,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.node import SimulationNode
from tests.property import coupled_graphs as cg
from tests.property import coupled_topologies as ct
from tests.property.sysid_transform_grid import precision

EPS32 = float(np.finfo(np.float32).eps)


class _Affine(SimulationNode):
    """``x <- G @ u + b`` in its own dtype, declaring ``evaluations`` per update."""

    def __init__(self, name, G, b, x0, dtype, evaluations=1.0):
        super().__init__(name, 1.0, G=jnp.asarray(G, dtype), b=jnp.asarray(b, dtype))
        self._dtype = dtype
        self._x0 = jnp.asarray(x0, dtype)
        self._evaluations = evaluations

    def initial_state(self):
        return {"x": self._x0}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = jnp.asarray(boundary_inputs["u"]).astype(self._dtype)
        return {"x": (p["G"] @ u + p["b"]).astype(self._dtype)}

    def update_evaluations(self):
        return self._evaluations


def _pair(Ga, Gb, ba, bb, xa, xb, *, dtypes=(jnp.float32, jnp.float32), evaluations=1.0,
          **knobs):
    """One step of the pair ``a <- Ga b + ba``, ``b <- Gb a + bb``; ``(report, gm)``."""
    gm = GraphManager()
    gm.add_node(_Affine("a", Ga, ba, xa, dtypes[0], evaluations))
    gm.add_node(_Affine("b", Gb, bb, xb, dtypes[1], evaluations))
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    gm.add_coupling_group(["a", "b"], diagnostics=True, **knobs)
    gm.compile()
    gm.step()
    return gm.coupling_diagnostics()["a+b"], gm


# ---------------------------------------------------------------------------
# The floor
# ---------------------------------------------------------------------------


def test_the_floor_is_four_units_of_eps_per_entry():
    """``4 eps sqrt(n)`` under ``"l2"`` and ``4 eps / rtol`` under ``"mixed"``.

    The number, not the constant's name: a floor of two units passes every
    test that multiplies by ``PRECISION_FLOOR_ULPS``.
    """
    state = {"n": {"x": jnp.asarray([2.0, -1.0, 0.5], jnp.float32)}}
    l2 = float(residual_precision_floor(state, ["n"], "l2"))
    mixed = float(residual_precision_floor(state, ["n"], "mixed", rtol=1e-3))
    assert l2 == pytest.approx(4.0 * EPS32 * np.sqrt(3.0), rel=1e-6), l2 / EPS32
    assert mixed == pytest.approx(4.0 * EPS32 / 1e-3, rel=1e-6), mixed / EPS32


def _stalled_scalar_pair(evaluations, **knobs):
    """A float32 scalar pair started on its fixed point: ``x = 0.5 x + 1`` at 2."""
    half, one, two = np.array([[0.5]]), np.array([1.0]), np.array([2.0])
    return _pair(half, half, one, one, two, two, evaluations=evaluations,
                 iteration_mode="jacobi", max_iterations=5, **knobs)[0]


@pytest.mark.parametrize("knobs", [
    dict(convergence_norm="l2", tolerance=1e-6),
    dict(convergence_norm="interface", rtol=1e-4),
], ids=["l2", "interface"])
def test_a_declared_evaluation_count_multiplies_the_floor_in_every_number_built_on_it(knobs):
    """Three evaluations per update: three times the floor, the bound and the gradient bound.

    At a stalled iterate (``residual == 0.0``) ``spectral_error_bound`` is
    the floor times its factor, and the gradient bound's distance is the
    floor's; the two groups differ in the declared count alone.  Under
    ``"l2"`` the report takes the floor from the function of the returned
    state, under ``"interface"`` from the step's own measurement: both
    carry the count.
    """
    one = _stalled_scalar_pair(1.0, **knobs)
    three = _stalled_scalar_pair(3.0, **knobs)
    for d in (one, three):
        assert d["residual"] == 0.0 and d["precision_limited"] and d["spectral_usable"], dict(d)
    assert three["spectral_error_bound"] == pytest.approx(
        3.0 * one["spectral_error_bound"], rel=1e-5), (dict(one), dict(three))
    assert one["gradient_relative_error_bound"] > 0, dict(one)
    assert three["gradient_relative_error_bound"] == pytest.approx(
        3.0 * one["gradient_relative_error_bound"], rel=1e-3), (dict(one), dict(three))


def test_the_report_multiplies_a_floor_the_step_measured_by_the_declared_count():
    """Behind a mapped edge the step measures the floor per evaluation and the report counts it.

    Where the interface norm reads an edge with a mapping, the floor
    depends on the weights the step ran with, so the step stores it per
    evaluation (``coupling_<key>_reading_floor``) and the report
    multiplies by the pass's count: the other branch of the report from
    the one the unmapped groups above take.
    """
    half, one, two = np.array([[0.5]]), np.array([1.0]), np.array([2.0])

    def stalled(evaluations):
        gm = GraphManager()
        gm.add_node(_Affine("a", half, one, two, jnp.float32, evaluations))
        gm.add_node(_Affine("b", half, one, two, jnp.float32, evaluations))
        gm.add_edge("b", "a", "x", "u", mapping=matrix_mapping(np.ones((1, 1), np.float32)))
        gm.add_edge("a", "b", "x", "u")
        gm.add_coupling_group(["a", "b"], diagnostics=True, iteration_mode="jacobi",
                              max_iterations=5, convergence_norm="interface", rtol=1e-4)
        gm.compile()
        gm.step()
        unit = float(gm._state["_meta"]["coupling_a+b_reading_floor"])      # noqa: SLF001
        return gm.coupling_diagnostics()["a+b"], unit

    (one_eval, unit_one), (three_evals, unit_three) = stalled(1.0), stalled(3.0)
    # The premise: the step measured the floor, the same per evaluation in both.
    assert np.isfinite(unit_one) and unit_one > 0.0 and unit_three == unit_one
    for d in (one_eval, three_evals):
        assert d["residual"] == 0.0 and d["precision_limited"], dict(d)
    assert three_evals["spectral_error_bound"] == pytest.approx(
        3.0 * one_eval["spectral_error_bound"], rel=1e-5), (dict(one_eval), dict(three_evals))


def test_the_gradient_bound_carries_the_floor_along_the_step_and_in_any_direction():
    """Just above the floor the gradient bound is ``A (r + m f) + B m f``, both parts present.

    ``x = 0.5 x + 1`` twice over, started ``2**-16`` (relatively) from
    its fixed point and stopped after two Jacobi passes: the residual
    ``r`` is four float floors ``f``, so the iterate is resolved
    and the floor is a part of the distance, not the whole of it.  The
    solve does not read the declared evaluation count ``m``; the floor
    is ``m f``.  The bound is then affine in ``m``:

    * the distance along the Newton step is ``r + m f`` through the
      resolvent (``A``), so the intercept is ``A r`` and that part of
      the slope ``A f``;
    * the floor has no known direction, and is taken through the
      resolvent in the worst one (``B m f``); on this pair ``B = sqrt(2) A``.

    So ``slope / (intercept f / r)`` is ``1 + sqrt(2)``.  With the floor
    left out of the distance it is ``sqrt(2)``, with the undirected part
    left out above the floor it is 1, and with no floor at all 0.
    """
    half, one = np.array([[0.5]]), np.array([1.0])
    start = np.array([2.0 * (1.0 + 2.0 ** -16)])
    bounds = []
    for evaluations in (1.0, 2.0, 3.0):
        d, gm = _pair(half, half, one, one, start, start, evaluations=evaluations,
                      iteration_mode="jacobi", max_iterations=2, convergence_norm="l2",
                      tolerance=1e-12)
        floor = float(residual_precision_floor(gm._state, ["a", "b"], "l2"))  # noqa: SLF001
        assert d["residual"] == pytest.approx(4.0 * floor, rel=1e-4), dict(d)
        assert d["gradient_bound_usable"] and not d["precision_limited"], dict(d)
        bounds.append(d["gradient_relative_error_bound"])
    slope = bounds[1] - bounds[0]
    assert bounds[2] - bounds[1] == pytest.approx(slope, rel=1e-4), bounds
    intercept = bounds[0] - slope
    assert intercept > 0.0, bounds
    assert slope / (intercept / 4.0) == pytest.approx(1.0 + np.sqrt(2.0), rel=1e-3), bounds


# ---------------------------------------------------------------------------
# The bound's two factors
# ---------------------------------------------------------------------------


def test_the_bound_uses_the_rate_where_it_exceeds_the_resolvent_factor():
    """``1 / (1 - rho_safe)`` with the resolvent factor at one: each half of the ``max`` alone."""
    assert float(spectral_error_bound(1.0, 0.9, 0.0, 1.0)) == pytest.approx(10.0, rel=1e-6)
    # ``rho_safe = rho + 2 * arnoldi_residual``.
    assert float(spectral_error_bound(1.0, 0.8, 0.05, 1.0)) == pytest.approx(10.0, rel=1e-6)
    assert float(spectral_error_bound(1.0, 0.5, 0.0, 7.0)) == pytest.approx(7.0, rel=1e-6)


def test_a_part_of_the_residual_outside_the_space_is_reported_above_rounding():
    """A residual 1e-6 outside an invariant space is unresolved where ``1 - rho`` is 1e-6.

    Seven modes and a start with one component in the null space: the
    Krylov space is eight-dimensional and breaks down on its last step.
    The residual handed in has a part of 1e-6 of its norm along a second
    null direction, which the space never takes in.  That part is eleven
    orders above float64 rounding, and at ``rho = 1 - 1e-6`` it is twenty
    times the settle margin: it must be reported, not rounded away at a
    fixed 1e-5.
    """
    with precision(True):
        lam = np.zeros(10)
        lam[:7] = (1.0 - 1e-6) * np.array([1.0, -0.9, 0.8, -0.7, 0.6, -0.5, 0.4])
        A = jnp.asarray(np.diag(lam))
        v0 = np.ones(10)
        v0[8:] = 0.0                       # one null direction in the start
        extra = np.zeros(10)
        extra[:7] = 1.0 / np.sqrt(7.0)
        extra[9] = 1e-6                    # the other null direction, in the residual alone
        rho, residual, _amp = arnoldi_spectral_radius(
            lambda v: A @ v, jnp.asarray(v0), v_extra=jnp.asarray(extra))
        assert float(rho) == pytest.approx(1.0 - 1e-6, abs=1e-12)
        assert float(residual) == pytest.approx(1e-6, rel=1e-3), float(residual)
        assert not bool(spectral_rate_settled(rho, residual))


# ---------------------------------------------------------------------------
# The products' rounding in a mixed-dtype group
# ---------------------------------------------------------------------------


def test_a_float32_member_beside_a_float64_one_settles_at_float32_rounding():
    """The breakdown test is scaled to the coarsest field's ``eps``, not the flat vector's.

    Under x64 a float32 member beside a float64 one is analysed in
    float64, and its products round at float32.  Six entries a side under
    Gauss-Seidel: the pass has rank six in a state of twelve, so the
    Krylov space closes at seven vectors and what is left of the eighth
    product is float32 rounding -- eight orders above float64's.  Held to
    float64's ``eps`` it is taken for a direction, the space is still
    "growing" at the cap, and the spectrum reads unsettled.
    """
    rng = np.random.default_rng(3)
    Qa, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    Qb, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    Ga, Gb = 0.8 * Qa, 0.75 * Qb           # the pass's radius is 0.6
    with precision(True):
        d, _gm = _pair(Ga, Gb, np.ones(6), np.ones(6), np.zeros(6), np.zeros(6),
                       dtypes=(jnp.float32, jnp.float64), iteration_mode="gauss-seidel",
                       convergence_norm="l2", tolerance=1e-5, max_iterations=200)
    assert d["converged"], dict(d)
    assert d["rho_spectral"] == pytest.approx(0.6, abs=1e-5), dict(d)
    assert d["spectral_usable"], dict(d)


def test_the_resolvent_factor_of_a_mixed_dtype_group_is_its_float64_twins():
    """A float32 member's rounding is not a direction of the compressed Jacobian.

    Three entries a side under Gauss-Seidel: the pass has rank three, the
    Krylov space closes at four vectors, and what is left of the next
    product is the float32 member's rounding.  Held to the analysis
    dtype's ``eps`` (float64's) instead of the coarsest field's, it is
    kept as a direction and enters the compressed operator: the factor
    the bound applies read 3.544 for 2.784.  The same pair with both
    members in float64 has no such leftover and reads 2.787 (the two
    stop a float32 rounding apart, hence the 0.1%).
    """
    rng = np.random.default_rng(3)
    Qa, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Qb, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    # Representable in float32, so the two pairs are the same map.
    Ga, Gb = (0.8 * Qa).astype(np.float32), (0.75 * Qb).astype(np.float32)
    factor = {}
    with precision(True):
        for first in (jnp.float32, jnp.float64):
            d, gm = _pair(Ga, Gb, np.ones(3), np.ones(3), np.zeros(3), np.zeros(3),
                          dtypes=(first, jnp.float64), iteration_mode="gauss-seidel",
                          convergence_norm="l2", tolerance=1e-5, max_iterations=200)
            assert d["converged"] and d["spectral_usable"], dict(d)
            factor[first] = float(
                gm._state["_meta"]["coupling_a+b_spectral_amplification"])   # noqa: SLF001
    assert factor[jnp.float32] == pytest.approx(factor[jnp.float64], rel=0.01), factor


# ---------------------------------------------------------------------------
# How many interface scalars a settled spectrum can have
# ---------------------------------------------------------------------------


def _pair_topology(n):
    b = ct.TopologyBuilder()
    b.node("a", n, alpha=0.0)
    b.node("b", n, alpha=0.0)
    b.edge("b", "a")
    b.edge("a", "b")
    b.group("a", "b")
    return b.build(f"pair-{n}")


#: ``(entries a side, schedule) -> (rank of the pass, entries of the state, settled)``.
#: Under Gauss-Seidel the pass of a pair has rank ``n`` in a state of
#: ``2 n``; under Jacobi rank ``2 n``, the whole state.
BOUNDARY = {
    (7, "gauss-seidel"): (7, 14, True),
    (8, "gauss-seidel"): (8, 16, False),
    (4, "jacobi"): (8, 8, True),
    (5, "jacobi"): (10, 10, False),
}


@pytest.mark.parametrize("n,mode", sorted(BOUNDARY))
def test_a_spectrum_settles_up_to_seven_scalars_and_eight_where_they_are_the_state(n, mode):
    """Eight Krylov vectors hold the start and seven more: the boundary, both sides.

    The Krylov space is grown from a start that is not in the Jacobian's
    range, so it closes once it holds the start *and* the range: a pass of
    rank seven in a larger state settles, one of rank eight does not
    (``spectral_usable=False``, the space still growing at the cap).
    Where the state is exactly the interface scalars the start is in the
    span, and eight settle.  ``rho_spectral`` of a settled one is the
    radius.
    """
    rank, size, settled = BOUNDARY[(n, mode)]
    topo = _pair_topology(n)
    knobs = cg.live_knobs(dict(acceleration="none", iteration_mode=mode, convergence_norm="l2",
                               tolerance=1e-10, solver="ift", diagnostics=True,
                               max_iterations=200))
    cfgs = ct.group_cfgs_of([knobs])
    with precision(True):
        built = ct.build(topo, knobs, dtype="float64", mapping_kind="matrix")
        values = ct.draw_values(topo, np.random.default_rng(n), 0.5, nonnormal=False,
                                dtype="float64", group_cfgs=cfgs, mapping_kind="matrix")
        (step,) = ct.run(built, values, 1)
    d = step.reports[0]
    model = ct.LinearModel(topo, values, dtype="float64", group_cfgs=cfgs)
    L, U = model.group_pass(0)
    J = np.linalg.solve(np.eye(L.shape[0]) - np.asarray(L, np.float64), np.asarray(U, np.float64))
    assert (np.linalg.matrix_rank(J), J.shape[0]) == (rank, size)
    assert rank <= SPECTRAL_KRYLOV_STEPS or not settled
    assert d["converged"] and bool(d["spectral_usable"]) is settled, dict(d)
    if settled:
        radius = float(np.max(np.abs(np.linalg.eigvals(J))))
        assert d["rho_spectral"] == pytest.approx(radius, abs=1e-9), (dict(d), radius)


# ---------------------------------------------------------------------------
# The rounding certificate
# ---------------------------------------------------------------------------


def _padded(H2, defect2):
    k = SPECTRAL_KRYLOV_STEPS
    H = np.zeros((k, k))
    H[:2, :2] = H2
    d = np.zeros(k)
    d[:2] = defect2
    active = np.zeros(k)
    active[:2] = 1.0
    return jnp.asarray(H), jnp.asarray(d), jnp.asarray(active)


def test_the_certificate_refuses_a_radius_rounding_can_move_inward():
    """Both circles are tested: the one about the dominant eigenvalue catches a fall.

    ``[[0.5, 1000], [e, 0.3]]`` has eigenvalues ``0.4 +/- sqrt(0.01 + 1000
    e)``: at ``e = -1.2e-5`` they meet and leave the real axis at modulus
    0.4025, a fall of 0.0975 from 0.5, while ``e = +1.2e-5`` raises the
    radius by 0.048 only.  With a margin of 0.08 no perturbation of that
    size puts an eigenvalue on ``|z| = 0.58``; one crosses the circle of
    radius 0.08 about 0.5.  A rounding of 1e-9 moves nothing.
    """
    with precision(True):
        H2 = np.array([[0.5, 1000.0], [0.0, 0.3]])
        rho, margin = jnp.asarray(0.5), jnp.asarray(0.08)
        fall = np.max(np.abs(np.linalg.eigvals(H2 + np.array([[0.0, 0.0], [-1.2e-5, 0.0]]))))
        rise = np.max(np.abs(np.linalg.eigvals(H2 + np.array([[0.0, 0.0], [1.2e-5, 0.0]]))))
        assert 0.5 - fall > 0.08 > rise - 0.5 > 0, (fall, rise)
        assert not bool(_rounding_keeps_the_radius(*_padded(H2, [0.0, 1.2e-5]), rho, margin))
        assert bool(_rounding_keeps_the_radius(*_padded(H2, [0.0, 1e-9]), rho, margin))
        assert bool(_rounding_keeps_the_radius(*_padded(H2, [0.0, 0.0]), rho, margin))


def test_the_certificate_holds_a_graded_matrix_to_its_own_rows_rounding():
    """Componentwise: a large entry's row may round at its size without moving the radius.

    The same matrix with the rounding on the *first* row (the row of the
    entry 1000, where a product rounds coarsely): ``E`` reaches the
    diagonal and the large entry only, the eigenvalues move by its size,
    and the certificate holds.  A normwise perturbation of that size on
    the second row is the one refused above.
    """
    with precision(True):
        H2 = np.array([[0.5, 1000.0], [0.0, 0.3]])
        assert bool(_rounding_keeps_the_radius(
            *_padded(H2, [1.2e-5, 0.0]), jnp.asarray(0.5), jnp.asarray(0.08)))


# ---------------------------------------------------------------------------
# The interpolated boundary value of a sub-cycled member
# ---------------------------------------------------------------------------


def test_the_interpolated_boundary_value_keeps_its_bits_and_has_an_exact_tangent():
    """``a + alpha (b - a)`` to the bit; its tangent ``(1 - alpha) a_dot + alpha b_dot``.

    At the last sub-step (``alpha = 1``) the tangent is ``b_dot`` alone,
    however small beside ``a_dot``: by the chain rule it was ``a_dot + (b_dot
    - a_dot)``, which rounds ``b_dot`` at one ``eps`` of ``a_dot`` and read
    ``0.0`` here.
    """
    rng = np.random.default_rng(0)
    a = jnp.asarray(rng.normal(size=5), jnp.float32)
    b = jnp.asarray(rng.normal(size=5), jnp.float32)
    for alpha in (0.25, 0.5, 1.0):
        al = jnp.float32(alpha)
        value = _coupled_block._interpolated(a, b, al)
        assert np.array_equal(np.asarray(value), np.asarray(a + al * (b - a)))
    a_dot = jnp.ones(5, jnp.float32)
    b_dot = jnp.full(5, 1e-9, jnp.float32)
    one = jnp.float32(1.0)
    _, last = jax.jvp(lambda x, y: _coupled_block._interpolated(x, y, one), (a, b), (a_dot, b_dot))
    assert np.array_equal(np.asarray(last), np.asarray(b_dot)), np.asarray(last)
    half = jnp.float32(0.5)
    _, mid = jax.jvp(lambda x, y: _coupled_block._interpolated(x, y, half), (a, b), (a_dot, b_dot))
    assert np.allclose(np.asarray(mid), 0.5 * (1.0 + 1e-9), rtol=1e-6)
    # Reverse mode is the transpose of the same rule, and the interpolation
    # weight is differentiable.
    grad_b = jax.grad(lambda y: jnp.sum(_coupled_block._interpolated(a, y, one)))(b)
    assert np.array_equal(np.asarray(grad_b), np.ones(5, np.float32))
    d_alpha = jax.grad(lambda t: jnp.sum(_coupled_block._interpolated(a, b, t)))(half)
    assert float(d_alpha) == pytest.approx(float(jnp.sum(b - a)), rel=1e-5)
    # An integer leaf is interpolated as it always was.
    i = _coupled_block._interpolated(jnp.asarray([1, 2]), jnp.asarray([3, 6]), half)
    assert np.array_equal(np.asarray(i), [2.0, 4.0])


def test_the_interpolated_boundary_value_can_be_differentiated_twice():
    """Forward over reverse, reverse over reverse and a Hessian through the tangent rule.

    The rule's tangent is ordinary arithmetic in the primals and the
    tangents, so JAX differentiates it again; a rule that closed over a
    tracer or stopped a gradient would fail here or read zero.  The
    second derivatives of ``a + alpha (b - a)``: zero in ``a`` and ``b``,
    and ``d2/(d alpha d b) = 1``, ``d2/(d alpha d a) = -1``.
    """
    a, b = jnp.asarray([1.0, -2.0], jnp.float32), jnp.asarray([0.5, 3.0], jnp.float32)
    alpha = jnp.asarray(0.25, jnp.float32)

    def value(a_, b_, t):
        return jnp.sum(_coupled_block._interpolated(a_, b_, t) ** 2)

    def exact(a_, b_, t):
        return jnp.sum((a_ + t * (b_ - a_)) ** 2)

    for transform in (lambda f: jax.jacfwd(jax.grad(f, argnums=(0, 1, 2)), argnums=(0, 1, 2)),
                      lambda f: jax.jacrev(jax.grad(f, argnums=(0, 1, 2)), argnums=(0, 1, 2)),
                      lambda f: jax.hessian(f, argnums=(0, 1, 2))):
        got, want = transform(value)(a, b, alpha), transform(exact)(a, b, alpha)
        for g_row, w_row in zip(got, want):
            for g, w in zip(g_row, w_row):
                np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=1e-6, atol=1e-6)
    mixed = jax.grad(lambda t: jax.grad(
        lambda b_: jnp.sum(_coupled_block._interpolated(a, b_, t)))(b)[0])(alpha)
    assert float(mixed) == 1.0

