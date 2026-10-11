"""What ``sharded_cg`` and ``sharded_gmres`` answer where a tolerance is
not a tolerance, where it is exactly zero, and where the solution does not
fit the dtype.  One rule for all three: ``converged`` is never True beside
a value that is not the solution, and a call that has no answer says so
with NaN in every entry of ``value``, a NaN ``residual_norm`` and
``converged=False``, on every route.

* **A tolerance that is not a number >= 0 (MADD-ANO-271).**  Given as a
  concrete value, ``rtol`` is refused as ``atol`` is (a ``ValueError``).
  Given as a traced value (an argument of a jitted or vmapped function) it
  cannot raise, and gets the NaN answer.  Until 0.4.0 a traced ``atol``
  and any ``rtol`` were passed on, and the lineax routes read
  ``converged=True`` on a value 72 % off, or on zeros.
* **A tolerance of zero.**  lineax tests ``|r_i| / (floor + rtol |b_i|)``
  against one; with a floor of zero that quotient is ``0 / 0`` at every
  entry where the residual and the right-hand side are both exactly zero,
  and lineax read the NaN as success.  ``rtol=0``, an explicit
  ``atol=0.0`` beside a right-hand side with zero entries, and an ``rtol``
  that underflows all got there.  Where the floor is zero ``converged`` is
  the statement of the loop backends,
  ``residual_norm <= max(atol, rtol * ||b||)``: for a tolerance of zero,
  ``residual_norm == 0``.
* **A solution that overflows the dtype (MADD-ANO-272).**  The solve runs
  on a rescaled right-hand side and is scaled back; the answer is tested
  after that, so infinities are not returned beside a finite norm and the
  frame's flag.
* **The unit of ``sharded_gmres``'s budget on the lineax route**: restart
  cycles, not steps.  Pinned as it is; registered open.

Systems: the 1-D Dirichlet Laplacian shifted on its diagonal, 64 unknowns,
float32.  No iteration count of a converged solve is pinned.  Where a flag
is asserted, the solve either stops for a structural reason (entries that
are exactly zero) or sits a measured factor away from its tolerance, which
is said beside the assertion; the tolerances asked of a solve that has to
converge are a hundred times or more above the float32 floor of the system
(``eps * kappa``, 1e-5 at the shift 0.05).
"""

# No ``from __future__ import annotations`` here: the device-policy test of
# this directory executes every test module without registering it in
# ``sys.modules``.
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.iterative_solver import (
    jacobi_preconditioner,
    sharded_cg,
    sharded_gmres,
)

N = 64
FLOAT32_MAX = float(np.finfo(np.float32).max)


def _operator(diagonal):
    def matvec(x):
        zero = jnp.zeros((1,), x.dtype)
        return (diagonal * x - jnp.concatenate([zero, x[:-1]])
                - jnp.concatenate([x[1:], zero]))
    return matvec


def _matrix(diagonal, n=N):
    return (np.diag(np.full(n, float(np.float32(diagonal))))
            - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1))


def _relative_error(got, want):
    got = np.asarray(got, np.float64)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


def _norm(b):
    return float(np.linalg.norm(np.asarray(b, np.float64)))


#: Every route of the two solvers.  ``"auto"`` with a preconditioner is the
#: loop backend.
ROUTES = {
    "cg default": (sharded_cg, {}),
    "cg lineax": (sharded_cg, {"backend": "lineax"}),
    "cg loop": (sharded_cg, {"backend": "loop"}),
    "cg auto with Jacobi": (sharded_cg, {"preconditioner": "jacobi"}),
    "gmres default": (sharded_gmres, {}),
    "gmres lineax": (sharded_gmres, {"backend": "lineax"}),
    "gmres loop": (sharded_gmres, {"backend": "loop"}),
    "gmres auto with Jacobi": (sharded_gmres, {"preconditioner": "jacobi"}),
}
#: The routes that read lineax's verdict.
LINEAX_ROUTES = ("cg default", "cg lineax", "gmres default", "gmres lineax")
_routes = pytest.mark.parametrize("route", list(ROUTES))
_ordinary_and_differentiable = pytest.mark.parametrize(
    "differentiable", [False, True], ids=["ordinary", "differentiable"])


def _call(route, diagonal, b, **kw):
    solver, selects = ROUTES[route]
    selects = dict(selects)
    if selects.get("preconditioner") == "jacobi":
        selects["preconditioner"] = jacobi_preconditioner(
            jnp.full((N,), diagonal, jnp.float32))
    return solver(_operator(diagonal), b, **selects, **kw)


def _fields(result):
    return result.value, result.converged, result.residual_norm


def _is_the_answer_of_no_answer(value, converged, residual_norm):
    return (bool(np.all(np.isnan(np.asarray(value))))
            and bool(np.all(np.isnan(np.asarray(residual_norm))))
            and not bool(np.any(np.asarray(converged))))


NOISE = np.random.default_rng(11).standard_normal(N).astype(np.float32)
#: One unit vector, and three entries: right-hand sides most of whose
#: entries are exactly zero (a local source behind a stencil).
UNIT = np.zeros(N, np.float32)
UNIT[0] = 1.0
THREE_ENTRIES = np.zeros(N, np.float32)
THREE_ENTRIES[[3, 20, 41]] = [1.0, -0.5, 0.25]


# ---------------------------------------------------------------------------
# MADD-ANO-272: a solution that overflows the dtype
# ---------------------------------------------------------------------------

#: Every entry a quarter of the largest float32.  Behind a diagonal of 2.05
#: the solution is about twenty times the right-hand side and overflows;
#: behind 10 it is an eighth of it and fits, while the 2-norm of the
#: right-hand side itself (twice the largest float32) does not.
HUGE = np.full(N, FLOAT32_MAX / 4, np.float32)
OVERFLOWS, FITS = 2.05, 10.0


@_ordinary_and_differentiable
@_routes
def test_a_solution_that_overflows_is_answered_with_nan(route, differentiable):
    """One jitted program per route.  The right-hand side whose solution
    overflows gets NaN in every entry, a NaN norm and ``converged=False``;
    the same right-hand side a thousand times smaller is solved by the
    same program (the rule does not answer NaN to everything).  Until the
    fix: 62 to 64 of the 64 entries infinite, a ``residual_norm`` of 1.5e33
    to 4.7e35, and ``converged=True`` at this ``rtol`` on every route."""
    solve = jax.jit(lambda b: _fields(_call(
        route, OVERFLOWS, b, rtol=1e-3, differentiable=differentiable)))

    assert _is_the_answer_of_no_answer(*solve(jnp.asarray(HUGE)))

    smaller = (HUGE / np.float32(1024)).astype(np.float32)
    value, _, residual_norm = solve(jnp.asarray(smaller))
    want = np.linalg.solve(_matrix(OVERFLOWS), smaller.astype(np.float64))
    assert np.all(np.isfinite(np.asarray(value)))
    assert _relative_error(value, want) < 1e-2
    # Not ``converged``: the lineax routes stop on an entrywise test that
    # allows twice the norm of the flag under ``differentiable=True``.
    assert float(residual_norm) < 3e-3 * _norm(smaller)


@pytest.mark.parametrize("route", ["cg default", "cg lineax", "cg loop", "gmres default"])
def test_the_overflow_gets_the_same_answer_when_the_call_is_not_jitted(route):
    """Eagerly the default CG route takes its branch in Python (lineax's
    value is finite in the frame, so the result is lineax's): it is the
    common exit that answers.  At the default ``rtol`` too, which float32
    does not meet on this system."""
    for tolerance in ({"rtol": 1e-3}, {}):
        result = _call(route, OVERFLOWS, jnp.asarray(HUGE), **tolerance)
        assert _is_the_answer_of_no_answer(*_fields(result)), tolerance


@pytest.mark.parametrize("route", ["cg default", "cg loop", "gmres default", "gmres loop"])
def test_under_differentiation_a_solution_that_overflows_has_no_derivative(route):
    """``differentiable=True``.  The tangent and cotangent solves have
    right-hand sides of their own (a direction of ones here), whose
    solutions fit: left alone they returned a finite derivative beside the
    value that is not one (measured before the fix: all 64 entries of the
    JVP and of the gradient finite).  NaN, in both modes; and the control a
    thousand times smaller has its derivative."""
    direction = jnp.ones((N,), jnp.float32)

    def value(b):
        return _call(route, OVERFLOWS, b, rtol=1e-3, differentiable=True).value

    tangent = jax.jit(lambda b: jax.jvp(value, (b,), (direction,))[1])
    gradient = jax.jit(jax.grad(lambda b: jnp.vdot(direction, value(b))))
    for derivative in (tangent, gradient):
        assert np.all(np.isnan(np.asarray(derivative(jnp.asarray(HUGE)))))
        smaller = derivative(jnp.asarray(HUGE / np.float32(1024)))
        want = np.linalg.solve(_matrix(OVERFLOWS), np.ones(N))
        assert _relative_error(smaller, want) < 1e-2


@_ordinary_and_differentiable
@_routes
def test_a_right_hand_side_whose_norm_overflows_is_solved_where_its_solution_fits(
        route, differentiable):
    """The part of the entry that 0.4.0 had resolved, and that testing the
    answer after it is scaled back must not take away.  The 2-norm of 64
    entries of a quarter of the largest float32 is twice that number;
    behind a diagonal of 10 the solution is an eighth of the right-hand
    side.  Measured within 2e-5 on every route, the residual at a tenth of
    ``rtol * ||b||`` or less.  In 0.3.0 and 0.3.1 the CG loop returned
    zeros with ``converged=True``, ``iters=0`` and an infinite
    ``residual_norm``, and the other routes NaN."""
    result = _call(route, FITS, jnp.asarray(HUGE), rtol=1e-4,
                   differentiable=differentiable)
    want = np.linalg.solve(_matrix(FITS), HUGE.astype(np.float64))
    assert bool(result.converged)
    assert np.all(np.isfinite(np.asarray(result.value)))
    assert np.isfinite(float(result.residual_norm))
    assert _relative_error(result.value, want) < 1e-3


#: Calls on the right-hand side whose norm overflows that leave it far from
#: solved, so that the norm of the residual overflows too when it is scaled
#: back, and tolerances wide enough to pass anything.
UNSOLVED = {
    "an infinite atol, one step": {"atol": float("inf"), "max_iters": 1},
    "an infinite rtol": {"rtol": float("inf")},
    "rtol 2, no step": {"rtol": 2.0, "max_iters": 0},
    "rtol 0.9, one step": {"rtol": 0.9, "max_iters": 1},
}


@pytest.mark.parametrize("case", list(UNSOLVED))
@_routes
def test_a_result_is_a_finite_value_beside_a_finite_norm_or_nan_in_both(route, case):
    """The general form, on what is RETURNED: every entry of ``value``
    finite beside a finite ``residual_norm``, or NaN in both with
    ``converged=False``.  Until the fix the norm was tested in the frame
    the solve ran in: ``sharded_cg`` given an infinite ``atol`` (default,
    lineax and loop) and ``sharded_gmres``'s default route returned
    ``residual_norm=inf`` beside ``converged=True``, and so did the CG loop
    at ``rtol=2`` with no step."""
    value, converged, residual_norm = _fields(_call(
        route, FITS, jnp.asarray(HUGE), **UNSOLVED[case]))
    if np.isfinite(float(residual_norm)):
        assert np.all(np.isfinite(np.asarray(value)))
    else:
        assert _is_the_answer_of_no_answer(value, converged, residual_norm)


@pytest.mark.parametrize("route", ["cg default", "cg lineax", "cg loop", "gmres default"])
def test_an_infinite_atol_does_not_read_converged_beside_a_norm_that_overflowed(route):
    """The measured case of that form, asserted by name: one step towards
    a right-hand side whose norm overflows leaves a residual whose norm
    overflows too, and an infinite ``atol`` passed it (``inf <= inf`` in
    the units of ``b``; in the frame both were finite)."""
    for differentiable in (False, True):
        result = _call(route, FITS, jnp.asarray(HUGE), atol=float("inf"), max_iters=1,
                       differentiable=differentiable)
        assert _is_the_answer_of_no_answer(*_fields(result)), differentiable


#: A solution of 0.7 of the largest float32: representable, and within a
#: factor of the diagonal (2.05) of the top.
NEAR_THE_TOP = np.full(N, FLOAT32_MAX / 28, np.float32)


@pytest.mark.parametrize("primal", ["plain", "with an atol", "with an x0"])
@_routes
def test_a_solution_just_under_the_top_of_the_dtype_is_answered(route, primal):
    """What the overflow rule must not take away: a solution that fits is
    returned, however close to the top, on every ordinary route (the
    solve and its report run in the frame of ``b``).  The same call under
    ``differentiable=True`` is the open case pinned in
    ``test_what_the_sharded_solvers_do_not_report_or_refuse_yet.py``."""
    want = np.linalg.solve(_matrix(OVERFLOWS), NEAR_THE_TOP.astype(np.float64))
    assert 0.6 * FLOAT32_MAX < np.abs(want).max() < 0.8 * FLOAT32_MAX
    kw = {"plain": {}, "with an atol": {"atol": 0.0},
          "with an x0": {"x0": 0.5 * jnp.asarray(want.astype(np.float32))}}[primal]
    result = _call(route, OVERFLOWS, jnp.asarray(NEAR_THE_TOP), rtol=1e-3, **kw)
    assert np.all(np.isfinite(np.asarray(result.value)))
    assert _relative_error(result.value, want) < 1e-2
    assert float(result.residual_norm) < 3e-3 * _norm(NEAR_THE_TOP)


@pytest.mark.parametrize("route", ["cg default", "cg loop", "gmres default", "gmres loop"])
def test_under_vmap_only_the_member_whose_solution_overflows_is_nan(route):
    batch = jnp.asarray(np.stack([HUGE, HUGE / np.float32(1024)]).astype(np.float32))
    value, converged, residual_norm = jax.vmap(
        lambda b: _fields(_call(route, OVERFLOWS, b, rtol=1e-3)))(batch)
    assert _is_the_answer_of_no_answer(value[0], converged[0], residual_norm[0])
    assert np.all(np.isfinite(np.asarray(value[1])))
    assert np.isfinite(float(residual_norm[1]))


# ---------------------------------------------------------------------------
# MADD-ANO-271: a tolerance that is not a number >= 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rtol", [float("nan"), -1e-6, -1.0, float("-inf"),
                                  np.float32("nan"), np.float64(-1e-3)],
                         ids=["nan", "-1e-6", "-1", "-inf", "numpy nan", "numpy -1e-3"])
@pytest.mark.parametrize("backend", ["auto", "lineax", "loop"])
@pytest.mark.parametrize("solver", [sharded_cg, sharded_gmres])
def test_a_nan_or_negative_rtol_is_refused(solver, backend, rtol):
    """As ``atol`` is, on every backend, before anything is solved.  Until
    the fix ``rtol`` was not read: a NaN gave zeros with ``converged=True``
    on the lineax routes, and a negative one was refused by lineax on its
    own routes only (the loops took it)."""
    with pytest.raises(ValueError, match=f"{solver.__name__}: rtol must be a number >= 0"):
        solver(_operator(2.5), jnp.asarray(NOISE), rtol=rtol, backend=backend)


@pytest.mark.parametrize("solver", [sharded_cg, sharded_gmres])
def test_a_concrete_rtol_is_refused_as_an_array_under_jit_and_with_the_flag(solver):
    """A JAX array that is not traced is a concrete value; a Python number
    closed over by a jitted function is one too (the refusal is raised
    when the function is traced); and ``differentiable=True`` passes the
    same door."""
    b = jnp.asarray(NOISE)
    with pytest.raises(ValueError, match="rtol must be a number >= 0"):
        solver(_operator(2.5), b, rtol=jnp.float32(-1.0))
    with pytest.raises(ValueError, match="rtol must be a number >= 0"):
        jax.jit(lambda rhs: solver(_operator(2.5), rhs, rtol=float("nan")).value)(b)
    with pytest.raises(ValueError, match="rtol must be a number >= 0"):
        solver(_operator(2.5), b, rtol=float("nan"), differentiable=True)
    with pytest.raises(ValueError, match="atol must be None"):
        solver(_operator(2.5), b, atol=float("nan"), differentiable=True)


@_routes
def test_infinity_is_a_tolerance(route):
    """What stays as it was.  Infinity is a number >= 0: with an infinite
    ``rtol`` or ``atol`` every residual passes, and the flag is True."""
    b = jnp.asarray(NOISE)
    for wide in ({"rtol": float("inf")}, {"atol": float("inf")}):
        result = _call(route, 2.5, b, **wide)
        assert bool(result.converged), wide
        assert np.all(np.isfinite(np.asarray(result.value))), wide
        assert np.isfinite(float(result.residual_norm)), wide


@pytest.mark.parametrize("atol", [None, 0.0, -0.0], ids=["None", "0.0", "-0.0"])
@_routes
def test_none_and_zero_are_the_relative_test(route, atol):
    """What stays as it was.  ``atol=None``, ``atol=0.0`` and
    ``atol=-0.0`` are the relative test, and a system float32 can solve to
    ``rtol=1e-3`` is solved (measured: the residual at 0.87 of the
    tolerance or less on the loop routes, where the flag is that
    comparison, and far under it on the lineax routes with a zero
    ``atol``, whose entrywise test without a floor is the stricter one)."""
    result = _call(route, 2.5, jnp.asarray(NOISE), rtol=1e-3, atol=atol)
    want = np.linalg.solve(_matrix(2.5), NOISE.astype(np.float64))
    assert bool(result.converged)
    assert _relative_error(result.value, want) < 1e-2


#: ``(rtol, atol)`` given as traced values: one of the two is not a number
#: >= 0.  The measured cases of the entry first.
NOT_TOLERANCES = {
    "atol -1": (1e-4, -1.0),
    "atol -1e-3": (1e-4, -1e-3),
    "atol NaN": (1e-4, float("nan")),
    "rtol NaN": (float("nan"), 1e-9),
    "rtol -1": (-1.0, 1e-9),
    "rtol -inf": (float("-inf"), 1e-9),
    "both": (float("nan"), -1.0),
}


@_ordinary_and_differentiable
@_routes
def test_a_traced_tolerance_that_is_not_a_number_is_answered_with_nan(route, differentiable):
    """One jitted program per route, the two tolerances its arguments.  A
    valid pair is solved by it; each pair of which one is negative or NaN
    gets NaN in every entry, a NaN norm and ``converged=False``.  Until the
    fix (measured on ``sharded_cg`` default and lineax): ``atol=-1.0``
    returned after one step, ``converged=True``, the value 72 % off;
    a NaN returned zeros with ``converged=True``."""
    b = jnp.asarray(NOISE)
    solve = jax.jit(lambda rtol, atol: _fields(_call(
        route, 2.5, b, rtol=rtol, atol=atol, differentiable=differentiable)))

    value, _, residual_norm = solve(jnp.float32(1e-3), jnp.float32(1e-9))
    want = np.linalg.solve(_matrix(2.5), NOISE.astype(np.float64))
    assert _relative_error(value, want) < 1e-2
    assert float(residual_norm) < 3e-3 * _norm(NOISE)
    for name, (rtol, atol) in NOT_TOLERANCES.items():
        assert _is_the_answer_of_no_answer(
            *solve(jnp.float32(rtol), jnp.float32(atol))), name


@_routes
def test_a_traced_rtol_alone_is_answered_the_same(route):
    """Without an ``atol`` the lineax routes derive their floor from
    ``rtol``; the answer does not depend on which of the two is bad."""
    b = jnp.asarray(NOISE)
    solve = jax.jit(lambda rtol: _fields(_call(route, 2.5, b, rtol=rtol)))
    assert np.all(np.isfinite(np.asarray(solve(jnp.float32(1e-3))[0])))
    for rtol in (float("nan"), -1e-3):
        assert _is_the_answer_of_no_answer(*solve(jnp.float32(rtol))), rtol


@_routes
def test_under_vmap_only_the_members_with_a_bad_tolerance_are_nan(route):
    b = jnp.asarray(NOISE)
    atols = jnp.asarray([1e-9, -1.0, float("nan"), 0.0], jnp.float32)
    value, converged, residual_norm = jax.vmap(
        lambda atol: _fields(_call(route, 2.5, b, rtol=1e-3, atol=atol)))(atols)
    for member, bad in enumerate((False, True, True, False)):
        if bad:
            assert _is_the_answer_of_no_answer(
                value[member], converged[member], residual_norm[member]), member
        else:
            assert np.all(np.isfinite(np.asarray(value[member]))), member
            assert bool(converged[member]), member


@pytest.mark.parametrize("route", ["cg default", "gmres loop"])
def test_under_vmap_with_the_flag_only_the_members_with_a_bad_tolerance_are_nan(route):
    """The same under ``differentiable=True``, where the tolerance the
    batch carries is closed over by ``lax.custom_linear_solve``'s solve."""
    b = jnp.asarray(NOISE)
    rtols = jnp.asarray([1e-3, float("nan"), -1e-3], jnp.float32)
    value, converged, residual_norm = jax.vmap(
        lambda rtol: _fields(_call(route, 2.5, b, rtol=rtol, differentiable=True)))(rtols)
    assert np.all(np.isfinite(np.asarray(value[0])))
    for member in (1, 2):
        assert _is_the_answer_of_no_answer(
            value[member], converged[member], residual_norm[member]), member


@pytest.mark.parametrize("route", ["cg default", "cg loop", "gmres default", "gmres loop"])
def test_under_differentiation_a_bad_traced_tolerance_has_no_derivative(route):
    """``differentiable=True``: the tangent and cotangent solves are
    purely relative and never see the call's ``atol``; they are refused
    with the call they belong to.  A derivative of NaN in both modes, for
    a bad ``atol`` as for a bad ``rtol``; a valid pair has its derivative."""
    b = jnp.asarray(NOISE)
    direction = jnp.ones((N,), jnp.float32)

    def value(rhs, rtol, atol):
        return _call(route, 2.5, rhs, rtol=rtol, atol=atol, differentiable=True).value

    tangent = jax.jit(lambda rtol, atol: jax.jvp(
        lambda rhs: value(rhs, rtol, atol), (b,), (direction,))[1])
    gradient = jax.jit(lambda rtol, atol: jax.grad(
        lambda rhs: jnp.vdot(direction, value(rhs, rtol, atol)))(b))
    want = np.linalg.solve(_matrix(2.5), np.ones(N))
    for derivative in (tangent, gradient):
        assert _relative_error(derivative(jnp.float32(1e-4), jnp.float32(1e-9)), want) < 1e-2
        for name in ("atol -1", "rtol NaN"):
            rtol, atol = NOT_TOLERANCES[name]
            assert np.all(np.isnan(np.asarray(
                derivative(jnp.float32(rtol), jnp.float32(atol))))), name


# ---------------------------------------------------------------------------
# A tolerance of zero
# ---------------------------------------------------------------------------

RIGHT_HAND_SIDES = {"one unit vector": UNIT, "white noise": NOISE,
                    "zeros": np.zeros(N, np.float32)}


@pytest.mark.parametrize("rhs", list(RIGHT_HAND_SIDES))
@_ordinary_and_differentiable
@_routes
def test_with_a_tolerance_of_zero_converged_is_a_residual_norm_of_zero(
        route, differentiable, rhs):
    """``rtol=0`` and no ``atol``: the test the call asks for is
    ``||b - A x|| <= 0``.  ``converged`` says exactly that of the value
    returned, on every route: a property with no tolerance in it.  (A zero
    right-hand side is solved by zeros and reads True; the GMRES loop
    solves the unit vector exactly in float32 and reads True too.)"""
    result = _call(route, 2.5, jnp.asarray(RIGHT_HAND_SIDES[rhs]), rtol=0.0,
                   differentiable=differentiable)
    assert bool(result.converged) == (float(result.residual_norm) == 0.0)


@_routes
def test_the_same_for_a_zero_tolerance_that_is_traced(route):
    """Zero is a number >= 0, so a traced zero is not answered with NaN:
    the call returns the value it reached, and the same property holds."""
    solve = jax.jit(lambda b, rtol, atol: _fields(_call(route, 2.5, b, rtol=rtol, atol=atol)))
    for rhs in (UNIT, NOISE):
        for atol in (0.0, -0.0):
            value, converged, residual_norm = solve(
                jnp.asarray(rhs), jnp.float32(0.0), jnp.float32(atol))
            assert np.all(np.isfinite(np.asarray(value)))
            assert bool(converged) == (float(residual_norm) == 0.0)


#: Calls whose floor for lineax's entrywise test is zero, on a right-hand
#: side with entries that are exactly zero.
ZERO_FLOORS = {
    "rtol=0": {"rtol": 0.0},
    "atol=0.0 beside the default rtol": {"atol": 0.0},
    "atol=0.0 beside rtol=1e-3": {"atol": 0.0, "rtol": 1e-3},
    "rtol=1e-44, which float32 flushes": {"rtol": 1e-44},
}


@pytest.mark.parametrize("floor", list(ZERO_FLOORS))
@pytest.mark.parametrize("route", LINEAX_ROUTES)
def test_lineax_stopping_on_zero_over_zero_is_not_reported_as_converged(route, floor):
    """The right-hand side is one unit vector.  After one CG step, or one
    GMRES cycle of two steps, every entry the Krylov space has not reached
    is exactly zero in the residual, in the right-hand side, in the update
    and in the iterate: lineax's quotients are ``0 / 0`` there, their
    maximum is NaN, and lineax stops and reports success.  The stop is
    structural (exact zeros, not rounding).  The value it stops on has a
    residual of 0.40 ``||b||`` (CG) and 0.17 (GMRES); until the fix that
    read ``converged=True``."""
    kw = dict(ZERO_FLOORS[floor])
    if route.startswith("gmres"):
        kw["restart"] = 2
    result = _call(route, 2.5, jnp.asarray(UNIT), **kw)
    assert float(result.residual_norm) > 0.1 * _norm(UNIT)
    assert not bool(result.converged)


@_routes
def test_an_explicit_atol_of_zero_is_the_relative_test_on_every_route(route):
    """``atol=0.0`` is a number >= 0 and means what ``None`` means for the
    flag, ``residual_norm <= rtol * ||b||``, on a right-hand side with
    zero entries too.  At ``rtol=1e-2`` the lineax routes stop on their
    ``0 / 0`` far above it (False); with the floor ``None`` gives them
    they solve it."""
    b = jnp.asarray(THREE_ENTRIES)
    bound = 1e-2 * _norm(THREE_ENTRIES)
    with_zero = _call(route, 2.05, b, rtol=1e-2, atol=0.0,
                      **({"restart": 2} if route.startswith("gmres") else {}))
    if bool(with_zero.converged):
        assert float(with_zero.residual_norm) <= 1.01 * bound
    if route in LINEAX_ROUTES:
        assert float(with_zero.residual_norm) > 10 * bound
        assert not bool(with_zero.converged)


@pytest.mark.parametrize("route", ["cg default", "cg lineax"])
def test_where_its_floor_is_positive_the_lineax_route_reports_lineaxs_verdict(route):
    """What the zero-floor rule must not touch: an ordinary call's flag on
    the lineax route is lineax's own verdict, not the loop backends'
    statement.  With the floor ``rtol * max|b|`` lineax's entrywise test
    lets each entry of the residual reach ``rtol * (max|b| + |b_i|)``; on
    a right-hand side of three entries that is a norm above
    ``rtol * ||b||``.  Measured (lineax 0.0.7 and 0.1.1): True at 1.46
    times ``rtol * ||b||``, where the loop backends' statement is False.
    ``rtol=1e-2`` is a thousand times the float32 floor of the system."""
    result = _call(route, 2.05, jnp.asarray(THREE_ENTRIES), rtol=1e-2)
    assert bool(result.converged)
    assert float(result.residual_norm) > 1.2 * 1e-2 * _norm(THREE_ENTRIES)


# ---------------------------------------------------------------------------
# The unit of ``sharded_gmres``'s budget on the lineax route
# ---------------------------------------------------------------------------


def _products_and_iters(backend, restart, max_iters):
    """Operator products of one ``sharded_gmres`` call, counted on the
    host, and the ``iters`` it reports.  A system GMRES(5) cannot solve in
    its budget: the shift 1e-3, condition number about 4000, 256 unknowns,
    white noise, the default ``rtol``."""
    applied = []

    def matvec(x):
        # The callback takes an entry of ``x`` so that it depends on the
        # operand and is executed once per product.
        jax.debug.callback(lambda _: applied.append(1), x[0])
        zero = jnp.zeros((1,), x.dtype)
        return (2.001 * x - jnp.concatenate([zero, x[:-1]])
                - jnp.concatenate([x[1:], zero]))

    b = jnp.asarray(np.random.default_rng(0).standard_normal(256).astype(np.float32))
    result = sharded_gmres(matvec, b, backend=backend, restart=restart, max_iters=max_iters)
    jax.effects_barrier()
    assert not bool(result.converged)
    return len(applied), int(result.iters)


@pytest.mark.parametrize("backend", ["auto", "lineax"])
def test_the_budget_and_the_count_of_gmres_on_lineax_are_in_restart_cycles(backend):
    """Pinned as it is (registered open; the docstring of ``max_iters``).
    The module hands lineax ``max_steps = max(4 * restart, max_iters)``,
    and lineax counts a step per restart CYCLE of ``restart`` products.
    With ``restart=5`` and ``max_iters=1`` that is 20: were it a budget in
    steps the call could execute some 20 products.  Measured, on lineax
    0.0.7 and 0.1.1: 116 products, ``iters=20``.  Asserted with room: more
    than twice a budget in steps, and at least ``restart`` products per
    unit of ``iters``."""
    restart, max_iters = 5, 1
    products, iters = _products_and_iters(backend, restart, max_iters)
    steps_handed_to_lineax = max(4 * restart, max_iters)
    assert products > 2 * steps_handed_to_lineax
    assert products > 10 * max_iters
    assert products >= restart * iters
    assert iters <= steps_handed_to_lineax


def test_the_loop_backend_bounds_the_work_of_gmres_by_its_budget():
    """What a caller who needs a bound on the work does: ``backend="loop"``
    runs ``max_iters`` steps rounded up to whole cycles (at least one)
    and reports the steps.  Each cycle is ``restart`` products and two for
    its residuals, after one product for the first residual."""
    for restart, max_iters, cycles in ((5, 1, 1), (5, 12, 3)):
        products, iters = _products_and_iters("loop", restart, max_iters)
        assert iters == cycles * restart
        assert products == cycles * (restart + 2) + 1
