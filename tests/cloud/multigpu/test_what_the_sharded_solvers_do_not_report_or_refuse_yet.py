"""Three things ``sharded_cg`` and ``sharded_gmres`` do not report or refuse
in 0.4.0, each pinned as it behaves today.  All three are registered open;
a fix turns the matching test red, and the entry is then closed with it.

* **MADD-ANO-270.**  Under ``differentiable=True`` the result's
  ``converged`` is the primal solve's.  Each derivative is one more solve
  of the same system, from zero, under the caller's ``rtol`` and
  ``max_iters``, and nothing reports whether THAT solve met its tolerance.
  A primal that needs fewer steps than a cold solve (a warm start, an
  ``atol`` above ``||b||``, a right-hand side that is one eigenvector)
  reads ``converged=True`` beside a derivative that is far off.
* **MADD-ANO-271.**  A tolerance that is not a number >= 0 is refused only
  where the module can read it: an ``atol`` given as a concrete value.  An
  ``atol`` that is traced (an argument of a jitted function) and an
  ``rtol`` of NaN are passed on, and the lineax routes then read
  ``converged=True`` on a value that is not the solution.
* **MADD-ANO-272.**  A solution too large for the dtype: the solve runs on
  a rescaled right-hand side and is scaled back afterwards, the scaling
  back overflows, and the result holds infinities beside a finite
  ``residual_norm`` and, where the rescaled solve met its tolerance,
  ``converged=True``.  (What 0.4.0 did fix there: a right-hand side whose
  NORM overflows while its solution fits is solved on every route; in
  0.3.0 and 0.3.1 the CG loop answered it with zeros and
  ``converged=True``, and the other routes with NaN.)

Systems: the 1-D Dirichlet Laplacian shifted on its diagonal, float32.  No
iteration count is pinned, and every tolerance asked of a solve that has
to converge is a hundred times or more above the float32 floor of its
system, so that the flags do not depend on the CPU.
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
    sharded_cg,
    sharded_gmres,
)


def _operator(diagonal):
    def matvec(x):
        zero = jnp.zeros((1,), x.dtype)
        return (diagonal * x - jnp.concatenate([zero, x[:-1]])
                - jnp.concatenate([x[1:], zero]))
    return matvec


def _matrix(diagonal, n):
    return (np.diag(np.full(n, float(np.float32(diagonal))))
            - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1))


def _relative_error(got, want):
    got = np.asarray(got, np.float64)
    return float(np.linalg.norm(got - want) / np.linalg.norm(want))


# ---------------------------------------------------------------------------
# MADD-ANO-270: ``converged`` does not cover the derivative solves
# ---------------------------------------------------------------------------

#: 256 unknowns, condition number about 81: a cold CG solve to 1e-5 needs
#: some fifty steps.
_N, _DIAGONAL = 256, 2.05
_A = _matrix(_DIAGONAL, _N)
#: One eigenvector of the operator, from the middle of its spectrum: CG and
#: GMRES solve for it in one step, whatever the budget.
_EIGENVECTOR = np.linalg.eigh(_A)[1][:, _N // 2]
_RNG = np.random.default_rng(5)
_DIRECTION = _RNG.standard_normal(_N)
_COVECTOR = _RNG.standard_normal(_N)

_DIFFERENTIATED_ROUTES = {
    "cg default": (sharded_cg, {}),
    "cg loop": (sharded_cg, {"backend": "loop"}),
    "gmres loop": (sharded_gmres, {"backend": "loop", "restart": 5}),
}


def _value_and_derivatives(route, max_iters):
    solver, selection = _DIFFERENTIATED_ROUTES[route]
    b = jnp.asarray(_EIGENVECTOR, jnp.float32)

    def solve(rhs):
        return solver(_operator(_DIAGONAL), rhs, rtol=1e-5,
                      max_iters=max_iters, differentiable=True, **selection)

    result = solve(b)
    _, tangent = jax.jvp(lambda rhs: solve(rhs).value, (b,),
                         (jnp.asarray(_DIRECTION, jnp.float32),))
    gradient = jax.grad(
        lambda rhs: jnp.vdot(jnp.asarray(_COVECTOR, jnp.float32),
                             solve(rhs).value))(b)
    return result, tangent, gradient


@pytest.mark.parametrize("route", sorted(_DIFFERENTIATED_ROUTES))
def test_converged_is_true_beside_a_derivative_whose_own_solve_ran_out_of_steps(route):
    """The right-hand side is one eigenvector, so the primal solve is done
    in one step and ``converged`` is True with a budget of five.  The
    tangent solve (for a white-noise direction) and the cotangent solve
    get the same five steps from zero, need about fifty, and come back
    far off: measured 46 % and 49 % on the CG routes, 67 % and 64 % on
    the GMRES loop.  Nothing in the result says so."""
    result, tangent, gradient = _value_and_derivatives(route, max_iters=5)
    assert bool(result.converged)
    assert _relative_error(result.value, np.linalg.solve(_A, _EIGENVECTOR)) < 1e-3
    assert _relative_error(tangent, np.linalg.solve(_A, _DIRECTION)) > 0.1
    assert _relative_error(gradient, np.linalg.solve(_A.T, _COVECTOR)) > 0.1


@pytest.mark.parametrize("route", sorted(_DIFFERENTIATED_ROUTES))
def test_the_same_derivatives_are_right_when_the_budget_covers_a_cold_solve(route):
    """The workaround of the entry: give ``max_iters`` what a solve from
    zero needs.  The derivative solves then meet ``rtol`` and the same
    derivatives are right (the bound is ``kappa * rtol``, 8e-4 here)."""
    result, tangent, gradient = _value_and_derivatives(route, max_iters=500)
    assert bool(result.converged)
    assert _relative_error(tangent, np.linalg.solve(_A, _DIRECTION)) < 1e-2
    assert _relative_error(gradient, np.linalg.solve(_A.T, _COVECTOR)) < 1e-2


# ---------------------------------------------------------------------------
# MADD-ANO-271: a tolerance the module cannot read, or does not check
# ---------------------------------------------------------------------------

_M = 64
_B = np.random.default_rng(11).standard_normal(_M).astype(np.float32)
_SOLUTION = np.linalg.solve(_matrix(2.5, _M), _B.astype(np.float64))

_LINEAX_ROUTES = {
    "cg default": (sharded_cg, {}),
    "cg lineax": (sharded_cg, {"backend": "lineax"}),
    "gmres default": (sharded_gmres, {}),
}


def _fields(result):
    return result.value, result.converged, result.residual_norm


@pytest.mark.parametrize("route", ["cg default", "cg lineax"])
def test_a_negative_atol_that_is_traced_is_not_refused_and_lineax_vouches_for_the_value(route):
    """The same ``atol=-1.0`` as a Python float raises a ``ValueError``.
    Given as an argument of a jitted function it cannot be read at trace
    time; lineax's entrywise test ``|r_i| <= atol + rtol * |b_i|`` then
    passes after one step.  Measured: the value 72 % off, ``||b - A x||``
    at 5.7e5 times ``rtol * ||b||``, ``converged=True``."""
    solver, selection = _LINEAX_ROUTES[route]
    with pytest.raises(ValueError, match="atol"):
        solver(_operator(2.5), jnp.asarray(_B), atol=-1.0, **selection)

    value, converged, _ = jax.jit(
        lambda b, atol: _fields(solver(_operator(2.5), b, atol=atol, **selection))
    )(jnp.asarray(_B), jnp.float32(-1.0))
    assert bool(converged)
    assert _relative_error(value, _SOLUTION) > 0.1


@pytest.mark.parametrize("route", sorted(_LINEAX_ROUTES))
def test_a_nan_atol_that_is_traced_gives_zeros_that_read_converged(route):
    solver, selection = _LINEAX_ROUTES[route]
    value, converged, _ = jax.jit(
        lambda b, atol: _fields(solver(_operator(2.5), b, atol=atol, **selection))
    )(jnp.asarray(_B), jnp.float32(np.nan))
    assert bool(converged)
    assert not np.any(np.asarray(value))


@pytest.mark.parametrize("route", sorted(_LINEAX_ROUTES))
def test_a_nan_rtol_is_not_refused_and_gives_zeros_that_read_converged(route):
    """``rtol`` is not validated at all: a NaN, as a plain Python float in
    a call that is not jitted, is taken as it comes."""
    solver, selection = _LINEAX_ROUTES[route]
    result = solver(_operator(2.5), jnp.asarray(_B), rtol=float("nan"), **selection)
    assert bool(result.converged)
    assert not np.any(np.asarray(result.value))


def test_the_cg_loop_does_not_vouch_for_those_tolerances():
    """What separates the routes: the loop's flag is a comparison with
    ``max(atol, rtol * ||b||)``.  A negative ``atol`` drops out of that
    maximum and the solve is the relative one; a NaN makes the comparison
    False."""
    b = jnp.asarray(_B)
    negative = jax.jit(
        lambda rhs, atol: _fields(sharded_cg(_operator(2.5), rhs, atol=atol, rtol=1e-4,
                                             backend="loop"))
    )(b, jnp.float32(-1.0))
    assert bool(negative[1])
    assert _relative_error(negative[0], _SOLUTION) < 1e-3

    not_a_number = sharded_cg(_operator(2.5), b, rtol=float("nan"), backend="loop")
    assert not bool(not_a_number.converged)


# ---------------------------------------------------------------------------
# MADD-ANO-272: a solution that overflows the dtype
# ---------------------------------------------------------------------------

#: Every entry a quarter of the largest float32; the solution of the
#: operator below is about twenty times the right-hand side.
_HUGE = np.full(_M, np.finfo(np.float32).max / 4, np.float32)

_EVERY_ROUTE = dict(_LINEAX_ROUTES)
_EVERY_ROUTE.update({
    "cg loop": (sharded_cg, {"backend": "loop"}),
    "gmres loop": (sharded_gmres, {"backend": "loop"}),
})


@pytest.mark.parametrize("route", sorted(_EVERY_ROUTE))
def test_a_solution_that_overflows_is_returned_as_infinities_beside_a_finite_norm(route):
    """``residual_norm`` is documented as the norm of ``b - A x`` for the
    returned value, and NaN where there is no answer.  Here the returned
    value holds infinities and the norm beside it is finite: it is the
    norm in the rescaled frame the solve ran in, multiplied back."""
    solver, selection = _EVERY_ROUTE[route]
    result = solver(_operator(2.05), jnp.asarray(_HUGE), rtol=1e-3, **selection)
    assert np.any(np.isinf(np.asarray(result.value)))
    assert np.isfinite(float(result.residual_norm))


@pytest.mark.parametrize("route", sorted(_EVERY_ROUTE))
def test_those_infinities_read_converged_on_every_route(route):
    """At a tolerance float32 can meet (``rtol=1e-3`` here; the rescaled
    solve stops at 0.7 of it or less) the flag is the rescaled solve's,
    on every route.  At the default ``rtol=1e-6``, which float32 does not
    meet on this system, the loop routes read False and the lineax routes
    True (lineax's own verdict): measured, not pinned."""
    solver, selection = _EVERY_ROUTE[route]
    result = solver(_operator(2.05), jnp.asarray(_HUGE), rtol=1e-3, **selection)
    assert bool(result.converged)
    assert np.any(np.isinf(np.asarray(result.value)))


def test_the_same_system_a_thousand_times_smaller_is_solved():
    """The control: the defect is the overflow, not the system."""
    b = (_HUGE / np.float32(1024)).astype(np.float32)
    result = sharded_cg(_operator(2.05), jnp.asarray(b), rtol=1e-3)
    want = np.linalg.solve(_matrix(2.05, _M), b.astype(np.float64))
    assert bool(result.converged)
    assert np.all(np.isfinite(np.asarray(result.value)))
    assert _relative_error(result.value, want) < 1e-2


@pytest.mark.parametrize("route", sorted(_EVERY_ROUTE))
def test_a_right_hand_side_whose_norm_overflows_is_solved_where_its_solution_fits(route):
    """The part of the entry that 0.4.0 resolved.  The 2-norm of 64
    entries of a quarter of the largest float32 is twice that number, so
    it is infinite in float32; behind a diagonal of 10 the solution is an
    eighth of the right-hand side and fits.  The solve runs on the
    right-hand side rescaled by a power of two, so nothing overflows:
    measured within 2e-5 on every route.  In 0.3.0 and 0.3.1 the CG loop
    returned zeros with ``converged=True``, ``iters=0`` and an infinite
    ``residual_norm`` (its tolerance ``rtol * ||b||`` was infinite), and
    lineax's CG and both GMRES routes NaN with ``converged=False``."""
    solver, selection = _EVERY_ROUTE[route]
    result = solver(_operator(10.0), jnp.asarray(_HUGE), rtol=1e-4, **selection)
    want = np.linalg.solve(_matrix(10.0, _M), _HUGE.astype(np.float64))
    assert bool(result.converged)
    assert np.all(np.isfinite(np.asarray(result.value)))
    assert np.isfinite(float(result.residual_norm))
    assert _relative_error(result.value, want) < 1e-3
