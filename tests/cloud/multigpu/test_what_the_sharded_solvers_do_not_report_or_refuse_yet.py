"""Two things ``sharded_cg`` and ``sharded_gmres`` do not report in
0.4.0, each pinned as it behaves today.  Both are registered open; a fix
turns the matching tests red, and the entry is then closed with it.

* **MADD-ANO-270.**  Under ``differentiable=True`` the result's
  ``converged`` is the primal solve's.  Each derivative is one more solve
  of the same system, from zero, under the caller's ``rtol`` and
  ``max_iters``, and nothing reports whether THAT solve met its tolerance.
  A primal that needs fewer steps than a cold solve (a warm start, an
  ``atol`` above ``||b||``, a right-hand side that is one eigenvector)
  reads ``converged=True`` beside a derivative that is far off.
* **MADD-ANO-282.**  The report of ``differentiable=True``, and the
  residual it attaches the derivative to, are products on the vectors as
  they are, not in the power-of-two frame the solve itself runs in.  For
  a solution within a factor of the operator's diagonal of the dtype's
  largest number such a product overflows although the solution fits: the
  call returns the right value with ``converged=False`` and an infinite
  ``residual_norm``, and with an ``atol`` or an ``x0`` NaN in every entry,
  where the ordinary call answers.

The two entries this file pinned beside it are resolved, and their tests
assert the fix in
``test_what_the_sharded_solvers_answer_at_the_edges_of_a_tolerance_and_of_the_dtype.py``:
a tolerance that is not a number >= 0 (MADD-ANO-271) and a solution that
overflows the dtype (MADD-ANO-272).

System: the 1-D Dirichlet Laplacian shifted on its diagonal, float32.  No
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
# MADD-ANO-282: ``differentiable=True`` just under the top of the dtype
# ---------------------------------------------------------------------------

_M = 64
#: Every entry 1/28 of the largest float32; behind a diagonal of 2.05 the
#: solution reaches 0.71 of that number, and ``2.05 * x`` overflows.
_NEAR_THE_TOP = np.full(_M, np.finfo(np.float32).max / 28, np.float32)
_NEAR_THE_TOP_SOLUTION = np.linalg.solve(_matrix(2.05, _M), _NEAR_THE_TOP.astype(np.float64))

_TOP_ROUTES = {
    "cg default": (sharded_cg, {}),
    "cg loop": (sharded_cg, {"backend": "loop"}),
    "gmres default": (sharded_gmres, {}),
    "gmres loop": (sharded_gmres, {"backend": "loop"}),
}


@pytest.mark.parametrize("route", sorted(_TOP_ROUTES))
def test_with_the_flag_a_solution_just_under_the_top_reads_not_converged_beside_an_infinite_norm(
        route):
    """The value is right (it is the ordinary call's) and the report is
    not: its product ``A x`` on the bare value overflows in ``2.05 * x``,
    by a factor of 1.46 at the largest entry (structural, not rounding).
    The ordinary call, whose report is taken in the frame of ``b``, reads
    a finite norm."""
    solver, selection = _TOP_ROUTES[route]
    b = jnp.asarray(_NEAR_THE_TOP)
    ordinary = solver(_operator(2.05), b, rtol=1e-3, **selection)
    assert np.isfinite(float(ordinary.residual_norm))
    assert _relative_error(ordinary.value, _NEAR_THE_TOP_SOLUTION) < 1e-2

    flagged = solver(_operator(2.05), b, rtol=1e-3, differentiable=True, **selection)
    assert _relative_error(flagged.value, _NEAR_THE_TOP_SOLUTION) < 1e-2
    assert np.isinf(float(flagged.residual_norm))
    assert not bool(flagged.converged)


@pytest.mark.parametrize("route", sorted(_TOP_ROUTES))
def test_with_the_flag_and_an_atol_the_same_solution_is_nan(route):
    """With an ``atol`` (or an ``x0``) the derivative is attached through
    the residual of the ordinary solve's value, ``b - A x``, taken on the
    bare vectors: it overflows the same way, and the value is NaN in
    every entry with ``converged=False``."""
    solver, selection = _TOP_ROUTES[route]
    flagged = solver(_operator(2.05), jnp.asarray(_NEAR_THE_TOP), rtol=1e-3, atol=0.0,
                     differentiable=True, **selection)
    assert np.all(np.isnan(np.asarray(flagged.value)))
    assert not bool(flagged.converged)
