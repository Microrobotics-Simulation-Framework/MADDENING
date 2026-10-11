"""What ``sharded_cg`` and ``sharded_gmres`` answer to an operator that
returns a non-finite value, what they refuse, and what their flag tests.

* **MADD-ANO-269.**  An operator that returns a NaN or an infinity gets
  what a non-finite right-hand side gets (MADD-ANO-155), on every route:
  NaN in every entry of ``value``, a NaN ``residual_norm``,
  ``converged=False``.  Until 0.4.0 ``sharded_cg``'s default call and
  ``backend="lineax"`` returned zeros with ``converged=True`` and
  ``iters=0`` (lineax stops at once on its zero start, which is finite),
  and the CG loop returned zeros with ``converged=False``.  The general
  form: on no route is ``converged`` True beside a ``residual_norm`` that
  is not finite.
* A negative (or NaN) ``atol`` is refused with a ``ValueError``.  It was
  taken as it came, and lineax's entrywise test then read True at 2e5
  times the tolerance asked for.
* ``converged`` on the loop backends and under ``differentiable=True``
  tests ``residual_norm`` against the LARGER of ``atol`` and
  ``rtol * ||b||``, not their sum; on the lineax backend it is lineax's
  verdict, which is not "the value is finite".
* ``sharded_gmres``'s ``max_iters`` on the loop backend is a budget in
  whole restart cycles, at least one.

The system is the 1-D Dirichlet Laplacian shifted on its diagonal by 0.5,
64 unknowns, float32, white noise.  No iteration count of a converged solve
is pinned.
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

UNKNOWNS = 64
SHIFT = 0.5


def _laplacian(x, shift=SHIFT):
    left = jnp.concatenate([jnp.zeros((1,), x.dtype), x[:-1]])
    right = jnp.concatenate([x[1:], jnp.zeros((1,), x.dtype)])
    return (2 + shift) * x - left - right


def _b() -> np.ndarray:
    return np.random.default_rng(0).standard_normal(UNKNOWNS).astype(np.float32)


def _jacobi():
    return jacobi_preconditioner(jnp.full((UNKNOWNS,), 2 + SHIFT, jnp.float32))


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
_routes = pytest.mark.parametrize("route", list(ROUTES))


def _call(route: str, matvec, b, **kw):
    solver, selects = ROUTES[route]
    selects = dict(selects)
    if selects.get("preconditioner") == "jacobi":
        selects["preconditioner"] = _jacobi()
    return solver(matvec, b, **selects, **kw)


def _fields(result) -> tuple:
    return result.value, result.converged, result.iters, result.residual_norm


# ---------------------------------------------------------------------------
# MADD-ANO-269: an operator that returns a NaN or an infinity
# ---------------------------------------------------------------------------


def _poison(kind: str) -> np.ndarray:
    """What is added to the operator's diagonal: nothing, or a non-finite
    coefficient at one unknown, or at every unknown."""
    poison = np.zeros(UNKNOWNS, np.float32)
    if kind == "a NaN at one unknown":
        poison[UNKNOWNS // 3] = np.nan
    elif kind == "an infinity at one unknown":
        poison[UNKNOWNS // 3] = np.inf
    elif kind == "minus infinity everywhere":
        poison[:] = -np.inf
    else:
        assert kind == "healthy"
    return poison


POISONS = ("a NaN at one unknown", "an infinity at one unknown", "minus infinity everywhere")


@pytest.mark.parametrize("differentiable", [False, True], ids=["ordinary", "differentiable"])
@_routes
def test_an_operator_that_returns_a_non_finite_value_is_answered_with_nan(route, differentiable):
    """One jitted program per route, given the operator's extra diagonal as
    an argument: the healthy operator is solved (so the rule is not
    answering NaN to everything), and each poisoned one gets NaN in every
    entry, a NaN norm and ``converged=False``.  (``rtol=1e-4``: at the
    default 1e-6 lineax's float32 value sits at the tolerance, and under
    ``differentiable=True`` the flag has no allowance for rounding.)"""
    b = jnp.asarray(_b())

    @jax.jit
    def solve(poison):
        return _fields(_call(route, lambda x: _laplacian(x) + poison * x, b,
                             rtol=1e-4, differentiable=differentiable))

    value, converged, _, residual_norm = solve(jnp.asarray(_poison("healthy")))
    assert bool(converged) and np.all(np.isfinite(np.asarray(value)))
    assert np.isfinite(float(residual_norm))
    for kind in POISONS:
        value, converged, _, residual_norm = solve(jnp.asarray(_poison(kind)))
        assert np.all(np.isnan(np.asarray(value))), kind
        assert np.isnan(float(residual_norm)), kind
        assert not bool(converged), kind


@pytest.mark.parametrize("route", ["cg default", "cg lineax", "cg loop", "gmres default"])
def test_the_same_answer_when_the_call_is_not_jitted(route):
    """Eagerly the default CG route takes its branch in Python: lineax's
    zeros are finite, so the result is lineax's, and it is the common exit
    that turns it into NaN."""
    poison = jnp.asarray(_poison("a NaN at one unknown"))
    value, converged, _, residual_norm = _fields(
        _call(route, lambda x: _laplacian(x) + poison * x, jnp.asarray(_b())))
    assert np.all(np.isnan(np.asarray(value)))
    assert np.isnan(float(residual_norm))
    assert not bool(converged)


@pytest.mark.parametrize("route", ["cg loop", "cg auto with Jacobi", "gmres loop"])
def test_converged_is_not_true_beside_a_residual_norm_that_is_not_finite(route):
    """The general form.  An infinite ``atol`` makes the tolerance of the
    flag infinite, and an operator of scale 1e30 applied to a start of
    scale 1e30 overflows float32: the residual norm is infinite, and
    ``inf <= inf`` read True.  The flag is False, and the value and the
    norm are NaN, as for any operator that returned an infinity."""
    b = jnp.asarray(_b())
    start = jnp.full((UNKNOWNS,), 1e30, jnp.float32)
    value, converged, _, residual_norm = jax.jit(lambda bb: _fields(_call(
        route, lambda x: 1e30 * x, bb, x0=start, atol=float("inf"), max_iters=0)))(b)
    assert not bool(converged)
    assert np.isnan(float(residual_norm))
    assert np.all(np.isnan(np.asarray(value)))


# ---------------------------------------------------------------------------
# A negative atol is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("atol", [-1.0, -1e-30, float("nan"), np.float32(-1e-3)],
                         ids=["-1", "-1e-30", "nan", "numpy -1e-3"])
@pytest.mark.parametrize("solver", [sharded_cg, sharded_gmres])
def test_a_negative_atol_is_refused(solver, atol):
    with pytest.raises(ValueError, match=f"{solver.__name__}: atol must be None"):
        solver(_laplacian, jnp.asarray(_b()), atol=atol)


@pytest.mark.parametrize("atol", [None, 0.0, 1e-9, np.float32(1e-9)],
                         ids=["None", "0", "1e-9", "numpy 1e-9"])
@pytest.mark.parametrize("solver", [sharded_cg, sharded_gmres])
def test_none_zero_and_a_positive_atol_are_accepted(solver, atol):
    result = solver(_laplacian, jnp.asarray(_b()), atol=atol, backend="loop")
    assert bool(result.converged)


# ---------------------------------------------------------------------------
# What the flag tests
# ---------------------------------------------------------------------------

#: Routes whose flag is ``residual_norm <= max(atol, rtol * ||b||)``, each
#: computed at a different place in the module: the common report of the
#: loop backends, the GMRES cycles' own test, and ``differentiable=True``.
MAX_NOT_SUM = {
    "cg loop": (sharded_cg, {"backend": "loop", "max_iters": 3}),
    "gmres loop": (sharded_gmres, {"backend": "loop", "restart": 3, "max_iters": 3}),
    "gmres loop behind Jacobi": (sharded_gmres, {"backend": "loop", "restart": 3,
                                                 "max_iters": 3, "preconditioner": "jacobi"}),
    "cg loop, differentiable": (sharded_cg, {"backend": "loop", "max_iters": 3,
                                             "differentiable": True}),
    "gmres loop, differentiable": (sharded_gmres, {"backend": "loop", "restart": 3,
                                                   "max_iters": 3, "differentiable": True}),
}


@pytest.mark.parametrize("case", list(MAX_NOT_SUM))
def test_the_flag_tests_the_larger_of_atol_and_rtol_times_b_not_their_sum(case):
    """A solve stopped by its budget after three steps has a residual norm
    ``rho``.  Asked again with ``atol = 0.6 rho`` and
    ``rtol * ||b|| = 0.6 rho`` it stops on the same step with the same
    value: ``rho`` is over the larger of the two (not converged) and under
    their sum."""
    solver, kw = MAX_NOT_SUM[case]
    kw = dict(kw)
    if kw.get("preconditioner") == "jacobi":
        kw["preconditioner"] = _jacobi()
    b = jnp.asarray(_b())
    b_norm = float(np.linalg.norm(_b().astype(np.float64)))

    first = solver(_laplacian, b, rtol=1e-12, atol=1e-30, **kw)
    rho = float(first.residual_norm)
    assert not bool(first.converged) and rho > 1e-3 * b_norm

    again = solver(_laplacian, b, rtol=0.6 * rho / b_norm, atol=0.6 * rho, **kw)
    assert np.asarray(again.value).tobytes() == np.asarray(first.value).tobytes()
    assert float(again.residual_norm) == rho
    assert not bool(again.converged)

    # ... and each of the two alone, once it is over ``rho``, reads True.
    assert bool(solver(_laplacian, b, rtol=1e-12, atol=1.2 * rho, **kw).converged)
    assert bool(solver(_laplacian, b, rtol=1.2 * rho / b_norm, atol=1e-30, **kw).converged)


@pytest.mark.parametrize("route", ["cg lineax", "cg default", "gmres lineax"])
def test_the_lineax_route_reports_lineaxs_verdict_not_that_its_value_is_finite(route):
    """lineax stopped by its budget on an ill-conditioned system (diagonal
    shift 0.01, condition number about 400) returns a finite value far
    from the solution; its verdict, and so ``converged``, is False.  For
    ``sharded_cg``'s default route this is the case in which lineax's
    value is finite and the loop is not asked."""
    b = jnp.asarray(_b())
    kw = {"restart": 2, "max_iters": 1} if route.startswith("gmres") else {"max_iters": 5}
    result = _call(route, lambda x: _laplacian(x, shift=0.01), b, **kw)
    assert np.all(np.isfinite(np.asarray(result.value)))
    assert float(result.residual_norm) > 0.1 * float(np.linalg.norm(_b()))
    assert not bool(result.converged)


def test_the_budget_of_the_gmres_loop_is_in_whole_cycles_and_at_least_one():
    """``max_iters`` of 0 or 1 on the loop backend still runs one restart
    cycle (``restart`` steps), and a budget between two multiples of
    ``restart`` is rounded up: what the docstring of ``max_iters`` says."""
    b = jnp.asarray(_b())
    for budget, steps in ((0, 8), (1, 8), (8, 8), (9, 16)):
        result = sharded_gmres(_laplacian, b, backend="loop", restart=8, max_iters=budget,
                               rtol=1e-12)
        assert int(result.iters) == steps, budget


@pytest.mark.parametrize("differentiable", [False, True], ids=["ordinary", "differentiable"])
def test_a_zero_right_hand_side_is_answered_with_zeros_and_an_ordinary_call_counts_its_cycle(
        differentiable):
    """A zero ``b`` from a zero start is solved by zeros, ``converged=True``
    with a zero norm.  An ordinary GMRES-loop call still runs (and counts)
    its one cycle there, as it always did: only the zero right-hand side
    that ``differentiable=True`` itself solves, to attach the derivative
    to a call with an ``x0`` or an ``atol``, skips the cycle."""
    zeros = jnp.zeros((UNKNOWNS,), jnp.float32)
    result = sharded_gmres(_laplacian, zeros, backend="loop", restart=8,
                           differentiable=differentiable)
    assert not np.any(np.asarray(result.value))
    assert bool(result.converged) and float(result.residual_norm) == 0.0
    if not differentiable:
        assert int(result.iters) == 8
