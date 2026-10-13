"""The public Krylov solvers give the same answer in any units.

``sharded_cg`` and ``sharded_gmres`` (STABLE, both backends) and
``ift_linear_solve`` (EXPERIMENTAL) stopped on ``||r|| <= max(atol, rtol *
||b||)`` -- lineax's entrywise form for the lineax backends -- with an
absolute default ``atol=1e-8``.  A right-hand side below about ``1e-2`` was
solved only to that absolute floor, and one below about ``1e-8`` passed the
test with the zero initial guess before a single step: on a 12-unknown SPD
system in float32 every backend returned relative errors of 0.4-1.0 at
``||b|| ~ 1e-9`` and reported ``converged=True``.  The solves now run on ``b``
rescaled by an exact power of two (``maddening.core._pow2_frame``), and the
default floor is ``1e-8`` of that rescaled ``b``: the same answer, to the bit,
at every power-of-two scale, and at the scale where ``max|b|`` is in
``[0.5, 1)`` exactly the answer the old default gave.  An explicit ``atol`` is
still an absolute floor in ``b``'s units.

The loop backend of ``sharded_gmres`` also returned NaN with
``converged=False`` whenever ``restart`` exceeded the system size -- the
default 50 against any system of fewer than 50 unknowns; it now clamps
``restart`` as the lineax backend always did.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.iterative_solver import sharded_cg, sharded_gmres
from maddening.core.solver_utils import ift_linear_solve

N = 12
_rng = np.random.default_rng(0)
_M = _rng.standard_normal((N, N)).astype(np.float32)
A = jnp.asarray(_M @ _M.T + N * np.eye(N, dtype=np.float32))     # SPD, cond ~ 10
B0 = jnp.asarray(_rng.standard_normal(N).astype(np.float32))
X_TRUE = np.linalg.solve(np.asarray(A, np.float64), np.asarray(B0, np.float64))


def _mv(v):
    return A @ v


#: Every Krylov path, as ``rhs -> (x, converged)``; each is jitted once, so
#: the scales below reuse one program per backend.
SOLVERS = {
    "sharded_cg/lineax": lambda b, **kw: sharded_cg(_mv, b, backend="lineax", **kw),
    "sharded_cg/loop": lambda b, **kw: sharded_cg(_mv, b, backend="loop", **kw),
    "sharded_gmres/lineax": lambda b, **kw: sharded_gmres(_mv, b, backend="lineax", **kw),
    "sharded_gmres/loop": lambda b, **kw: sharded_gmres(_mv, b, backend="loop", **kw),
    "ift_linear_solve/gmres": lambda b, **kw: ift_linear_solve(_mv, b, solver="gmres", **kw),
    "ift_linear_solve/cg": lambda b, **kw: ift_linear_solve(_mv, b, solver="cg", **kw),
}


def _value(result):
    return result if isinstance(result, jax.Array) else result.value


def _converged(result):
    return True if isinstance(result, jax.Array) else bool(result.converged)


@pytest.fixture(scope="module")
def solved():
    """``{solver: {k: (x, converged)}}`` for ``b = B0 * 2**k``."""
    out = {}
    for name, solve in SOLVERS.items():
        def run(b, solve=solve):
            r = solve(b)
            return _value(r), (jnp.asarray(True) if isinstance(r, jax.Array) else r.converged)
        fn = jax.jit(run)
        out[name] = {}
        for k in (0, -20, -30, -100, 40):
            x, conv = fn(B0 * jnp.float32(2.0 ** k))
            out[name][k] = (np.asarray(x), bool(conv))
    return out


@pytest.mark.parametrize("name", list(SOLVERS))
def test_every_krylov_solve_gives_the_same_answer_at_every_power_of_two_scale(solved, name):
    x1, conv1 = solved[name][0]
    err = float(np.max(np.abs(x1 - X_TRUE)) / np.max(np.abs(X_TRUE)))
    assert conv1 and err < 1e-4, (name, err)
    for k, (xk, convk) in solved[name].items():
        assert convk, (name, k)
        assert np.array_equal(xk, (x1 * np.float32(2.0 ** k)).astype(np.float32)), (
            f"{name}: the solve of b * 2**{k} is not the solve of b times 2**{k}; "
            f"relative error {np.max(np.abs(xk / 2.0 ** k - X_TRUE)) / np.max(np.abs(X_TRUE)):.2e}")


@pytest.mark.parametrize("name", ["sharded_cg/loop", "sharded_gmres/loop"])
def test_on_the_loop_backends_the_default_stops_where_it_did_above_one_hundredth(name):
    """``||r|| <= max(1e-8, rtol ||b||)`` is ``rtol ||b||`` once ``||b|| >= 1e-2``:
    there the old default and the new one are the same test, to the bit."""
    for b in (B0 * jnp.float32(0.03), B0, B0 * jnp.float32(1e4)):
        default = _value(SOLVERS[name](b))
        old = _value(SOLVERS[name](b, atol=1e-8))
        assert np.array_equal(np.asarray(default), np.asarray(old)), name


def test_a_right_hand_side_gmres_broke_down_on_now_solves():
    """At ``max|b| ~ 0.95`` the absolute ``1e-8`` held this system's
    near-zero residual entries below float32 rounding of the large ones, and
    lineax GMRES raised "iterative breakdown" (it solved at ``max|b| ~ 1.9``
    and ``0.475``).  The floor is now ``rtol * max|b|``."""
    x = np.asarray(ift_linear_solve(_mv, B0 * jnp.float32(0.5), solver="gmres"))
    assert np.max(np.abs(x / 0.5 - X_TRUE)) / np.max(np.abs(X_TRUE)) < 1e-4


def test_an_explicit_atol_is_still_an_absolute_floor_in_b_units():
    b = B0 * jnp.float32(1e-6)          # ||b|| ~ 3e-6, under an asserted floor of 1e-3
    r = sharded_cg(_mv, b, backend="loop", atol=1e-3)
    # The zero initial guess already meets the floor: the solve stops on it.
    assert np.all(np.asarray(r.value) == 0.0) and bool(r.converged)
    r = sharded_cg(_mv, b, backend="loop")
    assert np.max(np.abs(np.asarray(r.value) / 1e-6 - X_TRUE)) / np.max(np.abs(X_TRUE)) < 1e-4


def test_the_loop_gmres_clamps_restart_to_the_system_size():
    for restart in (N, N + 1, 50):
        r = sharded_gmres(_mv, B0, backend="loop", restart=restart)
        x = np.asarray(r.value)
        assert np.all(np.isfinite(x)) and bool(r.converged), restart
        assert np.max(np.abs(x - X_TRUE)) / np.max(np.abs(X_TRUE)) < 1e-4, restart


@pytest.mark.parametrize("name", ["sharded_cg/loop", "sharded_gmres/lineax", "ift_linear_solve/gmres"])
def test_the_derivative_through_a_solve_is_the_same_at_every_power_of_two_scale(name):
    """``d/db sum(x(b)**2) = 2 A^-T x``: the cotangent solve's right-hand side
    is ``2 x``, as small as the solution; at ``2**-30`` it was answered with
    zeros (or a few steps' worth) and the gradient was wrong without an error."""
    if name.startswith("sharded"):
        def solve(b):
            return SOLVERS[name](b, differentiable=True)
    else:
        solve = SOLVERS[name]
    grad = jax.jit(jax.grad(lambda b: jnp.sum(_value(solve(b)) ** 2)))
    g1 = np.asarray(grad(B0))
    want = 2 * np.linalg.solve(np.asarray(A, np.float64).T, X_TRUE)
    assert np.max(np.abs(g1 - want)) / np.max(np.abs(want)) < 1e-4, name
    for k in (-30, -60):
        s = np.float32(2.0 ** k)
        gk = np.asarray(grad(B0 * s))
        assert np.array_equal(gk, (g1 * s).astype(np.float32)), (name, k)


def test_a_differentiable_solve_reports_convergence_at_small_scale():
    r = sharded_cg(_mv, B0 * jnp.float32(2.0 ** -100), differentiable=True)
    assert bool(r.converged)
    assert float(r.residual_norm) <= 1e-6 * float(jnp.linalg.norm(B0)) * 2.0 ** -100 * 10
