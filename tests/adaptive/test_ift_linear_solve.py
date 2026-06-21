"""M1 tests for ``maddening.core.solver_utils.ift_linear_solve``.

Test list (numbering matches
``plans/MADDENING_ADAPTIVE_NODE_IMPLEMENTATION_PLAN.md`` §4 M1):

1. test_dense_baseline_agreement
2. test_gmres_matches_dense_on_indefinite
3. test_cg_matches_dense_on_spd
4. test_autodiff_correctness_dense
5. test_autodiff_correctness_gmres
6. test_autodiff_correctness_cg
7. test_gmres_restart_clamp
8. test_preconditioner_passes_through
9. test_preconditioner_gradient_blocked
10. test_bcoo_operator_compatibility
11. test_unsupported_solver_raises
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from maddening.core.solver_utils import ift_linear_solve


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _spd_matrix(n: int, seed: int = 0) -> jnp.ndarray:
    """Build a deterministic SPD matrix of size n."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return jnp.asarray(a @ a.T + n * np.eye(n))


def _indefinite_matrix(n: int, seed: int = 1) -> jnp.ndarray:
    """Build a non-symmetric matrix with bounded condition number."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return jnp.asarray(a + n * np.eye(n))


# ----------------------------------------------------------------------
# 1. test_dense_baseline_agreement
# ----------------------------------------------------------------------

def test_dense_baseline_agreement():
    """solver='dense' on a 16x16 SPD system matches jnp.linalg.solve to 1e-12.

    Guards: the dense fallback materialises A correctly via vmap+eye.
    """
    A = _spd_matrix(16)
    b = jnp.asarray(np.random.default_rng(2).standard_normal(16))
    x = ift_linear_solve(lambda v: A @ v, b, solver="dense")
    x_ref = jnp.linalg.solve(A, b)
    assert jnp.allclose(x, x_ref, atol=1e-12, rtol=0)


# ----------------------------------------------------------------------
# 2. test_gmres_matches_dense_on_indefinite
# ----------------------------------------------------------------------

def test_gmres_matches_dense_on_indefinite():
    """GMRES on a 64x64 non-symmetric system matches dense to 1e-6."""
    A = _indefinite_matrix(64)
    b = jnp.asarray(np.random.default_rng(3).standard_normal(64))
    x_gmres = ift_linear_solve(lambda v: A @ v, b, solver="gmres",
                                rtol=1e-10, atol=1e-12)
    x_ref = jnp.linalg.solve(A, b)
    assert jnp.allclose(x_gmres, x_ref, atol=1e-6, rtol=1e-6)


# ----------------------------------------------------------------------
# 3. test_cg_matches_dense_on_spd
# ----------------------------------------------------------------------

def test_cg_matches_dense_on_spd():
    """CG on a 64x64 SPD system matches dense to 1e-6."""
    A = _spd_matrix(64)
    b = jnp.asarray(np.random.default_rng(4).standard_normal(64))
    x_cg = ift_linear_solve(lambda v: A @ v, b, solver="cg",
                             rtol=1e-10, atol=1e-12)
    x_ref = jnp.linalg.solve(A, b)
    assert jnp.allclose(x_cg, x_ref, atol=1e-6, rtol=1e-6)


# ----------------------------------------------------------------------
# 4. test_autodiff_correctness_dense
# ----------------------------------------------------------------------

def test_autodiff_correctness_dense():
    """jax.grad through a theta-dependent dense solve matches FD to 1e-5.

    Guards: lineax's native autodiff propagates the linear-solve adjoint
    correctly through the dense backend.
    """
    A0 = _spd_matrix(8)
    b0 = jnp.asarray(np.random.default_rng(5).standard_normal(8))

    def J(theta):
        # Linear theta-perturbation of both A and b.
        A = A0 + theta * jnp.eye(8)
        b = b0 + theta * jnp.arange(8.0)
        x = ift_linear_solve(lambda v: A @ v, b, solver="dense")
        return jnp.sum(x ** 2)

    theta0 = 0.5
    g_auto = jax.grad(J)(theta0)
    h = 1e-5
    g_fd = (J(theta0 + h) - J(theta0 - h)) / (2 * h)
    assert abs(float(g_auto) - g_fd) / abs(g_fd) < 1e-5


# ----------------------------------------------------------------------
# 5. test_autodiff_correctness_gmres
# ----------------------------------------------------------------------

def test_autodiff_correctness_gmres():
    """jax.grad through GMRES matches FD to 1e-5."""
    A0 = _indefinite_matrix(32)
    b0 = jnp.asarray(np.random.default_rng(6).standard_normal(32))

    def J(theta):
        A = A0 + theta * jnp.eye(32)
        b = b0 + theta * jnp.arange(32.0)
        x = ift_linear_solve(lambda v: A @ v, b, solver="gmres",
                              rtol=1e-12, atol=1e-14)
        return jnp.sum(x ** 2)

    theta0 = 0.5
    g_auto = jax.grad(J)(theta0)
    h = 1e-5
    g_fd = (J(theta0 + h) - J(theta0 - h)) / (2 * h)
    assert abs(float(g_auto) - g_fd) / abs(g_fd) < 1e-5


# ----------------------------------------------------------------------
# 6. test_autodiff_correctness_cg
# ----------------------------------------------------------------------

def test_autodiff_correctness_cg():
    """jax.grad through CG on SPD matches FD to 1e-5."""
    A0 = _spd_matrix(32)
    b0 = jnp.asarray(np.random.default_rng(7).standard_normal(32))

    def J(theta):
        # Keep A SPD: add a positive multiple of I that scales with theta.
        A = A0 + jnp.abs(theta) * jnp.eye(32)
        b = b0 + theta * jnp.arange(32.0)
        x = ift_linear_solve(lambda v: A @ v, b, solver="cg",
                              rtol=1e-12, atol=1e-14)
        return jnp.sum(x ** 2)

    theta0 = 0.5
    g_auto = jax.grad(J)(theta0)
    h = 1e-5
    g_fd = (J(theta0 + h) - J(theta0 - h)) / (2 * h)
    assert abs(float(g_auto) - g_fd) / abs(g_fd) < 1e-5


# ----------------------------------------------------------------------
# 7. test_gmres_restart_clamp
# ----------------------------------------------------------------------

def test_gmres_restart_clamp():
    """At N=200, GMRES uses restart=min(N, 50)=50, not the lineax default of 20.

    Guards: the silent-low-rank-adjoint bug documented in
    ``graph_manager._ift_solve_bwd:430-451``.  We inspect the lineax
    solver object's ``restart`` attribute via a monkeypatched
    ``lineax.GMRES`` to capture the value passed.
    """
    import lineax as lx
    captured = {}
    original_gmres = lx.GMRES

    def spy_gmres(*args, **kwargs):
        captured["restart"] = kwargs.get("restart")
        captured["max_steps"] = kwargs.get("max_steps")
        return original_gmres(*args, **kwargs)

    lx.GMRES = spy_gmres
    try:
        A = _indefinite_matrix(200)
        b = jnp.asarray(np.random.default_rng(8).standard_normal(200))
        _ = ift_linear_solve(lambda v: A @ v, b, solver="gmres")
    finally:
        lx.GMRES = original_gmres
    assert captured["restart"] == 50, (
        f"Expected restart=min(200, 50)=50; got {captured['restart']}. "
        "Regression vs graph_manager._ift_solve_bwd silent-low-rank-adjoint guard."
    )
    assert captured["max_steps"] >= 4 * 50


# ----------------------------------------------------------------------
# 8. test_preconditioner_passes_through
# ----------------------------------------------------------------------

def test_preconditioner_passes_through():
    """Jacobi preconditioner on a diagonal-dominant system gives the same
    final solution as the unpreconditioned solve, within tolerance.

    Guards: the preconditioner kwarg is wired through to lineax.
    """
    A = _spd_matrix(64)
    A = A + 50.0 * jnp.eye(64)  # diagonal-dominant
    b = jnp.asarray(np.random.default_rng(9).standard_normal(64))
    diag_inv = 1.0 / jnp.diag(A)
    x_no_pc = ift_linear_solve(lambda v: A @ v, b, solver="cg",
                                rtol=1e-10, atol=1e-12)
    x_pc = ift_linear_solve(
        lambda v: A @ v, b, solver="cg",
        preconditioner=lambda v: diag_inv * v,
        rtol=1e-10, atol=1e-12,
    )
    # Both should converge to the same answer.
    assert jnp.allclose(x_pc, x_no_pc, atol=1e-6, rtol=1e-6)
    # And both should match dense.
    x_ref = jnp.linalg.solve(A, b)
    assert jnp.allclose(x_pc, x_ref, atol=1e-6, rtol=1e-6)


# ----------------------------------------------------------------------
# 9. test_preconditioner_gradient_blocked
# ----------------------------------------------------------------------

def test_preconditioner_gradient_blocked():
    """A theta-dependent preconditioner does not contribute gradient noise.

    Guards: the round-7 finding that ``M`` is gradient-irrelevant at
    convergence is enforced by the wrapper's ``stop_gradient`` on the
    preconditioner output.
    """
    A = _spd_matrix(32) + 20.0 * jnp.eye(32)
    b = jnp.asarray(np.random.default_rng(10).standard_normal(32))
    diag = jnp.diag(A)

    def J_with_theta_pc(theta):
        # A theta-dependent preconditioner: scale the diagonal precond
        # by (1 + theta).  At convergence the solution shouldn't depend
        # on theta through M, so this should have zero gradient.
        x = ift_linear_solve(
            lambda v: A @ v, b, solver="cg",
            preconditioner=lambda v: ((1.0 + theta) / diag) * v,
            rtol=1e-12, atol=1e-14,
        )
        return jnp.sum(x ** 2)

    g = jax.grad(J_with_theta_pc)(0.5)
    # The gradient should be machine-zero -- M's theta-dependence is
    # blocked by stop_gradient inside the wrapper.
    assert abs(float(g)) < 1e-8


# ----------------------------------------------------------------------
# 10. test_bcoo_operator_compatibility
# ----------------------------------------------------------------------

def test_bcoo_operator_compatibility():
    """A BCOO-backed operator solves correctly and ``jax.grad`` is correct.

    Guards: spike round-6 Inv 3B BCOO+lineax compatibility (rel err 7e-15).
    """
    import jax.experimental.sparse as jsparse
    A_dense = _spd_matrix(64) + 5.0 * jnp.eye(64)
    A_bcoo = jsparse.BCOO.fromdense(A_dense)
    b = jnp.asarray(np.random.default_rng(11).standard_normal(64))

    def matvec(v):
        return A_bcoo @ v

    x = ift_linear_solve(matvec, b, solver="cg",
                          rtol=1e-10, atol=1e-12)
    x_ref = jnp.linalg.solve(A_dense, b)
    assert jnp.allclose(x, x_ref, atol=1e-6, rtol=1e-6)

    # And jax.grad through it should be correct (compare to FD).
    def J(theta):
        b_t = b + theta * jnp.arange(64.0)
        x = ift_linear_solve(matvec, b_t, solver="cg",
                              rtol=1e-12, atol=1e-14)
        return jnp.sum(x ** 2)

    g_auto = jax.grad(J)(0.5)
    h = 1e-5
    g_fd = (J(0.5 + h) - J(0.5 - h)) / (2 * h)
    assert abs(float(g_auto) - g_fd) / abs(g_fd) < 1e-4


# ----------------------------------------------------------------------
# 11. test_unsupported_solver_raises
# ----------------------------------------------------------------------

def test_unsupported_solver_raises():
    """An unknown solver name raises ValueError with the valid options listed."""
    A = jnp.eye(4)
    b = jnp.ones(4)
    with pytest.raises(ValueError, match=r"unsupported solver"):
        ift_linear_solve(lambda v: A @ v, b, solver="nonexistent")  # type: ignore[arg-type]
