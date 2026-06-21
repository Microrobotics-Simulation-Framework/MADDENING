"""Round-3 Investigation 2: can anything simpler than full MG-CG
make the Haar PoC viable?

Two distinct senses of "preconditioning" we must keep separate:

  (A) Iteration preconditioning -- accelerate GMRES on the masked
      system to convergence.  Does NOT change the converged answer,
      so J_err and g_err at convergence equal round-2 direct-solve
      numbers.  Iteration count varies.

  (B) Basis preconditioning (BPX-style) -- transform the BASIS
      itself so the operator becomes diagonally dominant.  The
      truncation criterion top-|c_tilde| in the rescaled basis
      then captures more of the solution.  This is where MG-CG
      derives its power (BPX is a multi-level basis preconditioner).

This script implements both and reports them side-by-side, so the
v1.1+ plan can distinguish the two.

Convergence target: J_err < 5% AND g_err < 10% at k <= N/4.

Setup: same Haar basis as locality_theorem.py (N=256, FD Dirichlet
(-u'' + u), Gaussian source).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from locality_theorem import (
    N, W_HAAR, A_HAAR, EYE_N, source_grid, x_grid, SENSOR_IDX,
    J_haar_full, S_J, EIGVALS_SINE,
)

THETA = 0.42
theta = jnp.asarray(THETA)


# ---------- Haar level structure ----------
def haar_level_of_idx():
    """Index -> wavelet level (-1 for scaling, 0 for global wavelet,
    j>=1 for level-j detail)."""
    levels = np.zeros(N, dtype=int)
    levels[0] = -1
    levels[1] = 0
    for lvl in range(1, int(np.log2(N))):
        start = 2 ** lvl
        end = 2 ** (lvl + 1)
        levels[start:end] = lvl
    return levels


LEVEL = haar_level_of_idx()

# Precompute level-mean diagonal scales once (static).
_A_DIAG_NP = np.asarray(jnp.diag(A_HAAR))
_LEVEL_SCALES_NP = np.ones(N)
for _lvl in np.unique(LEVEL):
    _sel = LEVEL == _lvl
    _LEVEL_SCALES_NP[_sel] = _A_DIAG_NP[_sel].mean()
LEVEL_SCALES = jnp.asarray(_LEVEL_SCALES_NP)


# ---------- (A) Iteration preconditioning: lineax GMRES ----------

def jit_masked_lineax(precond_kind: str):
    """Build a JIT-traced solver: (theta, k) -> (J, n_iters)."""
    import lineax as lx

    A_diag = jnp.diag(A_HAAR)

    @jax.jit
    def solve(theta, k_active):
        b_full = W_HAAR @ source_grid(theta)
        c_for_mask = jnp.linalg.solve(A_HAAR, b_full)
        mag = jnp.abs(c_for_mask)
        threshold = jnp.sort(mag)[-k_active]
        mask = jax.lax.stop_gradient(mag >= threshold)
        A_eff = jnp.where(mask[:, None], mask[None, :] * A_HAAR, EYE_N)
        b_eff = mask * b_full

        # Preconditioned matrix-vector product
        if precond_kind == "none":
            matvec = lambda v: A_eff @ v
        elif precond_kind == "jacobi":
            diag_eff = jnp.where(mask, A_diag, 1.0)
            # Left preconditioning: solve M^{-1} A x = M^{-1} b
            # where M = diag(A_eff).  Equivalent to scaling rows.
            matvec = lambda v: (A_eff @ v) / diag_eff
            b_eff = b_eff / diag_eff
        elif precond_kind == "level":
            diag_eff = jnp.where(mask, LEVEL_SCALES, 1.0)
            matvec = lambda v: (A_eff @ v) / diag_eff
            b_eff = b_eff / diag_eff
        else:
            raise ValueError(precond_kind)

        op = lx.FunctionLinearOperator(matvec, jax.eval_shape(lambda: b_eff))
        solver = lx.GMRES(rtol=1e-8, atol=1e-10, restart=min(N, 50),
                          max_steps=400)
        result = lx.linear_solve(op, b_eff, solver=solver, throw=False)
        u = W_HAAR.T @ result.value
        n_iters = result.stats.get("num_steps", -1) if result.stats else -1
        return u[SENSOR_IDX], jnp.asarray(n_iters)

    return solve


# ---------- (B) Basis preconditioning (Jacobi-symmetric BPX-lite) ----------

def J_haar_basis_jacobi(theta, k_active):
    """Symmetric Jacobi basis transform: A_tilde = D^{-1/2} A D^{-1/2}.
    Top-|c_tilde| selection in the rescaled basis."""
    D_sqrt = jnp.sqrt(jnp.abs(jnp.diag(A_HAAR)))
    D_inv_sqrt = 1.0 / D_sqrt
    A_tilde = D_inv_sqrt[:, None] * A_HAAR * D_inv_sqrt[None, :]
    b_phys = W_HAAR @ source_grid(theta)
    b_tilde = D_inv_sqrt * b_phys
    c_tilde_full = jnp.linalg.solve(A_tilde, b_tilde)
    mag_tilde = jnp.abs(c_tilde_full)
    threshold = jnp.sort(mag_tilde)[-k_active]
    mask = jax.lax.stop_gradient(mag_tilde >= threshold)
    A_eff = jnp.where(mask[:, None], mask[None, :] * A_tilde, EYE_N)
    b_eff = mask * b_tilde
    c_tilde = jnp.linalg.solve(A_eff, b_eff)
    # Back-transform: c_orig = D^{-1/2} c_tilde
    c_orig = D_inv_sqrt * c_tilde
    u = W_HAAR.T @ c_orig
    return u[SENSOR_IDX]


def J_haar_basis_level(theta, k_active):
    """Level-block symmetric preconditioning."""
    D_sqrt = jnp.sqrt(LEVEL_SCALES)
    D_inv_sqrt = 1.0 / D_sqrt
    A_tilde = D_inv_sqrt[:, None] * A_HAAR * D_inv_sqrt[None, :]
    b_phys = W_HAAR @ source_grid(theta)
    b_tilde = D_inv_sqrt * b_phys
    c_tilde_full = jnp.linalg.solve(A_tilde, b_tilde)
    mag_tilde = jnp.abs(c_tilde_full)
    threshold = jnp.sort(mag_tilde)[-k_active]
    mask = jax.lax.stop_gradient(mag_tilde >= threshold)
    A_eff = jnp.where(mask[:, None], mask[None, :] * A_tilde, EYE_N)
    b_eff = mask * b_tilde
    c_tilde = jnp.linalg.solve(A_eff, b_eff)
    c_orig = D_inv_sqrt * c_tilde
    u = W_HAAR.T @ c_orig
    return u[SENSOR_IDX]


grad_basis_jacobi = jax.jit(jax.grad(J_haar_basis_jacobi), static_argnums=1)
grad_basis_level = jax.jit(jax.grad(J_haar_basis_level), static_argnums=1)
grad_haar_full_local = jax.jit(jax.grad(J_haar_full))


def relerr(x, ref):
    return float(abs(x - ref) / (abs(ref) + 1e-30))


def part_a_iteration_precond():
    """(A) iteration preconditioning -- expected: same J at convergence."""
    print("\n# Part A -- iteration preconditioning (GMRES + diagonal M)")
    print("# Expectation: J at convergence is identical across "
          "preconditioners;")
    print("# only the iteration count varies.")
    print()
    print(f"{'k_active':>8} | {'no_precond':>20} | {'jacobi':>20} | "
          f"{'level':>20}")
    print(f"{'':>8} | {'J        iters':>20} | "
          f"{'J        iters':>20} | "
          f"{'J        iters':>20}")
    solvers = {kind: jit_masked_lineax(kind)
               for kind in ["none", "jacobi", "level"]}
    for k in [N // 16, N // 8, N // 4, N // 2]:
        cells = []
        for kind in ["none", "jacobi", "level"]:
            J_val, n_iters = solvers[kind](theta, k)
            cells.append((float(J_val), int(n_iters)))
        cols = [f"{J:+.4e}  {ni:>4d}" for (J, ni) in cells]
        print(f"{k:>8d} | {cols[0]:>20} | {cols[1]:>20} | {cols[2]:>20}")


def part_b_basis_precond():
    """(B) basis preconditioning -- top-|c_tilde| in rescaled basis."""
    print("\n# Part B -- basis preconditioning (BPX-lite)")
    print("# A_tilde = D^{-1/2} A D^{-1/2}; truncate on |c_tilde|.")
    print()
    J_full_val = float(J_haar_full(theta))
    g_full_val = float(grad_haar_full_local(theta))
    print(f"# theta={THETA}: J_full = {J_full_val:+.6e}, "
          f"dJ/dtheta_full = {g_full_val:+.6e}")
    print()
    print(f"{'k_active':>8} | {'JACOBI BASIS':>26} | "
          f"{'LEVEL BASIS':>26}")
    print(f"{'':>8} | {'J_err':>10} {'g_err':>10}    | "
          f"{'J_err':>10} {'g_err':>10}    ")
    for k in [N // 16, N // 8, N // 4, N // 2]:
        Jj = float(J_haar_basis_jacobi(theta, k))
        gj = float(grad_basis_jacobi(theta, k))
        Jl = float(J_haar_basis_level(theta, k))
        gl = float(grad_basis_level(theta, k))
        print(f"{k:>8d} | {relerr(Jj, J_full_val):>10.3e} "
              f"{relerr(gj, g_full_val):>10.3e}    | "
              f"{relerr(Jl, J_full_val):>10.3e} "
              f"{relerr(gl, g_full_val):>10.3e}    ")


def part_c_adjoint_correctness():
    """(C) verify FD agrees with jax.grad through preconditioned solve."""
    print("\n# Part C -- preconditioned adjoint correctness")
    print("# FD vs jax.grad on the basis-preconditioned solve at "
          "theta=0.42, k=N/4")
    print()
    k = N // 4
    h = 1e-5
    fd_j = float((J_haar_basis_jacobi(theta + h, k)
                  - J_haar_basis_jacobi(theta - h, k)) / (2 * h))
    g_j = float(grad_basis_jacobi(theta, k))
    err_j = abs(fd_j - g_j) / (abs(fd_j) + 1e-30)
    fd_l = float((J_haar_basis_level(theta + h, k)
                  - J_haar_basis_level(theta - h, k)) / (2 * h))
    g_l = float(grad_basis_level(theta, k))
    err_l = abs(fd_l - g_l) / (abs(fd_l) + 1e-30)
    print(f"  JACOBI BASIS:  jax.grad = {g_j:+.6e}, FD = {fd_j:+.6e}, "
          f"rel_err = {err_j:.2e}")
    print(f"  LEVEL BASIS :  jax.grad = {g_l:+.6e}, FD = {fd_l:+.6e}, "
          f"rel_err = {err_l:.2e}")


if __name__ == "__main__":
    part_a_iteration_precond()
    part_b_basis_precond()
    part_c_adjoint_correctness()
