"""Round-6 Investigation 2: correct CDD implementation + trap immunity test.

Implements the three-step Cohen-Dahmen-DeVore algorithm:
  APPLY:  r = b - A_Λ c_Λ
  GROW:   bulk-chasing Doerfler: add smallest set with sum |r_i|^2
          >= theta_D * sum |r|^2
  COARSE: prune indices where |c_i| < eps_coarse * max |c|

Parts A (Haar 1D convergence), B (smooth trajectory comparison),
C (TRAP IMMUNITY at theta=0.5), D (theta_D, eps_coarse sensitivity).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from locality_theorem import (
    N, W_HAAR, A_HAAR, EYE_N, source_grid, x_grid, SENSOR_IDX,
    J_haar_full,
)


def haar_solve_on_mask(b, mask):
    """Solve A_HAAR[mask, mask] c[mask] = b[mask]; c[~mask] = 0."""
    A_arr = np.asarray(A_HAAR)
    A_eff = np.where(mask[:, None], mask[None, :] * A_arr, np.eye(N))
    b_eff = mask * b
    return np.linalg.solve(A_eff, b_eff)


def grow_doerfler(r, mask, theta_D):
    """Doerfler bulk-chasing: add indices NOT in mask such that
    sum of |r_added|^2 >= theta_D * sum |r|^2.  Returns updated mask."""
    r_sq = r * r
    target = theta_D * r_sq.sum()
    # exclude already-active indices from candidacy
    candidate = ~mask
    cand_sq = np.where(candidate, r_sq, 0.0)
    # sort descending
    sorted_idx = np.argsort(-cand_sq)
    cumsum = np.cumsum(cand_sq[sorted_idx])
    # find smallest count whose cumsum reaches target
    above = cumsum >= target
    if not above.any():
        # add ALL candidates if target unattainable
        n_add = int(candidate.sum())
    else:
        n_add = int(np.argmax(above)) + 1
    new_mask = mask.copy()
    if n_add > 0:
        new_mask[sorted_idx[:n_add]] = True
    return new_mask, n_add


def coarse_prune(c, mask, eps_coarse):
    """Prune mask indices where |c_i| < eps_coarse * max |c|."""
    if not mask.any():
        return mask
    mag = np.abs(c)
    max_c = mag.max()
    if max_c == 0:
        return mask
    keep = mag >= eps_coarse * max_c
    return mask & keep


def cdd_solve(b, *, theta_D=0.5, eps_coarse=0.1, rtol=1e-4,
              max_outer=20, init_mask=None, verbose=False):
    """Returns (final_mask, c, n_iters, residual_history)."""
    A_arr = np.asarray(A_HAAR)
    if init_mask is None:
        mask = np.zeros(N, dtype=bool)
    else:
        mask = init_mask.copy()
    if mask.any():
        c = haar_solve_on_mask(b, mask)
    else:
        c = np.zeros(N)
    b_norm = np.linalg.norm(b)
    res_history = []
    for it in range(max_outer):
        r = b - A_arr @ c
        rn = np.linalg.norm(r)
        res_history.append(rn)
        if verbose:
            n_active = int(mask.sum())
            print(f"    iter {it}: |Lambda|={n_active}, "
                  f"||r||/||b||={rn/b_norm:.3e}")
        if rn / (b_norm + 1e-30) < rtol:
            break
        # GROW
        mask_g, n_add = grow_doerfler(r, mask, theta_D)
        # SOLVE
        c = haar_solve_on_mask(b, mask_g)
        # COARSE
        mask = coarse_prune(c, mask_g, eps_coarse)
        # re-solve after coarsening if pruned
        if (mask != mask_g).any():
            c = haar_solve_on_mask(b, mask)
    return mask, c, it + 1, res_history


# ---- Part A ----
def part_a():
    print("# Part A -- correct CDD implementation on Haar 1D")
    print()
    theta = 0.42
    b = np.asarray(W_HAAR @ source_grid(jnp.asarray(theta)))
    print(f"  theta = {theta}, N = {N}, theta_D = 0.5, "
          f"eps_coarse = 0.1, rtol = 1e-4")
    mask, c, n_iters, history = cdd_solve(b, verbose=True)
    print(f"\n  Final |Lambda| = {int(mask.sum())} / {N}")
    print(f"  Outer iterations: {n_iters}")
    # J and gradient evaluation
    u = np.asarray(W_HAAR).T @ c
    J_cdd = float(u[SENSOR_IDX])
    J_ref = float(J_haar_full(jnp.asarray(theta)))
    err = abs(J_cdd - J_ref) / (abs(J_ref) + 1e-30)
    print(f"  J_cdd = {J_cdd:+.5e}, J_full = {J_ref:+.5e}, "
          f"rel_err = {err:.3e}")
    print()
    # Compare to oracle top-|c|
    c_full = np.linalg.solve(np.asarray(A_HAAR), b)
    sorted_c = np.sort(np.abs(c_full))
    K_target = int(mask.sum())   # match the size CDD picked
    thresh = sorted_c[-K_target]
    mask_or = np.abs(c_full) >= thresh
    c_or = haar_solve_on_mask(b, mask_or)
    u_or = np.asarray(W_HAAR).T @ c_or
    J_or = float(u_or[SENSOR_IDX])
    err_or = abs(J_or - J_ref) / (abs(J_ref) + 1e-30)
    print(f"  Oracle top-|c| at |Lambda|={K_target}: "
          f"J_err = {err_or:.3e}")


# ---- Part B (smooth trajectory) ----
def part_b():
    print("\n\n# Part B -- CDD vs rolling on smooth trajectory")
    print()
    T = 30
    def traj(t): return 0.3 + 0.3 * np.sin(2 * np.pi * t / T)
    cdd_errs = []
    cdd_iters = []
    roll_errs = []
    A_arr = np.asarray(A_HAAR)
    prev_mask = None
    c_prev = None
    for t in range(T):
        theta = traj(t)
        b = np.asarray(W_HAAR @ source_grid(jnp.asarray(theta)))
        J_ref = float(J_haar_full(jnp.asarray(theta)))
        # CDD with warm start
        mask, c, n_iters, _ = cdd_solve(b, init_mask=prev_mask,
                                         max_outer=10)
        u = np.asarray(W_HAAR).T @ c
        J_cdd = float(u[SENSOR_IDX])
        cdd_errs.append(abs(J_cdd - J_ref) / (abs(J_ref) + 1e-30))
        cdd_iters.append(n_iters)
        prev_mask = mask
        # rolling top-|c_prev|
        if c_prev is None:
            mag = np.abs(b)
        else:
            mag = np.abs(c_prev)
        K_BASE = 32
        sorted_m = np.sort(mag)
        roll_mask = mag >= sorted_m[-K_BASE]
        c_roll = haar_solve_on_mask(b, roll_mask)
        u_roll = np.asarray(W_HAAR).T @ c_roll
        J_roll = float(u_roll[SENSOR_IDX])
        roll_errs.append(abs(J_roll - J_ref) / (abs(J_ref) + 1e-30))
        c_prev = np.linalg.solve(A_arr, b)

    print(f"  T = {T}, K_base for rolling = 32 (Haar 1D)")
    print(f"  CDD:     mean J_err = {np.mean(cdd_errs):.3e}, "
          f"max = {max(cdd_errs):.3e}, mean iters = "
          f"{np.mean(cdd_iters):.1f}")
    print(f"  rolling: mean J_err = {np.mean(roll_errs):.3e}, "
          f"max = {max(roll_errs):.3e}")


# ---- Part C (TRAP IMMUNITY -- THE BIG ONE) ----
def part_c():
    print("\n\n# Part C -- TRAP IMMUNITY at theta = 0.5")
    print()
    print("  Setup: cold-start CDD (Lambda = empty) at theta = 0.5")
    print("  Question: does GROW ever add 'gradient-sensitive' modes?")
    print()

    # 1D Sine analog (using existing sine basis)
    from trap_characterisation import (
        N_BASIS, K_ACTIVE, rhs_coeffs, grad_with_mask, grad_full,
        LAMBDAS, PHI, SENSOR_IDX as S_IDX_1D,
    )
    print("## 1D sine basis (A is DIAGONAL: c_k = b_k / lambda_k)")
    theta = 0.5
    b_sine = np.asarray(rhs_coeffs(jnp.asarray(theta)))
    lam = np.asarray(LAMBDAS)
    # Sine CDD (much simpler -- no full matrix needed)
    mask = np.zeros(N_BASIS, dtype=bool)
    c = np.zeros(N_BASIS)
    b_norm = np.linalg.norm(b_sine)
    for it in range(10):
        # APPLY: r = b - diag(lambda)*c   (A is diagonal)
        r = b_sine - lam * c
        rn = np.linalg.norm(r)
        if rn / (b_norm + 1e-30) < 1e-4:
            break
        # GROW
        r_sq = r * r
        target = 0.5 * r_sq.sum()
        cand = ~mask
        cand_sq = np.where(cand, r_sq, 0.0)
        sorted_idx = np.argsort(-cand_sq)
        cumsum = np.cumsum(cand_sq[sorted_idx])
        n_add = int(np.argmax(cumsum >= target)) + 1
        new_mask = mask.copy()
        new_mask[sorted_idx[:n_add]] = True
        # SOLVE (diagonal trivial)
        c = np.where(new_mask, b_sine / lam, 0.0)
        # Count even-k in mask
        n_even = int(np.sum(new_mask[::2]))  # idx 0,2,4 = k=1,3,5 (odd)
        # Wait: ks = arange(1, N+1), so idx i -> k = i+1.  Even k means odd index.
        # idx 0 -> k=1 (odd), idx 1 -> k=2 (even).
        n_oddk = int(np.sum(new_mask[::2]))  # idx 0, 2, 4 -> k=1, 3, 5
        n_evenk = int(np.sum(new_mask[1::2]))  # idx 1, 3, 5 -> k=2, 4, 6
        print(f"  iter {it}: |Lambda|={int(new_mask.sum())}, "
              f"odd-k count={n_oddk}, even-k count={n_evenk}, "
              f"||r||={rn:.3e}")
        mask = new_mask
        if rn / b_norm < 1e-4:
            break

    g_frozen_x = float(grad_with_mask(jnp.asarray(theta),
                                       jnp.asarray(mask)))
    g_full = float(grad_full(jnp.asarray(theta)))
    print(f"\n  Final mask: odd-k={int(np.sum(mask[::2]))}, "
          f"even-k={int(np.sum(mask[1::2]))}")
    print(f"  Frozen-set gradient at theta=0.5: g_frozen = "
          f"{g_frozen_x:.4e}")
    print(f"  Full-basis gradient:               g_full   = "
          f"{g_full:.4e}")
    if abs(g_frozen_x) < 1e-10:
        print(f"  -> CDD in sine basis: TRAP NOT ESCAPED (g_frozen = 0)")
    else:
        print(f"  -> CDD in sine basis: ESCAPED (g_frozen nonzero)")

    print()
    print("## 1D Haar basis (A is NOT diagonal -- cross-level coupling)")
    theta_h = 0.5
    b_haar = np.asarray(W_HAAR @ source_grid(jnp.asarray(theta_h)))
    A_arr = np.asarray(A_HAAR)
    mask = np.zeros(N, dtype=bool)
    c = np.zeros(N)
    bn = np.linalg.norm(b_haar)
    for it in range(10):
        r = b_haar - A_arr @ c
        rn = np.linalg.norm(r)
        print(f"  iter {it}: |Lambda|={int(mask.sum())}, ||r||/||b||="
              f"{rn/bn:.3e}, ||r restricted to ~mask||="
              f"{np.linalg.norm(np.where(~mask, r, 0.0)):.3e}")
        if rn / bn < 1e-4:
            break
        mask, _ = grow_doerfler(r, mask, 0.5)
        c = haar_solve_on_mask(b_haar, mask)
    # check whether mask has support in V_G^perp
    # V_G = G-invariant subspace in Haar basis under x -> 1-x reflection.
    # Numerically: b_haar has nonzero only on V_G at theta=0.5.
    # So V_G = supp(|b_haar| > 1e-10).
    is_VG = np.abs(b_haar) > 1e-10
    n_in_VG = int(np.sum(mask & is_VG))
    n_in_VGperp = int(np.sum(mask & ~is_VG))
    print(f"\n  V_G (modes with |b_haar| > 1e-10): {int(is_VG.sum())} modes")
    print(f"  Final mask: in V_G = {n_in_VG}, in V_G^perp = {n_in_VGperp}")

    # Compute frozen-set gradient in Haar at theta=0.5
    def J_haar_frozen_fn(theta_in, mask_in):
        b = W_HAAR @ source_grid(theta_in)
        m = jax.lax.stop_gradient(jnp.asarray(mask_in))
        A_eff = jnp.where(m[:, None], m[None, :] * A_HAAR, EYE_N)
        b_eff = m * b
        c_loc = jnp.linalg.solve(A_eff, b_eff)
        u = W_HAAR.T @ c_loc
        return u[SENSOR_IDX]
    grad_fn = jax.jit(jax.grad(J_haar_frozen_fn, argnums=0))
    g_frozen_haar = float(grad_fn(jnp.asarray(theta_h), mask))
    g_full_haar = float(jax.grad(J_haar_full)(jnp.asarray(theta_h)))
    print(f"  g_frozen (Haar CDD at theta=0.5):  {g_frozen_haar:.4e}")
    print(f"  g_full (Haar full basis):          {g_full_haar:.4e}")
    if abs(g_frozen_haar) < 1e-10:
        print(f"  -> CDD in Haar basis: TRAP NOT ESCAPED")
    else:
        print(f"  -> CDD in Haar basis: ESCAPED")


# ---- Part D (theta_D, eps_coarse sensitivity) ----
def part_d():
    print("\n\n# Part D -- theta_D and eps_coarse sensitivity")
    print()
    theta = 0.42
    b = np.asarray(W_HAAR @ source_grid(jnp.asarray(theta)))
    J_ref = float(J_haar_full(jnp.asarray(theta)))
    print(f"  theta = {theta}, J_ref = {J_ref:+.5e}")
    print(f"  {'theta_D':>8} {'eps_c':>8} {'|Lambda|':>10} "
          f"{'iters':>6} {'J_err':>11}")
    for theta_D in [0.3, 0.5, 0.7]:
        for eps_coarse in [0.05, 0.1, 0.2]:
            mask, c, n_iters, _ = cdd_solve(
                b, theta_D=theta_D, eps_coarse=eps_coarse, rtol=1e-4
            )
            u = np.asarray(W_HAAR).T @ c
            J = float(u[SENSOR_IDX])
            err = abs(J - J_ref) / (abs(J_ref) + 1e-30)
            print(f"  {theta_D:>8.2f} {eps_coarse:>8.2f} "
                  f"{int(mask.sum()):>10d} {n_iters:>6d} {err:>11.3e}")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
