"""Round-7 Investigation 1: CDD with Dahmen-Kunoth scaling.

Round-6 finding: CDD without scaling diverges in Haar.  Round-5
subagent A says BPX for wavelets is trivial diagonal scaling
D_λλ = 2^|λ| (Dahmen-Kunoth 1992).  This investigation closes
the loop: does CDD-with-scaling actually outperform rolling?

Parts:
  A. Implement D_λλ = 2^|λ|, measure κ(A) vs κ(A_scaled) at N=64,128,256
  B. CDD-with-scaling on 1D Haar -- residual decreases? J_err vs rolling?
  C. 2D Haar Dahmen-Kunoth product scaling D = 2^(|λ_x| + |λ_y|)
  D. θ_D sensitivity post-scaling
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import os, sys
sys.path.insert(0, os.path.dirname(__file__))


def haar_matrix(n):
    W = np.eye(n)
    sz = n
    while sz > 1:
        H = np.zeros((sz, sz))
        inv = 1.0 / np.sqrt(2.0)
        for i in range(sz // 2):
            H[i, 2*i] = inv
            H[i, 2*i+1] = inv
            H[sz//2 + i, 2*i] = inv
            H[sz//2 + i, 2*i+1] = -inv
        W[:sz] = H @ W[:sz]
        sz //= 2
    return W


def haar_levels_1d(N):
    """Level index of each Haar basis function.  Convention:
    idx 0 = scaling, idx 1 = level-0 wavelet, then level-j has
    2^j wavelets at indices [2^j, 2^(j+1))."""
    levels = np.zeros(N, dtype=int)
    levels[0] = 0  # scaling
    levels[1] = 0  # coarsest wavelet
    for j in range(1, int(np.log2(N))):
        levels[2**j:2**(j+1)] = j
    return levels


def make_problem_1d(N, sigma=0.05):
    dx = 1.0 / (N + 1)
    x_grid = np.arange(1, N + 1) * dx
    A_PHYS = ((2.0/dx**2) * np.eye(N) + np.eye(N)
              - (1.0/dx**2) * (np.eye(N, k=1) + np.eye(N, k=-1)))
    W = haar_matrix(N)
    A_HAAR = W @ A_PHYS @ W.T
    levels = haar_levels_1d(N)
    D = 2.0 ** levels                    # Dahmen-Kunoth diagonal
    D_inv = 1.0 / D
    A_scaled = (D_inv[:, None]) * A_HAAR * (D_inv[None, :])
    return dict(N=N, dx=dx, x_grid=x_grid, W=W, A_PHYS=A_PHYS,
                A_HAAR=A_HAAR, levels=levels, D=D, A_scaled=A_scaled,
                sigma=sigma)


def part_a():
    print("# Part A -- condition number with vs without scaling")
    print()
    for N in [64, 128, 256]:
        p = make_problem_1d(N)
        kappa_unscaled = np.linalg.cond(p['A_HAAR'])
        kappa_scaled = np.linalg.cond(p['A_scaled'])
        print(f"  N = {N:>3d}:  κ(A_HAAR) = {kappa_unscaled:>10.3e}   "
              f"κ(A_scaled) = {kappa_scaled:>10.3e}   "
              f"ratio = {kappa_unscaled/kappa_scaled:>8.3f}")
    print()
    print("  Dahmen-Kunoth 1992 prediction: κ(A_scaled) = O(1), "
          "independent of N.")


def cdd_with_scaling(p, theta, theta_D=0.5, eps_coarse=0.1, rtol=1e-3,
                      max_outer=30, init_mask=None, verbose=False):
    """CDD APPLY/GROW/SOLVE/COARSE on the SCALED problem.
    Returns (mask, c_unscaled, n_iters, residual_history)."""
    N = p['N']
    W, A_scaled, D, D_inv = p['W'], p['A_scaled'], p['D'], 1.0/p['D']
    x_grid = p['x_grid']
    sigma = p['sigma']
    # Build b and scale
    f = np.exp(-((x_grid - theta) / sigma) ** 2)
    b_haar = W @ f
    b_scaled = D_inv * b_haar
    b_scaled_norm = np.linalg.norm(b_scaled)
    # Initialize
    if init_mask is None:
        mask = np.zeros(N, dtype=bool)
    else:
        mask = init_mask.copy()
    c_scaled = np.zeros(N)
    if mask.any():
        A_eff = np.where(mask[:, None], mask[None, :] * A_scaled, np.eye(N))
        c_scaled = np.linalg.solve(A_eff, mask * b_scaled)
    history = []
    for it in range(max_outer):
        r = b_scaled - A_scaled @ c_scaled
        rn = np.linalg.norm(r)
        rel_r = rn / (b_scaled_norm + 1e-30)
        history.append(rel_r)
        if verbose:
            print(f"    it {it}: |Λ|={int(mask.sum()):>3d}, "
                  f"||r||/||b||={rel_r:.3e}")
        if rel_r < rtol:
            break
        # GROW (Doerfler bulk chase)
        r_sq = r * r
        target = theta_D * r_sq.sum()
        cand_sq = np.where(mask, 0.0, r_sq)
        sorted_idx = np.argsort(-cand_sq)
        cumsum = np.cumsum(cand_sq[sorted_idx])
        above = cumsum >= target
        if not above.any():
            n_add = int((~mask).sum())
        else:
            n_add = int(np.argmax(above)) + 1
        new_mask = mask.copy()
        if n_add > 0:
            new_mask[sorted_idx[:n_add]] = True
        # SOLVE
        A_eff = np.where(new_mask[:, None],
                         new_mask[None, :] * A_scaled, np.eye(N))
        c_scaled = np.linalg.solve(A_eff, new_mask * b_scaled)
        # COARSE: prune based on |c_scaled|
        mag = np.abs(c_scaled)
        if mag.max() > 0:
            keep = mag >= eps_coarse * mag.max()
            new_mask = new_mask & keep
            # re-solve after pruning if changed
            if (new_mask != mask).any():
                A_eff = np.where(new_mask[:, None],
                                 new_mask[None, :] * A_scaled, np.eye(N))
                c_scaled = np.linalg.solve(A_eff, new_mask * b_scaled)
        mask = new_mask
    c_haar = D_inv * c_scaled
    return mask, c_haar, it + 1, history


def part_b():
    print("\n# Part B -- CDD-with-scaling on 1D Haar, smooth trajectory")
    print()
    N = 256
    p = make_problem_1d(N)
    THETA = 0.42
    print(f"## Single-point test at θ = {THETA}")
    mask, c, n_iters, hist = cdd_with_scaling(p, THETA, verbose=True)
    f = np.exp(-((p['x_grid'] - THETA) / p['sigma']) ** 2)
    u = p['W'].T @ c
    # sensor at N//3
    sensor_idx = N // 3
    J_cdd = float(u[sensor_idx])
    # Reference
    c_full = np.linalg.solve(p['A_HAAR'], p['W'] @ f)
    u_full = p['W'].T @ c_full
    J_full = float(u_full[sensor_idx])
    J_err = abs(J_cdd - J_full) / (abs(J_full) + 1e-30)
    print(f"\n  Final: |Λ|={int(mask.sum())}, n_iters={n_iters}, "
          f"J_err={J_err:.3e}")
    # Compare to oracle top-|c| at same |Λ|
    sorted_c = np.sort(np.abs(c_full))
    K = int(mask.sum())
    mask_or = np.abs(c_full) >= sorted_c[-K]
    A_eff_or = np.where(mask_or[:, None],
                        mask_or[None, :] * p['A_HAAR'], np.eye(N))
    c_or = np.linalg.solve(A_eff_or, mask_or * (p['W'] @ f))
    u_or = p['W'].T @ c_or
    J_or = float(u_or[sensor_idx])
    print(f"  Oracle |c| at K={K}: J_err = "
          f"{abs(J_or-J_full)/(abs(J_full)+1e-30):.3e}")

    # Trajectory
    print()
    print("## Smooth trajectory θ(t) = 0.3 + 0.3·sin(2π t/T), T=30")
    T = 30
    def traj(t): return 0.3 + 0.3 * np.sin(2 * np.pi * t / T)
    K_TARGET = 64
    cdd_errs, roll_errs, cdd_iters = [], [], []
    c_prev = None
    init_mask = None
    for t in range(T):
        theta = traj(t)
        f = np.exp(-((p['x_grid'] - theta) / p['sigma']) ** 2)
        b_haar = p['W'] @ f
        # CDD warm-start
        mask, c, n_it, _ = cdd_with_scaling(p, theta, init_mask=init_mask)
        u = p['W'].T @ c
        c_full = np.linalg.solve(p['A_HAAR'], b_haar)
        u_full = p['W'].T @ c_full
        J_f = float(u_full[sensor_idx])
        J_cdd = float(u[sensor_idx])
        cdd_errs.append(abs(J_cdd - J_f) / (abs(J_f) + 1e-30))
        cdd_iters.append(n_it)
        init_mask = mask
        # Rolling
        if c_prev is None:
            mag = np.abs(b_haar)
        else:
            mag = np.abs(c_prev)
        thr = np.sort(mag)[-K_TARGET]
        m_roll = mag >= thr
        A_eff = np.where(m_roll[:, None],
                         m_roll[None, :] * p['A_HAAR'], np.eye(N))
        c_roll = np.linalg.solve(A_eff, m_roll * b_haar)
        u_roll = p['W'].T @ c_roll
        roll_errs.append(abs(float(u_roll[sensor_idx]) - J_f)
                         / (abs(J_f) + 1e-30))
        c_prev = c_full
    print(f"  CDD-with-scaling:  mean J_err = {np.mean(cdd_errs):.3e}, "
          f"max = {max(cdd_errs):.3e}, mean iters = {np.mean(cdd_iters):.1f}")
    print(f"  rolling top-|c|:   mean J_err = {np.mean(roll_errs):.3e}, "
          f"max = {max(roll_errs):.3e}")
    print()
    cdd_better = np.mean(cdd_errs) < np.mean(roll_errs)
    print(f"  CDD better than rolling? {cdd_better}")


def part_c():
    print("\n\n# Part C -- 2D CDD with product scaling")
    print()
    N_1D = 32
    N = N_1D * N_1D
    sigma = 0.1
    dx = 1.0 / (N_1D + 1)
    x_grid = np.arange(1, N_1D + 1) * dx
    A_LAP = (2.0/dx**2) * np.eye(N_1D) - (1.0/dx**2) * (
        np.eye(N_1D, k=1) + np.eye(N_1D, k=-1))
    A_PHYS = (np.kron(A_LAP, np.eye(N_1D))
              + np.kron(np.eye(N_1D), A_LAP) + np.eye(N))
    W_1D = haar_matrix(N_1D)
    W_2D = np.kron(W_1D, W_1D)
    A_HAAR_2D = W_2D @ A_PHYS @ W_2D.T
    # 2D levels: idx i in flat 2D corresponds to (i_x, i_y) = (i // N_1D, i % N_1D)
    lev_1d = haar_levels_1d(N_1D)
    LEVELS_2D = (lev_1d[:, None] + lev_1d[None, :]).reshape(-1)
    D_2D = 2.0 ** LEVELS_2D
    D_inv = 1.0 / D_2D
    A_scaled = D_inv[:, None] * A_HAAR_2D * D_inv[None, :]

    print(f"  N = {N} ({N_1D}x{N_1D})")
    kappa_un = np.linalg.cond(A_HAAR_2D)
    kappa_sc = np.linalg.cond(A_scaled)
    print(f"  κ(A_HAAR_2D) = {kappa_un:.3e},   κ(A_scaled) = {kappa_sc:.3e}")
    print(f"  ratio = {kappa_un / kappa_sc:.3f}")

    THETA_X, THETA_Y = 0.42, 0.35
    SENSOR_X, SENSOR_Y = 0.7, 0.6
    sensor_ix = int(round(SENSOR_X / dx)) - 1
    sensor_iy = int(round(SENSOR_Y / dx)) - 1
    SENSOR_FLAT = sensor_ix * N_1D + sensor_iy

    X = x_grid[:, None]
    Y = x_grid[None, :]
    f = np.exp(-((X - THETA_X)**2 + (Y - THETA_Y)**2) / sigma**2).reshape(-1)
    b_haar = W_2D @ f
    b_scaled = D_inv * b_haar
    b_scaled_norm = np.linalg.norm(b_scaled)
    c_full = np.linalg.solve(A_HAAR_2D, b_haar)
    J_full = float((W_2D.T @ c_full)[SENSOR_FLAT])

    # CDD with k_active budget = 128
    mask = np.zeros(N, dtype=bool)
    c_scaled = np.zeros(N)
    THETA_D = 0.5
    EPS_COARSE = 0.1
    print()
    print(f"  CDD-with-scaling at θ=(0.42, 0.35), k_budget=128, θ_D={THETA_D}")
    for it in range(30):
        r = b_scaled - A_scaled @ c_scaled
        rn = np.linalg.norm(r) / (b_scaled_norm + 1e-30)
        if it < 8 or it == 29:
            print(f"    it {it}: |Λ|={int(mask.sum())}, ||r||/||b||={rn:.3e}")
        if rn < 1e-3:
            print(f"    it {it}: |Λ|={int(mask.sum())}, ||r||/||b||={rn:.3e} (converged)")
            break
        r_sq = r * r
        target = THETA_D * r_sq.sum()
        cand_sq = np.where(mask, 0.0, r_sq)
        sorted_idx = np.argsort(-cand_sq)
        cumsum = np.cumsum(cand_sq[sorted_idx])
        above = cumsum >= target
        n_add = int(np.argmax(above)) + 1 if above.any() else int((~mask).sum())
        new_mask = mask.copy()
        new_mask[sorted_idx[:n_add]] = True
        # Cap at 128
        if int(new_mask.sum()) > 128:
            cur_active_mag = np.where(new_mask, np.abs(c_scaled), -np.inf)
            new_active_idx = sorted_idx[:n_add]
            new_active_mag = np.abs(r[new_active_idx])
            # mix scores
            score_total = np.where(new_mask, 0.0, -np.inf)
            score_total[new_active_idx] = np.maximum(
                score_total[new_active_idx], new_active_mag)
            score_in = np.abs(c_scaled)
            score_combined = np.where(new_mask & ~mask,
                                       np.abs(r), score_in)
            top128 = np.argsort(-score_combined)[:128]
            new_mask = np.zeros(N, dtype=bool)
            new_mask[top128] = True
        A_eff = np.where(new_mask[:, None],
                         new_mask[None, :] * A_scaled, np.eye(N))
        c_scaled = np.linalg.solve(A_eff, new_mask * b_scaled)
        mag = np.abs(c_scaled)
        if mag.max() > 0:
            keep = mag >= EPS_COARSE * mag.max()
            new_mask = new_mask & keep
            A_eff = np.where(new_mask[:, None],
                             new_mask[None, :] * A_scaled, np.eye(N))
            c_scaled = np.linalg.solve(A_eff, new_mask * b_scaled)
        mask = new_mask
    c_final = D_inv * c_scaled
    u_final = W_2D.T @ c_final
    J_cdd = float(u_final[SENSOR_FLAT])
    err = abs(J_cdd - J_full) / (abs(J_full) + 1e-30)
    print(f"\n  Final |Λ|={int(mask.sum())}, J_err = {err:.3e}")
    # Compare to oracle top-|c|
    K_FINAL = int(mask.sum())
    if K_FINAL > 0:
        thr_or = np.sort(np.abs(c_full))[-K_FINAL]
        m_or = np.abs(c_full) >= thr_or
        A_eff_or = np.where(m_or[:, None],
                            m_or[None, :] * A_HAAR_2D, np.eye(N))
        c_or = np.linalg.solve(A_eff_or, m_or * b_haar)
        J_or = float((W_2D.T @ c_or)[SENSOR_FLAT])
        print(f"  Oracle |c| at K={K_FINAL}: J_err = "
              f"{abs(J_or-J_full)/(abs(J_full)+1e-30):.3e}")


def part_d():
    print("\n\n# Part D -- θ_D sensitivity post-scaling (1D)")
    print()
    N = 256
    p = make_problem_1d(N)
    THETA = 0.42
    print(f"  Sweep θ_D at θ={THETA}, ε_coarse=0.1, rtol=1e-3")
    print(f"  {'θ_D':>5} {'|Λ|':>5} {'iters':>6} {'J_err':>11}")
    sensor_idx = N // 3
    f = np.exp(-((p['x_grid'] - THETA) / p['sigma']) ** 2)
    c_full = np.linalg.solve(p['A_HAAR'], p['W'] @ f)
    J_full = float((p['W'].T @ c_full)[sensor_idx])
    for theta_D in [0.3, 0.5, 0.7]:
        mask, c, n_it, _ = cdd_with_scaling(p, THETA, theta_D=theta_D)
        J_cdd = float((p['W'].T @ c)[sensor_idx])
        err = abs(J_cdd - J_full) / (abs(J_full) + 1e-30)
        print(f"  {theta_D:>5.2f} {int(mask.sum()):>5d} {n_it:>6d} "
              f"{err:>11.3e}")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
