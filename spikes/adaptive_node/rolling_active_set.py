"""Round-3 Investigation 3: rolling active set top-|c_prev| vs
top-|b| and top-|c_current| (oracle).

Setup: sine basis 1D Poisson + Gaussian source (same as trap
characterisation).  Smooth theta trajectory theta(t) = 0.3 +
0.3 * sin(2 pi t / T) with T = 30 timesteps.

Three selection strategies at each step:
  (i)  top-|b|         -- RHS magnitude (round-2 baseline)
  (ii) top-|c_prev|    -- previous-timestep solution magnitudes
  (iii) top-|c_current| -- oracle, requires full preliminary solve

Parts:
  A. Lag under smooth motion -- which strategy tracks the oracle?
  B. Symmetry-trap interaction -- does warm-start escape?
  C. Cold-start protocol -- what to do at step 0?
  D. (memo only)  Combined recommendation.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from trap_characterisation import (
    N_BASIS, N_GRID, SIGMA, SENSOR_IDX, K_ACTIVE,
    ks, LAMBDAS, x_grid, dx, PHI,
    rhs_coeffs, J_full, J_with_mask, grad_with_mask, topk_mask,
)

T = 30  # number of timesteps


def trajectory(t):
    return 0.3 + 0.3 * np.sin(2.0 * np.pi * t / T)


def solve_for_c(theta, mask):
    """Full c vector under given mask."""
    b = np.asarray(rhs_coeffs(jnp.asarray(theta)))
    c = np.where(mask, b / np.asarray(LAMBDAS), 0.0)
    return c


def J_under_mask(theta, mask):
    return float(J_with_mask(jnp.asarray(theta), jnp.asarray(mask)))


def J_full_value(theta):
    return float(J_full(jnp.asarray(theta)))


def relerr(x, ref):
    return abs(x - ref) / (abs(ref) + 1e-30)


# ----- Part A: smooth-trajectory lag -----
def part_a():
    print("# Part A -- lag under smooth theta motion")
    print(f"# theta(t) = 0.3 + 0.3 sin(2pi t/T), T={T}, K={K_ACTIVE}")
    print()
    # For each strategy run the trajectory and record J_err at each step.
    results = {"top|b|": [], "top|c_prev|": [], "top|c_cur| (oracle)": []}
    c_prev = None
    for t in range(T):
        theta_t = trajectory(t)
        b = np.asarray(rhs_coeffs(jnp.asarray(theta_t)))
        J_ref = J_full_value(theta_t)

        # (i) top-|b|
        mask_b = topk_mask(np.abs(b), K_ACTIVE)
        J_b = J_under_mask(theta_t, mask_b)
        results["top|b|"].append(relerr(J_b, J_ref))

        # (ii) top-|c_prev| (cold-start = top-|b| on step 0)
        if c_prev is None:
            mask_cp = mask_b
        else:
            mask_cp = topk_mask(np.abs(c_prev), K_ACTIVE)
        J_cp = J_under_mask(theta_t, mask_cp)
        results["top|c_prev|"].append(relerr(J_cp, J_ref))

        # (iii) top-|c_current| oracle
        c_full = b / np.asarray(LAMBDAS)
        mask_co = topk_mask(np.abs(c_full), K_ACTIVE)
        J_co = J_under_mask(theta_t, mask_co)
        results["top|c_cur| (oracle)"].append(relerr(J_co, J_ref))

        # update c_prev: use current full solution to seed next step
        c_prev = c_full

    print(f"{'t':>3} {'theta':>7} {'top|b|':>11} "
          f"{'top|c_prev|':>13} {'oracle':>11}  "
          f"{'lag_factor':>11}")
    for t in range(T):
        theta_t = trajectory(t)
        lb, lc, lo = results["top|b|"][t], results["top|c_prev|"][t], \
            results["top|c_cur| (oracle)"][t]
        lag = lc / (lo + 1e-30)
        print(f"{t:>3d} {theta_t:>7.4f} {lb:>11.3e} "
              f"{lc:>13.3e} {lo:>11.3e}  {lag:>11.2f}")
    print()
    print("  summary:")
    lags = [results['top|c_prev|'][t] /
            (results['top|c_cur| (oracle)'][t] + 1e-30)
            for t in range(T)]
    print(f"    mean lag (c_prev / oracle): {np.mean(lags):.2f}")
    # threshold-velocity: at what speed does c_prev lag?
    dtheta = np.diff([trajectory(t) for t in range(T)])
    print(f"    theta velocity range: |dtheta/dt| in "
          f"[{abs(dtheta).min():.4e}, {abs(dtheta).max():.4e}]")


# ----- Part B: symmetry trap warm-start -----
def part_b():
    print("\n# Part B -- symmetry-trap warm-start interaction")
    print("# theta_0 = 0.5; first step cold-start top-|b| (trap),")
    print("# subsequent steps top-|c_prev|.  Does it escape?")
    print()

    theta = 0.5
    lr = 0.04
    c_prev = None
    history = [theta]
    for step in range(30):
        b = np.asarray(rhs_coeffs(jnp.asarray(theta)))
        if c_prev is None:
            mask = topk_mask(np.abs(b), K_ACTIVE)
        else:
            mask = topk_mask(np.abs(c_prev), K_ACTIVE)
        g = float(grad_with_mask(jnp.asarray(theta), jnp.asarray(mask)))
        c_prev = np.where(mask, b / np.asarray(LAMBDAS), 0.0)
        theta = theta - lr * g
        history.append(theta)

    print(f"  rolling top-|c_prev| from theta_0=0.5:")
    print(f"    theta trajectory: {['%.6f' % x for x in history[::5]]} "
          f"(every 5 steps)")
    print(f"    moved: {abs(history[-1] - history[0]) > 1e-4}")
    print(f"    final theta: {history[-1]}")
    print()
    print("  Diagnosis -- mask comparison at step 0 vs step 1:")
    b0 = np.asarray(rhs_coeffs(jnp.asarray(0.5)))
    mask_step0 = topk_mask(np.abs(b0), K_ACTIVE)
    c0 = np.where(mask_step0, b0 / np.asarray(LAMBDAS), 0.0)
    mask_step1 = topk_mask(np.abs(c0), K_ACTIVE)
    print(f"    step 0 mask = top-{K_ACTIVE} by |b|:    "
          f"{int(mask_step0.sum())} modes, "
          f"odd-k count: {int(mask_step0[::2].sum())}")  # ks index 0,2,4... = k=1,3,5...
    print(f"    step 1 mask = top-{K_ACTIVE} by |c_prev|: "
          f"{int(mask_step1.sum())} modes, "
          f"odd-k count: {int(mask_step1[::2].sum())}")
    print(f"    masks identical: {bool(np.array_equal(mask_step0, mask_step1))}")
    print()
    print("  -> rolling top-|c_prev| does NOT escape the trap because the")
    print("     trap state's c is supported on the same odd-k modes as |b|,")
    print("     so top-|c_prev| reproduces the trap mask exactly.")


# ----- Part C: cold-start protocol -----
def part_c():
    print("\n# Part C -- cold-start protocol comparison")
    print(f"# At step 0 we have no c_prev.  Test 3 strategies for the first")
    print(f"# step's J_err vs the full-basis solution.")
    print()
    test_thetas = [0.30, 0.42, 0.55, 0.5]
    print(f"  {'theta':>7} | {'(i) top|b|':>14} {'(ii) coarse':>14} "
          f"{'(iii) rand x5':>16}")
    rng = np.random.default_rng(seed=12345)
    for theta_val in test_thetas:
        b = np.asarray(rhs_coeffs(jnp.asarray(theta_val)))
        J_ref = J_full_value(theta_val)

        # (i) top-|b|
        mask_i = topk_mask(np.abs(b), K_ACTIVE)
        J_i = J_under_mask(theta_val, mask_i)
        err_i = relerr(J_i, J_ref)

        # (ii) coarse k=N/4 preliminary -> seed for k=K_ACTIVE
        mask_coarse = topk_mask(np.abs(b), N_BASIS // 4)
        c_coarse = np.where(mask_coarse, b / np.asarray(LAMBDAS), 0.0)
        # Now pick top-K_ACTIVE by |c_coarse|
        mask_ii = topk_mask(np.abs(c_coarse), K_ACTIVE)
        J_ii = J_under_mask(theta_val, mask_ii)
        err_ii = relerr(J_ii, J_ref)

        # (iii) random x5 -- best of 5
        best_err = float("inf")
        for _ in range(5):
            idx = rng.choice(N_BASIS, size=K_ACTIVE, replace=False)
            mask_rand = np.zeros(N_BASIS, dtype=bool)
            mask_rand[idx] = True
            J_r = J_under_mask(theta_val, mask_rand)
            err_r = relerr(J_r, J_ref)
            if err_r < best_err:
                best_err = err_r
        print(f"  {theta_val:>7.4f} | {err_i:>14.3e} {err_ii:>14.3e} "
              f"{best_err:>16.3e}")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
