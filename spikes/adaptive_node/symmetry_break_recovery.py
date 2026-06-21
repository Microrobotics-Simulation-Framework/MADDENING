"""Round-7 Investigation 2: symmetry_break end-to-end recovery test.

Protocol:
  1. Detect trap via blindness_ratio < threshold (0.7 from round-6)
  2. Compute g_full at theta (one full-basis solve)
  3. Perturb: theta_new = theta + delta * g_full / ||g_full||
  4. Resume optimisation from theta_new

Parts:
  A. delta calibration
  B. recovery speed
  C. multiple perturbations needed?
  D. (memo only) cost accounting
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from trap_characterisation import (
    K_ACTIVE, rhs_coeffs, J_with_mask, grad_with_mask, grad_full, J_full,
    LAMBDAS, x_grid, PHI, SENSOR_IDX, topk_mask,
)
from locality_2d import (
    N as N2, N_1D, source_grid_2d, J_sine_full as J_sine_full_2d,
    J_sine_frozen as J_sine_frozen_2d, SENSOR_FLAT,
)


def blind_1d(theta, k=K_ACTIVE):
    b = np.asarray(rhs_coeffs(jnp.asarray(theta)))
    mask = topk_mask(np.abs(b), k)
    g_fr = float(grad_with_mask(jnp.asarray(theta), jnp.asarray(mask)))
    g_f = float(grad_full(jnp.asarray(theta)))
    return abs(g_fr) / (abs(g_f) + 1e-30), g_fr, g_f


def blind_2d(tx, ty, k=N2 // 8):
    gx_f = float(jax.grad(J_sine_full_2d, argnums=0)(jnp.asarray(tx),
                                                      jnp.asarray(ty)))
    gy_f = float(jax.grad(J_sine_full_2d, argnums=1)(jnp.asarray(tx),
                                                      jnp.asarray(ty)))
    gx_fr = float(jax.grad(J_sine_frozen_2d, argnums=0)(
        jnp.asarray(tx), jnp.asarray(ty), k, "b"))
    gy_fr = float(jax.grad(J_sine_frozen_2d, argnums=1)(
        jnp.asarray(tx), jnp.asarray(ty), k, "b"))
    nf = np.hypot(gx_f, gy_f)
    nr = np.hypot(gx_fr, gy_fr)
    return nr / (nf + 1e-30), (gx_fr, gy_fr), (gx_f, gy_f)


# ---- Part A ----
def part_a():
    print("# Part A -- delta calibration (1D sine, theta=0.5 trap)")
    print()
    theta = 0.5
    ratio0, g_fr, g_f = blind_1d(theta)
    print(f"  At theta = 0.5:  g_full = {g_f:+.4e}, ratio = {ratio0:.4f}")
    direction = np.sign(g_f)   # 1D scalar; unit g_full direction
    print(f"  Perturbation direction = sign(g_full) = {direction:+.0f}")
    print()
    print(f"  {'delta':>8}  {'theta_new':>10}  {'ratio':>8}  "
          f"{'passes 0.7':>11}  {'J_err':>10}")
    min_pass = None
    for delta in [1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2]:
        theta_new = theta + delta * direction
        ratio_new, _, _ = blind_1d(theta_new)
        J_new = float(J_full(jnp.asarray(theta_new)))
        J_ref = float(J_full(jnp.asarray(theta)))
        passes = ratio_new > 0.7
        if passes and min_pass is None:
            min_pass = delta
        print(f"  {delta:>8.0e}  {theta_new:>10.5f}  {ratio_new:>8.4f}  "
              f"{str(passes):>11}  "
              f"{abs(J_new-J_ref)/(abs(J_ref)+1e-30):>10.3e}")
    print(f"\n  Minimum delta for ratio > 0.7 in 1D: "
          f"{min_pass if min_pass else '> 0.05'}")

    # 2D trap at (0.5, 0.5)
    print()
    print("## 2D sine, theta = (0.5, 0.5) trap")
    ratio0, _, g_f = blind_2d(0.5, 0.5)
    nf = np.hypot(g_f[0], g_f[1])
    ux, uy = g_f[0] / (nf + 1e-30), g_f[1] / (nf + 1e-30)
    print(f"  g_full = ({g_f[0]:+.4e}, {g_f[1]:+.4e}), ratio = {ratio0:.4f}")
    print(f"  Unit direction = ({ux:+.4f}, {uy:+.4f})")
    print()
    print(f"  {'delta':>8}  {'theta_new':>17}  {'ratio':>8}  "
          f"{'passes 0.7':>11}")
    min_pass_2d = None
    for delta in [1e-3, 5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 7e-2]:
        tx_new = 0.5 + delta * ux
        ty_new = 0.5 + delta * uy
        ratio_new, _, _ = blind_2d(tx_new, ty_new)
        passes = ratio_new > 0.7
        if passes and min_pass_2d is None:
            min_pass_2d = delta
        print(f"  {delta:>8.0e}  ({tx_new:.4f}, {ty_new:.4f})  "
              f"{ratio_new:>8.4f}  {str(passes):>11}")
    print(f"\n  Minimum delta for ratio > 0.7 in 2D: "
          f"{min_pass_2d if min_pass_2d else '> 0.07'}")
    return min_pass or 0.04, min_pass_2d or 0.05


# ---- Part B: recovery speed ----
def part_b(delta_1d, delta_2d):
    print("\n\n# Part B -- recovery speed after perturbation")
    print()
    # 1D
    theta = 0.5
    _, _, g_f = blind_1d(theta)
    direction = np.sign(g_f)
    theta_new = theta + delta_1d * direction
    print(f"## 1D: theta_0 = 0.5, perturbed to theta_new = {theta_new:.4f}")
    LR = 0.04
    N_STEPS = 50
    history = [theta_new]
    ratios_history = []
    consecutive_good = 0
    first_consecutive_5 = None
    cur = theta_new
    for step in range(N_STEPS):
        b = np.asarray(rhs_coeffs(jnp.asarray(cur)))
        mask = topk_mask(np.abs(b), K_ACTIVE)
        g = float(grad_with_mask(jnp.asarray(cur), jnp.asarray(mask)))
        ratio, _, _ = blind_1d(cur)
        ratios_history.append(ratio)
        if ratio > 0.7:
            consecutive_good += 1
            if consecutive_good >= 5 and first_consecutive_5 is None:
                first_consecutive_5 = step - 4   # back-date
        else:
            consecutive_good = 0
        # ASCENT (maximize J)
        cur = cur + LR * g
        history.append(cur)
    print(f"  Steps to 5 consecutive ratio > 0.7: "
          f"{first_consecutive_5 if first_consecutive_5 is not None else '>= 50'}")
    print(f"  Trajectory: theta = {[f'{t:.4f}' for t in history[::10]]} "
          f"(every 10 steps)")
    print(f"  Ratios:     {[f'{r:.3f}' for r in ratios_history[::10]]} "
          f"(every 10 steps)")
    # drift back?
    drift_back = any(abs(t - 0.5) < abs(theta_new - 0.5)
                     for t in history[1:])
    print(f"  Drift back toward Fix(G) = 0.5? {drift_back}")

    # 2D
    print()
    print(f"## 2D: theta_0 = (0.5, 0.5), perturbed by delta = {delta_2d}")
    _, _, g_f = blind_2d(0.5, 0.5)
    nf = np.hypot(g_f[0], g_f[1])
    ux, uy = g_f[0]/nf, g_f[1]/nf
    tx_new = 0.5 + delta_2d * ux
    ty_new = 0.5 + delta_2d * uy
    print(f"  theta_new = ({tx_new:.4f}, {ty_new:.4f})")
    cur_x, cur_y = tx_new, ty_new
    history2d = [(cur_x, cur_y)]
    ratios2d = []
    consec_good_2d = 0
    first_5_2d = None
    for step in range(N_STEPS):
        ratio, g_fr, _ = blind_2d(cur_x, cur_y)
        ratios2d.append(ratio)
        if ratio > 0.7:
            consec_good_2d += 1
            if consec_good_2d >= 5 and first_5_2d is None:
                first_5_2d = step - 4
        else:
            consec_good_2d = 0
        gx, gy = g_fr
        cur_x = cur_x + LR * gx
        cur_y = cur_y + LR * gy
        history2d.append((cur_x, cur_y))
    print(f"  Steps to 5 consecutive ratio > 0.7: "
          f"{first_5_2d if first_5_2d is not None else '>= 50'}")
    print(f"  Trajectory: {[(round(x,3), round(y,3)) for (x,y) in history2d[::10]]}")
    print(f"  Ratios: {[f'{r:.3f}' for r in ratios2d[::10]]}")
    return first_consecutive_5, first_5_2d


# ---- Part C: multiple perturbations ----
def part_c(delta_1d):
    print("\n\n# Part C -- multiple perturbations needed?")
    print()
    print("## 1D: starting at theta=0.5, perturb -> descent -> re-check")
    theta = 0.5
    n_perturbations = 0
    cur = theta
    LR = 0.04
    for outer in range(20):
        # Check blindness
        ratio, _, g_f = blind_1d(cur)
        if ratio > 0.7:
            # 10 steps to confirm escaped
            consec = 0
            test_cur = cur
            for inner in range(10):
                b_t = np.asarray(rhs_coeffs(jnp.asarray(test_cur)))
                mask_t = topk_mask(np.abs(b_t), K_ACTIVE)
                g_t = float(grad_with_mask(jnp.asarray(test_cur),
                                            jnp.asarray(mask_t)))
                r_t, _, _ = blind_1d(test_cur)
                if r_t > 0.7:
                    consec += 1
                test_cur = test_cur + LR * g_t
            if consec == 10:
                print(f"  Outer {outer}: escaped, ratio = {ratio:.4f}, "
                      f"theta = {cur:.4f}, perturbations = {n_perturbations}")
                break
        # Perturb
        direction = np.sign(g_f) if g_f != 0 else 1
        cur = cur + delta_1d * direction
        n_perturbations += 1
        # Run up to 20 descent steps
        for step in range(20):
            b = np.asarray(rhs_coeffs(jnp.asarray(cur)))
            mask = topk_mask(np.abs(b), K_ACTIVE)
            g = float(grad_with_mask(jnp.asarray(cur), jnp.asarray(mask)))
            r_step, _, _ = blind_1d(cur)
            cur = cur + LR * g
            if r_step < 0.7:
                # re-perturb next time we check
                pass
    else:
        print(f"  Outer loop exhausted: perturbations = {n_perturbations}")
    print(f"  Total perturbations used: {n_perturbations}")


if __name__ == "__main__":
    d1, d2 = part_a()
    part_b(d1, d2)
    part_c(d1)
