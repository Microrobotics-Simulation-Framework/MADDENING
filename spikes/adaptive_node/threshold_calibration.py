"""Round-6 Investigation 1: blindness_threshold calibration and
is_trapped_at coverage.

Three parts:
  A. Optimizer damage from partial-blindness starts (2D problem).
  B. FP/FN rates at candidate thresholds 0.2, 0.3, 0.5, 0.7.
  C. is_trapped_at coverage in the partial-blindness zone.

Setup matches round-4 Inv 3C: 32x32, Gaussian source, sensor (0.7, 0.6).
We MAXIMIZE J via gradient ascent (theta moves toward sensor).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from locality_2d import (
    N, N_1D, S_2D, SINE_EIGS_2D, SENSOR_FLAT, J_sine_full,
    J_sine_frozen, source_grid_2d, x_grid_1d,
)


def grad2_full(tx, ty):
    return (float(jax.grad(J_sine_full, argnums=0)(jnp.asarray(tx),
                                                    jnp.asarray(ty))),
            float(jax.grad(J_sine_full, argnums=1)(jnp.asarray(tx),
                                                    jnp.asarray(ty))))


def grad2_frozen(tx, ty, k, sel="b"):
    return (float(jax.grad(J_sine_frozen, argnums=0)(
                jnp.asarray(tx), jnp.asarray(ty), k, sel)),
            float(jax.grad(J_sine_frozen, argnums=1)(
                jnp.asarray(tx), jnp.asarray(ty), k, sel)))


def blindness_2d(tx, ty, k):
    gx_f, gy_f = grad2_full(tx, ty)
    gx_fr, gy_fr = grad2_frozen(tx, ty, k, "b")
    nf = np.hypot(gx_f, gy_f)
    nr = np.hypot(gx_fr, gy_fr)
    return nr / (nf + 1e-30)


def is_trapped_at_proxy(tx, ty, k, eps=1e-3, rng=None):
    """Re-thresh estimator from round-5 cheap_diagnostic_v2.py.
    For 2D theta, perturb in random direction."""
    if rng is None:
        rng = np.random.default_rng(seed=0)
    gx0, gy0 = grad2_frozen(tx, ty, k, "b")
    n0 = np.hypot(gx0, gy0)
    # Random 2D Rademacher
    v = rng.choice([-1.0, 1.0], size=2)
    txp = tx + eps * v[0]
    typ = ty + eps * v[1]
    gxp, gyp = grad2_frozen(txp, typ, k, "b")
    dg = np.array([gxp - gx0, gyp - gy0]) / eps
    est = np.linalg.norm(dg)
    return n0 / (est + 1e-30)


# ---- Part A ----
def part_a():
    print("# Part A -- optimizer damage from partial-blindness starts")
    print()
    # Pick 6 starting points from round-4 7x7 grid
    starts = [
        ("good (high)", 0.150, 0.150),
        ("good (high)", 0.850, 0.150),
        ("partial low", 0.617, 0.500),     # ratio 0.557
        ("partial low", 0.500, 0.733),     # ratio 0.674
        ("partial high (danger)", 0.500, 0.617),  # ratio 0.130
        ("partial high (danger)", 0.500, 0.383),  # ratio 0.795 -- not quite 0.13-0.3
    ]
    k = N // 8
    LR = 0.04
    N_STEPS = 50
    print(f"  N={N}, k={k}, lr={LR}, {N_STEPS} steps, gradient ASCENT")
    print(f"  Sensor at (0.7, 0.6).  Maximizing J means theta drifts toward sensor.")
    print()
    for label, tx0, ty0 in starts:
        r0 = blindness_2d(tx0, ty0, k)
        tx, ty = tx0, ty0
        # ascent: theta += lr * g
        wasted_steps = 0
        traj = [(tx, ty)]
        J_ref0 = float(J_sine_full(jnp.asarray(tx0), jnp.asarray(ty0)))
        exit_step = None
        for step in range(N_STEPS):
            gx, gy = grad2_frozen(tx, ty, k, "b")
            tx = tx + LR * gx
            ty = ty + LR * gy
            traj.append((tx, ty))
            r_now = blindness_2d(tx, ty, k)
            if r_now > 0.7 and exit_step is None:
                exit_step = step
            if r_now < 0.7:
                wasted_steps += 1
        J_ref_end = float(J_sine_full(jnp.asarray(tx), jnp.asarray(ty)))
        # optimal J at the sensor location
        J_opt = float(J_sine_full(jnp.asarray(0.7), jnp.asarray(0.6)))
        progress = (J_ref_end - J_ref0) / (J_opt - J_ref0 + 1e-30)
        print(f"  {label:<25} start=({tx0:.3f}, {ty0:.3f}) "
              f"ratio0={r0:.3f}")
        print(f"     wasted (in partial zone): {wasted_steps}/{N_STEPS}, "
              f"exit step: {exit_step}")
        print(f"     traj: [(%.3f,%.3f) ... (%.3f,%.3f)]" %
              (traj[0][0], traj[0][1], traj[-1][0], traj[-1][1]))
        print(f"     J0={J_ref0:.4e}, Jfinal={J_ref_end:.4e}, "
              f"J_opt={J_opt:.4e}, progress={progress:.3f}")
        print()


# ---- Part B ----
def part_b():
    print("\n# Part B -- FP/FN at threshold values 0.2, 0.3, 0.5, 0.7")
    print()
    # 1D sweep ratios
    from trap_characterisation import (
        K_ACTIVE, rhs_coeffs, grad_with_mask, grad_full, topk_mask,
        LAMBDAS,
    )
    print("  1D sweep: 41 points across theta in [0.1, 0.9]")
    thetas_1d = np.linspace(0.1, 0.9, 41)
    r1d = []
    for t in thetas_1d:
        b = np.asarray(rhs_coeffs(jnp.asarray(t)))
        mask = topk_mask(np.abs(b), K_ACTIVE)
        g_fr = float(grad_with_mask(jnp.asarray(t), jnp.asarray(mask)))
        g_f = float(grad_full(jnp.asarray(t)))
        r1d.append(abs(g_fr) / (abs(g_f) + 1e-30))
    r1d = np.array(r1d)

    # 2D sweep ratios (from Inv 1A-like)
    print("  2D sweep: 49 points across (tx, ty) in [0.15, 0.85]^2")
    grid = np.linspace(0.15, 0.85, 7)
    r2d = []
    for tx in grid:
        for ty in grid:
            r2d.append(blindness_2d(tx, ty, N // 8))
    r2d = np.array(r2d)

    # Define "true blind/partial/good" by ratio < 0.3, 0.3-0.7, > 0.7
    print()
    for thresh in [0.2, 0.3, 0.5, 0.7]:
        # FP: triggered but actually good (ratio > 0.7)
        # FN: not triggered but actually blind/partial (ratio < threshold but flagged > thresh? wait)
        # Triggering: ratio < threshold
        # "Actually blind" = ratio < 0.3 (round-3/4 boundary)
        # "Actually partial" = 0.3 <= ratio < 0.7
        # "Actually good" = ratio >= 0.7
        for label, ratios in [("1D (41 pts)", r1d), ("2D (49 pts)", r2d)]:
            trig = ratios < thresh
            actually_good = ratios >= 0.7
            actually_blind = ratios < 0.3
            actually_partial = (ratios >= 0.3) & (ratios < 0.7)
            n_trig = int(np.sum(trig))
            FP = int(np.sum(trig & actually_good))
            FN = int(np.sum(~trig & (actually_blind | actually_partial)))
            TP = int(np.sum(trig & ~actually_good))
            TN = int(np.sum(~trig & actually_good))
            print(f"  thresh={thresh:.1f}  {label}: "
                  f"triggered={n_trig}, FP={FP}, FN={FN}, "
                  f"TP={TP}, TN={TN}")
    print()
    print("  Expected cost: E[cost] = FP_rate * 1 step + FN_rate * mean_wasted")
    # From Part A typical wasted = 5-15 steps for partial-blind starts
    # Use a mean of 10
    MEAN_WASTED = 10.0
    for thresh in [0.2, 0.3, 0.5, 0.7]:
        rates = []
        for ratios in [r1d, r2d]:
            n_total = len(ratios)
            trig = ratios < thresh
            actually_good = ratios >= 0.7
            actually_blind_or_partial = ratios < 0.7
            FP = int(np.sum(trig & actually_good))
            FN = int(np.sum(~trig & actually_blind_or_partial))
            cost = FP / n_total * 1.0 + FN / n_total * MEAN_WASTED
            rates.append(cost)
        print(f"  thresh={thresh:.1f}: 1D cost={rates[0]:.2f}, "
              f"2D cost={rates[1]:.2f}")


# ---- Part C ----
def part_c():
    print("\n# Part C -- is_trapped_at coverage of partial-blindness zone")
    print()
    grid = np.linspace(0.15, 0.85, 7)
    k = N // 8
    print(f"  {'tx':>6} {'ty':>6} {'true_ratio':>11} {'proxy':>9} "
          f"{'bucket':>10}")
    rng = np.random.default_rng(seed=42)
    pairs = []
    for tx in grid:
        for ty in grid:
            r_true = blindness_2d(tx, ty, k)
            proxy = is_trapped_at_proxy(tx, ty, k, eps=1e-3, rng=rng)
            pairs.append((tx, ty, r_true, proxy))
            if r_true < 0.7 or proxy < 0.3:
                if r_true < 0.3: bucket = "blind"
                elif r_true < 0.7: bucket = "partial"
                else: bucket = "good"
                print(f"  {tx:>6.3f} {ty:>6.3f} {r_true:>11.4f} "
                      f"{proxy:>9.4f} {bucket:>10}")
    # correlation
    true_ratios = np.array([p[2] for p in pairs])
    proxies = np.array([p[3] for p in pairs])
    # Spearman-like rank correlation
    from scipy.stats import spearmanr
    rho, _ = spearmanr(true_ratios, proxies)
    print()
    print(f"  Spearman rank correlation (true_ratio vs proxy): {rho:.4f}")
    # Binary agreement at threshold 0.3
    bin_true = true_ratios < 0.3
    bin_proxy = proxies < 0.3
    agree = int(np.sum(bin_true == bin_proxy))
    print(f"  Binary agreement at thresh=0.3: {agree}/{len(pairs)} "
          f"({100*agree/len(pairs):.1f}%)")
    bin_true_partial = true_ratios < 0.7
    bin_proxy_partial = proxies < 0.7
    agree_p = int(np.sum(bin_true_partial == bin_proxy_partial))
    print(f"  Binary agreement at thresh=0.7: {agree_p}/{len(pairs)} "
          f"({100*agree_p/len(pairs):.1f}%)")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
