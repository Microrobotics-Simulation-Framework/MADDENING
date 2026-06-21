"""Round-4 Investigation 2: cheap approximation to the blindness
diagnostic.

Round-3 established that |grad J_frozen| / |grad J_full| catches
the symmetry trap.  Cost is 2x per evaluation (the full-basis
solve is the expensive step adaptivity exists to avoid).

Parts:
  A. Theoretical lower bound from coarse-then-fine cold start.
     Hypothesis: cold-start blindness >= some threshold by
     construction.
  B. Randomised estimator following the user's specification:
       est = (1/r) sum_i (v_i . (g_frozen(theta + eps v_i) -
                                  g_frozen(theta)))^2 / eps^2
     For 1D theta, v_i in {-1, +1}.
  C. Recommendation -- design-reasoning section in the memo.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from trap_characterisation import (
    N_BASIS, N_GRID, SIGMA, SENSOR_IDX, K_ACTIVE,
    LAMBDAS, x_grid, dx, PHI,
    rhs_coeffs, J_full, J_with_mask, grad_with_mask, grad_full,
    topk_mask,
)


def true_blindness(theta, k):
    b = np.asarray(rhs_coeffs(jnp.asarray(theta)))
    mask = topk_mask(np.abs(b), k)
    g_fr = float(grad_with_mask(jnp.asarray(theta), jnp.asarray(mask)))
    g_full = float(grad_full(jnp.asarray(theta)))
    return abs(g_fr) / (abs(g_full) + 1e-30), g_fr, g_full


def coldstart_mask(theta, k_active, k_coarse):
    """Coarse-then-fine: top-|c_coarse| from a preliminary k_coarse solve."""
    b = np.asarray(rhs_coeffs(jnp.asarray(theta)))
    mask_coarse = topk_mask(np.abs(b), k_coarse)
    c_coarse = np.where(mask_coarse, b / np.asarray(LAMBDAS), 0.0)
    return topk_mask(np.abs(c_coarse), k_active)


def coldstart_blindness(theta, k_active, k_coarse):
    mask = coldstart_mask(theta, k_active, k_coarse)
    g_fr = float(grad_with_mask(jnp.asarray(theta), jnp.asarray(mask)))
    g_full = float(grad_full(jnp.asarray(theta)))
    return abs(g_fr) / (abs(g_full) + 1e-30), mask


# ----- Part A: theoretical lower bound from cold-start -----
def part_a():
    print("# Part A -- cold-start blindness lower bound")
    print()
    print(f"  Hypothesis: coarse-then-fine cold-start (k_coarse > k_active)")
    print(f"  yields blindness ratio >= some threshold by construction.")
    print()
    test_thetas = [0.30, 0.42, 0.50, 0.60, 0.70]
    for k_active in [8, 16, 32]:
        for k_coarse in [N_BASIS // 4, N_BASIS // 8]:
            print(f"\n  k_active = {k_active:>3d}, k_coarse = {k_coarse:>3d}")
            print(f"  {'theta':>7}  {'ratio_topB':>11}  {'ratio_cold':>11}  "
                  f"{'improves':>10}")
            for t in test_thetas:
                r_b, _, _ = true_blindness(t, k_active)
                r_c, _ = coldstart_blindness(t, k_active, k_coarse)
                imp = ""
                if r_c < 0.3:
                    imp = "<<-- still blind"
                elif r_c > r_b * 1.2:
                    imp = "<-- helps"
                elif r_c < r_b * 0.8:
                    imp = "<-- HURTS"
                print(f"  {t:>7.4f}  {r_b:>11.4f}  {r_c:>11.4f}  {imp:>10}")

    # Specific test: at the trap
    print()
    print("  --- Symmetry trap (theta = 0.5) ---")
    for k_active in [8, 16, 32]:
        for k_coarse in [N_BASIS // 4, N_BASIS // 8]:
            r_c, _ = coldstart_blindness(0.5, k_active, k_coarse)
            print(f"    k_active={k_active:>3d}, k_coarse={k_coarse:>3d}: "
                  f"cold-start ratio = {r_c:.6f}  "
                  f"{'(blind)' if r_c < 0.3 else ''}")


# ----- Part B: randomised estimator -----
def randomized_estimator(theta, k, eps, r, rng):
    """Literal implementation of the user's formula:
      est = (1/r) sum (v_i . (g_frozen(theta + eps v_i) - g_frozen(theta)))^2
            / eps^2
    For 1D theta, v_i in {-1, +1} (Rademacher).  Mask is recomputed
    at each perturbed theta (the rationale being that variation in
    g_frozen across perturbations captures cross-mode info that the
    fixed-mask version misses).
    """
    b = np.asarray(rhs_coeffs(jnp.asarray(theta)))
    mask = topk_mask(np.abs(b), k)
    g0 = float(grad_with_mask(jnp.asarray(theta), jnp.asarray(mask)))

    sumsq = 0.0
    for _ in range(r):
        v = float(rng.choice([-1.0, 1.0]))
        theta_p = theta + eps * v
        b_p = np.asarray(rhs_coeffs(jnp.asarray(theta_p)))
        mask_p = topk_mask(np.abs(b_p), k)
        g_p = float(grad_with_mask(jnp.asarray(theta_p),
                                   jnp.asarray(mask_p)))
        sumsq += ((v * (g_p - g0)) / eps) ** 2
    return (sumsq / r) ** 0.5  # sqrt to get magnitude-like estimate


def estimator_blindness_proxy(theta, k, eps, r, rng):
    """Use estimator as a proxy for the |g_full| magnitude.

    Then blindness_proxy = |g_frozen| / estimator.
    If proxy < 0.3, predict 'blind'.
    """
    b = np.asarray(rhs_coeffs(jnp.asarray(theta)))
    mask = topk_mask(np.abs(b), k)
    g_fr = float(grad_with_mask(jnp.asarray(theta), jnp.asarray(mask)))
    est = randomized_estimator(theta, k, eps, r, rng)
    return abs(g_fr) / (est + 1e-30)


def classify_ratio(r):
    if r < 0.3:
        return "blind"
    if r < 0.7:
        return "partial"
    return "good"


def part_b():
    print("\n\n# Part B -- randomised estimator")
    print()
    print("  est = sqrt((1/r) sum (v . (g_fr(th+eps v) - g_fr(th)))^2 / eps^2)")
    print("  Proxy ratio = |g_frozen| / est.  Threshold proxy < 0.3 = 'blind'.")
    print()
    thetas = np.linspace(0.1, 0.9, 41)
    rng = np.random.default_rng(seed=42)

    # Compute the true blindness classification first
    truth = {}
    for t in thetas:
        r, _, _ = true_blindness(t, K_ACTIVE)
        truth[float(t)] = (r, classify_ratio(r))

    for r_samples in [1, 3, 5]:
        for eps in [1e-3, 1e-2]:
            n_correct = 0
            n_blind_correct = 0
            n_blind_truth = 0
            misclass_cases = []
            rng2 = np.random.default_rng(seed=42 + r_samples * 10
                                         + int(1000 * eps))
            for t in thetas:
                true_r, true_cls = truth[float(t)]
                proxy = estimator_blindness_proxy(t, K_ACTIVE, eps,
                                                  r_samples, rng2)
                proxy_cls = classify_ratio(proxy)
                if proxy_cls == true_cls:
                    n_correct += 1
                else:
                    misclass_cases.append((t, true_r, true_cls, proxy,
                                           proxy_cls))
                if true_cls == "blind":
                    n_blind_truth += 1
                    if proxy_cls == "blind":
                        n_blind_correct += 1
            print(f"  r = {r_samples}, eps = {eps:.0e}:  "
                  f"correctly classified {n_correct}/41  "
                  f"({100*n_correct/41:.1f}%); "
                  f"blind detection: {n_blind_correct}/{n_blind_truth}")
            # Show a couple misclassification examples
            for c in misclass_cases[:3]:
                t, tr, tc, p, pc = c
                print(f"    misclass at theta={t:.4f}: "
                      f"true_r={tr:.4f} ({tc}) vs proxy={p:.4f} ({pc})")

    # Specifically: detection of theta=0.5 trap with r=1
    print()
    print("  --- Trap detection (theta = 0.5) ---")
    for r_samples in [1, 3, 5]:
        for eps in [1e-3, 1e-2]:
            rng2 = np.random.default_rng(seed=999)
            proxy = estimator_blindness_proxy(0.5, K_ACTIVE, eps, r_samples,
                                              rng2)
            print(f"    r={r_samples}, eps={eps:.0e}: "
                  f"proxy = {proxy:.4f}  "
                  f"{'detected as BLIND' if proxy < 0.3 else 'MISSED'}")


if __name__ == "__main__":
    part_a()
    part_b()
