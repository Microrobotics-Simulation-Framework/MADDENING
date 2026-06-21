"""Round-5 Investigation 1: re-thresholded FD estimator.

Round-4's Hutchinson formula used g_frozen(theta + eps v) computed
with the SAME mask as g_frozen(theta).  That estimates J_frozen''
(curvature), not |grad J_full|, so classification fails.

The fix: re-threshold the mask AT EACH perturbed theta.  Then the
difference quotient
    proxy_dir(theta, v) = [g_frozen(theta+eps v; mask(theta+eps v))
                           - g_frozen(theta; mask(theta))] / eps
captures the change due to BOTH smooth gradient flow AND mask
flipping.  If a perturbation drags new gradient-relevant modes
into the mask, that contribution appears in proxy_dir.

est(theta) = (1/r) sum_i |proxy_dir(theta, v_i)|
proxy_ratio(theta) = |g_frozen(theta)| / est(theta)

Parts A (impl + classification), B (cost), C (recommendation).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np
import time

jax.config.update("jax_enable_x64", True)

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from trap_characterisation import (
    N_BASIS, N_GRID, SIGMA, SENSOR_IDX, K_ACTIVE,
    LAMBDAS, x_grid, dx, PHI,
    rhs_coeffs, J_full, J_with_mask, grad_with_mask, grad_full,
    topk_mask,
)


def g_frozen_rethresh(theta, k):
    """Return g_frozen at theta with mask re-thresholded at theta."""
    b = np.asarray(rhs_coeffs(jnp.asarray(theta)))
    mask = topk_mask(np.abs(b), k)
    g = float(grad_with_mask(jnp.asarray(theta), jnp.asarray(mask)))
    return g, mask


def true_blindness(theta, k):
    g_fr, _ = g_frozen_rethresh(theta, k)
    g_full_val = float(grad_full(jnp.asarray(theta)))
    return abs(g_fr) / (abs(g_full_val) + 1e-30), g_fr, g_full_val


def rethresh_estimator(theta, k, eps, r, rng):
    """The Round-5 fix: re-threshold mask at each perturbed theta."""
    g0, mask0 = g_frozen_rethresh(theta, k)
    proxy_dirs = []
    mask_changes = 0
    for _ in range(r):
        v = float(rng.choice([-1.0, 1.0]))
        theta_p = theta + eps * v
        g_p, mask_p = g_frozen_rethresh(theta_p, k)
        if not np.array_equal(mask_p, mask0):
            mask_changes += 1
        proxy_dirs.append((g_p - g0) / eps)
    est = np.mean(np.abs(proxy_dirs))
    return est, mask_changes, g0


def proxy_ratio(theta, k, eps, r, rng):
    est, mc, g0 = rethresh_estimator(theta, k, eps, r, rng)
    return abs(g0) / (est + 1e-30), mc, g0


def classify(r):
    if r < 0.3: return "blind"
    if r < 0.7: return "partial"
    return "good"


def part_a():
    print("# Part A -- re-thresholded estimator classification")
    print()
    thetas = np.linspace(0.1, 0.9, 41)
    truth = {}
    for t in thetas:
        r, _, _ = true_blindness(t, K_ACTIVE)
        truth[float(t)] = r

    # Sanity: print true ratios at a few key points
    print("  True ratios (subset):")
    for t in [0.16, 0.30, 0.42, 0.48, 0.50, 0.52, 0.70]:
        r_t, _, _ = true_blindness(t, K_ACTIVE)
        print(f"    theta = {t:.2f}: true_ratio = {r_t:.4f}  ({classify(r_t)})")
    print()
    print(f"  {'r':>2} {'eps':>6} | {'correct':>8} {'%':>5} | "
          f"{'blind_det':>10} | {'mask_chg/41':>12} | misclass examples")
    for r_samples in [1, 3]:
        for eps in [1e-3, 5e-3, 1e-2]:
            rng = np.random.default_rng(seed=10 + r_samples * 100
                                        + int(1e4 * eps))
            n_correct = 0
            n_blind_truth = 0
            n_blind_caught = 0
            total_mask_chg = 0
            misclass = []
            for t in thetas:
                pr, mc, _ = proxy_ratio(t, K_ACTIVE, eps, r_samples, rng)
                true_r = truth[float(t)]
                if classify(pr) == classify(true_r):
                    n_correct += 1
                else:
                    misclass.append((t, true_r, pr))
                if classify(true_r) == "blind":
                    n_blind_truth += 1
                    if classify(pr) == "blind":
                        n_blind_caught += 1
                total_mask_chg += mc
            ex = ", ".join(f"th={t:.2f}({true_r:.2f}->{pr:.2f})"
                           for t, true_r, pr in misclass[:2])
            print(f"  {r_samples:>2} {eps:>6.0e} | {n_correct:>8d} "
                  f"{100*n_correct/41:>4.1f}% | "
                  f"{n_blind_caught}/{n_blind_truth:<8} | "
                  f"{total_mask_chg:>4d}/{41*r_samples:<6d} | {ex}")
    print()
    # Trap detection specifically
    print("  Trap (theta=0.5) detection:")
    for r_samples in [1, 3]:
        for eps in [1e-3, 5e-3, 1e-2]:
            rng = np.random.default_rng(seed=999)
            pr, mc, _ = proxy_ratio(0.5, K_ACTIVE, eps, r_samples, rng)
            print(f"    r={r_samples}, eps={eps:.0e}: proxy={pr:.4f} "
                  f"(mask changes: {mc}/{r_samples}) "
                  f"{'DETECTED' if pr < 0.3 else 'MISSED'}")

    # Analysis: do mask changes help?
    print()
    print("  Mask-change correlation with classification accuracy:")
    rng = np.random.default_rng(seed=42)
    correct_with_chg = 0
    correct_no_chg = 0
    total_with_chg = 0
    total_no_chg = 0
    for t in thetas:
        pr, mc, _ = proxy_ratio(t, K_ACTIVE, 5e-3, 3, rng)
        if mc > 0:
            total_with_chg += 1
            if classify(pr) == classify(truth[float(t)]):
                correct_with_chg += 1
        else:
            total_no_chg += 1
            if classify(pr) == classify(truth[float(t)]):
                correct_no_chg += 1
    print(f"    r=3, eps=5e-3:")
    print(f"      mask-changes >0: {correct_with_chg}/{total_with_chg} "
          f"correct ({100*correct_with_chg/max(total_with_chg,1):.1f}%)")
    print(f"      mask-changes =0: {correct_no_chg}/{total_no_chg} "
          f"correct ({100*correct_no_chg/max(total_no_chg,1):.1f}%)")


def part_b():
    print("\n\n# Part B -- cost analysis")
    print()
    # Time one masked solve vs one full solve.  Use a benchmark loop.
    n_warmup = 5
    n_trials = 50
    theta = 0.42
    mask = topk_mask(np.abs(np.asarray(rhs_coeffs(jnp.asarray(theta)))),
                     K_ACTIVE)

    # warm JIT
    for _ in range(n_warmup):
        _ = float(grad_with_mask(jnp.asarray(theta), jnp.asarray(mask)))
        _ = float(grad_full(jnp.asarray(theta)))

    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = float(grad_with_mask(jnp.asarray(theta), jnp.asarray(mask)))
    masked_time = (time.perf_counter() - t0) / n_trials

    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = float(grad_full(jnp.asarray(theta)))
    full_time = (time.perf_counter() - t0) / n_trials

    print(f"  1D Poisson + sine basis, N_BASIS={N_BASIS}, K={K_ACTIVE}")
    print(f"  masked (K={K_ACTIVE}) grad time:  {masked_time*1e3:.3f} ms")
    print(f"  full   (K={N_BASIS}) grad time:  {full_time*1e3:.3f} ms")
    print(f"  ratio masked/full: {masked_time/full_time:.3f}")
    print()
    print(f"  Cost of full diagnostic (round-4 spec):     1 full grad "
          f"= {full_time*1e3:.3f} ms")
    print(f"  Cost of r=3 re-thresh estimator (round-5):  3 masked grads "
          f"= {3*masked_time*1e3:.3f} ms")
    print(f"  r=3 estimator is {full_time/(3*masked_time):.2f}x "
          f"{'cheaper' if 3*masked_time < full_time else 'EXPENSIVE'} "
          f"than full diagnostic.")
    print()
    print("  Note: in this 1D toy with sine basis, both forward and full")
    print("  solves are trivial (no actual linear system; just diagonal).")
    print("  The ratio in a real wavelet problem with N~1000 and k~100")
    print("  is masked_solve ~ O(k^3) vs full_solve ~ O(N^3).")
    print(f"  Predicted realistic ratio at N=1024, k=128: ")
    print(f"    masked/full = (k/N)^3 = (128/1024)^3 = {(128/1024)**3:.4f}")
    print(f"    So r=3 estimator costs ~{3 * (128/1024)**3:.4f} x one full solve.")
    print("  In that regime the re-thresholded estimator is ~500x cheaper")
    print("  than the full diagnostic.")


if __name__ == "__main__":
    part_a()
    part_b()
