"""Round-4 Investigation 1: wrong-sign boundary result -- mechanism
and rolling robustness.

Round-3 reported J_err = 1.247 at theta ~ 0.04 under top-|b| in
the smooth trajectory.  This is more dangerous than the symmetry
trap: the *forward solution* has the wrong sign at the sensor, not
just the gradient.

Parts:
  A. Mechanism -- decompose u_sensor by mode contribution, find the
     cancellation
  B. Rolling top-|c_prev| -- theoretical and empirical robustness
  C. Haar basis -- does the failure mode exist in a local basis?
  D. Direction accuracy in blindness_ratio > 1.0 regime
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
    ks, LAMBDAS, x_grid, dx, PHI,
    rhs_coeffs, J_full, J_with_mask, grad_with_mask, grad_full,
    topk_mask,
)


def J_under(theta, mask):
    return float(J_with_mask(jnp.asarray(theta), jnp.asarray(mask)))


def J_full_val(theta):
    return float(J_full(jnp.asarray(theta)))


# ----- Part A: mechanism -----
def part_a():
    print("# Part A -- mechanism of wrong-sign at boundary")
    print()

    # Reproduce round-3 boundary failure: theta ~ 0.04, k=16, top-|b|
    theta_bad = 0.040  # near round-3's 0.0402
    b = np.asarray(rhs_coeffs(jnp.asarray(theta_bad)))
    mag = np.abs(b)
    mask = topk_mask(mag, K_ACTIVE)
    J_frozen = J_under(theta_bad, mask)
    J_ref = J_full_val(theta_bad)
    err = abs(J_frozen - J_ref) / (abs(J_ref) + 1e-30)
    print(f"  theta = {theta_bad:.4f}, K = {K_ACTIVE}, sensor at "
          f"x = {float(x_grid[SENSOR_IDX]):.4f}")
    print(f"  J_full    = {J_ref:+.6e}")
    print(f"  J_frozen  = {J_frozen:+.6e}")
    print(f"  rel_err   = {err:.3e}  "
          f"{'(WRONG SIGN)' if np.sign(J_frozen)!=np.sign(J_ref) else ''}")
    print()

    # Decompose into per-mode contributions
    phi_at_sensor = np.asarray(PHI[SENSOR_IDX, :])  # shape (N_BASIS,)
    lam = np.asarray(LAMBDAS)
    c_full = b / lam
    contrib_full = c_full * phi_at_sensor   # mode -> contribution to u_sensor

    # Mask selected modes
    sel_idx = np.where(mask)[0]
    sel_ks = sel_idx + 1  # 1-indexed mode number
    sel_contribs = contrib_full[sel_idx]

    # Excluded modes' contributions
    excl_idx = np.where(~mask)[0]
    excl_ks = excl_idx + 1
    excl_contribs = contrib_full[excl_idx]

    print("  ## Mode-by-mode decomposition (modes in mask)")
    print(f"  {'rank':>4} {'k':>4} {'|b_k|':>11} {'c_k':>13} "
          f"{'phi(x_s)':>10} {'contrib':>13}")
    order_in = np.argsort(-mag[sel_idx])
    for r, j in enumerate(order_in[:K_ACTIVE]):
        i = sel_idx[j]
        print(f"  {r+1:>4d} {i+1:>4d} {mag[i]:>11.3e} "
              f"{c_full[i]:>+13.3e} {phi_at_sensor[i]:>+10.3f} "
              f"{contrib_full[i]:>+13.3e}")
    sum_in = sel_contribs.sum()
    print(f"  SUM over in-mask:  {sum_in:+.4e}  (= J_frozen if direct sum)")

    # Largest excluded contributions
    print()
    print("  ## Top-20 EXCLUDED modes by |contribution|")
    excl_abs = np.abs(excl_contribs)
    order_excl = np.argsort(-excl_abs)
    print(f"  {'rank':>4} {'k':>4} {'|b_k|':>11} {'c_k':>13} "
          f"{'phi(x_s)':>10} {'contrib':>13}")
    for r in range(min(20, len(excl_idx))):
        j = order_excl[r]
        i = excl_idx[j]
        print(f"  {r+1:>4d} {i+1:>4d} {mag[i]:>11.3e} "
              f"{c_full[i]:>+13.3e} {phi_at_sensor[i]:>+10.3f} "
              f"{contrib_full[i]:>+13.3e}")
    sum_excl = excl_contribs.sum()
    print(f"  SUM over excluded: {sum_excl:+.4e}")
    print(f"  Total (in + excl): {sum_in + sum_excl:+.4e}  "
          f"~ J_full = {J_ref:+.4e}")

    # Why does the in-mask sum have wrong sign?
    pos_in = sel_contribs[sel_contribs > 0].sum()
    neg_in = sel_contribs[sel_contribs < 0].sum()
    print()
    print(f"  In-mask positive contributions sum: {pos_in:+.4e}")
    print(f"  In-mask negative contributions sum: {neg_in:+.4e}")
    print(f"  In-mask net:                        {pos_in+neg_in:+.4e}")
    pos_excl = excl_contribs[excl_contribs > 0].sum()
    neg_excl = excl_contribs[excl_contribs < 0].sum()
    print(f"  Excluded positive contributions sum: {pos_excl:+.4e}")
    print(f"  Excluded negative contributions sum: {neg_excl:+.4e}")
    print(f"  Excluded net:                        {pos_excl+neg_excl:+.4e}")

    return theta_bad


# ----- Part B: rolling top-|c_prev| robustness -----
def part_b(theta_bad):
    print("\n\n# Part B -- does rolling top-|c_prev| avoid the wrong-sign?")
    print()

    # Theory:
    # top-|b| ranks modes by |b_k|.  At boundary theta, sin(k pi theta)
    # gives small |b| for low-k (near-zero of sine).  Higher-k modes
    # have larger sin amplitudes but tiny 1/lambda_k.  Top-|b| therefore
    # selects modes whose CONTRIBUTION c_k = b_k/lambda_k is NOT
    # dominant -- selection is on b, not on c.
    #
    # top-|c_prev| ranks by |c_k| = |b_k|/lambda_k, which directly
    # weights by 1/k^2.  Low-k modes (with large 1/lambda) dominate
    # |c| ranking even when their |b| is small.  This selects the
    # modes that ACTUALLY contribute to u, regardless of whether
    # they contribute to b.
    print("  THEORY: top-|b| selects by RHS magnitude.  top-|c| selects")
    print("  by SOLUTION magnitude = |b|/lambda.  At a boundary-")
    print("  adjacent theta, sin(k pi theta) suppresses low-k |b|, so")
    print("  top-|b| picks high-k modes whose |c| = |b|/k^2 is small.")
    print("  Top-|c| corrects this by re-weighting by 1/k^2.")
    print()

    # Empirical: trajectory sweep
    T = 30
    def trajectory(t):
        return 0.3 + 0.3 * np.sin(2.0 * np.pi * t / T)

    print(f"  {'t':>3} {'theta':>7}  {'top|b| J_err':>13}  "
          f"{'rolling J_err':>14}  {'oracle J_err':>13}  flag")
    c_prev = None
    err_wrong_sign_b = 0
    err_wrong_sign_cp = 0
    err_wrong_sign_oracle = 0
    for t in range(T):
        theta_t = trajectory(t)
        b = np.asarray(rhs_coeffs(jnp.asarray(theta_t)))
        J_ref = J_full_val(theta_t)
        mag = np.abs(b)
        # top-|b|
        mask_b = topk_mask(mag, K_ACTIVE)
        Jb = J_under(theta_t, mask_b)
        eb = abs(Jb - J_ref) / (abs(J_ref) + 1e-30)
        # rolling top-|c_prev|
        if c_prev is None:
            mask_cp = mask_b
        else:
            mask_cp = topk_mask(np.abs(c_prev), K_ACTIVE)
        Jcp = J_under(theta_t, mask_cp)
        ecp = abs(Jcp - J_ref) / (abs(J_ref) + 1e-30)
        # oracle
        c_full = b / np.asarray(LAMBDAS)
        mask_or = topk_mask(np.abs(c_full), K_ACTIVE)
        Jor = J_under(theta_t, mask_or)
        eor = abs(Jor - J_ref) / (abs(J_ref) + 1e-30)
        flag = ""
        if np.sign(Jb) != np.sign(J_ref): flag += "[b-WRONG-SIGN] "; err_wrong_sign_b += 1
        if np.sign(Jcp) != np.sign(J_ref): flag += "[cp-WRONG-SIGN] "; err_wrong_sign_cp += 1
        if np.sign(Jor) != np.sign(J_ref): flag += "[or-WRONG-SIGN] "; err_wrong_sign_oracle += 1
        if eb > 1.0 and not flag: flag += "[b-LARGE] "
        print(f"  {t:>3d} {theta_t:>7.4f}  {eb:>13.3e}  "
              f"{ecp:>14.3e}  {eor:>13.3e}  {flag}")
        c_prev = c_full
    print()
    print(f"  top-|b|       wrong-sign steps: {err_wrong_sign_b} / {T}")
    print(f"  rolling cprev wrong-sign steps: {err_wrong_sign_cp} / {T}")
    print(f"  oracle        wrong-sign steps: {err_wrong_sign_oracle} / {T}")


# ----- Part C: Haar basis behavior -----
def part_c():
    print("\n\n# Part C -- does the wrong-sign failure occur in Haar?")
    print()
    from locality_theorem import (
        N as N2, W_HAAR, A_HAAR, EYE_N, source_grid, SENSOR_IDX as SENS_2,
        J_haar_full, x_grid as x_grid2,
    )

    def J_haar_under_mask(theta, mask):
        b = W_HAAR @ source_grid(jnp.asarray(theta))
        m = jax.lax.stop_gradient(jnp.asarray(mask))
        A_eff = jnp.where(m[:, None], m[None, :] * A_HAAR, EYE_N)
        b_eff = m * b
        c = jnp.linalg.solve(A_eff, b_eff)
        u = W_HAAR.T @ c
        return float(u[SENS_2])

    K2 = N2 // 16   # match round-2 setup proportionally
    T = 30

    def trajectory(t):
        return 0.3 + 0.3 * np.sin(2.0 * np.pi * t / T)

    print(f"  Haar 1D Poisson, N={N2}, K={K2}, sensor x={float(x_grid2[SENS_2]):.4f}")
    print(f"  {'t':>3} {'theta':>7}  {'top|b|':>11}  {'top|c_prev|':>11}  "
          f"{'oracle':>9}  flag")
    c_prev = None
    n_wrong_b = 0
    n_wrong_cp = 0
    for t in range(T):
        theta_t = trajectory(t)
        b_haar = np.asarray(W_HAAR @ source_grid(jnp.asarray(theta_t)))
        J_ref = float(J_haar_full(jnp.asarray(theta_t)))
        # top-|b|
        mag_b = np.abs(b_haar)
        mask_b = topk_mask(mag_b, K2)
        Jb = J_haar_under_mask(theta_t, mask_b)
        eb = abs(Jb - J_ref) / (abs(J_ref) + 1e-30)
        # top-|c_prev|
        if c_prev is None:
            mask_cp = mask_b
        else:
            mask_cp = topk_mask(np.abs(c_prev), K2)
        Jcp = J_haar_under_mask(theta_t, mask_cp)
        ecp = abs(Jcp - J_ref) / (abs(J_ref) + 1e-30)
        # oracle
        c_full = np.linalg.solve(np.asarray(A_HAAR), b_haar)
        mask_or = topk_mask(np.abs(c_full), K2)
        Jor = J_haar_under_mask(theta_t, mask_or)
        eor = abs(Jor - J_ref) / (abs(J_ref) + 1e-30)
        flag = ""
        if np.sign(Jb) != np.sign(J_ref): flag += "[b-WS] "; n_wrong_b += 1
        if np.sign(Jcp) != np.sign(J_ref): flag += "[cp-WS] "; n_wrong_cp += 1
        if eb > 1.0: flag += "[b>1] "
        print(f"  {t:>3d} {theta_t:>7.4f}  {eb:>11.3e}  {ecp:>11.3e}  "
              f"{eor:>9.3e}  {flag}")
        c_prev = c_full
    print()
    print(f"  Haar: top-|b|       wrong-sign steps: {n_wrong_b} / {T}")
    print(f"  Haar: rolling cprev wrong-sign steps: {n_wrong_cp} / {T}")
    if n_wrong_b == 0:
        print("  -> Haar locality prevents wrong-sign in this trajectory.")


# ----- Part D: direction accuracy when blindness_ratio > 1 -----
def part_d():
    print("\n\n# Part D -- direction accuracy in blindness_ratio > 1.0 regime")
    print()
    print("  In 1D, theta is scalar -- 'direction' reduces to SIGN.")
    print("  We check sign agreement between g_frozen and g_full at all")
    print("  sweep points, with attention to ratio > 1.0 cases.")
    print()
    thetas = np.linspace(0.1, 0.9, 41)
    print(f"  {'theta':>7}  {'ratio':>8}  {'sign(g_fr)':>10}  "
          f"{'sign(g_full)':>13}  {'agree':>7}  flag")
    n_over_1 = 0
    n_over_1_sign_disagree = 0
    n_sign_disagree = 0
    for t in thetas:
        b = np.asarray(rhs_coeffs(jnp.asarray(t)))
        mask = topk_mask(np.abs(b), K_ACTIVE)
        g_fr = float(grad_with_mask(jnp.asarray(t), jnp.asarray(mask)))
        g_full = float(grad_full(jnp.asarray(t)))
        ratio = abs(g_fr) / (abs(g_full) + 1e-30)
        s_fr = int(np.sign(g_fr))
        s_full = int(np.sign(g_full))
        agree = s_fr == s_full
        flag = ""
        if ratio > 1.0:
            n_over_1 += 1
            flag += "[r>1] "
        if not agree:
            n_sign_disagree += 1
            flag += "[SIGN-FLIP] "
            if ratio > 1.0:
                n_over_1_sign_disagree += 1
        print(f"  {t:>7.4f}  {ratio:>8.4f}  {s_fr:>+10d}  "
              f"{s_full:>+13d}  {str(agree):>7}  {flag}")
    print()
    print(f"  Total points with ratio > 1.0:                     {n_over_1}")
    print(f"  Of those, sign disagreement (direction-distorted): "
          f"{n_over_1_sign_disagree}")
    print(f"  Total sign disagreements:                          "
          f"{n_sign_disagree}")


if __name__ == "__main__":
    tb = part_a()
    part_b(tb)
    part_c()
    part_d()
