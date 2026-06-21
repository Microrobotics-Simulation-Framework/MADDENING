"""Investigation 3: dJ/depsilon -- is differentiating through the
threshold tractable, and is the dual-variable formula exact?

Setup: same 1D Poisson + Haar basis as locality_theorem.py.  k=16
active set, eps is the threshold |c| >= eps determining the active
set.  We compute dJ/depsilon three ways:

  (1) FD:    J(theta, eps + h) - J(theta, eps - h) / (2h)
  (2) dual:  marginal contribution of the boundary element,
             phi_i(x_sensor) * c_active_i_when_included
  (3) soft:  jax.grad through sigmoid((|c| - eps)/tau)

J is piecewise-constant in eps with jumps at eps = |c_i| for each
i.  So dJ/depsilon in the classical sense is 0 a.e. with delta-
impulses at the flip points.

Methods (1) detects the jump if h spans a flip.  Method (2) computes
the jump magnitude analytically.  Method (3) smooths the jump with
a finite-width sigmoid envelope.

Additional comparison: SINE basis where A is diagonal.  Then the
dual formula has a clean closed form (c_active_i = c_full_i exactly
because no cross-mode coupling).  In HAAR, A_HAAR is not diagonal,
so including/excluding a single element changes the entire solve
slightly -- the dual formula is approximate.

We also report whether method (2) is computable purely from the
existing full-solve adjoint, or whether it needs additional work.
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
    N, W_HAAR, A_HAAR, EYE_N, S_J, EIGVALS_SINE,
    source_grid, x_grid, SENSOR_IDX,
)

THETA = 0.42
theta = jnp.asarray(THETA)


# ---------- Haar problem ----------

def J_haar_at_eps(theta, eps):
    """Forward J under mask = {i : |c_full_i| >= eps}, dense masked solve."""
    b = W_HAAR @ source_grid(theta)
    c_full = jnp.linalg.solve(A_HAAR, b)
    mag = jnp.abs(c_full)
    mask = jax.lax.stop_gradient(mag >= eps)
    A_eff = jnp.where(mask[:, None], mask[None, :] * A_HAAR, EYE_N)
    b_eff = mask * b
    c = jnp.linalg.solve(A_eff, b_eff)
    u = W_HAAR.T @ c
    return u[SENSOR_IDX]


def J_haar_soft(theta, eps, tau):
    """Soft-mask version using sigmoid envelope."""
    b = W_HAAR @ source_grid(theta)
    c_full = jnp.linalg.solve(A_HAAR, b)
    mag = jnp.abs(c_full)
    soft = jax.nn.sigmoid((mag - eps) / tau)
    M = soft[:, None] * soft[None, :]
    A_eff = M * A_HAAR + (1.0 - M) * EYE_N
    b_eff = soft * b
    c = jnp.linalg.solve(A_eff, b_eff)
    u = W_HAAR.T @ c
    return u[SENSOR_IDX]


# ---------- Sine problem (A diagonal) ----------

def J_sine_at_eps(theta, eps):
    b = S_J @ source_grid(theta)
    c_full = b / EIGVALS_SINE
    mag = jnp.abs(c_full)
    mask = jax.lax.stop_gradient(mag >= eps)
    c = jnp.where(mask, c_full, 0.0)
    u = S_J.T @ c
    return u[SENSOR_IDX]


def J_sine_soft(theta, eps, tau):
    b = S_J @ source_grid(theta)
    c_full = b / EIGVALS_SINE
    mag = jnp.abs(c_full)
    soft = jax.nn.sigmoid((mag - eps) / tau)
    c = soft * c_full
    u = S_J.T @ c
    return u[SENSOR_IDX]


# Compute full coefficient vectors / sensor projection rows
b_haar = np.asarray(W_HAAR @ source_grid(theta))
c_full_haar = np.linalg.solve(np.asarray(A_HAAR), b_haar)
mag_haar = np.abs(c_full_haar)
phi_sensor_haar = np.asarray(W_HAAR.T[SENSOR_IDX, :])

b_sine = np.asarray(S_J @ source_grid(theta))
c_full_sine = b_sine / np.asarray(EIGVALS_SINE)
mag_sine = np.abs(c_full_sine)
phi_sensor_sine = np.asarray(S_J.T[SENSOR_IDX, :])

# Sort descending so rank-k corresponds to top-k
sorted_haar = np.sort(mag_haar)[::-1]
sorted_sine = np.sort(mag_sine)[::-1]


def boundary_dual(eps, mag, c_full, phi_sensor, A=None):
    """Method 2: marginal contribution of boundary element."""
    # element about to enter as eps decreases = largest |c| < eps
    out_mask = mag < eps
    if not np.any(out_mask):
        return 0.0, None, 0.0
    cand_mag = mag[out_mask]
    cand_idx = np.where(out_mask)[0]
    i_enter = int(cand_idx[np.argmax(cand_mag)])
    # contribution if this element entered: depends on basis.
    # For SINE (A diagonal), c_active_i = c_full_i exactly.
    # For HAAR (A not diagonal), we compute the actual jump by re-solving
    # with this element added.
    if A is None:
        # diagonal case
        c_active_i = c_full[i_enter]
        contrib = phi_sensor[i_enter] * c_active_i
        return contrib, i_enter, abs(mag[i_enter] - eps)
    # Haar: re-solve with this element added to the current active set
    cur_mask = mag >= eps
    new_mask = cur_mask.copy()
    new_mask[i_enter] = True
    # Solve under new_mask
    A_arr = np.asarray(A)
    A_eff_new = np.where(new_mask[:, None],
                         new_mask[None, :] * A_arr,
                         np.eye(N))
    b_eff_new = new_mask * b_haar
    c_new = np.linalg.solve(A_eff_new, b_eff_new)
    # Solve under cur_mask
    A_eff_cur = np.where(cur_mask[:, None],
                         cur_mask[None, :] * A_arr,
                         np.eye(N))
    b_eff_cur = cur_mask * b_haar
    c_cur = np.linalg.solve(A_eff_cur, b_eff_cur)
    u_new = (np.asarray(W_HAAR).T @ c_new)[SENSOR_IDX]
    u_cur = (np.asarray(W_HAAR).T @ c_cur)[SENSOR_IDX]
    # Jump magnitude when element enters: u_new - u_cur (J increases by this)
    # We define dual = -(jump) so it lines up with FD which gives -(jump)/(2h)
    # for an upward crossing of eps. Actually: as eps increases past |c_i|,
    # element i exits, J -> u_cur (smaller mask).  J change = u_cur - u_new.
    return u_cur - u_new, i_enter, abs(mag[i_enter] - eps)


def report(label, J_at_eps, J_soft, mag, c_full, phi_sensor, A=None):
    print(f"\n## {label}")
    print(f"   N = {N}, theta = {THETA}, sensor at x = "
          f"{float(x_grid[SENSOR_IDX]):.4f}")
    sorted_mag = np.sort(mag)[::-1]
    print(f"   |c| ranks 13..18 (descending):")
    for r in range(13, 19):
        print(f"       rank {r:>2}: |c| = {sorted_mag[r]:.6e}")
    print()

    # Choose two eps values: one safe (midway between rank 15 and 16)
    eps_safe = 0.5 * (sorted_mag[15] + sorted_mag[16])
    # one near a flip (just above rank 15's |c|)
    eps_flip = sorted_mag[15] - 1e-15
    # FD step that does NOT span a flip vs one that DOES
    h_small = (sorted_mag[15] - sorted_mag[16]) * 0.01
    h_span = (sorted_mag[15] - sorted_mag[16]) * 0.6

    grad_soft = jax.grad(J_soft, argnums=1)

    for label2, eps, h in [
        ("eps midway (no flip in window)", eps_safe, h_small),
        ("eps midway, h spans a flip   ", eps_safe, h_span),
        ("eps AT flip point            ", eps_flip, h_small),
    ]:
        m_lo = mag >= (eps - h)
        m_hi = mag >= (eps + h)
        n_flip = int(np.sum(m_lo ^ m_hi))
        Jlo = float(J_at_eps(theta, eps - h))
        Jhi = float(J_at_eps(theta, eps + h))
        fd = (Jhi - Jlo) / (2.0 * h)
        dual, i_in, gap = boundary_dual(float(eps), mag, c_full,
                                        phi_sensor, A=A)
        # Smooth FD: jax.grad through sigmoid
        sg_1m3 = float(grad_soft(theta, float(eps), 1e-3))
        sg_1m4 = float(grad_soft(theta, float(eps), 1e-4))
        print(f"   {label2}  eps={float(eps):.6e}  h={h:.2e}")
        print(f"       flips between eps+/-h: {n_flip:d}")
        print(f"       FD       = {fd:+.4e}")
        print(f"       dual     = {dual:+.4e}    "
              f"(boundary i={i_in}, gap={gap:.2e})")
        print(f"       soft tau=1e-3 = {sg_1m3:+.4e}")
        print(f"       soft tau=1e-4 = {sg_1m4:+.4e}")


report("SINE basis (A diagonal -- dual formula closed-form)",
       J_sine_at_eps, J_sine_soft, mag_sine, c_full_sine, phi_sensor_sine,
       A=None)

report("HAAR basis (A non-diagonal -- dual is empirical re-solve)",
       J_haar_at_eps, J_haar_soft, mag_haar, c_full_haar, phi_sensor_haar,
       A=A_HAAR)


# ----- Cleaner element-wise jump test -----
# For each rank r, jump_r = J(eps just below sorted[r]) - J(eps just
# above sorted[r]).  This is the *true* contribution of rank-r mode
# under the actual adaptive solve.  Dual prediction: phi[i] * c_full[i]
# for the index i at rank r.  In sine A is diagonal so c_active_i =
# c_full_i, dual is exact.  In Haar A is non-diagonal, dual is an
# approximation; the size of the disagreement quantifies how much the
# cross-mode coupling moves things.

def element_jump_test(label, mag, c_full, phi_sensor, J_at_eps, ranks):
    print(f"\n## Element-wise jump test -- {label}")
    print(f"   {'rank':>4} {'mode_idx':>9} {'|c| at rank':>13} "
          f"{'true_jump':>13} {'dual_pred':>13} {'rel_err':>10}")
    sorted_mag = np.sort(mag)[::-1]
    idx_by_rank = np.argsort(-mag)
    for r in ranks:
        eps_val = sorted_mag[r]
        h = 1e-4 * eps_val
        # mask includes rank 0..r-1 at eps_val + h (rank r excluded)
        # mask includes rank 0..r at eps_val - h (rank r included)
        J_above = float(J_at_eps(theta, jnp.asarray(eps_val + h)))
        J_below = float(J_at_eps(theta, jnp.asarray(eps_val - h)))
        # As eps increases through sorted_mag[r], rank-r exits mask.
        # J(eps_above) lacks rank-r's contribution.
        true_jump = J_below - J_above   # rank-r contribution
        idx_r = int(idx_by_rank[r])
        dual_pred = float(phi_sensor[idx_r] * c_full[idx_r])
        err = abs(true_jump - dual_pred) / (abs(true_jump) + 1e-30)
        print(f"   {r:>4d} {idx_r:>9d} {eps_val:>13.4e} "
              f"{true_jump:>+13.4e} {dual_pred:>+13.4e} "
              f"{err:>10.2e}")


element_jump_test("SINE basis", mag_sine, c_full_sine, phi_sensor_sine,
                  J_sine_at_eps, ranks=[5, 10, 15, 20, 30, 50])
element_jump_test("HAAR basis", mag_haar, c_full_haar, phi_sensor_haar,
                  J_haar_at_eps, ranks=[5, 10, 15, 20, 30, 50])
