"""Investigate the 7% gradient gap at k=128 even though forward J is exact."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from q2_frozen_set_gradient import (
    rhs_coeffs, LAMBDAS, PHI, SENSOR_IDX, J_full, J_frozen,
)

theta0 = jnp.asarray(0.35)

# Full per-mode contributions to dJ/dtheta
db_dtheta = jax.jacrev(rhs_coeffs)(theta0)            # shape (N_BASIS,)
phi_sensor = np.asarray(PHI[SENSOR_IDX])              # shape (N_BASIS,)
inv_lam = 1.0 / np.asarray(LAMBDAS)
per_mode = np.asarray(db_dtheta) * inv_lam * phi_sensor   # per-mode dJ/dtheta

# Active set ranking
b = np.asarray(rhs_coeffs(theta0))
mag = np.abs(b)
rank_by_mag = np.argsort(-mag)            # mode indices sorted by descending |b|

print("# per-mode contribution to dJ/dtheta (sorted by |b| descending)")
print(f"{'rank':>4} {'mode_k':>6} {'|b|':>11} {'db/dtheta':>13} "
      f"{'1/lambda':>11} {'phi(x_s)':>9} {'contrib':>13}")
for r in range(40):
    i = rank_by_mag[r]
    print(f"{r+1:>4} {i+1:>6} {mag[i]:>11.3e} "
          f"{float(db_dtheta[i]):>+13.3e} {inv_lam[i]:>11.3e} "
          f"{phi_sensor[i]:>+9.3f} {per_mode[i]:>+13.3e}")

print()
print("# Cumulative sum of per-mode contributions in |b|-rank order")
cumsum = np.cumsum(per_mode[rank_by_mag])
total = float(cumsum[-1])
print(f"total (all 256 modes)         = {total:+.6e}")
for K in [4, 8, 16, 32, 64, 96, 128, 192, 256]:
    partial = float(cumsum[K - 1])
    print(f"  top-{K:>3d}  by |b|: partial = {partial:+.6e}  "
          f"miss = {total - partial:+.3e}")
