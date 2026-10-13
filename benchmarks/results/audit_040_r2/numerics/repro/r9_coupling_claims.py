"""Every numeric claim in coupling/acceleration.py's error-estimate docstrings."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from maddening.core.coupling.acceleration import (
    error_amplification, estimated_error, relaxation_step_scale)

print("CLAIM 1: residuals 0.5, 5, 0.25, 2.5 converge at rho=0.71/pass, but every")
print("         second one-step ratio reads 0.05 and flatters the bound 14x.")
r = [0.5, 5.0, 0.25, 2.5]
print(f"   one-step r2/r1 = {r[2]/r[1]:.4f}   two-step sqrt(r2/r0) = "
      f"{np.sqrt(r[2]/r[0]):.4f}")
amp_1step = 1.0/(1.0 - r[2]/r[1])
amp = float(error_amplification(jnp.asarray(r[2]), jnp.asarray(r[1]),
                                jnp.asarray(r[0])))
print(f"   1/(1-rho) with one-step only = {amp_1step:.4f}")
print(f"   error_amplification(...)     = {amp:.4f}   "
      f"ratio = {amp_1step/amp:.2f}x  (claim: 14)")

print("\nCLAIM 2: two-mode contraction (0.999, 0.2): rho reads 0.2 while the")
print("         distance still to travel is 122x the estimate.")
# x_k = a*0.999^k + b*0.2^k ; residual r_k = |x_k - x_{k+1}|
a, b = 1e-3, 1.0
k = np.arange(0, 12)
x = a*0.999**k + b*0.2**k
res = np.abs(np.diff(x))
for i in (3, 4, 5):
    amp_i = float(error_amplification(jnp.asarray(res[i]), jnp.asarray(res[i-1]),
                                      jnp.asarray(res[i-2])))
    est = float(estimated_error(jnp.asarray(res[i]), jnp.asarray(amp_i)))
    true_remaining = abs(x[i+1] - 0.0)   # fixed point is 0
    print(f"   k={i}: rho_eff={res[i]/res[i-1]:.4f} amp={amp_i:.4f} "
          f"est={est:.4e} true={true_remaining:.4e} "
          f"understatement={true_remaining/est:.1f}x")

print("\nCLAIM 3: relaxation_step_scale returns omega for 'fixed' and 1.0 otherwise")
for acc in ("none", "fixed", "aitken", "iqn-ils", "iqn-imvj"):
    print(f"   {acc:10s} omega=1.5 -> {relaxation_step_scale(acc, 1.5)}")

print("\nCLAIM 4: estimated_error is never smaller than the residual")
bad = 0
rng = np.random.default_rng(7)
for _ in range(20000):
    r0 = float(rng.lognormal(-3, 3))
    amp = float(rng.uniform(0.0, 3.0))
    ss = float(rng.uniform(0.0, 2.0))
    e = float(estimated_error(jnp.asarray(r0), jnp.asarray(amp), ss))
    if e < r0 * (1 - 1e-12):
        bad += 1
print(f"   violations over 20000 random (residual, amplification, step_scale): {bad}")

print("\nCLAIM 5: rejection cases return 0.0 (impossible amplification)")
cases = {
    "non-decreasing residual": (1.0, 0.5, 0.5),
    "equal residuals":         (1.0, 1.0, 1.0),
    "zero predecessor":        (1.0, 0.0, 1.0),
    "nan current":             (np.nan, 1.0, 1.0),
    "inf predecessor":         (1.0, np.inf, 1.0),
    "negative predecessor":    (1.0, -1.0, 1.0),
    "healthy contraction":     (0.1, 1.0, 10.0),
}
for name, (a_, b_, c_) in cases.items():
    v = float(error_amplification(jnp.asarray(a_), jnp.asarray(b_), jnp.asarray(c_)))
    print(f"   {name:24s} -> {v}")

print("\nCLAIM 6: 'the two-step term is sqrt of the one-step one' when prev2 defaults")
r_k, r_km1 = 0.25, 1.0
v_default = float(error_amplification(jnp.asarray(r_k), jnp.asarray(r_km1)))
rho = max(r_k/r_km1, np.sqrt(r_k/r_km1))
print(f"   default prev2: amp={v_default:.5f}   expected 1/(1-{rho:.4f})="
      f"{1/(1-rho):.5f}")
