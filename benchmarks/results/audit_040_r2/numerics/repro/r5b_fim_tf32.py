"""One short measurement: does fim() return a false 'identifiable' verdict
when J.T @ J is formed at the GPU's default matmul precision?

Construction: residual depends only on (a+b), so F = J.T J has an EXACT
null direction and the true rank is 1 of 2.  In float32 with an
fp32-accurate matmul the null eigenvalue comes back <= 0 and fim says
rank=1, crb=[inf, inf].  If the matmul runs at TF32 (10-bit mantissa) the
null eigenvalue picks up noise ~ sqrt(m) * eps_tf32 * lam_max, which is
far above the cutoff max(n, sqrt(m)) * eps_float32, so fim reports rank=2
with a FINITE crb and no PrecisionLimitWarning.
"""
import os, warnings, sys
import jax, jax.numpy as jnp, numpy as np
from maddening.sysid import fim
from maddening.warnings import PrecisionLimitWarning

M = 4000
rng = np.random.default_rng(20260920)
A = jnp.asarray(rng.normal(size=(M,)), dtype=jnp.float32)

def residual_fn(p):
    return {"r": A * (p["a"] + p["b"]) + 0.5}

params = {"a": jnp.float32(1.0), "b": jnp.float32(1.0)}
cutoff = max(2, np.sqrt(M)) * np.finfo(np.float32).eps

def one(label, precision=None):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if precision:
            with jax.default_matmul_precision(precision):
                rep = fim(residual_fn, params, scale=None)
        else:
            rep = fim(residual_fn, params, scale=None)
        warned = any(issubclass(x.category, PrecisionLimitWarning) for x in w)
    ev = np.asarray(rep.eigvals, dtype=np.float64)
    ratio = ev[0]/ev[-1]
    print(f"  {label:26s} rank={rep.rank}/2  lam0/lam_max={ratio: .4e}  "
          f"({ratio/cutoff: .1f}x cutoff)  crb={np.asarray(rep.crb)}  "
          f"PrecisionLimitWarning={warned}")

print(f"backend={jax.default_backend()} devices={jax.devices()} jax={jax.__version__}")
print(f"jax_default_matmul_precision={jax.config.jax_default_matmul_precision}")
print(f"cutoff max(n,sqrt(m))*eps = {cutoff:.4e};  true rank = 1 of 2")
one("GPU default")
one("precision=highest", "highest")
one("precision=tensorfloat32", "tensorfloat32")
