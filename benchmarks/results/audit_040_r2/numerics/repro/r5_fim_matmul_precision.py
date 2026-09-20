"""Does fim()'s rank cutoff know what precision J.T @ J was formed at?

_resolve_rank_rtol derives the cutoff from np.finfo(eigvals.dtype).eps.
`F = J.T @ J` in sysid.fim is a plain jnp matmul, which honours JAX's
*matmul precision*, a separate knob from the array dtype.  If the matmul
runs at reduced precision (TF32 / bfloat16 accumulation), the error in F
grows by orders of magnitude while the cutoff does not move -- putting
the cutoff back below the noise floor, which is the exact failure the
sqrt(m) term was added to remove.

Construction: an EXACTLY rank-deficient problem (residual depends only
on a+b, so F has a true null direction).  float64 says lambda_0/lambda_max
is ~0; anything float32 reports is its own rounding.
"""
import os, sys, warnings
os.environ.setdefault("JAX_PLATFORMS", os.environ.get("JAX_PLATFORMS", "cpu"))
import jax, jax.numpy as jnp, numpy as np
from maddening.sysid import fim
from maddening.warnings import PrecisionLimitWarning

rng = np.random.default_rng(20260920)
M = 4000                        # m residual rows
A = jnp.asarray(rng.normal(size=(M,)), dtype=jnp.float32)

def residual_fn(p):
    # depends only on (a + b) -> exact rank 1 out of 2
    return {"r": A * (p["a"] + p["b"]) + 0.5}

params = {"a": jnp.float32(1.0), "b": jnp.float32(1.0)}


def one(label, precision=None):
    ctx = jax.default_matmul_precision(precision) if precision else None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if ctx:
            with ctx:
                rep = fim(residual_fn, params, scale=None)
        else:
            rep = fim(residual_fn, params, scale=None)
        warned = [x for x in w if issubclass(x.category, PrecisionLimitWarning)]
    ev = np.asarray(rep.eigvals, dtype=np.float64)
    ratio = ev[0] / ev[-1]
    n, m = 2, M
    cutoff = max(n, np.sqrt(m)) * np.finfo(np.float32).eps
    print(f"  {label:34s} rank={rep.rank}  lam0/lam_max={ratio: .4e}  "
          f"cutoff={cutoff:.3e}  ratio/cutoff={ratio/cutoff: 9.2f}  "
          f"crb={np.asarray(rep.crb)}  warned={bool(warned)}")
    return ratio, cutoff, rep.rank


print(f"backend = {jax.default_backend()}   devices = {jax.devices()}")
print(f"jax version = {jax.__version__}")
print(f"jax_default_matmul_precision = {jax.config.jax_default_matmul_precision}")
print("\nExactly rank-deficient F (true rank 1 of 2), m=4000, n=2, float32:")
one("default matmul precision")
for p in ("highest", "float32", "tensorfloat32", "bfloat16"):
    try:
        one(f"precision={p}", p)
    except Exception as e:
        print(f"  precision={p}: {type(e).__name__}: {e}")
