"""M6: nnz/row of A_wave at 64^3 by RANDOM COLUMN SAMPLING (unbiased), so the
64^3 BCOO memory is measured, not extrapolated.
"""
from __future__ import annotations
import numpy as np, sys
import jax, jax.numpy as jnp
from common import build

def sample(nl, nc, dim, kind, contrast, n_samp=256, seed=0):
    st = build(nl, nc, dim, kind=kind, contrast=contrast)
    N = st["N"]
    rng = np.random.default_rng(seed)
    js = rng.choice(N, size=min(n_samp, N), replace=False)
    f = jax.jit(jax.vmap(lambda j: st["apply"](jax.nn.one_hot(j, N, dtype=jnp.float64))))
    # global max: estimate from the sample (diag-dominant => max is on the sampled cols)
    per = {t: [] for t in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)}
    allc = []
    B = 32
    for s in range(0, len(js), B):
        cols = np.asarray(f(jnp.asarray(js[s:s+B])))
        allc.append(cols)
    cols = np.concatenate(allc, 0)
    gmax = float(np.abs(cols).max())
    out = {}
    for t in per:
        out[t] = float((np.abs(cols) >= t * gmax).sum(axis=1).mean())
    return N, out, gmax

if __name__ == "__main__":
    print("nnz per ROW of A_wave, random-column sample (n=256), thr relative to max|A|")
    for (dim, nl, nc, kind, ct) in ((3, 3, 2, "varcoeff", 100.0),
                                    (3, 4, 2, "varcoeff", 100.0),
                                    (3, 5, 2, "varcoeff", 100.0),
                                    (3, 5, 2, "laplacian", 1.0),
                                    (3, 4, 4, "varcoeff", 100.0)):
        N, out, gmax = sample(nl, nc, dim, kind, ct)
        side = nc * 2 ** nl
        mem = out[1e-12] * N * 16 / 1e9
        print(f"side={side:3d} N={N:7d} n_coarse={nc} {kind:10s} | " +
              "  ".join(f"{t:.0e}:{out[t]:7.1f}" for t in sorted(out)) +
              f"  | BCOO@1e-12 = {mem:6.2f} GB", flush=True)
