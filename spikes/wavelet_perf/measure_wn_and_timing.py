"""M2: nnz(Wn) (the synthesis matrix itself), and baseline matvec timings.

nnz(Wn) bounds any sparse-triple-product assembly route: A_wave = Wn^T A_phys Wn
cannot be assembled sparsely if Wn itself does not fit.
"""
from __future__ import annotations
import time
import numpy as np
import jax
import jax.numpy as jnp
from common import build
from maddening.nodes.adaptive.wavelets import transform as T


def nnz_wn(n_levels, n_coarse, dim, order=4, thr=1e-12):
    """nnz of the synthesis matrix W, column by column (O(N) memory)."""
    N = T.n_dofs(n_levels, n_coarse, dim)
    synth = T._SYNTH[dim]
    f = jax.jit(jax.vmap(lambda j: synth(jax.nn.one_hot(j, N, dtype=jnp.float64),
                                         n_levels, n_coarse, order)))
    batch = max(1, min(128, 2 ** 22 // N))
    tot = 0
    per_level = {}
    lev = np.asarray({1: T.levels_1d, 2: T.levels_2d, 3: T.levels_3d}[dim](
        n_levels, n_coarse))
    for s in range(0, N, batch):
        e = min(s + batch, N)
        cols = np.abs(np.asarray(f(jnp.arange(s, e))))
        m = cols.max()
        cnt = (cols >= thr * m).sum(axis=1)
        tot += int(cnt.sum())
        for i, j in enumerate(range(s, e)):
            per_level.setdefault(int(lev[j]), []).append(int(cnt[i]))
    return N, tot, {k: (len(v), float(np.mean(v))) for k, v in sorted(per_level.items())}


def time_matvec(n_levels, n_coarse, dim, kind, contrast=1.0, reps=20):
    st = build(n_levels, n_coarse, dim, kind=kind, contrast=contrast)
    N = st["N"]
    ap = jax.jit(st["apply"])
    v = jax.random.normal(jax.random.PRNGKey(0), (N,), dtype=jnp.float64)
    ap(v).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(reps):
        r = ap(v)
    r.block_until_ready()
    dt = (time.perf_counter() - t0) / reps
    return N, dt


if __name__ == "__main__":
    print("=== nnz(Wn): synthesis matrix sparsity ===")
    for dim, nls in ((1, (4, 5, 6, 7)), (2, (3, 4, 5)), (3, (2, 3, 4))):
        for nl in nls:
            N, tot, pl = nnz_wn(nl, 2, dim)
            print(f"dim={dim} side={2*2**nl:4d} N={N:6d}  nnz(Wn)/N={tot/N:8.1f}  "
                  f"dense%={100*tot/N**2:6.2f}  per-level(count,mean nnz/col): "
                  + " ".join(f"L{k}:({c},{m:.0f})" for k, (c, m) in pl.items()), flush=True)

    print()
    print("=== matrix-free matvec timing (device default) ===")
    print("backend:", jax.default_backend())
    for dim, nl, nc, kind in ((3, 4, 2, "varcoeff"), (3, 5, 2, "varcoeff"),
                              (3, 4, 2, "laplacian"), (3, 5, 2, "laplacian"),
                              (2, 5, 2, "varcoeff"), (2, 6, 2, "varcoeff")):
        N, dt = time_matvec(nl, nc, dim, kind, contrast=100.0)
        print(f"dim={dim} side={nc*2**nl:4d} N={N:7d} {kind:10s} "
              f"matvec={dt*1e3:8.3f} ms   ({N/dt/1e6:.0f} Mdof/s)", flush=True)
