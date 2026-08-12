"""M3: the decisive test for approach (a).

1. Run a REAL CDD selection to get a realistic Lambda at K=N/16.
2. Report the level distribution of Lambda.
3. Assemble A_wave sparsely (column by column, thresholded) at a size that fits.
4. Measure nnz(A_LL) for that Lambda.
5. Benchmark BCOO matvec (full and restricted) vs the matrix-free matvec.
6. Measure the truncation error a threshold introduces.
"""
from __future__ import annotations
import time, sys
import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsparse
from common import build
from maddening.nodes.adaptive.wavelets import cdd as CDD
from maddening.nodes.adaptive.wavelets import matrixfree as mf


def real_lambda(st, K=None):
    """Run the real CDD loop; return (mask, converged)."""
    N = st["N"]
    K = K or max(8, N // 16)
    side, dim = st["side"], st["dim"]
    # gaussian source like the node's default
    coords = np.meshgrid(*([np.arange(side) / side] * dim), indexing="ij")
    r2 = sum((c - 0.42 if i == 0 else c - 0.5) ** 2 for i, c in enumerate(coords))
    f = jnp.asarray(np.exp(-r2 / 0.10 ** 2).reshape(-1))
    _, wnT = mf.make_wn_ops(st["n_levels"], st["n_coarse"], st["order"], dim, st["norms"])
    b = wnT(f) / st["D"]

    def solve_masked(mask, rhs):
        return mf.masked_cg_solve(st["apply"], mask, rhs, rtol=1e-8, atol=1e-10)

    t0 = time.time()
    mask, c, conv = CDD.cdd_select(st["apply"], solve_masked, b, st["coarse"], K)
    mask = jax.block_until_ready(mask)
    return np.asarray(mask), bool(conv), time.time() - t0, K, b


def assemble_sparse(st, thr_rel=1e-12, batch=64):
    """Assemble A_wave as (rows, cols, vals) numpy arrays, thresholded."""
    N = st["N"]
    f = jax.jit(jax.vmap(lambda j: st["apply"](jax.nn.one_hot(j, N, dtype=jnp.float64))))
    # pass 1: global max
    gmax = 0.0
    for s in range(0, N, batch):
        e = min(s + batch, N)
        gmax = max(gmax, float(jnp.abs(f(jnp.arange(s, e))).max()))
    cut = thr_rel * gmax
    R, C, V = [], [], []
    for s in range(0, N, batch):
        e = min(s + batch, N)
        cols = np.asarray(f(jnp.arange(s, e)))   # cols[i] = A[:, s+i]
        keep = np.abs(cols) >= cut
        ii, rr = np.nonzero(keep)                # ii -> column offset, rr -> row
        R.append(rr.astype(np.int32)); C.append((ii + s).astype(np.int32))
        V.append(cols[ii, rr])
    return (np.concatenate(R), np.concatenate(C), np.concatenate(V), gmax)


def bench(fn, v, reps=20):
    fn(v).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(reps):
        r = fn(v)
    r.block_until_ready()
    return (time.perf_counter() - t0) / reps


def main():
    dim, nl, nc = 3, 4, 2          # 32^3
    st = build(nl, nc, dim, kind="varcoeff", contrast=100.0)
    N = st["N"]
    print(f"=== {st['side']}^{dim}  N={N} varcoeff contrast=100 ===")

    mask, conv, tcdd, K, b = real_lambda(st)
    lev = st["lev_np"]
    print(f"CDD: |Lambda|={mask.sum()} (K={K}, N/16={N//16}) converged={conv} "
          f"({tcdd:.1f}s)")
    print("  level distribution of Lambda (active/total):")
    for L in sorted(set(lev.tolist())):
        m = lev == L
        print(f"    L{L}: {mask[m].sum():6d} / {m.sum():6d}  ({100*mask[m].sum()/m.sum():5.1f}%)")

    print("\nAssembling A_wave sparsely (thr=1e-12) ...", flush=True)
    t0 = time.time()
    R, C, V, gmax = assemble_sparse(st)
    tasm = time.time() - t0
    nnz = len(V)
    print(f"  nnz={nnz} ({nnz/N:.1f}/row)  assembly={tasm:.1f}s  "
          f"BCOO mem={nnz*16/1e6:.0f} MB")

    # nnz restricted to Lambda x Lambda
    mR, mC = mask[R], mask[C]
    nnz_LL = int((mR & mC).sum())
    nnz_rowL = int(mR.sum())
    print(f"  nnz(A_LL)={nnz_LL} ({100*nnz_LL/nnz:.1f}% of nnz; "
          f"{nnz_LL/max(mask.sum(),1):.1f} per active row)  mem={nnz_LL*16/1e6:.0f} MB")
    print(f"  nnz(rows in Lambda, all cols)={nnz_rowL} ({100*nnz_rowL/nnz:.1f}%)")

    # ---- benchmarks ----
    v = jax.random.normal(jax.random.PRNGKey(1), (N,), dtype=jnp.float64)
    mf_apply = jax.jit(st["apply"])
    t_mf = bench(mf_apply, v)

    idx = jnp.stack([jnp.asarray(R), jnp.asarray(C)], axis=1)
    A = jsparse.BCOO((jnp.asarray(V), idx), shape=(N, N))
    A = A.sort_indices()
    sp_apply = jax.jit(lambda x: A @ x)
    t_sp = bench(sp_apply, v)

    sel = np.nonzero(mR & mC)[0]
    ALL = jsparse.BCOO((jnp.asarray(V[sel]),
                        jnp.stack([jnp.asarray(R[sel]), jnp.asarray(C[sel])], axis=1)),
                       shape=(N, N)).sort_indices()
    spLL_apply = jax.jit(lambda x: ALL @ x)
    t_spLL = bench(spLL_apply, v)

    mfmask = jax.jit(mf.make_masked_operator_fn(st["apply"], jnp.asarray(mask)))
    t_mfmask = bench(mfmask, v)

    print(f"\n=== matvec timings (A2000, float64) ===")
    print(f"  matrix-free full        : {t_mf*1e3:8.3f} ms")
    print(f"  matrix-free masked      : {t_mfmask*1e3:8.3f} ms")
    print(f"  BCOO full   (nnz={nnz:9d}): {t_sp*1e3:8.3f} ms   "
          f"speedup vs mf = {t_mf/t_sp:5.2f}x")
    print(f"  BCOO A_LL   (nnz={nnz_LL:9d}): {t_spLL*1e3:8.3f} ms   "
          f"speedup vs mf = {t_mf/t_spLL:5.2f}x")

    # ---- truncation error vs threshold ----
    print(f"\n=== truncation error: ||A_thr v - A v|| / ||A v|| (random v) ===")
    ref = mf_apply(v)
    nref = float(jnp.linalg.norm(ref))
    av = np.abs(V)
    for t in (1e-12, 1e-10, 1e-8, 1e-6, 1e-4):
        k = av >= t * gmax
        At = jsparse.BCOO((jnp.asarray(V[k]),
                           jnp.stack([jnp.asarray(R[k]), jnp.asarray(C[k])], axis=1)),
                          shape=(N, N)).sort_indices()
        err = float(jnp.linalg.norm(At @ v - ref)) / nref
        print(f"  thr={t:.0e}  nnz={int(k.sum()):9d} ({k.sum()/N:7.1f}/row)  "
              f"rel err={err:.3e}")


if __name__ == "__main__":
    main()
