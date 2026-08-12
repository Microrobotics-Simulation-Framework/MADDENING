"""R2 extension: the ONLY version of (a) that could reduce matvec cost is a
sparse operator RESTRICTED to the active block Lambda x Lambda.

Measures, for a REAL CDD-selected Lambda:
  1. nnz of A_wave restricted to Lambda rows AND Lambda cols, at thresholds.
  2. estimated sparse-matvec cost (bytes moved) vs the matrix-free matvec.
  3. truncation error: ||A_trunc v - A_full v|| / ||A_full v|| at each threshold,
     for a Lambda-supported v (what CG actually multiplies).
"""
from __future__ import annotations
import time, numpy as np
import jax, jax.numpy as jnp
from common import build
from maddening.nodes.adaptive.wavelets import cdd as CDD
from maddening.nodes.adaptive.wavelets import matrixfree as mf


def get_rhs(st):
    side, dim = st["side"], st["dim"]
    coords = np.meshgrid(*([np.arange(side) / side] * dim), indexing="ij")
    r2 = sum((c - (0.42 if i == 0 else 0.5)) ** 2 for i, c in enumerate(coords))
    f = jnp.asarray(np.exp(-r2 / 0.10 ** 2).reshape(-1))
    _, wnT = mf.make_wn_ops(st["n_levels"], st["n_coarse"], st["order"], dim, st["norms"])
    return wnT(f) / st["D"]


def real_lambda(st, K):
    b = get_rhs(st)
    ap = st["apply"]
    def solve_masked(mask, rhs):
        return mf.masked_cg_solve(ap, mask, rhs, rtol=1e-8, atol=1e-10)
    mask, c, conv = CDD.cdd_select(ap, solve_masked, b, st["coarse"], K)
    return np.asarray(mask), b


def main():
    THRESH = [1e-12, 1e-8, 1e-6, 1e-4]
    for (dim, nl, nc, kind, ct) in ((3, 3, 2, "varcoeff", 100.0),
                                    (3, 4, 2, "varcoeff", 100.0)):
        st = build(nl, nc, dim, kind=kind, contrast=ct)
        N = st["N"]; K = max(8, N // 16)
        mask, b = real_lambda(st, K)
        L = np.nonzero(mask)[0]
        kL = len(L)
        print(f"\n==== {st['side']}^3 N={N} {kind} contrast={ct} |Lambda|={kL} (N/16={N//16}) ====")

        # assemble the Lambda columns of A_wave (only kL columns, O(kL*N) not O(N^2))
        ap = jax.jit(jax.vmap(lambda j: st["apply"](jax.nn.one_hot(j, N, dtype=jnp.float64))))
        cols = []
        B = 64
        for s in range(0, kL, B):
            cols.append(np.asarray(ap(jnp.asarray(L[s:s+B]))))
        cols = np.concatenate(cols, 0)          # [kL, N]  columns j in Lambda
        gmax = float(np.abs(cols).max())

        # restrict to Lambda rows -> the Lambda x Lambda block (what a masked
        # sparse matvec would use)
        block = cols[:, L]                       # [kL, kL]
        for t in THRESH:
            nz_block = int((np.abs(block) >= t * gmax).sum())
            nz_full  = int((np.abs(cols)  >= t * gmax).sum())
            print(f"  thr={t:.0e}: LxL nnz={nz_block:9d} ({nz_block/kL:6.1f}/row)  "
                  f"full-row-in-Lambda nnz={nz_full:9d} ({nz_full/kL:6.1f}/row)  "
                  f"LxL mem={nz_block*16/1e6:6.1f}MB")

        # --- matvec cost comparison (bytes moved, the GPU-bound proxy) ---
        # matrix-free: 2 synthesis passes touch all N; measure wall directly
        apj = jax.jit(st["apply"])
        v = jnp.where(jnp.asarray(mask), jax.random.normal(jax.random.PRNGKey(1), (N,), jnp.float64), 0.0)
        apj(v).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(30): r = apj(v)
        r.block_until_ready()
        t_mf = (time.perf_counter() - t0) / 30

        # sparse LxL matvec: nnz*16 bytes moved (value+idx); ~200 GB/s on A2000
        nz12 = int((np.abs(block) >= 1e-12 * gmax).sum())
        nz8  = int((np.abs(block) >= 1e-8  * gmax).sum())
        bw = 200e9
        print(f"  matrix-free masked matvec (measured): {t_mf*1e3:7.3f} ms")
        print(f"  sparse LxL @1e-12 bytes={nz12*16/1e6:.0f}MB -> ~{nz12*16/bw*1e3:6.3f} ms (BW-bound est)")
        print(f"  sparse LxL @1e-8  bytes={nz8*16/1e6:.0f}MB -> ~{nz8*16/bw*1e3:6.3f} ms (BW-bound est)")

        # --- truncation error of the LxL block on a real vector ---
        vb = np.asarray(v)[L]                    # restricted vector
        full = block @ vb
        for t in THRESH:
            Bt = np.where(np.abs(block) >= t * gmax, block, 0.0)
            err = np.linalg.norm(Bt @ vb - full) / (np.linalg.norm(full) + 1e-300)
            print(f"  truncation@{t:.0e}: rel matvec err = {err:.3e}")


if __name__ == "__main__":
    main()
