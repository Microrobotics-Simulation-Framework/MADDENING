"""M1: how sparse IS A_wave? nnz vs N, several thresholds, 1D/2D/3D.

Computes columns of the SCALED operator Ahat matrix-free (O(N) memory per
column), accumulates a log-magnitude histogram, and reports nnz at thresholds
relative to global max|A| (what operator.sparsity_pattern does).
"""
from __future__ import annotations
import sys, time, json
import numpy as np
import jax.numpy as jnp
from common import build, columns

THRESH = [1e-12, 1e-10, 1e-8, 1e-6]
# log10 bins for |a| ; wide enough for any scaling
BINS = np.arange(-30.0, 12.0001, 0.05)


def nnz_scan(n_levels, n_coarse, dim, kind, contrast=1.0, batch=None):
    st = build(n_levels, n_coarse, dim, kind=kind, contrast=contrast)
    N = st["N"]
    if batch is None:
        batch = max(1, min(128, 2 ** 22 // max(N, 1)))
    hist = np.zeros(len(BINS) - 1, dtype=np.int64)
    gmax = 0.0
    nz_exact = 0
    t0 = time.time()
    for s, cols in columns(st["apply"], N, batch=batch):
        a = np.abs(np.asarray(cols))
        gmax = max(gmax, float(a.max()))
        nz_exact += int((a > 0).sum())
        la = np.log10(np.where(a > 0, a, 1e-300))
        h, _ = np.histogram(la[a > 0], bins=BINS)
        hist += h
    dt = time.time() - t0
    cum = np.cumsum(hist[::-1])[::-1]  # cum[i] = count with log10|a| >= BINS[i]
    out = {}
    for t in THRESH:
        cut = np.log10(t * gmax)
        i = np.searchsorted(BINS, cut, side="left")
        i = min(max(i, 0), len(cum) - 1)
        out[t] = int(cum[i])
    return dict(N=N, side=st["side"], dim=dim, kind=kind, contrast=contrast,
                gmax=gmax, nnz_exact=nz_exact, nnz=out, secs=dt)


def main():
    cases = []
    # 1D: sweep N
    for nl in (3, 4, 5, 6, 7, 8):
        cases.append((nl, 2, 1, "laplacian", 1.0))
        cases.append((nl, 2, 1, "varcoeff", 100.0))
    # 2D
    for nl in (2, 3, 4, 5):
        cases.append((nl, 2, 2, "laplacian", 1.0))
        cases.append((nl, 2, 2, "varcoeff", 100.0))
    # 3D
    for nl in (2, 3, 4):
        cases.append((nl, 2, 3, "laplacian", 1.0))
        cases.append((nl, 2, 3, "varcoeff", 100.0))

    res = []
    for nl, nc, dim, kind, ct in cases:
        side = nc * 2 ** nl
        N = side ** dim
        if N > 40000:
            continue
        r = nnz_scan(nl, nc, dim, kind, ct)
        res.append(r)
        d = r["nnz"]
        print(f"dim={dim} side={r['side']:4d} N={N:7d} {kind:10s} "
              f"nnz/N: " + "  ".join(f"{t:.0e}:{d[t]/N:8.1f}" for t in THRESH)
              + f"   dense%={100*d[1e-12]/N**2:6.2f}  ({r['secs']:.1f}s)", flush=True)
    with open("nnz_results.json", "w") as f:
        json.dump([{**r, "nnz": {str(k): v for k, v in r["nnz"].items()}} for r in res], f, indent=1)


if __name__ == "__main__":
    main()
