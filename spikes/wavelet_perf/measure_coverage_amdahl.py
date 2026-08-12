"""M4: (b) support coverage, and the Amdahl ceiling on lever R2.

Part A: synthesise a REAL CDD-selected Lambda and measure what fraction of grid
        cells are actually touched. This is the make-or-break test for the
        "restricted synthesis" idea. Also tested WITHOUT the coarse level, to
        isolate whether coarse-inclusion alone forces full coverage.
Part B: where does the solve time actually go? Counts matvecs in a real solve and
        computes the Amdahl bound: the best possible speedup if the matvec were FREE.
Part C: does translation invariance survive a variable coefficient? (It must, for
        any direct block-Toeplitz assembly of A_wave.)
"""
from __future__ import annotations
import time
import numpy as np
import jax
import jax.numpy as jnp
from common import build
from maddening.nodes.adaptive.wavelets import cdd as CDD
from maddening.nodes.adaptive.wavelets import matrixfree as mf
from maddening.nodes.adaptive.wavelets import transform as T


def get_rhs(st):
    side, dim = st["side"], st["dim"]
    coords = np.meshgrid(*([np.arange(side) / side] * dim), indexing="ij")
    r2 = sum((c - (0.42 if i == 0 else 0.5)) ** 2 for i, c in enumerate(coords))
    f = jnp.asarray(np.exp(-r2 / 0.10 ** 2).reshape(-1))
    _, wnT = mf.make_wn_ops(st["n_levels"], st["n_coarse"], st["order"],
                            dim, st["norms"])
    return wnT(f) / st["D"]


def run_cdd(st, K=None, count=False):
    N = st["N"]
    K = K or max(8, N // 16)
    b = get_rhs(st)
    n = {"mv": 0}
    ap = st["apply"]
    if count:
        def ap(v, _a=st["apply"]):
            n["mv"] += 1
            return _a(v)

    def solve_masked(mask, rhs):
        return mf.masked_cg_solve(ap, mask, rhs, rtol=1e-8, atol=1e-10)

    t0 = time.perf_counter()
    mask, c, conv = CDD.cdd_select(ap, solve_masked, b, st["coarse"], K)
    mask, c = jax.block_until_ready((mask, c))
    return np.asarray(mask), np.asarray(c), bool(conv), time.perf_counter() - t0, n["mv"], b


def part_a(st, mask):
    """Fraction of grid cells touched by synthesis of a Lambda-supported vector."""
    N = st["N"]
    lev = st["lev_np"]
    synth = T._SYNTH[st["dim"]]
    key = jax.random.PRNGKey(7)
    v = jax.random.normal(key, (N,), dtype=jnp.float64)

    def cover(m):
        vv = jnp.where(jnp.asarray(m), v, 0.0)
        u = synth(vv / st["norms"], st["n_levels"], st["n_coarse"], st["order"])
        u = np.abs(np.asarray(u))
        return u

    print("\n--- (b) SUPPORT COVERAGE of synthesis(Lambda-supported v) ---")
    for name, m in (("full Lambda (coarse included, as CDD returns)", mask),
                    ("Lambda MINUS coarse level", mask & (lev != lev.min())),
                    ("coarse level ONLY", (lev == lev.min())),
                    ("single coarse DOF", np.eye(N, dtype=bool)[0]),
                    ("finest level of Lambda only", mask & (lev == lev.max()))):
        u = cover(m)
        mx = u.max() if u.max() > 0 else 1.0
        for tol in (0.0, 1e-12, 1e-6):
            frac = (u > tol * mx).mean() if tol else (u != 0).mean()
        f0 = (u != 0).mean(); f12 = (u > 1e-12 * mx).mean(); f6 = (u > 1e-6 * mx).mean()
        print(f"  {name:45s} |m|={int(np.sum(m)):6d}  cells touched: "
              f"exact={100*f0:6.2f}%  >1e-12={100*f12:6.2f}%  >1e-6={100*f6:6.2f}%")


def part_c(st):
    """Is A_wave block-Toeplitz (translation invariant) under a variable coeff?"""
    print("\n--- (c') TRANSLATION INVARIANCE of A_wave columns ---")
    N = st["N"]
    ids, reps = T.structural_blocks(st["n_levels"], st["n_coarse"], st["dim"])
    for kind, ct in (("laplacian", 1.0), ("varcoeff", 100.0)):
        s2 = build(st["n_levels"], st["n_coarse"], st["dim"], kind=kind, contrast=ct)
        ap = jax.jit(s2["apply"])
        # take the finest structural block; compare diag entries of two members
        b_last = int(ids.max())
        members = np.nonzero(ids == b_last)[0]
        j0, j1 = int(members[0]), int(members[len(members) // 2])
        c0 = np.asarray(ap(jax.nn.one_hot(j0, N, dtype=jnp.float64)))
        c1 = np.asarray(ap(jax.nn.one_hot(j1, N, dtype=jnp.float64)))
        # compare the SORTED magnitude profile (translation-invariant columns have
        # identical value multisets, just permuted)
        p0 = np.sort(np.abs(c0))[::-1][:50]
        p1 = np.sort(np.abs(c1))[::-1][:50]
        rel = np.linalg.norm(p0 - p1) / (np.linalg.norm(p0) + 1e-300)
        print(f"  {kind:10s}: two columns of the SAME structural block differ in "
              f"sorted-magnitude profile by {rel:.3e}  "
              f"({'TRANSLATION-INVARIANT' if rel < 1e-10 else 'NOT invariant'})")


def main():
    for (dim, nl, nc) in ((3, 4, 2), (3, 5, 2)):
        st = build(nl, nc, dim, kind="varcoeff", contrast=100.0)
        N = st["N"]
        print(f"\n================ {st['side']}^3  N={N}  varcoeff contrast=100 ================")
        mask, c, conv, t_tot, nmv, b = run_cdd(st, count=False)
        print(f"CDD solve: |Lambda|={mask.sum()}  converged={conv}  wall={t_tot:.2f}s")

        # count matvecs on a traced-but-counted run (python counter under jit trace
        # counts trace-time calls, so instead: time a matvec and infer)
        ap = jax.jit(st["apply"])
        v = jax.random.normal(jax.random.PRNGKey(0), (N,), dtype=jnp.float64)
        ap(v).block_until_ready()
        reps = 20
        t0 = time.perf_counter()
        for _ in range(reps):
            r = ap(v)
        r.block_until_ready()
        t_mv = (time.perf_counter() - t0) / reps

        part_a(st, mask)

        print(f"\n--- (B) AMDAHL CEILING on R2 ---")
        print(f"  one matvec                       : {t_mv*1e3:.3f} ms")
        print(f"  full CDD solve wall              : {t_tot*1e3:.1f} ms")
        est_mv = t_tot / t_mv
        print(f"  => solve is equivalent to ~{est_mv:.0f} matvecs of wall time")
        for frac in (0.3, 0.5, 0.7):
            print(f"     if matvec is {100*frac:.0f}% of the solve, a FREE matvec gives "
                  f"at most {1/(1-frac):.2f}x")
        if dim == 3 and nl == 4:
            part_c(st)


if __name__ == "__main__":
    main()
