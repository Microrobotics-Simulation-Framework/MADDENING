"""R2 crux: is the restricted-block (a) approach ECONOMICAL?

Two deciding measurements:

(1) PROPER translation-invariance test (the salvaged one compared only sorted
    top-50 magnitudes -> passes trivially because the contrast blob is tiny and
    two random finest-level DOFs both sit in a=1 region). Here: take the true
    column of a member, take the representative column of the SAME structural
    block ROLLED to that member's grid offset, compare the FULL vectors. If they
    disagree under varcoeff, assembly needs kL full matvecs (no amortization).

(2) REAL jax BCOO spmv on the LxL block (not a bandwidth-bound estimate) vs the
    matrix-free masked matvec, jitted, at 32^3 and 64^3. Plus the assembly cost
    (kL matvecs) to judge amortization against the ~180-iter inner solve.
"""
from __future__ import annotations
import time, numpy as np
import jax, jax.numpy as jnp
from jax.experimental import sparse as jsparse
from common import build
from maddening.nodes.adaptive.wavelets import cdd as CDD
from maddening.nodes.adaptive.wavelets import matrixfree as mf
from maddening.nodes.adaptive.wavelets import transform as T


def get_rhs(st):
    side, dim = st["side"], st["dim"]
    coords = np.meshgrid(*([np.arange(side) / side] * dim), indexing="ij")
    r2 = sum((c - (0.42 if i == 0 else 0.5)) ** 2 for i, c in enumerate(coords))
    f = jnp.asarray(np.exp(-r2 / 0.10 ** 2).reshape(-1))
    _, wnT = mf.make_wn_ops(st["n_levels"], st["n_coarse"], st["order"], dim, st["norms"])
    return wnT(f) / st["D"]


def real_lambda(st, K):
    b = get_rhs(st); ap = st["apply"]
    def solve_masked(mask, rhs):
        return mf.masked_cg_solve(ap, mask, rhs, rtol=1e-8, atol=1e-10)
    mask, c, conv = CDD.cdd_select(ap, solve_masked, b, st["coarse"], K)
    return np.asarray(mask), b


def col(st, j):
    N = st["N"]
    return np.asarray(st["apply"](jax.nn.one_hot(j, N, dtype=jnp.float64)))


def test_invariance(dim=3, nl=4, nc=2):
    """Compare a member's TRUE column to the representative column rolled into
    place, for laplacian vs varcoeff. Full-vector relative error."""
    print("=== (1) PROPER translation-invariance (full-column) ===")
    side = nc * 2 ** nl
    ids, reps = T.structural_blocks(nl, nc, dim)
    ids = np.asarray(ids)
    for kind, ct in (("laplacian", 1.0), ("varcoeff", 100.0)):
        st = build(nl, nc, dim, kind=kind, contrast=ct)
        N = st["N"]
        a_np = None if st["a_grid"] is None else np.asarray(st["a_grid"]).reshape([side]*dim)
        # finest structural block
        b_last = int(ids.max())
        members = np.nonzero(ids == b_last)[0]
        synth = T._SYNTH[dim]
        def cell_of(j):
            u = np.asarray(synth(jax.nn.one_hot(int(j), N, dtype=jnp.float64), nl, nc, 4))
            return np.unravel_index(int(np.argmax(np.abs(u))), tuple([side]*dim))
        rep = int(members[0]); rc = cell_of(rep); crep = col(st, rep)
        crep_g = crep.reshape([side]*dim)
        # try to reconstruct another member's column by rolling the representative
        errs = []
        for j in members[1:12]:
            jc = cell_of(int(j))
            shift = tuple(int(a - b) for a, b in zip(jc, rc))
            pred = np.roll(crep_g, shift, axis=tuple(range(dim))).reshape(-1)
            true = col(st, int(j))
            rel = np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-300)
            inblob = "" if a_np is None else (" [near-blob]" if a_np[jc] > 1 or a_np[rc] > 1 else "")
            errs.append(rel)
        errs = np.array(errs)
        print(f"  {kind:10s}: roll-representative vs true column, rel err over "
              f"{len(errs)} members: min={errs.min():.2e} median={np.median(errs):.2e} max={errs.max():.2e}")
        print(f"             -> {'INVARIANT (amortizable)' if errs.max()<1e-8 else 'NOT invariant -> needs kL full matvecs'}")


def bench_spmv(dim, nl, nc, thr=1e-8):
    st = build(nl, nc, dim, kind="varcoeff", contrast=100.0)
    N = st["N"]; K = max(8, N // 16)
    mask, b = real_lambda(st, K)
    L = np.nonzero(mask)[0]; kL = len(L)
    # assemble kL columns (this IS the assembly cost)
    apv = jax.jit(jax.vmap(lambda j: st["apply"](jax.nn.one_hot(j, N, dtype=jnp.float64))))
    t0 = time.perf_counter()
    cols = []
    for s in range(0, kL, 64):
        cols.append(np.asarray(apv(jnp.asarray(L[s:s+64]))))
    cols = np.concatenate(cols, 0)
    t_assemble = time.perf_counter() - t0
    gmax = float(np.abs(cols).max())
    block = cols[:, L]                      # [kL,kL]
    keep = np.abs(block) >= thr * gmax
    rows, colsi = np.nonzero(keep)
    vals = block[rows, colsi]
    nnz = len(vals)
    idx = jnp.asarray(np.stack([rows, colsi], 1))
    data = jnp.asarray(vals)
    A = jsparse.BCOO((data, idx), shape=(kL, kL))
    x = jax.random.normal(jax.random.PRNGKey(2), (kL,), jnp.float64)

    @jax.jit
    def spmv(x): return A @ x
    spmv(x).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(100): r = spmv(x)
    r.block_until_ready()
    t_spmv = (time.perf_counter() - t0) / 100

    # matrix-free masked matvec (full N)
    op = jax.jit(mf.make_masked_operator_fn(st["apply"], jnp.asarray(mask)))
    v = jnp.where(jnp.asarray(mask), jax.random.normal(jax.random.PRNGKey(3), (N,), jnp.float64), 0.0)
    op(v).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(100): r = op(v)
    r.block_until_ready()
    t_mf = (time.perf_counter() - t0) / 100

    print(f"\n=== (2) {st['side']}^3 N={N} |Lambda|={kL} thr={thr:.0e} ===")
    print(f"  LxL nnz={nnz} ({nnz/kL:.0f}/row)  mem={nnz*16/1e6:.1f}MB")
    print(f"  matrix-free masked matvec : {t_mf*1e3:8.3f} ms")
    print(f"  REAL jax BCOO spmv (LxL)  : {t_spmv*1e3:8.3f} ms   -> speedup {t_mf/t_spmv:.2f}x")
    print(f"  assembly cost (kL matvecs): {t_assemble*1e3:8.1f} ms = {t_assemble/max(t_mf,1e-9):.0f} matvecs")
    print(f"  break-even: assembly pays off only if inner solve does "
          f">{t_assemble/max(t_mf-t_spmv,1e-12):.0f} iters at this Lambda")


def main():
    test_invariance(3, 4, 2)
    for (dim, nl, nc) in ((3, 4, 2), (3, 5, 2)):
        bench_spmv(dim, nl, nc, thr=1e-8)


if __name__ == "__main__":
    main()
