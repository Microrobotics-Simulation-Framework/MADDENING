"""M5: the Amdahl ceiling done properly (jitted production path) + a correct
translation-invariance test.

Previous run timed an UNJITTED cdd_select (87 s) which is not the production
path. Here the whole solve is jitted, and the matvec count is obtained from an
instrumented CG that returns its iteration count.
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


def cg_counting(op, b, rtol=1e-8, atol=1e-10, maxit=5000):
    """Plain CG returning (x, iters). Same stopping rule shape as lineax CG."""
    bnorm = jnp.linalg.norm(b)
    tol = jnp.maximum(rtol * bnorm, atol)

    def cond(s):
        x, r, p, rs, k = s
        return (jnp.sqrt(rs) > tol) & (k < maxit)

    def body(s):
        x, r, p, rs, k = s
        Ap = op(p)
        alpha = rs / (jnp.vdot(p, Ap) + 1e-300)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = jnp.vdot(r, r)
        p = r + (rs_new / (rs + 1e-300)) * p
        return (x, r, p, rs_new, k + 1)

    x0 = jnp.zeros_like(b)
    s = (x0, b, b, jnp.vdot(b, b), jnp.int32(0))
    x, r, p, rs, k = jax.lax.while_loop(cond, body, s)
    return x, k


def solve_counting(st, K):
    """Run the CDD loop, accumulating the TOTAL matvec count."""
    N = st["N"]
    b = get_rhs(st)
    ap = st["apply"]
    total = {"mv": 0, "outer": 0}

    masked_cg = jax.jit(lambda mask, rhs: cg_counting(
        mf.make_masked_operator_fn(ap, mask), jnp.where(mask, rhs, 0.0)))
    ap_j = jax.jit(ap)

    mask = st["coarse"]
    c, k = masked_cg(mask, b)
    total["mv"] += int(k); total["outer"] += 1
    resid = b - ap_j(c); total["mv"] += 1
    bnorm = float(jnp.linalg.norm(b))
    for it in range(CDD.MAX_OUTER):
        rel = float(jnp.linalg.norm(resid)) / bnorm
        if rel < 1e-6 or int(jnp.sum(mask)) >= K:
            break
        mask = CDD._doerfler_grow(mask, resid, CDD.THETA_D, K)
        c, k = masked_cg(mask, b)
        total["mv"] += int(k); total["outer"] += 1
        resid = b - ap_j(c); total["mv"] += 1
    return mask, c, total


def main():
    for (dim, nl, nc) in ((3, 4, 2), (3, 5, 2)):
        st = build(nl, nc, dim, kind="varcoeff", contrast=100.0)
        N = st["N"]
        K = max(8, N // 16)
        print(f"\n========== {st['side']}^3 N={N} varcoeff contrast=100 K=N/16={K} ==========")

        # 1. one matvec
        ap = jax.jit(st["apply"])
        v = jax.random.normal(jax.random.PRNGKey(0), (N,), dtype=jnp.float64)
        ap(v).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(20):
            r = ap(v)
        r.block_until_ready()
        t_mv = (time.perf_counter() - t0) / 20

        # 2. fully-jitted production solve
        b = get_rhs(st)

        def full_solve(b):
            def solve_masked(mask, rhs):
                return mf.masked_cg_solve(st["apply"], mask, rhs,
                                          rtol=1e-8, atol=1e-10)
            return CDD.cdd_select(st["apply"], solve_masked, b, st["coarse"], K)

        fs = jax.jit(full_solve)
        t0 = time.perf_counter()
        out = jax.block_until_ready(fs(b))
        t_compile = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = jax.block_until_ready(fs(b))
        t_solve = time.perf_counter() - t0

        # 3. matvec count
        mask, c, tot = solve_counting(st, K)

        n_mv = tot["mv"]
        t_mv_total = n_mv * t_mv
        frac = t_mv_total / t_solve
        print(f"  one matvec              : {t_mv*1e3:8.3f} ms")
        print(f"  jitted solve (compile)  : {t_compile:8.2f} s")
        print(f"  jitted solve (run)      : {t_solve:8.3f} s")
        print(f"  CDD outer iterations    : {tot['outer']}")
        print(f"  TOTAL matvecs in solve  : {n_mv}")
        print(f"  matvec time in solve    : {t_mv_total:8.3f} s  = {100*frac:.1f}% of solve")
        print(f"  ==> AMDAHL: a FREE matvec gives at most {1/(1-min(frac,0.999)):.2f}x")
        print(f"  ==> a 2x faster matvec gives at most     {1/(1-frac/2):.2f}x")
        print(f"  ==> a 4x faster matvec gives at most     {1/(1-0.75*frac):.2f}x")

    # ---- corrected translation-invariance test ----
    print("\n========== translation invariance of A_wave (CORRECTED) ==========")
    st = build(4, 2, 3, kind="varcoeff", contrast=100.0)
    N = st["N"]; side = st["side"]
    ids, reps = T.structural_blocks(4, 2, 3)
    a_np = np.asarray(st["a_grid"]).reshape(side, side, side)
    print(f"  coefficient field: min={a_np.min()} max={a_np.max()} "
          f"blob cells={int((a_np > 1).sum())}/{N}")
    for kind, ct in (("laplacian", 1.0), ("varcoeff", 100.0)):
        s2 = build(4, 2, 3, kind=kind, contrast=ct)
        ap = jax.jit(s2["apply"])
        b_last = int(ids.max())
        members = np.nonzero(ids == b_last)[0]
        # pick the member whose grid cell is INSIDE the blob and one far OUTSIDE
        # finest-level detail j maps ~1:1 to a grid cell; find via synthesis support
        synth = T._SYNTH[3]
        def cell_of(j):
            u = np.asarray(synth(jax.nn.one_hot(j, N, dtype=jnp.float64), 4, 2, 4))
            return np.unravel_index(int(np.argmax(np.abs(u))), (side, side, side))
        j_in = j_out = None
        for j in members:
            cx = cell_of(int(j))
            if a_np[cx] > 1 and j_in is None: j_in = int(j)
            if a_np[cx] == 1 and j_out is None: j_out = int(j)
            if j_in is not None and j_out is not None: break
        c0 = np.asarray(ap(jax.nn.one_hot(j_in, N, dtype=jnp.float64)))
        c1 = np.asarray(ap(jax.nn.one_hot(j_out, N, dtype=jnp.float64)))
        p0 = np.sort(np.abs(c0))[::-1][:50]; p1 = np.sort(np.abs(c1))[::-1][:50]
        rel = np.linalg.norm(p0 - p1) / (np.linalg.norm(p0) + 1e-300)
        print(f"  {kind:10s}: cols j_in={j_in} (inside blob) vs j_out={j_out} "
              f"(outside), SAME structural block")
        print(f"              sorted-magnitude profiles differ by {rel:.3e}  -> "
              f"{'TRANSLATION-INVARIANT' if rel < 1e-10 else 'NOT INVARIANT'}")
        print(f"              diag: A[j_in,j_in]={c0[j_in]:.6e}  A[j_out,j_out]={c1[j_out]:.6e}"
              f"   ratio={c0[j_in]/c1[j_out]:.4f}")


if __name__ == "__main__":
    main()
