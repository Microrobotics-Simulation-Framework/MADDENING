"""Light, reliable Amdahl fraction: what % of a PRODUCTION jitted masked-CG
solve is the matvec? Avoids compiling the whole nested CDD while-loop (which
compiles for >10 min). Times one jitted masked matvec, and one jitted masked-CG
solve with a KNOWN iteration count, at a realistic Lambda.
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


def cg_counting(op, b, rtol=1e-8, atol=1e-10, maxit=5000):
    bnorm = jnp.linalg.norm(b); tol = jnp.maximum(rtol * bnorm, atol)
    def cond(s):
        x, r, p, rs, k = s
        return (jnp.sqrt(rs) > tol) & (k < maxit)
    def body(s):
        x, r, p, rs, k = s
        Ap = op(p); alpha = rs / (jnp.vdot(p, Ap) + 1e-300)
        x = x + alpha * p; r = r - alpha * Ap
        rs_new = jnp.vdot(r, r); p = r + (rs_new / (rs + 1e-300)) * p
        return (x, r, p, rs_new, k + 1)
    s = (jnp.zeros_like(b), b, b, jnp.vdot(b, b), jnp.int32(0))
    x, r, p, rs, k = jax.lax.while_loop(cond, body, s)
    return x, k


def main():
    for (dim, nl, nc) in ((3, 4, 2), (3, 5, 2)):
        st = build(nl, nc, dim, kind="varcoeff", contrast=100.0)
        N = st["N"]; K = max(8, N // 16)
        b = get_rhs(st)
        # realistic Lambda: grow doerfler from coarse to ~K (cheap, done eagerly)
        ap_j = jax.jit(st["apply"])
        mask = st["coarse"]
        c = jnp.zeros(N, jnp.float64)
        # a couple of doerfler growths to reach a realistic active set
        for _ in range(6):
            resid = b - ap_j(jnp.where(mask, c, 0.0))
            if int(jnp.sum(mask)) >= K: break
            mask = CDD._doerfler_grow(mask, resid, CDD.THETA_D, K)
            c = mf.masked_cg_solve(st["apply"], mask, b, rtol=1e-8, atol=1e-10)
        mask = jax.block_until_ready(mask)
        kL = int(jnp.sum(mask))

        op = mf.make_masked_operator_fn(st["apply"], mask)
        # one masked matvec
        v = jnp.where(mask, jax.random.normal(jax.random.PRNGKey(1), (N,), jnp.float64), 0.0)
        mvj = jax.jit(op); mvj(v).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(50): r = mvj(v)
        r.block_until_ready()
        t_mv = (time.perf_counter() - t0) / 50

        # one full masked-CG solve, jitted, with iteration count
        cg_j = jax.jit(lambda rhs: cg_counting(op, jnp.where(mask, rhs, 0.0)))
        (x, k) = jax.block_until_ready(cg_j(b))
        iters = int(k)
        t0 = time.perf_counter()
        for _ in range(5): out = cg_j(b)
        jax.block_until_ready(out)
        t_solve = (time.perf_counter() - t0) / 5

        frac = (iters * t_mv) / t_solve
        print(f"\n==== {st['side']}^3 N={N} |Lambda|={kL} (N/16={N//16}) ====")
        print(f"  one masked matvec         : {t_mv*1e3:8.3f} ms")
        print(f"  masked-CG iters           : {iters}")
        print(f"  masked-CG solve wall      : {t_solve*1e3:8.1f} ms")
        print(f"  matvec compute in solve   : {iters*t_mv*1e3:8.1f} ms = {100*frac:.1f}% of solve")
        print(f"  ==> AMDAHL: free matvec gives at most {1/(1-min(frac,0.999)):.2f}x on the inner solve")
        print(f"  ==> matvec made O(|L|)=1/16 cost -> inner solve at most "
              f"{1/(1-frac*15/16):.2f}x (if matvec 16x cheaper)")


if __name__ == "__main__":
    main()
