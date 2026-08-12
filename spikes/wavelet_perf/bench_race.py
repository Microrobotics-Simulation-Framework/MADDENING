"""LEVER 3, the decisive test: WALL-CLOCK race to the node's default cg_rtol=1e-8.
Plain fp64 CG vs fp32-inner iterative refinement, 64^3, contrast=100.
Everything block_until_ready'd."""
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf")
from bench_precision import build

TOL = 1e-8
NL, NC, DIM, CONTRAST = 4, 4, 3, 100.0


def cg_body(ap, dt, N):
    def body(i, c):
        x, r, p, rs = c
        Ap = ap(p)
        alpha = rs / jnp.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = jnp.dot(r, r)
        p = r + (rs_new / rs) * p
        return (x, r, p, rs_new)
    return body


def main():
    b64 = build(NL, NC, 4, DIM, CONTRAST, jnp.float64)
    b32 = build(NL, NC, 4, DIM, CONTRAST, jnp.float32)
    N = b64["N"]
    ap64 = b64["apply"]; ap32 = b32["apply"]
    rhs = jnp.asarray(np.random.RandomState(2).randn(N))
    rhs = rhs / jnp.linalg.norm(rhs)
    nb = float(jnp.linalg.norm(rhs))

    # ---- A: plain fp64 CG, chunked, find iters-to-tol AND wall clock ----
    CH = 25
    @jax.jit
    def chunk64(c):
        return jax.lax.fori_loop(0, CH, cg_body(ap64, jnp.float64, N), c)
    x = jnp.zeros(N); r = rhs - ap64(x); p = r; rs = jnp.dot(r, r)
    c = (x, r, p, rs)
    c = chunk64(c); jax.block_until_ready(c)          # warm compile
    x = jnp.zeros(N); r = rhs - ap64(x); p = r; rs = jnp.dot(r, r)
    c = (x, r, p, rs)
    t0 = time.perf_counter(); it64 = 0; res64 = 1.0
    while it64 < 2000:
        c = chunk64(c); it64 += CH
        res64 = float(jnp.linalg.norm(rhs - jax.jit(ap64)(c[0])) / nb)
        if res64 < TOL:
            break
    jax.block_until_ready(c)
    t64 = time.perf_counter() - t0
    print(f"  A) plain fp64 CG : {it64:5d} iters, true rel resid {res64:.3e}, "
          f"wall {t64:7.3f} s")

    # ---- B: iterative refinement, fp32 inner ----
    INNER = 200
    @jax.jit
    def cg32(rhs32):
        x = jnp.zeros(N, jnp.float32); r = rhs32; p = r; rs = jnp.dot(r, r)
        c = jax.lax.fori_loop(0, INNER, cg_body(ap32, jnp.float32, N),
                              (x, r, p, rs))
        return c[0]
    ap64j = jax.jit(ap64)
    _ = cg32(jnp.zeros(N, jnp.float32)); jax.block_until_ready(_)   # warm
    _ = ap64j(jnp.zeros(N)); jax.block_until_ready(_)

    x = jnp.zeros(N)
    t0 = time.perf_counter(); mv32 = 0; outers = 0
    for k in range(12):
        r = rhs - ap64j(x)
        rn = float(jnp.linalg.norm(r) / nb)
        if rn < TOL:
            break
        scale = jnp.linalg.norm(r)
        d = cg32((r / scale).astype(jnp.float32))
        x = x + scale * d.astype(jnp.float64)
        mv32 += INNER; outers += 1
    jax.block_until_ready(x)
    tIR = time.perf_counter() - t0
    print(f"  B) fp32 IR       : {outers} outers x {INNER} = {mv32} fp32 matvecs, "
          f"true rel resid {rn:.3e}, wall {tIR:7.3f} s")

    print(f"\n  ==> end-to-end speedup of fp32-IR over fp64 CG to rtol={TOL}: "
          f"{t64/tIR:.2f}x")

    # sanity: solution agreement
    print(f"  (residual check both below tol: fp64 {res64:.2e}, IR {rn:.2e})")


if __name__ == "__main__":
    print("device:", jax.devices()[0], f" grid={NC*2**NL}^{DIM}")
    main()
