"""LEVER 3, FAIR wall-clock race. bench_race.py was invalid: the fp64 arm
synced to host every 25 iters (float(...)), the fp32 arm ran 200 iters inside a
single fori_loop. That measured dispatch overhead, not precision.

Here BOTH arms run their entire iteration count inside fori_loops with the same
structure, using the iteration counts to-tolerance established previously
(fp64: 975 iters; fp32-IR: 6 x 200 inner + 6 fp64 residuals)."""
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf")
from bench_precision import build

NL, NC, DIM, CONTRAST = 4, 4, 3, 100.0
IT64, INNER, OUTERS = 975, 200, 6


def cg_body(ap):
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
    ap64, ap32 = b64["apply"], b32["apply"]
    rhs = jnp.asarray(np.random.RandomState(2).randn(N))
    rhs = rhs / jnp.linalg.norm(rhs)
    nb = float(jnp.linalg.norm(rhs))

    # ---- A: plain fp64 CG, ALL 975 iters inside one fori_loop ----
    @jax.jit
    def solve64(rhs):
        x = jnp.zeros(N); r = rhs - ap64(x); p = r; rs = jnp.dot(r, r)
        c = jax.lax.fori_loop(0, IT64, cg_body(ap64), (x, r, p, rs))
        return c[0]

    x64 = solve64(rhs); jax.block_until_ready(x64)          # warm compile
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        x64 = solve64(rhs); jax.block_until_ready(x64)
        ts.append(time.perf_counter() - t0)
    t64 = min(ts)
    r64 = float(jnp.linalg.norm(rhs - jax.jit(ap64)(x64)) / nb)

    # ---- B: fp32 IR, whole thing (all outers) inside one jit ----
    @jax.jit
    def solve_ir(rhs):
        ap64j, ap32j = ap64, ap32
        def outer(k, x):
            r = rhs - ap64j(x)
            scale = jnp.linalg.norm(r)
            r32 = (r / scale).astype(jnp.float32)
            xi = jnp.zeros(N, jnp.float32)
            c = jax.lax.fori_loop(0, INNER, cg_body(ap32j),
                                  (xi, r32, r32, jnp.dot(r32, r32)))
            return x + scale * c[0].astype(jnp.float64)
        return jax.lax.fori_loop(0, OUTERS, outer, jnp.zeros(N))

    xir = solve_ir(rhs); jax.block_until_ready(xir)         # warm compile
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        xir = solve_ir(rhs); jax.block_until_ready(xir)
        ts.append(time.perf_counter() - t0)
    tIR = min(ts)
    rIR = float(jnp.linalg.norm(rhs - jax.jit(ap64)(xir)) / nb)

    print(f"  A) plain fp64 CG : {IT64} iters (1 fori_loop), "
          f"resid {r64:.3e}, wall {t64*1e3:8.2f} ms")
    print(f"  B) fp32 IR       : {OUTERS}x{INNER} fp32 + {OUTERS} fp64 matvecs, "
          f"resid {rIR:.3e}, wall {tIR*1e3:8.2f} ms")
    print(f"\n  ==> FAIR end-to-end speedup of fp32-IR over fp64 CG = {t64/tIR:.2f}x")
    print(f"  (per-matvec fp64/fp32 ratio measured earlier: 2.13x — IR cannot beat that,")
    print(f"   and pays for it with {OUTERS*INNER/IT64:.2f}x more matvecs from Krylov restarts)")


if __name__ == "__main__":
    print("device:", jax.devices()[0], f" grid={NC*2**NL}^{DIM}")
    main()
