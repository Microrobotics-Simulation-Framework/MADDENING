"""LEVER 3: (a) fair gradient test fp32-vs-fp64 (no FD cancellation),
(b) iterative refinement: does fp32 inner + fp64 residual actually pay?"""
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf")
from bench_precision import build
from maddening.nodes.adaptive.wavelets import transform as T
from maddening.nodes.adaptive.wavelets import operator as op
from maddening.nodes.adaptive.wavelets import matrixfree as mf


def make_J(dt, nl=2, nc=4, dim=2, contrast=10.0):
    side = nc * 2 ** nl
    h = 1.0 / side
    N = T.n_dofs(nl, nc, dim)
    norms = op.column_norms_fast(nl, nc, 4, dim, h).astype(dt)
    D = jnp.ones(N, dtype=dt)
    srow = jnp.asarray(np.random.RandomState(5).randn(N), dtype=dt)
    rhs = jnp.asarray(np.random.RandomState(6).randn(N), dtype=dt)

    def J(theta):
        a = 1.0 + (contrast - 1.0) * jax.nn.sigmoid(theta)
        a_phys = mf.make_varcoeff_apply(a, side, dim, h, mass=1.0)
        apply = mf.make_wave_apply(nl, nc, 4, dim, norms, a_phys, D)
        x = jnp.zeros(N, dtype=dt)
        r = rhs - apply(x); p = r; rs = jnp.dot(r, r)
        for _ in range(40):
            Ap = apply(p)
            alpha = rs / jnp.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = jnp.dot(r, r)
            p = r + (rs_new / rs) * p
            rs = rs_new
        return jnp.dot(srow, x)
    return J, N


def fair_grad_test():
    """Compare the fp32 gradient DIRECTLY against the fp64 gradient at the same
    theta. This isolates gradient error from finite-difference cancellation."""
    J64, N = make_J(jnp.float64)
    J32, _ = make_J(jnp.float32)
    theta64 = jnp.asarray(np.random.RandomState(7).randn(N) * 0.5, dtype=jnp.float64)
    g64 = jax.grad(J64)(theta64)
    g32 = jax.grad(J32)(theta64.astype(jnp.float32)).astype(jnp.float64)
    rel = float(jnp.linalg.norm(g32 - g64) / jnp.linalg.norm(g64))
    # also cosine, to see if it is a direction error or a scale error
    cos = float(jnp.dot(g32, g64) / (jnp.linalg.norm(g32) * jnp.linalg.norm(g64)))
    return rel, cos


def iterative_refinement(nl=4, nc=4, dim=3, contrast=100.0, inner_iters=200,
                         outers=6):
    """fp32 inner CG + fp64 residual computation. Measure: does the true fp64
    residual keep descending across outer iterations, and at what matvec cost?"""
    b64 = build(nl, nc, 4, dim, contrast, jnp.float64)
    b32 = build(nl, nc, 4, dim, contrast, jnp.float32)
    N = b64["N"]
    ap64 = jax.jit(b64["apply"])
    ap32 = jax.jit(b32["apply"])
    rhs = jnp.asarray(np.random.RandomState(2).randn(N), dtype=jnp.float64)
    rhs = rhs / jnp.linalg.norm(rhs)

    @jax.jit
    def cg32(rhs32, iters=inner_iters):
        x = jnp.zeros(N, dtype=jnp.float32)
        r = rhs32; p = r; rs = jnp.dot(r, r)
        def body(i, c):
            x, r, p, rs = c
            Ap = ap32(p)
            alpha = rs / jnp.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = jnp.dot(r, r)
            p = r + (rs_new / rs) * p
            return (x, r, p, rs_new)
        x, r, p, rs = jax.lax.fori_loop(0, iters, body, (x, r, p, rs))
        return x

    x = jnp.zeros(N, dtype=jnp.float64)
    nb = float(jnp.linalg.norm(rhs))
    print(f"    IR: inner = {inner_iters} fp32 CG iters per outer")
    for k in range(outers):
        r = rhs - ap64(x)                       # fp64 residual
        rn = float(jnp.linalg.norm(r) / nb)
        print(f"      outer {k}: true rel resid {rn:.3e}   (matvecs so far ~{k*inner_iters})")
        if rn < 1e-10:
            break
        scale = jnp.linalg.norm(r)
        d = cg32((r / scale).astype(jnp.float32))   # scaled to keep fp32 range
        x = x + scale * d.astype(jnp.float64)
    r = rhs - ap64(x)
    print(f"      final:   true rel resid {float(jnp.linalg.norm(r)/nb):.3e}")


if __name__ == "__main__":
    print("device:", jax.devices()[0])
    print("\n3b) FAIR gradient test: fp32 grad vs fp64 grad (same theta):")
    rel, cos = fair_grad_test()
    print(f"    rel L2 error of fp32 gradient = {rel:.3e}")
    print(f"    cosine(g32, g64)              = {cos:.10f}")

    print("\n4) Iterative refinement, 64^3, contrast=100:")
    iterative_refinement()
