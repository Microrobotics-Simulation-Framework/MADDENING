"""LEVER 3 measurement: fp32 vs fp64 for the matrix-free wavelet matvec.

Run: XLA_PYTHON_CLIENT_PREALLOCATE=false python bench_precision.py [--cpu]
"""
import sys, time, functools
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from maddening.nodes.adaptive.wavelets import transform as T
from maddening.nodes.adaptive.wavelets import matrixfree as mf
from maddening.nodes.adaptive.wavelets import operator as op
from maddening.nodes.adaptive.wavelets import precond as pc


def build(n_levels, n_coarse, order, dim, contrast, dt):
    side = n_coarse * 2 ** n_levels
    N = T.n_dofs(n_levels, n_coarse, dim)
    h = 1.0 / side
    norms = op.column_norms_fast(n_levels, n_coarse, order, dim, h).astype(dt)
    # variable coefficient: smooth checker-ish field with given contrast
    xs = [np.arange(side) / side] * dim
    mesh = np.meshgrid(*xs, indexing="ij")
    s = np.ones(mesh[0].shape)
    for m in mesh:
        s = s * np.sin(2 * np.pi * m)
    a_np = 1.0 + (contrast - 1.0) * 0.5 * (1.0 + np.tanh(8.0 * s))
    a = jnp.asarray(a_np.reshape(-1), dtype=dt)
    a_phys = mf.make_varcoeff_apply(a, side, dim, h, mass=1.0)
    # reference-lagged diagonal (built in fp64 always; cast after)
    a_ref = jnp.ones(N, dtype=jnp.float64)
    aref_phys = mf.make_varcoeff_apply(a_ref, side, dim, h, mass=1.0)
    levels = {1: T.levels_1d, 2: T.levels_2d, 3: T.levels_3d}[dim](n_levels, n_coarse)
    diagA = mf.wave_diagonal_fast(n_levels, n_coarse, order, dim,
                                  op.column_norms_fast(n_levels, n_coarse, order, dim, h),
                                  aref_phys)
    D = pc.diagonal_scaling(diagA, levels, "hybrid").astype(dt)
    apply = mf.make_wave_apply(n_levels, n_coarse, order, dim, norms, a_phys, D)
    return dict(side=side, N=N, h=h, apply=apply, norms=norms, D=D, a=a,
                levels=levels, a_phys=a_phys)


def timeit(fn, x, n=30, warm=3):
    f = jax.jit(fn)
    for _ in range(warm):
        y = f(x); y.block_until_ready()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        y = f(x); y.block_until_ready()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts))


if __name__ == "__main__":
    print("device:", jax.devices()[0])
    for (nl, nc, dim) in [(4, 4, 3), (5, 2, 3), (3, 4, 3)]:
        side = nc * 2 ** nl
        row = {}
        for dt, name in [(jnp.float64, "fp64"), (jnp.float32, "fp32")]:
            b = build(nl, nc, 4, dim, 100.0, dt)
            key = jax.random.PRNGKey(0)
            x = jax.random.normal(key, (b["N"],), dtype=dt)
            med, mn = timeit(b["apply"], x)
            row[name] = (med, mn, b["N"])
            print(f"  side={side:3d} N={b['N']:8d} {name}: median {med*1e3:8.3f} ms  min {mn*1e3:8.3f} ms")
        r = row["fp64"][0] / row["fp32"][0]
        print(f"  side={side:3d} -> fp64/fp32 matvec speedup = {r:.2f}x\n")
