"""Is the 2.1x bandwidth-bound? Control experiments.

(a) ALU-bound control: does this A2000 actually show ~1/64 fp64:fp32?
(b) Pure-bandwidth control: a copy/axpy kernel -> expect ~2x.
(c) Larger grid (128^3): does the matvec ratio hold, i.e. not launch-bound?
"""
import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf")
from bench_precision import build, timeit


def alu_bound(dt, n=1 << 16, iters=2000):
    """FMA chain on a tiny array: fits in cache, so ALU-limited."""
    x = jnp.asarray(np.random.rand(n), dtype=dt)

    def f(x):
        def body(i, v):
            return v * jnp.asarray(1.0000001, dt) + jnp.asarray(1e-7, dt)
        return jax.lax.fori_loop(0, iters, body, x)
    return timeit(f, x, n=20)


def bw_bound(dt, n=1 << 24):
    """axpy on a big array: memory-bandwidth-limited."""
    x = jnp.asarray(np.random.rand(n), dtype=dt)

    def f(x):
        return 2.0 * x + 1.0
    med, mn = timeit(f, x, n=30)
    gb = 2 * n * jnp.dtype(dt).itemsize / 1e9  # read+write
    return med, mn, gb / mn


if __name__ == "__main__":
    print("device:", jax.devices()[0])
    print("\n(a) ALU-bound control (FMA chain, cache-resident):")
    r = {}
    for dt, name in [(jnp.float64, "fp64"), (jnp.float32, "fp32")]:
        med, mn = alu_bound(dt)
        r[name] = mn
        print(f"    {name}: min {mn*1e3:8.3f} ms")
    print(f"    -> ALU fp64/fp32 ratio = {r['fp64']/r['fp32']:.1f}x")

    print("\n(b) Bandwidth-bound control (axpy, 16M elems):")
    r = {}
    for dt, name in [(jnp.float64, "fp64"), (jnp.float32, "fp32")]:
        med, mn, bw = bw_bound(dt)
        r[name] = mn
        print(f"    {name}: min {mn*1e3:8.3f} ms   achieved {bw:6.1f} GB/s")
    print(f"    -> BW fp64/fp32 ratio = {r['fp64']/r['fp32']:.2f}x")

    print("\n(c) wave_apply at 128^3 (N=2.097M):")
    r = {}
    for dt, name in [(jnp.float64, "fp64"), (jnp.float32, "fp32")]:
        try:
            b = build(5, 4, 4, 3, 100.0, dt)
            x = jax.random.normal(jax.random.PRNGKey(0), (b["N"],), dtype=dt)
            med, mn = timeit(b["apply"], x, n=20)
            r[name] = med
            print(f"    {name}: N={b['N']} median {med*1e3:8.3f} ms")
        except Exception as e:
            print(f"    {name}: FAILED {type(e).__name__}: {str(e)[:120]}")
    if len(r) == 2:
        print(f"    -> fp64/fp32 ratio at 128^3 = {r['fp64']/r['fp32']:.2f}x")
