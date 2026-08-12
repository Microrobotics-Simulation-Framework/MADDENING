"""Honest end-to-end: EVERYTHING inside one jit, including preconditioner setup.

In production the coefficient field `a` is traced (theta -> a -> A), so the
multigrid hierarchy (coefficient coarsening, or the RAP probing) is rebuilt on
every solve and MUST be counted.  This benchmark jits  a -> (build M) -> PCG.
"""
from __future__ import annotations

import sys, time
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf")

import jax, jax.numpy as jnp, numpy as np
from harness import Problem
from mg import MG
from stencil_mg import StencilMG
from e2e import make_pcg


def timeit(fn, *args, reps=3):
    r = fn(*args); jax.block_until_ready(r)
    t0 = time.perf_counter()
    for _ in range(reps):
        r = fn(*args)
    jax.block_until_ready(r)
    return (time.perf_counter() - t0) / reps


def build(dim, nl, nc, kind, ct, mass=1.0, tol=1e-8, maxit=3000):
    p = Problem(nl, nc, dim, kind=kind, contrast=ct, mass=mass)
    N = p.N
    f = jnp.asarray(np.random.default_rng(0).normal(size=N))

    def hybrid(a):
        pa = _swap(p, a)
        D = pa["D"]
        cg = make_pcg(lambda v: pa["A"](v / D) / D, None, N, tol, maxit)
        return cg(pa["b"] / D)

    def mg_arith(a):
        pa = _swap(p, a)
        m = MG(a, p.side, dim, p.h, mass, n_levels=nl, how="arith")
        Minv = lambda v: p.wn_inv(m.apply(p.wn_inv_T(v)))
        return make_pcg(pa["A"], Minv, N, tol, maxit)(pa["b"])

    def mg_rap(a):
        pa = _swap(p, a)
        m = StencilMG(a, p.side, dim, p.h, mass, n_levels=nl)
        Minv = lambda v: p.wn_inv(m.apply(p.wn_inv_T(v)))
        return make_pcg(pa["A"], Minv, N, tol, maxit)(pa["b"])

    def _swap(p, a):
        ap = __import__("maddening.nodes.adaptive.wavelets.matrixfree",
                        fromlist=["x"]).make_varcoeff_apply(a, p.side, dim, p.h, mass)
        A = lambda v: p.wn_transpose(ap(p.wn_apply(v)))
        from maddening.nodes.adaptive.wavelets.matrixfree import wave_diagonal_fast
        from maddening.nodes.adaptive.wavelets.precond import diagonal_scaling
        d = wave_diagonal_fast(nl, nc, p.order, dim, p.norms, ap)
        return {"A": A, "b": p.wn_transpose(f),
                "D": diagonal_scaling(d, p.levels, "hybrid")}

    return p, {k: jax.jit(v) for k, v in
               [("hybrid", hybrid), ("mg-arith", mg_arith), ("mg-rap", mg_rap)]}
