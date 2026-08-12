"""End-to-end wavelet-space PCG: hybrid-Jacobi (current) vs MG-pullback (operator mode).

Measures iterations to 1e-8 AND wall clock, all jitted, on the real matrix-free path.
"""
from __future__ import annotations

import sys, time, functools
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf")

import jax, jax.numpy as jnp, numpy as np
from harness import Problem
from mg import MG


def make_pcg(apply, Minv, N, tol=1e-8, maxit=2000):
    """Jitted PCG returning (x, iters, relres). lax.while_loop -> real early exit."""
    def run(b):
        bn = jnp.linalg.norm(b)
        def body(s):
            x, r, p, rz, k, _ = s
            Ap = apply(p)
            alpha = rz / jnp.vdot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            z = Minv(r) if Minv is not None else r
            rz2 = jnp.vdot(r, z)
            p = z + (rz2 / rz) * p
            return (x, r, p, rz2, k + 1, jnp.linalg.norm(r) / bn)
        def cond(s):
            return (s[5] > tol) & (s[4] < maxit)
        x0 = jnp.zeros_like(b); r0 = b
        z0 = Minv(r0) if Minv is not None else r0
        s = (x0, r0, z0, jnp.vdot(r0, z0), 0, jnp.linalg.norm(r0) / bn)
        x, r, p, rz, k, rn = jax.lax.while_loop(cond, body, s)
        return x, k, rn
    return jax.jit(run)


def bench(dim, nl, nc, kind, contrast, mass=1.0, reps=3):
    p = Problem(nl, nc, dim, kind=kind, contrast=contrast, mass=mass)
    N = p.N
    rng = np.random.default_rng(0)
    f = jnp.asarray(rng.normal(size=N))
    out = {}

    # ---- baseline: hybrid-Jacobi, scaled coordinates (current production) ----
    D = p.D("hybrid")
    Ahat = lambda v: p.A_wave(v / D) / D
    bhat = p.wn_transpose(f) / D
    cg0 = make_pcg(Ahat, None, N)
    x, k, rn = cg0(bhat); x.block_until_ready()
    t = min(_time(cg0, bhat) for _ in range(reps))
    out["hybrid"] = (int(k), float(rn), t)

    # ---- operator mode: identity coords, MG pullback inner_precond ----
    m = MG(p.a, p.side, dim, p.h, p.mass, n_levels=nl, how="arith")
    Minv = lambda v: p.wn_inv(m.apply(p.wn_inv_T(v)))
    b = p.wn_transpose(f)
    cg1 = make_pcg(p.A_wave, Minv, N)
    x1, k1, rn1 = cg1(b); x1.block_until_ready()
    t1 = min(_time(cg1, b) for _ in range(reps))
    out["mg-pullback"] = (int(k1), float(rn1), t1)

    # cross-check both solve the same system
    u0 = p.wn_apply(x / D); u1 = p.wn_apply(x1)
    out["agree"] = float(jnp.linalg.norm(u0 - u1) / jnp.linalg.norm(u1))

    # cost of one matvec / one M^-1 application
    out["t_matvec"] = _time(jax.jit(p.A_wave), b)
    out["t_minv"] = _time(jax.jit(Minv), b)
    return p, out


def _time(fn, arg):
    r = fn(arg)
    jax.block_until_ready(r)
    t0 = time.perf_counter()
    for _ in range(3):
        r = fn(arg)
    jax.block_until_ready(r)
    return (time.perf_counter() - t0) / 3
