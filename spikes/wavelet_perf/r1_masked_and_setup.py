"""R1 CRITIQUE GATES before implementing:

GATE A (setup cost): the MG hierarchy is rebuilt every solve in production (a is
traced). Does counting the build inside the timed/jitted solve eat the win?

GATE B (MASKED operator): production solves the CDD-masked active-set system with a
MASKED preconditioner. (M^-1)_LambdaLambda != (A_LambdaLambda)^-1 in general, so the
masked MG might not precondition the masked operator. Measure CG iterations on the
REAL masked system: masked-hybrid vs masked-MG. This is make-or-break.
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf")
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from harness import Problem, pcg_count
from mg import MG
from maddening.nodes.adaptive.wavelets import cdd as CDD
from maddening.nodes.adaptive.wavelets import matrixfree as MF


def real_mask(p, K=None):
    """A REAL CDD-selected active set on this problem (hybrid-scaled, as production)."""
    N = p.N
    K = K or max(8, N // 16)
    D = p.D("hybrid")
    Ahat = lambda v: p.A_wave(v / D) / D
    f = jnp.asarray(np.random.default_rng(0).normal(size=N))
    b = p.wn_transpose(f) / D
    lev = np.asarray(p.levels); coarse = jnp.asarray(lev == lev.min())
    sm = lambda m, r: MF.masked_cg_solve(Ahat, m, r, rtol=1e-8, atol=1e-10)
    mask, _, _ = CDD.cdd_select(Ahat, sm, b, coarse, K)
    return np.asarray(mask), D, b


def masked(fn, mask):
    m = jnp.asarray(mask)
    def op(v):
        vm = jnp.where(m, v, 0.0)
        return jnp.where(m, fn(vm), v)
    return op


def gate_b(dim, nl, nc, kind, contrasts):
    print(f"\n=== GATE B: MASKED active-set system, {nc*2**nl}^{dim} {kind} ===")
    print(f"{'contrast':>9} | {'|Lambda|':>7} | {'masked-hybrid its':>17} | "
          f"{'masked-MG its':>13} | {'iter ratio':>10}")
    for ct in contrasts:
        p = Problem(nl, nc, dim, kind=kind, contrast=ct, mass=1.0)
        mask, D, b = real_mask(p)
        # masked hybrid: scaled operator, no inner precond
        Ahat = lambda v: p.A_wave(v / D) / D
        A_eff = masked(Ahat, mask)
        b_eff = jnp.where(jnp.asarray(mask), b, 0.0)
        _, kh, _ = pcg_count(A_eff, b_eff, Minv=None, tol=1e-8, maxit=5000)
        # masked MG: identity coords, masked MG-pullback inner precond
        m = MG(p.a, p.side, dim, p.h, p.mass, n_levels=nl, how="arith")
        Minv_full = lambda v: p.wn_inv(m.apply(p.wn_inv_T(v)))
        A_eff2 = masked(p.A_wave, mask)
        Minv_masked = masked(Minv_full, mask)
        b2 = jnp.where(jnp.asarray(mask), p.wn_transpose(jnp.asarray(
            np.random.default_rng(0).normal(size=p.N))), 0.0)
        _, km, _ = pcg_count(A_eff2, b2, Minv=Minv_masked, tol=1e-8, maxit=5000)
        print(f"{ct:>9.0e} | {int(mask.sum()):>7d} | {kh:>17d} | {km:>13d} | "
              f"{(kh/max(km,1)):>9.1f}x")


def gate_a(dim, nl, nc, kind, contrasts):
    """Setup-included: time (build MG hierarchy + solve) fully jitted vs hybrid."""
    from e2e import make_pcg
    print(f"\n=== GATE A: setup-INCLUDED jitted solve, {nc*2**nl}^{dim} {kind} ===")
    print(f"{'contrast':>9} | {'hybrid wall':>11} | {'MG(+build) wall':>15} | {'net x':>7}")
    for ct in contrasts:
        p = Problem(nl, nc, dim, kind=kind, contrast=ct, mass=1.0)
        N = p.N
        f = jnp.asarray(np.random.default_rng(0).normal(size=N))

        def hybrid(a):
            ap = MF.make_varcoeff_apply(a, p.side, dim, p.h, p.mass)
            A = lambda v: p.wn_transpose(ap(p.wn_apply(v)))
            d = MF.wave_diagonal_fast(nl, nc, p.order, dim, p.norms, ap)
            from maddening.nodes.adaptive.wavelets.precond import diagonal_scaling
            D = diagonal_scaling(d, p.levels, "hybrid")
            return make_pcg(lambda v: A(v / D) / D, None, N, 1e-8, 5000)(p.wn_transpose(f) / D)

        def mg_full(a):
            ap = MF.make_varcoeff_apply(a, p.side, dim, p.h, p.mass)
            A = lambda v: p.wn_transpose(ap(p.wn_apply(v)))
            mm = MG(a, p.side, dim, p.h, p.mass, n_levels=nl, how="arith")  # BUILD counted
            Minv = lambda v: p.wn_inv(mm.apply(p.wn_inv_T(v)))
            return make_pcg(A, Minv, N, 1e-8, 5000)(p.wn_transpose(f))

        jh, jm = jax.jit(hybrid), jax.jit(mg_full)
        for fn in (jh, jm):
            jax.block_until_ready(fn(p.a))
        def t(fn):
            t0 = time.perf_counter()
            for _ in range(3):
                r = fn(p.a)
            jax.block_until_ready(r); return (time.perf_counter() - t0) / 3
        th, tm = t(jh), t(jm)
        print(f"{ct:>9.0e} | {th*1e3:>9.1f}ms | {tm*1e3:>13.1f}ms | {th/tm:>6.2f}x")


if __name__ == "__main__":
    gate_b(3, 4, 2, "smooth", [1.0, 1e2, 1e3])
    gate_b(2, 5, 2, "jump", [1e2, 1e3])
    gate_a(3, 4, 2, "smooth", [1.0, 1e2, 1e3])
