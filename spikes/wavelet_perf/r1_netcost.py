"""R1 net-cost: wall-clock speedup + cost of M^-1 in matvec-equivalents.

Uses the jitted while-loop PCG (real early exit) from e2e.make_pcg.
Reports, per contrast: iterations, cost(M^-1)/cost(matvec), and NET wall-clock
ratio hybrid/MG.  Also times FULL setup+solve (e2e_full style) since in
production the coefficient is traced and the hierarchy is rebuilt per solve.
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf")
import numpy as np, jax, jax.numpy as jnp
from harness import Problem
from mg import MG
from stencil_mg import StencilMG
from e2e import make_pcg


def _time(fn, arg, reps=5):
    r = fn(arg); jax.block_until_ready(r)
    t0 = time.perf_counter()
    for _ in range(reps):
        r = fn(arg)
    jax.block_until_ready(r)
    return (time.perf_counter() - t0) / reps


def bench(dim, nl, nc, kind, contrasts, mgkind="harm", mass=1.0):
    print(f"\n### dim={dim} nl={nl} nc={nc} kind={kind} mg={mgkind}  N={(nc*2**nl)**dim}")
    print(f"{'ct':>7} | {'hyb_it':>6} {'mg_it':>6} | {'t_mv(ms)':>9} {'t_Minv(ms)':>10} {'Minv/mv':>7} |"
          f" {'hyb_solve':>9} {'mg_solve':>9} {'speedup':>7} | {'hyb_full':>8} {'mg_full':>8} {'full_su':>7}")
    for ct in contrasts:
        p = Problem(nl, nc, dim, kind=kind, contrast=ct, mass=mass)
        N = p.N
        f = jnp.asarray(np.random.default_rng(0).normal(size=N))
        b = p.wn_transpose(f)
        D = p.D("hybrid")

        # hybrid (scaled coords, no inner precond)
        Ahat = lambda w: p.A_wave(w / D) / D
        cg_h = make_pcg(Ahat, None, N)
        xh, kh, rh = cg_h(b / D); jax.block_until_ready(xh)

        # MG-pullback operator mode
        if mgkind == "rap":
            m = StencilMG(p.a, p.side, dim, p.h, p.mass, n_levels=nl)
        else:
            m = MG(p.a, p.side, dim, p.h, p.mass, n_levels=nl, how=mgkind)
        Minv = lambda v: p.wn_inv(m.apply(p.wn_inv_T(v)))
        cg_m = make_pcg(p.A_wave, Minv, N)
        xm, km, rm = cg_m(b); jax.block_until_ready(xm)

        # per-op cost
        t_mv = _time(jax.jit(p.A_wave), b)
        t_mi = _time(jax.jit(Minv), b)

        # solve wall-clock (hierarchy already built, jit warm)
        t_hs = _time(cg_h, b / D)
        t_ms = _time(cg_m, b)

        # FULL: setup(hierarchy from a) + solve, all counted, jitted per call
        def full_hybrid(a):
            pa = _swap(p, a, nl, nc, dim, mass, f)
            return make_pcg(lambda v: pa["A"](v/pa["D"])/pa["D"], None, N)(pa["b"]/pa["D"])[1]
        def full_mg(a):
            pa = _swap(p, a, nl, nc, dim, mass, f)
            if mgkind == "rap":
                mm = StencilMG(a, p.side, dim, p.h, mass, n_levels=nl)
            else:
                mm = MG(a, p.side, dim, p.h, mass, n_levels=nl, how=mgkind)
            Mi = lambda v: p.wn_inv(mm.apply(p.wn_inv_T(v)))
            return make_pcg(pa["A"], Mi, N)(pa["b"])[1]
        fh = jax.jit(full_hybrid); fm = jax.jit(full_mg)
        _ = fh(p.a); _ = fm(p.a); jax.block_until_ready(_)
        t_hf = _time(fh, p.a); t_mf = _time(fm, p.a)

        print(f"{ct:7.0f} | {int(kh):6d} {int(km):6d} | {t_mv*1e3:9.3f} {t_mi*1e3:10.3f}"
              f" {t_mi/t_mv:7.2f} | {t_hs*1e3:9.2f} {t_ms*1e3:9.2f} {t_hs/t_ms:7.2f} |"
              f" {t_hf*1e3:8.2f} {t_mf*1e3:8.2f} {t_hf/t_mf:7.2f}")


def _swap(p, a, nl, nc, dim, mass, f):
    from maddening.nodes.adaptive.wavelets.matrixfree import make_varcoeff_apply, wave_diagonal_fast
    from maddening.nodes.adaptive.wavelets.precond import diagonal_scaling
    ap = make_varcoeff_apply(a, p.side, dim, p.h, mass)
    A = lambda v: p.wn_transpose(ap(p.wn_apply(v)))
    d = wave_diagonal_fast(nl, nc, p.order, dim, p.norms, ap)
    return {"A": A, "b": p.wn_transpose(f), "D": diagonal_scaling(d, p.levels, "hybrid")}


if __name__ == "__main__":
    cs = [1, 10, 100, 1000, 10000]
    bench(2, 4, 2, "jump", cs, mgkind="harm")
    bench(2, 4, 2, "jump", cs, mgkind="rap")
    bench(3, 3, 2, "jump", cs, mgkind="harm")
