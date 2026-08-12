"""R1 iteration counts vs contrast, jitted PCG (fast, exact early-exit counts).

hybrid (current) vs mg-harm (rediscretised harmonic) vs mg-rap (op-dep P + exact
Galerkin RAP by colored probing).  2D and 3D.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf")
import numpy as np, jax, jax.numpy as jnp
from harness import Problem
from mg import MG
from stencil_mg import StencilMG
from e2e import make_pcg


def iters(dim, nl, nc, kind, contrasts, mass=1.0, maxit=6000, tol=1e-8):
    print(f"\n### dim={dim} nl={nl} nc={nc} kind={kind} mass={mass}  N={(nc*2**nl)**dim}", flush=True)
    print(f"{'contrast':>8} {'hybrid':>8} {'mg-harm':>8} {'mg-rap':>8}   (agree hyb,rap vs mg-harm)", flush=True)
    for ct in contrasts:
        p = Problem(nl, nc, dim, kind=kind, contrast=ct, mass=mass)
        N = p.N
        f = jnp.asarray(np.random.default_rng(0).normal(size=N))
        b = p.wn_transpose(f)
        D = p.D("hybrid")
        # hybrid
        cg_h = make_pcg(lambda w: p.A_wave(w / D) / D, None, N, tol, maxit)
        xh, kh, rh = cg_h(b / D)
        uh = p.wn_apply(xh / D)
        # mg-harm
        mh = MG(p.a, p.side, dim, p.h, p.mass, n_levels=nl, how="harm")
        Mih = lambda v: p.wn_inv(mh.apply(p.wn_inv_T(v)))
        cg_mh = make_pcg(p.A_wave, Mih, N, tol, maxit)
        xmh, kmh, rmh = cg_mh(b)
        umh = p.wn_apply(xmh)
        # mg-rap
        mr = StencilMG(p.a, p.side, dim, p.h, p.mass, n_levels=nl)
        Mir = lambda v: p.wn_inv(mr.apply(p.wn_inv_T(v)))
        cg_mr = make_pcg(p.A_wave, Mir, N, tol, maxit)
        xmr, kmr, rmr = cg_mr(b)
        umr = p.wn_apply(xmr)
        jax.block_until_ready((uh, umh, umr))
        ag_h = float(jnp.linalg.norm(uh - umh) / jnp.linalg.norm(umh))
        ag_r = float(jnp.linalg.norm(umr - umh) / jnp.linalg.norm(umh))
        print(f"{ct:8.0f} {int(kh):8d} {int(kmh):8d} {int(kmr):8d}   ({ag_h:.1e}, {ag_r:.1e})"
              f"  res hyb/harm/rap {float(rh):.0e}/{float(rmh):.0e}/{float(rmr):.0e}", flush=True)


if __name__ == "__main__":
    cs = [1, 10, 100, 1000, 10000]
    iters(2, 4, 2, "jump", cs)
    iters(2, 4, 2, "checker", cs)
    iters(2, 5, 2, "jump", cs)          # N=4096, finer -> more levels
    iters(3, 3, 2, "jump", cs)          # N=32^3=32768
    iters(3, 4, 2, "jump", cs)          # N=64^3, the documented size
