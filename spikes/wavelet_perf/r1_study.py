"""R1 core study: iteration count / kappa vs contrast, hybrid vs MG-pullback variants.

Runs the matrix-free PCG (harness.pcg_count) on A_wave with different inner
preconditioners, all wrapped into wavelet-coefficient space via the exact
Wn^{-1}/Wn^{-T} maps. Reports iterations-to-1e-8 vs contrast.

For small N also reports exact kappa via dense eig.
"""
from __future__ import annotations
import sys, time
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf")
import numpy as np, jax, jax.numpy as jnp
from harness import Problem, dense_of, kappa_plain, kappa_pre, pcg_count
from mg import MG
from mg_opdep import MGOpDep
from stencil_mg import StencilMG


def make_minv(variant, p, nl):
    if variant == "hybrid":
        return None  # handled specially (two-sided scaling)
    if variant == "mg-arith":
        m = MG(p.a, p.side, p.dim, p.h, p.mass, n_levels=nl, how="arith")
    elif variant == "mg-harm":
        m = MG(p.a, p.side, p.dim, p.h, p.mass, n_levels=nl, how="harm")
    elif variant == "mg-opdep":
        m = MGOpDep(p.a, p.side, p.dim, p.h, p.mass, n_levels=nl, how="harm")
    elif variant == "mg-rap":
        m = StencilMG(p.a, p.side, p.dim, p.h, p.mass, n_levels=nl)
    else:
        raise ValueError(variant)
    return lambda v: p.wn_inv(m.apply(p.wn_inv_T(v)))


def study(dim, nl, nc, kind, contrasts, variants, dense=False, mass=1.0, tol=1e-8):
    rng = np.random.default_rng(0)
    print(f"\n### dim={dim} nl={nl} nc={nc} kind={kind} mass={mass}  N={ (nc*2**nl)**dim }")
    header = "contrast   " + "".join(f"{v:>16}" for v in variants)
    print(header)
    results = {}
    for ct in contrasts:
        p = Problem(nl, nc, dim, kind=kind, contrast=ct, mass=mass)
        N = p.N
        f = jnp.asarray(rng.normal(size=N))
        b = p.wn_transpose(f)
        D = p.D("hybrid")
        row = {}
        cells = []
        for v in variants:
            if v == "hybrid":
                Ahat = lambda w: p.A_wave(w / D) / D
                bhat = b / D
                x, it, hist = pcg_count(Ahat, bhat, None, tol=tol, maxit=6000)
                if dense:
                    Aw = dense_of(p.A_wave, N)
                    Dn = np.asarray(D)
                    Ah = (np.asarray(Aw)/Dn[:,None])/Dn[None,:]
                    kap = kappa_plain(jnp.asarray(Ah))
                else:
                    kap = None
            else:
                Minv = make_minv(v, p, nl)
                x, it, hist = pcg_count(p.A_wave, b, Minv, tol=tol, maxit=6000)
                if dense:
                    Aw = dense_of(p.A_wave, N)
                    Mid = dense_of(Minv, N)
                    kap = kappa_pre(Aw, Mid)
                else:
                    kap = None
            row[v] = (it, kap, hist[-1])
            cells.append(f"{it:4d}" + (f"/{kap:8.1f}" if kap is not None else "        "))
        print(f"{ct:8.0f}   " + "".join(f"{c:>16}" for c in cells))
        results[ct] = row
    return results


if __name__ == "__main__":
    contrasts = [1, 10, 100, 1000, 10000]
    # 1D dense: exact kappa
    study(1, 4, 2, "jump", contrasts, ["hybrid", "mg-arith", "mg-harm", "mg-rap"], dense=True)
    # 2D matrix-free iteration counts
    study(2, 4, 2, "jump", contrasts, ["hybrid", "mg-arith", "mg-harm", "mg-opdep", "mg-rap"])
    study(2, 4, 2, "checker", contrasts, ["hybrid", "mg-arith", "mg-harm", "mg-rap"])
