"""R1 DECISIVE MEASUREMENT: is the MG-preconditioned wavelet operator
contrast-INDEPENDENT, or merely better than hybrid-Jacobi?

Uses the harness EXACT kappa (dense generalized eigenvalue) — the gold standard,
no CG-iteration noise. Small dense N so eigh is exact. Compares, across contrast:
  - hybrid-Jacobi (current production): kappa of  D^-1 A_wave D^-1
  - mg-arith  (geometric MG, linear P):  kappa_pre(A_wave, M^-1 = Wn^-1 MG Wn^-T)
  - mg-rap    (op-dependent Galerkin):   same, StencilMG

The whole R1 question is: does a curve stay FLAT vs contrast (contrast-independent,
the goal) or grow like contrast (hybrid-Jacobi, the problem)?

Also reports the M^-1 SPD check (CG needs an SPD preconditioner) and the
preconditioned kappa's sensitivity to coefficient KIND (smooth vs jump), because
op-dependent MG is supposed to shine specifically on jumps.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf")
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from harness import Problem, dense_of, kappa_pre, kappa_plain
from mg import MG
from stencil_mg import StencilMG


def kappa_row(dim, nl, nc, kind, contrast):
    p = Problem(nl, nc, dim, kind=kind, contrast=contrast, mass=1.0)
    N = p.N
    # dense A_wave (unscaled) and scaled Ahat
    A = dense_of(p.A_wave, N)
    D = p.D("hybrid")
    Ahat = dense_of(lambda v: p.A_wave(v / D) / D, N)
    k_hybrid = kappa_plain(Ahat)

    out = {"N": N, "hybrid": k_hybrid, "unprec": kappa_plain(A)}

    for name, MGclass in (("mg-arith", MG), ("mg-rap", StencilMG)):
        try:
            if MGclass is MG:
                m = MGclass(p.a, p.side, dim, p.h, p.mass, n_levels=nl, how="arith")
            else:
                m = MGclass(p.a, p.side, dim, p.h, p.mass, n_levels=nl)
            Minv = lambda v: p.wn_inv(m.apply(p.wn_inv_T(v)))
            Mi = dense_of(Minv, N)
            # SPD check of the (symmetrised) preconditioner
            Msym = 0.5 * (Mi + Mi.T)
            evM = np.linalg.eigvalsh(Msym)
            spd = bool(evM.min() > 1e-10 * evM.max())
            out[name] = kappa_pre(A, Mi)
            out[name + "_spd"] = spd
        except Exception as ex:
            out[name] = float("nan")
            out[name + "_err"] = repr(ex)[:80]
    return out


def main():
    dim, nl, nc = 2, 4, 2          # side=32, N=1024 — dense eigh fine
    contrasts = [1.0, 10.0, 1e2, 1e3, 1e4]
    for kind in ("smooth", "jump"):
        print(f"\n===== 2D side=32 N=1024, coeff kind={kind} — EXACT kappa vs contrast =====")
        print(f"{'contrast':>9} {'unprec':>11} {'hybrid-Jac':>11} {'mg-arith':>11} "
              f"{'mg-rap':>11}  {'arith_spd/rap_spd'}")
        for ct in contrasts:
            r = kappa_row(dim, nl, nc, kind, ct)
            print(f"{ct:>9.0e} {r['unprec']:>11.3e} {r['hybrid']:>11.3e} "
                  f"{r.get('mg-arith', float('nan')):>11.3e} "
                  f"{r.get('mg-rap', float('nan')):>11.3e}   "
                  f"{r.get('mg-arith_spd','?')}/{r.get('mg-rap_spd','?')}"
                  + (f"  ERR {r.get('mg-rap_err','')}" if 'mg-rap_err' in r else ''))


if __name__ == "__main__":
    main()
