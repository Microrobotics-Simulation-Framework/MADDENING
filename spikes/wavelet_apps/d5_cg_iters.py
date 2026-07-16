"""D5 derisk — PCG iteration counts at chi <= 10^3 (matrix-free viability).

Under matrix-free, gather_solve's direct dense solve is replaced by CG. kappa ~
contrast (measurement 1), so the question is whether the iteration count at the
targeted chi<=10^3 stays affordable with hybrid-Jacobi alone, or whether a
contrast-robust preconditioner (R1) becomes a prerequisite for M16.

Measure: iterations for CG to reach rel. residual 1e-8 on the hybrid-Jacobi
symmetrically-scaled operator Ah = D^-1 A D^-1, 2D, contrast 1 -> 10^3.  The
masked (active-set) system's conditioning is bounded above by the full Ah
(Cauchy interlacing), so the full-operator count is a conservative ceiling.

Pass: <~100 iterations at chi=10^3.
On fail: promote R1 to prerequisite, STOP.
"""
from __future__ import annotations
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as spla
from maddening.nodes.adaptive.wavelets import operator as OP, precond as PC


def measure(side_levels=(4, 2), inclusion_r=0.15):
    nl, nc = side_levels
    side = nc * 2 ** nl
    N = side ** 2
    h = 1.0 / side
    c1 = np.arange(side) / side
    X, Y = np.meshgrid(c1, c1, indexing="ij")
    incl = (X - 0.5) ** 2 + (Y - 0.5) ** 2 < inclusion_r ** 2
    # RHS: a smooth Gaussian (well-posed, non-degenerate for all contrasts)
    f = jnp.asarray(np.exp(-(((X - 0.42) ** 2 + (Y - 0.5) ** 2) / 0.12 ** 2)).reshape(-1))

    print(f"2D side={side} N={N}, CG to rel-resid 1e-8 on hybrid-Jacobi Ah")
    print(f"{'contrast':>9} {'kappa(Ah)':>11} {'CG iters':>9} {'unprec iters':>13}")
    rows = []
    for contrast in [1.0, 10.0, 1e2, 1e3]:
        a = jnp.asarray(np.where(incl, contrast, 1.0))
        r = OP.assemble_wave_dense(nl, nc, 4, 2, a_grid=a, mass=1.0)
        A, Wn, lev = r["A_dense"], r["Wn"], r["levels"]
        D = PC.diagonal_scaling(jnp.diag(A), lev, "hybrid")
        Ah = np.asarray((A / D[:, None]) / D[None, :])
        bh = np.asarray(((h ** 2) * (Wn.T @ f)) / D)
        ev = np.abs(np.linalg.eigvalsh(Ah)); kappa = ev.max() / ev.min()

        def count(M):
            it = [0]
            spla.cg(M, bh, rtol=1e-8, maxiter=5000, callback=lambda xk: it.__setitem__(0, it[0] + 1))
            return it[0]

        ni = count(Ah)
        nu = count(np.asarray(A))
        rows.append((contrast, kappa, ni, nu))
        print(f"{contrast:>9.0e} {kappa:>11.3e} {ni:>9} {nu:>13}")
    return rows


if __name__ == "__main__":
    rows = measure()
    at_1e3 = [r for r in rows if r[0] == 1e3][0]
    ni = at_1e3[2]
    print()
    print(f"CG iters at chi=1e3: {ni}  ->",
          "PASS — matrix-free CG affordable, M16 proceeds" if ni <= 100
          else "FAIL — promote R1 (contrast-robust preconditioner) to prerequisite, STOP")
