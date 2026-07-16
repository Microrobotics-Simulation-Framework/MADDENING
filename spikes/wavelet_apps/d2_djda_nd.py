"""D2 derisk — dJ/da w.r.t. the coefficient field in 2D and 3D (dense path).

All existing varcoeff coverage is 1D only (test_wavelet_basis.py:155-168). App 1
(magnetics) rests entirely on dJ/da flowing through operator assembly in 2D/3D.
Validate jax.grad vs central FD before any architecture work.

Objective mirrors the 1D test: J(a) = phi(x_sensor) for a fixed Gaussian source,
solving (-div(a grad) + m) u = f in the L2-normalised wavelet basis with the
preconditioner frozen at a0 (gradient-irrelevant at convergence).

Pass: max relative error < 1e-3 vs central FD at a few probed voxels, 2D and 3D.
"""
from __future__ import annotations
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from maddening.nodes.adaptive.wavelets import operator as OP, precond as PC


def build(dim, nl, nc):
    side = nc * 2 ** nl
    h = 1.0 / side
    N = side ** dim
    c1 = np.arange(side) / side
    mesh = np.meshgrid(*([c1] * dim), indexing="ij")
    # fixed Gaussian source, off-centre
    r2 = (mesh[0] - 0.42) ** 2 + sum((mesh[d] - 0.5) ** 2 for d in range(1, dim))
    f = jnp.asarray(np.exp(-(r2) / 0.12 ** 2).reshape(-1))
    a0 = jnp.ones(N)
    r0 = OP.assemble_wave_dense(nl, nc, 4, dim, a_grid=a0.reshape((side,) * dim), mass=1.0)
    D = PC.diagonal_scaling(jnp.diag(r0["A_dense"]), r0["levels"], "hybrid")
    sidx = int(np.argmin(np.abs(c1 - 0.30))) * (side ** (dim - 1))  # a corner-ish sensor
    Wn = r0["Wn"]
    srow = Wn[sidx]

    def J(a_flat):
        a = a_flat.reshape((side,) * dim)
        r = OP.assemble_wave_dense(nl, nc, 4, dim, a_grid=a, mass=1.0)
        Aa, Wn = r["A_dense"], r["Wn"]
        Ah = (Aa / D[:, None]) / D[None, :]
        b = ((h ** dim) * (Wn.T @ f)) / D
        c = jnp.linalg.solve(Ah, b)
        return (srow / D) @ c

    return J, N, side


def run(dim, nl, nc, n_probe=6):
    J, N, side = build(dim, nl, nc)
    a0 = jnp.ones(N)
    g = jax.grad(J)(a0)
    rng = np.random.default_rng(0)
    probes = rng.choice(N, size=min(n_probe, N), replace=False)
    # e=1e-4, not 1e-6: at low-sensitivity voxels (|dJ/da|~1e-11) a 1e-6 central
    # step is roundoff-dominated (subtracting near-equal J).  See d2_diag.py /
    # FINDINGS_D2.md — the gradient is correct to ~2e-5 vs a 4th-order stencil.
    e = 1e-4
    errs = []
    for k in probes:
        fd = float((J(a0.at[k].add(e)) - J(a0.at[k].add(-e))) / (2 * e))
        rel = abs(float(g[k]) - fd) / (abs(fd) + 1e-30)
        errs.append(rel)
    return side, N, max(errs), np.mean(errs)


if __name__ == "__main__":
    cases = [(2, 3, 2), (2, 4, 2), (3, 3, 1)]   # 16^2, 32^2, 8^3
    print(f"{'dim':>3} {'side':>5} {'N':>6} {'max relerr':>12} {'mean relerr':>12}")
    all_pass = True
    for dim, nl, nc in cases:
        side, N, mx, mn = run(dim, nl, nc)
        ok = mx < 1e-3
        all_pass &= ok
        print(f"{dim:>3} {side:>5} {N:>6} {mx:>12.3e} {mn:>12.3e}  {'PASS' if ok else 'FAIL'}")
    print()
    print("Verdict:", "PASS — dJ/da flows through assembly in 2D/3D"
          if all_pass else "FAIL — STOP (engine varcoeff assembly wrong above 1D)")
