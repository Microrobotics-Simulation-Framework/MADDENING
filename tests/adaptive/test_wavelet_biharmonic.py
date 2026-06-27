"""M5 (part 1) — biharmonic operator + CDD on the biharmonic residual.

The stream-function formulation of the driven cavity solves a biharmonic (t=2,
H²-elliptic) problem.  FINDINGS §5 validated the *conditioning* (t=2 DK / Jacobi)
but CDD selection had only ever been tested on Laplacian (2nd-order) residuals
(spike Limitation 12).  Here we validate CDD on the biharmonic (4th-order)
residual, whose spatial structure differs.  conftest provides float64.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from maddening.nodes.adaptive.wavelets import cdd as CDD
from maddening.nodes.adaptive.wavelets import operator as OP
from maddening.nodes.adaptive.wavelets import precond as PC


def _kappa(A):
    ev = np.linalg.eigvalsh(np.asarray(A))
    return ev[-1] / ev[0]


def test_biharmonic_t2_and_jacobi_conditioning():
    """DD-4 biharmonic: t=2 DK κ≈8.6e3, hybrid-Jacobi κ≈1.2e3 (FINDINGS §5)."""
    res = OP.assemble_wave_operator(7, 2, order=4, dim=1, mass=1.0,
                                    kind="biharmonic")
    A, levels = res["A_dense"], res["levels"]
    k_dk = _kappa((A / (D := PC.diagonal_scaling(jnp.diag(A), levels, "dk", t=2.0))[:, None]) / D[None, :])
    Dj = PC.diagonal_scaling(jnp.diag(A), levels, "hybrid")
    k_j = _kappa((A / Dj[:, None]) / Dj[None, :])
    assert 5e3 < k_dk < 1.5e4, f"t=2 DK κ={k_dk}"
    assert 5e2 < k_j < 2.5e3, f"Jacobi κ={k_j}"
    assert k_j < k_dk            # Jacobi better, as the spike found


def test_biharmonic_needs_order_4():
    """DD-2 (piecewise linear, not in H²) is ill-conditioned for the biharmonic;
    DD-4 is O(1)-ish — confirms the H²-order requirement (FINDINGS §5)."""
    def jac_kappa(order, nl):
        res = OP.assemble_wave_operator(nl, 2, order=order, dim=1, mass=1.0,
                                        kind="biharmonic")
        A, lv = res["A_dense"], res["levels"]
        D = PC.diagonal_scaling(jnp.diag(A), lv, "hybrid")
        return _kappa((A / D[:, None]) / D[None, :])
    # DD-2 grows steeply with N; DD-4 stays bounded
    k2_lo, k2_hi = jac_kappa(2, 5), jac_kappa(2, 7)
    k4_lo, k4_hi = jac_kappa(4, 5), jac_kappa(4, 7)
    assert k2_hi / k2_lo > 3.0, "DD-2 biharmonic κ should grow fast with N"
    assert k4_hi / k4_lo < 2.0, "DD-4 biharmonic κ should be ~bounded"


def test_biharmonic_mms_converges():
    """Manufactured biharmonic: u=cos(2πx), (Δ²+1)u = ((2π)^4+1)u; converge."""
    errs = []
    for nl in (5, 6):
        res = OP.assemble_wave_operator(nl, 2, order=4, dim=1, mass=1.0,
                                        kind="biharmonic")
        A, Wn, side, h = res["A_dense"], res["Wn"], res["side"], res["h"]
        x = np.arange(side) / side
        u_ex = np.cos(2 * np.pi * x)
        f = ((2 * np.pi) ** 4 + 1.0) * u_ex
        b = h * (Wn.T @ jnp.asarray(f))
        u_h = np.asarray(Wn @ jnp.linalg.solve(A, b))
        errs.append(np.linalg.norm(u_h - u_ex) / np.linalg.norm(u_ex))
    assert errs[-1] < 5e-3, f"biharmonic MMS relL2={errs[-1]}"
    assert errs[0] > errs[-1]          # converging


def test_cdd_on_biharmonic_residual():
    """CDD's residual marking works on the 4th-order biharmonic residual
    (Limitation 12): converges in ≤25 outer iters at < N/4, J_err small."""
    res = OP.assemble_wave_operator(7, 2, order=4, dim=1, mass=1.0,
                                    kind="biharmonic")
    A, Wn, levels, side, h, N = (res["A_dense"], res["Wn"], res["levels"],
                                 res["side"], res["h"], res["N"])
    D = PC.diagonal_scaling(jnp.diag(A), levels, "hybrid")
    Ah = (A / D[:, None]) / D[None, :]
    x = np.arange(side) / side
    u_ex = np.cos(2 * np.pi * x)
    f = ((2 * np.pi) ** 4 + 1.0) * u_ex
    bh = (h * (Wn.T @ jnp.asarray(f))) / D
    coarse = jnp.asarray(levels) == int(np.asarray(levels).min())
    K = N // 8

    def solve_masked(mask, rhs):
        return OP.gather_solve(Ah, mask, rhs, K)

    # count outer iterations
    mask = coarse
    c = solve_masked(mask, bh)
    nout = 0
    for _ in range(CDD.MAX_OUTER):
        if int(jnp.sum(mask)) >= K:
            break
        r = bh - Ah @ c
        mask = CDD._doerfler_grow(mask, r, 0.5, K)
        c = solve_masked(mask, bh)
        nout += 1
    u_h = np.asarray(Wn @ (c / D))
    j_err = np.linalg.norm(u_h - u_ex) / np.linalg.norm(u_ex)
    assert nout <= 25, f"CDD outer iters {nout}"
    assert int(jnp.sum(mask)) <= N // 4
    assert j_err < 1e-3, f"biharmonic CDD J_err={j_err}"
