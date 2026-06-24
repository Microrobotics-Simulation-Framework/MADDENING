"""M0 tests — production DD wavelet engine (transform / operator / precond / cdd).

Validates the numerical core that ``WaveletAdaptiveNode`` (M1+) builds on,
against the derisking-spike reference numbers (FINDINGS continuation Inv 1/2).
The ``tests/adaptive/conftest.py`` autouse fixture provides float64.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.solver_utils import ift_linear_solve
from maddening.nodes.adaptive.wavelets import cdd as CDD
from maddening.nodes.adaptive.wavelets import operator as OP
from maddening.nodes.adaptive.wavelets import precond as PC
from maddening.nodes.adaptive.wavelets import transform as T


# ----------------------------------------------------------------------
# transform.py
# ----------------------------------------------------------------------

@pytest.mark.parametrize("dim,nl,nc", [(1, 7, 2), (2, 4, 2), (3, 3, 1)])
def test_roundtrip_analysis_synthesis(dim, nl, nc):
    """analysis ∘ synthesis is the identity to machine precision."""
    synth = {1: T.synthesis_1d, 2: T.synthesis_2d, 3: T.synthesis_3d}[dim]
    ana = {1: T.analysis_1d, 2: T.analysis_2d, 3: T.analysis_3d}[dim]
    N = T.n_dofs(nl, nc, dim)
    v = jnp.asarray(np.random.default_rng(0).standard_normal(N))
    back = synth(ana(v, nl, nc, 4), nl, nc, 4)
    assert float(jnp.linalg.norm(back - v) / jnp.linalg.norm(v)) < 1e-12


def test_synthesis_matrix_consistency():
    """The materialised W is consistent with the matrix-free synthesis."""
    W = T.synthesis_matrix(5, 2, 4, dim=1)
    N = W.shape[0]
    j = 7
    ej = jnp.zeros(N).at[j].set(1.0)
    assert float(jnp.linalg.norm(T.synthesis_1d(ej, 5, 2, 4) - W[:, j])) < 1e-12


def test_dd4_midpoint_order():
    """DD-4 midpoint prediction is 4th-order: error drops ~16x per doubling."""
    errs = []
    for n in (64, 128, 256):
        x = np.arange(n) / n
        f = jnp.asarray(np.sin(2 * np.pi * x))
        pred = T._predict_axis(f, 0, 4)
        true = jnp.asarray(np.sin(2 * np.pi * (x + 0.5 / n)))
        errs.append(float(jnp.max(jnp.abs(pred - true))))
    # each refinement should cut the error by ~2^4=16 (allow margin)
    assert errs[0] / errs[1] > 10 and errs[1] / errs[2] > 10


def test_synthesis_jit_compiles():
    f = jax.jit(lambda c: T.synthesis_2d(c, 4, 2, 4))
    out = f(jnp.zeros(T.n_dofs(4, 2, 2))).block_until_ready()
    assert out.shape == (T.n_dofs(4, 2, 2),)


# ----------------------------------------------------------------------
# operator.py + precond.py — condition number matches the spike
# ----------------------------------------------------------------------

def _kappa(A):
    ev = np.linalg.eigvalsh(np.asarray(A))
    return ev[-1] / ev[0]


@pytest.mark.parametrize("dim,nl,nc,kappa_lo,kappa_hi", [
    (1, 7, 2, 18.0, 23.0),    # FINDINGS Inv 1: 1D hybrid ≈ 20.4
    (2, 4, 2, 34.0, 42.0),    # FINDINGS Inv 1: 2D hybrid ≈ 37.7
])
def test_hybrid_jacobi_kappa_matches_spike(dim, nl, nc, kappa_lo, kappa_hi):
    res = OP.assemble_wave_operator(nl, nc, order=4, dim=dim, mass=1.0)
    A, levels = res["A_dense"], res["levels"]
    D_h = PC.diagonal_scaling(jnp.diag(A), levels, "hybrid")
    D_f = PC.diagonal_scaling(jnp.diag(A), levels, "full")
    k_h = _kappa((A / D_h[:, None]) / D_h[None, :])
    k_f = _kappa((A / D_f[:, None]) / D_f[None, :])
    assert kappa_lo < k_h < kappa_hi, f"hybrid κ={k_h}"
    # hybrid ≡ full to 4 sig figs (FINDINGS Inv 1 headline)
    assert abs(k_h - k_f) / k_f < 1e-3


def test_bcoo_operator_assembled():
    """A_wave is assembled as a sparse BCOO (not dense)."""
    import jax.experimental.sparse as jsparse
    res = OP.assemble_wave_operator(4, 2, order=4, dim=2, mass=1.0)
    assert isinstance(res["A_bcoo"], jsparse.BCOO)
    nnz = int(res["A_bcoo"].nse)
    assert nnz < res["N"] ** 2  # genuinely sparse


# ----------------------------------------------------------------------
# Differentiability — through the solve (w.r.t. θ) and through assembly (w.r.t. a)
# ----------------------------------------------------------------------

def _setup_1d(nl=6, nc=2):
    res = OP.assemble_wave_operator(nl, nc, order=4, dim=1, mass=1.0)
    A, Wn, levels = res["A_dense"], res["Wn"], res["levels"]
    side, h, N = res["side"], res["h"], res["N"]
    D = PC.diagonal_scaling(jnp.diag(A), levels, "hybrid")
    Ah = (A / D[:, None]) / D[None, :]
    x = np.arange(side) / side
    sidx = int(np.argmin(np.abs(x - 0.30)))
    return dict(Ah=Ah, Wn=Wn, levels=levels, side=side, h=h, N=N, x=x,
                sidx=sidx, D=D)


def test_grad_through_cdd_solve_matches_fd():
    """jax.grad of a sensor functional through CDD + frozen solve = FD."""
    s = _setup_1d()
    Ah, Wn, D, x, h, N, sidx = (s["Ah"], s["Wn"], s["D"], s["x"], s["h"],
                                s["N"], s["sidx"])
    levels = np.asarray(s["levels"])
    coarse = jnp.asarray(levels == levels.min())
    K = N // 16

    def solve_masked(mask, rhs):
        op = OP.make_masked_operator(Ah, mask)
        return ift_linear_solve(op, jnp.where(mask, rhs, 0.0),
                                solver="cg", rtol=1e-10, atol=1e-12)

    def J(theta):
        f = jnp.exp(-((jnp.asarray(x) - theta) / 0.06) ** 2)
        b = (h * (Wn.T @ f)) / D
        _, c = CDD.cdd_select(lambda v: Ah @ v, solve_masked, b, coarse, K)
        return (Wn[sidx] / D) @ c

    th = jnp.asarray(0.42)
    g = float(jax.grad(J)(th))
    e = 1e-5
    fd = float((J(th + e) - J(th - e)) / (2 * e))
    assert abs(g - fd) / (abs(fd) + 1e-30) < 1e-5


def test_grad_through_coefficient_field_matches_fd():
    """jax.grad w.r.t. the coefficient field a(x) flows through operator
    assembly (Amendment 1).  Preconditioner held fixed (solver-only)."""
    nl, nc = 6, 2
    res0 = OP.assemble_wave_operator(nl, nc, order=4, dim=1, mass=1.0)
    levels, side, h = res0["levels"], res0["side"], res0["h"]
    x = np.arange(side) / side
    sidx = int(np.argmin(np.abs(x - 0.30)))
    a0 = jnp.asarray(1.0 + 0.5 * np.sin(2 * np.pi * x))
    D = PC.diagonal_scaling(
        jnp.diag(OP.assemble_wave_operator(nl, nc, 4, 1, a_grid=a0)["A_dense"]),
        levels, "hybrid")

    def J_of_a(a_grid):
        r = OP.assemble_wave_operator(nl, nc, 4, 1, a_grid=a_grid, mass=1.0)
        Aa, Wn = r["A_dense"], r["Wn"]
        Aha = (Aa / D[:, None]) / D[None, :]
        f = jnp.exp(-((jnp.asarray(x) - 0.42) / 0.06) ** 2)
        b = (h * (Wn.T @ f)) / D
        c = jnp.linalg.solve(Aha, b)
        return (Wn[sidx] / D) @ c

    ga = jax.grad(J_of_a)(a0)
    k = side // 2
    e = 1e-6
    fd = float((J_of_a(a0.at[k].add(e)) - J_of_a(a0.at[k].add(-e))) / (2 * e))
    assert abs(float(ga[k]) - fd) / (abs(fd) + 1e-30) < 1e-3


# ----------------------------------------------------------------------
# cdd.py — selection behaviour
# ----------------------------------------------------------------------

def test_cdd_includes_coarse_and_is_sparse():
    """CDD always retains the coarse level and stays near the budget K."""
    s = _setup_1d()
    Ah, Wn, D, x, h, N = s["Ah"], s["Wn"], s["D"], s["x"], s["h"], s["N"]
    levels = np.asarray(s["levels"])
    coarse = jnp.asarray(levels == levels.min())
    K = N // 16

    def solve_masked(mask, rhs):
        op = OP.make_masked_operator(Ah, mask)
        return ift_linear_solve(op, jnp.where(mask, rhs, 0.0),
                                solver="cg", rtol=1e-10, atol=1e-12)

    f = jnp.exp(-((jnp.asarray(x) - 0.42) / 0.06) ** 2)
    b = (h * (Wn.T @ f)) / D
    mask, c = CDD.cdd_select(lambda v: Ah @ v, solve_masked, b, coarse, K)
    # coarse fully retained
    assert bool(jnp.all(mask[jnp.asarray(coarse)]))
    # sparse: active set well below full N
    assert int(jnp.sum(mask)) < N // 2
    # accurate vs the full solve
    c_full = jnp.linalg.solve(Ah, b)
    sidx = s["sidx"]
    J_full = float((Wn[sidx] / D) @ c_full)
    J_cdd = float((Wn[sidx] / D) @ c)
    assert abs(J_cdd - J_full) / (abs(J_full) + 1e-30) < 1e-2
