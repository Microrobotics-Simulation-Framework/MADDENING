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

    # The node's actual gradient path: CDD selects the mask (a discrete,
    # non-differentiable step, stop_gradient'd), then the sensor is
    # differentiated through the FROZEN re-solve.  At a non-kink theta the mask
    # is locally constant, so this equals differentiating through the whole
    # selection -- the frozen-active-set adjoint.  (cdd_select's inner loop is a
    # lax.while_loop and is not reverse-differentiable; the node never asks it
    # to be, because it stop_gradients the mask.)
    def _mask_at(theta):
        f = jnp.exp(-((jnp.asarray(x) - theta) / 0.06) ** 2)
        b = (h * (Wn.T @ f)) / D
        mask, _, _ = CDD.cdd_select(lambda v: Ah @ v, solve_masked, b, coarse, K)
        return jax.lax.stop_gradient(mask)

    def J(theta, mask):
        f = jnp.exp(-((jnp.asarray(x) - theta) / 0.06) ** 2)
        b = (h * (Wn.T @ f)) / D
        return (Wn[sidx] / D) @ solve_masked(mask, b)

    th = jnp.asarray(0.42)
    mask = _mask_at(th)
    g = float(jax.grad(lambda t: J(t, mask))(th))
    e = 1e-5
    fd = float((J(th + e, mask) - J(th - e, mask)) / (2 * e))
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

# ----------------------------------------------------------------------
# JIT recompilation audit + JIT+grad (the spike's 1e-9 was eager-only).
#
# These guard the production hot path: the spike validated gradients in EAGER
# mode (jax.grad without jit), where BCOO.fromdense sees concrete arrays. Under
# jax.jit everything is abstract, so the masked operator must NOT call
# fromdense per call (it closes over a pre-assembled constant BCOO instead).
# We assert (a) no recompilation across same-shape calls, and (b) jax.grad
# through a *jit-compiled* solve still matches FD.
# ----------------------------------------------------------------------

def test_no_recompilation_transform():
    count = {"n": 0}

    @jax.jit
    def f(c):
        count["n"] += 1            # Python body runs once per trace
        return T.synthesis_2d(c, 4, 2, 4)

    x = jnp.zeros(T.n_dofs(4, 2, 2))
    f(x); f(x); f(x)
    assert count["n"] == 1


def test_no_recompilation_masked_solve_and_cdd():
    """The masked solve + CDD select compile once and do not recompile, and
    run under jax.jit at all (the fromdense-under-jit failure mode)."""
    s = _setup_1d()
    Ah, D = s["Ah"], s["D"]
    import jax.experimental.sparse as jsparse
    Ah_bcoo = jsparse.BCOO.fromdense(Ah)            # assembled once (constant)
    levels = np.asarray(s["levels"])
    coarse = jnp.asarray(levels == levels.min())
    N = s["N"]
    K = N // 16

    def solve_masked(mask, rhs):
        op = OP.make_masked_operator(Ah_bcoo, mask)
        return ift_linear_solve(op, jnp.where(mask, rhs, 0.0),
                                solver="cg", rtol=1e-10, atol=1e-12)

    count = {"n": 0}

    @jax.jit
    def run(b):
        count["n"] += 1
        _, c, _ = CDD.cdd_select(lambda v: Ah_bcoo @ v, solve_masked, b, coarse, K)
        return c

    b = jnp.asarray(np.random.default_rng(0).standard_normal(N))
    run(b); run(b); run(b)
    assert count["n"] == 1          # compiled once, reused (no silent recompile)


def test_jit_grad_through_solve_matches_fd():
    """jax.grad through a jit-compiled CDD solve matches FD -- the production
    JIT+grad path (the spike's 1e-9 was eager-only)."""
    s = _setup_1d()
    Ah, Wn, D, x, h, N, sidx = (s["Ah"], s["Wn"], s["D"], s["x"], s["h"],
                                s["N"], s["sidx"])
    import jax.experimental.sparse as jsparse
    Ah_bcoo = jsparse.BCOO.fromdense(Ah)
    levels = np.asarray(s["levels"])
    coarse = jnp.asarray(levels == levels.min())
    K = N // 16

    def solve_masked(mask, rhs):
        op = OP.make_masked_operator(Ah_bcoo, mask)
        return ift_linear_solve(op, jnp.where(mask, rhs, 0.0),
                                solver="cg", rtol=1e-10, atol=1e-12)

    # Frozen-active-set adjoint under jit (the production path): select+freeze
    # the mask, then jit(grad(.)) through the frozen re-solve.  cdd_select's
    # while_loop is forward-only; the node stop_gradients its mask.
    def _mask_at(theta):
        f = jnp.exp(-((jnp.asarray(x) - theta) / 0.06) ** 2)
        b = (h * (Wn.T @ f)) / D
        mask, _, _ = CDD.cdd_select(lambda v: Ah_bcoo @ v, solve_masked, b, coarse, K)
        return jax.lax.stop_gradient(mask)

    @jax.jit
    def J(theta, mask):
        f = jnp.exp(-((jnp.asarray(x) - theta) / 0.06) ** 2)
        b = (h * (Wn.T @ f)) / D
        return (Wn[sidx] / D) @ solve_masked(mask, b)

    th = jnp.asarray(0.42)
    mask = _mask_at(th)
    g = float(jax.jit(jax.grad(J))(th, mask))
    e = 1e-5
    fd = float((J(th + e, mask) - J(th - e, mask)) / (2 * e))
    assert abs(g - fd) / (abs(fd) + 1e-30) < 1e-5


# ----------------------------------------------------------------------
# Variable coefficient: static structure, traced data, dJ/da under jit.
# ----------------------------------------------------------------------

def test_sparsity_pattern_independent_of_coefficient():
    """The A_wave nonzero pattern is the same for a=1 and a Brinkman field."""
    side = 2 * 2 ** 5
    x = np.arange(side) / side
    a_unit = jnp.ones(side)
    a_jump = jnp.asarray(1.0 + 99.0 * (x > 0.5))
    A1 = OP.assemble_wave_dense(5, 2, 4, 1, a_grid=a_unit)["A_dense"]
    A2 = OP.assemble_wave_dense(5, 2, 4, 1, a_grid=a_jump)["A_dense"]
    r1, c1 = OP.sparsity_pattern(A1)
    r2, c2 = OP.sparsity_pattern(A2)
    p1 = set(zip(np.asarray(r1).tolist(), np.asarray(c1).tolist()))
    p2 = set(zip(np.asarray(r2).tolist(), np.asarray(c2).tolist()))
    # the jump field's pattern is a (near) superset; the structural support is
    # coefficient-independent up to entries that happen to vanish at a=1.
    assert p1.issubset(p2) or p2.issubset(p1) or len(p1 ^ p2) < 0.02 * len(p1)


def test_grad_through_traced_data_bcoo_under_jit():
    """BCOO with static indices + traced data: dJ/da flows under jax.jit
    (structure static, values differentiable in a -- the Brinkman pattern)."""
    nl, nc = 5, 2
    ref = OP.assemble_wave_dense(nl, nc, 4, 1)["A_dense"]
    rows, cols = OP.sparsity_pattern(ref)
    side = nc * 2 ** nl
    x = np.arange(side) / side
    sidx = int(np.argmin(np.abs(x - 0.30)))
    h = 1.0 / side
    a0 = jnp.asarray(1.0 + 0.5 * np.sin(2 * np.pi * x))
    r0 = OP.assemble_wave_dense(nl, nc, 4, 1, a_grid=a0)
    D = _setup_pc_diag(r0["A_dense"], r0["levels"])

    @jax.jit
    def J_of_a(a_grid):
        r = OP.assemble_wave_dense(nl, nc, 4, 1, a_grid=a_grid)
        A_bcoo = OP.bcoo_with_traced_data(r["A_dense"], rows, cols)  # static idx
        Wn = r["Wn"]
        f = jnp.exp(-((jnp.asarray(x) - 0.42) / 0.06) ** 2)
        b = (h * (Wn.T @ f)) / D
        # masked-matvec solve over the traced-data BCOO (full mask = all active)
        full = jnp.ones(r["N"], dtype=bool)
        op = OP.make_masked_operator(A_bcoo, full)

        def scaled_op(v):
            return op(v / D) / D
        c = ift_linear_solve(scaled_op, b, solver="gmres", rtol=1e-10, atol=1e-12)
        return (Wn[sidx] / D) @ c

    ga = jax.grad(J_of_a)(a0)
    k = side // 2
    e = 1e-6
    fd = float((J_of_a(a0.at[k].add(e)) - J_of_a(a0.at[k].add(-e))) / (2 * e))
    assert abs(float(ga[k]) - fd) / (abs(fd) + 1e-30) < 1e-3


def _setup_pc_diag(A, levels):
    return PC.diagonal_scaling(jnp.diag(A), levels, "hybrid")


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
    mask, c, _ = CDD.cdd_select(lambda v: Ah @ v, solve_masked, b, coarse, K)
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


# ----------------------------------------------------------------------
# M3 — high-contrast regression guard + MAX_OUTER recalibration
# ----------------------------------------------------------------------

def _chi_inclusion_2d(nl=4, nc=2, contrast=1e5):
    """2D χ inclusion problem: (-∇·((1+χ)∇) + m)φ = -∇·(χ H₀), H₀=(0,1)."""
    side = nc * 2 ** nl
    N = side * side
    h = 1.0 / side
    c1 = np.arange(side) / side
    X, Y = np.meshgrid(c1, c1, indexing="ij")
    chi = np.where((X - 0.5) ** 2 + (Y - 0.5) ** 2 < 0.15 ** 2, contrast, 0.0)
    a = jnp.asarray(1.0 + chi)
    f = jnp.asarray((-(np.roll(chi, -1, axis=1) - np.roll(chi, 1, axis=1))
                     / (2 * h)).reshape(-1))
    r = OP.assemble_wave_dense(nl, nc, 4, 2, a_grid=a, mass=1.0)
    A, Wn, lev = r["A_dense"], r["Wn"], r["levels"]
    D = PC.diagonal_scaling(jnp.diag(A), lev, "hybrid")
    Ah = (A / D[:, None]) / D[None, :]
    import jax.experimental.sparse as jsparse
    Ah_bcoo = jsparse.BCOO.fromdense(Ah)
    bh = ((h ** 2) * (Wn.T @ f)) / D
    lev_np = np.asarray(lev)
    coarse = jnp.asarray(lev_np == lev_np.min())
    c_full = jnp.linalg.solve(Ah, bh)
    phi_full = Wn @ (c_full / D)
    return dict(Ah=Ah, Ah_bcoo=Ah_bcoo, Wn=Wn, D=D, bh=bh, coarse=coarse,
                phi_full=phi_full, N=N)


def test_high_contrast_regression_max_outer():
    """At χ=1e5 with an ADEQUATE budget, the recalibrated MAX_OUTER=200 gives an
    accurate direct solve, and the old ceiling (30) reproduces the silent
    worse-than-zero defect the flag now catches.

    This is the M3 regression guard: it FAILS on the pre-M1/M3 behaviour.
    """
    s = _chi_inclusion_2d(contrast=1e5)
    Ah, Ah_bcoo, Wn, D, bh, coarse, phi_full, N = (
        s["Ah"], s["Ah_bcoo"], s["Wn"], s["D"], s["bh"], s["coarse"],
        s["phi_full"], s["N"])
    K = N // 4                       # budget scaled to the contrast (see caveat)
    nrm = float(jnp.linalg.norm(phi_full)) + 1e-30

    def solve_masked(mask, rhs):
        return OP.gather_solve(Ah, mask, rhs, K)

    def relerr(c):
        return float(jnp.linalg.norm(Wn @ (c / D) - phi_full)) / nrm

    # Recalibrated ceiling: accurate, and reported converged.
    mask, c, conv = CDD.cdd_select(lambda v: Ah_bcoo @ v, solve_masked, bh,
                                   coarse, K, max_outer=200)
    assert relerr(c) < 1e-2, f"χ=1e5 relerr={relerr(c)}"
    assert bool(conv) is True

    # Old ceiling: worse-than-zero, and CORRECTLY flagged not-converged
    # (iteration starvation — the marking hadn't finished growing the set).
    mask30, c30, conv30 = CDD.cdd_select(lambda v: Ah_bcoo @ v, solve_masked,
                                         bh, coarse, K, max_outer=30)
    assert relerr(c30) > 0.5, "expected the max_outer=30 defect to reproduce"
    assert bool(conv30) is False


def test_high_contrast_small_budget_is_a_documented_limitation():
    """Budget too small for the contrast: the solve is inaccurate even though the
    active set fills and the scaled residual is small (κ~contrast means small
    residual does not bound the error).  This guards the honest scope boundary —
    adequate budget or a contrast-robust preconditioner (R1) is required — so
    nobody mistakes the small residual / converged=True for success.
    """
    s = _chi_inclusion_2d(contrast=1e5)
    Ah, Ah_bcoo, Wn, D, bh, coarse, phi_full, N = (
        s["Ah"], s["Ah_bcoo"], s["Wn"], s["D"], s["bh"], s["coarse"],
        s["phi_full"], s["N"])
    K = N // 16                      # too small for χ=1e5
    nrm = float(jnp.linalg.norm(phi_full)) + 1e-30

    def solve_masked(mask, rhs):
        return OP.gather_solve(Ah, mask, rhs, K)

    mask, c, conv = CDD.cdd_select(lambda v: Ah_bcoo @ v, solve_masked, bh,
                                   coarse, K, max_outer=200)
    rel_resid = float(jnp.linalg.norm(bh - Ah @ c) / (jnp.linalg.norm(bh) + 1e-30))
    sol_err = float(jnp.linalg.norm(Wn @ (c / D) - phi_full)) / nrm
    # budget filled, flag reads healthy, residual is small ...
    assert bool(conv) is True
    assert rel_resid < 1e-2
    # ... yet the solution error is large: the flag cannot catch this regime.
    assert sol_err > 0.5


# ----------------------------------------------------------------------
# M4 — traceable diagonal_scaling (JAX segment_sum), equivalence + grad
# ----------------------------------------------------------------------

def _ref_diagonal_scaling(diag, levels, kind="hybrid", t=1.0):
    """The pre-M4 NumPy implementation, kept here as the equivalence oracle."""
    d = np.abs(np.asarray(diag, dtype=np.float64))
    lev = np.asarray(levels).astype(int)
    uniq = sorted(set(lev.tolist()))
    if kind == "full":
        D = np.sqrt(d)
    elif kind == "dk":
        D = 2.0 ** (t * lev)
    else:
        D = np.zeros_like(d)
        for i, l in enumerate(uniq):
            m = lev == l
            if kind == "hybrid" and i == 0:
                D[m] = np.sqrt(d[m])
            else:
                D[m] = np.sqrt(d[m].mean())
    return np.where(D > 0, D, 1.0)


@pytest.mark.parametrize("kind", ["hybrid", "full", "level", "dk"])
def test_diagonal_scaling_matches_numpy_reference(kind):
    """JAX diagonal_scaling reproduces the old NumPy loop to round-off."""
    res = OP.assemble_wave_operator(5, 2, order=4, dim=2, mass=1.0)
    diag = jnp.diag(res["A_dense"])
    levels = res["levels"]
    got = np.asarray(PC.diagonal_scaling(diag, levels, kind))
    ref = _ref_diagonal_scaling(diag, levels, kind)
    assert np.allclose(got, ref, rtol=0, atol=1e-12), \
        f"{kind}: max|Δ|={np.max(np.abs(got - ref))}"


def test_diagonal_scaling_is_traceable_and_differentiable():
    """diag depends on a(x); D flows a gradient through segment_sum (the θ→A
    prerequisite -- the old NumPy version raised under jax.grad)."""
    nl, nc, side = 5, 2, 2 * 2 ** 5
    a0 = jnp.asarray(1.0 + 0.3 * np.sin(2 * np.pi * np.arange(side) / side))
    res0 = OP.assemble_wave_dense(nl, nc, 4, 1, a_grid=a0, mass=1.0)
    levels = res0["levels"]

    def scalar_of_a(a_grid):
        r = OP.assemble_wave_dense(nl, nc, 4, 1, a_grid=a_grid, mass=1.0)
        D = PC.diagonal_scaling(jnp.diag(r["A_dense"]), levels, "hybrid")
        return jnp.sum(D)                      # any differentiable reduction

    g = jax.grad(scalar_of_a)(a0)              # must not raise
    assert g.shape == a0.shape
    assert jnp.all(jnp.isfinite(g))
    # non-trivial dependence (D genuinely varies with a)
    assert float(jnp.linalg.norm(g)) > 0
