"""M13–M18 tests — matrix-free wavelet operator path.

The matrix-free path must be a transparent drop-in for the dense path on
problems that fit in memory: it applies the same A_wave = Wnᵀ A_phys Wn without
materialising (N, N). The conftest.py autouse fixture provides float64.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.nodes.adaptive.wavelets import matrixfree as MF
from maddening.nodes.adaptive.wavelets import operator as OP
from maddening.nodes.adaptive.wavelets import transform as T


# ----------------------------------------------------------------------
# M13 — matrix-free Wn·v / Wnᵀ·u
# ----------------------------------------------------------------------

@pytest.mark.parametrize("dim,nl,nc", [(1, 5, 2), (2, 3, 2), (3, 2, 2)])
def test_matrixfree_wn_apply_and_transpose_match_dense(dim, nl, nc):
    """Wn v and Wnᵀ u match the dense L²-normalised synthesis to 1e-12."""
    side = nc * 2 ** nl
    h = 1.0 / side
    N = T.n_dofs(nl, nc, dim)
    # dense reference Wn
    W = T.synthesis_matrix(nl, nc, 4, dim=dim)
    norms = jnp.sqrt((h ** dim) * jnp.sum(W ** 2, axis=0))
    norms = jnp.where(norms > 0, norms, 1.0)
    Wn = W / norms[None, :]

    wn_apply, wn_transpose = MF.make_wn_ops(nl, nc, 4, dim, norms)
    rng = np.random.default_rng(0)
    v = jnp.asarray(rng.standard_normal(N))
    u = jnp.asarray(rng.standard_normal(N))

    assert float(jnp.linalg.norm(wn_apply(v) - Wn @ v)) < 1e-12
    assert float(jnp.linalg.norm(wn_transpose(u) - Wn.T @ u)) < 1e-12


@pytest.mark.parametrize("dim,nl,nc", [(1, 5, 2), (2, 3, 2)])
def test_matrixfree_transpose_is_not_analysis(dim, nl, nc):
    """Guard the D3 trap: analysis_* is NOT Wnᵀ (off by order unity)."""
    side = nc * 2 ** nl
    h = 1.0 / side
    N = T.n_dofs(nl, nc, dim)
    W = T.synthesis_matrix(nl, nc, 4, dim=dim)
    norms = jnp.sqrt((h ** dim) * jnp.sum(W ** 2, axis=0))
    norms = jnp.where(norms > 0, norms, 1.0)
    Wn = W / norms[None, :]
    _, wn_transpose = MF.make_wn_ops(nl, nc, 4, dim, norms)

    u = jnp.asarray(np.random.default_rng(1).standard_normal(N))
    ana = {1: T.analysis_1d, 2: T.analysis_2d, 3: T.analysis_3d}[dim]
    # analysis(u)/norms would be the WRONG "transpose"; confirm it disagrees
    wrong = ana(u, nl, nc, 4) / norms
    assert float(jnp.linalg.norm(wn_transpose(u) - Wn.T @ u)) < 1e-12
    assert float(jnp.linalg.norm(wrong - Wn.T @ u)) > 0.1


# ----------------------------------------------------------------------
# M14 — O(log N) column norms via structural-block representatives
# ----------------------------------------------------------------------

@pytest.mark.parametrize("dim,nl,nc", [(1, 6, 2), (2, 4, 2), (3, 3, 1), (2, 5, 2)])
def test_column_norms_fast_matches_dense(dim, nl, nc):
    """The O(log N) block-representative norms reproduce the dense column_norms
    to machine zero (derisk D4 recipe)."""
    side = nc * 2 ** nl
    h = 1.0 / side
    ref = OP.column_norms(nl, nc, 4, dim, h)
    fast = OP.column_norms_fast(nl, nc, 4, dim, h)
    assert fast.shape == ref.shape
    assert float(jnp.max(jnp.abs(fast - ref))) < 1e-12


def test_structural_blocks_count_is_log_n():
    """#blocks = 1 + n_levels·n_subband (O(log N)), and reps index each block."""
    for dim, n_sub in [(1, 1), (2, 3), (3, 7)]:
        ids, reps = T.structural_blocks(4, 2, dim)
        assert len(reps) == 1 + 4 * n_sub
        assert ids.max() + 1 == len(reps)
        # reps are the first index of each block (strictly increasing)
        assert bool(np.all(np.diff(reps) > 0))


# ----------------------------------------------------------------------
# M15 — matrix-free A_phys stencil (matches the dense operators)
# ----------------------------------------------------------------------

@pytest.mark.parametrize("dim,side", [(1, 32), (2, 16), (3, 8)])
def test_matrixfree_laplacian_matches_dense(dim, side):
    h = 1.0 / side
    A = OP.physical_laplacian(side, dim, h, mass=1.3)
    apply = MF.make_laplacian_apply(side, dim, h, mass=1.3)
    rng = np.random.default_rng(0)
    v = jnp.asarray(rng.standard_normal(side ** dim))
    assert float(jnp.linalg.norm(apply(v) - A @ v)) < 1e-11


@pytest.mark.parametrize("dim,side", [(1, 32), (2, 16), (3, 8)])
def test_matrixfree_varcoeff_matches_dense(dim, side):
    rng = np.random.default_rng(1)
    a = jnp.asarray(1.0 + 0.5 * rng.random((side,) * dim))
    h = 1.0 / side
    A = OP.physical_varcoeff(a, dim, h, mass=0.7)
    apply = MF.make_varcoeff_apply(a, side, dim, h, mass=0.7)
    v = jnp.asarray(rng.standard_normal(side ** dim))
    assert float(jnp.linalg.norm(apply(v) - A @ v)) < 1e-11


def test_matrixfree_varcoeff_djda_flows():
    """dJ/da flows through the matrix-free varcoeff apply (application-1 path)."""
    side, dim, h = 16, 2, 1.0 / 16
    v = jnp.asarray(np.random.default_rng(2).standard_normal(side ** dim))
    a0 = jnp.ones(side ** dim)

    def J(a_flat):
        apply = MF.make_varcoeff_apply(a_flat, side, dim, h, mass=1.0)
        return jnp.sum(apply(v) ** 2)

    g = jax.grad(J)(a0)
    assert g.shape == a0.shape and jnp.all(jnp.isfinite(g))
    assert float(jnp.linalg.norm(g)) > 0


# ----------------------------------------------------------------------
# M16 — masked CG replaces gather_solve (matrix-free frozen solve)
# ----------------------------------------------------------------------

def _matrixfree_setup(nl, nc, dim, mass=1.0, a_grid=None):
    from maddening.nodes.adaptive.wavelets import precond as PC
    side = nc * 2 ** nl
    h = 1.0 / side
    res = (OP.assemble_wave_operator(nl, nc, 4, dim, mass=mass) if a_grid is None
           else OP.assemble_wave_operator(nl, nc, 4, dim, mass=mass, a_grid=a_grid))
    A_wave, levels = res["A_dense"], res["levels"]
    D = PC.diagonal_scaling(jnp.diag(A_wave), levels, "hybrid")
    Ah = (A_wave / D[:, None]) / D[None, :]
    norms = OP.column_norms_fast(nl, nc, 4, dim, h)
    if a_grid is None:
        a_phys = MF.make_laplacian_apply(side, dim, h, mass=mass)
    else:
        a_phys = MF.make_varcoeff_apply(a_grid, side, dim, h, mass=mass)
    wave_apply = MF.make_wave_apply(nl, nc, 4, dim, norms, a_phys, D)
    return dict(side=side, h=h, N=side ** dim, Ah=Ah, D=D, norms=norms,
                wave_apply=wave_apply, levels=levels)


@pytest.mark.parametrize("dim,nl,nc", [(2, 3, 2), (3, 3, 1)])   # 16^2, 8^3
def test_matrixfree_wave_apply_matches_dense_scaled(dim, nl, nc):
    """The matrix-free scaled matvec Â equals the dense D⁻¹ A_wave D⁻¹."""
    s = _matrixfree_setup(nl, nc, dim)
    v = jnp.asarray(np.random.default_rng(0).standard_normal(s["N"]))
    assert float(jnp.linalg.norm(s["wave_apply"](v) - s["Ah"] @ v)) < 1e-10


@pytest.mark.parametrize("dim,nl,nc", [(2, 3, 2), (3, 3, 1)])
def test_masked_cg_matches_gather_solve(dim, nl, nc):
    """masked CG reproduces gather_solve on the same active-block system."""
    from maddening.nodes.adaptive.wavelets import cdd as CDD
    import jax.experimental.sparse as jsparse
    s = _matrixfree_setup(nl, nc, dim)
    Ah, N = s["Ah"], s["N"]
    Ah_bcoo = jsparse.BCOO.fromdense(Ah)
    lev = np.asarray(s["levels"]); coarse = jnp.asarray(lev == lev.min())
    K = N // 4
    rhs = jnp.asarray(np.random.default_rng(1).standard_normal(N))

    def solve_masked(mask, r):
        return OP.gather_solve(Ah, mask, r, K)

    mask, _, _ = CDD.cdd_select(lambda v: Ah_bcoo @ v, solve_masked, rhs,
                                coarse, K)
    c_gather = OP.gather_solve(Ah, mask, rhs, K)
    c_cg = MF.masked_cg_solve(s["wave_apply"], mask, rhs, rtol=1e-12, atol=1e-14)
    assert float(jnp.linalg.norm(c_cg - c_gather)) / \
        (float(jnp.linalg.norm(c_gather)) + 1e-30) < 1e-8
