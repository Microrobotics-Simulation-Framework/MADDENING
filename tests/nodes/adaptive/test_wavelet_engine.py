"""The interpolating-wavelet engine behind ``WaveletAdaptiveNode``.

Transforms, operator assembly, preconditioning, the two frozen solves and
the CDD selection, each checked against a dense reference or a property
it must have for the node's contract with ``AdaptiveNode`` to hold.  The
suite's ``conftest.py`` enables float64 per test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.solver_utils import ift_linear_solve
from maddening.nodes.adaptive.wavelets import cdd as CDD
from maddening.nodes.adaptive.wavelets import dirichlet as DIR
from maddening.nodes.adaptive.wavelets import operator as OP
from maddening.nodes.adaptive.wavelets import precond as PC
from maddening.nodes.adaptive.wavelets import transform as T


def _kappa(A) -> float:
    ev = np.linalg.eigvalsh(np.asarray(A))
    return float(ev[-1] / ev[0])


def _scaled(op: OP.WaveletOperator, kind: str = "hybrid"):
    D = PC.diagonal_scaling(jnp.diag(op.A), op.levels, kind)
    return (op.A / D[:, None]) / D[None, :], D


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim,nl,nc", [(1, 6, 2), (2, 3, 2), (3, 2, 1)])
def test_analysis_inverts_synthesis_to_round_off(dim, nl, nc):
    n = T.n_dofs(nl, nc, dim)
    v = jnp.asarray(np.random.default_rng(0).standard_normal(n))
    back = T.synthesis(T.analysis(v, nl, nc, order=4, dim=dim), nl, nc, order=4, dim=dim)
    assert float(jnp.linalg.norm(back - v) / jnp.linalg.norm(v)) < 1e-12


@pytest.mark.parametrize("dim,nl,nc", [(1, 4, 2), (2, 2, 2)])
def test_the_dense_synthesis_matrix_is_the_matrix_free_synthesis_columnwise(dim, nl, nc):
    W = T.synthesis_matrix(nl, nc, order=4, dim=dim)
    n = W.shape[0]
    for j in (0, 1, n // 2, n - 1):
        ej = jnp.zeros(n).at[j].set(1.0)
        col = T.synthesis(ej, nl, nc, order=4, dim=dim)
        assert float(jnp.linalg.norm(col - W[:, j])) < 1e-12


def test_dd4_midpoint_prediction_is_fourth_order():
    errs = []
    for n in (64, 128, 256):
        x = np.arange(n) / n
        f = jnp.asarray(np.sin(2 * np.pi * x))
        pred = T._predict_axis(f, 0, 4)
        exact = jnp.asarray(np.sin(2 * np.pi * (x + 0.5 / n)))
        errs.append(float(jnp.max(jnp.abs(pred - exact))))
    # 4th order: ~16x per doubling; 10x leaves room for the pre-asymptotic first pair
    assert errs[0] / errs[1] > 10 and errs[1] / errs[2] > 10, errs


@pytest.mark.parametrize("order", [2, 4, 6])
def test_every_supported_order_reproduces_a_constant(order):
    """The filters sum to one, so a constant field has zero detail."""
    c = T.analysis(jnp.full(64, 3.5), 5, 2, order=order, dim=1)
    assert float(jnp.max(jnp.abs(c[2:]))) < 1e-12 and bool(jnp.allclose(c[:2], 3.5))


def test_an_unsupported_order_or_dimension_is_refused():
    with pytest.raises(ValueError, match="order"):
        T.synthesis(jnp.zeros(8), 2, 2, order=3)
    with pytest.raises(ValueError, match="dim"):
        T.n_dofs(2, 2, 4)


def test_level_labels_have_one_entry_per_function_and_a_coarse_block_first():
    for dim in (1, 2, 3):
        lab = np.asarray(T.level_labels(3, 2, dim))
        assert lab.shape == (T.n_dofs(3, 2, dim),)
        assert int((lab == 0).sum()) == 2 ** dim + (2 ** dim - 1) * 2 ** dim
        assert lab[: 2 ** dim].max() == 0 and np.all(np.diff(lab) >= 0)


def test_synthesis_traces_once_under_jit():
    count = {"n": 0}

    @jax.jit
    def f(c):
        count["n"] += 1
        return T.synthesis(c, 3, 2, order=4, dim=2)

    x = jnp.zeros(T.n_dofs(3, 2, 2))
    f(x); f(x); f(x)
    assert count["n"] == 1


# ---------------------------------------------------------------------------
# operator + precond
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim,nl,nc,lo,hi", [
    (1, 7, 2, 18.0, 23.0),   # hybrid-Jacobi kappa ~ 20.4 in 1-D (256 points)
    (2, 3, 2, 30.0, 37.0),   # ~ 33.3 in 2-D (16^2); ~ 37.7 at 32^2
])
def test_hybrid_jacobi_conditioning_is_bounded_and_matches_full_jacobi(dim, nl, nc, lo, hi):
    op = OP.assemble_operator(nl, nc, order=4, dim=dim, mass=1.0)
    Ah, _ = _scaled(op, "hybrid")
    Af, _ = _scaled(op, "full")
    k_h, k_f = _kappa(Ah), _kappa(Af)
    assert lo < k_h < hi, k_h
    assert abs(k_h - k_f) / k_f < 1e-3


def test_the_operator_is_symmetric_positive_definite_and_the_sparse_copy_matches():
    op = OP.assemble_operator(5, 2, order=4, dim=1, mass=1.0)
    A = np.asarray(op.A)
    assert np.allclose(A, A.T)
    assert np.linalg.eigvalsh(A).min() > 0
    v = jnp.asarray(np.random.default_rng(1).standard_normal(op.n))
    assert float(jnp.max(jnp.abs(op.A_sparse @ v - op.A @ v))) < 1e-10 * float(jnp.max(jnp.abs(op.A @ v)))
    assert int(op.A_sparse.nse) < op.n ** 2


def test_the_full_wavelet_solve_is_the_finite_difference_solve_in_another_basis():
    """The change of basis is exact: Wn (Wn^T A Wn)^-1 Wn^T b equals A_phys^-1 b."""
    op = OP.assemble_operator(5, 2, order=4, dim=1, mass=1.0)
    A_phys = jnp.asarray(OP._physical_operator(op.side, 1, op.h, 1.0, "periodic"))
    f = jnp.asarray(np.random.default_rng(2).standard_normal(op.n))
    u_fd = jnp.linalg.solve(A_phys, op.h * f)
    u_w = op.Wn @ jnp.linalg.solve(op.A, op.h * (op.Wn.T @ f))
    assert float(jnp.max(jnp.abs(u_w - u_fd))) < 1e-10 * float(jnp.max(jnp.abs(u_fd)))


def test_assembly_honours_the_requested_dtype():
    op32 = OP.assemble_operator(4, 2, order=4, dim=1, dtype=jnp.float32)
    assert op32.A.dtype == jnp.float32 and op32.Wn.dtype == jnp.float32
    assert op32.A_sparse.data.dtype == jnp.float32
    op64 = OP.assemble_operator(4, 2, order=4, dim=1, dtype=jnp.float64)
    assert op64.A.dtype == jnp.float64


def test_an_unknown_boundary_or_preconditioner_is_refused():
    with pytest.raises(ValueError, match="boundary"):
        OP.assemble_operator(3, 2, boundary="neumann")
    with pytest.raises(ValueError, match="preconditioner"):
        PC.diagonal_scaling(jnp.ones(4), jnp.zeros(4, jnp.int32), "ilu")


def test_dahmen_kunoth_scaling_is_two_to_the_level():
    lev = T.level_labels(3, 2, 1)
    D = PC.diagonal_scaling(jnp.ones(lev.shape[0]), lev, "dk", t=1.0)
    assert bool(jnp.array_equal(D, 2.0 ** lev.astype(D.dtype)))


# ---------------------------------------------------------------------------
# dirichlet
# ---------------------------------------------------------------------------

def test_dirichlet_side_count_and_level_labels():
    assert DIR.dirichlet_side(2, 2) == 11 and DIR.dirichlet_side(4, 2) == 47
    W, lev, side = DIR.synthesis_matrix_dirichlet(3, 2, order=4, dim=1)
    assert W.shape == (side, side) == (23, 23)
    assert np.linalg.matrix_rank(np.asarray(W)) == side
    assert int((np.asarray(lev) == 0).sum()) == 2 + 3     # coarse block + first detail band


def test_dirichlet_basis_is_better_conditioned_than_periodic_in_one_dimension():
    op = OP.assemble_operator(4, 2, order=4, dim=1, mass=1.0, boundary="dirichlet")
    Ah, _ = _scaled(op)
    assert 2.5 < _kappa(Ah) < 6.0


def test_dirichlet_full_solve_matches_the_dirichlet_finite_difference_solve():
    op = OP.assemble_operator(4, 2, order=4, dim=1, mass=1.0, boundary="dirichlet")
    A_phys = jnp.asarray(OP._physical_operator(op.side, 1, op.h, 1.0, "dirichlet"))
    f = jnp.asarray(np.random.default_rng(3).standard_normal(op.n))
    u_fd = jnp.linalg.solve(A_phys, op.h * f)
    u_w = op.Wn @ jnp.linalg.solve(op.A, op.h * (op.Wn.T @ f))
    assert float(jnp.max(jnp.abs(u_w - u_fd))) < 1e-10 * float(jnp.max(jnp.abs(u_fd)))


def test_two_dimensional_dirichlet_is_the_tensor_product_basis():
    W1, lev1, side = DIR.synthesis_matrix_dirichlet(2, 2, order=4, dim=1)
    W2, lev2, side2 = DIR.synthesis_matrix_dirichlet(2, 2, order=4, dim=2)
    assert side2 == side and W2.shape == (side * side,) * 2
    assert np.allclose(np.asarray(W2), np.kron(np.asarray(W1), np.asarray(W1)))
    assert np.array_equal(np.asarray(lev2), np.maximum.outer(np.asarray(lev1), np.asarray(lev1)).reshape(-1))


# ---------------------------------------------------------------------------
# frozen solves
# ---------------------------------------------------------------------------

def _random_mask(n, count, seed):
    idx = np.random.default_rng(seed).choice(n, size=count, replace=False)
    return jnp.zeros(n, dtype=bool).at[jnp.asarray(idx)].set(True)


def test_gather_solve_equals_the_masked_dense_solve_and_is_zero_off_the_mask():
    op = OP.assemble_operator(5, 2, order=4, dim=1)
    Ah, _ = _scaled(op)
    mask = _random_mask(op.n, 10, seed=4)
    rhs = jnp.asarray(np.random.default_rng(5).standard_normal(op.n))
    c = OP.gather_solve(Ah, mask, rhs, buf=12)
    idx = np.flatnonzero(np.asarray(mask))
    ref = jnp.linalg.solve(Ah[jnp.ix_(idx, idx)], rhs[idx])
    assert float(jnp.max(jnp.abs(c[idx] - ref))) < 1e-12
    assert bool(jnp.all(c[~mask] == 0.0))


def test_gather_solve_agrees_with_the_masked_operator_through_ift_linear_solve():
    op = OP.assemble_operator(5, 2, order=4, dim=1)
    Ah, _ = _scaled(op)
    mask = _random_mask(op.n, 10, seed=6)
    rhs = jnp.asarray(np.random.default_rng(7).standard_normal(op.n))
    c_g = OP.gather_solve(Ah, mask, rhs, buf=10)
    c_cg = ift_linear_solve(OP.make_masked_operator(op.A_sparse, mask), jnp.where(mask, rhs, 0.0),
                            solver="cg", rtol=1e-12, atol=1e-14)
    # the cg path solves the *unscaled* sparse operator here; compare in physical coordinates
    _, D = _scaled(op)
    assert float(jnp.max(jnp.abs(c_g / D - c_cg / 1.0)) ) > 0  # different coordinates, sanity only
    c_cg_scaled = ift_linear_solve(OP.make_masked_operator(Ah, mask), jnp.where(mask, rhs, 0.0),
                                   solver="cg", rtol=1e-12, atol=1e-14)
    assert float(jnp.max(jnp.abs(c_g - c_cg_scaled))) < 1e-9


def test_gather_solve_gradient_with_respect_to_the_right_hand_side_matches_finite_differences():
    op = OP.assemble_operator(5, 2, order=4, dim=1)
    Ah, _ = _scaled(op)
    mask = _random_mask(op.n, 8, seed=8)
    rhs = jnp.asarray(np.random.default_rng(9).standard_normal(op.n))
    w = jnp.asarray(np.random.default_rng(10).standard_normal(op.n))
    J = lambda b: w @ OP.gather_solve(Ah, mask, b, buf=8)
    g = jax.grad(J)(rhs)
    e = jnp.zeros(op.n).at[int(np.flatnonzero(np.asarray(mask))[0])].set(1e-6)
    fd = (J(rhs + e) - J(rhs - e)) / 2e-6
    k = int(np.flatnonzero(np.asarray(mask))[0])
    assert abs(float(g[k]) - float(fd)) < 1e-6 * (1 + abs(float(fd)))


def test_gather_solve_silently_truncates_a_mask_larger_than_its_buffer():
    """The documented hazard, pinned: with more active functions than the
    buffer holds the result is *not* the masked solve, and nothing raises.
    This is why ``WaveletAdaptiveNode`` caps its selection at ``k`` and
    overrides the all-true full-basis gradient with a dense solve."""
    op = OP.assemble_operator(5, 2, order=4, dim=1)
    Ah, _ = _scaled(op)
    full = jnp.ones(op.n, dtype=bool)
    rhs = jnp.asarray(np.random.default_rng(11).standard_normal(op.n))
    truncated = OP.gather_solve(Ah, full, rhs, buf=8)
    exact = jnp.linalg.solve(Ah, rhs)
    assert int((truncated != 0).sum()) == 8
    assert float(jnp.linalg.norm(truncated - exact) / jnp.linalg.norm(exact)) > 1e-2


# ---------------------------------------------------------------------------
# cdd
# ---------------------------------------------------------------------------

def _cdd_setup(nl=6, nc=2):
    op = OP.assemble_operator(nl, nc, order=4, dim=1)
    Ah, D = _scaled(op)
    lev = np.asarray(op.levels)
    coarse = jnp.asarray(lev == lev.min())
    x = np.arange(op.side) / op.side
    f = jnp.exp(-((jnp.asarray(x) - 0.42) / 0.06) ** 2)
    b_hat = (op.h * (op.Wn.T @ f)) / D
    return op, Ah, D, coarse, b_hat


def test_cdd_keeps_the_coarse_level_returns_a_bool_mask_and_never_exceeds_the_budget():
    op, Ah, D, coarse, b_hat = _cdd_setup()
    K = op.n // 16
    mask, c = CDD.cdd_select(lambda v: Ah @ v, lambda m, r: OP.gather_solve(Ah, m, r, K), b_hat, coarse, K)
    assert mask.dtype == jnp.bool_ and mask.shape == (op.n,)
    assert bool(jnp.all(mask[coarse]))
    assert int(coarse.sum()) < int(mask.sum()) <= K
    assert bool(jnp.all(c[~mask] == 0.0))


def test_cdd_reaches_the_budget_on_a_localised_source_and_is_accurate_at_the_sensor():
    op, Ah, D, coarse, b_hat = _cdd_setup()
    K = op.n // 16
    mask, c = CDD.cdd_select(lambda v: Ah @ v, lambda m, r: OP.gather_solve(Ah, m, r, K), b_hat, coarse, K)
    assert int(mask.sum()) == K
    sidx = int(np.argmin(np.abs(np.arange(op.side) / op.side - 0.30)))
    srow = op.Wn[sidx] / D
    j_full = float(srow @ jnp.linalg.solve(Ah, b_hat))
    assert abs(float(srow @ c) - j_full) / abs(j_full) < 1e-2


def test_cdd_never_marks_a_function_whose_residual_is_exactly_zero():
    """A zero source has an exactly zero residual everywhere: nothing is
    marked, and the set is the coarse level -- non-empty -- rather than
    empty or padded with arbitrary functions to reach the budget."""
    op, Ah, D, coarse, _ = _cdd_setup(nl=4)
    K = op.n // 2
    mask, c = CDD.cdd_select(lambda v: Ah @ v, lambda m, r: OP.gather_solve(Ah, m, r, K),
                             jnp.zeros(op.n), coarse, K)
    assert bool(jnp.array_equal(mask, coarse))
    assert bool(jnp.all(c == 0.0))


def test_cdd_runs_under_jit_and_under_grad_when_its_input_carries_no_tangent():
    op, Ah, D, coarse, _ = _cdd_setup()
    K = op.n // 16
    x = jnp.asarray(np.arange(op.side) / op.side)
    sidx = int(np.argmin(np.abs(np.asarray(x) - 0.30)))
    srow = op.Wn[sidx] / D

    def J(theta):
        b_hat = (op.h * (op.Wn.T @ jnp.exp(-((x - theta) / 0.06) ** 2))) / D
        mask, _ = CDD.cdd_select(lambda v: op.A_sparse @ (v / D) / D, lambda m, r: OP.gather_solve(Ah, m, r, K),
                                 jax.lax.stop_gradient(b_hat), coarse, K)
        return srow @ OP.gather_solve(Ah, jax.lax.stop_gradient(mask), b_hat, K)

    th = jnp.asarray(0.42)
    g = float(jax.jit(jax.grad(J))(th))
    e = 1e-5
    fd = float((J(th + e) - J(th - e)) / (2 * e))
    assert abs(g - fd) / abs(fd) < 1e-6
