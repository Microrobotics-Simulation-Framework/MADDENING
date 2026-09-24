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


def test_gather_solve_poisons_a_mask_larger_than_its_buffer_instead_of_truncating():
    """The former hazard, closed.  With more active functions than the buffer
    holds the result used to be a plausible wrong answer -- the first ``buf``
    of them solved, the rest dropped, nothing raised -- which is the
    mechanism behind the 5x gradient error the audit found at ``dim=2,
    n_levels=2``.  A jitted function cannot raise, so the gathered block is
    NaN and nothing downstream can mistake it for the masked solve; exactly
    at the buffer size, and at the full basis, the solve is exact."""
    op = OP.assemble_operator(5, 2, order=4, dim=1)
    Ah, _ = _scaled(op)
    rhs = jnp.asarray(np.random.default_rng(11).standard_normal(op.n))
    nine = jnp.zeros(op.n, dtype=bool).at[:9].set(True)
    poisoned = OP.gather_solve(Ah, nine, rhs, buf=8)
    assert bool(jnp.all(jnp.isnan(poisoned[:8]))) and bool(jnp.all(poisoned[8:] == 0.0))
    jitted = jax.jit(OP.gather_solve, static_argnums=3)(Ah, nine, rhs, 8)
    assert bool(jnp.all(jnp.isnan(jitted[:8])))
    everything = OP.gather_solve(Ah, jnp.ones(op.n, dtype=bool), rhs, buf=8)
    assert int(jnp.isnan(everything).sum()) == 8 and int((everything != 0).sum()) == 8
    eight = jnp.zeros(op.n, dtype=bool).at[:8].set(True)
    exact = jnp.linalg.solve(Ah[:8, :8], rhs[:8])
    assert float(jnp.max(jnp.abs(OP.gather_solve(Ah, eight, rhs, buf=8)[:8] - exact))) < 1e-12
    full = OP.gather_solve(Ah, jnp.ones(op.n, dtype=bool), rhs, buf=op.n)
    assert float(jnp.max(jnp.abs(full - jnp.linalg.solve(Ah, rhs)))) < 1e-10


def test_assembly_refuses_a_non_symmetric_physical_stencil_instead_of_symmetrising_it(monkeypatch):
    """The audit's one missed mutation (M7s): a one-sided ``[-1, 2, -1] / h``
    stencil is first order, but ``0.5 (A + A^T)`` turned it into a consistent
    second-order one and the MMS order gate passed (observed order 2.04).
    The symmetry check runs before the symmetrisation now, so the stencil
    is refused by name at construction, in 1-D and through the tensor sum."""
    def one_sided(side, h):
        idx = np.arange(side)
        S = np.zeros((side, side))
        S[idx, idx] = -1.0 / h
        S[idx, (idx + 1) % side] += 2.0 / h
        S[idx, (idx + 2) % side] += -1.0 / h
        return S

    monkeypatch.setattr(OP, "_stiffness_periodic", one_sided)
    with pytest.raises(ValueError, match="not symmetric"):
        OP.assemble_operator(4, 2, order=4, dim=1)
    with pytest.raises(ValueError, match="not symmetric"):
        OP.assemble_operator(2, 2, order=4, dim=2)


@pytest.mark.parametrize("kw", [
    dict(n_levels=6, n_coarse=2, dim=1), dict(n_levels=3, n_coarse=2, dim=2),
    dict(n_levels=2, n_coarse=2, dim=3), dict(n_levels=4, n_coarse=2, dim=1, order=6),
    dict(n_levels=5, n_coarse=2, dim=1, boundary="dirichlet"),
    dict(n_levels=2, n_coarse=2, dim=2, boundary="dirichlet"),
], ids=lambda kw: "-".join(f"{k}={v}" for k, v in kw.items()))
def test_the_correct_assembly_is_symmetric_two_orders_inside_the_tolerance_the_check_uses(kw):
    """A check is only safe if what it guards is far from its threshold:
    rebuild the *un-symmetrised* Galerkin product the way ``assemble_operator``
    does and measure it.  1e-17 .. 1.1e-16 over 15 configurations when the
    tolerance was set; ``SYMMETRY_TOL = 1e-12`` leaves four orders, and this
    pins at least two."""
    op = OP.assemble_operator(**{k: v for k, v in kw.items()}, mass=1.0)
    boundary = kw.get("boundary", "periodic")
    A_phys = OP._physical_operator(op.side, op.dim, op.h, 1.0, boundary)
    Wn = np.asarray(op.Wn, dtype=np.float64)
    raw = Wn.T @ A_phys @ Wn
    asym = np.max(np.abs(raw - raw.T)) / np.max(np.abs(raw))
    assert asym < 1e-2 * OP.SYMMETRY_TOL, asym
    assert np.array_equal(np.asarray(op.A), np.asarray(op.A).T)


def test_assembly_is_legal_inside_a_jit_trace_and_gives_the_eager_operator():
    """Every input is a static setting, so the assembly runs on the host under
    ``ensure_compile_time_eval``; the returned arrays are constants of the
    trace and equal the eager ones.  ``levels`` and ``diagonal`` are host
    arrays, which is what lets the node count its seed inside a trace."""
    eager = OP.assemble_operator(3, 2, order=4, dim=2)

    @jax.jit
    def f(v):
        op = OP.assemble_operator(3, 2, order=4, dim=2)
        return op.A @ v, op.A_sparse @ v, op.Wn @ v

    v = jnp.asarray(np.random.default_rng(12).standard_normal(eager.n))
    a, a_s, w = f(v)
    assert float(jnp.max(jnp.abs(a - eager.A @ v))) < 1e-12
    assert float(jnp.max(jnp.abs(a_s - eager.A_sparse @ v))) < 1e-12
    assert float(jnp.max(jnp.abs(w - eager.Wn @ v))) < 1e-12
    assert isinstance(eager.levels, np.ndarray) and isinstance(eager.diagonal, np.ndarray)
    assert np.allclose(eager.diagonal, np.diag(np.asarray(eager.A)))


# ---------------------------------------------------------------------------
# cdd
# ---------------------------------------------------------------------------

def _cdd_setup(nl=6, nc=2, sigma=0.06):
    op = OP.assemble_operator(nl, nc, order=4, dim=1)
    Ah, D = _scaled(op)
    lev = np.asarray(op.levels)
    coarse = jnp.asarray(lev == lev.min())
    x = np.arange(op.side) / op.side
    f = jnp.exp(-((jnp.asarray(x) - 0.42) / sigma) ** 2)
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


def test_cdd_reports_its_iteration_count_and_the_bound_is_the_exit_for_a_budget_near_half_the_basis():
    """``K = 8`` stops on the budget in a few iterations.  ``K = 64`` on the
    128-point basis (the node's own source, sigma 0.10) stops on
    ``MAX_OUTER = 30`` short of the budget -- 54 measured, the count not
    pinned across jaxlib lanes -- and 200 iterations do reach it.  The
    Doerfler step marks a fixed fraction of the *remaining* residual, so
    the steps shrink once the source is resolved.  ``cdd_select`` is the
    same loop without the count."""
    op, Ah, D, coarse, b_hat = _cdd_setup(sigma=0.10)
    apply = lambda v: Ah @ v
    solve = lambda K: (lambda m, r: OP.gather_solve(Ah, m, r, K))
    mask, c, n8 = CDD.cdd_select_with_iterations(apply, solve(8), b_hat, coarse, 8)
    assert int(mask.sum()) == 8 and 0 < int(n8) < CDD.MAX_OUTER
    m2, c2 = CDD.cdd_select(apply, solve(8), b_hat, coarse, 8)
    assert bool(jnp.array_equal(mask, m2)) and bool(jnp.array_equal(c, c2))
    stalled, _, n64 = CDD.cdd_select_with_iterations(apply, solve(64), b_hat, coarse, 64)
    assert int(n64) == CDD.MAX_OUTER == 30
    assert int(coarse.sum()) < int(stalled.sum()) < 64
    reached, _, n200 = CDD.cdd_select_with_iterations(apply, solve(64), b_hat, coarse, 64, max_outer=200)
    assert int(reached.sum()) == 64 and 30 < int(n200) <= 200


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


# ---------------------------------------------------------------------------
# conditioning: the estimate the node's dtype guard reads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    dict(n_levels=6, mass=1.0), dict(n_levels=6, mass=1e-4), dict(n_levels=6, mass=1e-8),
    dict(n_levels=4, n_coarse=3, order=6, mass=1e-6),
    dict(n_levels=3, dim=2, mass=1e-6),
    dict(n_levels=5, boundary="dirichlet", mass=1e-8),
    dict(n_levels=6, mass=1e-3, preconditioner="dk"),
])
def test_the_condition_estimate_is_a_tight_lower_bound_on_the_exact_spectrum(kw):
    """``assemble_operator(preconditioner=...)`` estimates kappa(D^-1 A D^-1)
    within 3% of ``eigvalsh`` (0.03% measured) and not above it: both
    extremes are approached from inside the spectrum, so a refusal made on
    the estimate is never spurious.  "Not above" is up to 1e-5 relative:
    at small mass the closed-form constant-mode quotient describes the
    exact operator and ``eigvalsh`` the assembled one, and the two differ
    by the assembly's rounding (4e-7 here; 1.9% at mass 1e-10, a mass the
    node refuses)."""
    kw = dict(kw)
    kind = kw.pop("preconditioner", "hybrid")
    nl, nc = kw.pop("n_levels"), kw.pop("n_coarse", 2)
    op = OP.assemble_operator(nl, nc, dtype=jnp.float64, preconditioner=kind, **kw)
    D = PC._diagonal_scaling_np(op.diagonal, op.levels, kind)
    exact = _kappa(np.asarray(op.A) / D[:, None] / D[None, :])
    est = op.condition_number
    assert est is not None
    assert est <= exact * (1.0 + 1e-5), (est, exact)
    assert est >= 0.97 * exact, (est, exact)


@pytest.mark.parametrize("side,dim,boundary", [
    (16, 1, "periodic"), (15, 1, "dirichlet"), (8, 2, "periodic"), (7, 2, "dirichlet"),
    (4, 3, "periodic"),
])
@pytest.mark.parametrize("mass", [1.0, 1e-3])
def test_the_physical_condition_number_is_the_closed_form_of_the_grid_spectrum(side, dim, boundary, mass):
    h = 1.0 / side if boundary == "periodic" else 1.0 / (side + 1)
    ev = np.linalg.eigvalsh(OP._physical_operator(side, dim, h, mass, boundary))
    assert OP.physical_condition_number(side, dim, mass, boundary) == pytest.approx(
        ev[-1] / ev[0], rel=1e-9)


def test_without_a_preconditioner_the_assembly_reports_no_condition_number():
    assert OP.assemble_operator(4, 2).condition_number is None


def test_on_the_periodic_basis_the_condition_number_grows_like_one_over_the_mass():
    """The smallest eigenvalue of the periodic operator belongs to the
    constant function and is proportional to ``mass``; the Dirichlet
    operator has no such mode and does not care."""
    per = [OP.assemble_operator(6, 2, mass=m, preconditioner="hybrid").condition_number
           for m in (1e-2, 1e-4)]
    assert per[0] is not None and per[1] is not None
    assert 95.0 < per[1] / per[0] < 105.0, per
    dir_ = [OP.assemble_operator(5, 2, mass=m, boundary="dirichlet",
                                 preconditioner="hybrid").condition_number for m in (1e-2, 1e-8)]
    assert dir_[0] is not None and dir_[1] is not None
    assert abs(dir_[1] / dir_[0] - 1.0) < 0.01, dir_


def test_the_constant_mode_bound_supplies_the_small_eigenvalue_a_short_lanczos_run_misses():
    """At small mass the constant function's eigenvalue is isolated far
    below the rest; a few Lanczos steps from a random start do not resolve
    it, and the closed-form Rayleigh quotient does."""
    op = OP.assemble_operator(6, 2, mass=1e-8, dtype=jnp.float64)
    D = PC._diagonal_scaling_np(op.diagonal, op.levels, "hybrid")
    Ah = np.asarray(op.A) / D[:, None] / D[None, :]
    exact = _kappa(Ah)
    q = OP._constant_mode_rayleigh(np.asarray(op.Wn), D, op.levels, 1e-8, op.h, 1)
    assert q is not None
    short = OP.condition_estimate(Ah, steps=4)
    assert short < 0.5 * exact
    assert abs(OP.condition_estimate(Ah, steps=4, rayleigh_bound=q) / exact - 1.0) < 0.03
    wall = OP.assemble_operator(4, 2, mass=1e-8, boundary="dirichlet", dtype=jnp.float64)
    Dw = PC._diagonal_scaling_np(wall.diagonal, wall.levels, "hybrid")
    assert OP._constant_mode_rayleigh(np.asarray(wall.Wn), Dw, wall.levels, 1e-8, wall.h, 1) is None


def test_an_operator_that_is_not_positive_definite_estimates_an_infinite_condition_number():
    A = np.diag([1.0, 2.0, -1e-3])
    assert OP.condition_estimate(A) == float("inf")
    assert OP.condition_estimate(np.eye(3), rayleigh_bound=0.0) == float("inf")


# ---------------------------------------------------------------------------
# cdd: the marking step reads no difference below the rounding floor
# ---------------------------------------------------------------------------

def _grow(r, *, cap, mask=None, tol=1e-9):
    r = jnp.asarray(r, dtype=jnp.float64)
    mask = jnp.zeros(r.shape[0], bool) if mask is None else jnp.asarray(mask)
    return np.asarray(CDD._doerfler_grow(mask, r, CDD.THETA_D, cap, jnp.float64(tol)))


_TIED = np.array([0.0, 0.3, 0.3, 0.0, 0.3, 0.3, 0.0, 0.3, 0.3])   # six equal, at 1 2 4 5 7 8


@pytest.mark.parametrize("bump", [0.0, 1e-15, -1e-15])
def test_a_tie_at_the_cap_is_broken_by_basis_index_not_by_the_last_bits(bump):
    """Six equal residuals, so the Doerfler count is 2 (the first two carry
    ``theta_D**2 = 0.25`` of the squared mass).  With room for one, or for
    the two, the lowest indices are taken -- whichever member the last
    bits make larger.  The old marking took ``argsort``'s order, i.e. the
    rounding's."""
    r = _TIED.copy()
    r[8] += bump                     # the highest index, larger or smaller by 1 ulp-ish
    r[1] -= bump
    assert np.flatnonzero(_grow(r, cap=1)).tolist() == [1]
    assert np.flatnonzero(_grow(r, cap=2)).tolist() == [1, 2]
    assert np.flatnonzero(_grow(r, cap=9)).tolist() == [1, 2]


def test_a_magnitude_gap_wider_than_the_tolerance_is_respected_over_index_order():
    r = _TIED.copy()
    r[8] += 1e-6                     # a real gap: 1000x the tolerance below
    assert np.flatnonzero(_grow(r, cap=1, tol=1e-9)).tolist() == [8]
    assert np.flatnonzero(_grow(r, cap=2, tol=1e-9)).tolist() == [1, 8]


def test_a_doerfler_crossing_inside_a_tie_band_takes_the_lowest_index():
    """Three equal residuals: the Doerfler count is 1 and the band holds
    all three; index order decides, not the sort."""
    r = np.array([0.0, 0.3, 0.0, 0.3, 0.0, 0.3 + 1e-15])
    assert np.flatnonzero(_grow(r, cap=6)).tolist() == [1]


def test_residuals_at_or_below_the_rounding_floor_are_never_marked():
    r = np.array([0.0, 1e-9, 5e-10, 1e-12])
    assert not _grow(r, cap=4, tol=1e-9).any()
    r2 = np.array([0.0, 1e-9, 2e-9, 1e-12])
    assert _grow(r2, cap=4, tol=1e-9).tolist() == [False, False, True, False]


def test_the_rounding_floor_is_the_stated_multiple_of_eps_and_scales_with_b_and_c():
    b = jnp.asarray([0.5, -2.0], dtype=jnp.float32)
    c = jnp.asarray([3.0, -1.0], dtype=jnp.float32)
    got = float(CDD.rounding_floor(b, c))
    want = CDD.NOISE_FACTOR * float(jnp.finfo(jnp.float32).eps) * (2.0 + 3.0)
    assert got == pytest.approx(want, rel=1e-6)
    assert float(CDD.rounding_floor(b.astype(jnp.float64), c.astype(jnp.float64))) < 1e-12


def test_cdd_stops_once_nothing_above_the_floor_is_left_instead_of_spinning_to_the_bound():
    """A right-hand side whose residual vanishes after the seed solve (it
    lies in the coarse span) leaves nothing to mark: the loop exits after
    one marking attempt, not after ``MAX_OUTER`` solves."""
    op, Ah, D, coarse, _ = _cdd_setup(nl=4)
    K = op.n // 2
    b = jnp.where(coarse, 1.0, 0.0)
    b = Ah @ OP.gather_solve(Ah, coarse, b, K)     # exactly representable on the seed
    mask, _, n_outer = CDD.cdd_select_with_iterations(
        lambda v: Ah @ v, lambda m, r: OP.gather_solve(Ah, m, r, K), b, coarse, K,
    )
    assert bool(jnp.array_equal(mask, coarse)) and int(n_outer) == 1
