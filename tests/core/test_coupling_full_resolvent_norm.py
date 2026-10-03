"""``_full_resolvent_norm`` is ``||(I - J)^{-1}||_2`` over the whole space, against a dense inverse.

The Newton-Kantorovich check behind ``gradient_bound_usable`` takes
``beta = ||(I - J(x_k))^{-1}||``.  Until 0.4.0's round-5 fix it took the
Arnoldi factor, the resolvent restricted to the Krylov space of the start
vector and the residual, which can be several times smaller (MADD-ANO-142).
The full norm comes from the range basis: with ``range(J)`` inside
``span(U)`` and ``B = U^T J`` the resolvent is ``I + U (I - M)^{-1} B``,
exact from a small SVD.  Its compression to ``span(U)`` alone -- what
``inv(I - M)`` gives -- is the whole operator only where ``U`` is square;
on a non-normal map whose range and co-range differ it falls short (a
nilpotent chain: 1.83 where the full norm is 7.12), which is what these
cases pin, against ``numpy.linalg.inv`` in float64.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import SPECTRAL_KRYLOV_STEPS, jacobian_range_basis
from maddening.core.graph_manager import _full_resolvent_norm


def _low_rank(n, rank, seed, scale):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, rank))
    b = rng.standard_normal((rank, n))
    J = a @ b
    return scale * J / np.max(np.abs(np.linalg.eigvals(J)))


def _chain(n, gain, last):
    """``x_i <- gain * x_{i+1}`` on the first ``k`` entries, the last link ``last``: nilpotent.

    Its range is the first ``k`` entries and its co-range the next ``k``,
    so the compression to the range misses the last link's column.
    """
    k = SPECTRAL_KRYLOV_STEPS
    J = np.zeros((n, n))
    for i in range(k):
        J[i, i + 1] = gain
    J[k - 1, k] = last
    return J


def _ring(n, gains):
    """The rank-one ring of the round-5 audit, padded with uncoupled entries to ``n``."""
    J = np.zeros((n, n))
    J[0, 2], J[1, 0], J[2, 1] = gains
    return J


CASES = {
    "square-dense": _low_rank(5, 5, 0, 0.8),
    "wide-rank-one": _low_rank(12, 1, 1, 0.9),
    "wide-rank-three": _low_rank(30, 3, 2, 0.95),
    "wide-rank-eight": _low_rank(20, SPECTRAL_KRYLOV_STEPS, 3, 0.7),
    "wide-symmetric": (lambda J: 0.5 * (J + J.T))(_low_rank(16, 4, 4, 0.8)),
    "wide-chain": _chain(14, 0.5, 6.0),
    "wide-ring": _ring(12, (-12.137987, 0.0320999, -2.2632)),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_full_resolvent_norm_matches_the_dense_inverse(case):
    J = CASES[case]
    n = J.shape[0]
    Jf = jnp.asarray(J, jnp.float32)
    U, M, captured = jacobian_range_basis(lambda v: Jf @ v, n)
    assert bool(captured), case
    got = float(_full_resolvent_norm(U, M, lambda: (lambda u: Jf.T @ u)))
    want = float(np.linalg.norm(np.linalg.inv(np.eye(n) - J), ord=2))
    assert got == pytest.approx(want, rel=2e-3), (case, got, want)


def test_the_compression_alone_falls_short_where_range_and_co_range_differ():
    """The fixture's premise: on the chain ``inv(I - M)`` is not the operator's norm."""
    J = CASES["wide-chain"]
    n = J.shape[0]
    Jf = jnp.asarray(J, jnp.float32)
    U, M, _ = jacobian_range_basis(lambda v: Jf @ v, n)
    compressed = float(jnp.linalg.norm(jnp.linalg.inv(jnp.eye(U.shape[1]) - M), ord=2))
    want = float(np.linalg.norm(np.linalg.inv(np.eye(n) - J), ord=2))
    assert compressed < want / 3, (compressed, want)


def test_without_a_transpose_a_wide_basis_has_no_norm_and_a_square_one_needs_none():
    J = CASES["wide-rank-one"]
    Jf = jnp.asarray(J, jnp.float32)
    U, M, _ = jacobian_range_basis(lambda v: Jf @ v, J.shape[0])
    assert _full_resolvent_norm(U, M, lambda: None) is None
    Js = jnp.asarray(CASES["square-dense"], jnp.float32)
    Us, Ms, _ = jacobian_range_basis(lambda v: Js @ v, 5)

    def never():
        raise AssertionError("a square basis needs no transpose")

    want = float(np.linalg.norm(np.linalg.inv(np.eye(5) - CASES["square-dense"]), ord=2))
    assert float(_full_resolvent_norm(Us, Ms, never)) == pytest.approx(want, rel=2e-3)
