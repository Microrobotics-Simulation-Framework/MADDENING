"""The fitters' rank guard decomposes a long Jacobian without a square of its rows.

``fit_lm`` ends by asking its Jacobian, over all ``m`` residual rows, which
directions it resolves (``_curvature_rank`` and the hold's
``_gauss_newton_flatness``, through ``_rotation_by_singular_values``).  That
helper took the *full* singular value decomposition to get a square rotation
of the ``n`` parameter directions, and the full decomposition of an ``m x n``
matrix also forms an ``m x m`` left factor nobody reads: 6.4 GB at 20,000
rows, and a well-posed two-parameter fit over 100,000 rows raised
``MemoryError`` after it had converged.  It takes the reduced decomposition
now.

Nothing here allocates what the old code did at those sizes.  The sizes are
small (the old code's factor is 32 MB to 128 MB at them) and the assertions
are on what is **allocated or returned**, which is the same number on every
machine: the peak of the traced allocations (numpy reports its array buffers
to ``tracemalloc``), and the sizes of the factors ``numpy.linalg.svd`` hands
back.  Neither is a timing or a resident-set reading, and neither depends on
what else the test session holds.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import tracemalloc
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening import sysid
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode


@pytest.fixture
def svd_calls(monkeypatch):
    """Every ``numpy.linalg.svd`` call of the test: the input's shape and
    the number of entries of the largest array it returned."""
    original = np.linalg.svd
    calls = []

    def recording(a, *args, **kwargs):
        out = original(a, *args, **kwargs)
        arrays = out if isinstance(out, tuple) else (out,)
        calls.append((np.shape(a), max(int(np.size(x)) for x in arrays)))
        return out

    monkeypatch.setattr(np.linalg, "svd", recording)
    return calls


def test_the_rotation_returns_no_factor_larger_than_its_input(svd_calls):
    rng = np.random.default_rng(0)
    m, k = 500, 3
    W, s = sysid._rotation_by_singular_values(rng.standard_normal((m, k)), k)
    assert W.shape == (k, k) and s.shape == (k,)
    assert svd_calls and svd_calls[0][0] == (m, k)
    # The full decomposition's left factor is m * m = 250,000 entries.
    assert max(size for _, size in svd_calls) <= m * k, svd_calls


def test_the_rank_guard_over_a_long_jacobian_allocates_no_square_of_its_rows():
    """4,000 rows: the full decomposition's left factor alone is 128 MB;
    everything the guard needs is a few copies of ``J`` (64 KB each)."""
    m = 4000
    rng = np.random.default_rng(1)
    J = rng.standard_normal((m, 2)).astype(np.float32)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        rank = sysid._curvature_rank(J, np.array([1.0, 1.0]))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert rank == 2
    assert peak < 4 * 2 ** 20, f"peak of traced allocations: {peak / 2 ** 20:.1f} MiB"


class _TwoConstants(SimulationNode):
    """A node whose two constants a residual below reads; it does nothing."""

    def __init__(self, name, a, b):
        super().__init__(name=name, timestep=0.01, a=a, b=b)

    def initial_state(self):
        return {"x": jnp.float32(0.0)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": state["x"]}


def test_fit_lm_over_a_long_residual_reaches_its_guard_with_small_factors(svd_calls):
    """A well-posed two-parameter fit over 2,000 rows, end to end: the
    guard is reached (a decomposition of a ``2000 x 2`` matrix is seen),
    and no decomposition of the run returns more than ``m * n`` entries
    (the old route returned ``m * m``, 4,000,000)."""
    m = 2000
    gm = GraphManager()
    gm.add_node(_TwoConstants("s", 1.0, 2.0))
    gm.compile()
    rng = np.random.default_rng(2)
    u1 = jnp.asarray(rng.standard_normal(m), jnp.float32)
    u2 = jnp.asarray(rng.standard_normal(m), jnp.float32)
    y = 1.5 * u1 + 2.5 * u2

    def residual(p):
        return p["nodes"]["s"]["a"] * u1 + p["nodes"]["s"]["b"] * u2 - y

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = sysid.fit_lm(gm, residual, n_iter=10, notify_every=0)
    assert result.converged and result.excited_rank == 2
    np.testing.assert_allclose(
        [float(result.params["nodes"]["s"][k]) for k in ("a", "b")], [1.5, 2.5], rtol=1e-5)
    over_the_rows = [(shape, size) for shape, size in svd_calls if shape[0] == m]
    assert over_the_rows, f"the guard's decomposition was not seen: {svd_calls}"
    assert max(size for _, size in svd_calls) <= m * 2, svd_calls


def test_fewer_rows_than_directions_leaves_the_missing_ones_at_exactly_zero():
    M = np.array([[3.0, 0.0, 4.0]])
    W, s = sysid._rotation_by_singular_values(M, 3)
    assert W.shape == (3, 3) and s.tolist() == [5.0, 0.0, 0.0]
    np.testing.assert_allclose(W.T @ W, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(np.abs(W[:, 0]), [0.6, 0.0, 0.8], atol=1e-14)
    np.testing.assert_allclose(M @ W[:, 1:], 0.0, atol=1e-14)


@pytest.mark.parametrize("m, k", [(40, 4), (7, 7), (4, 4), (3, 5), (1, 4), (900, 6)])
def test_the_rotation_is_the_full_decompositions_to_rounding(m, k):
    """Against the decomposition it replaces: the same singular values, and
    the same right singular vectors up to the sign each is defined to (the
    values of a random matrix are distinct); where ``m < k`` the missing
    directions are a basis of the same null space, which is all a caller
    reads of them (the span of the directions it calls flat)."""
    rng = np.random.default_rng(100 * m + k)
    M = rng.standard_normal((m, k))
    _, s_full, wt_full = np.linalg.svd(M, full_matrices=True)
    W, s = sysid._rotation_by_singular_values(M, k)
    r = min(m, k)
    np.testing.assert_allclose(s[:r], s_full, rtol=1e-12, atol=1e-13 * s_full[0])
    assert np.all(s[r:] == 0.0)
    overlap = np.abs(wt_full[:r] @ W[:, :r])
    np.testing.assert_allclose(overlap, np.eye(r), atol=1e-9)
    np.testing.assert_allclose(W[:, r:] @ W[:, r:].T, wt_full[r:].T @ wt_full[r:], atol=1e-12)
    np.testing.assert_allclose(W.T @ W, np.eye(k), atol=1e-13)
