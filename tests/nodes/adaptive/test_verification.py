"""verify_node battery for the adaptive toys and the MADD-VER-004 benchmark.

MADD-VER-004 compares the frozen-active-set solution of
``-u'' + u = f`` on ``(0, 1)``, ``u(0) = u(1) = 0``, against the exact
Green's-function solution

    u(x) = int_0^1 G(x, s) f(s) ds,   G(x, s) = sinh(x_<) sinh(1 - x_>) / sinh(1)

(the classical two-point boundary-value Green's function, e.g. LeVeque
2007 ch. 2): with every mode active the sine-Galerkin solution must
match it, and the top-K adaptive solution must converge to it as the
budget K grows.
"""

from __future__ import annotations

import numpy as np
import pytest

from maddening.core.compliance.validation import (
    BenchmarkType, _BENCHMARK_REGISTRY, verification_benchmark,
)
from maddening.testing.verification import DEFAULT_CHECKS, verify_node

from tests.nodes.adaptive._toys import MaskedDenseNode, PoissonSineTopKNode

SIGMA = 0.04
THETA = 0.42


def _greens_reference(x: np.ndarray, theta: float, sigma: float, n_quad: int = 20001) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n_quad)
    f = np.exp(-((s - theta) / sigma) ** 2)
    lo = np.minimum.outer(x, s)
    hi = np.maximum.outer(x, s)
    G = np.sinh(lo) * np.sinh(1.0 - hi) / np.sinh(1.0)
    return np.trapezoid(G * f, s, axis=1)


@pytest.mark.parametrize("make", [
    lambda: PoissonSineTopKNode(n=64, k=16),
    lambda: MaskedDenseNode(n=16, k=4, blindness_gate=False),
], ids=["sine_topk", "masked_dense"])
def test_verify_node_battery_passes(make):
    node = make()
    results = verify_node(
        node, bounds={"c": (-1.0, 1.0)}, dtype=np.float64,
        max_examples=25, derandomize=True,
    )
    assert set(results) == set(DEFAULT_CHECKS)
    bad = [str(r) for r in results.values() if not r.passed]
    assert not bad, "\n".join(bad)
    # The params contract is exercised, not skipped: theta is trainable
    # and update reads it from the injected pytree.
    assert all(results[k].status == "PASS" for k in
               ("params_consistent", "params_gradient_finite", "params_effective"))


@verification_benchmark(
    benchmark_id="MADD-VER-004",
    description=(
        "AdaptiveNode frozen-active-set solve of -u'' + u = Gaussian on "
        "(0, 1) with Dirichlet ends vs the exact Green's-function solution: "
        "full active set reproduces it; top-K error decreases with K"
    ),
    node_type="AdaptiveNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    acceptance_criteria=(
        "Full-basis (K = n = 256) L2 relative error < 1e-4 on the grid and "
        "sensor error < 1e-6; top-K sensor error strictly decreasing over "
        "K in (4, 8, 16, 32) and < 1e-4 at K = 32"
    ),
    references=("LeVeque2007: Green's function for the 1-D two-point BVP",),
)
def test_adaptive_solve_matches_greens_function_and_converges_in_k():
    n = 256
    full = PoissonSineTopKNode(n=n, k=n, theta=THETA, sigma=SIGMA)
    s = full.initial_state()
    x = np.asarray(full._x)
    u_ref = _greens_reference(x, THETA, SIGMA)
    u_full = np.asarray(full.field(s["c"]))
    l2 = np.sqrt(np.sum((u_full - u_ref) ** 2) / np.sum(u_ref ** 2))
    assert l2 < 1e-4, f"full-basis L2 relative error {l2:.3e}"

    x_s = 1.0 / 3.0
    j_ref = float(_greens_reference(np.array([x_s]), THETA, SIGMA)[0])
    j_full = float(full.objective(s, full.params))
    assert abs(j_full - j_ref) < 1e-6, (j_full, j_ref)

    errors = []
    for k in (4, 8, 16, 32):
        node = PoissonSineTopKNode(n=n, k=k, theta=THETA, sigma=SIGMA, blindness_gate=False)
        errors.append(abs(float(node.objective(node.initial_state(), node.params)) - j_ref))
    assert all(a > b for a, b in zip(errors, errors[1:])), errors
    assert errors[-1] < 1e-4, errors


def test_benchmark_is_registered():
    bm = _BENCHMARK_REGISTRY["MADD-VER-004"]
    assert bm.node_type == "AdaptiveNode"
    assert bm.benchmark_type == BenchmarkType.ANALYTICAL
