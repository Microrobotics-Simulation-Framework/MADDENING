"""Each flag in ``coupling_diagnostics()`` reads the condition it documents.

The report's booleans are derived, on the host, from numbers the step
stored.  A derivation that dropped one of its conditions would still
produce a plausible ``True`` on every fixture where the dropped
condition happens to hold -- and a mutation audit found three such
drops that no test in the suite could see:

* ``converged`` computed from the raw residual (the pre-0.4.0 rule)
  instead of from the error estimate;
* ``spectral_usable`` ignoring whether the Krylov space had settled;
* ``gradient_bound_usable`` ignoring ``spectral_usable``.

Each test below builds the one configuration where the condition in
question is the *deciding* one, and says so in its premise asserts.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import SPECTRAL_KRYLOV_STEPS
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode


class _Linear(SimulationNode):
    """``x <- M @ u + c`` on a vector field; ``c`` is a parameter, so it is probed."""

    def __init__(self, name, M, c):
        M = np.asarray(M, np.float32)
        super().__init__(name=name, timestep=1.0, c=jnp.asarray(c, jnp.float32))
        self._M = jnp.asarray(M)
        self._n = M.shape[0]

    def initial_state(self):
        return {"x": jnp.zeros(self._n, jnp.float32)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._n,), dtype=jnp.float32,
                                       default=jnp.zeros(self._n, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": self._M @ boundary_inputs["u"] + p["c"]}


def _relay_pair(M, c, **group_kw):
    """``a <- M b + c``, ``b <- a``: a Gauss-Seidel Jacobian of rank ``rank(M)``."""
    n = np.asarray(M).shape[0]
    gm = GraphManager()
    gm.add_node(_Linear("a", M, c))
    gm.add_node(_Linear("b", np.eye(n), np.zeros(n)))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(["a", "b"], diagnostics=True, **group_kw)
    gm.compile()
    return gm


def test_converged_reads_the_error_estimate_not_the_residual():
    """Residual under the tolerance, estimate over it: not converged.

    ``x_a <- 0.9 x_b + 1`` at ``tolerance=0.02`` for 30 passes: the last
    pass moved 6.2e-3 of the state, but at a contraction of 0.9 the
    distance still to go is ten times that, and the estimate (6.0e-2)
    says so.  Before 0.4.0 the flag was the residual test and read
    ``True`` here; a report that reverted to it passed every other test.
    """
    gm = _relay_pair([[0.9]], [1.0], max_iterations=30, tolerance=0.02)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert d["residual"] <= 0.02 < d["error_estimate"], (
        f"fixture premise: residual {d['residual']:.3e} under and estimate "
        f"{d['error_estimate']:.3e} over the tolerance"
    )
    assert d["ratio_usable"] is True
    assert d["iterations"] == 30
    assert d["converged"] is False


def _spread_contraction(n, seed, radius):
    """A symmetric ``n x n`` map with ``n`` distinct eigenvalues in ``[-radius, radius]``."""
    rng = np.random.default_rng(seed)
    lam = np.linspace(-radius, radius, n)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return (Q * lam) @ Q.T, rng.uniform(-1.0, 1.0, n)


def test_spectral_usable_is_false_where_the_krylov_space_did_not_settle():
    """Twelve independent interface scalars against eight Arnoldi steps.

    The bound is finite -- the Ritz radius plus its margin stays below
    one -- so finiteness alone would call it usable.  The space has
    *not* settled (the Arnoldi residual is far above 5% of the gap), the
    radius is an estimate from below, and the flag has to say so.
    """
    M, c = _spread_contraction(12, seed=4, radius=0.4)
    gm = _relay_pair(M, c, max_iterations=40, tolerance=1e-4)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert 12 > SPECTRAL_KRYLOV_STEPS, "fixture premise"
    assert math.isfinite(d["spectral_error_bound"]), (
        f"fixture premise: the bound is finite, so only the settled test "
        f"can clear the flag: {d}"
    )
    assert d["spectral_usable"] is False, d
    assert d["gradient_bound_usable"] is False, d


def test_gradient_bound_usable_requires_a_settled_spectrum():
    """Rank exactly eight: the range basis captures it, the Krylov space does not settle.

    Eight random images span a rank-eight range, so the gradient bound
    is computed and finite.  But the Krylov space grown from one start
    vector needs ``rank + 1`` steps to become invariant, and it has
    eight, so the spectrum is not settled -- and the gradient bound's
    distance and resolvent factor come from that spectrum.  The flag
    has to follow ``spectral_usable`` here, not only its own finiteness.
    """
    M, c = _spread_contraction(SPECTRAL_KRYLOV_STEPS, seed=7, radius=0.4)
    gm = _relay_pair(M, c, max_iterations=40, tolerance=1e-4)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert math.isfinite(d["gradient_relative_error_bound"]), (
        f"fixture premise: the gradient bound is computed and finite: {d}"
    )
    assert d["spectral_usable"] is False, (
        f"fixture premise: the spectrum is not settled: {d}"
    )
    assert d["gradient_bound_usable"] is False, d


@pytest.mark.parametrize("n", (2, 4))
def test_a_resolved_spectrum_is_usable(n):
    """The control for the two above: a small group settles and both flags hold."""
    M, c = _spread_contraction(n, seed=3, radius=0.4)
    gm = _relay_pair(M, c, max_iterations=40, tolerance=1e-4)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assert d["spectral_usable"] is True, d
    assert d["gradient_bound_usable"] is True, d
