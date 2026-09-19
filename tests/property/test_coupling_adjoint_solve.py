"""``jax.grad`` on the default coupling path does not raise.

The coupling audit (2026-09-19, ``audit_040_final/coupling``) found
``jax.grad`` through ``solver="ift", linear_solver="gmres"`` -- both
defaults -- raising ``_EquinoxRuntimeError: iterative breakdown`` on a
four-DOF two-node cycle.  ``jax.jvp`` on the same graph was fine and
``linear_solver="dense"`` was fine: it was the transpose (adjoint)
solve, and it was ill-conditioning meeting an unreachable float32
tolerance rather than a Krylov breakdown.  See
``graph_manager._ift_linear_solve`` for the mechanism and
``tests/core/test_coupling_ift_lineax.py`` for the pinned example.

Why this is a property test and not three more examples: the failure
was **non-monotone in stiffness** (0.998 and 0.999 raised, 0.9995 did
not) because whether the float32 iterate lands inside a tolerance it
cannot reach depends on round-off in the cotangent.  A fixed list of
rates therefore proves very little -- the next release's spectrum, or
the next user's loss function, draws from the same lottery.  So the
spectrum *and* the cotangent are both generated here, and the
assertion is the one a user actually relies on: differentiating a
coupled graph on the library's own defaults returns a finite number.

The map is affine with a known fixed point, so the expected gradient
is in closed form and the assertion can be on the value rather than on
the absence of an exception; a fallback that quietly returned the
unconverged Krylov iterate would satisfy the weaker statement.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

from tests.conftest import EXAMPLES_COSTLY


def _build(rho: tuple[float, ...], c: tuple[float, ...]) -> GraphManager:
    """A two-node cycle carrying a diagonal contraction ``diag(rho)``.

    ``a`` maps its boundary input ``u`` to ``rho * u + gain * c`` and
    ``b`` relays it back, so one Gauss-Seidel sweep of the group is the
    affine map ``x -> rho * x + gain * c`` and the fixed point is
    ``gain * c / (1 - rho)`` per mode.  The flat group state is
    ``2 * len(rho)`` floats.
    """
    n = len(rho)
    rho_a = jnp.asarray(rho)
    c_a = jnp.asarray(c)

    class _Contract(SimulationNode):
        def __init__(self, name, dt, gain=1.0):
            super().__init__(name, dt, gain=gain)

        def initial_state(self):
            return {"x": jnp.zeros(n)}

        def boundary_input_spec(self):
            return {"u": BoundaryInputSpec(shape=(n,), description="u")}

        def update(self, state, bi, dt, *, params=None):
            p = self.params if params is None else {**self.params, **params}
            return {"x": rho_a * bi.get("u", jnp.zeros(n)) + p["gain"] * c_a}

    class _Relay(SimulationNode):
        def initial_state(self):
            return {"y": jnp.zeros(n)}

        def boundary_input_spec(self):
            return {"v": BoundaryInputSpec(shape=(n,), description="v")}

        def update(self, state, bi, dt, *, params=None):
            return {"y": bi.get("v", jnp.zeros(n))}

    gm = GraphManager()
    gm.add_node(_Contract("a", 0.01))
    gm.add_node(_Relay("b", 0.01))
    gm.add_edge("a", "b", "x", "v")
    gm.add_edge("b", "a", "y", "u")
    # No ``solver=`` / ``linear_solver=`` here on purpose: the defaults
    # are the surface under test.
    gm.add_coupling_group(
        ["a", "b"], max_iterations=80, tolerance=1e-6, diagnostics=True,
    )
    gm.compile()
    return gm


#: The stiff mode's spectral gap, ``1 - rho``.  Bounded below at 1e-4:
#: ``cond(I - dF/dx) ~ 1/gap``, so that is ``cond = 1e4``, where float32
#: already carries ``eps * cond ~ 1e-3`` of relative error and the
#: closed-form comparison below stops being a fair reference.  Bounded
#: above at 1e-2 because the *point* of the generator is to sit in the
#: band the audit failed in; the well-conditioned band is covered by
#: the parametrised example test in ``tests/core``.
_GAP = st.floats(min_value=1e-4, max_value=1e-2)

#: One fast mode alongside it.  A single stiff mode on its own is not
#: the failing shape: the audit's group was stiff *and* had a mode that
#: converged immediately, which is what leaves GMRES with a spread of
#: eigenvalues to resolve in float32.
_FAST = st.floats(min_value=0.0, max_value=0.5)

#: The cotangent, through the per-mode weights of the loss.  Which
#: cotangents fail is the lottery: ``(1, 1)`` and ``(-1, 1)`` failed at
#: rates where ``(1, 0)``, ``(0, 1)``, ``(1e3, 1)`` and ``(1e-3, 1)``
#: did not, so the two that fail must not be special-cased into the
#: generator and the four that pass must not be dropped from it.  Exact
#: zeros are kept for the same reason they were in the audit: a loss
#: touching only some fields is what makes lineax's elementwise
#: tolerance unreachable, and it is the common multi-physics shape.
_WEIGHT = st.sampled_from([0.0, 1.0, -1.0, 0.5, 1e-3, 1e3])


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(
    gap=_GAP,
    fast=_FAST,
    w_slow=_WEIGHT,
    w_fast=_WEIGHT,
    forcing=st.sampled_from([1e-5, 1.0]),
)
def test_grad_through_the_default_coupling_solver_never_raises(
    gap, fast, w_slow, w_fast, forcing,
):
    """A generated spectrum and cotangent still differentiate.

    The gradient of ``sum(w * x*)`` with respect to ``gain`` is
    ``sum(w * c / (1 - rho))`` for the affine map above, so the value
    is checked too.  ``rel=1e-2`` is float32 on an operator whose
    condition number reaches 1e4: the analytic answer is dominated by
    the stiffest mode, and ``1/(1 - rho)`` amplifies round-off in the
    same proportion.
    """
    rho = (1.0 - gap, fast)
    c = (forcing, 1.0)
    weights = (w_slow, w_fast)
    w = jnp.asarray(weights)

    def loss(p):
        return jnp.sum(w * _build(rho, c).run_scan(1, params=p)["a"]["x"])

    base = _build(rho, c).params
    grad = jax.grad(loss)(base)["nodes"]["a"]["gain"]
    exact = sum(
        wi * ci / (1.0 - ri) for wi, ci, ri in zip(weights, c, rho)
    )
    note(f"rho={rho} c={c} w={weights} grad={grad} exact={exact}")
    assert jnp.isfinite(grad), (
        f"non-finite adjoint on the default path: rho={rho} w={weights}"
    )
    assert float(grad) == pytest.approx(exact, rel=1e-2, abs=1e-6)
