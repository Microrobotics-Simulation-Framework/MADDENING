"""The ``jnp.where`` gradient trap on a class whose whole job is masking.

``_solve_and_pack`` applies ``jnp.where(mask, c, 0)``.  That protects the
*value*, never the *tangent*: if ``solve_frozen`` evaluates an expression
that is singular on the inactive entries, the forward pass is clean and
``jax.grad`` returns ``NaN`` (audit A9).  The base class cannot repair
that -- the tangent is poisoned inside the subclass's own expression --
so it does two things instead: it warns when it can see the damage, and
it offers :meth:`AdaptiveNode.mask_safe`, the inner half of the
double-``where`` idiom, which sanitises the *input* of the unsafe
operation.  These tests pin all three behaviours.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.nodes.adaptive import AdaptiveNode

N = 6
OFF_MASK_SIGN = jnp.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])


class _SqrtTrap(AdaptiveNode):
    """``sqrt`` of a negative number on the inactive entries.

    ``style`` selects how the subclass handles the masking:

    * ``"unmasked"`` -- returns the raw (non-finite) coefficients and
      lets the base class zero them: the one case the base class can see;
    * ``"output"`` -- masks its own output, which hides the damage from
      the base class but not from ``jax.grad``;
    * ``"mask_safe"`` -- the double-``where`` fix.
    """

    def __init__(self, style: str):
        super().__init__("trap", 1.0, n_max=N, theta=1.0, blindness_gate=False)
        self.style = style

    def compute_active_set(self, state, params, *, prev=None, is_cold_start=False):
        return jnp.arange(N) < 3

    def solve_frozen(self, state, mask, params):
        if self.style == "mask_safe":
            operand = self.mask_safe(mask, OFF_MASK_SIGN * params["theta"], fill=1.0)
            return {"c": jnp.where(mask, jnp.sqrt(operand), 0.0)}
        c = jnp.sqrt(OFF_MASK_SIGN * params["theta"])
        return {"c": c if self.style == "unmasked" else jnp.where(mask, c, 0.0)}

    def objective(self, state, params):
        return jnp.sum(state["c"])


def _forward_and_gradient(node):
    empty = {"c": jnp.zeros(N), "mask": jnp.zeros(N, dtype=bool)}

    def J(theta):
        return node.objective(node.update(empty, {}, 1.0, params={"theta": theta}), {})

    return float(J(jnp.asarray(1.0))), float(jax.grad(J)(jnp.asarray(1.0)))


def test_gradient_is_nan_when_solve_frozen_evaluates_a_singular_expression_off_mask():
    """The documented limitation, asserted rather than assumed: the value
    is exactly right and only the gradient is poisoned."""
    value, gradient = _forward_and_gradient(_SqrtTrap("output"))
    assert value == pytest.approx(3.0)
    assert np.isnan(gradient)


def test_the_base_class_warns_when_it_can_see_the_non_finite_coefficients():
    """When the subclass leaves the masking to the base class, the base
    class sees the ``NaN`` before it erases it -- and says so, naming the
    remedy, instead of silently returning a clean-looking state."""
    node = _SqrtTrap("unmasked")
    empty = {"c": jnp.zeros(N), "mask": jnp.zeros(N, dtype=bool)}
    with pytest.warns(UserWarning, match="mask_safe"):
        out = node.update(empty, {}, 1.0, params={"theta": jnp.asarray(1.0)})
    assert bool(jnp.all(jnp.isfinite(out["c"])))
    assert bool(jnp.all(out["c"][~out["mask"]] == 0.0))


def test_mask_safe_double_where_guard_keeps_the_gradient_finite():
    """The remedy the guide prescribes: sanitise the input, not the
    output.  Same value, finite and correct gradient
    (``d/dtheta 3 sqrt(theta) = 1.5`` at ``theta = 1``)."""
    value, gradient = _forward_and_gradient(_SqrtTrap("mask_safe"))
    assert value == pytest.approx(3.0)
    assert np.isfinite(gradient)
    assert gradient == pytest.approx(1.5)


def test_mask_safe_leaves_the_active_entries_untouched():
    mask = jnp.array([True, False, True, False, False, False])
    out = AdaptiveNode.mask_safe(mask, jnp.arange(N, dtype=float), fill=1.0)
    assert np.array_equal(np.asarray(out), np.array([0.0, 1.0, 2.0, 1.0, 1.0, 1.0]))
    assert out.dtype == jnp.arange(N, dtype=float).dtype
