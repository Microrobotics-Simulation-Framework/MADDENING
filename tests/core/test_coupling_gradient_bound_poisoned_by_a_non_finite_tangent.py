"""A probe whose tangent is not finite makes the gradient bound NaN, not a bound over the other probes.

``gradient_relative_error_bound`` covers the gradient in every scalar
constant.  A constant whose Jacobian-vector product through the group is
not finite has no gradient the bound can speak for, so the bound is NaN --
``gradient_bound_usable`` ``False``.  Before 0.4.0's round-5 fix such a
probe dropped out of the maximum silently (``NaN > 0`` is ``False``, so it
was taken as a probe the fixed point does not respond to), and the bound
over the remaining constants read usable: 0.078 here at three passes,
although the gradient in that constant is not computed (NaN under the
dense solve and ``"fori"``, 0.0 under the default GMRES solve) where the
fixed point's derivative is 1/(1 - 0.4) = 1.67.  That case
reached the bound in practice through a tangent overflow -- an all-zero
constant probed at 1.0 beside a state near 1e-35 -- which the probe's
power-of-two frame now prevents; a node whose derivative in a constant
cannot be evaluated is the same case without the overflow.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode


@jax.custom_jvp
def _no_derivative(c):
    """The identity, whose derivative along any non-zero tangent is NaN (and 0 along a zero one)."""
    return c


@_no_derivative.defjvp
def _no_derivative_jvp(primals, tangents):
    (c,), (t,) = primals, tangents
    return c, jnp.where(t != 0, jnp.nan, 0.0).astype(t.dtype)


class _Relay(SimulationNode):
    """``x <- b + g u (+ opaque(c))``."""

    def __init__(self, name, g, b, c=None):
        consts = dict(g=jnp.float32(g), b=jnp.float32(b))
        if c is not None:
            consts["c"] = jnp.float32(c)
        super().__init__(name, 1.0, **consts)

    def initial_state(self):
        return {"x": jnp.zeros(1, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(1,), dtype=jnp.float32,
                                       default=jnp.zeros(1, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        out = p["b"] + p["g"] * boundary_inputs["u"]
        if "c" in p:
            out = out + _no_derivative(p["c"])
        return {"x": out}


def _pair(cap, opaque):
    gm = GraphManager()
    gm.add_node(_Relay("A", 0.5, 1.0, c=1.0 if opaque else None))
    gm.add_node(_Relay("B", 0.8, 0.5))
    gm.add_edge("A", "B", "x", "u")
    gm.add_edge("B", "A", "x", "u")
    gm.add_coupling_group(["A", "B"], max_iterations=cap, tolerance=1e-7, diagnostics=True)
    gm.compile()
    gm.step()
    return gm.coupling_diagnostics()["A+B"]


@pytest.mark.parametrize("cap", [3, 6])
def test_a_constant_with_no_finite_tangent_leaves_no_usable_bound(cap):
    d = _pair(cap, opaque=True)
    assert d["spectral_usable"], dict(d)
    assert math.isnan(d["gradient_relative_error_bound"]), dict(d)
    assert d["gradient_bound_usable"] is False, dict(d)


@pytest.mark.parametrize("cap", [3, 6])
def test_without_that_constant_the_same_group_has_a_usable_bound(cap):
    """The control: the group's other constants alone are bounded, so the NaN above is that probe's."""
    d = _pair(cap, opaque=False)
    assert d["gradient_bound_usable"] is True, dict(d)
    assert math.isfinite(d["gradient_relative_error_bound"]), dict(d)
