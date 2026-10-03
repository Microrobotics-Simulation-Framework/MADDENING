"""The gradient bound holds, or is withdrawn, where the forward stops far from its fixed point.

``gradient_relative_error_bound`` is leading-order in the distance and
carries a Newton-Kantorovich check for the rest.  Two gaps were found by the
round-6 coupling audit (CPL-093) on pairs whose map is bilinear in the state:

* at ``max_iterations=2`` the check passed (``h = 0.19``) and the bound read
  0.986x the true gradient error with ``gradient_bound_usable=True``: the
  secant across the Newton step ``delta`` stood in for the change across
  ``x* - x_k``, and the part of ``x* - x_k`` the Newton step misses was not
  carried at all;
* ``L``, the Jacobian's Lipschitz constant, was its change along ``delta``
  alone.  On a pair 57% from its fixed point it read ``h = 0.37`` where the
  operator norm of the same change gives 0.69 -- past Kantorovich's 1/2 --
  the fixed point lay outside the radius the check certified, and the bound
  read 0.81x (0.65x before the first fix) with the flag set.

The bound now adds the Newton step's second-order miss and takes ``L`` as
an operator across the step.  The oracle is the IFT tangent at the float64
fixed point, by central differences of the float64 map, measured in the
group's L2 norm at the returned state.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode

KEY = "A+B"


def _f32(v):
    return float(np.float32(v))


class _Bilinear(SimulationNode):
    """``x <- a x_pre + c + g (inp * inp[::-1])``: bilinear in the coupled state."""

    def __init__(self, name, x0, **params):
        super().__init__(name, 1.0, **params)
        self._x0 = np.asarray(x0, np.float32)

    def initial_state(self):
        return {"x": jnp.asarray(self._x0)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else params
        inp = boundary_inputs.get("inp", jnp.zeros((2,), jnp.float32))
        return {"x": p["a"] * state["x"] + p["c"] + p["g"] * inp * inp[::-1]}

    def update_evaluations(self):
        return 1.0


#: ``(gA, gB, A0, B0, cA, cB)``.  The audit's pair, and the pair the
#: Kantorovich check passed along ``delta`` 57% from its fixed point.
PAIRS = {
    "audit": (0.8, 0.9, [0.1, 0.2], [0.3, 0.1], [0.3, 0.5], [0.2, 0.1]),
    "far": (1.3056230968020888, 0.9512193440850327, [-0.31664374, -0.20407315],
            [0.07441077, -0.3569979], [0.10686893, 0.31694561], [0.30487887, 0.24566291]),
}


def _graph(pair, cap, acceleration="none"):
    gA, gB, A0, B0, cA, cB = PAIRS[pair]
    gm = GraphManager()
    gm.add_node(_Bilinear("A", A0, a=0.2, c=np.asarray(cA, np.float32), g=gA))
    gm.add_node(_Bilinear("B", B0, a=-0.1, c=np.asarray(cB, np.float32), g=gB))
    gm.add_edge("A", "B", "x", "inp")
    gm.add_edge("B", "A", "x", "inp")
    gm.add_coupling_group(["A", "B"], max_iterations=cap, tolerance=1e-8,
                          acceleration=acceleration, diagnostics=True)
    gm.compile()
    return gm


def _true_relative_error(pair, gm):
    """The relative error of the IFT tangent in ``g_A`` at the returned state, float64."""
    gA, gB, A0, B0, cA, cB = PAIRS[pair]
    A0 = np.asarray(A0, np.float32).astype(np.float64)
    B0 = np.asarray(B0, np.float32).astype(np.float64)
    cA = np.asarray(cA, np.float32).astype(np.float64)
    cB = np.asarray(cB, np.float32).astype(np.float64)
    g0 = _f32(gA)

    def F(x, g):
        na = _f32(0.2) * A0 + cA + g * x[2:] * x[2:][::-1]
        nb = _f32(-0.1) * B0 + cB + _f32(gB) * na * na[::-1]
        return np.concatenate([na, nb])

    def tangent(x, h=1e-7):
        J = np.stack([(F(x + h * e, g0) - F(x - h * e, g0)) / (2 * h) for e in np.eye(4)],
                     axis=1)
        Fc = (F(x, g0 + h) - F(x, g0 - h)) / (2 * h) * g0     # probed by its own magnitude
        return np.linalg.solve(np.eye(4) - J, Fc)

    xk = np.concatenate([np.asarray(gm.get_node_state(n)["x"], np.float64) for n in ("A", "B")])
    xs = xk.copy()
    for _ in range(20000):
        xs = F(xs, g0)
    assert np.max(np.abs(F(xs, g0) - xs)) < 1e-12, "fixture premise: a contracting pair"
    w = np.concatenate([np.full(2, 1.0 / np.max(np.abs(xk[:2]))),
                        np.full(2, 1.0 / np.max(np.abs(xk[2:])))])
    tk, ts = tangent(xk), tangent(xs)
    return float(np.linalg.norm(w * (tk - ts)) / np.linalg.norm(w * tk))


@pytest.mark.parametrize("cap", [2, 3, 6])
def test_a_usable_gradient_bound_holds_on_the_audit_pair_at_every_cap(cap):
    """At ``max_iterations=2`` it read 0.986x the true error, usable."""
    gm = _graph("audit", cap)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.step()
    d = gm.coupling_diagnostics()[KEY]
    true = _true_relative_error("audit", gm)
    assert d["gradient_bound_usable"], dict(d)      # the fixture premise: certified
    assert d["gradient_relative_error_bound"] >= true, (
        d["gradient_relative_error_bound"] / true, dict(d))


@pytest.mark.parametrize("acceleration", ["none", "aitken", "iqn-ils"])
def test_far_from_its_fixed_point_the_gradient_bound_holds_or_is_withdrawn(acceleration):
    """The pair 57% from its fixed point: never a usable bound below the true error.

    Its Kantorovich ``h`` is 0.69 with ``L`` as an operator, so the check
    fails and the bound is withdrawn (``inf``, unusable); with ``L`` along
    ``delta`` it read 0.37, passed, and the bound read 0.65-0.81x.
    """
    gm = _graph("far", 2, acceleration)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.step()
    d = gm.coupling_diagnostics()[KEY]
    true = _true_relative_error("far", gm)
    if d["gradient_bound_usable"]:
        assert d["gradient_relative_error_bound"] >= true, (
            d["gradient_relative_error_bound"] / true, dict(d))
    else:
        assert not np.isfinite(d["gradient_relative_error_bound"]) or not d["spectral_usable"], \
            dict(d)
