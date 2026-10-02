"""The IFT tangent and adjoint do not depend on the units of the group or of the loss.

Under ``solver="ift"`` both derivatives of a coupled step go through one
linear solve, ``(I - dF/dx) v = b`` (``_ift_linear_solve``): ``b`` is the
tangent rhs in forward mode (``jax.jacfwd``) and the cotangent in reverse
mode (``jax.grad``).  The Krylov backends stopped on ``|r| <= 1e-8 + rtol
* max|b|``.  With the zero initial guess the absolute ``1e-8`` passed
before a single step whenever ``max|b| <= ~1e-8``, and the solve returned
``0``, reported successful: the derivative of the fixed point read exactly
zero for a group written in small units, and in reverse mode for any loss
close to its minimum (the cotangent of ``(x - obs)**2`` is ``2 (x - obs)``).
Above the dense fallback's size (50 coupled DOF) a moderately small ``b``
raised the "ill-conditioned" adjoint error instead.  The solve now runs on
``b`` rescaled by an exact power of two, with a purely relative tolerance.

Each case here is linear in its scale ``c``, so the derivative relative to
its exact value is the same at every scale, and ``linear_solver="dense"``
(an LU solve, no tolerance) is the second reference.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

#: Down to where the cotangent of a squared error is still a normal float32.
#: (Below about 1e-19 the old solve's own norms underflowed, it reported a
#: breakdown, and the dense fallback happened to answer correctly; between
#: 1e-8 and there it answered zero.)
SCALES = (1.0, 1e-9, 1e-12, 1e-16, 1e-30)
G = 0.9


class _Rel(SimulationNode):
    """``x <- g * u + c`` on an ``n``-entry field; ``g`` and ``c`` are parameters."""

    def __init__(self, name, g, c):
        g, c = np.atleast_1d(np.asarray(g, np.float32)), np.atleast_1d(np.asarray(c, np.float32))
        super().__init__(name, 1.0, g=jnp.asarray(g), c=jnp.asarray(c))
        self._n = c.shape[0]

    def initial_state(self):
        return {"x": jnp.zeros(self._n, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._n,), dtype=jnp.float32,
                                       default=jnp.zeros(self._n, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": p["g"] * boundary_inputs["u"] + p["c"]}


def _pair(g, c, **kw):
    n = np.atleast_1d(c).shape[0]
    gm = GraphManager()
    gm.add_node(_Rel("a", g, c))
    gm.add_node(_Rel("b", np.ones(n), np.zeros(n)))
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    gm.add_coupling_group(["a", "b"], tolerance=1e-6, max_iterations=2000, **kw)
    gm.compile()
    return gm


def _scalar_derivatives(c, **kw):
    """``(d x_a / d g`` by jacfwd, ``d/dg (x_a - 1.05 x*)**2`` by grad, exact of each)."""
    g32, c32 = float(np.float32(G)), float(np.float32(c))
    xstar = c32 / (1.0 - g32)
    obs = jnp.float32(1.05 * xstar)

    def state_of(g):
        gm = _pair(G, [c], **kw)
        p = jax.tree.map(lambda v: v, gm.params)
        p["nodes"]["a"]["g"] = g
        return gm.run_scan(1, params=p)["a"]["x"][0]

    g0 = _pair(G, [c], **kw).params["nodes"]["a"]["g"]
    fwd = float(jax.jacfwd(state_of)(g0)[0])
    rev = float(jax.grad(lambda g: (state_of(g) - obs) ** 2)(g0)[0])
    x = float(state_of(g0))
    exact_fwd = c32 / (1.0 - g32) ** 2
    return fwd, exact_fwd, rev, 2.0 * (x - float(obs)) * exact_fwd


@pytest.mark.parametrize("c, linear_solver",
                         [(c, "gmres") for c in SCALES] + [(1e-12, "dense")])
def test_the_tangent_and_the_adjoint_are_the_same_at_every_scale(c, linear_solver):
    """Two coupled DOF: forward and reverse mode against the exact derivative.

    Before the fix ``"gmres"`` returned exactly 0 in reverse mode from
    ``c = 1e-8`` down and in forward mode from ``c = 1e-9`` down.
    """
    fwd, exact_fwd, rev, exact_rev = _scalar_derivatives(c, linear_solver=linear_solver)
    assert fwd == pytest.approx(exact_fwd, rel=1e-4), (c, fwd, exact_fwd)
    assert rev == pytest.approx(exact_rev, rel=2e-3), (c, rev, exact_rev)


#: 48 distinct modes per node, 96 coupled DOF: above the dense fallback, so
#: GMRES alone answers.
_GV = np.linspace(0.5, 0.97, 48).astype(np.float32)
_CV = np.linspace(1.0, 2.0, 48).astype(np.float32)


@pytest.mark.parametrize("scale", (1.0, 1e-8, 1e-12, 1e-30))
def test_the_adjoint_above_the_dense_fallback_is_the_same_at_every_scale(scale):
    """96 coupled DOF, reverse mode: the gradient of a least-squares loss in ``c``.

    Before the fix it raised the "ill-conditioned" adjoint error at
    ``1e-8`` and returned an exactly zero gradient below ``1e-9``.
    """
    c = (_CV * np.float32(scale)).astype(np.float32)
    g64, c64 = _GV.astype(np.float64), c.astype(np.float64)
    obs = jnp.asarray(1.05 * c64 / (1 - g64), jnp.float32)

    def loss(cc):
        gm = _pair(_GV, c)
        p = jax.tree.map(lambda v: v, gm.params)
        p["nodes"]["a"]["c"] = cc
        return jnp.sum((gm.run_scan(1, params=p)["a"]["x"] - obs) ** 2)

    gm0 = _pair(_GV, c)
    got = np.asarray(jax.grad(loss)(gm0.params["nodes"]["a"]["c"]), np.float64)
    x = np.asarray(gm0.run_scan(1)["a"]["x"], np.float64)
    exact = 2.0 * (x - np.asarray(obs, np.float64)) / (1 - g64)   # dx_i/dc_i = 1/(1 - g_i)
    err = float(np.linalg.norm(got - exact) / np.linalg.norm(exact))
    assert err < 1e-4, (scale, err)


def test_a_zero_cotangent_gives_an_exactly_zero_gradient():
    """A loss the group's state does not reach: the adjoint rhs is zero, and so is the answer.

    The zero rhs is answered with exact zeros rather than handed to the
    Krylov solver, whose normalisation by ``|b|`` has nothing to divide.
    """
    def loss(c):
        gm = _pair(_GV, _CV)
        p = jax.tree.map(lambda v: v, gm.params)
        p["nodes"]["a"]["c"] = c
        out = gm.run_scan(1, params=p)
        return 0.0 * jnp.sum(out["a"]["x"]) + jnp.sum(c)

    got = np.asarray(jax.grad(loss)(jnp.asarray(_CV)))
    np.testing.assert_array_equal(got, np.ones_like(_CV))
