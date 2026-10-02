"""The spectral rate and the gradient bound do not depend on the group's units.

``coupling_diagnostics()`` reports two numbers measured after the loop by
Jacobian-vector products at the returned state: ``rho_spectral`` (an
Arnoldi estimate of ``dF/dx`` in the group's norm, ``_spectral_rate_at``)
and ``gradient_relative_error_bound`` (``_gradient_error_bound_body``: the
residual, a curvature step and a secant, all at the returned iterate).
Both are dimensionless.  They were formed in state units: the residual
``F(x) - x`` and the secant as raw differences, the curvature step as
``x + delta * max|x|``, the tangent weighted by ``1/|x|`` -- and each of
those flushes to zero once the field is small enough (XLA on CPU flushes
subnormals).  A pair near ``1e-34`` reported a gradient bound of exactly
``0.0`` with ``gradient_bound_usable=True``, against a true relative error
of ``6.9e-5``; a ring near ``2**-118`` reported a shifted ``rho_spectral``
(and from there a different ``spectral_error_bound``).  They are now
formed in power-of-two frames, so a group scaled by ``2**-k`` reports the
same numbers to the bit wherever its own state is a normal float.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

N = 16


class _Rel(SimulationNode):
    """``x <- g u + c`` on an ``n``-entry field; ``g`` and ``c`` are parameters."""

    def __init__(self, name, g, c):
        c = np.atleast_1d(np.asarray(c, np.float32))
        super().__init__(name, 1.0, g=jnp.float32(g), c=jnp.asarray(c))
        self._c = c

    def initial_state(self):
        return {"x": jnp.asarray(self._c)}

    def boundary_input_spec(self):
        n = self._c.shape[0]
        return {"u": BoundaryInputSpec(shape=(n,), dtype=jnp.float32,
                                       default=jnp.zeros(n, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": p["g"] * boundary_inputs["u"] + p["c"]}

    def update_evaluations(self):
        return 1


_C = {nm: np.random.default_rng(7 + i).uniform(0.5, 1.5, N).astype(np.float32)
      for i, nm in enumerate("abc")}
_G = {"a": 0.97, "b": 0.96, "c": 0.98}


@functools.lru_cache(maxsize=None)
def _ring(mode, acceleration):
    """A Gauss-Seidel or Jacobi ring of three 16-entry relays, compiled once."""
    gm = GraphManager()
    for nm in "abc":
        gm.add_node(_Rel(nm, _G[nm], _C[nm]))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "c", "x", "u")
    gm.add_edge("c", "a", "x", "u")
    gm.add_coupling_group(list("abc"), max_iterations=200, tolerance=1e-3,
                          iteration_mode=mode, acceleration=acceleration, diagnostics=True)
    gm.compile()
    return gm


def _ring_report(mode, acceleration, k):
    gm = _ring(mode, acceleration)
    s = np.float32(2.0 ** k)
    gm.reset_state()
    p = jax.tree.map(lambda v: v, gm.params)
    for nm in "abc":
        p["nodes"][nm]["c"] = jnp.asarray(_C[nm] * s)
        gm.set_node_state(nm, {"x": jnp.asarray(_C[nm] * s)})
    gm.step(params=p)
    x = np.concatenate([np.asarray(gm.get_node_state(nm)["x"]) for nm in "abc"])
    return gm.coupling_diagnostics()["a+b+c"], x


_KEYS = ("iterations", "residual", "converged", "rho_spectral", "spectral_error_bound",
         "spectral_usable", "gradient_relative_error_bound", "gradient_bound_usable")


@pytest.mark.parametrize("k", [-118, -122])
@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
def test_the_spectral_rate_and_the_gradient_bound_are_the_same_at_every_scale(mode, k):
    """A ring at ``2**k`` reports what it reports at ``2**0``, every key bit for bit.

    ``acceleration="none"``: the forward is exactly scaled already (checked
    here too), so any difference is the report's.  Before the frames:
    under Gauss-Seidel ``rho_spectral`` moved at ``2**-118`` and the
    gradient bound read ``0.0``, then NaN at ``2**-122``; under Jacobi, at
    ``2**-122``, ``rho_spectral`` moved and ``spectral_usable`` dropped.
    """
    ref, x_ref = _ring_report(mode, "none", 0)
    got, x = _ring_report(mode, "none", k)
    np.testing.assert_array_equal(x, x_ref * np.float32(2.0 ** k))
    moved = {key: (ref[key], got[key]) for key in _KEYS
             if not (ref[key] == got[key] or (ref[key] != ref[key] and got[key] != got[key]))}
    assert not moved, f"2**{k}: {moved}"


def _pair(c, linear_solver="gmres"):
    gm = GraphManager()
    gm.add_node(_Rel("a", 0.9, [c]))
    gm.add_node(_Rel("b", 1.0, [0.0]))
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    gm.add_coupling_group(["a", "b"], tolerance=1e-4, diagnostics=True, max_iterations=400,
                          linear_solver=linear_solver)
    gm.compile()
    return gm


@pytest.mark.parametrize("k", [14, 20])
def test_a_usable_gradient_bound_is_not_zero_on_a_tiny_group(k):
    """A pair near ``1e-34``: the control's bound, which is above the true error.

    ``x <- 0.9 u + c`` read back by a relay, ``c = 1e-30 * 2**-k``.  The true
    relative error of the IFT gradient ``dx/dg`` at the returned iterate is
    ``|x_b - x*| / x*`` (the tangent there is ``x_b / (1 - g)`` in both
    nodes), the same at every scale.  Before the frames the bound read
    exactly ``0.0`` with ``gradient_bound_usable=True`` from ``k = 14``.
    """
    c0 = float(np.float32(1e-30))
    ref = _pair(c0)
    ref.step()
    want = ref.coupling_diagnostics()["a+b"]
    c = float(np.float32(c0 * 2.0 ** -k))
    gm = _pair(c)
    gm.step()
    got = gm.coupling_diagnostics()["a+b"]
    xb = float(gm.get_node_state("b")["x"][0])
    g32 = float(np.float32(0.9))
    xs = c / (1.0 - g32)
    true_rel = abs(xb - xs) / xs
    assert got["gradient_bound_usable"]
    assert got["gradient_relative_error_bound"] == want["gradient_relative_error_bound"]
    assert got["gradient_relative_error_bound"] >= true_rel, (got, true_rel)
