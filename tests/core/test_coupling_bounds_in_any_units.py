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


class _Bent(SimulationNode):
    """``x <- g u + c + k (u1 / |u0|) u1`` on two entries: smooth, nonlinear, of degree one.

    The ratio is formed as ``u1 * exp(-log|u0|)``, whose derivative
    intermediates stay at the state's own relative size under a tangent in
    state units (JAX's rule for ``u1 / u0`` forms ``u0**-2``, which
    overflows below ``|u0| ~ 1e-19`` whatever the tangent).  Its Jacobian
    depends on the point's direction, not its size, so the dimensionless
    report is the same at every scale up to rounding -- ``exp`` and ``log``
    do not scale exactly, so not to the bit.
    """

    def __init__(self, name, g, c, k):
        c = np.asarray(c, np.float32)
        super().__init__(name, 1.0, g=jnp.float32(g), c=jnp.asarray(c), k=jnp.float32(k))
        self._c = c

    def initial_state(self):
        return {"x": jnp.asarray(self._c)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(2,), dtype=jnp.float32,
                                       default=jnp.zeros(2, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs["u"]
        ratio = u[1] * jnp.exp(-jnp.log(jnp.abs(u[0])))
        bend = jnp.stack([jnp.float32(0.0), ratio * u[1]])
        return {"x": p["g"] * u + p["c"] + p["k"] * bend}

    def update_evaluations(self):
        return 1


_CB = np.array([1.0, 0.5], np.float32)


@functools.lru_cache(maxsize=None)
def _bent_pair():
    gm = GraphManager()
    # Loop gain 0.82 at the fixed point (2.64 in the bent entry).
    gm.add_node(_Bent("a", 0.8, _CB, 0.02))
    gm.add_node(_Bent("b", 1.0, [0.0, 0.0], 0.0))
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    gm.add_coupling_group(["a", "b"], tolerance=1e-3, max_iterations=200, diagnostics=True)
    gm.compile()
    return gm


def _bent_report(k):
    gm = _bent_pair()
    s = np.float32(2.0 ** k)
    gm.reset_state()
    p = jax.tree.map(lambda v: v, gm.params)
    p["nodes"]["a"]["c"] = jnp.asarray(_CB * s)
    for nm in ("a", "b"):
        gm.set_node_state(nm, {"x": jnp.asarray(_CB * s)})
    gm.step(params=p)
    return gm.coupling_diagnostics()["a+b"]


@pytest.mark.parametrize("k", [60, 100, -100, -116, -122])
def test_a_nonlinear_group_never_reports_a_usable_bound_its_control_disowns(k):
    """A degree-one nonlinear pair at ``2**k``: the control's numbers, or no usable bound.

    The report's Jacobian-vector products take their tangent in state
    units, lifted out of the underflow range only where the group is that
    small (``_tangent_lift``).  A tangent framed to order one at every
    magnitude was tried first and was wrong both ways on this pair: at
    ``2**60`` the gradient bound read 0.79x its control with
    ``gradient_bound_usable=True`` (the node's derivative intermediates,
    relative perturbations of ``2**-60``, underflowed), and at ``2**-100``
    nothing was usable.  Before any frame, at ``2**-116`` and below, the
    gradient bound read exactly ``0.0``, usable.  At ``2**-122`` the lifted
    tangent overflows this node's ``1/|u0|`` derivative: no usable report,
    which is the most it can say there.
    """
    ref = _bent_report(0)
    got = _bent_report(k)
    assert ref["gradient_bound_usable"] and ref["spectral_usable"], dict(ref)
    assert got["iterations"] == ref["iterations"]
    if got["gradient_bound_usable"]:
        assert got["gradient_relative_error_bound"] == pytest.approx(
            ref["gradient_relative_error_bound"], rel=2e-2), (k, dict(got))
    if got["spectral_usable"]:
        assert got["rho_spectral"] == pytest.approx(ref["rho_spectral"], rel=1e-3), (k, dict(got))
    if k >= -116:
        assert got["gradient_bound_usable"] and got["spectral_usable"], (k, dict(got))
