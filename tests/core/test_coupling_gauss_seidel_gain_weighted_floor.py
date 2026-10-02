"""A Gauss-Seidel pass's float floor weights each same-pass read by its measured gain.

The floor ``spectral_error_bound`` and the gradient bound add is
``PRECISION_FLOOR_ULPS`` units per evaluation the pass rounds like.  Under
Gauss-Seidel a rounding made by a member reaches every member reading it
from the same pass, and was counted along the longest chain of such reads
on the premise that a read passes a relative rounding on at most unchanged
-- relative gains summing to at most one at each node.  A squaring relay
``x = u * u`` doubles it, exactly, with no cancellation and one correctly
rounded multiply; a ring of them read ``spectral_error_bound`` at 0.20x the
true distance and ``gradient_relative_error_bound`` at 0.018x the true
gradient error, both flags set.

With ``diagnostics=True`` the step now measures each same-pass read's
relative gain at the returned state (one JVP of the reading node's update
along the source's own state) and counts ``depth(n) = d_n e_n + sum_m g_nm
depth(m)``, never below the structural count; the report reads that count
from the step (``coupling_{key}_pass_evaluations``).
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


class _Affine(SimulationNode):
    """``x <- g * u + c`` (scalar), one evaluation, declared."""

    def __init__(self, name, g, c, x0):
        super().__init__(name, 1.0, g=jnp.float32(g), c=jnp.float32(c))
        self._x0 = float(x0)

    def initial_state(self):
        return {"x": jnp.asarray([self._x0], jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(1,), dtype=jnp.float32,
                                       default=jnp.zeros(1, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": p["g"] * boundary_inputs["u"] + p["c"]}

    def update_evaluations(self):
        return 1


class _Square(_Affine):
    """``x <- u * u``: one multiply, relative gain exactly 2, one evaluation."""

    def __init__(self, name, x0):
        super().__init__(name, 1.0, 0.0, x0)

    def update(self, state, boundary_inputs, dt, *, params=None):
        u = boundary_inputs["u"]
        return {"x": u * u}


def _squaring_chain(n, rho=0.9, head=1e-3):
    """``n0 <- g sN + c``, ``s_k <- s_{k-1}**2``: a Gauss-Seidel ring of ``n`` squares.

    The loop contracts at ``rho`` at its fixed point (``n0`` near 1.001),
    every node declares one evaluation, the Jacobian has rank one.
    Returns ``(gm, names, exact fixed point)``.
    """
    names = ["n00"] + [f"s{k:02d}" for k in range(1, n + 1)]
    n0t = 1.001
    m = 2 ** n
    sNt = n0t ** m
    g = float(np.float32(rho / m * n0t / sNt))
    c = float(np.float32(n0t - g * sNt))
    x = n0t
    for _ in range(200):          # n0 = g n0**m + c, Newton in float64
        x -= (g * x ** m + c - x) / (g * m * x ** (m - 1) - 1.0)
    xs = [x ** (2 ** k) for k in range(n + 1)]
    gm = GraphManager()
    gm.add_node(_Affine("n00", g, c, xs[0] * (1 - head)))
    for k in range(1, n + 1):
        gm.add_node(_Square(names[k], xs[k] * (1 - head)))
    for k in range(1, n + 1):
        gm.add_edge(names[k - 1], names[k], "x", "u")
    gm.add_edge(names[n], names[0], "x", "u")
    gm.add_coupling_group(names, max_iterations=3000, tolerance=1e-12, diagnostics=True)
    gm.compile()
    return gm, names, np.asarray(xs, np.float64), (g, c)


def _affine_ring(n, rho=0.9):
    """Plain relays of gain ``rho**(1/n)`` below one: the structural count stands."""
    names = [f"a{k:02d}" for k in range(n)]
    gk = float(np.float32(rho ** (1.0 / n)))
    gm = GraphManager()
    for nm in names:
        gm.add_node(_Affine(nm, gk, 1.0 - gk, 0.999))
    for k in range(n):
        gm.add_edge(names[k - 1], names[k], "x", "u")
    gm.add_coupling_group(names, max_iterations=3000, tolerance=1e-12, diagnostics=True)
    gm.compile()
    return gm, names


def _slot(gm, names):
    return float(gm._state["_meta"][f"coupling_{'+'.join(sorted(names))}_pass_evaluations"])


@functools.lru_cache(maxsize=None)
def _stepped_chain(n):
    gm, names, xs, gc = _squaring_chain(n)
    gm.step()
    return gm, names, xs, gc


def test_each_same_pass_read_is_weighted_by_its_measured_gain():
    """Eight squares: depth ``2**(k+1) - 1`` at the k-th, 511 at the last.

    The structural count of the same chain is 9.  Each square's read of
    its predecessor measures a relative gain of exactly 2 (the JVP of
    ``u * u`` along ``u`` is ``2 u**2``), so the count is exact.
    """
    gm, names, _xs, _gc = _stepped_chain(8)
    assert gm._committed_floor_inputs["+".join(sorted(names))][0] == 9.0
    assert _slot(gm, names) == 511.0


def test_reads_of_gain_below_one_keep_the_structural_count():
    """An affine ring of gain below one: the weighted depth is smaller; 6 stands."""
    gm, names = _affine_ring(6)
    gm.step()
    assert _slot(gm, names) == 6.0


def test_the_spectral_bound_holds_on_a_stalled_squaring_chain():
    """Usable, and at least the true distance to the exact fixed point.

    Before the gain weighting it read 0.66x the true distance at eight
    squares (0.20x at twelve) with ``spectral_usable=True``.
    """
    gm, names, xs, _gc = _stepped_chain(8)
    d = gm.coupling_diagnostics()["+".join(sorted(names))]
    x = np.array([float(gm.get_node_state(nm)["x"][0]) for nm in names], np.float64)
    dist = float(np.sqrt(np.sum(((x - xs) / np.maximum(np.abs(x), np.abs(xs))) ** 2)))
    assert d["precision_limited"] and d["spectral_usable"], dict(d)
    assert d["spectral_error_bound"] >= dist, (d["spectral_error_bound"], dist)


# Slow: the gradient compiles the twelve-node step twice more (~20 s on a
# six-core slice).  Per push the weighting and the spectral bound it feeds
# are pinned above; the gradient bound shares the count.
# Per push: tests/core/test_coupling_gauss_seidel_gain_weighted_floor.py::test_each_same_pass_read_is_weighted_by_its_measured_gain
@pytest.mark.slow
@pytest.mark.parametrize("n", [8, 12])
def test_the_gradient_bound_holds_on_a_stalled_squaring_chain(n):
    """``gradient_relative_error_bound`` against the exact gradient of the fixed point.

    Read 0.91x (eight squares) and 0.018x (twelve) the true relative error
    of ``dx/dc`` with ``gradient_bound_usable=True`` before the weighting.
    """
    gm, names, xs, (g, _c) = _stepped_chain(n)
    d = gm.coupling_diagnostics()["+".join(sorted(names))]
    xk = np.array([float(gm.get_node_state(nm)["x"][0]) for nm in names], np.float64)

    def f(c):
        g2, _n, _x, _gc = _squaring_chain(n)
        p = jax.tree.map(lambda v: v, g2.params)
        p["nodes"]["n00"]["c"] = c
        out = g2.run_scan(1, params=p)
        return jnp.stack([out[nm]["x"][0] for nm in names])

    tk = np.asarray(jax.jacfwd(f)(gm.params["nodes"]["n00"]["c"]), np.float64)
    m = 2 ** n
    dn0 = 1.0 / (1.0 - g * m * xs[0] ** (m - 1))
    tstar = np.array([(2 ** k) * xs[0] ** (2 ** k - 1) * dn0 for k in range(n + 1)])
    w = 1.0 / np.abs(xk)
    rel = float(np.linalg.norm(w * (tk - tstar)) / np.linalg.norm(w * tk))
    assert d["gradient_bound_usable"], dict(d)
    assert d["gradient_relative_error_bound"] >= rel, (d["gradient_relative_error_bound"], rel)
