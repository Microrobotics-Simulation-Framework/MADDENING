"""The Kantorovich check reads ``h = beta L eta`` with ``beta`` the full resolvent norm, as CPL-095 states.

``gradient_bound_usable`` is ``False`` -- the bound ``inf`` -- where
``h >= 1/2``, with ``beta = ||(I - J(x_k))^{-1}||`` over the whole space in
the group's norm, ``eta`` the Newton step and ``L`` the change of the
Jacobian along it.  Until 0.4.0's round-5 fix ``beta`` was the Arnoldi
factor, the resolvent restricted to the Krylov space of the start vector
and the residual (MADD-ANO-138).  On the round-5 audit's rank-one ring
made nonlinear (``x0 <- b0 + g0 u + q u**2``) that factor is several times
smaller than the full norm, and with it the check passed at a float64
``h`` of 0.67-1.75: every case below that must read unusable read usable
(bounds 0.82-187).  The verdict is checked here against ``h`` computed
from the analytic Jacobian in float64 at the returned state.

Two widths: one entry a node (3 coupled entries, the norm from the
range basis's own square compression) and four (12 entries, more than
the basis's eight vectors, so ``beta`` is assembled through the
transposed map).  The four entries are copies of the scalar ring scaled
by powers of two, so their ``h`` is the scalar ring's exactly.
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

#: Design 1 of ``test_coupling_gradient_bound_applies_the_full_resolvent.py``.
_G = (-12.137987144851225, 0.03209990587719496, -2.2631998902941737)
_B = (0.3133808845080941, -1.9064101527339117, -7.043231919131743)
_X0 = (1.7952182122084204, -0.66232274948191, 0.7460887496060612)
_NAMES = ("n0", "n1", "n2")
#: Per-entry scale of the wide ring: powers of two, so each entry is the
#: scalar ring scaled exactly (``q`` divided by the same factor).
_SCALES = {1: (1.0,), 4: (1.0, 0.5, 2.0, 0.25)}


class _Quad(SimulationNode):
    """``x <- b + g u + q u**2`` entry by entry; ``g`` and ``b`` parameters."""

    def __init__(self, name, g, b, q, x0):
        super().__init__(name, 1.0, g=jnp.asarray(g, jnp.float32), b=jnp.asarray(b, jnp.float32))
        self._q = np.asarray(q, np.float32)
        self._x0 = np.asarray(x0, np.float32)

    def initial_state(self):
        return {"x": jnp.asarray(self._x0)}

    def boundary_input_spec(self):
        m = self._x0.shape[0]
        return {"u": BoundaryInputSpec(shape=(m,), dtype=jnp.float32,
                                       default=jnp.zeros(m, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs["u"]
        return {"x": p["b"] + p["g"] * u + jnp.asarray(self._q) * u * u}


def _arrays(q, width):
    f = np.asarray(_SCALES[width])
    g = [np.full(width, v) for v in _G]
    b = [v * f for v in _B]
    qs = [q / f, np.zeros(width), np.zeros(width)]
    x0 = [v * f for v in _X0]
    return g, b, qs, x0


def _ring(q, cap, width):
    g, b, qs, x0 = _arrays(q, width)
    gm = GraphManager()
    for i, nm in enumerate(_NAMES):
        gm.add_node(_Quad(nm, g[i], b[i], qs[i], x0[i]))
    gm.add_edge("n2", "n0", "x", "u")
    gm.add_edge("n0", "n1", "x", "u")
    gm.add_edge("n1", "n2", "x", "u")
    gm.add_coupling_group(list(_NAMES), max_iterations=cap, tolerance=1e-7, diagnostics=True)
    gm.compile()
    return gm


def _dense(q, width, x):
    """``(h, x*, dx*/dtheta)`` in float64 for one Gauss-Seidel pass map, from its float32 constants."""
    g, b, qs, _ = (np.asarray([np.float32(v) for v in arr], np.float64) for arr in _arrays(q, width))
    m = width

    def F(x):
        x = x.reshape(3, m)
        y0 = b[0] + g[0] * x[2] + qs[0] * x[2] ** 2
        y1 = b[1] + g[1] * y0
        return np.concatenate([y0, y1, b[2] + g[2] * y1])

    def J(x):
        x = x.reshape(3, m)
        d0 = g[0] + 2 * qs[0] * x[2]
        out = np.zeros((3 * m, 3 * m))
        for e in range(m):
            out[e, 2 * m + e] = d0[e]
            out[m + e, 2 * m + e] = g[1][e] * d0[e]
            out[2 * m + e, 2 * m + e] = g[2][e] * g[1][e] * d0[e]
        return out

    # The group's norm: each field divided by its own max|field| at x_k.
    s = np.repeat(1.0 / np.max(np.abs(x.reshape(3, m)), axis=1), m)
    eye = np.eye(3 * m)
    Js = s[:, None] * J(x) / s[None, :]
    R = np.linalg.inv(eye - Js)
    delta_s = R @ (s * (F(x) - x))
    delta = delta_s / s
    jac_change = s * ((J(x + delta) - J(x)) @ delta)
    h = np.linalg.norm(R, 2) * np.linalg.norm(jac_change) / np.linalg.norm(delta_s)
    xs = x.copy()
    for _ in range(100):
        xs = xs - np.linalg.solve(eye - J(xs), xs - F(xs))
    return h, xs, np.linalg.inv(eye - J(xs))


#: ``(q, cap)`` and the float64 ``h`` at the returned state, for the record.
UNUSABLE = [(-0.05, 5), (-0.10, 5), (-0.20, 4), (-0.40, 3)]     # h = 0.755, 0.674, 1.51, 0.916
USABLE = [(-0.10, 6), (-0.20, 6), (-0.40, 5)]                    # h = 0.368, 0.199, 0.289


def _run(q, cap, width):
    gm = _ring(q, cap, width)
    step, st0, ext, params = gm._compiled_step, gm._state, gm._default_external_inputs(), gm.params
    out = step(st0, ext, params)
    gm._store_state(out)
    d = gm.coupling_diagnostics()["n0+n1+n2"]
    x = np.concatenate([np.asarray(out[nm]["x"], np.float64) for nm in _NAMES])
    return gm, step, st0, ext, params, out, d, x


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("q, cap", UNUSABLE)
def test_the_bound_is_unusable_where_the_full_norm_puts_h_above_one_half(q, cap, width):
    *_, d, x = _run(q, cap, width)
    h, _, _ = _dense(q, width, x)
    assert h > 0.6, ("fixture premise: h well above one half", h)
    assert d["gradient_bound_usable"] is False, (h, dict(d))
    assert np.isinf(d["gradient_relative_error_bound"]), (h, dict(d))


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("q, cap", USABLE)
def test_the_bound_is_usable_and_holds_where_the_full_norm_puts_h_below_one_half(q, cap, width):
    """Against every scalar entry of every ``g`` and ``b``: the IFT gradient vs ``(I - J(x*))^{-1} F_theta``."""
    _, step, st0, ext, params, out, d, x = _run(q, cap, width)
    h, xs, R = _dense(q, width, x)
    assert h < 0.4, ("fixture premise: h well below one half", h)
    assert d["gradient_bound_usable"] is True, (h, dict(d))
    m = width
    s = np.repeat(1.0 / np.max(np.abs(x.reshape(3, m)), axis=1), m)
    g = [np.asarray(np.float32(v), np.float64) for v in _G]
    xs3 = xs.reshape(3, m)
    worst = 0.0
    for nm, i in zip(_NAMES, range(3)):
        for p in ("g", "b"):
            for e in range(m):
                tang = jax.tree.map(jnp.zeros_like, params)
                tang["nodes"][nm][p] = tang["nodes"][nm][p].at[e].set(1.0)
                _, dout = jax.jvp(lambda pp: step(st0, ext, pp), (params,), (tang,))
                g_k = np.concatenate([np.asarray(dout[n]["x"], np.float64) for n in _NAMES])
                # dF/dtheta of the pass map at x*: the entry's own column of
                # the chain n_i -> n_{i+1} -> ... -> n2.
                src = {0: xs3[2, e], 1: xs3[0, e], 2: xs3[1, e]}[i] if p == "g" else 1.0
                col = np.zeros(3 * m)
                col[i * m + e] = src
                if i < 1:
                    col[m + e] = float(g[1]) * col[e]
                if i < 2:
                    col[2 * m + e] = float(g[2]) * col[m + e]
                t_star = R @ col
                worst = max(worst, float(np.linalg.norm(s * (g_k - t_star)) / np.linalg.norm(s * g_k)))
    assert d["gradient_relative_error_bound"] >= worst, (d["gradient_relative_error_bound"], worst, h)
