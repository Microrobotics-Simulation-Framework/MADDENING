"""The gradient bound applies the resolvent to each secant exactly, and probes each entry.

``gradient_relative_error_bound`` is ``distance * ||(I - J)^{-1} secant||
/ (||delta|| * ||t||)`` per probe.  Until 0.4.0's round-5 fix it took
``amplification * ||secant||`` instead, with ``amplification`` the Arnoldi
resolvent norm -- ``(I - J)^{-1}`` restricted to the Krylov space
``span(v0, r)``.  The residual lies in that space; a secant need not.  On a
Gauss-Seidel ring of three scalar relays (rank-one Jacobian) stopped at
three passes the restricted norm was 8.57 where the full one is 45.2, and
the bound read 0.19x the true relative error of the gradient in a gain,
with ``gradient_bound_usable=True`` (the round-5 audit's designs, below).
And an array-valued constant was probed along one |c|-weighted random
direction, which its large entry dominated: on a two-entry gain whose
small entry's gradient was 28x off, the bound read 1.2 (23.6x short).
Each scalar entry is now its own probe, up to
``GRADIENT_PROBE_ENTRY_LIMIT`` entries per constant; a larger constant is
probed as a whole and ``coupling_report()`` names it.

Exact references: float64 dense solves of the linear coupled system.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GRADIENT_PROBE_ENTRY_LIMIT, GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode


class _Lin(SimulationNode):
    """``x <- a * x_pre + G @ u + b`` with ``G``, ``a``, ``b`` parameters."""

    def __init__(self, name, a, G, b, x0):
        G = np.atleast_2d(np.asarray(G, np.float32))
        b = np.atleast_1d(np.asarray(b, np.float32))
        super().__init__(name, 1.0, a=jnp.float32(a), G=jnp.asarray(G), b=jnp.asarray(b))
        self._x0 = np.atleast_1d(np.asarray(x0, np.float32))
        self._m = G.shape[1]

    def initial_state(self):
        return {"x": jnp.asarray(self._x0)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._m,), dtype=jnp.float32,
                                       default=jnp.zeros(self._m, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": p["a"] * state["x"] + p["G"] @ boundary_inputs["u"] + p["b"]}


#: The audit's two scalar rings (``repro_gradient_bound_restricted_resolvent.py``).
DESIGNS = {
    "design-1": dict(g=[-12.137987144851225, 0.03209990587719496, -2.2631998902941737],
                     a=[-0.27253763571799705, 0.33819271091271386, -0.467254282363038],
                     b=[0.3133808845080941, -1.9064101527339117, -7.043231919131743],
                     xp=[1.7952182122084204, -0.66232274948191, 0.7460887496060612]),
    "design-2": dict(g=[-16.933654872794946, 0.37028521444489765, -0.15017834144620343],
                     a=[-0.335199409298542, -0.06077049309210725, 0.193372263797991],
                     b=[-5.29958273575953, -6.136493989756406, -0.5895405902075863],
                     xp=[-0.4538054566644179, -0.5431879807054575, -2.2646394410021444]),
}
_RING = ["n0", "n1", "n2"]


def _ring(design, cap):
    d = DESIGNS[design]
    gm = GraphManager()
    for i, nm in enumerate(_RING):
        gm.add_node(_Lin(nm, d["a"][i], [[d["g"][i]]], [d["b"][i]], [d["xp"][i]]))
    gm.add_edge("n2", "n0", "x", "u")
    gm.add_edge("n0", "n1", "x", "u")
    gm.add_edge("n1", "n2", "x", "u")
    gm.add_coupling_group(_RING, max_iterations=cap, tolerance=1e-6, diagnostics=True)
    gm.compile()
    return gm


def _f32(v):
    return float(np.float32(v))


def _ring_exact(design):
    """``(x*, (I - M)^{-1})`` of the ring, float64, from its float32 constants."""
    d = DESIGNS[design]
    a = [_f32(v) for v in d["a"]]
    g = [_f32(v) for v in d["g"]]
    b = [_f32(v) for v in d["b"]]
    xp = [_f32(v) for v in d["xp"]]
    M = np.zeros((3, 3))
    M[0, 2], M[1, 0], M[2, 1] = g[0], g[1], g[2]
    c = np.array([a[i] * xp[i] + b[i] for i in range(3)])
    R = np.linalg.inv(np.eye(3) - M)
    return R @ c, R


@pytest.mark.parametrize("design", sorted(DESIGNS))
@pytest.mark.parametrize("cap", [3, 5])
def test_a_usable_gradient_bound_covers_every_gain_of_a_rank_one_ring(design, cap):
    """Each relay's gain: the IFT gradient against ``(I - M)^{-1} e_i u_i*``.

    Before the fix: 0.19x (design-1, cap 3), 0.53x (cap 5), 0.36x and
    0.82x (design-2), all with ``gradient_bound_usable=True``.
    """
    gm = _ring(design, cap)
    step, st0, ext, params = gm._compiled_step, gm._state, gm._default_external_inputs(), gm.params
    out = step(st0, ext, params)
    gm._store_state(out)
    d = gm.coupling_diagnostics()["n0+n1+n2"]
    x = np.array([float(out[nm]["x"][0]) for nm in _RING])
    xs, R = _ring_exact(design)
    w = 1.0 / np.abs(x)
    src = {"n0": 2, "n1": 0, "n2": 1}
    worst = 0.0
    for i, nm in enumerate(_RING):
        tang = jax.tree.map(jnp.zeros_like, params)
        tang["nodes"][nm]["G"] = jnp.ones_like(params["nodes"][nm]["G"])
        _, dout = jax.jvp(lambda p: step(st0, ext, p), (params,), (tang,))
        g_k = np.array([float(dout[m]["x"][0]) for m in _RING])
        e = np.zeros(3)
        e[i] = xs[src[nm]]
        t_star = R @ e
        worst = max(worst, float(np.linalg.norm(w * (g_k - t_star)) / np.linalg.norm(w * g_k)))
    assert d["gradient_bound_usable"], dict(d)
    assert d["gradient_relative_error_bound"] >= worst, (d["gradient_relative_error_bound"], worst)


def _field_pair(cap):
    """The audit's two-node group: ``A <- G_A @ x_B`` with a 1x2 gain, ``B`` relays ``A``."""
    gm = GraphManager()
    gm.add_node(_Lin("A", 0.0, [[1.0, 0.9]], [0.0], [0.0]))
    gm.add_node(_Lin("B", 0.0, [[0.0], [1.0]], [100.0, -99.0], [0.0, 0.0]))
    gm.add_edge("B", "A", "x", "u")
    gm.add_edge("A", "B", "x", "u")
    gm.add_coupling_group(["A", "B"], max_iterations=cap, tolerance=1e-7, diagnostics=True)
    gm.compile()
    return gm


@pytest.mark.parametrize("cap", [20, 24, 40])
def test_each_entry_of_an_array_gain_is_its_own_probe(cap):
    """``d x / d G_A[0, 1]``: 1.6-23.6x the old one-direction bound; covered now."""
    gm = _field_pair(cap)
    step, st0, ext, params = gm._compiled_step, gm._state, gm._default_external_inputs(), gm.params
    out = step(st0, ext, params)
    gm._store_state(out)
    d = gm.coupling_diagnostics()["A+B"]
    xa, xb = (np.asarray(out[m]["x"], np.float64) for m in ("A", "B"))
    xs_b = np.array([100.0, 10.0])        # the fixed point of B: (100, x_A* - 99) with x_A* = 109
    w = np.concatenate([1.0 / np.abs(xa), 1.0 / np.max(np.abs(xb)) * np.ones(2)])
    worst = 0.0
    for j in range(2):
        tang = jax.tree.map(jnp.zeros_like, params)
        tang["nodes"]["A"]["G"] = tang["nodes"]["A"]["G"].at[0, j].set(1.0)
        _, dout = jax.jvp(lambda p: step(st0, ext, p), (params,), (tang,))
        g_k = np.concatenate([np.asarray(dout["A"]["x"], np.float64), np.asarray(dout["B"]["x"], np.float64)])
        # d x*/d G_A[0, j] = (I - M)^{-1} e_A x_B*[j]; with M's loop gain 0.9 (A reads B[1]
        # with 0.9, B[1] reads A with 1): the A entry is x_B*[j] / (1 - 0.9), B[1] the same.
        val = xs_b[j] / (1.0 - float(np.float32(0.9)))
        t_star = np.array([val, 0.0, val])
        worst = max(worst, float(np.linalg.norm(w * (g_k - t_star)) / np.linalg.norm(w * g_k)))
    assert d["gradient_bound_usable"], dict(d)
    assert d["gradient_relative_error_bound"] >= worst, (d["gradient_relative_error_bound"], worst)


class _Wide(SimulationNode):
    """A field of ``n`` entries with a gain vector ``g`` (``n`` entries): ``x <- g * u + c``."""

    def __init__(self, name, n, g):
        super().__init__(name, 1.0, g=jnp.full((n,), g, jnp.float32), c=jnp.ones(n, jnp.float32))
        self._n = n

    def initial_state(self):
        return {"x": jnp.zeros(self._n, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._n,), dtype=jnp.float32,
                                       default=jnp.zeros(self._n, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": p["g"] * boundary_inputs["u"] + p["c"]}


@pytest.mark.parametrize("n, whole", [(GRADIENT_PROBE_ENTRY_LIMIT, False),
                                      (GRADIENT_PROBE_ENTRY_LIMIT + 1, True)])
def test_a_constant_above_the_entry_limit_is_probed_whole_and_named(n, whole):
    """``coupling_report()`` names each constant probed as a whole; none at the limit."""
    gm = GraphManager()
    gm.add_node(_Wide("p", n, 0.5))
    gm.add_node(_Wide("q", n, 0.9))
    gm.add_edge("p", "q", "x", "u")
    gm.add_edge("q", "p", "x", "u")
    gm.add_coupling_group(["p", "q"], max_iterations=4, tolerance=1e-6, diagnostics=True)
    gm.compile()
    gm.step()
    assert gm.coupling_diagnostics()["p+q"]["gradient_bound_usable"]
    row = next(iter(gm.coupling_report()))
    flagged = [f for f in row["flags"] if "probed as a whole" in f]
    if not whole:
        assert not flagged, flagged
        return
    assert len(flagged) == 1, row["flags"]
    for name in ("p.g", "q.g", "p.c", "q.c"):
        assert f"{name} ({n} entries)" in flagged[0], flagged[0]
