"""``run_adaptive_scan``'s gradient is finite where step doubling's two estimates agree.

``run_adaptive_scan`` differentiates through its own step-size controller:
the next timestep is ``dt * safety * max(error_norm, 1e-10)**(-1/(order+1))``
and ``error_norm`` is the RMS of the step-doubling differences, a square root.
When the full step and the two half steps land on one value -- a memoryless
or steady state, a coupled group already at its fixed point, a clock whose
``dt/2 + dt/2`` is exact -- the RMS is exactly zero, the square root's
derivative there is infinite, and the backward pass multiplied it by the zero
that ``max`` sends back on its flat side: NaN in every gradient through the
scan (MADD-ANO-160).  The parameters never moved the step size there, so the
right derivative of the controller's branch is zero and the gradient is the
state's own.

The fixture is MADD-ANO-160's: two scalar relays ``x <- 0.6 u + b`` reading
each other, each with a clock ``t <- t + dt``, ``b`` constant.  The coupled
fixed point is ``x_a = (b_a + 0.6 b_b) / (1 - 0.36)``, so ``d x_a / d b_a`` is
``1 / 0.64 = 1.5625`` at every step and every step size.  The gradient is
checked against that closed form, against ``run_scan``'s gradient, and
against a central difference of ``run_adaptive`` (the host loop, which takes
the same acceptance rule and is not traced at all).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.simulation.adaptive import _tree_error_norm

DT = 0.125
T_END = 3 * DT
KW = dict(dt_initial=DT, dt_max=2 * DT)
GAIN = 0.6
#: ``d x_a / d b_a`` of the coupled pair: ``1 / (1 - GAIN**2)``.
SENSITIVITY = 1.0 / (1.0 - GAIN * GAIN)


class _Relay(SimulationNode):
    """``x <- g u + b + s t``, the clock ``t`` advanced by the step's ``dt``.

    Memoryless in ``x``; with ``s = 0`` the full step and the two half steps
    give the same ``x`` exactly, and ``t + dt`` equals ``t + dt/2 + dt/2``
    for these power-of-two steps.
    """

    def __init__(self, name, *, g, b, s=0.0):
        super().__init__(name, DT, g=jnp.float32(g), b=jnp.float32(b), s=jnp.float32(s))

    def initial_state(self):
        return {"x": jnp.zeros((), jnp.float32), "t": jnp.zeros((), jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.zeros((), jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        t = state["t"] + dt
        u = boundary_inputs.get("u", jnp.zeros((), jnp.float32))
        return {"x": p["g"] * u + p["b"] + p["s"] * t, "t": t}


def _pair(acceleration):
    gm = GraphManager()
    gm.add_node(_Relay("a", g=GAIN, b=1.0))
    gm.add_node(_Relay("b", g=GAIN, b=0.5))
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    gm.add_coupling_group(["a", "b"], acceleration=acceleration, tolerance=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


def _with_b(p, value):
    q = jax.tree.map(lambda v: v, p)
    q["nodes"]["a"]["b"] = jnp.float32(value)
    return q


def _adaptive_scan_xa(gm):
    def xa(p):
        gm.reset_state()
        final, _h, _i = gm.run_adaptive_scan(T_END, max_steps=16, params=p, **KW)
        return final["a"]["x"]
    return xa


def _run_adaptive_xa(gm, p):
    gm.reset_state()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final, _info = gm.run_adaptive(T_END, params=p, **KW)
    return float(np.asarray(final["a"]["x"], np.float64))


@pytest.fixture(scope="module")
def aitken_pair():
    return _pair("aitken")


@pytest.fixture(scope="module")
def aitken_gradient(aitken_pair):
    """The gradient of ``x_a`` through ``run_adaptive_scan``, taken once."""
    gm = aitken_pair
    p = jax.tree.map(lambda v: v, gm.params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return jax.grad(_adaptive_scan_xa(gm))(p)


def test_the_error_norm_has_a_finite_derivative_where_the_estimates_agree():
    """The unit: two identical estimates give a norm of zero and a zero
    derivative, not ``inf`` (whose product with any cotangent was NaN); two
    that differ keep the RMS's own derivative."""
    fine = {"n": {"x": jnp.array([1.0, -2.0], jnp.float32), "t": jnp.float32(0.5)}}
    atol, rtol = 1e-6, 1e-3
    g = jax.grad(lambda f: _tree_error_norm(f, fine, atol, rtol))(fine)
    for leaf in jax.tree.leaves(g):
        assert np.all(np.asarray(leaf) == 0.0), g
    assert float(_tree_error_norm(fine, fine, atol, rtol)) == 0.0

    coarse = {"n": {"x": jnp.array([1.001, -2.0], jnp.float32), "t": jnp.float32(0.5)}}
    value, g = jax.value_and_grad(lambda f: _tree_error_norm(f, coarse, atol, rtol))(fine)
    eps = 1e-3

    def at(dx):
        f = {"n": {"x": fine["n"]["x"].at[0].add(dx), "t": fine["n"]["t"]}}
        return float(_tree_error_norm(f, coarse, atol, rtol))

    assert float(value) > 0.0
    fd = (at(eps * 1e-2) - at(-eps * 1e-2)) / (2 * eps * 1e-2)
    assert np.isclose(float(g["n"]["x"][0]), fd, rtol=1e-2), (g, fd)


def test_the_adaptive_scan_gradient_of_a_group_at_its_fixed_point_is_the_coupled_sensitivity(
        aitken_pair, aitken_gradient):
    """MADD-ANO-160's reproducer: ``1 / (1 - 0.36)`` under ``"aitken"``, as
    ``run_scan`` gives; the scan used to return NaN."""
    gm, g = aitken_pair, aitken_gradient
    p = jax.tree.map(lambda v: v, gm.params)

    def xa_scan(q):
        gm.reset_state()
        return gm.run_scan(3, params=q)["a"]["x"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs = jax.grad(xa_scan)(p)
    for leaf in jax.tree.leaves(g):
        assert np.all(np.isfinite(np.asarray(leaf))), g
    got = float(g["nodes"]["a"]["b"])
    assert got == pytest.approx(SENSITIVITY, rel=1e-5)
    assert got == pytest.approx(float(gs["nodes"]["a"]["b"]), rel=1e-6)


def test_the_adaptive_scan_gradient_matches_a_central_difference_of_run_adaptive(
        aitken_pair, aitken_gradient):
    """The same derivative from the host stepper's forward, which no trace
    touches: a central difference of ``run_adaptive`` in ``b_a``."""
    gm = aitken_pair
    p = jax.tree.map(lambda v: v, gm.params)
    got = float(aitken_gradient["nodes"]["a"]["b"])
    h = 2.0 ** -6
    b = float(p["nodes"]["a"]["b"])
    fd = (_run_adaptive_xa(gm, _with_b(p, b + h)) - _run_adaptive_xa(gm, _with_b(p, b - h))) / (2 * h)
    assert fd == pytest.approx(SENSITIVITY, rel=1e-4)
    assert got == pytest.approx(fd, rel=1e-4)


# Slow: each acceleration compiles its own scan and its transpose (~3 s on
# three cores).  "aitken", the reproducer's, stays on every push.
# Per push: tests/core/test_adaptive_scan_gradient_where_the_error_estimate_is_exact.py::test_the_adaptive_scan_gradient_of_a_group_at_its_fixed_point_is_the_coupled_sensitivity
@pytest.mark.slow
@pytest.mark.parametrize("acceleration", ["iqn-ils", "none"])
def test_every_acceleration_gives_the_coupled_sensitivity_through_the_adaptive_scan(acceleration):
    """The accelerations that reach the fixed point exactly (the IQN pair, as
    ``"aitken"``) and one that stops short of it (``"none"``, whose estimates
    differ by its truncation) give the same derivative."""
    gm = _pair(acceleration)
    p = jax.tree.map(lambda v: v, gm.params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = float(jax.grad(_adaptive_scan_xa(gm))(p)["nodes"]["a"]["b"])
    assert got == pytest.approx(SENSITIVITY, rel=1e-4)


def test_an_uncoupled_node_whose_estimates_agree_has_a_finite_adaptive_scan_gradient():
    """No coupling group at all: one memoryless relay with a clock.  The
    controller's branch is the graph's, not the coupling solve's, so the NaN
    reached every graph whose step doubling was exact."""
    gm = GraphManager()
    gm.add_node(_Relay("a", g=0.0, b=2.0))
    gm.compile()
    p = jax.tree.map(lambda v: v, gm.params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = jax.grad(_adaptive_scan_xa(gm))(p)
    assert float(g["nodes"]["a"]["b"]) == 1.0


# Slow: a second graph's scan and transpose (~4 s on three cores).
# Per push: tests/core/test_adaptive_scan_gradient_where_the_error_estimate_is_exact.py::test_the_error_norm_has_a_finite_derivative_where_the_estimates_agree
@pytest.mark.slow
def test_a_moving_forcing_keeps_the_controllers_derivative():
    """Where the estimates differ (a forcing that moves with the clock), the
    gradient still matches a central difference of ``run_adaptive``: the
    guard changes only the zero-error branch."""
    gm = GraphManager()
    gm.add_node(_Relay("a", g=GAIN, b=1.0, s=0.5))
    gm.add_node(_Relay("b", g=GAIN, b=0.5, s=-0.25))
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    gm.add_coupling_group(["a", "b"], acceleration="aitken", tolerance=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
        p = jax.tree.map(lambda v: v, gm.params)
        got = float(jax.grad(_adaptive_scan_xa(gm))(p)["nodes"]["a"]["b"])
    h = 2.0 ** -6
    b = float(p["nodes"]["a"]["b"])
    fd = (_run_adaptive_xa(gm, _with_b(p, b + h)) - _run_adaptive_xa(gm, _with_b(p, b - h))) / (2 * h)
    assert np.isfinite(got)
    assert got == pytest.approx(fd, rel=1e-3)
