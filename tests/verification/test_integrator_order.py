"""Measured order of accuracy of the explicit integrators, with a
boundary input that genuinely varies in time.

``maddening.core.simulation.integrators`` names its methods after their
classical orders, and a name is a numerical claim.  The claim is
conditional, because none of the three steppers is given a time: every
stage sees the one ``boundary_inputs`` object the caller passed for the
step.  So the claimed order is the order for

.. math::  dx/dt = f(x, u),   u fixed over the step

and this module measures what happens on both sides of that condition.

Three kinds of test, because no one of them is sufficient:

**The frozen-input ladder** pins the degraded behaviour -- 1st order for
all three methods, ``MADD-ANO-014`` -- so it cannot quietly become
untrue in either direction while the docstrings still describe it.

**The stage-time ladder** pins 1 / 2 / 4, reached with no new API by
carrying time as a state field with derivative 1.  This is the test that
sees a wrong Butcher coefficient as a lost order.

**The stability-polynomial identity** pins each method's coefficients
exactly, because a refinement ladder cannot tell one first-order scheme
from another: in the frozen-input study below, ``euler``, ``heun`` and
``rk4`` all measure ~1.02 and their errors agree to three digits.  A
ladder answers "what order is this?", never "which scheme is this?".
One step of ``x' = a x`` answers the second question exactly: the result
is ``x0`` times the method's stability polynomial ``R(z)``, ``z = a*dt``,
and every stage weight appears in it.

Precision: the studies run under ``jax_enable_x64``.  A 4th-order ladder
exhausts float32 immediately -- the finest level below reaches 1.5e-10
relative, which is not representable as a *converging* error in single
precision.
"""

import contextlib
import math

import jax
import jax.numpy as jnp
import pytest

from maddening.core.node import SimulationNode
from maddening.core.simulation.integrators import (
    euler_step,
    heun_step,
    integrate_node,
    rk4_step,
)
from maddening.testing.mms import (
    RefinementAxis,
    check_order,
    measure_order,
)

# Symmetric and tighter than the harness default excess of 1.0.  The
# point of every ladder here is to separate order 1 from order 2 from
# order 4; a band of [expected - 0.25, expected + 1.0] around 1 admits
# 1.999, which is exactly the outcome these tests exist to distinguish
# from 1.02.  The measured orders below sit within 0.03 of theory on the
# finest pair, so 0.25 either way is ~8x the observed wobble.
_BAND = {"shortfall": 0.25, "excess": 0.25}

_LEVELS = (10, 20, 40, 80, 160)
_LAMBDA = 1.0
_OMEGA = 2.0 * math.pi
#: sin(2*pi*0.75) = -1, so the endpoint of the manufactured trajectory is
#: O(1) and the relative error is not a ratio against something near zero.
_T_FINAL = 0.75


@contextlib.contextmanager
def _float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def float64():
    with _float64():
        yield


# ---------------------------------------------------------------------------
# The manufactured problem
# ---------------------------------------------------------------------------
#
# Manufacture the trajectory, then read off the forcing that makes it
# exact -- the ODE analogue of ManufacturedSolution.  For
#
#     x' = -lambda*x + u(t),   x(0) = 0
#
# the input that makes x*(t) = sin(omega*t) an exact solution is
#
#     u(t) = omega*cos(omega*t) + lambda*sin(omega*t).
#
# Both terms matter: the -lambda*x term is what the stage arithmetic can
# integrate to full order, and u(t) is what it cannot see at stage times.


def _exact(t):
    return jnp.sin(_OMEGA * jnp.asarray(t))


def _forcing(t):
    t = jnp.asarray(t)
    return _OMEGA * jnp.cos(_OMEGA * t) + _LAMBDA * jnp.sin(_OMEGA * t)


def _derivatives(state, boundary_inputs):
    return {"x": -_LAMBDA * state["x"] + boundary_inputs["u"]}


class _ForcedDecayNode(SimulationNode):
    """Minimal node exposing the manufactured ODE through ``derivatives``.

    Local to this module on purpose: the integrators are generic over any
    ``derivatives_fn``, and pinning their order against a shipped node
    would measure that node's choices too.
    """

    def initial_state(self):
        return {"x": jnp.asarray(0.0)}

    def update(self, state, boundary_inputs, dt):
        return euler_step(self.derivatives, state, boundary_inputs, dt)

    def derivatives(self, state, boundary_inputs):
        return _derivatives(state, boundary_inputs)


def _relative_endpoint_error(x_final):
    reference = float(_exact(_T_FINAL))
    return abs(float(x_final) - reference) / abs(reference)


def _frozen_input_error_at(stepper):
    """Error after driving ``stepper`` with one input value per step.

    The natural way to use this API with a time-varying input, and the
    one that costs the order.
    """

    def error_at(n_steps):
        dt = _T_FINAL / n_steps
        state = {"x": jnp.asarray(0.0)}
        for i in range(n_steps):
            state = stepper(state, {"u": _forcing(i * dt)}, dt)
        return _relative_endpoint_error(state["x"])

    return error_at


def _stage_time_error_at(stepper):
    """Error after driving ``stepper`` with time carried in the state.

    ``time`` has derivative 1, so each stage state arrives holding
    ``t + c_i*dt`` for that method's Butcher node, and the forcing is
    read there.  No integrator API is involved beyond the existing
    ``derivatives_fn``.
    """

    def forced(state, _boundary_inputs):
        rest = {k: v for k, v in state.items() if k != "time"}
        derivs = _derivatives(rest, {"u": _forcing(state["time"])})
        return {**derivs, "time": jnp.asarray(1.0)}

    def error_at(n_steps):
        dt = _T_FINAL / n_steps
        state = {"x": jnp.asarray(0.0), "time": jnp.asarray(0.0)}
        for _ in range(n_steps):
            state = stepper(forced, state, {}, dt)
        return _relative_endpoint_error(state["x"])

    return error_at


def _assert_order(error_at, expected, name):
    measurement = measure_order(
        error_at, _LEVELS, axis=RefinementAxis.TIME,
        h_of=lambda n: _T_FINAL / n,
    )
    result = check_order(measurement, expected, name=name, **_BAND)
    assert result.passed, result.detail


# ---------------------------------------------------------------------------
# Frozen boundary inputs: every method is 1st order (MADD-ANO-014)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,stepper",
    [("euler", euler_step), ("heun", heun_step), ("rk4", rk4_step)],
)
def test_a_boundary_input_frozen_across_stages_caps_every_method_at_first_order(
    float64, method, stepper
):
    """The documented limitation, measured rather than asserted.

    A zero-order hold on ``u`` injects an O(dt^2) error per step, hence
    O(dt) globally, no matter how the stages combine.  Pinned so that a
    change which silently made ``rk4`` second order here would fail and
    force the docstrings and MADD-ANO-014 to be revisited, and so that a
    change which made it order 0 would fail too.
    """
    _assert_order(
        _frozen_input_error_at(
            lambda state, bi, dt: stepper(_derivatives, state, bi, dt)
        ),
        expected=1.0,
        name=f"{method}_frozen_input_temporal_order",
    )


@pytest.mark.parametrize("method", ["euler", "heun", "rk4"])
def test_integrate_node_inherits_the_frozen_input_first_order(float64, method):
    """``integrate_node`` has nowhere to put a stage time, so it cannot
    do better than the frozen-input order for any method it offers.

    Separate from the function-level test because ``integrate_node`` is
    the entry point whose ``method="rk4"`` default is what a user reads
    as a promise of fourth order.
    """
    node = _ForcedDecayNode(name="forced_decay", timestep=0.01)
    _assert_order(
        _frozen_input_error_at(
            lambda state, bi, dt: integrate_node(node, state, bi, dt, method=method)
        ),
        expected=1.0,
        name=f"integrate_node_{method}_frozen_input_temporal_order",
    )


def test_rk4_with_a_frozen_input_is_no_more_accurate_than_euler(float64):
    """The counter-intuitive half of MADD-ANO-014, pinned.

    Reaching for the higher-order method is not a conservative choice
    here: on this problem RK4's error at the finest level is *larger*
    than Euler's, because the stages refine a term that is no longer
    dominant.  Recorded because a user who knows only "RK4 is more
    accurate" would never test for it.
    """
    n_steps = 160
    euler_err = _frozen_input_error_at(
        lambda s, bi, dt: euler_step(_derivatives, s, bi, dt)
    )(n_steps)
    rk4_err = _frozen_input_error_at(
        lambda s, bi, dt: rk4_step(_derivatives, s, bi, dt)
    )(n_steps)
    assert rk4_err > euler_err, (
        f"RK4 error {rk4_err:.6g} is no longer above Euler's {euler_err:.6g} "
        f"at dt={_T_FINAL / n_steps:g}; if the frozen-input handling changed, "
        f"MADD-ANO-014 and the integrators docstring need revisiting"
    )


# ---------------------------------------------------------------------------
# Time as a state field: the classical orders, with no new API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,stepper,expected",
    [
        ("euler", euler_step, 1.0),
        ("heun", heun_step, 2.0),
        ("rk4", rk4_step, 4.0),
    ],
)
def test_carrying_time_in_the_state_restores_each_methods_classical_order(
    float64, method, stepper, expected
):
    """Same non-autonomous problem, full order, existing signature.

    Each stage state is built as ``state + a_ij*dt*k_j``, so a field with
    derivative 1 arrives holding exactly ``t + c_i*dt``.  This is the
    test that turns a wrong stage coefficient into a lost order.
    """
    _assert_order(
        _stage_time_error_at(stepper),
        expected=expected,
        name=f"{method}_stage_time_temporal_order",
    )


# ---------------------------------------------------------------------------
# Scheme identity: which method is this, not what order is it
# ---------------------------------------------------------------------------

#: R(z) for each method: one step of x' = a*x from x0 multiplies x0 by
#: this, with z = a*dt.  Truncations of exp(z) to the method's order,
#: which for these three methods is also a complete statement of their
#: Butcher coefficients up to the compensations the polynomial cannot
#: see (there are none for a 4-stage explicit method of this shape).
_STABILITY_POLYNOMIAL = {
    "euler": lambda z: 1.0 + z,
    "heun": lambda z: 1.0 + z + z**2 / 2.0,
    "rk4": lambda z: 1.0 + z + z**2 / 2.0 + z**3 / 6.0 + z**4 / 24.0,
}


@pytest.mark.parametrize(
    "method,stepper",
    [("euler", euler_step), ("heun", heun_step), ("rk4", rk4_step)],
)
@pytest.mark.parametrize("z", [-2.5, -0.3, 0.7])
def test_each_stepper_reproduces_its_own_stability_polynomial(
    float64, method, stepper, z
):
    """One step of ``x' = a x`` pins every stage weight, exactly.

    A convergence ladder cannot do this: the frozen-input study above
    measures ~1.02 for all three methods with errors agreeing to three
    significant figures, so a scheme swap passes it untouched.  This
    check is algebraic and closes that hole -- any change to a stage
    offset or an output weight moves ``R(z)`` and fails here, at a
    tolerance of a few ulp rather than a convergence band.

    Three values of ``z``, including one outside RK4's real stability
    interval (``z = -2.5``, where the polynomial is still the right
    answer for one step), so the check cannot be satisfied by a method
    that merely agrees near ``z = 0``.
    """
    dt = 0.25
    a = z / dt
    x0 = 1.5
    out = stepper(
        lambda state, _bi: {"x": a * state["x"]},
        {"x": jnp.asarray(x0)},
        {},
        dt,
    )
    expected = x0 * _STABILITY_POLYNOMIAL[method](z)
    assert float(out["x"]) == pytest.approx(expected, rel=1e-13, abs=1e-15)


def test_the_three_steppers_are_distinguishable_on_a_single_step(float64):
    """Redundant given the polynomials above, and kept anyway.

    It states the property the polynomial test exists to protect -- that
    these are three different schemes -- in a form that survives someone
    editing ``_STABILITY_POLYNOMIAL`` and the implementation together.
    """
    dt = 0.25
    results = {
        name: float(
            stepper(
                lambda state, _bi: {"x": -1.2 * state["x"]},
                {"x": jnp.asarray(1.5)},
                {},
                dt,
            )["x"]
        )
        for name, stepper in (
            ("euler", euler_step), ("heun", heun_step), ("rk4", rk4_step)
        )
    }
    assert len(set(results.values())) == 3, (
        f"two of the three steppers produced the same step: {results}"
    )
