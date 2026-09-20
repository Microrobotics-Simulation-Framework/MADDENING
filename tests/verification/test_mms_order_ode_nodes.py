"""Observed order of convergence for MADDENING's four ODE nodes.

``tests/verification/test_mms_order.py`` put three of the eleven node
types through the Method of Manufactured Solutions harness (``heat``,
``lbm``, ``rigid_body``).  The other eight declared no order and the
harness reported them ``SKIP`` -- honest, but a skip is not a clean
result, and a green run of that file is easy to read as coverage it
does not have.  This module adds the four that are straightforward,
because all four integrate an ODE: ``spring``, ``ball``,
``rigid_body_2d`` and ``heart_pump``.

Each study manufactures a trajectory, derives by automatic
differentiation the forcing that makes that trajectory an exact
solution of the equation the node claims to solve, drives the node
with it, and measures the rate at which the error falls as the
timestep is refined.  All four declare ``temporal=1.0`` and all four
measure it.  None of them declares a spatial order -- there is no grid
-- and the harness is asked to confirm it *skips* that axis rather
than passing it vacuously.

Where the source is injected, and why it differs per node
---------------------------------------------------------
MMS needs a source.  Only two of the four expose one as a force:

* ``rigid_body_2d`` takes ``force`` and ``torque`` directly.
* ``spring`` takes ``anchor_position``, which enters the force
  linearly, so the anchor that makes a chosen trajectory exact can be
  solved for in closed form.
* ``heart_pump`` takes ``backpressure``, which enters the outflow
  linearly; same argument.
* ``ball`` exposes **no forcing input at all** -- its only boundary
  input is ``table_position``, a collision surface.  Its source has to
  go in through the ``gravity`` parameter, which *is* the acceleration.
  That is a finding about the node's coupling surface, not about MMS:
  a caller cannot drive a BallNode with an external force.

Three defects this module records
---------------------------------
None of them is an order shortfall -- all four nodes meet their
declared order.  Each is pinned below as a strict xfail so that fixing
the node turns it into an XPASS that has to be dealt with:

* ``BallNode`` names forward Euler and implements semi-implicit
  (symplectic) Euler (MADD-ANO-011);
* ``HeartPumpNode`` names forward Euler and samples its inflow
  waveform at the *end* of the step, disagreeing with its own
  ``derivatives()`` (MADD-ANO-012);
* ``HeartPumpNode`` downcasts ``backpressure`` to float32, which puts
  a floor under any convergence study or adjoint through that coupling
  variable (MADD-ANO-013).

The first two are invisible to an order study -- forward and
semi-implicit Euler are both 1st order -- which is why each has its
own scheme-identity test rather than being left to the ladder.

Precision: the studies run under ``jax_enable_x64``, for the reason
``test_mms_order.py`` gives -- the observed order is a ratio of small
numbers and float32 runs out of signal before a refinement ladder
does.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import contextlib  # noqa: E402
import math  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from maddening.core.compliance.validation import (  # noqa: E402
    BenchmarkType,
    verification_benchmark,
)
from maddening.core.simulation.integrators import euler_step  # noqa: E402
from maddening.nodes.ball import BallNode  # noqa: E402
from maddening.nodes.heart_pump import HeartPumpNode  # noqa: E402
from maddening.nodes.rigid_body_2d import RigidBody2DNode  # noqa: E402
from maddening.nodes.spring import SpringDamperNode  # noqa: E402
from maddening.testing.mms import (  # noqa: E402
    RefinementAxis,
    UndeclaredOrderError,
    assert_node_order_verified,
    declared_order,
    manufactured_acceleration,
    measure_order,
    verify_node_order,
)

# --------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _float64():
    """Run the body in double precision, restoring the global setting."""
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


def _relative_l2(got: dict, exact: dict) -> float:
    """One relative error over every field of a state, jointly.

    Taken over the whole state rather than one field: semi-implicit
    Euler integrates a position at 2nd order when the force does not
    depend on the state, so a position-only norm would measure the
    wrong thing (the same trap ``test_mms_order.py`` records for
    RigidBodyNode).
    """
    num = sum(float(np.sum((np.asarray(got[k], np.float64) - v) ** 2))
              for k, v in exact.items())
    den = sum(float(np.sum(v**2)) for v in exact.values())
    return math.sqrt(num / den)


# The two timestep perturbations the mutation tests below apply.  Each is
# a defect the harness has to *catch*; see ``TestTheLaddersCanFail``.


def _scaled(dt, factor):
    """A timestep mis-scaled by a constant factor: the classic unit bug."""
    return dt * factor


def _order_halving(dt):
    """A timestep that drifts with the resolution: global order 1/2.

    Advancing by ``dt * (1 + sqrt(dt))`` while counting time as ``dt``
    lands the run at ``T * (1 + sqrt(dt))`` instead of ``T``, so the
    error falls like ``sqrt(dt)``.  Unlike a plain mis-scaled timestep
    this keeps the ladder monotone, so it tests the acceptance *band*
    rather than the non-convergence guard.
    """
    return dt * (1.0 + math.sqrt(dt))


# --------------------------------------------------------------------------
# SpringDamperNode -- temporal order
# --------------------------------------------------------------------------

_K, _C, _M, _REST = 40.0, 0.7, 1.3, 0.5


def _spring_x(t):
    return 0.3 * jnp.sin(3.0 * t) + 0.1 * jnp.cos(2.0 * t)


_spring_v = jax.jacfwd(_spring_x)
_spring_a = manufactured_acceleration(_spring_x)


def _spring_anchor(t):
    """The anchor that makes :func:`_spring_x` an exact solution.

    The node integrates ``m x'' = -k (x - a - rest) - c x'``, which is
    linear in the anchor ``a``, so the source can be solved for
    exactly::

        a(t) = x*(t) - rest + (m x*''(t) + c x*'(t)) / k
    """
    return _spring_x(t) - _REST + (_M * _spring_a(t) + _C * _spring_v(t)) / _K


def _spring_error(n_steps, *, t_final=1.0, dt_map=None, freeze_source=False):
    """Relative L2 error of (position, velocity) after ``n_steps`` steps."""
    dt = t_final / n_steps
    node = SpringDamperNode(
        "mms_spring", timestep=dt, stiffness=_K, damping=_C, mass=_M,
        rest_length=_REST,
    )
    step_dt = dt if dt_map is None else dt_map(dt)
    state = {
        "position": jnp.asarray(_spring_x(0.0)),
        "velocity": jnp.asarray(_spring_v(0.0)),
    }

    def step(k, s):
        t = 0.0 if freeze_source else k * dt
        return node.update(s, {"anchor_position": _spring_anchor(t)}, step_dt)

    state = jax.lax.fori_loop(0, n_steps, jax.jit(step), state)
    return _relative_l2(
        state,
        {"position": np.asarray(_spring_x(t_final), np.float64),
         "velocity": np.asarray(_spring_v(t_final), np.float64)},
    )


@verification_benchmark(
    benchmark_id="MADD-VER-009",
    description=(
        "SpringDamperNode temporal order of accuracy by the Method of "
        "Manufactured Solutions: manufactured displacement injected through "
        "anchor_position, the node's own boundary input, timestep refined at "
        "fixed final time"
    ),
    node_type="SpringDamperNode",
    benchmark_type=BenchmarkType.MANUFACTURED_SOLUTION,
    acceptance_criteria=(
        "Observed temporal order over the finest pair of a 100/200/400/800 "
        "step ladder within [-0.25, +1.0] of the declared 1.0 "
        "(measured: 1.029)"
    ),
    references=(
        "Hairer2006: Geometric Numerical Integration, Ch. VI (symplectic Euler)",
        "Roache2002: Code Verification by the Method of Manufactured Solutions",
    ),
)
def test_spring_converges_at_its_declared_temporal_order(float64):
    """Semi-implicit Euler meets its claim of 1st order."""
    node = SpringDamperNode("mms_spring", timestep=0.01, stiffness=_K,
                            damping=_C, mass=_M, rest_length=_REST)
    assert_node_order_verified(
        node,
        axis=RefinementAxis.TIME,
        error_at=_spring_error,
        levels=(100, 200, 400, 800),
    )


# --------------------------------------------------------------------------
# BallNode -- temporal order
# --------------------------------------------------------------------------


def _ball_x(t):
    return 0.4 * jnp.sin(2.5 * t) + 1.0 * t + 2.0


_ball_v = jax.jacfwd(_ball_x)
_ball_a = manufactured_acceleration(_ball_x)


def _ball_error(n_steps, *, t_final=1.0, dt_map=None, freeze_source=False):
    """Relative L2 error of (position, velocity) after ``n_steps`` steps.

    The manufactured acceleration goes in through ``params["gravity"]``,
    not through a boundary input: BallNode exposes none that carries a
    force (see the module docstring).  ``gravity`` *is* the
    acceleration, so ``gravity = x*''(t)`` makes the trajectory an exact
    solution of ``dv/dt = g``, ``dx/dt = v``.  No ``table_position`` is
    supplied, so the collision branch stays out of the study: a contact
    is a non-smooth event and the node claims no order across one.
    """
    dt = t_final / n_steps
    node = BallNode("mms_ball", timestep=dt)
    step_dt = dt if dt_map is None else dt_map(dt)
    state = {
        "position": jnp.asarray(_ball_x(0.0)),
        "velocity": jnp.asarray(_ball_v(0.0)),
    }

    def step(k, s):
        t = 0.0 if freeze_source else k * dt
        return node.update(s, {}, step_dt, params={"gravity": _ball_a(t)})

    state = jax.lax.fori_loop(0, n_steps, jax.jit(step), state)
    return _relative_l2(
        state,
        {"position": np.asarray(_ball_x(t_final), np.float64),
         "velocity": np.asarray(_ball_v(t_final), np.float64)},
    )


@verification_benchmark(
    benchmark_id="MADD-VER-010",
    description=(
        "BallNode temporal order of accuracy by the Method of Manufactured "
        "Solutions: manufactured trajectory injected as a time-varying "
        "gravity parameter (the node exposes no force input), timestep "
        "refined at fixed final time, collision-free regime"
    ),
    node_type="BallNode",
    benchmark_type=BenchmarkType.MANUFACTURED_SOLUTION,
    acceptance_criteria=(
        "Observed temporal order over the finest pair of a 100/200/400/800 "
        "step ladder within [-0.25, +1.0] of the declared 1.0 "
        "(measured: 1.002).  Smooth regime only: no table_position is "
        "supplied, so no order is claimed across a collision."
    ),
    references=(
        "Hairer2006: Geometric Numerical Integration, Ch. VI (symplectic Euler)",
        "Roache2002: Code Verification by the Method of Manufactured Solutions",
    ),
)
def test_ball_converges_at_its_declared_temporal_order(float64):
    """The ball's 1st-order claim holds in the collision-free regime."""
    node = BallNode("mms_ball", timestep=0.01)
    assert_node_order_verified(
        node,
        axis=RefinementAxis.TIME,
        error_at=_ball_error,
        levels=(100, 200, 400, 800),
    )


# --------------------------------------------------------------------------
# RigidBody2DNode -- temporal order
# --------------------------------------------------------------------------

_RB2_M, _RB2_I = 1.7, 2.3


def _rb2_x(t):
    return jnp.stack([0.3 * jnp.sin(2.0 * t), 0.2 * jnp.cos(3.0 * t)])


def _rb2_angle(t):
    return 0.5 * jnp.sin(1.5 * t) + 0.2 * t


_rb2_v = jax.jacfwd(_rb2_x)
_rb2_a = manufactured_acceleration(_rb2_x)
_rb2_omega = jax.jacfwd(_rb2_angle)
_rb2_alpha = manufactured_acceleration(_rb2_angle)


def _rb2_error(n_steps, *, t_final=1.0, dt_map=None, freeze_source=False):
    """Relative L2 error over the whole 2D state after ``n_steps`` steps.

    Gravity is zeroed so the manufactured force and torque are the only
    drive; both are the node's own additive boundary inputs, sampled at
    the start of each step, which is where an explicit scheme reads its
    source.
    """
    dt = t_final / n_steps
    node = RigidBody2DNode(
        "mms_rb2", timestep=dt, mass=_RB2_M, inertia=_RB2_I,
        gravity=(0.0, 0.0),
    )
    step_dt = dt if dt_map is None else dt_map(dt)
    state = {
        "x": jnp.asarray(_rb2_x(0.0)),
        "angle": jnp.asarray(_rb2_angle(0.0)),
        "v": jnp.asarray(_rb2_v(0.0)),
        "omega": jnp.asarray(_rb2_omega(0.0)),
    }

    def step(k, s):
        t = 0.0 if freeze_source else k * dt
        return node.update(
            s,
            {"force": _RB2_M * _rb2_a(t), "torque": _RB2_I * _rb2_alpha(t)},
            step_dt,
        )

    state = jax.lax.fori_loop(0, n_steps, jax.jit(step), state)
    return _relative_l2(
        state,
        {"x": np.asarray(_rb2_x(t_final), np.float64),
         "angle": np.asarray(_rb2_angle(t_final), np.float64),
         "v": np.asarray(_rb2_v(t_final), np.float64),
         "omega": np.asarray(_rb2_omega(t_final), np.float64)},
    )


@verification_benchmark(
    benchmark_id="MADD-VER-011",
    description=(
        "RigidBody2DNode temporal order of accuracy by the Method of "
        "Manufactured Solutions: manufactured planar trajectory injected as "
        "force and torque, timestep refined at fixed final time"
    ),
    node_type="RigidBody2DNode",
    benchmark_type=BenchmarkType.MANUFACTURED_SOLUTION,
    acceptance_criteria=(
        "Observed temporal order over the finest pair of a 100/200/400/800 "
        "step ladder within [-0.25, +1.0] of the declared 1.0 "
        "(measured: 1.000)"
    ),
    references=(
        "Hairer2006: Geometric Numerical Integration, Ch. VI (symplectic Euler)",
        "Roache2002: Code Verification by the Method of Manufactured Solutions",
    ),
)
def test_rigid_body_2d_converges_at_its_declared_temporal_order(float64):
    """Semi-implicit Euler in both DOF groups meets its claim of 1st order."""
    node = RigidBody2DNode("mms_rb2", timestep=0.01, mass=_RB2_M,
                           inertia=_RB2_I, gravity=(0.0, 0.0))
    assert_node_order_verified(
        node,
        axis=RefinementAxis.TIME,
        error_at=_rb2_error,
        levels=(100, 200, 400, 800),
    )


# --------------------------------------------------------------------------
# HeartPumpNode -- temporal order
# --------------------------------------------------------------------------

_HP = {
    "resistance": 1.1,
    "compliance": 0.9,
    "heart_rate": 72.0,
    "stroke_volume": 70.0,
    "venous_pressure": 0.0,
    "systole_fraction": 0.35,
}
_HP_Q_MAX = (
    _HP["stroke_volume"] * math.pi * (_HP["heart_rate"] / 60.0)
    / (2.0 * _HP["systole_fraction"])
)


def _hp_inflow(t):
    """``Q_heart(t)``: the node's own cardiac waveform, as a function of time.

    The cardiac phase is integrated exactly by the node
    (``fmod(phase + dt * f, 1)``), so it can be written down: ``phase(t)
    = fmod(t * heart_rate / 60, 1)``.  The waveform is continuous but
    has a corner at the systole/diastole transition, which is part of
    what the ladder has to cope with.
    """
    phase = jnp.fmod(t * _HP["heart_rate"] / 60.0, 1.0)
    return jnp.where(
        phase < _HP["systole_fraction"],
        _HP_Q_MAX * jnp.sin(jnp.pi * phase / _HP["systole_fraction"]),
        0.0,
    )


def _hp_pressure(t):
    return 95.0 + 18.0 * jnp.sin(2.0 * t) + 4.0 * jnp.cos(5.0 * t)


_hp_dpdt = jax.jacfwd(_hp_pressure)


def _hp_backpressure(t):
    """The downstream pressure that makes :func:`_hp_pressure` exact.

    ``C dP/dt = Q_heart - (P - P_down) / R`` is linear in ``P_down``::

        P_down(t) = P*(t) + R (C P*'(t) - Q_heart(t))
    """
    return _hp_pressure(t) + _HP["resistance"] * (
        _HP["compliance"] * _hp_dpdt(t) - _hp_inflow(t)
    )


def _hp_state(pressure):
    """A strongly-typed float64 carry.

    Explicit ``dtype`` rather than ``jnp.asarray(80.0)``: a *weakly*
    typed float64 pressure meets the node's hard float32 cast of
    ``backpressure`` and is silently demoted, which makes the state
    returned by ``update()`` a different type from the one passed in
    and fails inside ``lax.fori_loop``.  That is MADD-ANO-013, pinned
    below.
    """
    return {
        "arterial_pressure": jnp.asarray(pressure, dtype=jnp.float64),
        "phase": jnp.asarray(0.0, dtype=jnp.float64),
        "flow_rate": jnp.asarray(0.0, dtype=jnp.float64),
    }


def _heart_pump_error(n_steps, *, t_final=1.0, dt_map=None, freeze_source=False):
    """Relative error of the arterial pressure after ``n_steps`` steps."""
    dt = t_final / n_steps
    node = HeartPumpNode(
        "mms_heart", timestep=dt,
        initial_pressure=float(_hp_pressure(0.0)), **_HP,
    )
    step_dt = dt if dt_map is None else dt_map(dt)
    state = _hp_state(_hp_pressure(0.0))

    def step(k, s):
        t = 0.0 if freeze_source else k * dt
        return node.update(s, {"backpressure": _hp_backpressure(t)}, step_dt)

    state = jax.lax.fori_loop(0, n_steps, jax.jit(step), state)
    return _relative_l2(
        {"arterial_pressure": state["arterial_pressure"]},
        {"arterial_pressure": np.asarray(_hp_pressure(t_final), np.float64)},
    )


@verification_benchmark(
    benchmark_id="MADD-VER-012",
    description=(
        "HeartPumpNode temporal order of accuracy by the Method of "
        "Manufactured Solutions: manufactured arterial pressure injected "
        "through backpressure, the node's own boundary input, timestep "
        "refined at fixed final time over one second of cardiac cycles"
    ),
    node_type="HeartPumpNode",
    benchmark_type=BenchmarkType.MANUFACTURED_SOLUTION,
    acceptance_criteria=(
        "Observed temporal order over the finest pair of a 200/400/800/1600 "
        "step ladder within [-0.25, +1.0] of the declared 1.0 "
        "(measured: 1.000).  The ladder stops at 1600 steps: the float32 "
        "downcast of backpressure (MADD-ANO-013) turns it over below about "
        "6e-6 relative error, two orders of magnitude finer than the 1.9e-3 "
        "this ladder reaches."
    ),
    references=(
        "Roache2002: Code Verification by the Method of Manufactured Solutions",
        "LeVeque2007: Finite Difference Methods for ODEs and PDEs",
    ),
)
def test_heart_pump_converges_at_its_declared_temporal_order(float64):
    """The Windkessel pressure update meets its claim of 1st order."""
    node = HeartPumpNode("mms_heart", timestep=0.005, **_HP)
    assert_node_order_verified(
        node,
        axis=RefinementAxis.TIME,
        error_at=_heart_pump_error,
        levels=(200, 400, 800, 1600),
    )


# --------------------------------------------------------------------------
# The studies as a table, for the parametrised checks below
# --------------------------------------------------------------------------

#: ``node factory -> (error function, refinement ladder)``.
_STUDIES = {
    "SpringDamperNode": (
        lambda: SpringDamperNode("mms_spring", timestep=0.01, stiffness=_K,
                                 damping=_C, mass=_M, rest_length=_REST),
        _spring_error,
        (100, 200, 400),
    ),
    "BallNode": (
        lambda: BallNode("mms_ball", timestep=0.01),
        _ball_error,
        (100, 200, 400),
    ),
    "RigidBody2DNode": (
        lambda: RigidBody2DNode("mms_rb2", timestep=0.01, mass=_RB2_M,
                                inertia=_RB2_I, gravity=(0.0, 0.0)),
        _rb2_error,
        (100, 200, 400),
    ),
    "HeartPumpNode": (
        lambda: HeartPumpNode("mms_heart", timestep=0.005, **_HP),
        _heart_pump_error,
        (200, 400, 800),
    ),
}


@pytest.mark.parametrize("node_name", sorted(_STUDIES))
def test_each_ode_node_declares_a_temporal_order_rather_than_skipping(node_name):
    """The point of the exercise: none of the four reports SKIP any more.

    Written as its own test because the order studies above would
    *also* pass if the harness silently skipped them -- ``SKIP`` is
    what :func:`assert_node_order_verified` turns into
    :class:`UndeclaredOrderError`, but only a test that asserts the
    declaration can tell a measured 1.0 from an unmeasured one.
    """
    node = _STUDIES[node_name][0]()
    declared = declared_order(node)
    assert declared is not None, f"{node_name} declares no order at all"
    assert declared.temporal == 1.0
    assert declared.notes, f"{node_name} declares an order with no provenance"


@pytest.mark.parametrize("node_name", sorted(_STUDIES))
def test_an_ode_node_skips_the_spatial_axis_rather_than_passing_it(node_name):
    """``spatial=None`` is a deliberate claim, and has to read as a skip.

    An ODE node has no grid to refine.  The harness must say so
    explicitly; a vacuous pass on the spatial axis would be a claim
    about a discretisation that does not exist.
    """
    node = _STUDIES[node_name][0]()
    assert declared_order(node).spatial is None
    result = verify_node_order(
        node, axis=RefinementAxis.SPACE,
        error_at=lambda level: 1.0 / level, levels=(10, 20),
    )
    assert result.status == "SKIP"
    assert "spatial order" in result.detail

    with pytest.raises(UndeclaredOrderError):
        assert_node_order_verified(
            node, axis=RefinementAxis.SPACE,
            error_at=lambda level: 1.0 / level, levels=(10, 20),
        )


# --------------------------------------------------------------------------
# Mutation tests: the ladders have to be able to fail
# --------------------------------------------------------------------------


class TestTheLaddersCanFail:
    """A convergence study that passes whatever the node does is worthless.

    Each case perturbs a node so that its order *should* drop, and
    requires the harness to fail and to name the node it failed.  The
    perturbations go through the timestep the node is handed and the
    source it is fed, so nothing in ``src/`` is mutated and there is no
    mutation left behind to revert.
    """

    @pytest.mark.parametrize("node_name", sorted(_STUDIES))
    def test_a_mis_scaled_timestep_is_caught(self, node_name, float64):
        """A node advanced by 1.5*dt while time is counted as dt.

        The classic unit bug.  The error tends to a constant instead of
        to zero, so the ladder either stops converging or measures an
        order near zero; both are failures.
        """
        factory, error_at, levels = _STUDIES[node_name]
        with pytest.raises(AssertionError) as exc:
            assert_node_order_verified(
                factory(),
                axis=RefinementAxis.TIME,
                error_at=lambda n: error_at(n, dt_map=lambda dt: _scaled(dt, 1.5)),
                levels=levels,
            )
        assert node_name in str(exc.value)

    @pytest.mark.parametrize("node_name", sorted(_STUDIES))
    def test_a_half_order_perturbation_is_caught(self, node_name, float64):
        """A monotone ladder at order 1/2 must still fail the band.

        This is the case that tests the acceptance band rather than the
        non-convergence guard: the error does fall at every refinement,
        it just falls at the wrong rate.
        """
        factory, error_at, levels = _STUDIES[node_name]
        with pytest.raises(AssertionError) as exc:
            assert_node_order_verified(
                factory(),
                axis=RefinementAxis.TIME,
                error_at=lambda n: error_at(n, dt_map=_order_halving),
                levels=levels,
            )
        assert node_name in str(exc.value)
        assert "below the declared" in str(exc.value)

    @pytest.mark.parametrize("node_name", sorted(_STUDIES))
    def test_a_source_frozen_at_the_initial_time_is_caught(self, node_name, float64):
        """The manufactured forcing held at ``t = 0`` for the whole run.

        A boundary input the graph forgets to refresh is a real coupling
        bug and leaves the node solving a different problem, so the
        error stops depending on the timestep.
        """
        factory, error_at, levels = _STUDIES[node_name]
        with pytest.raises(AssertionError) as exc:
            assert_node_order_verified(
                factory(),
                axis=RefinementAxis.TIME,
                error_at=lambda n: error_at(n, freeze_source=True),
                levels=levels,
            )
        assert node_name in str(exc.value)

    def test_an_order_study_cannot_tell_forward_from_semi_implicit_euler(
        self, float64,
    ):
        """The limit of the method, asserted rather than assumed.

        Both schemes are 1st order, so swapping one for the other leaves
        the measured order inside the band.  That is why MADD-ANO-011
        and MADD-ANO-012 -- both of which are exactly this swap -- are
        pinned by the scheme-identity tests below and not by a ladder.
        A reader who takes "measured 1.0" as evidence that the node
        implements the scheme its metadata names is reading more into
        the measurement than it contains.
        """

        class _ForwardEulerSpring(SpringDamperNode):
            """The spring with the position advanced by the *old* velocity."""

            def update(self, state, boundary_inputs, dt, *, params=None):
                out = super().update(state, boundary_inputs, dt, params=params)
                return {
                    "position": state["position"] + state["velocity"] * dt,
                    "velocity": out["velocity"],
                }

        original = SpringDamperNode(
            "mms_spring", timestep=0.01, stiffness=_K, damping=_C, mass=_M,
            rest_length=_REST,
        )
        mutant = _ForwardEulerSpring(
            "mms_spring_fe", timestep=0.01, stiffness=_K, damping=_C, mass=_M,
            rest_length=_REST,
        )

        def error_at(n_steps, *, t_final=1.0):
            dt = t_final / n_steps
            state = {
                "position": jnp.asarray(_spring_x(0.0)),
                "velocity": jnp.asarray(_spring_v(0.0)),
            }

            def step(k, s):
                return mutant.update(
                    s, {"anchor_position": _spring_anchor(k * dt)}, dt
                )

            state = jax.lax.fori_loop(0, n_steps, jax.jit(step), state)
            return _relative_l2(
                state,
                {"position": np.asarray(_spring_x(t_final), np.float64),
                 "velocity": np.asarray(_spring_v(t_final), np.float64)},
            )

        measurement = measure_order(
            error_at, (100, 200, 400), axis=RefinementAxis.TIME,
        )
        assert measurement.monotone
        assert 0.75 <= measurement.observed <= 2.0, (
            "the forward-Euler mutant was expected to stay inside the band; "
            f"it measured {measurement.observed:.3f}.\n{measurement.table()}"
        )
        # And the original is what it is regardless of the mutant.
        assert declared_order(original).temporal == 1.0


# --------------------------------------------------------------------------
# Scheme identity: what the metadata names vs what update() does
# --------------------------------------------------------------------------


def _forward_euler_agrees(node, state, boundary_inputs, dt, field):
    """Whether ``update()`` equals a forward-Euler step of ``derivatives()``.

    ``derivatives()`` is the node's own statement of the right-hand
    side, so ``euler_step(node.derivatives, ...)`` is precisely the
    forward-Euler discretisation of the equation the node claims to
    solve.  A node whose ``NodeMeta.discretization`` names forward
    Euler has to reproduce it.
    """
    got = node.update(state, boundary_inputs, dt)[field]
    want = euler_step(node.derivatives, state, boundary_inputs, dt)[field]
    return bool(np.allclose(np.asarray(got, np.float64),
                            np.asarray(want, np.float64), rtol=1e-6, atol=1e-9))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MADD-ANO-011: BallNode.meta.discretization names 'Forward Euler "
        "(explicit, 1st-order)', but update() advances the position with the "
        "already-updated velocity -- semi-implicit (symplectic) Euler.  A "
        "forward-Euler step built from the node's own derivatives() gives a "
        "different position at O(dt).  Both schemes are 1st order, so the "
        "order study cannot see this.  Remove this xfail when the metadata "
        "and the implementation agree, whichever of the two is changed."
    ),
)
def test_ball_implements_the_scheme_its_metadata_names():
    node = BallNode("ball", timestep=0.1, initial_position=0.0,
                    initial_velocity=0.0, gravity=-9.81)
    state = {"position": jnp.asarray(0.0, dtype=jnp.float32),
             "velocity": jnp.asarray(0.0, dtype=jnp.float32)}
    agrees = _forward_euler_agrees(node, state, {}, 0.1, "position")
    scheme = "forward" if agrees else "semi-implicit"
    assert scheme in BallNode.meta.discretization.lower(), (
        f"update() implements {scheme} Euler, but NodeMeta.discretization "
        f"says {BallNode.meta.discretization!r}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MADD-ANO-012: HeartPumpNode.meta.discretization names 'Forward Euler "
        "(explicit, 1st-order)', but update() advances the cardiac phase "
        "first and evaluates the inflow waveform at the END of the step, "
        "while the node's own derivatives() evaluates it at the start.  The "
        "two disagree at O(dt) in the source term, so the node's explicit "
        "path and its derivatives()-based paths (integrate_node, "
        "implicit_residual) integrate different waveforms.  Remove this "
        "xfail when the two samplings agree."
    ),
)
def test_heart_pump_implements_the_scheme_its_metadata_names():
    node = HeartPumpNode("heart", timestep=0.05, **_HP)
    state = {"arterial_pressure": jnp.asarray(80.0, dtype=jnp.float32),
             "phase": jnp.asarray(0.0, dtype=jnp.float32),
             "flow_rate": jnp.asarray(0.0, dtype=jnp.float32)}
    assert _forward_euler_agrees(node, state, {}, 0.05, "arterial_pressure"), (
        "update() is not the forward-Euler step of the node's own "
        "derivatives(); see MADD-ANO-012"
    )


def test_spring_implements_the_semi_implicit_scheme_its_metadata_names():
    """The control for the two xfails above.

    SpringDamperNode names semi-implicit Euler and implements it, so
    the same probe that fails for BallNode and HeartPumpNode passes
    here.  Without this the two xfails could be read as an artefact of
    the probe rather than as a property of those two nodes.
    """
    node = SpringDamperNode("spring", timestep=0.01, stiffness=_K, damping=_C,
                            mass=_M, rest_length=_REST)
    state = {"position": jnp.asarray(0.3, dtype=jnp.float32),
             "velocity": jnp.asarray(0.0, dtype=jnp.float32)}
    agrees = _forward_euler_agrees(node, state, {}, 0.01, "position")
    assert not agrees, (
        "SpringDamperNode.update matched a forward-Euler step, but its "
        "metadata claims semi-implicit Euler"
    )
    assert "semi-implicit" in SpringDamperNode.meta.discretization.lower()


# --------------------------------------------------------------------------
# MADD-ANO-013: the float32 downcast of backpressure
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MADD-ANO-013: HeartPumpNode.update casts backpressure to float32 "
        "unconditionally, so a difference below float32 resolution in the "
        "coupling variable is discarded even under jax_enable_x64.  That "
        "puts a floor under any convergence study or adjoint through the "
        "pressure coupling.  Remove this xfail when the cast follows the "
        "state's dtype instead of being pinned to float32."
    ),
)
def test_heart_pump_resolves_a_backpressure_difference_below_float32(float64):
    node = HeartPumpNode("heart", timestep=0.05, **_HP)
    state = _hp_state(80.0)
    # 1e-6 on 100.0 is ~1/7 of a float32 ulp there, and ~5e8 float64 ulps.
    coarse = node.update(
        state, {"backpressure": jnp.asarray(100.0, dtype=jnp.float64)}, 0.05,
    )["arterial_pressure"]
    fine = node.update(
        state, {"backpressure": jnp.asarray(100.0 + 1e-6, dtype=jnp.float64)},
        0.05,
    )["arterial_pressure"]
    assert float(coarse) != float(fine), (
        "the two backpressures differ by 1e-6 and the node returned "
        f"bit-identical pressures ({float(coarse)!r}); see MADD-ANO-013"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MADD-ANO-013: the same float32 cast demotes a weakly-typed float64 "
        "pressure carry -- which is what jnp.asarray(80.0) produces under "
        "jax_enable_x64 -- to float32, so update() returns a state of a "
        "different dtype from the one it was given and lax.scan/fori_loop "
        "rejects the carry.  Remove this xfail with the cast."
    ),
)
def test_heart_pump_does_not_demote_the_dtype_of_the_pressure_it_is_given(float64):
    node = HeartPumpNode("heart", timestep=0.05, **_HP)
    state = {"arterial_pressure": jnp.asarray(80.0),
             "phase": jnp.asarray(0.0),
             "flow_rate": jnp.asarray(0.0)}
    out = node.update(state, {}, 0.05)
    assert out["arterial_pressure"].dtype == jnp.float64, (
        f"a float64 pressure came back as {out['arterial_pressure'].dtype}; "
        "see MADD-ANO-013"
    )
