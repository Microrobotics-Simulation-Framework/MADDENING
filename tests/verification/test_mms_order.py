"""Observed order of convergence, by the Method of Manufactured Solutions.

MADDENING's ~3,440 tests verify that the code does what the code says.
They compare the code to itself: a Laplacian with the wrong weight at
the boundary is finite, deterministic, JIT-consistent, differentiable
and passes every one of them.  These tests compare the code to the
*mathematics*.

Each study substitutes an analytic field into the equation a node
claims to solve, feeds the node the source term that makes that field
exact, and measures the rate at which the error falls under
refinement — not its size.  A wrong stencil weight or a mishandled
boundary leaves the absolute error looking perfectly acceptable and
shows up as order 1 where order 2 was claimed.

Two of the studies below currently fail, and are recorded as strict
xfails so that fixing the node turns them into an XPASS that has to be
dealt with rather than a silent pass:

* ``stencil_order=4`` measures order 1.0 against its claim of 4
  (MADD-ANO-008), and is *less* accurate than the default second-order
  stencil at every resolution measured;
* supplying the Dirichlet data at the rod ends, which is what
  ``boundary_input_spec`` documents, measures order 1.0 against the
  node's claim of 2 (MADD-ANO-007).

Both sat comfortably inside the acceptance criteria of MADD-VER-001 and
MADD-VER-002, which are pointwise-error and "rate between 0.7 and 2.5"
tests over the same node.

Precision: the studies run under ``jax_enable_x64``.  The observed
order is a ratio of small numbers, and in float32 these ladders measure
a clean order out to about 80 cells and then turn over as round-off
overtakes the discretisation error — the failure the harness reports as
a non-monotone ladder.  ``tests/verification/hypothesis/
test_hypothesis_integrators.py`` sets the same precedent for
precision-sensitive tests here.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import contextlib  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from maddening.core.compliance.metadata import DiscretizationOrder  # noqa: E402
from maddening.core.compliance.validation import (  # noqa: E402
    _BENCHMARK_REGISTRY,
    BenchmarkType,
    verification_benchmark,
)
from maddening.core.node import SimulationNode  # noqa: E402
from maddening.nodes.heat import HeatNode  # noqa: E402
from maddening.nodes.lbm import LBMNode  # noqa: E402
from maddening.nodes.rigid_body import RigidBodyNode  # noqa: E402
from maddening.testing.mms import (  # noqa: E402
    DEFAULT_ORDER_EXCESS,
    DEFAULT_ORDER_SHORTFALL,
    ManufacturedSolution,
    OrderMeasurement,
    RefinementAxis,
    UndeclaredOrderError,
    assert_node_order_verified,
    check_order,
    declared_order,
    diffusion_operator,
    manufactured_acceleration,
    measure_order,
    verify_node_order,
)

# --------------------------------------------------------------------------
# Shared fixtures
# --------------------------------------------------------------------------


@contextlib.contextmanager
def _float64():
    """Run the body in double precision, restoring the global setting.

    The suite runs float32 by default; see the module docstring for why
    a convergence ladder needs more than that.
    """
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


# --------------------------------------------------------------------------
# HeatNode — spatial order
# --------------------------------------------------------------------------

_L = 1.0
_ALPHA = 1.0

#: Steady manufactured profile.  Non-symmetric (the linear term) and
#: with a non-vanishing fourth derivative (the sine), so neither the
#: second- nor the fourth-order stencil is accidentally exact on it.
_STEADY = ManufacturedSolution(
    exact=lambda x, t: jnp.sin(2.0 * jnp.pi * x / _L) + 0.5 * x + 1.0,
    operator=diffusion_operator(_ALPHA),
)


def _cell_centres(n_cells):
    dx = _L / n_cells
    return np.linspace(dx / 2, _L - dx / 2, n_cells)


def _relaxation_steps(n_cells, fourier, decay=16.0):
    """Steps for the slowest mode to decay by ``exp(-decay)``.

    The slowest mode of the discrete operator decays by roughly
    ``1 - fourier * pi**2 / n**2`` per step, so the count grows like
    ``n**2``.  ``decay=16`` leaves a residual transient ~1e-7 of the
    discretisation error, i.e. far below it at every level here.
    """
    return int(decay * n_cells**2 / (fourier * np.pi**2)) + 50


def _heat_steady_error(n_cells, *, stencil_order=2, fourier=0.4, bc="cell_centre"):
    """Relative L2 error of the steady manufactured profile.

    A steady manufactured solution is what isolates the *spatial* order
    on an explicit scheme.  Forward Euler's temporal truncation error
    is proportional to the second time derivative of the exact
    solution, so for a time-independent one it is identically zero, and
    what the run converges to is the exact solution of the discrete
    steady problem — the spatial error and nothing else.  Refining
    space and time together (the usual CFL-locked ladder) would instead
    measure the minimum of the two orders.
    """
    dx = _L / n_cells
    dt = fourier * dx * dx / _ALPHA
    x = _cell_centres(n_cells)
    exact = np.asarray(_STEADY.field(x, 0.0), dtype=np.float64)
    source = _STEADY.source_field(x, 0.0)

    if bc == "cell_centre":
        # What the code implements: Dirichlet data at the first and
        # last cell centre, which is where update() writes it.
        t_left, t_right = exact[0], exact[-1]
    else:
        # What boundary_input_spec documents: "Dirichlet BC at left
        # end", i.e. the value at the rod's end, x = 0 and x = L.
        t_left = float(_STEADY.exact(jnp.float64(0.0), jnp.float64(0.0)))
        t_right = float(_STEADY.exact(jnp.float64(_L), jnp.float64(0.0)))

    node = HeatNode(
        "mms_heat", timestep=dt, n_cells=n_cells, length=_L,
        thermal_diffusivity=_ALPHA, initial_temperature=exact,
        stencil_order=stencil_order,
    )
    boundary = {
        "left_temperature": jnp.asarray(t_left),
        "right_temperature": jnp.asarray(t_right),
        "heat_source": jnp.asarray(source),
    }
    step = jax.jit(
        lambda T: node.update({"temperature": T}, boundary, dt)["temperature"]
    )
    T = jax.lax.fori_loop(
        0, _relaxation_steps(n_cells, fourier), lambda _, t: step(t),
        jnp.asarray(exact),
    )
    T = np.asarray(jax.device_get(T), dtype=np.float64)
    return float(
        np.sqrt(np.mean((T - exact) ** 2)) / np.sqrt(np.mean(exact**2))
    )


@verification_benchmark(
    benchmark_id="MADD-VER-005",
    description=(
        "HeatNode spatial order of accuracy by the Method of Manufactured "
        "Solutions: steady manufactured profile, source term derived by "
        "automatic differentiation, grid refined at fixed Fourier number"
    ),
    node_type="HeatNode",
    benchmark_type=BenchmarkType.MANUFACTURED_SOLUTION,
    acceptance_criteria=(
        "Observed spatial order over the finest pair of a 10/20/40/80/160 "
        "ladder within [-0.25, +1.0] of the declared 2.0 (measured: 1.982). "
        "Applies to the boundary convention the code implements, Dirichlet "
        "data at the first and last cell centre; see MADD-ANO-007."
    ),
    references=(
        "Roache2002: Code Verification by the Method of Manufactured Solutions",
        "LeVeque2007: Finite Difference Methods for ODEs and PDEs",
    ),
)
def test_heat_second_order_stencil_converges_at_its_declared_spatial_order(float64):
    """The default stencil meets its claim of 2nd order in space."""
    node = HeatNode("mms_heat", timestep=1e-4, n_cells=10, length=_L,
                    thermal_diffusivity=_ALPHA)
    assert_node_order_verified(
        node,
        axis=RefinementAxis.SPACE,
        error_at=_heat_steady_error,
        levels=(10, 20, 40, 80, 160),
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MADD-ANO-008: HeatNode(stencil_order=4) converges at order 1, not 4. "
        "The two ghost cells of the 5-point stencil are populated one cell "
        "out of position -- the ghost at x=-dx is given the boundary value, "
        "which is the value at x=0 -- so the local truncation error at the "
        "first interior cell is O(1/dx) and the global error is O(dx).  "
        "Remove this xfail when the ghost construction is corrected."
    ),
)
def test_heat_fourth_order_stencil_converges_at_its_declared_spatial_order(float64):
    """The 4th-order stencil does not meet its claim of 4th order.

    Run at a Fourier number of 0.3: the 5-point stencil is unstable
    above 3/8, not the 1/2 the node documents (MADD-ANO-009).
    """
    node = HeatNode("mms_heat4", timestep=1e-5, n_cells=10, length=_L,
                    thermal_diffusivity=_ALPHA, stencil_order=4)
    assert declared_order(node).spatial == 4.0
    assert_node_order_verified(
        node,
        axis=RefinementAxis.SPACE,
        error_at=lambda n: _heat_steady_error(n, stencil_order=4, fourier=0.3),
        levels=(10, 20, 40, 80),
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "MADD-ANO-007: HeatNode applies left_temperature/right_temperature at "
        "the first and last CELL CENTRE (x = dx/2, L - dx/2), while "
        "boundary_input_spec documents them as the Dirichlet BC 'at the left "
        "end' / 'at the right end'.  A caller who supplies T(0) and T(L), as "
        "documented, gets a globally 1st-order scheme.  Remove this xfail "
        "when the node and its documentation agree."
    ),
)
def test_heat_converges_at_its_declared_order_with_the_documented_boundary_data(
    float64,
):
    """The documented boundary semantics cost a full order of accuracy."""
    node = HeatNode("mms_heat_bc", timestep=1e-4, n_cells=10, length=_L,
                    thermal_diffusivity=_ALPHA)
    assert_node_order_verified(
        node,
        axis=RefinementAxis.SPACE,
        error_at=lambda n: _heat_steady_error(n, bc="rod_ends"),
        levels=(10, 20, 40, 80, 160),
    )


def test_the_fourth_order_stencil_is_unstable_below_the_documented_cfl_limit():
    """MADD-ANO-009: the 5-point stencil's stability bound is 3/8, not 1/2.

    ``NodeMeta.limitations`` and MADD-ANO-002 both give the limit as
    ``dt < dx^2 / (2*alpha)`` — a Fourier number of 1/2 — for the node
    as a whole.  That is the bound for the 3-point stencil.  The
    4th-order stencil's is ``3/8``, and between the two the run
    diverges silently, which is the failure mode MADD-ANO-002 exists to
    record.
    """
    stable = _heat_steady_error(20, stencil_order=4, fourier=0.37)
    assert np.isfinite(stable) and stable < 1.0, (
        f"Fo=0.37 should be stable for the 4th-order stencil, got {stable}"
    )
    unstable = _heat_steady_error(20, stencil_order=4, fourier=0.4)
    assert not np.isfinite(unstable) or unstable > 1.0, (
        "Fo=0.4 is above the 4th-order stencil's 3/8 stability bound and "
        f"should diverge, but the error was {unstable}; if the stencil has "
        "been changed, re-measure the bound and update MADD-ANO-009"
    )


# --------------------------------------------------------------------------
# HeatNode — temporal order
# --------------------------------------------------------------------------

_OMEGA = 20.0

#: Quadratic in x, so the second-order central difference reproduces
#: ``d2u/dx2`` exactly and the spatial error is identically zero: what
#: is left is the time integrator.  Oscillating fast enough in t that
#: the temporal error stays well clear of round-off.
_TRANSIENT = ManufacturedSolution(
    exact=lambda x, t: (1.0 + 0.7 * x + 0.4 * x * x) * jnp.cos(_OMEGA * t),
    operator=diffusion_operator(_ALPHA),
)


def _heat_transient_error(n_steps, *, n_cells=21, t_final=0.2):
    """Relative L2 error at ``t_final`` after ``n_steps`` steps.

    The grid is fixed, so this ladder refines time alone.
    """
    dt = t_final / n_steps
    x = _cell_centres(n_cells)
    xj = jnp.asarray(x)
    initial = np.asarray(_TRANSIENT.field(x, 0.0), dtype=np.float64)

    node = HeatNode(
        "mms_heat_t", timestep=dt, n_cells=n_cells, length=_L,
        thermal_diffusivity=_ALPHA, initial_temperature=initial,
    )

    def step(k, T):
        t = k * dt
        boundary = {
            "left_temperature": _TRANSIENT.exact(xj[0], t),
            "right_temperature": _TRANSIENT.exact(xj[-1], t),
            "heat_source": _TRANSIENT.source_field(xj, t),
        }
        return node.update({"temperature": T}, boundary, dt)["temperature"]

    T = jax.lax.fori_loop(0, n_steps, jax.jit(step), jnp.asarray(initial))
    T = np.asarray(jax.device_get(T), dtype=np.float64)
    exact = np.asarray(_TRANSIENT.field(x, t_final), dtype=np.float64)
    return float(
        np.sqrt(np.mean((T - exact) ** 2)) / np.sqrt(np.mean(exact**2))
    )


@verification_benchmark(
    benchmark_id="MADD-VER-006",
    description=(
        "HeatNode temporal order of accuracy by the Method of Manufactured "
        "Solutions: manufactured solution quadratic in x, so the spatial "
        "error vanishes identically and the timestep ladder measures the "
        "forward-Euler integration alone"
    ),
    node_type="HeatNode",
    benchmark_type=BenchmarkType.MANUFACTURED_SOLUTION,
    acceptance_criteria=(
        "Observed temporal order over the finest pair of a 250/500/1000/2000 "
        "step ladder within [-0.25, +1.0] of the declared 1.0 "
        "(measured: 0.998)"
    ),
    references=(
        "Roache2002: Code Verification by the Method of Manufactured Solutions",
    ),
)
def test_heat_converges_at_its_declared_temporal_order(float64):
    """Forward Euler in time meets its claim of 1st order."""
    node = HeatNode("mms_heat_t", timestep=1e-4, n_cells=21, length=_L,
                    thermal_diffusivity=_ALPHA)
    assert_node_order_verified(
        node,
        axis=RefinementAxis.TIME,
        error_at=_heat_transient_error,
        levels=(250, 500, 1000, 2000),
    )


# --------------------------------------------------------------------------
# LBMNode — spatial order
# --------------------------------------------------------------------------

_NU = 0.1            # lattice viscosity, held fixed: diffusive scaling
_U_REF, _N_REF = 0.04, 16
_LBM_NX = 4


def _lbm_kolmogorov_error(n_cells):
    """Relative L2 error of a steady, force-driven shear layer.

    Kolmogorov flow: ``u = (U sin(k y), 0)`` is an exact steady
    solution of the incompressible Navier-Stokes equations, because it
    is divergence-free and its own advection term vanishes, driven by
    the body force ``F_x = -mu d2u_x/dy2``.  The x-momentum equation
    therefore reduces to the same 1D diffusion problem the HeatNode
    study manufactures, and the same :class:`ManufacturedSolution`
    derives the force.

    The domain is fully periodic — no ``wall_mask``, no Zou-He faces —
    because the node's own metadata records bounce-back as 1st order at
    curved walls, which would be what the ladder measured instead.

    Refinement uses **diffusive scaling**: the lattice fixes
    ``dx = dt = 1``, so the only way to refine is to resolve the same
    physical wavelength with more cells, holding the lattice viscosity
    fixed and scaling the velocity with ``1/N`` so the Mach number
    falls with the grid.  Nothing in the node does this; it is the
    caller's scaling rule.
    """
    U = _U_REF * _N_REF / n_cells
    k = 2.0 * np.pi / n_cells
    y = np.arange(n_cells, dtype=np.float64)

    profile = ManufacturedSolution(
        exact=lambda yy, t: U * jnp.sin(k * yy),
        operator=diffusion_operator(_NU),   # mu = rho * nu, rho = 1
    )
    exact = np.asarray(profile.field(y, 0.0), dtype=np.float64)
    force = np.zeros((_LBM_NX, n_cells, 2), dtype=np.float64)
    force[:, :, 0] = np.asarray(profile.source_field(y, 0.0), dtype=np.float64)

    node = LBMNode(
        "mms_lbm", timestep=1.0, grid_shape=(_LBM_NX, n_cells),
        viscosity=_NU, lattice="D2Q9",
    )
    state = {
        key: (value if value.dtype == jnp.uint8 else jnp.asarray(value, jnp.float64))
        for key, value in node.initial_state().items()
    }
    boundary = {"body_force": jnp.asarray(force)}
    step = jax.jit(lambda s: node.update(s, boundary, 1.0))
    # The shear mode relaxes at nu*k^2 per lattice step; 14 e-foldings
    # leaves a residual transient far below the discretisation error.
    steps = int(14.0 / (_NU * k * k)) + 100
    state = jax.lax.fori_loop(0, steps, lambda _, s: step(s), state)

    u_x = np.asarray(jax.device_get(state["velocity"]), np.float64)[..., 0]
    u_x = u_x.mean(axis=0)          # the field is uniform along x
    return float(
        np.sqrt(np.mean((u_x - exact) ** 2)) / np.sqrt(np.mean(exact**2))
    )


@verification_benchmark(
    benchmark_id="MADD-VER-007",
    description=(
        "LBMNode spatial order of accuracy by the Method of Manufactured "
        "Solutions: steady Kolmogorov flow on a periodic D2Q9 lattice driven "
        "by a manufactured body force, refined under diffusive scaling"
    ),
    node_type="LBMNode",
    benchmark_type=BenchmarkType.MANUFACTURED_SOLUTION,
    acceptance_criteria=(
        "Observed spatial order over the finest pair of a 16/32/64 ladder "
        "within [-0.25, +1.0] of the declared 2.0 (measured: 1.998). "
        "Wall-free periodic domain; bounce-back walls are 1st order."
    ),
    references=(
        "Guo2002: Discrete lattice effects on the forcing term in the LBM",
        "Roache2002: Code Verification by the Method of Manufactured Solutions",
    ),
)
def test_lbm_converges_at_its_declared_spatial_order(float64):
    """BGK + Guo forcing meets its claim of 2nd order in space."""
    node = LBMNode("mms_lbm", timestep=1.0, grid_shape=(_LBM_NX, 16),
                   viscosity=_NU, lattice="D2Q9")
    assert_node_order_verified(
        node,
        axis=RefinementAxis.SPACE,
        error_at=_lbm_kolmogorov_error,
        levels=(16, 32, 64),
    )


def test_lbm_declares_no_temporal_order_and_is_skipped_rather_than_passed():
    """An axis a node cannot be refined along must skip, not pass.

    The lattice fixes ``dx = dt = 1`` and ``LBMNode.update`` ignores its
    ``dt`` argument entirely, so there is no timestep to refine.  The
    harness has to say so rather than report a vacuous pass.
    """
    node = LBMNode("lbm", timestep=1.0, grid_shape=(4, 4), viscosity=_NU,
                   lattice="D2Q9")
    assert declared_order(node).temporal is None
    result = verify_node_order(
        node, axis=RefinementAxis.TIME,
        error_at=lambda level: 1.0 / level, levels=(2, 4),
    )
    assert result.status == "SKIP"
    assert "temporal order" in result.detail


# --------------------------------------------------------------------------
# RigidBodyNode — temporal order
# --------------------------------------------------------------------------

_MASS = 1.0
_INERTIA = np.array([1.0, 2.0, 3.0])
_AMP = np.array([0.3, 0.2, 0.5])
_FREQ = np.array([2.0, 3.0, 5.0])
_OMEGA0 = np.array([0.4, 0.1, 0.2])
_OMEGA_AMP = np.array([0.2, 0.3, 0.1])
_OMEGA_FREQ = np.array([1.5, 2.5, 3.5])


def _rb_position(t):
    return _AMP * jnp.sin(_FREQ * t)


def _rb_angular_velocity(t):
    return _OMEGA0 + _OMEGA_AMP * jnp.sin(_OMEGA_FREQ * t)


_RB_ACCELERATION = manufactured_acceleration(_rb_position)
_RB_ANGULAR_ACCELERATION = jax.jacfwd(_rb_angular_velocity)


def _rigid_body_error(n_steps, *, t_final=1.0):
    """Relative L2 error of the whole state after ``n_steps`` steps.

    The manufactured trajectory is injected as ``force`` and ``torque``
    — the node's own additive boundary inputs — sampled at the start of
    each step, which is where an explicit scheme reads its source.
    Gravity is zeroed so the manufactured forcing is the only drive,
    and no constraints are set, so the update has no branches.

    The error is taken over velocity, angular velocity *and* position
    together: for a state-independent force, symplectic Euler
    integrates position at second order, so a position-only norm would
    measure the wrong thing.
    """
    dt = t_final / n_steps
    node = RigidBodyNode(
        "mms_rb", timestep=dt, mass=_MASS, inertia=tuple(_INERTIA),
        gravity=(0.0, 0.0, 0.0),
    )
    state = {
        "position": jnp.asarray(_rb_position(0.0)),
        "orientation": jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float64),
        "velocity": jnp.asarray(jax.jacfwd(_rb_position)(0.0)),
        "angular_velocity": jnp.asarray(_rb_angular_velocity(0.0)),
    }

    def step(k, s):
        t = k * dt
        return node.update(
            s,
            {
                "force": _MASS * _RB_ACCELERATION(t),
                "torque": jnp.asarray(_INERTIA) * _RB_ANGULAR_ACCELERATION(t),
            },
            dt,
        )

    state = jax.lax.fori_loop(0, n_steps, jax.jit(step), state)
    got = {k: np.asarray(jax.device_get(v), np.float64) for k, v in state.items()}
    exact = {
        "position": np.asarray(_rb_position(t_final), np.float64),
        "velocity": np.asarray(jax.jacfwd(_rb_position)(t_final), np.float64),
        "angular_velocity": np.asarray(
            _rb_angular_velocity(t_final), np.float64
        ),
    }
    num = sum(np.sum((got[k] - v) ** 2) for k, v in exact.items())
    den = sum(np.sum(v**2) for v in exact.values())
    return float(np.sqrt(num / den))


@verification_benchmark(
    benchmark_id="MADD-VER-008",
    description=(
        "RigidBodyNode temporal order of accuracy by the Method of "
        "Manufactured Solutions: manufactured 6-DOF trajectory injected as "
        "force and torque, timestep refined at fixed final time"
    ),
    node_type="RigidBodyNode",
    benchmark_type=BenchmarkType.MANUFACTURED_SOLUTION,
    acceptance_criteria=(
        "Observed temporal order over the finest pair of a 100/200/400/800 "
        "step ladder within [-0.25, +1.0] of the declared 1.0 "
        "(measured: 0.999)"
    ),
    references=(
        "Hairer2006: Geometric Numerical Integration, Ch. VI (symplectic Euler)",
    ),
)
def test_rigid_body_converges_at_its_declared_temporal_order(float64):
    """Semi-implicit Euler meets its claim of 1st order."""
    node = RigidBodyNode("mms_rb", timestep=0.01, mass=_MASS,
                         inertia=tuple(_INERTIA), gravity=(0.0, 0.0, 0.0))
    assert_node_order_verified(
        node,
        axis=RefinementAxis.TIME,
        error_at=_rigid_body_error,
        levels=(100, 200, 400, 800),
    )


# --------------------------------------------------------------------------
# The harness itself
# --------------------------------------------------------------------------


class _UndeclaredNode(SimulationNode):
    """A node that declares no order at all."""

    def initial_state(self):
        return {"x": jnp.zeros(())}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": state["x"]}


def _ladder(order, levels=(10, 20, 40, 80), axis=RefinementAxis.SPACE):
    """A synthetic ladder whose error falls exactly as ``h**order``."""
    return measure_order(
        lambda n: (1.0 / n) ** order, levels, axis=axis,
    )


class TestTheOrderGateCanFail:
    """Mutation tests: the band has to reject what it claims to reject.

    A gate is only worth what it fails on.  Each case here is a ladder
    the gate must *not* accept.
    """

    def test_a_ladder_at_the_declared_order_passes(self):
        assert check_order(_ladder(2.0), 2.0).passed

    @pytest.mark.parametrize("measured", [1.0, 1.5, 1.74])
    def test_a_ladder_short_of_the_declared_order_fails(self, measured):
        result = check_order(_ladder(measured), 2.0)
        assert result.failed
        assert "below the declared" in result.detail

    def test_a_ladder_just_inside_the_band_still_passes(self):
        """The band is the documented width, not a wider one by accident."""
        assert check_order(_ladder(2.0 - DEFAULT_ORDER_SHORTFALL / 2), 2.0).passed

    @pytest.mark.parametrize("measured", [3.5, 6.0])
    def test_a_ladder_far_above_the_declared_order_fails(self, measured):
        """A study that measures nothing is a failure, not a bonus."""
        result = check_order(_ladder(measured), 2.0)
        assert result.failed
        assert "above the declared" in result.detail

    def test_a_ladder_whose_error_stops_falling_fails_as_inconclusive(self):
        """Noise-floor contamination must not be read as a wrong order."""
        measurement = OrderMeasurement(
            axis=RefinementAxis.SPACE, levels=(10, 20, 40, 80),
            h=(0.1, 0.05, 0.025, 0.0125),
            errors=(1e-2, 2.5e-3, 6e-4, 9e-4),
        )
        result = check_order(measurement, 2.0)
        assert result.failed
        assert "not converging" in result.detail

    def test_an_exactly_represented_solution_fails_rather_than_passing(self):
        """A zero error means the study did not exercise the scheme."""
        measurement = OrderMeasurement(
            axis=RefinementAxis.SPACE, levels=(10, 20),
            h=(0.1, 0.05), errors=(1e-12, 0.0),
        )
        assert check_order(measurement, 2.0).failed

    def test_the_default_band_is_the_one_the_docstring_claims(self):
        assert (DEFAULT_ORDER_SHORTFALL, DEFAULT_ORDER_EXCESS) == (0.25, 1.0)


class TestMeasureOrder:
    def test_the_fit_and_the_finest_pair_agree_on_a_clean_ladder(self):
        measurement = _ladder(2.0)
        assert measurement.observed == pytest.approx(2.0, abs=1e-9)
        assert measurement.fitted == pytest.approx(2.0, abs=1e-9)
        assert measurement.monotone

    def test_levels_given_finest_first_are_refused(self):
        """Reversed levels would silently flip the sign of every order."""
        with pytest.raises(ValueError, match="coarsest first"):
            measure_order(lambda n: 1.0 / n, (80, 40, 20),
                          axis=RefinementAxis.SPACE)

    def test_a_single_level_is_refused(self):
        with pytest.raises(ValueError, match="at least two"):
            measure_order(lambda n: 1.0 / n, (10,), axis=RefinementAxis.SPACE)

    def test_an_explicit_h_mapping_is_used(self):
        """A ladder of timesteps indexes by dt directly, not by 1/level."""
        measurement = measure_order(
            lambda dt: dt**2, (0.1, 0.05, 0.025),
            axis=RefinementAxis.TIME, h_of=lambda dt: dt,
        )
        assert measurement.observed == pytest.approx(2.0, abs=1e-9)

    def test_the_table_names_the_axis_that_was_refined(self):
        assert "dt" in _ladder(1.0, axis=RefinementAxis.TIME).table()
        assert " h " in _ladder(1.0, axis=RefinementAxis.SPACE).table()


class TestDeclaredOrder:
    def test_a_node_declaring_nothing_is_skipped_not_passed(self):
        node = _UndeclaredNode("undeclared", 0.01)
        assert declared_order(node) is None
        result = verify_node_order(
            node, axis=RefinementAxis.SPACE,
            error_at=lambda n: (1.0 / n) ** 2, levels=(10, 20),
        )
        assert result.status == "SKIP"
        assert "declares no spatial order" in result.detail

    def test_the_assert_form_raises_rather_than_passing_silently(self):
        node = _UndeclaredNode("undeclared", 0.01)
        with pytest.raises(UndeclaredOrderError, match="declares no spatial order"):
            assert_node_order_verified(
                node, axis=RefinementAxis.SPACE,
                error_at=lambda n: (1.0 / n) ** 2, levels=(10, 20),
            )

    def test_the_instance_hook_wins_over_the_class_declaration(self):
        """A node whose order depends on construction can only answer per instance."""
        second = HeatNode("h2", 1e-4, n_cells=10)
        fourth = HeatNode("h4", 1e-5, n_cells=10, stencil_order=4)
        assert HeatNode.meta.discretization_order.spatial == 2.0
        assert declared_order(second).spatial == 2.0
        assert declared_order(fourth).spatial == 4.0

    def test_an_explicit_expectation_overrides_the_declaration(self):
        node = _UndeclaredNode("undeclared", 0.01)
        result = verify_node_order(
            node, axis=RefinementAxis.SPACE,
            error_at=lambda n: (1.0 / n) ** 2, levels=(10, 20, 40),
            expected=2.0,
        )
        assert result.passed and result.status == "PASS"


class TestManufacturedSource:
    """The AD-derived source must equal the hand-derived one.

    This is the step MMS is most often got wrong by hand, which is why
    the harness derives it — so it is the step that most needs pinning
    against an independent derivation.
    """

    def test_the_source_matches_a_hand_derivation_for_diffusion(self):
        alpha = 0.37
        omega, kx = 2.1, 3.3
        sol = ManufacturedSolution(
            exact=lambda x, t: jnp.sin(kx * x) * jnp.exp(-omega * t),
            operator=diffusion_operator(alpha),
        )
        for x in (0.0, 0.25, 1.7):
            for t in (0.0, 0.4):
                expected = (
                    -omega * np.sin(kx * x) * np.exp(-omega * t)
                    + alpha * kx**2 * np.sin(kx * x) * np.exp(-omega * t)
                )
                assert float(sol.source(x, t)) == pytest.approx(expected, rel=1e-5)

    def test_a_solution_of_the_unforced_equation_needs_no_source(self):
        """The heat kernel is its own witness: S must vanish on it."""
        alpha = 0.7
        kx = 1.9
        sol = ManufacturedSolution(
            exact=lambda x, t: jnp.sin(kx * x) * jnp.exp(-alpha * kx**2 * t),
            operator=diffusion_operator(alpha),
        )
        assert float(sol.source(0.6, 0.3)) == pytest.approx(0.0, abs=1e-6)

    def test_the_manufactured_acceleration_is_the_second_derivative(self):
        acc = manufactured_acceleration(lambda t: jnp.array([jnp.sin(3.0 * t)]))
        assert float(acc(0.4)[0]) == pytest.approx(-9.0 * np.sin(1.2), rel=1e-5)


class TestBenchmarkRegistration:
    """The new benchmarks reach the registry the compliance gates read."""

    @pytest.mark.parametrize(
        "benchmark_id", ["MADD-VER-005", "MADD-VER-006",
                         "MADD-VER-007", "MADD-VER-008"],
    )
    def test_the_benchmark_is_registered(self, benchmark_id):
        assert benchmark_id in _BENCHMARK_REGISTRY

    @pytest.mark.parametrize(
        ("benchmark_id", "node_type"),
        [("MADD-VER-005", "HeatNode"), ("MADD-VER-006", "HeatNode"),
         ("MADD-VER-007", "LBMNode"), ("MADD-VER-008", "RigidBodyNode")],
    )
    def test_the_benchmark_names_its_node_and_its_method(
        self, benchmark_id, node_type,
    ):
        benchmark = _BENCHMARK_REGISTRY[benchmark_id]
        assert benchmark.node_type == node_type
        assert benchmark.benchmark_type is BenchmarkType.MANUFACTURED_SOLUTION

    @pytest.mark.parametrize(
        ("node_type", "axis", "order"),
        [("HeatNode", "spatial", 2.0), ("HeatNode", "temporal", 1.0),
         ("LBMNode", "spatial", 2.0), ("RigidBodyNode", "temporal", 1.0)],
    )
    def test_every_measured_node_declares_the_order_that_was_measured(
        self, node_type, axis, order,
    ):
        """The benchmark and the declaration cannot drift apart."""
        nodes = {
            "HeatNode": HeatNode("h", 1e-4, n_cells=10),
            "LBMNode": LBMNode("l", 1.0, grid_shape=(4, 4), viscosity=_NU,
                               lattice="D2Q9"),
            "RigidBodyNode": RigidBodyNode("r", 0.01),
        }
        declared = declared_order(nodes[node_type])
        assert isinstance(declared, DiscretizationOrder)
        assert getattr(declared, axis) == order
