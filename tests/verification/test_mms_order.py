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

Two of the HeatNode studies below were strict xfails when this module
landed, recording defects the harness had just found.  Both are fixed
and both now assert:

* ``stencil_order=4`` measured order 0.954 against its claim of 4 and
  was 10x *less* accurate than the default stencil (MADD-ANO-008); it
  now measures 3.957;
* supplying the Dirichlet data at the rod ends, which is what
  ``boundary_input_spec`` documents, measured order 1.001 against the
  node's claim of 2 (MADD-ANO-007); it now measures 2.000, and the
  rod-end reading is the only one the node implements.

Both sat comfortably inside the acceptance criteria of MADD-VER-001 and
MADD-VER-002, which were a pointwise-error test and a convergence study
whose band had been widened to "rate between 0.7 and 2.5" — wide enough
to accept the 1.0 it was measuring.  MADD-VER-002's band is now
[1.7, 2.3]; see ``tests/verification/test_heat_analytical.py``.

Each fix is pinned twice over: by the ladder that measures the order,
and by a cheap direct test that fails on the defect alone
(:class:`TestHeatBoundaryPlacement`, :class:`TestHeatStencilGhosts`).
The order ladders are the specification; the direct tests are what
makes a regression legible without reading a refinement table.

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
import pathlib  # noqa: E402

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
from maddening.nodes.heat import MAX_FOURIER_NUMBER, HeatNode  # noqa: E402
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

#: Repository root: tests/verification/<this file> -> two levels up.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_L = 1.0
_ALPHA = 1.0

#: Steady manufactured profile.  Three properties, each load-bearing and
#: each pinned by :class:`TestTheSteadyProfileCanSeeABrokenScheme`:
#:
#: * non-symmetric (the linear term), so a symmetric error cannot cancel;
#: * non-vanishing fourth derivative (the sine), so neither stencil is
#:   accidentally exact on it in the interior;
#: * **non-vanishing second derivative at both rod ends** (the quadratic
#:   term).  This one was missing until 0.4.0 and the omission mattered.
#:
#: The boundary rows' leading error term is proportional to ``u''`` at
#: the rod end, so a profile flat there cannot see a wrong boundary
#: closure at all.  ``sin(2 pi x) + 0.5x + 1`` has ``u'' = 0`` at both
#: ends exactly, and on it a linear ghost extrapolation -- which is a
#: genuinely 2nd-order closure -- measures 4.357, 4.257, 4.150, 4.080
#: and sails through a band centred on 4.  Adding ``0.4 x^2`` makes the
#: same wrong closure measure 3.206, 2.129, 2.006, 2.000, caught by two
#: whole orders, and moves the correct closure not at all (3.957 either
#: way).  A manufactured solution that cannot fail is the same defect as
#: an acceptance band that cannot fail, one layer down.
_STEADY = ManufacturedSolution(
    exact=lambda x, t: (
        jnp.sin(2.0 * jnp.pi * x / _L) + 0.5 * x + 1.0 + 0.4 * x * x
    ),
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


def _heat_steady_error(n_cells, *, stencil_order=2, fourier=0.4, bc="rod_ends"):
    """Relative L2 error of the steady manufactured profile.

    A steady manufactured solution is what isolates the *spatial* order
    on an explicit scheme.  Forward Euler's temporal truncation error
    is proportional to the second time derivative of the exact
    solution, so for a time-independent one it is identically zero, and
    what the run converges to is the exact solution of the discrete
    steady problem — the spatial error and nothing else.  Refining
    space and time together (the usual CFL-locked ladder) would instead
    measure the minimum of the two orders.

    ``bc`` selects which reading of ``left_temperature`` the study
    feeds the node.  ``"rod_ends"`` is the documented one and the one
    the node implements; ``"cell_centre"`` is the reading the node used
    to implement, kept so that
    :class:`TestHeatBoundaryPlacement` can show the two are not
    interchangeable and that the wrong one still costs an order.
    """
    dx = _L / n_cells
    dt = fourier * dx * dx / _ALPHA
    x = _cell_centres(n_cells)
    exact = np.asarray(_STEADY.field(x, 0.0), dtype=np.float64)
    source = _STEADY.source_field(x, 0.0)

    if bc == "cell_centre":
        # The pre-0.4.0 reading: the value at the first and last cell
        # centre, which is where update() used to write it.  Supplying
        # it now is supplying the boundary datum half a cell out.
        t_left, t_right = exact[0], exact[-1]
    else:
        # What boundary_input_spec documents and what the node now
        # implements: the value at the rod's ends, x = 0 and x = L.
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
        "ladder within [-0.25, +1.0] of the declared 2.0 (measured: 2.000). "
        "Dirichlet data supplied at the rod ends x=0 and x=L, which is what "
        "boundary_input_spec documents and, since 0.4.0, what the node "
        "implements; before 0.4.0 the same ladder measured 1.001 "
        "(MADD-ANO-007)."
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


def test_heat_fourth_order_stencil_converges_at_its_declared_spatial_order(float64):
    """The 4th-order stencil meets its claim of 4th order in space.

    Run at a Fourier number of 0.3.  The limit for this stencil is
    5/16 = 0.3125, not the 1/2 the node used to document and not the
    3/8 of the bare 5-point symbol either: the cubic boundary closure
    that makes the stencil actually 4th-order tightens it further.  See
    :data:`maddening.nodes.heat.MAX_FOURIER_NUMBER` and
    :class:`TestHeatStabilityBound`.

    This was a strict xfail measuring 0.954 until the ghost
    construction was corrected (MADD-ANO-008).
    """
    node = HeatNode("mms_heat4", timestep=1e-5, n_cells=10, length=_L,
                    thermal_diffusivity=_ALPHA, stencil_order=4)
    assert declared_order(node).spatial == 4.0
    assert_node_order_verified(
        node,
        axis=RefinementAxis.SPACE,
        error_at=lambda n: _heat_steady_error(n, stencil_order=4, fourier=0.3),
        levels=(10, 20, 40, 80, 160),
    )


def test_the_fourth_order_stencil_beats_the_second_order_one(float64):
    """The option sold as more accurate has to actually be more accurate.

    MADD-ANO-008's sharpest symptom was not the order: it was that
    ``stencil_order=4`` was 10x *less* accurate than the default at
    every resolution measured, which no order test states in so many
    words.  A ghost construction that is wrong but not catastrophically
    wrong could recover an order and still lose this.
    """
    for n_cells in (20, 80):
        second = _heat_steady_error(n_cells, stencil_order=2, fourier=0.3)
        fourth = _heat_steady_error(n_cells, stencil_order=4, fourier=0.3)
        assert fourth < second, (
            f"at n={n_cells} the 4th-order stencil is less accurate than the "
            f"default: {fourth:.3e} against {second:.3e}"
        )


class TestTheSteadyProfileCanSeeABrokenScheme:
    """The manufactured solution must be able to fail the node.

    An order study is only as good as the field it refines.  These
    check the three properties ``_STEADY`` is chosen for, so that a
    later edit to it cannot quietly disarm every ladder above.
    """

    def test_the_curvature_does_not_vanish_at_either_rod_end(self):
        """Where the boundary closure's error term lives.

        The first and last rows of the discrete operator have a leading
        truncation error proportional to ``u''`` at the rod end.  A
        manufactured solution with ``u'' = 0`` there cannot distinguish
        a correct boundary closure from a wrong one: measured on the
        profile this module used before 0.4.0, a linear ghost
        extrapolation (genuinely 2nd order) read 4.080 on a 4th-order
        ladder.  With this term present the same closure reads 2.000.
        """
        d2 = jax.grad(jax.grad(lambda x: _STEADY.exact(x, 0.0)))
        for end, x in (("left", 0.0), ("right", _L)):
            curvature = float(d2(jnp.asarray(x)))
            assert abs(curvature) > 0.1, (
                f"the manufactured solution is flat at the {end} rod end "
                f"(u'' = {curvature:.3g}), so the boundary closure's error "
                f"term vanishes there and no ladder built on it can see a "
                f"wrong closure"
            )

    def test_the_fourth_derivative_does_not_vanish(self):
        """Otherwise the 5-point stencil is exact and measures nothing.

        Sampled across the rod rather than at a point: the sine's
        fourth derivative has zeros (at x = 0, L/2 and L for this
        profile), and a zero at one point says nothing about the
        truncation error of a ladder that integrates over all of them.
        """
        d4 = jax.grad(jax.grad(jax.grad(jax.grad(
            lambda x: _STEADY.exact(x, 0.0)
        ))))
        sampled = [
            abs(float(d4(jnp.asarray(x))))
            for x in np.linspace(0.0, _L, 21)
        ]
        assert max(sampled) > 1.0, (
            "the manufactured solution has no fourth derivative anywhere on "
            "the rod, so the 5-point stencil is exact on it and the "
            "4th-order ladder measures nothing"
        )

    def test_the_profile_is_not_symmetric_about_the_rod_centre(self):
        """A symmetric profile lets the two ends' errors cancel in L2."""
        left = float(_STEADY.exact(jnp.asarray(0.25 * _L), jnp.asarray(0.0)))
        right = float(_STEADY.exact(jnp.asarray(0.75 * _L), jnp.asarray(0.0)))
        assert abs(left - right) > 0.1


class TestHeatBoundaryPlacement:
    """MADD-ANO-007: where the Dirichlet datum actually lands.

    The order ladder above is the specification, but it takes five
    relaxation runs to read.  These tests fail on the defect alone, in
    milliseconds, and say which half-cell the boundary condition went
    to.
    """

    def test_a_linear_profile_between_the_rod_ends_is_reproduced_exactly(self):
        """The steady state of T(0)=0, T(L)=1 is T = x/L, at the cell centres.

        A linear profile is the exact solution of the steady heat
        equation with no source, and the discrete operator reproduces
        it exactly at every cell *if and only if* the boundary data is
        imposed at x=0 and x=L.  Impose it half a cell in and the
        steady state is instead the line through (dx/2, 0) and
        (L - dx/2, 1) — a different line, off by dx/2 of slope at every
        cell, which is the whole of MADD-ANO-007 in one number.
        """
        n_cells = 10
        node = HeatNode("rod", timestep=0.4, n_cells=n_cells, length=1.0,
                        thermal_diffusivity=1.0 / n_cells**2)
        boundary = {"left_temperature": jnp.float32(0.0),
                    "right_temperature": jnp.float32(1.0)}
        step = jax.jit(
            lambda T: node.update({"temperature": T}, boundary, 0.4)[
                "temperature"
            ]
        )
        # ~400 steps relax the slowest mode by exp(-16); 1200 is margin.
        T = jax.lax.fori_loop(
            0, 1200, lambda _, t: step(t),
            jnp.zeros(n_cells, dtype=jnp.float32),
        )
        x = np.linspace(0.05, 0.95, n_cells)
        np.testing.assert_allclose(np.asarray(T), x, atol=2e-4)

    def test_the_boundary_cell_is_not_pinned_to_the_supplied_value(self):
        """T[0] is a cell centre half a cell in, not the rod end.

        Until 0.4.0 ``update`` overwrote T[0] with ``left_temperature``,
        so this assertion was exactly inverted.  Pinning it is what
        stops the overwrite coming back as a "convenience".
        """
        node = HeatNode("rod", timestep=0.001, n_cells=10, length=1.0,
                        thermal_diffusivity=0.01)
        state = {"temperature": jnp.zeros(10, dtype=jnp.float32)}
        out = node.update(
            state, {"left_temperature": jnp.float32(100.0)}, 0.001,
        )
        first = float(out["temperature"][0])
        assert first != pytest.approx(100.0), (
            "T[0] was set to the Dirichlet value, so the boundary condition "
            "is being imposed at the first cell centre again (MADD-ANO-007)"
        )
        assert 0.0 < first < 100.0, (
            f"T[0] should warm towards the boundary value, got {first}"
        )

    def test_the_cell_centre_reading_of_the_boundary_data_still_costs_an_order(
        self, float64,
    ):
        """Supplying the old reading is now the thing that measures 1.

        The node implements one convention.  Feeding it the other one
        is a caller error, and this records what that error costs, so
        the 2.000 above cannot be mistaken for insensitivity to the
        boundary data.
        """
        measurement = measure_order(
            lambda n: _heat_steady_error(n, bc="cell_centre"),
            (10, 20, 40, 80, 160),
            axis=RefinementAxis.SPACE,
        )
        assert measurement.observed == pytest.approx(1.0, abs=0.1), (
            f"expected ~1 from the wrong boundary datum, got "
            f"{measurement.observed:.3f}\n{measurement.table()}"
        )


class TestHeatStencilGhosts:
    """MADD-ANO-008: the 5-point stencil's ghost values.

    Direct algebraic checks on ``_compute_laplacian``, so a regression
    is a one-line failure rather than a refinement table.
    """

    def test_the_laplacian_is_exact_on_a_cubic_for_the_fourth_order_stencil(self):
        """A 4th-order stencil with a cubic closure is exact on cubics.

        Including at the boundary cells: the ghost extrapolation is
        itself a cubic through the rod end, so nothing in the row has
        any error left on a cubic field.  The old ghosts failed this by
        a factor of order 1/dx at the first interior cell.
        """
        n_cells, length = 12, 1.0
        dx = length / n_cells
        x = np.linspace(dx / 2, length - dx / 2, n_cells)
        poly = lambda t: 1.0 + 0.3 * t - 0.7 * t**2 + 0.45 * t**3
        curvature = lambda t: -1.4 + 2.7 * t
        node = HeatNode("rod", timestep=1e-4, n_cells=n_cells, length=length,
                        thermal_diffusivity=1e-3, stencil_order=4)
        lap = node._compute_laplacian(
            jnp.asarray(poly(x), dtype=jnp.float32),
            jnp.float32(poly(0.0)),
            jnp.float32(poly(length)),
        )
        np.testing.assert_allclose(
            np.asarray(lap), curvature(x), rtol=2e-3, atol=2e-3,
        )

    def test_the_second_order_laplacian_is_exact_on_a_linear_profile(self):
        """The mirror ghost makes the 3-point row exact on linears.

        That is the property the conservative closure buys, and it is
        what lets the temporal study below isolate the time integrator.
        """
        n_cells, length = 12, 1.0
        dx = length / n_cells
        x = np.linspace(dx / 2, length - dx / 2, n_cells)
        line = lambda t: 2.0 - 1.3 * t
        node = HeatNode("rod", timestep=1e-4, n_cells=n_cells, length=length,
                        thermal_diffusivity=1e-3)
        lap = node._compute_laplacian(
            jnp.asarray(line(x), dtype=jnp.float32),
            jnp.float32(line(0.0)),
            jnp.float32(line(length)),
        )
        np.testing.assert_allclose(np.asarray(lap), 0.0, atol=1e-3)

    def test_an_unsupplied_boundary_condition_conserves_energy(self):
        """No Dirichlet data means no flux through the ends, exactly.

        With ``left_temperature`` defaulted to ``T[0]`` the mirror ghost
        is ``T[0]`` itself, so the end face carries zero gradient and
        the total heat is conserved to round-off.  Before 0.4.0 the end
        cells were frozen at their previous values instead, which both
        leaked energy and made the ends unphysically static.
        """
        node = HeatNode("rod", timestep=0.1, n_cells=8, length=1.0,
                        thermal_diffusivity=0.01)
        rng = np.random.default_rng(0)
        T = jnp.asarray(rng.uniform(10.0, 100.0, 8), dtype=jnp.float32)
        state = {"temperature": T}
        before = float(jnp.sum(T))
        for _ in range(50):
            state = node.update(state, {}, 0.1)
        after = float(jnp.sum(state["temperature"]))
        assert after == pytest.approx(before, rel=1e-5)


#: Exact gradient of :data:`_STEADY` at ``t = 0``, by AD rather than by
#: hand, for the same reason the source term is: a hand-differentiated
#: reference is a second place for the profile to be wrong.
_STEADY_GRADIENT = jax.grad(lambda x: _STEADY.exact(x, jnp.float64(0.0)))


def _heat_boundary_flux_error(
    n_cells, *, end="left", stencil_order=2, datum=True,
):
    """Relative error of a reported rod-end flux on the steady profile.

    The field is handed to the node exactly -- no relaxation -- because
    what is being measured is the boundary *reconstruction*, not the
    scheme that produced the field.  Mixing the two would let a
    2nd-order state error mask a 1st-order flux.

    ``datum=False`` withholds ``left_temperature`` /
    ``right_temperature``, which is the path
    ``compute_boundary_fluxes`` takes when a caller asks for a flux
    without a Dirichlet boundary condition.
    """
    dx = _L / n_cells
    x = _cell_centres(n_cells)
    T = np.asarray(_STEADY.field(x, 0.0), dtype=np.float64)
    node = HeatNode(
        "flux_heat", timestep=0.3 * dx * dx / _ALPHA, n_cells=n_cells,
        length=_L, thermal_diffusivity=_ALPHA, stencil_order=stencil_order,
    )
    boundary = {}
    if datum:
        boundary = {
            "left_temperature": jnp.asarray(
                float(_STEADY.exact(jnp.float64(0.0), jnp.float64(0.0)))
            ),
            "right_temperature": jnp.asarray(
                float(_STEADY.exact(jnp.float64(_L), jnp.float64(0.0)))
            ),
        }
    fluxes = node.compute_boundary_fluxes(
        {"temperature": jnp.asarray(T)}, boundary, dx * dx,
    )
    at = jnp.float64(0.0 if end == "left" else _L)
    true_flux = -_ALPHA * float(_STEADY_GRADIENT(at))
    reported = float(fluxes[f"{end}_heat_flux"])
    return abs(reported - true_flux) / abs(true_flux)


class TestTheReportedBoundaryFluxIsAtTheRodEnd:
    """``compute_boundary_fluxes`` returns the flux the spec names.

    Until 0.4.0 it returned ``-alpha (T[1] - T[0]) / dx``, the gradient
    of the line through the first two cell *centres*, which for a
    cell-centred grid sits at ``x = dx`` -- a full cell inside the end
    ``boundary_flux_spec`` calls "the left boundary".  On ``T = exp(x)``
    with ``alpha = 1`` and ``n = 10`` that is -1.10563 where the
    rod-end flux is -1.0, a 10.6% error, and it matched the flux at
    ``x = dx`` (-1.10517) to four figures.  It refined at order 1.005
    against the node's own 2.000, so a flux-coupled solve was capped at
    1st order by the flux alone.

    Nothing asserted the value.  The three tests that touched it
    checked ``isfinite`` and ``> 0.0``, and a *linear* profile cannot
    see the defect at all -- the scheme is exact there and both
    readings give exactly -1.0 -- so the profile these studies refine
    has to be curved at the ends, exactly as
    :class:`TestTheSteadyProfileCanSeeABrokenScheme` requires of the
    state ladder one layer down.
    """

    @pytest.mark.parametrize("end", ["left", "right"])
    def test_the_flux_converges_at_the_default_stencils_order(
        self, float64, end,
    ):
        """2nd order, matching the state: measured 1.999 (was 1.005)."""
        measurement = measure_order(
            lambda n: _heat_boundary_flux_error(n, end=end),
            levels=(10, 20, 40, 80, 160),
            axis=RefinementAxis.SPACE,
        )
        assert measurement.monotone, measurement.table()
        assert measurement.observed == pytest.approx(2.0, abs=0.2), (
            measurement.table()
        )

    @pytest.mark.parametrize("end", ["left", "right"])
    def test_the_flux_converges_at_the_fourth_order_stencils_order(
        self, float64, end,
    ):
        """4th order for ``stencil_order=4``: measured 3.993.

        The interface is not capped below the state it belongs to.  The
        reconstruction reads the rod-end datum plus ``stencil_order``
        cells, so it is accurate to ``stencil_order`` at either end.
        """
        measurement = measure_order(
            lambda n: _heat_boundary_flux_error(n, end=end, stencil_order=4),
            levels=(10, 20, 40, 80, 160),
            axis=RefinementAxis.SPACE,
        )
        assert measurement.monotone, measurement.table()
        assert measurement.observed == pytest.approx(4.0, abs=0.2), (
            measurement.table()
        )

    @pytest.mark.parametrize("end", ["left", "right"])
    @pytest.mark.parametrize("stencil_order", [2, 4])
    def test_the_flux_without_a_dirichlet_datum_is_still_at_the_rod_end(
        self, float64, stencil_order, end,
    ):
        """No boundary input: extrapolate to the end, do not move it.

        With no datum the reconstruction is a pure extrapolation from
        three cells and 2nd order for either stencil -- a higher-degree
        extrapolant costs conditioning (coefficient L1 norm per ``dx``
        of 6 for three cells against 28.3 for five) to buy accuracy
        against a boundary condition that was never supplied.  What
        matters is that it still reports the flux *at the rod end*:
        measured 1.996, where the old expression measured 1.007 here
        too.
        """
        measurement = measure_order(
            lambda n: _heat_boundary_flux_error(
                n, end=end, stencil_order=stencil_order, datum=False,
            ),
            levels=(10, 20, 40, 80, 160),
            axis=RefinementAxis.SPACE,
        )
        assert measurement.monotone, measurement.table()
        assert measurement.observed == pytest.approx(2.0, abs=0.2), (
            measurement.table()
        )

    def test_the_reported_value_is_the_rod_end_flux_and_not_the_first_face(
        self, float64,
    ):
        """The cheap direct test: one number, on ``T = exp(x)``.

        An order ladder says a regression happened; this says what the
        number is.  Both readings are pinned, so a fix that moved the
        flux to some third location would fail this too.
        """
        n_cells = 10
        dx = _L / n_cells
        x = _cell_centres(n_cells)
        node = HeatNode(
            "flux_exp", timestep=0.3 * dx * dx, n_cells=n_cells, length=_L,
            thermal_diffusivity=1.0,
        )
        fluxes = node.compute_boundary_fluxes(
            {"temperature": jnp.asarray(np.exp(x))},
            {"left_temperature": jnp.asarray(1.0),
             "right_temperature": jnp.asarray(float(np.exp(_L)))},
            dx * dx,
        )
        left = float(fluxes["left_heat_flux"])
        # -alpha exp'(0) = -1; the flux one cell in is -exp(dx).
        assert left == pytest.approx(-1.0, abs=5e-3), (
            f"left flux {left} is not the rod-end flux -1.0"
        )
        assert abs(left - -float(np.exp(dx))) > 1e-2, (
            f"left flux {left} is still the flux at x = dx "
            f"({-float(np.exp(dx))})"
        )
        right = float(fluxes["right_heat_flux"])
        assert right == pytest.approx(-float(np.exp(_L)), rel=5e-3), (
            f"right flux {right} is not the rod-end flux {-np.exp(_L)}"
        )
        assert abs(right - -float(np.exp(_L - dx))) > 1e-2

    def test_a_linear_profile_cannot_tell_the_two_readings_apart(
        self, float64,
    ):
        """Why the studies above may not use a straight line.

        On ``T = x`` the scheme is exact and *both* readings give
        exactly -1.0 at both ends, so a linear-profile test -- the
        obvious one to reach for, and the shape
        ``tests/core/test_flux_coupling.py`` happened to use -- is
        blind to the whole defect.  Recorded as an assertion so the
        next person to simplify these studies finds out here.
        """
        n_cells = 10
        dx = _L / n_cells
        x = _cell_centres(n_cells)
        node = HeatNode(
            "flux_linear", timestep=0.3 * dx * dx, n_cells=n_cells,
            length=_L, thermal_diffusivity=1.0,
        )
        T = jnp.asarray(x)
        fluxes = node.compute_boundary_fluxes(
            {"temperature": T},
            {"left_temperature": jnp.asarray(0.0),
             "right_temperature": jnp.asarray(_L)},
            dx * dx,
        )
        old_reading_left = -float(T[1] - T[0]) / dx
        assert float(fluxes["left_heat_flux"]) == pytest.approx(-1.0, abs=1e-12)
        assert old_reading_left == pytest.approx(-1.0, abs=1e-12)
        assert float(fluxes["right_heat_flux"]) == pytest.approx(
            -1.0, abs=1e-12,
        )

    def test_the_two_ends_of_one_rod_report_a_balanced_pair(self, float64):
        """A curved profile: the two ends must not err with one sign.

        The old reading was the flux one cell *inside* each end, so on
        any curved profile the two ends were wrong in opposite
        directions and their sum -- the net flux into the rod, which is
        what a conservation check reads -- carried the whole error.  On
        ``T = exp(x)`` at n = 10 the old pair gave a net 1.35500
        against the true 1.71828 -- 21.1% out -- while each end alone
        was only 10.6% and 9.5% out; it is now within 0.25%.
        """
        n_cells = 10
        dx = _L / n_cells
        x = _cell_centres(n_cells)
        node = HeatNode(
            "flux_balance", timestep=0.3 * dx * dx, n_cells=n_cells,
            length=_L, thermal_diffusivity=1.0,
        )
        fluxes = node.compute_boundary_fluxes(
            {"temperature": jnp.asarray(np.exp(x))},
            {"left_temperature": jnp.asarray(1.0),
             "right_temperature": jnp.asarray(float(np.exp(_L)))},
            dx * dx,
        )
        net = float(fluxes["left_heat_flux"]) - float(fluxes["right_heat_flux"])
        true_net = -1.0 + float(np.exp(_L))
        assert net == pytest.approx(true_net, rel=5e-3), (
            f"net flux {net} against the true {true_net}"
        )


class TestHeatStabilityBound:
    """MADD-ANO-009: the Fourier limit, per stencil, and the guard on it."""

    def test_the_second_order_stencil_is_stable_up_to_one_half(self, float64):
        error = _heat_steady_error(20, fourier=MAX_FOURIER_NUMBER[2])
        assert np.isfinite(error) and error < 1.0, (
            f"Fo=1/2 should be stable for the default stencil, got {error}"
        )

    def test_the_fourth_order_stencil_is_stable_up_to_five_sixteenths(
        self, float64,
    ):
        error = _heat_steady_error(
            20, stencil_order=4, fourier=MAX_FOURIER_NUMBER[4],
        )
        assert np.isfinite(error) and error < 1.0, (
            f"Fo=5/16 should be stable for the 4th-order stencil, got {error}"
        )

    def test_the_fourth_order_limit_is_below_the_three_eighths_of_its_symbol(
        self, float64,
    ):
        """Driven past 5/16 by hand, the 4th-order stencil does diverge.

        The constructor refuses this configuration, so the run has to be
        built by passing an oversized ``dt`` to ``update`` — which is
        also an honest demonstration of what the guard does *not*
        cover.  3/8 is the bound of the 5-point symbol alone and was
        what MADD-ANO-009 recorded; the cubic boundary closure brings
        the whole operator's bound down to 0.3249.
        """
        n_cells = 20
        dx = _L / n_cells
        safe_dt = 0.3 * dx * dx / _ALPHA
        node = HeatNode("rod4", timestep=safe_dt, n_cells=n_cells, length=_L,
                        thermal_diffusivity=_ALPHA, stencil_order=4)
        x = _cell_centres(n_cells)
        exact = np.asarray(_STEADY.field(x, 0.0), dtype=np.float64)
        boundary = {
            "left_temperature": jnp.asarray(
                float(_STEADY.exact(jnp.float64(0.0), jnp.float64(0.0)))
            ),
            "right_temperature": jnp.asarray(
                float(_STEADY.exact(jnp.float64(_L), jnp.float64(0.0)))
            ),
            "heat_source": jnp.asarray(_STEADY.source_field(x, 0.0)),
        }
        unstable_dt = 0.375 * dx * dx / _ALPHA
        T = jnp.asarray(exact)
        for _ in range(400):
            T = node.update({"temperature": T}, boundary, unstable_dt)[
                "temperature"
            ]
        peak = float(jnp.max(jnp.abs(T)))
        assert not np.isfinite(peak) or peak > 1e3, (
            f"Fo=3/8 is above this operator's 0.3249 bound and should "
            f"diverge, but max|T| was {peak}; if the boundary closure has "
            f"changed, re-measure MAX_FOURIER_NUMBER and MADD-ANO-009"
        )

    @pytest.mark.parametrize(
        ("stencil_order", "fourier"), [(2, 0.51), (4, 0.33)],
    )
    def test_the_constructor_refuses_an_unstable_configuration(
        self, stencil_order, fourier,
    ):
        """A silent NaN is the worst available behaviour, so it is an error.

        ``fourier=0.33`` is inside both the 1/2 the node used to
        document and the 3/8 of the bare 5-point symbol, and it
        diverges: that combination is why this is refused rather than
        merely written down.
        """
        n_cells = 20
        dx = _L / n_cells
        with pytest.raises(ValueError, match="Fourier number"):
            HeatNode(
                "rod", timestep=fourier * dx * dx / _ALPHA, n_cells=n_cells,
                length=_L, thermal_diffusivity=_ALPHA,
                stencil_order=stencil_order,
            )

    def test_a_stable_configuration_is_accepted(self):
        """The guard must not be a blanket refusal of the 4th-order stencil."""
        dx = _L / 20
        node = HeatNode(
            "rod", timestep=0.31 * dx * dx / _ALPHA, n_cells=20, length=_L,
            thermal_diffusivity=_ALPHA, stencil_order=4,
        )
        assert node.params["stencil_order"] == 4


# --------------------------------------------------------------------------
# HeatNode — temporal order
# --------------------------------------------------------------------------

_OMEGA = 20.0

#: Linear in x, so the discrete Laplacian returns exactly zero at every
#: cell and the spatial error vanishes identically: what is left is the
#: time integrator.  Oscillating fast enough in t that the temporal
#: error stays well clear of round-off.
#:
#: This was quadratic in x until 0.4.0, because the old scheme
#: *overwrote* the end cells with the exact solution and so was exact on
#: quadratics there for the wrong reason.  The conservative mirror
#: closure that replaced it (MADD-ANO-007) is exact on linears at the
#: boundary rows and 0.75x the curvature on quadratics, which would have
#: left a dt-independent spatial floor in this ladder and turned it
#: non-monotone.  A linear profile is the one this scheme represents
#: exactly everywhere, which is what the axis-isolation rule in
#: ``maddening.testing.mms`` asks for.  The spatial operator is measured
#: by MADD-VER-005, not here.
_TRANSIENT = ManufacturedSolution(
    exact=lambda x, t: (1.0 + 0.7 * x) * jnp.cos(_OMEGA * t),
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
            # At the rod ends, which is where the node imposes them.
            "left_temperature": _TRANSIENT.exact(jnp.asarray(0.0), t),
            "right_temperature": _TRANSIENT.exact(jnp.asarray(_L), t),
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
        "Solutions: manufactured solution linear in x, so the discrete "
        "Laplacian is identically zero, the spatial error vanishes and the "
        "timestep ladder measures the forward-Euler integration alone"
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


class TestTheOrderBandIsJustifiedByTheseNumbers:
    """Every figure the two band constants cite, checked two ways.

    ``DEFAULT_ORDER_SHORTFALL`` and ``DEFAULT_ORDER_EXCESS`` are
    justified in ``mms.py`` by measured figures.  An earlier revision
    justified them with figures no ladder in the tree produces -- "the
    corrected fourth-order stencil measures 5.02", where the largest
    pairwise order over four manufactured solutions is 4.126, and
    "as much as 0.16 low (1.847)", where the coarsest pairs measure
    2.007-2.023, *above* theory.  Worse, 5.02 was cited as the
    superconvergence the band accommodates while lying outside the
    band's own upper edge of ``4 + 1.0 = 5.00``, so the figure quoted
    in its defence would have failed it.

    So each figure is pinned twice: it has to appear in the recorded
    fixture it is attributed to, and ``check_order`` has to make the
    decision on it that the docstring claims.  A figure that drifts out
    of the band it is cited as being inside fails here rather than
    sitting on the page.
    """

    _FIXTURES = _REPO_ROOT / "benchmarks" / "results" / "audit_040_r2" / "numerics"

    #: ``(figure, fixture file, what check_order does with it against 4)``.
    _FOURTH_ORDER_EVIDENCE = (
        # Correct cubic ghost closure -- the band must admit all of these.
        ("4.126", "r8_order_band.log", True),   # largest pairwise, tanh_bump
        ("4.001", "r8_order_band.log", True),   # largest finest-pair
        ("3.957", "r8_order_band.log", True),   # release profile
        ("3.760", "r8_order_band.log", True),   # coarsest pair, same study
        # The disarmed-profile defect.  The band does NOT catch it.
        ("4.080", "r8_order_band.log", True),
    )

    _SECOND_ORDER_EVIDENCE = (
        ("2.0004", "r1_heat_order_independent.log", True),
        ("2.023", "r1_heat_order_independent.log", True),
        ("1.998", "r10_lbm_order.log", True),
    )

    _FIRST_ORDER_EVIDENCE = (
        ("1.0009", "r6_declared_orders.log", True),
        ("1.0003", "r6_declared_orders.log", True),
        ("1.0002", "r6_declared_orders.log", True),
    )

    @pytest.mark.parametrize(
        ("figure", "fixture"),
        [(f, x) for f, x, _ in
         _FOURTH_ORDER_EVIDENCE + _SECOND_ORDER_EVIDENCE + _FIRST_ORDER_EVIDENCE],
    )
    def test_the_figure_is_in_the_fixture_it_is_attributed_to(
        self, figure, fixture,
    ):
        """A cited measurement has to exist where it says it does."""
        path = self._FIXTURES / fixture
        assert path.is_file(), f"cited fixture {path} is missing"
        assert figure in path.read_text(), (
            f"{figure} is cited against {fixture} but does not appear in it; "
            f"re-run benchmarks/results/audit_040_r2/numerics/repro/ and "
            f"quote what it now says"
        )

    @pytest.mark.parametrize("fixture", [
        "r8_order_band.log", "r1_heat_order_independent.log",
        "r6_declared_orders.log", "r10_lbm_order.log",
    ])
    def test_every_fixture_mms_names_is_present(self, fixture):
        """``_ORDER_BAND_FIXTURES`` must not be a list of dead paths."""
        from maddening.testing.mms import _ORDER_BAND_FIXTURES
        relative = f"benchmarks/results/audit_040_r2/numerics/{fixture}"
        assert relative in _ORDER_BAND_FIXTURES
        assert (_REPO_ROOT / relative).is_file()

    @pytest.mark.parametrize(
        ("figure", "expected"),
        [(f, 4.0) for f, _, _ in _FOURTH_ORDER_EVIDENCE]
        + [(f, 2.0) for f, _, _ in _SECOND_ORDER_EVIDENCE]
        + [(f, 1.0) for f, _, _ in _FIRST_ORDER_EVIDENCE],
    )
    def test_the_band_admits_every_recorded_measurement(self, figure, expected):
        """Nothing the fixtures recorded as correct may fail the band."""
        result = check_order(_ladder(float(figure)), expected)
        assert result.passed, result.detail

    @pytest.mark.parametrize(
        ("observed", "expected"),
        [(1.001, 2.0),   # MADD-ANO-007, boundary datum half a cell in
         (0.954, 4.0)],  # MADD-ANO-008, ghosts at the wrong positions
    )
    def test_the_band_rejects_the_defects_it_is_credited_with(
        self, observed, expected,
    ):
        result = check_order(_ladder(observed), expected)
        assert result.failed
        assert "below the declared" in result.detail

    def test_the_band_does_not_catch_the_disarmed_profile_defect(self):
        """Recorded as a pass, because it is one, and that is the point.

        A linear ghost closure -- genuinely 2nd order -- measured 4.080
        on the pre-0.4.0 flat-ended profile and sailed through.  What
        catches it is
        :class:`TestTheSteadyProfileCanSeeABrokenScheme`, which forbids
        a manufactured solution flat at the rod ends; on a curved one
        the same closure measures 2.000 and the *shortfall* rejects it.
        """
        assert check_order(_ladder(4.080), 4.0).passed
        assert check_order(_ladder(2.000), 4.0).failed

    @pytest.mark.parametrize(
        "excess", [0.02, 0.05, 0.08, 0.1, 0.126, 0.15, 0.25, 0.5, 1.0, 1.5],
    )
    def test_no_excess_separates_the_defect_from_a_correct_study(self, excess):
        """The trap, made executable: do not tighten this band.

        ``check_order`` gates on the finest pair of whatever ladder it
        is handed, and this constant is global.  The defect's gated
        value is 4.080; a correct ``tanh_bump`` study read over its
        coarsest pair -- which is the gated value of any two- or
        three-level ladder of it -- is 4.126.  4.080 < 4.126, so every
        upper edge that rejects the defect also rejects that study.
        Tightening ``DEFAULT_ORDER_EXCESS`` cannot be the fix for the
        disarmed-profile case at any value whatsoever.
        """
        defect = check_order(_ladder(4.080), 4.0, excess=excess).passed
        correct = check_order(_ladder(4.126), 4.0, excess=excess).passed
        assert defect or not correct, (
            f"excess={excess} rejects the 4.080 defect while still admitting "
            f"the legitimate 4.126 study -- which the recorded fixtures say "
            f"is impossible, so one of them has moved"
        )

    def test_every_admissible_figure_is_below_the_edge_it_is_cited_against(
        self,
    ):
        """The arithmetic guard the old docstring failed.

        "1.0 is wide enough for X" is false whenever ``X > 4 + 1.0``.
        The figure that sentence used to name, 5.02, is exactly that
        case.
        """
        edge = 4.0 + DEFAULT_ORDER_EXCESS
        assert edge == 5.0
        for figure, _, admissible in self._FOURTH_ORDER_EVIDENCE:
            assert admissible and float(figure) < edge, (
                f"{figure} is cited as admissible but the band's upper edge "
                f"for a declared 4 is {edge}"
            )
        # The figure the docstring used to cite, kept as the negative
        # control for this check.
        assert 5.02 > edge

    def test_the_shortfall_sits_between_the_wobble_and_the_defects(self):
        """0.25 is a ratio to two measured quantities, not a round number.

        Largest honest finest-pair shortfall recorded: 0.043 (the
        4th-order heat stencil's 3.957 against 4).  Smallest defect it
        must reject: 1.0 (MADD-ANO-007's 1.001 against 2).
        """
        largest_honest_shortfall = 4.0 - 3.957
        smallest_defect_shortfall = 2.0 - 1.001
        assert largest_honest_shortfall < DEFAULT_ORDER_SHORTFALL
        assert DEFAULT_ORDER_SHORTFALL < smallest_defect_shortfall
        assert DEFAULT_ORDER_SHORTFALL / largest_honest_shortfall > 5.0
        assert smallest_defect_shortfall / DEFAULT_ORDER_SHORTFALL > 3.5


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
