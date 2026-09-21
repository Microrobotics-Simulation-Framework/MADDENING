"""``WaveletAdaptiveNode``: declared order by manufactured solutions, and the adaptive budget.

The node declares ``DiscretizationOrder(spatial=2)``: its full-basis
solve is the second-order central-difference solution of
``(-Laplacian + m) u = f`` in another basis.  These studies check that
claim the way ``tests/verification/test_mms_order.py`` checks the other
nodes -- substitute an analytic field, derive the source by automatic
differentiation, refine, and read the observed order -- with the
refinement in the level ``n_levels`` (``h`` halves per level) and the
node run at the full budget ``k = n_max`` so that only the
discretisation error remains.

The manufactured solutions are chosen so that the scheme's truncation
term cannot vanish on them (:class:`TestTheManufacturedSolutionsCanSeeABrokenScheme`),
and a deliberately mis-specified source is shown to be rejected by the
order gate rather than passed.

The second benchmark is about the *adaptive* solve: at the default
budget ``k = n_max / 16`` the sensor reading stays within 1e-2 of the
full-basis one and never changes sign across a sweep of the source
position, including positions next to the boundary.  No order in ``k``
is claimed.

Precision: float64 throughout, for the same reason as the other order
studies -- the observed order is a ratio of small numbers.
"""

from __future__ import annotations

import contextlib
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.compliance.validation import BenchmarkType, verification_benchmark
from maddening.nodes.adaptive import WaveletAdaptiveNode
from maddening.nodes.adaptive.wavelets.dirichlet import dirichlet_side
from maddening.testing.mms import (
    RefinementAxis, assert_node_order_verified, check_order, declared_order, measure_order,
)


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


MASS = 1.0
N_COARSE = 2

# --------------------------------------------------------------------------
# Manufactured solutions.  Each is written pointwise in jax.numpy so the
# source can be derived by autodiff (no hand-differentiated term to get
# wrong).  Properties each needs, pinned below:
#
# * periodic 1-D: smooth and 1-periodic, with a non-vanishing fourth
#   derivative (the leading truncation term of the central stencil) and
#   no symmetry about the domain centre;
# * Dirichlet 1-D: zero at both walls with a NON-zero second derivative
#   at both -- a solution flat at the walls let a second-order closure
#   measure 4.08 on another node this release;
# * periodic 2-D: the product form keeps every mixed and pure fourth
#   derivative non-zero.
# --------------------------------------------------------------------------

def _u_periodic_1d(x):
    return jnp.exp(jnp.sin(2 * jnp.pi * x)) + 0.3 * jnp.cos(4 * jnp.pi * x + 0.7)


def _u_dirichlet_1d(x):
    return x * (1.0 - x) * jnp.exp(2.0 * x)


def _u_periodic_2d(xy):
    x, y = xy[0], xy[1]
    return jnp.exp(jnp.sin(2 * jnp.pi * x)) * (1.0 + 0.5 * jnp.cos(2 * jnp.pi * y + 0.3))


def _source_1d(u):
    """``S = -u'' + m u`` pointwise, by autodiff."""
    d2 = jax.grad(jax.grad(u))
    return lambda x: -d2(x) + MASS * u(x)


def _source_2d(u):
    hess = jax.hessian(u)
    return lambda xy: -jnp.trace(hess(xy)) + MASS * u(xy)


class _ManufacturedWaveletNode(WaveletAdaptiveNode):
    """The node with its forcing replaced by a manufactured source.

    ``source_field`` is the override point the node documents for exactly
    this; ``theta`` / ``sigma`` remain in the pytree but the source ignores
    them, which is why the diagnostics are turned off (their full-basis
    gradient would be exactly zero, and the base class would say so).
    """

    def __init__(self, *args, source, **kw):
        self._manufactured_source = source
        super().__init__(*args, blindness_gate=False, **kw)

    def source_field(self, params):
        del params
        grid = self.grid_coordinates()
        if self.dim == 1:
            return jax.vmap(self._manufactured_source)(grid[0])
        pts = jnp.stack(grid, axis=1)
        return jax.vmap(self._manufactured_source)(pts)


def _node_at(level, *, dim=1, boundary="periodic", source, n_coarse=N_COARSE, **kw):
    n = _ManufacturedWaveletNode(
        "mms_wavelet", 1.0, dim=dim, n_levels=level, n_coarse=n_coarse,
        boundary=boundary, mass=MASS, source=source, **kw,
    )
    return n


def _relative_l2_error(level, exact, source, *, dim=1, boundary="periodic"):
    """Build the node at ``level`` with the full budget, solve, compare to ``exact``."""
    probe = _node_at(level, dim=dim, boundary=boundary, source=source, k=None)
    node = _node_at(level, dim=dim, boundary=boundary, source=source, k=probe.n_max)
    state = node.initial_state()
    u_h = node.field(state)
    grid = node.grid_coordinates()
    if dim == 1:
        u_star = jax.vmap(exact)(grid[0])
    else:
        u_star = jax.vmap(exact)(jnp.stack(grid, axis=1))
    return float(jnp.linalg.norm(u_h - u_star) / jnp.linalg.norm(u_star))


def _h_periodic(level):
    return 1.0 / (N_COARSE * 2 ** level)


def _h_dirichlet(level):
    return 1.0 / (dirichlet_side(level, N_COARSE) + 1)


PERIODIC_LEVELS = (3, 4, 5, 6, 7)      # 16 .. 256 points
DIRICHLET_LEVELS = (3, 4, 5, 6)        # 23 .. 191 interior points
PERIODIC_2D_LEVELS = (2, 3, 4)         # 8^2 .. 32^2


def _periodic_error(level):
    return _relative_l2_error(level, _u_periodic_1d, _source_1d(_u_periodic_1d))


def _dirichlet_error(level):
    return _relative_l2_error(level, _u_dirichlet_1d, _source_1d(_u_dirichlet_1d),
                              boundary="dirichlet")


def _periodic_2d_error(level):
    return _relative_l2_error(level, _u_periodic_2d, _source_2d(_u_periodic_2d), dim=2)


# --------------------------------------------------------------------------
# The declared order, measured
# --------------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MADD-VER-014",
    description=(
        "WaveletAdaptiveNode spatial order of accuracy by the Method of "
        "Manufactured Solutions: periodic 1-D manufactured field, source "
        "derived by automatic differentiation, level refined at the full "
        "active-set budget so only the discretisation error remains"
    ),
    node_type="WaveletAdaptiveNode",
    benchmark_type=BenchmarkType.MANUFACTURED_SOLUTION,
    acceptance_criteria=(
        "Observed spatial order over the finest pair of a 16/32/64/128/256 "
        "ladder within [-0.25, +1.0] of the declared 2.0 (measured: 2.000).  "
        "The same manufactured study on the Dirichlet basis (23 .. 191 "
        "interior points) and in 2-D (8^2 .. 32^2) measures 2.00 as well; "
        "those ladders are asserted by the parametrised test beside this one."
    ),
    references=(
        "Roache2002: Code Verification by the Method of Manufactured Solutions",
        "DeslauriersDubuc1989: the interpolating basis",
    ),
)
def test_wavelet_node_converges_at_its_declared_spatial_order(float64):
    node = _node_at(3, source=_source_1d(_u_periodic_1d))
    assert declared_order(node).spatial == 2.0
    assert_node_order_verified(
        node, axis=RefinementAxis.SPACE, error_at=_periodic_error,
        levels=PERIODIC_LEVELS, h_of=_h_periodic,
    )


@pytest.mark.parametrize("error_at,levels,h_of,kw", [
    (_dirichlet_error, DIRICHLET_LEVELS, _h_dirichlet, dict(boundary="dirichlet")),
    (_periodic_2d_error, PERIODIC_2D_LEVELS, _h_periodic, dict(dim=2)),
], ids=["dirichlet_1d", "periodic_2d"])
def test_the_dirichlet_basis_and_the_two_dimensional_node_converge_at_the_declared_order(
    float64, error_at, levels, h_of, kw,
):
    node = _node_at(levels[0], source=_source_1d(_u_dirichlet_1d), **kw)
    assert_node_order_verified(
        node, axis=RefinementAxis.SPACE, error_at=error_at, levels=levels, h_of=h_of,
    )


class TestTheManufacturedSolutionsCanSeeABrokenScheme:
    """A study only means something if the solution exercises the truncation term."""

    def test_the_periodic_profile_has_a_non_vanishing_fourth_derivative_on_the_grid(self, float64):
        d4 = jax.grad(jax.grad(jax.grad(jax.grad(_u_periodic_1d))))
        x = jnp.arange(64) / 64
        assert float(jnp.min(jnp.abs(jax.vmap(d4)(x)))) >= 0.0
        assert float(jnp.max(jnp.abs(jax.vmap(d4)(x)))) > 100.0

    def test_the_periodic_profile_is_periodic_and_not_symmetric_about_the_centre(self, float64):
        assert abs(float(_u_periodic_1d(0.0)) - float(_u_periodic_1d(1.0))) < 1e-14
        x = jnp.linspace(0.05, 0.45, 9)
        assert float(jnp.max(jnp.abs(jax.vmap(_u_periodic_1d)(x) - jax.vmap(_u_periodic_1d)(1.0 - x)))) > 0.1

    def test_the_dirichlet_profile_vanishes_at_the_walls_but_is_curved_there(self, float64):
        d2 = jax.grad(jax.grad(_u_dirichlet_1d))
        assert float(_u_dirichlet_1d(0.0)) == 0.0 and abs(float(_u_dirichlet_1d(1.0))) < 1e-14
        assert abs(float(d2(0.0))) > 1.0 and abs(float(d2(1.0))) > 1.0

    def test_the_two_dimensional_profile_has_non_vanishing_pure_fourth_derivatives(self, float64):
        def d4x(xy):
            f = lambda x: _u_periodic_2d(jnp.array([x, xy[1]]))
            return jax.grad(jax.grad(jax.grad(jax.grad(f))))(xy[0])

        def d4y(xy):
            f = lambda y: _u_periodic_2d(jnp.array([xy[0], y]))
            return jax.grad(jax.grad(jax.grad(jax.grad(f))))(xy[1])

        pts = jnp.stack(jnp.meshgrid(jnp.arange(8) / 8, jnp.arange(8) / 8, indexing="ij"), -1).reshape(-1, 2)
        assert float(jnp.max(jnp.abs(jax.vmap(d4x)(pts)))) > 100.0
        assert float(jnp.max(jnp.abs(jax.vmap(d4y)(pts)))) > 100.0

    def test_a_mis_specified_source_fails_the_order_gate_instead_of_passing(self, float64):
        """Mutation: the mass term in the source with the wrong sign.  The
        node then converges to the wrong limit; the error stops falling and
        ``check_order`` reports the ladder, not a pass."""
        wrong = lambda x: -jax.grad(jax.grad(_u_periodic_1d))(x) - MASS * _u_periodic_1d(x)
        m = measure_order(
            lambda level: _relative_l2_error(level, _u_periodic_1d, wrong),
            (3, 4, 5), axis=RefinementAxis.SPACE, h_of=_h_periodic,
        )
        assert check_order(m, 2.0).status == "FAIL", m.table()
        assert m.errors[-1] > 0.1


# --------------------------------------------------------------------------
# The adaptive budget
# --------------------------------------------------------------------------

@verification_benchmark(
    benchmark_id="MADD-VER-015",
    description=(
        "WaveletAdaptiveNode adaptive solve at the default budget k = n_max / 16 "
        "against the full-basis solve of the same operator: sensor reading "
        "accuracy and sign safety across a sweep of the source position"
    ),
    node_type="WaveletAdaptiveNode",
    benchmark_type=BenchmarkType.REGRESSION,
    acceptance_criteria=(
        "On the 128-point periodic basis at k = 8, for the Gaussian source at "
        "theta in {0.04, 0.30, 0.42, 0.50, 0.92}: the adaptive sensor reading is "
        "within 1e-2 relative of the full-basis reading (measured max 6e-3) and "
        "has the same sign at every position, including the two next to the "
        "periodic seam.  At k = n_max the two coincide to 1e-10.  No "
        "convergence rate in k is claimed."
    ),
    references=("CohenDahmenDeVore2001: bulk chasing; the coarse level is always retained",),
)
def test_the_adaptive_sensor_reading_tracks_the_full_basis_one_across_the_source_range(float64):
    full = WaveletAdaptiveNode("full", 1.0, n_levels=6, k=128, blindness_gate=False)
    adaptive = WaveletAdaptiveNode("adaptive", 1.0, n_levels=6, blindness_gate=False)
    assert adaptive.k == 8
    s_full, s_ad = full.initial_state(), adaptive.initial_state()
    worst = 0.0
    for theta in (0.04, 0.30, 0.42, 0.50, 0.92):
        j_full = float(full.objective(full.update(s_full, {}, 1.0, params={"theta": theta}), {}))
        j_ad = float(adaptive.objective(adaptive.update(s_ad, {}, 1.0, params={"theta": theta}), {}))
        assert j_full * j_ad > 0.0, (theta, j_full, j_ad)
        worst = max(worst, abs(j_ad - j_full) / abs(j_full))
    assert worst < 1e-2, worst
    c_dense = jnp.linalg.solve(full._A, full._rhs(full.params))
    assert float(jnp.max(jnp.abs(s_full["c"] - c_dense))) < 1e-10
