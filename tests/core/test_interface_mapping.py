"""Tests for spatial interpolation map factories (Phase 4b).

Verifies correctness, JAX compatibility, and integration with EdgeSpec
for nearest-neighbor, linear, RBF, and conservative projection maps.
"""

import jax
import jax.numpy as jnp
import pytest

from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.coupling.interface_mapping import (
    conservative_projection_1d,
    linear_interpolation_1d,
    nearest_neighbor_1d,
    rbf_interpolation,
)
from maddening.nodes.heat import HeatNode


# ==================================================================
# Nearest-neighbor 1D
# ==================================================================

class TestNearestNeighbor1D:
    """Tests for nearest_neighbor_1d."""

    def test_identity_same_grid(self):
        x = jnp.linspace(0, 1, 10)
        transform = nearest_neighbor_1d(x, x)
        values = jnp.sin(x)
        result = transform(values)
        assert jnp.allclose(result, values)

    def test_coarsening(self):
        """Fine grid to coarse grid picks nearest values."""
        source_x = jnp.linspace(0, 1, 20)
        target_x = jnp.linspace(0, 1, 5)
        transform = nearest_neighbor_1d(source_x, target_x)
        values = jnp.arange(20, dtype=jnp.float32)
        result = transform(values)
        assert result.shape == (5,)
        assert jnp.all(jnp.isfinite(result))

    def test_refinement(self):
        """Coarse grid to fine grid."""
        source_x = jnp.linspace(0, 1, 5)
        target_x = jnp.linspace(0, 1, 20)
        transform = nearest_neighbor_1d(source_x, target_x)
        values = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = transform(values)
        assert result.shape == (20,)
        # End points should match
        assert float(result[0]) == pytest.approx(0.0)
        assert float(result[-1]) == pytest.approx(4.0)


# ==================================================================
# Linear interpolation 1D
# ==================================================================

class TestLinearInterpolation1D:
    """Tests for linear_interpolation_1d."""

    def test_exact_at_nodes(self):
        """Interpolation is exact at source nodes."""
        source_x = jnp.linspace(0, 1, 10)
        transform = linear_interpolation_1d(source_x, source_x)
        values = jnp.sin(source_x)
        result = transform(values)
        assert jnp.allclose(result, values, atol=1e-6)

    def test_midpoint_is_average(self):
        """Midpoint between two nodes is their average."""
        source_x = jnp.array([0.0, 1.0])
        target_x = jnp.array([0.5])
        transform = linear_interpolation_1d(source_x, target_x)
        values = jnp.array([2.0, 4.0])
        result = transform(values)
        assert float(result[0]) == pytest.approx(3.0)

    def test_preserves_linear_function(self):
        """A linear function is exactly reproduced."""
        source_x = jnp.linspace(0, 1, 5)
        target_x = jnp.linspace(0, 1, 20)
        transform = linear_interpolation_1d(source_x, target_x)
        # f(x) = 2x + 1
        values = 2.0 * source_x + 1.0
        result = transform(values)
        expected = 2.0 * target_x + 1.0
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_shape_correct(self):
        source_x = jnp.linspace(0, 1, 10)
        target_x = jnp.linspace(0, 1, 50)
        transform = linear_interpolation_1d(source_x, target_x)
        values = jnp.ones(10)
        result = transform(values)
        assert result.shape == (50,)


# ==================================================================
# RBF interpolation
# ==================================================================

class TestRBFInterpolation:
    """Tests for rbf_interpolation."""

    def test_exact_at_source_points(self):
        """RBF interpolation is exact at source points."""
        source = jnp.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
        values = jnp.sin(source[:, 0] * jnp.pi)
        transform = rbf_interpolation(source, source, epsilon=2.0,
                                       kernel="gaussian")
        result = transform(values)
        assert jnp.allclose(result, values, atol=1e-4)

    def test_smooth_interpolation(self):
        """RBF produces smooth, finite interpolation."""
        source = jnp.linspace(0, 1, 5).reshape(-1, 1)
        target = jnp.linspace(0, 1, 20).reshape(-1, 1)
        values = jnp.sin(source[:, 0] * jnp.pi)
        transform = rbf_interpolation(source, target, epsilon=2.0,
                                       kernel="gaussian")
        result = transform(values)
        assert result.shape == (20,)
        assert jnp.all(jnp.isfinite(result))

    def test_multiquadric_kernel(self):
        source = jnp.linspace(0, 1, 5).reshape(-1, 1)
        target = jnp.linspace(0, 1, 10).reshape(-1, 1)
        values = jnp.ones(5)
        transform = rbf_interpolation(source, target, epsilon=1.0,
                                       kernel="multiquadric")
        result = transform(values)
        assert jnp.all(jnp.isfinite(result))
        # Constant function should map to constant
        assert jnp.allclose(result, 1.0, atol=0.1)

    def test_thin_plate_spline_kernel(self):
        source = jnp.linspace(0, 1, 5).reshape(-1, 1)
        target = jnp.linspace(0, 1, 10).reshape(-1, 1)
        values = source[:, 0]  # linear function
        transform = rbf_interpolation(source, target, epsilon=1.0,
                                       kernel="thin_plate_spline")
        result = transform(values)
        assert jnp.all(jnp.isfinite(result))

    def test_2d_points(self):
        """RBF works with 2D point coordinates."""
        source = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        target = jnp.array([[0.5, 0.5]])
        values = jnp.array([1.0, 2.0, 3.0, 4.0])
        transform = rbf_interpolation(source, target, epsilon=1.0,
                                       kernel="gaussian")
        result = transform(values)
        assert result.shape == (1,)
        assert jnp.isfinite(result[0])
        # Center value should be roughly the average
        assert 1.0 < float(result[0]) < 4.0


# ==================================================================
# Conservative projection 1D
# ==================================================================

class TestConservativeProjection1D:
    """Tests for conservative_projection_1d."""

    def test_preserves_integral(self):
        """Integral of projected field equals integral of source."""
        source_bounds = jnp.linspace(0, 1, 11)  # 10 cells
        target_bounds = jnp.linspace(0, 1, 6)   # 5 cells
        transform = conservative_projection_1d(source_bounds, target_bounds)

        values = jnp.sin(jnp.linspace(0.05, 0.95, 10) * jnp.pi)
        result = transform(values)
        assert result.shape == (5,)

        # Integral = sum(value * cell_width)
        src_widths = source_bounds[1:] - source_bounds[:-1]
        tgt_widths = target_bounds[1:] - target_bounds[:-1]
        src_integral = jnp.sum(values * src_widths)
        tgt_integral = jnp.sum(result * tgt_widths)
        assert float(src_integral) == pytest.approx(float(tgt_integral), abs=1e-5)

    def test_constant_field(self):
        """Constant field maps to constant field."""
        source_bounds = jnp.linspace(0, 1, 11)
        target_bounds = jnp.linspace(0, 1, 6)
        transform = conservative_projection_1d(source_bounds, target_bounds)
        values = jnp.ones(10) * 5.0
        result = transform(values)
        assert jnp.allclose(result, 5.0, atol=1e-5)

    def test_identity_same_grid(self):
        """Same grid produces identity mapping."""
        bounds = jnp.linspace(0, 1, 6)
        transform = conservative_projection_1d(bounds, bounds)
        values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = transform(values)
        assert jnp.allclose(result, values, atol=1e-5)


# ==================================================================
# Integration with EdgeSpec and graph
# ==================================================================

class TestEdgeSpecIntegration:
    """Tests that interpolation maps work as EdgeSpec transforms."""

    def test_transform_in_edge_spec(self):
        """Interpolation function can be used as EdgeSpec.transform."""
        source_x = jnp.linspace(0, 1, 10)
        target_x = jnp.linspace(0, 1, 20)
        interp = linear_interpolation_1d(source_x, target_x)
        edge = EdgeSpec("a", "b", "T", "T_in", transform=interp)
        assert edge.transform is not None

    def test_jit_compatible(self):
        """Transform function JIT-compiles."""
        source_x = jnp.linspace(0, 1, 10)
        target_x = jnp.linspace(0, 1, 20)
        interp = linear_interpolation_1d(source_x, target_x)
        jitted = jax.jit(interp)
        values = jnp.sin(source_x)
        result = jitted(values)
        assert result.shape == (20,)
        assert jnp.all(jnp.isfinite(result))

    def test_grad_compatible(self):
        """Gradient through transform function."""
        source_x = jnp.linspace(0, 1, 5)
        target_x = jnp.linspace(0, 1, 10)
        interp = linear_interpolation_1d(source_x, target_x)

        def loss_fn(values):
            return jnp.sum(interp(values) ** 2)

        g = jax.grad(loss_fn)(jnp.ones(5))
        assert g.shape == (5,)
        assert jnp.all(jnp.isfinite(g))

    def test_coupled_heat_nodes_different_resolution(self):
        """End-to-end test: two HeatNodes at different resolutions."""
        gm = GraphManager()
        dt = 0.001
        a = HeatNode(name="coarse", timestep=dt, n_cells=5,
                     thermal_diffusivity=0.01, length=1.0,
                     initial_temperature=100.0)
        b = HeatNode(name="fine", timestep=dt, n_cells=20,
                     thermal_diffusivity=0.01, length=1.0,
                     initial_temperature=0.0)
        gm.add_node(a)
        gm.add_node(b)

        # Create interpolation maps
        coarse_x = jnp.linspace(0, 1, 5)
        fine_x = jnp.linspace(0, 1, 20)
        coarse_to_fine = linear_interpolation_1d(coarse_x, fine_x)
        fine_to_coarse = linear_interpolation_1d(fine_x, coarse_x)

        # Coarse rightmost -> fine left BC (scalar extraction after interp)
        gm.add_edge("coarse", "fine", "temperature", "left_temperature",
                    transform=lambda T: T[-1])
        # Fine leftmost -> coarse right BC
        gm.add_edge("fine", "coarse", "temperature", "right_temperature",
                    transform=lambda T: T[0])

        gm.add_coupling_group(["coarse", "fine"],
                               max_iterations=10, tolerance=1e-8)
        gm.compile()
        state = gm.run_scan(200)
        assert jnp.all(jnp.isfinite(state["coarse"]["temperature"]))
        assert jnp.all(jnp.isfinite(state["fine"]["temperature"]))

    def test_rbf_grad_compatible(self):
        """Gradient through RBF transform."""
        source = jnp.linspace(0, 1, 5).reshape(-1, 1)
        target = jnp.linspace(0, 1, 10).reshape(-1, 1)
        interp = rbf_interpolation(source, target, epsilon=2.0,
                                    kernel="gaussian")

        def loss_fn(values):
            return jnp.sum(interp(values) ** 2)

        g = jax.grad(loss_fn)(jnp.ones(5))
        assert g.shape == (5,)
        assert jnp.all(jnp.isfinite(g))
