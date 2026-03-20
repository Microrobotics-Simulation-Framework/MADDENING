"""
Tests for spatial accuracy enhancements (Phase 8).

8a. Higher-order FD stencils (stencil_order=4)
8b. Non-uniform grids (grid_points parameter)
8c. Conservative 2D mapping (nearest_neighbor_2d, rbf_interpolation_2d)
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.nodes.heat import HeatNode
from maddening.core.graph_manager import GraphManager
from maddening.core.coupling.interface_mapping import (
    nearest_neighbor_2d,
    rbf_interpolation_2d,
)


# ======================================================================
# Phase 8a: Higher-order FD stencils
# ======================================================================

class TestHeatNodeStencilOrder:
    """Test 4th-order stencil support in HeatNode."""

    def test_stencil_order_2_default(self):
        node = HeatNode("rod", 0.001, n_cells=10)
        assert node.params["stencil_order"] == 2

    def test_stencil_order_4_creation(self):
        node = HeatNode("rod", 0.001, n_cells=10, stencil_order=4)
        assert node.params["stencil_order"] == 4

    def test_stencil_order_4_requires_5_cells(self):
        with pytest.raises(ValueError, match="at least 5 cells"):
            HeatNode("rod", 0.001, n_cells=4, stencil_order=4)

    def test_invalid_stencil_order_raises(self):
        with pytest.raises(ValueError, match="stencil_order must be 2 or 4"):
            HeatNode("rod", 0.001, n_cells=10, stencil_order=3)

    def test_stencil_order_4_runs(self):
        """4th-order node can update without errors."""
        node = HeatNode(
            "rod", 0.001, n_cells=20, length=1.0,
            thermal_diffusivity=0.01, stencil_order=4,
            initial_temperature=0.0,
        )
        state = node.initial_state()
        bi = {"left_temperature": jnp.float32(100.0),
              "right_temperature": jnp.float32(0.0)}
        new_state = node.update(state, bi, 0.001)
        assert new_state["temperature"].shape == (20,)
        assert jnp.isfinite(new_state["temperature"]).all()

    def test_4th_order_more_accurate_than_2nd(self):
        """4th-order stencil should converge faster for smooth solutions.

        Use a quadratic temperature profile T(x) = x^2 for which the
        exact Laplacian is 2.  The 4th-order stencil should be exact
        for polynomials up to degree 4.
        """
        n = 20
        L = 1.0
        dx = L / n
        x = jnp.linspace(dx / 2, L - dx / 2, n)

        # Quadratic profile: T(x) = x^2
        T = x ** 2
        T_left = jnp.float32(0.0)   # (0)^2
        T_right = jnp.float32(1.0)  # (1)^2

        # Exact Laplacian of x^2 is 2
        exact_laplacian = 2.0

        # 2nd order
        node2 = HeatNode("rod2", 0.001, n_cells=n, length=L, stencil_order=2)
        lap2 = node2._compute_laplacian(T, T_left, T_right)

        # 4th order
        node4 = HeatNode("rod4", 0.001, n_cells=n, length=L, stencil_order=4)
        lap4 = node4._compute_laplacian(T, T_left, T_right)

        # Interior cells (skip boundary cells)
        interior = slice(2, n - 2)
        err2 = float(jnp.abs(lap2[interior] - exact_laplacian).max())
        err4 = float(jnp.abs(lap4[interior] - exact_laplacian).max())

        # 4th-order should be much more accurate for interior cells
        # For x^2, the 4th-order stencil should be exact (up to float precision)
        assert err4 < err2 or err4 < 1e-4

    def test_4th_order_in_graph(self):
        """4th-order node works in a compiled graph."""
        gm = GraphManager()
        gm.add_node(HeatNode(
            "rod", 0.001, n_cells=20, length=1.0,
            thermal_diffusivity=0.01, stencil_order=4,
        ))
        gm.compile()
        state = gm.step()
        assert "rod" in state

    def test_4th_order_derivatives(self):
        """derivatives() uses the same stencil as update()."""
        node = HeatNode(
            "rod", 0.001, n_cells=10, length=1.0,
            thermal_diffusivity=0.01, stencil_order=4,
        )
        state = {"temperature": jnp.linspace(0.0, 100.0, 10)}
        bi = {"left_temperature": jnp.float32(0.0),
              "right_temperature": jnp.float32(100.0)}
        derivs = node.derivatives(state, bi)
        assert "temperature" in derivs
        assert derivs["temperature"].shape == (10,)
        assert jnp.isfinite(derivs["temperature"]).all()


# ======================================================================
# Phase 8b: Non-uniform grids
# ======================================================================

class TestHeatNodeNonUniformGrid:
    """Test non-uniform grid support in HeatNode."""

    def test_create_with_grid_points(self):
        x = np.linspace(0.0, 1.0, 10)
        node = HeatNode("rod", 0.001, n_cells=10, grid_points=x)
        assert node.params["grid_points"] is not None
        assert len(node.params["grid_points"]) == 10

    def test_grid_points_length_mismatch_raises(self):
        x = np.linspace(0.0, 1.0, 8)
        with pytest.raises(ValueError, match="grid_points length"):
            HeatNode("rod", 0.001, n_cells=10, grid_points=x)

    def test_nonuniform_update_runs(self):
        """Non-uniform grid node can update without errors."""
        # Cosine-clustered grid (fine at boundaries)
        n = 15
        theta = np.linspace(0, np.pi, n)
        x = 0.5 * (1.0 - np.cos(theta))  # [0, 1] clustered at ends

        node = HeatNode(
            "rod", 0.0001, n_cells=n, grid_points=x,
            thermal_diffusivity=0.01, initial_temperature=0.0,
        )
        state = node.initial_state()
        bi = {"left_temperature": jnp.float32(100.0),
              "right_temperature": jnp.float32(0.0)}
        new_state = node.update(state, bi, 0.0001)
        assert new_state["temperature"].shape == (n,)
        assert jnp.isfinite(new_state["temperature"]).all()

    def test_nonuniform_diffusion_direction(self):
        """Heat flows from hot to cold boundary on non-uniform grid."""
        n = 15
        x = np.linspace(0.0, 1.0, n)
        node = HeatNode(
            "rod", 0.0001, n_cells=n, grid_points=x,
            thermal_diffusivity=0.05, initial_temperature=50.0,
        )
        state = node.initial_state()
        bi = {"left_temperature": jnp.float32(100.0),
              "right_temperature": jnp.float32(0.0)}

        # Run a few steps
        for _ in range(10):
            state = node.update(state, bi, 0.0001)

        T = state["temperature"]
        # Left side should be hotter than right
        assert float(T[1]) > float(T[-2])

    def test_nonuniform_boundary_fluxes(self):
        """Boundary fluxes computed with non-uniform dx."""
        n = 10
        # Non-uniform grid: clustered at left
        x = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.7, 0.85, 1.0])
        node = HeatNode(
            "rod", 0.001, n_cells=n, grid_points=x,
            thermal_diffusivity=0.01,
        )
        state = {"temperature": jnp.linspace(0.0, 100.0, n)}
        bi = {}
        fluxes = node.compute_boundary_fluxes(state, bi, 0.001)
        assert "left_heat_flux" in fluxes
        assert "right_heat_flux" in fluxes
        assert jnp.isfinite(jnp.array(fluxes["left_heat_flux"]))
        assert jnp.isfinite(jnp.array(fluxes["right_heat_flux"]))

    def test_nonuniform_in_graph(self):
        """Non-uniform grid node works in a compiled graph."""
        x = np.linspace(0.0, 1.0, 10)
        gm = GraphManager()
        gm.add_node(HeatNode(
            "rod", 0.001, n_cells=10, grid_points=x,
            thermal_diffusivity=0.01,
        ))
        gm.compile()
        state = gm.step()
        assert "rod" in state

    def test_uniform_and_nonuniform_agree_for_uniform_grid(self):
        """Uniform grid with grid_points should give similar result as without.

        The two paths are not numerically identical because the ghost cell
        extrapolation differs slightly, but for a uniform grid they should
        agree to within a few percent for interior cells.
        """
        n = 20
        L = 1.0
        dx = L / n
        x = np.linspace(dx / 2, L - dx / 2, n)

        node_u = HeatNode("u", 0.001, n_cells=n, length=L,
                           thermal_diffusivity=0.01)
        node_nu = HeatNode("nu", 0.001, n_cells=n, grid_points=x,
                            thermal_diffusivity=0.01)

        # Use a smooth profile where the Laplacian is non-trivial
        T = jnp.sin(jnp.linspace(0.0, jnp.pi, n))
        T_left = jnp.float32(0.0)
        T_right = jnp.float32(0.0)

        lap_u = node_u._compute_laplacian(T, T_left, T_right)
        lap_nu = node_nu._compute_laplacian(T, T_left, T_right)

        # Interior cells (skip boundary cells)
        interior = slice(2, n - 2)
        np.testing.assert_allclose(
            np.asarray(lap_u[interior]),
            np.asarray(lap_nu[interior]),
            rtol=0.01,
            atol=1e-3,
        )

    def test_nonuniform_derivatives(self):
        """derivatives() works with non-uniform grid."""
        x = np.linspace(0.0, 1.0, 8)
        node = HeatNode(
            "rod", 0.001, n_cells=8, grid_points=x,
            thermal_diffusivity=0.01,
        )
        state = {"temperature": jnp.linspace(0.0, 100.0, 8)}
        bi = {"left_temperature": jnp.float32(0.0),
              "right_temperature": jnp.float32(100.0)}
        derivs = node.derivatives(state, bi)
        assert jnp.isfinite(derivs["temperature"]).all()

    def test_nonuniform_interface_correction(self):
        """Interface correction works with non-uniform grid."""
        x = np.linspace(0.0, 1.0, 8)
        node = HeatNode(
            "rod", 0.001, n_cells=8, grid_points=x,
            thermal_diffusivity=0.01,
        )
        state = {"temperature": jnp.linspace(0.0, 100.0, 8)}
        bi = {"left_temperature": jnp.float32(10.0),
              "right_temperature": jnp.float32(90.0)}
        corrections = node.compute_interface_correction(state, bi, 0.001)
        assert "temperature" in corrections
        for idx, val in corrections["temperature"]:
            assert jnp.isfinite(val)


# ======================================================================
# Phase 8c: 2D mapping functions
# ======================================================================

class TestNearestNeighbor2D:
    """Test nearest_neighbor_2d interpolation."""

    def test_identity_mapping(self):
        """Same source and target points: identity."""
        pts = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        transform = nearest_neighbor_2d(pts, pts)
        vals = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = transform(vals)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(vals))

    def test_simple_mapping(self):
        """Target points map to nearest source points."""
        src = jnp.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        tgt = jnp.array([[0.1, 0.1], [9.0, 0.5], [0.5, 9.0]])
        transform = nearest_neighbor_2d(src, tgt)
        vals = jnp.array([100.0, 200.0, 300.0])
        result = transform(vals)
        # [0.1,0.1] -> nearest src[0], [9,0.5] -> nearest src[1], etc.
        expected = jnp.array([100.0, 200.0, 300.0])
        np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))

    def test_different_grid_sizes(self):
        """Source and target grids can have different sizes."""
        src = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        tgt = jnp.array([[0.1, 0.1], [0.9, 0.9], [0.5, 0.5], [0.3, 0.3]])
        transform = nearest_neighbor_2d(src, tgt)
        vals = jnp.array([10.0, 20.0])
        result = transform(vals)
        assert result.shape == (4,)


class TestRBFInterpolation2D:
    """Test rbf_interpolation_2d."""

    def test_identity_mapping(self):
        """Same source and target: identity (approximately)."""
        pts = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        transform = rbf_interpolation_2d(pts, pts, epsilon=1.0)
        vals = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = transform(vals)
        np.testing.assert_allclose(
            np.asarray(result), np.asarray(vals), atol=0.1
        )

    def test_different_kernels(self):
        """All kernel types work for 2D."""
        src = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        tgt = jnp.array([[0.5, 0.5]])
        vals = jnp.array([1.0, 2.0, 3.0])

        for kernel in [
            "gaussian",
            "multiquadric",
            "inverse_multiquadric",
            "thin_plate_spline",
        ]:
            transform = rbf_interpolation_2d(
                src, tgt, epsilon=1.0, kernel=kernel
            )
            result = transform(vals)
            assert result.shape == (1,)
            assert jnp.isfinite(result).all(), f"kernel={kernel}"

    def test_smooth_interpolation(self):
        """RBF produces smooth interpolation for 2D data."""
        # Source: regular 3x3 grid
        src_x = jnp.array([0.0, 0.5, 1.0])
        src_y = jnp.array([0.0, 0.5, 1.0])
        src = jnp.array(
            [[x, y] for x in src_x for y in src_y]
        )
        vals = jnp.array([float(x + y) for x in src_x for y in src_y])

        # Target: single interior point
        tgt = jnp.array([[0.25, 0.25]])
        transform = rbf_interpolation_2d(src, tgt, epsilon=2.0)
        result = transform(vals)

        # For f(x,y) = x+y, at (0.25, 0.25) expect ~0.5
        assert abs(float(result[0]) - 0.5) < 0.3

    def test_multichannel(self):
        """RBF works with multi-channel (N_src, C) values."""
        src = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        tgt = jnp.array([[0.5, 0.5]])
        vals = jnp.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        transform = rbf_interpolation_2d(src, tgt, epsilon=1.0)
        result = transform(vals)
        assert result.shape == (1, 2)
