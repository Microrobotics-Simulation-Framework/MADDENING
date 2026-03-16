"""Tests for the general LBMNode and lattice descriptors."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.nodes.lbm import (
    LBMNode,
    LatticeDescriptor,
    d3q19,
    d2q9,
    _equilibrium,
    _compute_macroscopic,
    _stream,
    _guo_forcing,
    _get_opp_map,
)
from maddening.nodes.lbm_geometry import voxelize_vessel
from maddening.core.graph_manager import GraphManager


# ═══════════════════════════════════════════════════════════════════════
# 1. Lattice descriptor tests
# ═══════════════════════════════════════════════════════════════════════

class TestLatticeD3Q19:
    def test_weights_sum_to_one(self):
        lat = d3q19()
        np.testing.assert_allclose(np.sum(lat.w), 1.0, rtol=1e-12)

    def test_opposite_indices_correct(self):
        """opp[opp[q]] == q for all q, and e[opp[q]] == -e[q]."""
        lat = d3q19()
        for q in range(lat.Q):
            assert int(lat.opp[lat.opp[q]]) == q
            np.testing.assert_array_equal(lat.e[lat.opp[q]], -lat.e[q])

    def test_zero_velocity_weight(self):
        lat = d3q19()
        # Direction 0 is the rest direction
        np.testing.assert_array_equal(lat.e[0], [0, 0, 0])
        np.testing.assert_allclose(lat.w[0], 1.0 / 3.0, rtol=1e-12)

    def test_dimensions(self):
        lat = d3q19()
        assert lat.D == 3
        assert lat.Q == 19
        assert lat.e.shape == (19, 3)
        assert lat.w.shape == (19,)
        assert lat.opp.shape == (19,)

    def test_cs2(self):
        lat = d3q19()
        np.testing.assert_allclose(lat.cs2, 1.0 / 3.0, rtol=1e-12)

    def test_arrays_are_numpy(self):
        """Lattice constants must be numpy, not jnp (lesson from LBMPipeNode)."""
        lat = d3q19()
        assert isinstance(lat.e, np.ndarray)
        assert isinstance(lat.w, np.ndarray)
        assert isinstance(lat.opp, np.ndarray)


class TestLatticeD2Q9:
    def test_weights_sum_to_one(self):
        lat = d2q9()
        np.testing.assert_allclose(np.sum(lat.w), 1.0, rtol=1e-12)

    def test_opposite_indices_correct(self):
        lat = d2q9()
        for q in range(lat.Q):
            assert int(lat.opp[lat.opp[q]]) == q
            np.testing.assert_array_equal(lat.e[lat.opp[q]], -lat.e[q])

    def test_zero_velocity_weight(self):
        lat = d2q9()
        np.testing.assert_array_equal(lat.e[0], [0, 0])
        np.testing.assert_allclose(lat.w[0], 4.0 / 9.0, rtol=1e-12)

    def test_dimensions(self):
        lat = d2q9()
        assert lat.D == 2
        assert lat.Q == 9
        assert lat.e.shape == (9, 2)
        assert lat.w.shape == (9,)
        assert lat.opp.shape == (9,)

    def test_arrays_are_numpy(self):
        lat = d2q9()
        assert isinstance(lat.e, np.ndarray)
        assert isinstance(lat.w, np.ndarray)
        assert isinstance(lat.opp, np.ndarray)


# ═══════════════════════════════════════════════════════════════════════
# 2. LBMNode construction tests
# ═══════════════════════════════════════════════════════════════════════

class TestLBMConstructs:
    def test_default_3d(self):
        """Basic construction with default params (D3Q19)."""
        node = LBMNode(name="lbm", timestep=0.001, grid_shape=(8, 8, 8))
        assert node.name == "lbm"
        assert node._D == 3
        assert node._Q == 19
        assert node.tau > 0.5

    def test_2d(self):
        """Construction with D2Q9 lattice."""
        node = LBMNode(
            name="lbm2d", timestep=0.001,
            grid_shape=(16, 8), lattice="D2Q9",
        )
        assert node._D == 2
        assert node._Q == 9

    def test_invalid_lattice(self):
        with pytest.raises(ValueError, match="Unknown lattice"):
            LBMNode(name="x", timestep=0.001, grid_shape=(8, 8, 8),
                    lattice="D1Q3")

    def test_grid_shape_mismatch(self):
        """2D grid_shape with 3D lattice raises ValueError."""
        with pytest.raises(ValueError, match="dims"):
            LBMNode(name="x", timestep=0.001, grid_shape=(8, 8),
                    lattice="D3Q19")

    def test_viscosity_sets_tau(self):
        node = LBMNode(name="lbm", timestep=0.001,
                       grid_shape=(8, 8, 8), viscosity=0.15)
        expected_tau = 0.5 + 0.15 / (1.0 / 3.0)
        np.testing.assert_allclose(node.tau, expected_tau, rtol=1e-10)

    def test_wall_mask_shape_validation(self):
        """wall_mask must match grid_shape."""
        with pytest.raises(ValueError, match="wall_mask shape"):
            LBMNode(name="x", timestep=0.001, grid_shape=(8, 8, 8),
                    wall_mask=np.zeros((4, 4, 4), dtype=bool))


# ═══════════════════════════════════════════════════════════════════════
# 3. Initial state tests
# ═══════════════════════════════════════════════════════════════════════

class TestLBMInitialState:
    def test_state_shapes_3d(self):
        node = LBMNode(name="lbm", timestep=0.001, grid_shape=(8, 6, 4))
        state = node.initial_state()
        assert state["f"].shape == (8, 6, 4, 19)
        assert state["density"].shape == (8, 6, 4)
        assert state["velocity"].shape == (8, 6, 4, 3)
        assert state["pressure"].shape == (8, 6, 4)

    def test_state_shapes_2d(self):
        node = LBMNode(name="lbm2d", timestep=0.001,
                       grid_shape=(16, 8), lattice="D2Q9")
        state = node.initial_state()
        assert state["f"].shape == (16, 8, 9)
        assert state["density"].shape == (16, 8)
        assert state["velocity"].shape == (16, 8, 2)
        assert state["pressure"].shape == (16, 8)

    def test_initial_density_is_one(self):
        node = LBMNode(name="lbm", timestep=0.001, grid_shape=(8, 8, 8))
        state = node.initial_state()
        np.testing.assert_allclose(state["density"], 1.0, rtol=1e-5)

    def test_initial_velocity_is_zero(self):
        node = LBMNode(name="lbm", timestep=0.001, grid_shape=(8, 8, 8))
        state = node.initial_state()
        np.testing.assert_allclose(state["velocity"], 0.0, atol=1e-10)

    def test_initial_pressure(self):
        node = LBMNode(name="lbm", timestep=0.001, grid_shape=(8, 8, 8))
        state = node.initial_state()
        np.testing.assert_allclose(
            state["pressure"], 1.0 / 3.0, rtol=1e-5,
        )

    def test_distributions_sum_to_density(self):
        node = LBMNode(name="lbm", timestep=0.001, grid_shape=(8, 8, 8))
        state = node.initial_state()
        rho = jnp.sum(state["f"], axis=-1)
        np.testing.assert_allclose(rho, state["density"], rtol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# 4. Single step produces finite values
# ═══════════════════════════════════════════════════════════════════════

class TestLBMStepFinite:
    def test_step_produces_finite_3d(self):
        node = LBMNode(name="lbm", timestep=0.001, grid_shape=(8, 8, 8))
        state = node.initial_state()
        new_state = node.update(state, {}, node.delta_t)
        for key in ("f", "density", "velocity", "pressure"):
            assert jnp.all(jnp.isfinite(new_state[key])), (
                f"{key} has non-finite values after one step"
            )

    def test_step_produces_finite_2d(self):
        node = LBMNode(
            name="lbm2d", timestep=0.001,
            grid_shape=(16, 8), lattice="D2Q9",
        )
        state = node.initial_state()
        new_state = node.update(state, {}, node.delta_t)
        for key in ("f", "density", "velocity", "pressure"):
            assert jnp.all(jnp.isfinite(new_state[key])), (
                f"{key} has non-finite values after one step"
            )


# ═══════════════════════════════════════════════════════════════════════
# 5. Equilibrium stability
# ═══════════════════════════════════════════════════════════════════════

class TestLBMEquilibriumStable:
    def test_uniform_density_zero_velocity_stays_stable(self):
        """Uniform density + zero velocity should remain stable over many steps."""
        node = LBMNode(name="lbm", timestep=0.001, grid_shape=(8, 8, 8))
        state = node.initial_state()
        for _ in range(100):
            state = node.update(state, {}, node.delta_t)
        np.testing.assert_allclose(state["density"], 1.0, rtol=1e-3)
        np.testing.assert_allclose(state["velocity"], 0.0, atol=1e-4)

    def test_2d_equilibrium_stable(self):
        node = LBMNode(
            name="lbm2d", timestep=0.001,
            grid_shape=(16, 16), lattice="D2Q9",
        )
        state = node.initial_state()
        for _ in range(100):
            state = node.update(state, {}, node.delta_t)
        np.testing.assert_allclose(state["density"], 1.0, rtol=1e-3)
        np.testing.assert_allclose(state["velocity"], 0.0, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════
# 6. 2D Poiseuille flow verification
# ═══════════════════════════════════════════════════════════════════════

class TestLBMPoiseuille2D:
    """Verify Poiseuille flow in a 2D channel with pressure BCs.

    Channel: nx=64, ny=16 with walls at y=0 and y=ny-1.
    Pressure inlet at x_min, pressure outlet at x_max.
    After sufficient steps, the velocity profile should approach parabolic.
    """

    def test_poiseuille_profile(self):
        nx, ny = 64, 16

        # Wall mask: walls at top and bottom rows
        wall_mask = np.zeros((nx, ny), dtype=bool)
        wall_mask[:, 0] = True
        wall_mask[:, ny - 1] = True

        viscosity = 0.1
        node = LBMNode(
            name="poiseuille",
            timestep=0.001,
            grid_shape=(nx, ny),
            viscosity=viscosity,
            lattice="D2Q9",
            wall_mask=wall_mask,
            inlet_face="x_min",
            outlet_face="x_max",
        )

        state = node.initial_state()

        # Pressure BCs: small pressure gradient
        p_in = 1.0 / 3.0 * 1.01   # rho_in ~ 1.01
        p_out = 1.0 / 3.0 * 1.00  # rho_out ~ 1.00

        boundary_inputs = {
            "inlet_pressure": jnp.float32(p_in),
            "outlet_pressure": jnp.float32(p_out),
        }

        # Run for 2000 steps
        for _ in range(2000):
            state = node.update(state, boundary_inputs, node.delta_t)

        # Check finiteness
        assert jnp.all(jnp.isfinite(state["velocity"]))

        # Extract velocity profile at mid-channel (x = nx//2)
        u_profile = np.array(state["velocity"][nx // 2, :, 0])  # x-velocity

        # Walls should have zero velocity
        np.testing.assert_allclose(u_profile[0], 0.0, atol=1e-6)
        np.testing.assert_allclose(u_profile[ny - 1], 0.0, atol=1e-6)

        # Interior fluid should flow (positive x-velocity)
        u_interior = u_profile[1:ny - 1]
        assert np.all(u_interior > 0), (
            f"Expected positive flow, got min={u_interior.min():.6f}"
        )

        # Check parabolic shape: peak should be near the centre
        peak_y = np.argmax(u_interior) + 1  # offset for wall
        expected_peak = ny // 2
        assert abs(peak_y - expected_peak) <= 2, (
            f"Velocity peak at y={peak_y}, expected near y={expected_peak}"
        )

        # Analytical Poiseuille: u(y) = dp/dx / (2*mu) * y * (H-y)
        # where H = channel height (in lattice units), mu = viscosity
        H = ny - 2  # fluid cells between walls
        dp_dx = (1.01 - 1.0) / nx  # density gradient (pressure = rho*cs2)
        y_fluid = np.arange(1, ny - 1) - 0.5  # cell centres relative to wall
        u_analytical = dp_dx / (2.0 * viscosity) * y_fluid * (H - y_fluid)

        # Normalise both profiles for shape comparison
        u_sim_norm = u_interior / np.max(u_interior)
        u_ana_norm = u_analytical / np.max(u_analytical)

        # Allow 10% error in the shape (LBM needs many steps to fully develop)
        np.testing.assert_allclose(
            u_sim_norm, u_ana_norm, atol=0.15,
            err_msg="Poiseuille profile shape does not match analytical",
        )


# ═══════════════════════════════════════════════════════════════════════
# 7. Mass conservation
# ═══════════════════════════════════════════════════════════════════════

class TestLBMMassConservation:
    def test_mass_conserved_periodic(self):
        """Total mass should be conserved for a system with no BCs.

        No wall mask, no pressure BCs -> purely periodic, mass conserved.
        """
        node = LBMNode(
            name="lbm", timestep=0.001,
            grid_shape=(16, 16, 16), viscosity=0.1,
        )
        state = node.initial_state()
        mass_initial = float(jnp.sum(state["density"]))

        # Run 50 steps with no boundary inputs
        for _ in range(50):
            state = node.update(state, {}, node.delta_t)

        mass_final = float(jnp.sum(state["density"]))
        np.testing.assert_allclose(
            mass_final, mass_initial, rtol=1e-5,
            err_msg="Mass not conserved in periodic system",
        )


# ═══════════════════════════════════════════════════════════════════════
# 8. Wall mask test
# ═══════════════════════════════════════════════════════════════════════

class TestLBMWallMask:
    def test_wall_cells_have_zero_velocity(self):
        """Cells marked as wall should have zero velocity after stepping."""
        wall = np.zeros((8, 8, 8), dtype=bool)
        wall[3:5, :, :] = True  # wall slab in the middle

        node = LBMNode(
            name="lbm", timestep=0.001,
            grid_shape=(8, 8, 8), wall_mask=wall,
        )
        state = node.initial_state()

        # Apply some forcing to create flow
        force = jnp.zeros((8, 8, 8, 3), dtype=jnp.float32)
        force = force.at[:, :, :, 0].set(0.001)

        for _ in range(20):
            state = node.update(
                state, {"body_force": force}, node.delta_t,
            )

        # Wall cells should have zero velocity
        wall_vel = state["velocity"][3:5, :, :]
        np.testing.assert_allclose(wall_vel, 0.0, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════
# 9. Graph manager integration
# ═══════════════════════════════════════════════════════════════════════

class TestLBMWithGraphManager:
    def test_lbm_in_graph(self):
        """LBMNode should work inside a GraphManager."""
        node = LBMNode(
            name="lbm", timestep=0.01,
            grid_shape=(8, 8, 8), viscosity=0.1,
        )
        gm = GraphManager()
        gm.add_node(node)
        gm.compile()

        state = gm.get_node_state("lbm")
        assert "f" in state
        assert state["f"].shape == (8, 8, 8, 19)

        # Run a few steps
        gm.run(5)
        state_after = gm.get_node_state("lbm")
        assert jnp.all(jnp.isfinite(state_after["density"]))


# ═══════════════════════════════════════════════════════════════════════
# 10. Clot injection (wall_mask_update)
# ═══════════════════════════════════════════════════════════════════════

class TestLBMClotInjection:
    def test_wall_mask_update_changes_domain(self):
        """Providing wall_mask_update should change which cells are walls."""
        node = LBMNode(
            name="lbm", timestep=0.001,
            grid_shape=(16, 8, 8), viscosity=0.1,
        )
        state = node.initial_state()

        # Apply body force to create some flow
        force = jnp.zeros((16, 8, 8, 3), dtype=jnp.float32)
        force = force.at[:, :, :, 0].set(0.0005)

        # Step a few times without clot
        for _ in range(10):
            state = node.update(state, {"body_force": force}, node.delta_t)

        # Velocity at the future clot site should be nonzero
        vel_before = float(jnp.mean(jnp.abs(state["velocity"][8, 3:5, 3:5, 0])))
        assert vel_before > 1e-6, "Expected nonzero velocity before clot"

        # Inject a clot (wall block at x=8, y=3:5, z=3:5)
        clot_mask = np.zeros((16, 8, 8), dtype=bool)
        clot_mask[8, 3:5, 3:5] = True

        for _ in range(10):
            state = node.update(
                state,
                {"body_force": force, "wall_mask_update": jnp.asarray(clot_mask)},
                node.delta_t,
            )

        # Velocity at the clot site should be zero
        vel_after = state["velocity"][8, 3:5, 3:5, :]
        np.testing.assert_allclose(vel_after, 0.0, atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════
# 11. Voxelizer
# ═══════════════════════════════════════════════════════════════════════

class TestVoxelizeVessel:
    def test_reasonable_fluid_fraction(self):
        """Fluid fraction should be between 5% and 50%."""
        grid_shape = (64, 32, 32)
        params = {
            "parent_radius": 6.0,
            "daughter_radius": 4.0,
            "parent_length": 30.0,
            "daughter_length": 25.0,
            "bifurcation_angle": 30.0,
        }
        mask = voxelize_vessel(grid_shape, params)
        assert mask.shape == grid_shape
        assert mask.dtype == jnp.bool_

        fluid_fraction = float(jnp.sum(~mask)) / np.prod(grid_shape)
        assert 0.05 < fluid_fraction < 0.50, (
            f"Fluid fraction {fluid_fraction:.3f} outside expected range"
        )

    def test_parent_tube_is_fluid(self):
        """Centre of the parent tube should be fluid."""
        grid_shape = (64, 32, 32)
        params = {
            "parent_radius": 6.0,
            "daughter_radius": 4.0,
            "parent_length": 30.0,
            "daughter_length": 25.0,
            "bifurcation_angle": 30.0,
        }
        mask = voxelize_vessel(grid_shape, params)
        # Centre of parent tube at x=15, y=15, z=15
        assert not bool(mask[15, 15, 15]), "Centre of parent tube should be fluid"


# ═══════════════════════════════════════════════════════════════════════
# 12. Boundary input spec / derivatives
# ═══════════════════════════════════════════════════════════════════════

class TestLBMBoundarySpec:
    def test_boundary_input_spec(self):
        node = LBMNode(name="lbm", timestep=0.001, grid_shape=(8, 8, 8))
        spec = node.boundary_input_spec()
        assert "inlet_pressure" in spec
        assert "outlet_pressure" in spec
        assert "body_force" in spec
        assert "wall_mask_update" in spec
        assert spec["body_force"].coupling_type == "additive"

    def test_derivatives_raises(self):
        node = LBMNode(name="lbm", timestep=0.001, grid_shape=(8, 8, 8))
        state = node.initial_state()
        with pytest.raises(NotImplementedError, match="discrete"):
            node.derivatives(state, {})

    def test_compute_boundary_fluxes(self):
        node = LBMNode(name="lbm", timestep=0.001, grid_shape=(8, 8, 8))
        state = node.initial_state()
        fluxes = node.compute_boundary_fluxes(state, {}, node.delta_t)
        assert "outlet_pressure_avg" in fluxes
        assert jnp.isfinite(fluxes["outlet_pressure_avg"])
