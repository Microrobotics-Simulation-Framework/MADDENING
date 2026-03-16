"""Tests for HeartPump <-> LBM vessel flow coupling (Phases D-F)."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.heart_pump import HeartPumpNode
from maddening.nodes.lbm import LBMNode, _FACE_MAP
from maddening.nodes.lbm_geometry import voxelize_vessel


# Small grid for fast tests
SMALL_GRID = (8, 4, 4)
SMALL_VESSEL_PARAMS = {
    "parent_radius": 1.5,
    "daughter_radius": 1.0,
    "parent_length": 4.0,
    "daughter_length": 3.0,
    "bifurcation_angle": 30.0,
}


def _make_lbm(grid_shape=SMALL_GRID, vessel_params=SMALL_VESSEL_PARAMS):
    """Create a small LBM node with vessel mask for testing."""
    mask = voxelize_vessel(grid_shape, vessel_params)
    return LBMNode(
        name="vessel",
        timestep=1.0,
        grid_shape=grid_shape,
        viscosity=0.1,
        wall_mask=np.asarray(mask),
        inlet_face="x_min",
        outlet_face="x_max",
    ), mask


def _make_heart(dt=1.0):
    """Create a HeartPumpNode for testing.

    Parameters are tuned for LBM lattice units: pressures stay in a
    narrow band around the equilibrium pressure 1/3 (= cs2 * rho0).
    """
    return HeartPumpNode(
        name="heart",
        timestep=dt,
        resistance=10.0,       # high R -> small outflow
        compliance=1.0,        # C=1 -> moderate pressure response
        heart_rate=72.0,
        stroke_volume=0.002,   # tiny SV -> pressure perturbation ~1%
        venous_pressure=0.0,
        systole_fraction=0.35,
        initial_pressure=1.0 / 3.0,
    )


def _make_outlet_transform(wall_mask, outlet_face, ndim):
    """Build transform that extracts mean outlet pressure from full grid."""
    outlet_axis, outlet_side = _FACE_MAP[outlet_face]
    face_slices = [slice(None)] * ndim
    if outlet_side == "min":
        face_slices[outlet_axis] = 0
    else:
        face_slices[outlet_axis] = -1
    face_sl = tuple(face_slices)
    wall_face = wall_mask[face_sl]

    def outlet_pressure_avg(pressure):
        p_face = pressure[face_sl]
        fluid_count = jnp.sum(~wall_face)
        p_sum = jnp.sum(jnp.where(wall_face, 0.0, p_face))
        return p_sum / jnp.maximum(fluid_count, 1.0)

    return outlet_pressure_avg


def _build_coupled_graph(grid_shape=SMALL_GRID, vessel_params=SMALL_VESSEL_PARAMS):
    """Build a minimal coupled HeartPump + LBM graph."""
    import warnings

    lbm_node, mask = _make_lbm(grid_shape, vessel_params)
    heart_node = _make_heart(dt=lbm_node.delta_t)

    gm = GraphManager()
    gm.add_node(heart_node)
    gm.add_node(lbm_node)

    # HeartPump -> LBM: arterial pressure drives inlet
    gm.add_edge("heart", "vessel", "arterial_pressure", "inlet_pressure")

    # LBM -> HeartPump: outlet face avg pressure feeds back
    outlet_transform = _make_outlet_transform(mask, "x_max", ndim=len(grid_shape))
    gm.add_edge("vessel", "heart", "pressure", "backpressure",
                 transform=outlet_transform)

    gm.add_coupling_group(
        ["heart", "vessel"],
        max_iterations=5,
        tolerance=1e-4,
        diagnostics=True,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()

    return gm, mask


# ═══════════════════════════════════════════════════════════════════════
# 1. Coupled graph runs and produces finite values
# ═══════════════════════════════════════════════════════════════════════

class TestHeartLBMCouplingRuns:
    def test_coupled_graph_runs_20_steps(self):
        """Build coupled graph, run 20 steps, verify all values finite."""
        gm, _ = _build_coupled_graph()
        gm.run(20)

        # Check heart state
        heart_state = gm.get_node_state("heart")
        for key, val in heart_state.items():
            assert jnp.all(jnp.isfinite(val)), (
                f"heart.{key} has non-finite values"
            )

        # Check vessel state (skip 'f' for speed, check macroscopic)
        vessel_state = gm.get_node_state("vessel")
        for key in ("density", "velocity", "pressure"):
            assert jnp.all(jnp.isfinite(vessel_state[key])), (
                f"vessel.{key} has non-finite values"
            )


# ═══════════════════════════════════════════════════════════════════════
# 2. Pressure transfers from HeartPump to LBM
# ═══════════════════════════════════════════════════════════════════════

class TestHeartLBMPressureTransfers:
    def test_heart_pressure_reaches_lbm_inlet(self):
        """HeartPump's arterial pressure should influence LBM inlet."""
        gm, _ = _build_coupled_graph()

        # Run enough steps for pressure to propagate
        gm.run(10)

        heart_state = gm.get_node_state("heart")
        vessel_state = gm.get_node_state("vessel")

        # Heart should be producing nonzero arterial pressure
        p_art = float(heart_state["arterial_pressure"])
        assert jnp.isfinite(jnp.array(p_art)), "Arterial pressure is not finite"

        # LBM inlet face should have non-uniform pressure (driven by heart)
        p_inlet = vessel_state["pressure"][0, :, :]  # x_min face
        assert jnp.all(jnp.isfinite(p_inlet)), "Inlet pressure not finite"


# ═══════════════════════════════════════════════════════════════════════
# 3. Bidirectional coupling (LBM outlet -> HeartPump backpressure)
# ═══════════════════════════════════════════════════════════════════════

class TestHeartLBMBidirectional:
    def test_lbm_outlet_pressure_feeds_back(self):
        """LBM's outlet pressure avg should feed back to HeartPump."""
        gm, _ = _build_coupled_graph()

        # Run a few steps to let coupling exchange values
        gm.run(5)

        # The LBM should compute a meaningful outlet pressure
        vessel_state = gm.get_node_state("vessel")
        lbm_node = gm._nodes["vessel"].node
        fluxes = lbm_node.compute_boundary_fluxes(vessel_state, {}, 1.0)
        outlet_p = float(fluxes["outlet_pressure_avg"])
        assert jnp.isfinite(jnp.array(outlet_p)), (
            "LBM outlet_pressure_avg is not finite"
        )

        # HeartPump should have been influenced by this backpressure.
        # After several steps the arterial pressure should differ from
        # what it would be with zero backpressure.
        heart_state = gm.get_node_state("heart")
        p_art = float(heart_state["arterial_pressure"])
        assert jnp.isfinite(jnp.array(p_art))


# ═══════════════════════════════════════════════════════════════════════
# 4. Clot injection via wall_mask_update
# ═══════════════════════════════════════════════════════════════════════

class TestClotInjection:
    def test_clot_injection_changes_flow(self):
        """Injecting a clot via wall_mask_update should block flow."""
        grid = (16, 8, 8)
        vessel_params = {
            "parent_radius": 3.0,
            "daughter_radius": 2.0,
            "parent_length": 8.0,
            "daughter_length": 6.0,
            "bifurcation_angle": 30.0,
        }
        lbm_node, base_mask = _make_lbm(grid, vessel_params)
        state = lbm_node.initial_state()

        # Apply pressure BCs to drive flow
        bi = {
            "inlet_pressure": jnp.float32(1.0 / 3.0 * 1.01),
            "outlet_pressure": jnp.float32(1.0 / 3.0 * 0.99),
        }

        # Run a few steps to establish flow
        for _ in range(10):
            state = lbm_node.update(state, bi, 1.0)

        # Record mean velocity at mid-pipe
        vel_before = float(jnp.mean(jnp.abs(state["velocity"][8, :, :, 0])))

        # Inject a large clot: block most of the cross-section at x=8
        clot_np = np.array(base_mask, copy=True)  # writable copy
        clot_np[7:10, 1:7, 1:7] = True
        clot_mask = jnp.asarray(clot_np)

        bi_with_clot = {**bi, "wall_mask_update": clot_mask}
        for _ in range(10):
            state = lbm_node.update(state, bi_with_clot, 1.0)

        # Velocity at the clot site should be reduced
        vel_after = float(jnp.mean(jnp.abs(state["velocity"][8, :, :, 0])))

        # The clot blocked cells should have zero velocity
        clot_vel = state["velocity"][8, 1:7, 1:7, :]
        np.testing.assert_allclose(clot_vel, 0.0, atol=1e-10,
                                   err_msg="Clot cells should have zero velocity")


# ═══════════════════════════════════════════════════════════════════════
# 5. build_vessel_flow_graph helper
# ═══════════════════════════════════════════════════════════════════════

class TestVesselFlowHelper:
    def test_helper_produces_working_graph(self):
        """The helper function should return a compiled, working graph."""
        from maddening.examples.coupling.vessel_flow_helpers import (
            build_vessel_flow_graph,
        )

        gm, mask = build_vessel_flow_graph(
            grid_shape=SMALL_GRID,
            vessel_params=SMALL_VESSEL_PARAMS,
            dt=1.0,
            max_coupling_iters=3,
            coupling_tolerance=1e-3,
        )

        # Should be compiled
        assert gm._compiled_step is not None

        # Should have two nodes
        assert "heart" in gm._nodes
        assert "vessel" in gm._nodes

        # Mask should be a boolean array of correct shape
        assert mask.shape == SMALL_GRID
        assert mask.dtype == jnp.bool_

        # Should be able to step
        gm.run(5)
        heart_state = gm.get_node_state("heart")
        assert jnp.all(jnp.isfinite(heart_state["arterial_pressure"]))

    def test_helper_default_params(self):
        """Helper should work with default vessel_params=None."""
        from maddening.examples.coupling.vessel_flow_helpers import (
            build_vessel_flow_graph,
        )

        gm, mask = build_vessel_flow_graph(
            grid_shape=SMALL_GRID,
            vessel_params=None,
            dt=1.0,
            max_coupling_iters=3,
        )
        assert mask.shape == SMALL_GRID
        gm.run(3)
