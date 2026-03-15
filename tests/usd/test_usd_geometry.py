"""
Tests for USD geometry reader and vessel phantom (Phase 9).

9a. geometry_source on SimulationNode
9b. USD-initialized HeatNode
9c. Vessel phantom
9d. Bifurcation coupling example
"""

import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from pxr import Usd, UsdGeom, Gf, Vt

import maddening.usd  # noqa: F401 (schema registration)
from maddening.usd.geometry import (
    load_grid_from_usd,
    create_vessel_phantom,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.transforms import register_transform
from maddening.nodes.heat import HeatNode


# ======================================================================
# Phase 9a: geometry_source on SimulationNode
# ======================================================================

class TestGeometrySource:
    """Test geometry_source attribute on SimulationNode."""

    def test_default_none(self):
        node = HeatNode("rod", 0.001, n_cells=10)
        assert node.geometry_source is None

    def test_set_geometry_source(self):
        node = HeatNode("rod", 0.001, n_cells=10,
                        geometry_source="/Vessel/parent")
        assert node.geometry_source == "/Vessel/parent"

    def test_geometry_source_not_in_params(self):
        """geometry_source should be popped from params, not stored there."""
        node = HeatNode("rod", 0.001, n_cells=10,
                        geometry_source="/Vessel/parent")
        assert "geometry_source" not in node.params


# ======================================================================
# Phase 9b: USD-initialized HeatNode (load grid from USD)
# ======================================================================

class TestLoadGridFromUSD:
    """Test reading grid coordinates from USD prims."""

    def _make_points_stage(self, points_3d):
        """Create an in-memory stage with a BasisCurves prim."""
        stage = Usd.Stage.CreateInMemory()
        prim = UsdGeom.BasisCurves.Define(stage, "/curve")
        pts = Vt.Vec3fArray(
            [Gf.Vec3f(*p) for p in points_3d]
        )
        prim.GetPointsAttr().Set(pts)
        prim.GetCurveVertexCountsAttr().Set(Vt.IntArray([len(points_3d)]))
        return stage

    def test_load_x_axis(self):
        pts = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
        stage = self._make_points_stage(pts)
        x = load_grid_from_usd(stage, "/curve", axis=0)
        np.testing.assert_allclose(x, [0.0, 0.5, 1.0])

    def test_load_y_axis(self):
        pts = [(0.0, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 1.0, 0.0)]
        stage = self._make_points_stage(pts)
        y = load_grid_from_usd(stage, "/curve", axis=1)
        np.testing.assert_allclose(y, [0.0, 0.3, 1.0])

    def test_sorted_output(self):
        pts = [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
        stage = self._make_points_stage(pts)
        x = load_grid_from_usd(stage, "/curve", axis=0)
        np.testing.assert_allclose(x, [0.0, 0.5, 1.0])

    def test_duplicate_removal(self):
        pts = [
            (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0), (1.0, 0.0, 0.0),
        ]
        stage = self._make_points_stage(pts)
        x = load_grid_from_usd(stage, "/curve", axis=0)
        assert len(x) == 3

    def test_invalid_prim_raises(self):
        stage = Usd.Stage.CreateInMemory()
        with pytest.raises(ValueError, match="No prim"):
            load_grid_from_usd(stage, "/nonexistent")

    def test_no_points_attr_raises(self):
        stage = Usd.Stage.CreateInMemory()
        stage.DefinePrim("/empty", "Xform")
        with pytest.raises(ValueError, match="no 'points' attribute"):
            load_grid_from_usd(stage, "/empty")

    def test_initialize_heat_node_from_usd(self):
        """Create a HeatNode with grid_points loaded from a USD prim."""
        pts = [(float(x), 0.0, 0.0) for x in np.linspace(0, 1, 12)]
        stage = self._make_points_stage(pts)
        grid_x = load_grid_from_usd(stage, "/curve", axis=0)

        node = HeatNode(
            "rod", 0.001, n_cells=len(grid_x),
            grid_points=list(grid_x),
            thermal_diffusivity=0.01,
            geometry_source="/curve",
        )
        assert node.geometry_source == "/curve"
        assert node._is_nonuniform
        state = node.initial_state()
        assert state["temperature"].shape == (len(grid_x),)


# ======================================================================
# Phase 9c: Vessel phantom
# ======================================================================

class TestVesselPhantom:
    """Test vessel phantom creation."""

    def test_create_vessel(self, tmp_path):
        filepath = str(tmp_path / "vessel.usda")
        stage = create_vessel_phantom(filepath)

        # Check parent exists
        parent = stage.GetPrimAtPath("/Vessel/parent")
        assert parent.IsValid()
        points = UsdGeom.BasisCurves(parent).GetPointsAttr().Get()
        assert len(points) == 20  # default parent_n_points

    def test_vessel_has_daughters(self, tmp_path):
        filepath = str(tmp_path / "vessel.usda")
        stage = create_vessel_phantom(filepath)

        left = stage.GetPrimAtPath("/Vessel/daughter_left")
        right = stage.GetPrimAtPath("/Vessel/daughter_right")
        assert left.IsValid()
        assert right.IsValid()

    def test_vessel_daughter_diverge(self, tmp_path):
        """Daughter tubes diverge from the bifurcation point."""
        filepath = str(tmp_path / "vessel.usda")
        stage = create_vessel_phantom(
            filepath, bifurcation_angle=30.0
        )

        left = UsdGeom.BasisCurves(
            stage.GetPrimAtPath("/Vessel/daughter_left")
        )
        right = UsdGeom.BasisCurves(
            stage.GetPrimAtPath("/Vessel/daughter_right")
        )

        left_pts = left.GetPointsAttr().Get()
        right_pts = right.GetPointsAttr().Get()

        # First points (at bifurcation) should be the same
        assert abs(left_pts[0][0] - right_pts[0][0]) < 1e-5
        assert abs(left_pts[0][1] - right_pts[0][1]) < 1e-5

        # Last points should diverge in y
        assert left_pts[-1][1] < 0  # left goes negative y
        assert right_pts[-1][1] > 0  # right goes positive y

    def test_vessel_custom_params(self, tmp_path):
        filepath = str(tmp_path / "vessel.usda")
        stage = create_vessel_phantom(
            filepath,
            parent_length=2.0,
            daughter_length=1.5,
            parent_n_points=30,
            daughter_n_points=25,
        )

        parent = UsdGeom.BasisCurves(
            stage.GetPrimAtPath("/Vessel/parent")
        )
        left = UsdGeom.BasisCurves(
            stage.GetPrimAtPath("/Vessel/daughter_left")
        )

        assert len(parent.GetPointsAttr().Get()) == 30
        assert len(left.GetPointsAttr().Get()) == 25

    def test_vessel_loads_grid(self, tmp_path):
        """Grid coordinates can be loaded from the vessel phantom."""
        filepath = str(tmp_path / "vessel.usda")
        stage = create_vessel_phantom(filepath, parent_n_points=15)

        x = load_grid_from_usd(stage, "/Vessel/parent", axis=0)
        assert len(x) == 15
        assert x[0] < x[-1]  # sorted ascending

    def test_vessel_file_persists(self, tmp_path):
        """Vessel file can be saved and reloaded."""
        filepath = str(tmp_path / "vessel.usdc")
        create_vessel_phantom(filepath)

        stage2 = Usd.Stage.Open(filepath)
        parent = stage2.GetPrimAtPath("/Vessel/parent")
        assert parent.IsValid()


# ======================================================================
# Phase 9d: Bifurcation coupling example
# ======================================================================

@register_transform("_test_extract_right", "Extract rightmost temp")
def _extract_right(T):
    return T[-1]


@register_transform("_test_extract_left", "Extract leftmost temp")
def _extract_left(T):
    return T[0]


class TestBifurcationCoupling:
    """Test three-node coupled bifurcation."""

    @pytest.fixture
    def vessel_stage(self, tmp_path):
        filepath = str(tmp_path / "vessel.usda")
        return create_vessel_phantom(
            filepath,
            parent_n_points=12,
            daughter_n_points=10,
        )

    def test_three_node_coupling_runs(self, vessel_stage):
        """Three HeatNodes coupled at bifurcation can step."""
        parent_x = load_grid_from_usd(vessel_stage, "/Vessel/parent", axis=0)
        left_x = load_grid_from_usd(
            vessel_stage, "/Vessel/daughter_left", axis=0
        )
        right_x = load_grid_from_usd(
            vessel_stage, "/Vessel/daughter_right", axis=0
        )

        dt = 0.0001
        gm = GraphManager()
        gm.add_node(HeatNode(
            "parent", dt, n_cells=len(parent_x),
            grid_points=list(parent_x - parent_x[0]),
            thermal_diffusivity=0.01,
            initial_temperature=100.0,
        ))
        gm.add_node(HeatNode(
            "left", dt, n_cells=len(left_x),
            grid_points=list(left_x - left_x[0]),
            thermal_diffusivity=0.01,
            initial_temperature=20.0,
        ))
        gm.add_node(HeatNode(
            "right", dt, n_cells=len(right_x),
            grid_points=list(right_x - right_x[0]),
            thermal_diffusivity=0.01,
            initial_temperature=20.0,
        ))

        # Coupling edges
        gm.add_edge("parent", "left", "temperature", "left_temperature",
                     transform="_test_extract_right")
        gm.add_edge("parent", "right", "temperature", "left_temperature",
                     transform="_test_extract_right")
        gm.add_edge("left", "parent", "temperature", "right_temperature",
                     transform="_test_extract_left")

        gm.add_coupling_group(
            ["parent", "left", "right"],
            max_iterations=10,
            tolerance=1e-6,
        )
        gm.compile()

        for _ in range(5):
            state = gm.step()

        # Check temperatures are finite and in reasonable range
        for name in ["parent", "left", "right"]:
            T = state[name]["temperature"]
            assert jnp.isfinite(T).all()
            assert float(T.min()) >= -50.0
            assert float(T.max()) <= 200.0

    def test_bifurcation_heat_flows(self, vessel_stage):
        """Heat flows from hot parent into cool daughters."""
        parent_x = load_grid_from_usd(vessel_stage, "/Vessel/parent", axis=0)
        left_x = load_grid_from_usd(
            vessel_stage, "/Vessel/daughter_left", axis=0
        )

        dt = 0.0001
        gm = GraphManager()
        gm.add_node(HeatNode(
            "parent", dt, n_cells=len(parent_x),
            grid_points=list(parent_x - parent_x[0]),
            thermal_diffusivity=0.01,
            initial_temperature=100.0,
        ))
        gm.add_node(HeatNode(
            "left", dt, n_cells=len(left_x),
            grid_points=list(left_x - left_x[0]),
            thermal_diffusivity=0.01,
            initial_temperature=20.0,
        ))

        gm.add_edge("parent", "left", "temperature", "left_temperature",
                     transform="_test_extract_right")
        gm.add_edge("left", "parent", "temperature", "right_temperature",
                     transform="_test_extract_left")

        gm.add_coupling_group(
            ["parent", "left"],
            max_iterations=15,
            tolerance=1e-8,
        )
        gm.compile()

        T_left_initial = float(gm._state["left"]["temperature"][1])

        for _ in range(20):
            state = gm.step()

        T_left_after = float(state["left"]["temperature"][1])
        # Daughter should warm up
        assert T_left_after > T_left_initial
