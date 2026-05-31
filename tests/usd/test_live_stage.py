"""Tests for the generic :class:`LiveStage` writer (v0.3.0 §A3)."""

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

# Skip if USD bindings aren't installed.
pxr = pytest.importorskip("pxr", reason="usd-core not installed")
from pxr import UsdGeom

from maddening.usd.live_stage import (
    LiveStage,
    make_translate_orient_updater,
    make_translate_updater,
    quat_from_z_to_direction,
    quat_wxyz_to_gf,
    vec3_to_gf,
)


class TestLiveStageConstruction:

    def test_default_in_memory(self):
        stage = LiveStage()
        assert stage.stage is not None
        # The /World prim is created.
        world = stage.stage.GetPrimAtPath("/World")
        assert world.IsValid()
        # A default camera is set up.
        cam = stage.stage.GetPrimAtPath("/World/Camera")
        assert cam.IsValid()

    def test_meters_per_unit_default_is_si(self):
        stage = LiveStage()
        assert UsdGeom.GetStageMetersPerUnit(stage.stage) == 1.0

    def test_y_up_supported(self):
        stage = LiveStage(up_axis="Y")
        assert UsdGeom.GetStageUpAxis(stage.stage) == UsdGeom.Tokens.y

    def test_unknown_up_axis_rejected(self):
        with pytest.raises(ValueError, match="up_axis"):
            LiveStage(up_axis="x")


class TestDynamicPrimRegistry:
    """Domain-neutral prim registry + per-frame update dispatch."""

    def test_register_and_update_translate(self):
        stage = LiveStage()
        sphere = UsdGeom.Sphere.Define(stage.stage, "/World/Ball")
        sphere.GetRadiusAttr().Set(0.001)
        xform = UsdGeom.Xformable(sphere.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp()

        stage.register_prim(
            node_name="ball", prim_path="/World/Ball",
            updater=make_translate_updater(),
        )
        stage.update({"ball": {"position": np.array([1.0, 2.0, 3.0])}})

        ops = UsdGeom.Xformable(
            stage.stage.GetPrimAtPath("/World/Ball"),
        ).GetOrderedXformOps()
        translate = ops[0].Get()
        assert tuple(translate) == (1.0, 2.0, 3.0)

    def test_register_and_update_translate_orient(self):
        stage = LiveStage()
        sphere = UsdGeom.Sphere.Define(stage.stage, "/World/Body")
        sphere.GetRadiusAttr().Set(0.001)
        xform = UsdGeom.Xformable(sphere.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp()
        xform.AddOrientOp()

        stage.register_prim(
            node_name="rb", prim_path="/World/Body",
            updater=make_translate_orient_updater(),
        )
        stage.update({"rb": {
            "position": np.array([0.1, 0.2, 0.3]),
            "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
        }})

        ops = UsdGeom.Xformable(
            stage.stage.GetPrimAtPath("/World/Body"),
        ).GetOrderedXformOps()
        assert tuple(ops[0].Get()) == (0.1, 0.2, 0.3)
        # Identity quaternion comes back as Quatf(1, (0,0,0)).
        q = ops[1].Get()
        assert math.isclose(q.GetReal(), 1.0)

    def test_missing_node_state_silently_skipped(self):
        """If a registered prim's node isn't in `state`, just skip it
        (rather than raising) — the contract matches MIME's bridge."""
        stage = LiveStage()
        sphere = UsdGeom.Sphere.Define(stage.stage, "/World/Ball")
        sphere.GetRadiusAttr().Set(0.001)
        xform = UsdGeom.Xformable(sphere.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp()
        stage.register_prim(
            node_name="ball", prim_path="/World/Ball",
            updater=make_translate_updater(),
        )
        # Empty state — no error.
        stage.update({})

    def test_time_sampled_writes(self):
        from pxr import Usd
        stage = LiveStage()
        sphere = UsdGeom.Sphere.Define(stage.stage, "/World/Ball")
        sphere.GetRadiusAttr().Set(0.001)
        xform = UsdGeom.Xformable(sphere.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp()
        stage.register_prim(
            node_name="ball", prim_path="/World/Ball",
            updater=make_translate_updater(),
        )
        for t in range(5):
            stage.update(
                {"ball": {"position": np.array([float(t), 0.0, 0.0])}},
                time_code=Usd.TimeCode(t),
            )
        ops = UsdGeom.Xformable(
            stage.stage.GetPrimAtPath("/World/Ball"),
        ).GetOrderedXformOps()
        for t in range(5):
            tr = ops[0].Get(Usd.TimeCode(t))
            assert math.isclose(tr[0], float(t))


class TestSceneDressing:

    def test_dome_light(self):
        stage = LiveStage()
        stage.add_dome_light(intensity=750.0)
        # Light exists if USD shade is installed; skip if not.
        try:
            from pxr import UsdLux  # noqa: F401
        except ImportError:
            pytest.skip("UsdLux not installed")
        prim = stage.stage.GetPrimAtPath("/Lights/Dome")
        assert prim.IsValid()

    def test_ground_plane_default(self):
        stage = LiveStage()
        stage.add_ground_plane()
        prim = stage.stage.GetPrimAtPath("/World/Environment/Ground")
        assert prim.IsValid()

    def test_ground_plane_unknown_normal_raises(self):
        stage = LiveStage()
        with pytest.raises(ValueError, match="X/Y/Z"):
            stage.add_ground_plane(normal="oblique")

    def test_create_material(self):
        try:
            from pxr import UsdShade  # noqa: F401
        except ImportError:
            pytest.skip("UsdShade not installed")
        stage = LiveStage()
        path = stage.create_material("rubber", diffuse_color=(0.5, 0.1, 0.1))
        assert path == "/Materials/rubber"


class TestExport:

    def test_export_round_trip(self, tmp_path):
        from pxr import Usd
        stage = LiveStage()
        sphere = UsdGeom.Sphere.Define(stage.stage, "/World/Demo")
        sphere.GetRadiusAttr().Set(0.5)
        out = tmp_path / "out.usda"
        stage.export(str(out))
        assert out.exists()
        # Reload and confirm the prim survived.
        loaded = Usd.Stage.Open(str(out))
        assert loaded.GetPrimAtPath("/World/Demo").IsValid()


class TestConverters:

    def test_quat_from_z_identity(self):
        q = quat_from_z_to_direction(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0])

    def test_quat_from_z_minus_z(self):
        q = quat_from_z_to_direction(np.array([0.0, 0.0, -1.0]))
        # Half-rotation about an axis perpendicular to Z.
        np.testing.assert_allclose(q[0], 0.0, atol=1e-7)

    def test_quat_from_z_zero_vector_safe(self):
        q = quat_from_z_to_direction(np.zeros(3))
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0])

    def test_vec3_converter(self):
        gf = vec3_to_gf(np.array([1.0, 2.0, 3.0]))
        assert tuple(gf) == (1.0, 2.0, 3.0)

    def test_quat_wxyz_converter(self):
        gf = quat_wxyz_to_gf(np.array([0.7, 0.1, 0.2, 0.3]))
        # Gf.Quatf is float32 — match its precision.
        assert math.isclose(float(gf.GetReal()), 0.7, abs_tol=1e-6)


class TestStabilityTagging:

    def test_live_stage_tagged_evolving(self):
        from maddening.core.compliance.metadata import StabilityLevel
        assert LiveStage._stability_level == StabilityLevel.EVOLVING


class TestNonMimeDemo:
    """A3 acceptance: a non-MIME MADDENING demo renders live in
    MICROROBOTICA via the new maddening.usd.live_stage path."""

    def test_bouncing_ball_demo_writes_time_sampled_stage(self, tmp_path):
        """End-to-end exercise of the demo function.  Skipped if pxr
        is missing (the conftest already gates this directory)."""
        from maddening.examples.advanced.live_stage_bouncing_ball_demo import run
        out = run(out_path=tmp_path / "ball.usda", n_steps=5)
        assert out.exists()
        text = out.read_text()
        # Time-sampled translate written.
        assert "timeSamples" in text
        # Default scene dressing wired up.
        assert "Camera" in text
        # Verify the file actually opens as USD and the ball prim is valid.
        from pxr import Usd
        loaded = Usd.Stage.Open(str(out))
        assert loaded is not None
        assert loaded.GetPrimAtPath("/World/Ball").IsValid()
