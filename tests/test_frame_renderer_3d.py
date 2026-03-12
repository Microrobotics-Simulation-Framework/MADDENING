"""Tests for the PyVista/VTK 3D server-side frame renderer."""

import numpy as np
import pytest

from maddening.api.frame_renderer_3d import (
    ServerFrameRenderer3D,
    View3DConfig,
    SliceConfig,
    ArrowConfig,
    PipeWallConfig,
    StreamlineConfig,
)


def _make_state(nx=8, ny=8, nz=8, ux=0.02):
    """Create a simple mock fluid state for testing."""
    density = np.ones((nx, ny, nz), dtype=np.float32)
    velocity = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    # Simple parabolic-ish x-velocity profile (higher in center)
    cy, cz = (ny - 1) / 2.0, (nz - 1) / 2.0
    for iy in range(ny):
        for iz in range(nz):
            r = np.sqrt((iy - cy) ** 2 + (iz - cz) ** 2)
            rmax = min(ny, nz) / 2.0
            if r < rmax * 0.9:
                velocity[:, iy, iz, 0] = ux * (1 - (r / rmax) ** 2)
    return {"fluid": {"density": density, "velocity": velocity}}


class TestServerFrameRenderer3D:
    def test_basic_render_jpeg(self):
        config = View3DConfig(
            node="fluid",
            slices=[SliceConfig(node="fluid", field="density", normal="x")],
        )
        renderer = ServerFrameRenderer3D(config, width=320, height=240)
        state = _make_state()
        data = renderer.render(0.0, state)
        assert isinstance(data, bytes)
        assert len(data) > 100
        assert data[:2] == b"\xff\xd8"  # JPEG magic
        renderer.close()

    def test_png_format(self):
        config = View3DConfig(
            node="fluid",
            slices=[SliceConfig(node="fluid", field="density", normal="x")],
        )
        renderer = ServerFrameRenderer3D(config, width=320, height=240, fmt="png")
        state = _make_state()
        data = renderer.render(0.0, state)
        assert data[:4] == b"\x89PNG"
        renderer.close()

    def test_content_type(self):
        config = View3DConfig(node="fluid")
        r = ServerFrameRenderer3D(config, fmt="jpeg")
        assert r.content_type == "image/jpeg"
        r.set_format("png")
        assert r.content_type == "image/png"
        r.close()

    def test_velocity_magnitude_slice(self):
        config = View3DConfig(
            node="fluid",
            slices=[
                SliceConfig(node="fluid", field="velocity", component=-1,
                            normal="x", origin_frac=0.5, cmap="coolwarm"),
            ],
        )
        renderer = ServerFrameRenderer3D(config, width=320, height=240)
        state = _make_state(ux=0.05)
        data = renderer.render(0.0, state)
        assert len(data) > 100
        renderer.close()

    def test_multiple_slices(self):
        config = View3DConfig(
            node="fluid",
            slices=[
                SliceConfig(node="fluid", field="density", normal="x",
                            origin_frac=0.25),
                SliceConfig(node="fluid", field="density", normal="x",
                            origin_frac=0.75),
                SliceConfig(node="fluid", field="velocity", component=0,
                            normal="y", origin_frac=0.5),
            ],
        )
        renderer = ServerFrameRenderer3D(config, width=320, height=240)
        state = _make_state()
        data = renderer.render(0.0, state)
        assert len(data) > 100
        renderer.close()

    def test_arrow_glyphs(self):
        config = View3DConfig(
            node="fluid",
            arrows=[
                ArrowConfig(node="fluid", field="velocity",
                            normal="x", origin_frac=0.5,
                            scale=20.0, stride=2),
            ],
        )
        renderer = ServerFrameRenderer3D(config, width=320, height=240)
        state = _make_state(ux=0.05)
        data = renderer.render(0.0, state)
        assert len(data) > 100
        renderer.close()

    def test_pipe_wall(self):
        config = View3DConfig(
            node="fluid",
            pipe_wall=PipeWallConfig(radius_frac=0.9, opacity=0.15),
            slices=[SliceConfig(node="fluid", field="density", normal="x")],
        )
        renderer = ServerFrameRenderer3D(config, width=320, height=240)
        state = _make_state()
        data = renderer.render(0.0, state)
        assert len(data) > 100
        renderer.close()

    def test_multiple_frames(self):
        """Data update path: second+ frames update mesh data in place."""
        config = View3DConfig(
            node="fluid",
            slices=[SliceConfig(node="fluid", field="velocity", component=-1,
                                normal="x", origin_frac=0.5)],
            arrows=[ArrowConfig(node="fluid", scale=20.0, stride=2)],
        )
        renderer = ServerFrameRenderer3D(config, width=320, height=240)

        state1 = _make_state(ux=0.01)
        data1 = renderer.render(0.0, state1)

        state2 = _make_state(ux=0.05)
        data2 = renderer.render(0.1, state2)

        # Both should produce valid output
        assert len(data1) > 100
        assert len(data2) > 100
        # Different input should produce different output
        assert data1 != data2
        renderer.close()

    def test_set_format(self):
        config = View3DConfig(
            node="fluid",
            slices=[SliceConfig(node="fluid", field="density", normal="x")],
        )
        renderer = ServerFrameRenderer3D(config, width=320, height=240, fmt="jpeg")
        assert renderer.fmt == "jpeg"

        renderer.set_format("png", quality=90)
        assert renderer.fmt == "png"

        state = _make_state()
        data = renderer.render(0.0, state)
        assert data[:4] == b"\x89PNG"
        renderer.close()

    def test_camera_positions(self):
        """Different camera presets should produce different images."""
        state = _make_state()
        images = {}
        for cam in ["xz", "xy", "yz"]:
            config = View3DConfig(
                node="fluid",
                slices=[SliceConfig(node="fluid", field="density", normal="x")],
                pipe_wall=PipeWallConfig(),
                camera_position=cam,
            )
            renderer = ServerFrameRenderer3D(config, width=320, height=240)
            images[cam] = renderer.render(0.0, state)
            renderer.close()

        # At least some views should differ
        assert images["xz"] != images["xy"] or images["xz"] != images["yz"]

    def test_close_and_reuse(self):
        config = View3DConfig(
            node="fluid",
            slices=[SliceConfig(node="fluid", field="density", normal="x")],
        )
        renderer = ServerFrameRenderer3D(config, width=320, height=240)
        state = _make_state()

        data1 = renderer.render(0.0, state)
        renderer.close()

        # Should be able to render again after close (re-initializes)
        data2 = renderer.render(0.1, state)
        assert len(data1) > 100
        assert len(data2) > 100
        renderer.close()

    def test_fixed_clim(self):
        """Fixed clim should not error."""
        config = View3DConfig(
            node="fluid",
            slices=[
                SliceConfig(node="fluid", field="density", normal="x",
                            clim=(0.9, 1.1)),
            ],
        )
        renderer = ServerFrameRenderer3D(config, width=320, height=240)
        data = renderer.render(0.0, _make_state())
        assert len(data) > 100
        renderer.close()

    def test_z_normal_slice(self):
        config = View3DConfig(
            node="fluid",
            slices=[
                SliceConfig(node="fluid", field="density", normal="z",
                            origin_frac=0.5),
            ],
        )
        renderer = ServerFrameRenderer3D(config, width=320, height=240)
        data = renderer.render(0.0, _make_state())
        assert len(data) > 100
        renderer.close()

    def test_properties(self):
        config = View3DConfig(node="fluid")
        renderer = ServerFrameRenderer3D(config, width=640, height=480,
                                          fmt="jpeg", quality=90)
        assert renderer.width == 640
        assert renderer.height == 480
        assert renderer.fmt == "jpeg"
        assert renderer.content_type == "image/jpeg"
        renderer.close()
