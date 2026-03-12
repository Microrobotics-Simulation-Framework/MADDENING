"""Tests for the server-side frame renderer."""

import io

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.api.frame_renderer import (
    ServerFrameRenderer,
    SceneConfig,
    SceneObject,
    TimeSeriesConfig,
    HeatmapConfig,
)


@pytest.fixture
def sample_state():
    return {
        "ball": {
            "position": jnp.array(5.0),
            "velocity": jnp.array(-3.2),
        },
        "spring": {
            "position": jnp.array(3.0),
            "velocity": jnp.array(0.5),
        },
        "table": {
            "position": jnp.array(0.0),
        },
        "heat_rod": {
            "temperature": jnp.ones(20) * 20.0,
        },
    }


@pytest.fixture
def scene_config():
    return SceneConfig(
        objects=[
            SceneObject(node="table", y="position", kind="surface", color="#8B7355"),
            SceneObject(node="ball", y="position", kind="circle", color="#DD4444"),
            SceneObject(node="spring", y="position", kind="circle", x=0.8, color="#4488DD"),
        ],
        xlim=(-2, 2), ylim=(-1, 8),
    )


@pytest.fixture
def ts_configs():
    return [
        TimeSeriesConfig(
            fields=[("ball", "position", "Ball"), ("spring", "position", "Spring")],
            window=100,
        ),
    ]


class TestServerFrameRenderer:
    def test_render_produces_jpeg_bytes(self, scene_config, ts_configs, sample_state):
        renderer = ServerFrameRenderer(
            scene=scene_config, timeseries=ts_configs,
            width=640, height=480, fmt="jpeg",
        )
        frame = renderer.render(0.0, sample_state)
        assert isinstance(frame, bytes)
        assert len(frame) > 100
        # JPEG magic bytes
        assert frame[:2] == b'\xff\xd8'
        renderer.close()

    def test_render_produces_png_bytes(self, scene_config, sample_state):
        renderer = ServerFrameRenderer(
            scene=scene_config, width=320, height=240, fmt="png",
        )
        frame = renderer.render(0.5, sample_state)
        assert isinstance(frame, bytes)
        # PNG magic bytes
        assert frame[:4] == b'\x89PNG'
        renderer.close()

    def test_multiple_frames_same_size(self, scene_config, ts_configs, sample_state):
        renderer = ServerFrameRenderer(
            scene=scene_config, timeseries=ts_configs,
            width=640, height=480, fmt="png",
        )
        frames = [renderer.render(t * 0.01, sample_state) for t in range(5)]
        # PNG frames may vary slightly but should all be valid
        for f in frames:
            assert f[:4] == b'\x89PNG'
        renderer.close()

    def test_content_type(self):
        r = ServerFrameRenderer(width=100, height=100, fmt="jpeg")
        assert r.content_type == "image/jpeg"
        r.set_format("png")
        assert r.content_type == "image/png"
        r.close()

    def test_scene_only(self, scene_config, sample_state):
        renderer = ServerFrameRenderer(scene=scene_config, width=320, height=240)
        frame = renderer.render(0.0, sample_state)
        assert len(frame) > 0
        renderer.close()

    def test_timeseries_only(self, ts_configs, sample_state):
        renderer = ServerFrameRenderer(timeseries=ts_configs, width=320, height=240)
        frame = renderer.render(0.0, sample_state)
        assert len(frame) > 0
        renderer.close()

    def test_heatmap_panel(self, sample_state):
        renderer = ServerFrameRenderer(
            heatmaps=[HeatmapConfig(
                node="heat_rod", field="temperature",
                vmin=0, vmax=100, title="Heat",
            )],
            width=320, height=240,
        )
        frame = renderer.render(0.0, sample_state)
        assert len(frame) > 0
        renderer.close()

    def test_reset_clears_buffers(self, ts_configs, sample_state):
        renderer = ServerFrameRenderer(timeseries=ts_configs, width=320, height=240)
        renderer.render(0.0, sample_state)
        renderer.render(0.1, sample_state)
        assert len(renderer._time_buffers["ball.position"]) == 2
        renderer.reset()
        assert len(renderer._time_buffers["ball.position"]) == 0
        renderer.close()

    def test_resize_changes_output(self, scene_config, sample_state):
        renderer = ServerFrameRenderer(scene=scene_config, width=320, height=240, fmt="png")
        frame1 = renderer.render(0.0, sample_state)
        renderer.resize(160, 120)
        frame2 = renderer.render(0.0, sample_state)
        # Smaller resolution should produce smaller file
        assert len(frame2) < len(frame1)
        renderer.close()

    def test_set_format_and_quality(self):
        renderer = ServerFrameRenderer(width=100, height=100, fmt="jpeg", quality=50)
        assert renderer.fmt == "jpeg"
        assert renderer.quality == 50
        renderer.set_format("png", quality=None)
        assert renderer.fmt == "png"
        assert renderer.quality == 50  # unchanged
        renderer.set_format("jpeg", quality=95)
        assert renderer.quality == 95
        renderer.close()

    def test_composite_layout(self, scene_config, sample_state):
        """Scene + timeseries + heatmap all in one figure."""
        renderer = ServerFrameRenderer(
            scene=scene_config,
            timeseries=[
                TimeSeriesConfig(
                    fields=[("ball", "position", "Ball")],
                    window=50,
                ),
            ],
            heatmaps=[
                HeatmapConfig(node="heat_rod", field="temperature",
                              vmin=0, vmax=100),
            ],
            width=800, height=600, fmt="jpeg",
        )
        frame = renderer.render(1.0, sample_state)
        assert frame[:2] == b'\xff\xd8'
        assert len(frame) > 500
        renderer.close()

    def test_dict_scene_objects(self, sample_state):
        """SceneConfig accepts dicts as objects (not just SceneObject)."""
        cfg = SceneConfig(objects=[
            {"node": "ball", "y": "position", "kind": "circle", "color": "red"},
        ])
        renderer = ServerFrameRenderer(scene=cfg, width=320, height=240)
        frame = renderer.render(0.0, sample_state)
        assert len(frame) > 0
        renderer.close()

    def test_hline_object(self, sample_state):
        cfg = SceneConfig(objects=[
            SceneObject(node="table", y="position", kind="hline", color="gray"),
        ])
        renderer = ServerFrameRenderer(scene=cfg, width=320, height=240)
        frame = renderer.render(0.0, sample_state)
        assert len(frame) > 0
        renderer.close()
