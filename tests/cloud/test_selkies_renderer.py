"""Tests for SelkiesRenderer."""

import warnings
from typing import Optional

import pytest

from maddening.cloud.mock_streaming import MockStreamSession
from maddening.cloud.streaming import StreamConfig, QualityPreset
from maddening.viz.backends.selkies_renderer import SelkiesRenderer
from maddening.viz.renderer import GraphInfo, Renderer
from maddening.warnings import PerformanceWarning


# -- Minimal mock renderers -------------------------------------------

class _MockInnerRenderer(Renderer):
    """CPU-only inner renderer for testing."""

    def __init__(self, width: int = 64, height: int = 32):
        self.width = width
        self.height = height
        self.setup_called = False
        self.update_count = 0
        self.teardown_called = False

    def setup(self, graph_info: GraphInfo) -> None:
        self.setup_called = True

    def update(self, sim_time: float, state: dict[str, dict]) -> None:
        self.update_count += 1

    def teardown(self) -> None:
        self.teardown_called = True

    def read_framebuffer_cpu(self):
        pixels = b"\xFF" * (self.width * self.height * 4)
        return pixels, self.width, self.height, "RGBA"

    def requested_fields(self) -> Optional[dict[str, list[str]]]:
        return {"ball": ["position"]}


class _MockGPURenderer(Renderer):
    """GPU-capable inner renderer for testing."""

    def __init__(self):
        self.setup_called = False
        self.update_count = 0
        self.teardown_called = False

    def setup(self, graph_info: GraphInfo) -> None:
        self.setup_called = True

    def update(self, sim_time: float, state: dict[str, dict]) -> None:
        self.update_count += 1

    def teardown(self) -> None:
        self.teardown_called = True

    def read_framebuffer_gpu(self):
        from maddening.cloud.streaming import GPUFramebuffer
        return GPUFramebuffer(
            cuda_ptr=0xBEEF, width=1280, height=720,
            stride_bytes=1280 * 4,
        )


class _MockBareRenderer(Renderer):
    """Renderer with no read_framebuffer methods at all."""

    def setup(self, graph_info: GraphInfo) -> None:
        pass

    def update(self, sim_time: float, state: dict[str, dict]) -> None:
        pass

    def teardown(self) -> None:
        pass


# -- Dummy graph info -------------------------------------------------

_GRAPH_INFO = GraphInfo(
    node_names=["ball"],
    node_params={"ball": {}},
    node_state_fields={"ball": ["position", "velocity"]},
    edges=[],
    timestep=0.01,
)


# -- Tests ------------------------------------------------------------

class TestSelkiesRenderer:
    def test_setup_starts_session(self):
        inner = _MockInnerRenderer()
        session = MockStreamSession()
        renderer = SelkiesRenderer(inner, session)
        renderer.setup(_GRAPH_INFO)
        assert inner.setup_called
        assert session.is_alive()
        assert renderer.stream_info is not None
        assert renderer.url is not None

    def test_update_pushes_cpu_frames(self):
        inner = _MockInnerRenderer()
        session = MockStreamSession()
        renderer = SelkiesRenderer(inner, session)
        renderer.setup(_GRAPH_INFO)
        renderer.update(0.0, {"ball": {"position": 1.0}})
        assert inner.update_count == 1
        assert session.frame_count == 1

    def test_update_pushes_gpu_frames(self):
        inner = _MockGPURenderer()
        session = MockStreamSession()
        renderer = SelkiesRenderer(inner, session)
        renderer.setup(_GRAPH_INFO)
        renderer.update(0.0, {})
        assert inner.update_count == 1
        assert len(session._gpu_frames) == 1

    def test_cpu_fallback_warning(self):
        inner = _MockInnerRenderer()
        session = MockStreamSession()
        renderer = SelkiesRenderer(inner, session)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            renderer.setup(_GRAPH_INFO)
        perf_warnings = [x for x in w if issubclass(x.category, PerformanceWarning)]
        assert len(perf_warnings) == 1
        assert "CPU" in str(perf_warnings[0].message)

    def test_no_warning_for_gpu_renderer(self):
        inner = _MockGPURenderer()
        session = MockStreamSession()
        renderer = SelkiesRenderer(inner, session)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            renderer.setup(_GRAPH_INFO)
        perf_warnings = [x for x in w if issubclass(x.category, PerformanceWarning)]
        assert len(perf_warnings) == 0

    def test_teardown(self):
        inner = _MockInnerRenderer()
        session = MockStreamSession()
        renderer = SelkiesRenderer(inner, session)
        renderer.setup(_GRAPH_INFO)
        renderer.teardown()
        assert inner.teardown_called
        assert not session.is_alive()

    def test_requested_fields_delegates(self):
        inner = _MockInnerRenderer()
        session = MockStreamSession()
        renderer = SelkiesRenderer(inner, session)
        assert renderer.requested_fields() == {"ball": ["position"]}

    def test_url_none_before_setup(self):
        inner = _MockInnerRenderer()
        session = MockStreamSession()
        renderer = SelkiesRenderer(inner, session)
        assert renderer.url is None
        assert renderer.stream_info is None

    def test_bare_renderer_fallback(self):
        """Renderer without read_framebuffer_cpu still works."""
        inner = _MockBareRenderer()
        session = MockStreamSession()
        config = StreamConfig(width=16, height=16)
        renderer = SelkiesRenderer(inner, session, config=config)
        renderer.setup(_GRAPH_INFO)
        renderer.update(0.0, {})
        assert session.frame_count == 1
        # Placeholder frame: 16*16*4 bytes of zeros
        assert session.last_frame == b"\x00" * (16 * 16 * 4)

    def test_custom_config(self):
        inner = _MockInnerRenderer()
        session = MockStreamSession()
        config = StreamConfig.from_preset(QualityPreset.CAPTURE)
        renderer = SelkiesRenderer(inner, session, config=config)
        renderer.setup(_GRAPH_INFO)
        assert session.config.width == 1920
        assert session.config.fps == 60
