"""Tests for MockStreamSession."""

import pytest

from maddening.cloud.mock_streaming import MockStreamSession
from maddening.cloud.streaming import (
    GPUFramebuffer,
    QualityPreset,
    StreamConfig,
    StreamReconfigError,
    StreamStartError,
)


class TestMockStreamSession:
    def test_start_returns_stream_info(self):
        session = MockStreamSession()
        config = StreamConfig.from_preset(QualityPreset.STANDARD)
        info = session.start(config)
        assert info.session_id
        assert "signaling" in info.signaling_url
        assert session.is_alive()

    def test_stop(self):
        session = MockStreamSession()
        session.start(StreamConfig())
        assert session.is_alive()
        session.stop()
        assert not session.is_alive()

    def test_fail_on_start(self):
        session = MockStreamSession(fail_on_start=True)
        with pytest.raises(StreamStartError):
            session.start(StreamConfig())
        assert not session.is_alive()

    def test_cpu_framebuffer(self):
        session = MockStreamSession()
        session.start(StreamConfig())
        pixels = b"\x00" * (1280 * 720 * 4)
        session.update_framebuffer_cpu(pixels, 1280, 720)
        assert session.frame_count == 1
        assert session.last_frame == pixels

    def test_cpu_framebuffer_ring_buffer(self):
        session = MockStreamSession(max_frames=3)
        session.start(StreamConfig())
        for i in range(5):
            session.update_framebuffer_cpu(bytes([i]) * 4, 1, 1)
        assert session.frame_count == 3
        assert session.last_frame == bytes([4]) * 4

    def test_gpu_framebuffer(self):
        session = MockStreamSession()
        session.start(StreamConfig())
        buf = GPUFramebuffer(cuda_ptr=0x1000, width=1920, height=1080,
                             stride_bytes=1920 * 4)
        session.update_framebuffer_gpu(buf)
        assert len(session._gpu_frames) == 1
        assert session._gpu_frames[0].cuda_ptr == 0x1000

    def test_reconfigure_bitrate(self):
        session = MockStreamSession()
        config = StreamConfig(width=1280, height=720, bitrate_kbps=2000)
        session.start(config)
        new_config = StreamConfig(width=1280, height=720, bitrate_kbps=6000)
        session.reconfigure(new_config)
        assert session.config.bitrate_kbps == 6000

    def test_reconfigure_resolution_raises(self):
        session = MockStreamSession()
        session.start(StreamConfig(width=1280, height=720))
        with pytest.raises(StreamReconfigError, match="Resolution change"):
            session.reconfigure(StreamConfig(width=1920, height=1080))

    def test_reconfigure_before_start_raises(self):
        session = MockStreamSession()
        with pytest.raises(StreamReconfigError, match="not started"):
            session.reconfigure(StreamConfig())

    def test_set_input_handler(self):
        session = MockStreamSession()
        received = []
        session.set_input_handler(lambda evt: received.append(evt))
        assert session._input_handler is not None

    def test_properties_before_start(self):
        session = MockStreamSession()
        assert session.frame_count == 0
        assert session.last_frame is None
        assert session.config is None
        assert session.info is None
        assert not session.is_alive()
