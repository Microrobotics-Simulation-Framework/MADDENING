"""Tests for streaming ABCs and data types."""

import pytest

from maddening.cloud.streaming import (
    GPUFramebuffer,
    QualityPreset,
    StreamConfig,
    StreamInfo,
    StreamReconfigError,
    StreamStartError,
    StreamingSession,
)


class TestQualityPreset:
    def test_values(self):
        assert QualityPreset.PREVIEW.value == "preview"
        assert QualityPreset.STANDARD.value == "standard"
        assert QualityPreset.CAPTURE.value == "capture"

    def test_from_string(self):
        assert QualityPreset("preview") is QualityPreset.PREVIEW


class TestStreamConfig:
    def test_defaults(self):
        cfg = StreamConfig()
        assert cfg.width == 1280
        assert cfg.height == 720
        assert cfg.fps == 30
        assert cfg.codec == "h264"

    def test_from_preset_preview(self):
        cfg = StreamConfig.from_preset(QualityPreset.PREVIEW)
        assert cfg.width == 854
        assert cfg.height == 480
        assert cfg.fps == 24
        assert cfg.bitrate_kbps == 1500

    def test_from_preset_standard(self):
        cfg = StreamConfig.from_preset(QualityPreset.STANDARD)
        assert cfg.width == 1280
        assert cfg.height == 720

    def test_from_preset_capture(self):
        cfg = StreamConfig.from_preset(QualityPreset.CAPTURE)
        assert cfg.width == 1920
        assert cfg.height == 1080
        assert cfg.fps == 60

    def test_from_preset_with_overrides(self):
        cfg = StreamConfig.from_preset(QualityPreset.STANDARD, fps=60)
        assert cfg.fps == 60
        assert cfg.width == 1280  # inherited from preset

    def test_from_dict(self):
        d = {"width": 640, "height": 480, "fps": 15, "bitrate_kbps": 1000,
             "codec": "vp9", "pixel_format": "RGBA", "enable_audio": False,
             "ice_servers": []}
        cfg = StreamConfig.from_dict(d)
        assert cfg.width == 640
        assert cfg.codec == "vp9"

    def test_frozen(self):
        cfg = StreamConfig()
        with pytest.raises(AttributeError):
            cfg.width = 640  # type: ignore[misc]


class TestStreamInfo:
    def test_fields(self):
        info = StreamInfo(
            session_id="abc",
            signaling_url="ws://localhost:8080/signaling/abc",
            stream_url="http://localhost:8080/stream/abc",
            ice_servers=[{"urls": ["stun:stun.l.google.com:19302"]}],
        )
        assert info.session_id == "abc"
        assert info.control_endpoint == ""  # default

    def test_frozen(self):
        info = StreamInfo(
            session_id="x", signaling_url="ws://x",
            stream_url="http://x", ice_servers=[],
        )
        with pytest.raises(AttributeError):
            info.session_id = "y"  # type: ignore[misc]


class TestGPUFramebuffer:
    def test_fields(self):
        buf = GPUFramebuffer(
            cuda_ptr=0xDEADBEEF, width=1920, height=1080,
            stride_bytes=1920 * 4,
        )
        assert buf.cuda_ptr == 0xDEADBEEF
        assert buf.pixel_format == "RGBA"
        assert buf.cuda_stream == 0
        assert buf.fence_value == 0


class TestErrors:
    def test_stream_start_error(self):
        with pytest.raises(StreamStartError):
            raise StreamStartError("test")

    def test_stream_reconfig_error(self):
        with pytest.raises(StreamReconfigError):
            raise StreamReconfigError("test")


class TestStreamingSessionABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            StreamingSession()  # type: ignore[abstract]
