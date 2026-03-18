"""
Cloud module for MADDENING.

Provides WebRTC viewport streaming and cloud VM orchestration.

Pure-Python types (no external dependencies)::

    StreamConfig, StreamInfo, QualityPreset, GPUFramebuffer,
    StreamingSession, StreamStartError, StreamReconfigError

Cloud session (requires skypilot)::

    CloudSession, CloudConfig, CloudStage, CloudSessionInfo,
    CloudReadyResult, PreemptionPolicy, CloudSessionError

Mock implementations (zero deps, for testing)::

    MockStreamSession, MockCloudSession

GStreamer streaming (requires PyGObject + GStreamer)::

    SelkiesSession
"""

# Eagerly import pure-Python types (stdlib only, no deps)
from maddening.cloud.streaming import (
    GPUFramebuffer,
    QualityPreset,
    StreamConfig,
    StreamInfo,
    StreamReconfigError,
    StreamStartError,
    StreamingSession,
)
from maddening.cloud._auth import generate_session_token, validate_session_token


def __getattr__(name: str):
    """Lazy imports for components that need external dependencies."""
    _lazy = {
        # GStreamer streaming (requires PyGObject)
        "SelkiesSession": "maddening.cloud.selkies_session",
        # Cloud session (requires skypilot)
        "CloudSession": "maddening.cloud.session",
        "CloudConfig": "maddening.cloud.session",
        "CloudStage": "maddening.cloud.session",
        "CloudSessionInfo": "maddening.cloud.session",
        "CloudReadyResult": "maddening.cloud.session",
        "PreemptionPolicy": "maddening.cloud.session",
        "CloudSessionError": "maddening.cloud.session",
        # Mock implementations
        "MockStreamSession": "maddening.cloud.mock_streaming",
        "MockCloudSession": "maddening.cloud.mock_session",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'maddening.cloud' has no attribute {name!r}")


__all__ = [
    # Streaming types (eagerly imported)
    "GPUFramebuffer",
    "QualityPreset",
    "StreamConfig",
    "StreamInfo",
    "StreamReconfigError",
    "StreamStartError",
    "StreamingSession",
    "generate_session_token",
    "validate_session_token",
    # Lazy imports
    "SelkiesSession",
    "CloudSession",
    "CloudConfig",
    "CloudStage",
    "CloudSessionInfo",
    "CloudReadyResult",
    "PreemptionPolicy",
    "CloudSessionError",
    "MockStreamSession",
    "MockCloudSession",
]
