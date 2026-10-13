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

Resume-from-URL transport (fsspec only for cloud-storage schemes)::

    download_and_load_state
"""

# Eagerly import pure-Python types (stdlib only, no deps)
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    # PEP 562: the names below are resolved lazily by the module
    # `__getattr__`, which a type checker cannot see through -- without
    # these re-exports it reports every one of them as missing from
    # `__all__`, and a downstream `from maddening.cloud import X` is an error.
    # Nothing here runs at import time; the lazy table stays the only
    # runtime path.
    from maddening.cloud.launcher import (
        CloudJob,
        CloudLauncher,
        CostLimitError,
        CostPolicy,
        CredentialError,
        JobConfig,
        JobPhase,
        LaunchError,
    )
    from maddening.cloud.mock_session import MockCloudSession
    from maddening.cloud.mock_streaming import MockStreamSession
    from maddening.cloud.providers import (
        AWSProvider,
        CloudProvider,
        GCPProvider,
        LambdaLabsProvider,
        PROVIDERS,
        RunPodProvider,
    )
    from maddening.cloud.resume import download_and_load_state
    from maddening.cloud.selkies_session import SelkiesSession
    from maddening.cloud.session import (
        CloudConfig,
        CloudReadyResult,
        CloudSession,
        CloudSessionError,
        CloudSessionInfo,
        CloudStage,
        PreemptionPolicy,
    )



_INSTALL_HINTS = {
    "SelkiesSession": "streaming",
    "CloudSession": "cloud",
    "CloudConfig": "cloud",
    "CloudStage": "cloud",
    "CloudSessionInfo": "cloud",
    "CloudReadyResult": "cloud",
    "PreemptionPolicy": "cloud",
    "CloudSessionError": "cloud",
    "CloudLauncher": "runpod",
    "CloudJob": "runpod",
    "JobConfig": "runpod",
    "JobPhase": "runpod",
    "CostPolicy": "runpod",
    "CredentialError": "runpod",
    "CostLimitError": "runpod",
    "LaunchError": "runpod",
}


#: Lazy names -> defining module.  Resolved on first attribute access so
#: importing ``maddening.cloud`` never pulls in an optional dependency.
_LAZY = {
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
    # Launcher (requires skypilot)
    "CloudLauncher": "maddening.cloud.launcher",
    "CloudJob": "maddening.cloud.launcher",
    "JobConfig": "maddening.cloud.launcher",
    "JobPhase": "maddening.cloud.launcher",
    "CostPolicy": "maddening.cloud.launcher",
    "CredentialError": "maddening.cloud.launcher",
    "CostLimitError": "maddening.cloud.launcher",
    "LaunchError": "maddening.cloud.launcher",
    # Providers (no deps)
    "CloudProvider": "maddening.cloud.providers",
    "RunPodProvider": "maddening.cloud.providers",
    "LambdaLabsProvider": "maddening.cloud.providers",
    "AWSProvider": "maddening.cloud.providers",
    "GCPProvider": "maddening.cloud.providers",
    "PROVIDERS": "maddening.cloud.providers",
    # Multi-job (requires pyzmq for coordinator)
    "CloudGroup": "maddening.cloud.group",
    "GroupConfig": "maddening.cloud.group",
    "GroupFailureMode": "maddening.cloud.group",
    "SubgraphSpec": "maddening.cloud.group",
    "Coordinator": "maddening.cloud.multigpu.coordinator",
    # Resume-from-URL transport (imports JAX via core checkpoint; fsspec
    # is needed only at call time for the cloud-storage schemes)
    "download_and_load_state": "maddening.cloud.resume",
}


def __getattr__(name: str) -> Any:
    """Lazy imports for components that need external dependencies.

    Only a :class:`ModuleNotFoundError` for a module *outside*
    ``maddening`` is rewrapped with an install hint; any other
    ``ImportError`` (a broken module of ours, a circular import during
    development) propagates unchanged so the real cause is not hidden
    behind a wrong ``pip install`` suggestion.
    """
    if name in _LAZY:
        import importlib
        try:
            mod = importlib.import_module(_LAZY[name])
            return getattr(mod, name)
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if missing == "maddening" or missing.startswith("maddening."):
                raise
            extra = _INSTALL_HINTS.get(name, "cloud")
            raise ImportError(
                f"'{name}' requires the optional module {missing!r}. "
                f"Install with:  pip install maddening[{extra}]"
            ) from exc
    raise AttributeError(f"module 'maddening.cloud' has no attribute {name!r}")


def __dir__() -> list[str]:
    """``dir(maddening.cloud)`` lists the lazy names too."""
    return sorted(set(globals()) | set(__all__))


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
    "CloudLauncher",
    "CloudJob",
    "JobConfig",
    "JobPhase",
    "CostPolicy",
    "CredentialError",
    "CostLimitError",
    "LaunchError",
    "CloudProvider",
    "RunPodProvider",
    "LambdaLabsProvider",
    "AWSProvider",
    "GCPProvider",
    "PROVIDERS",
    "download_and_load_state",
]
