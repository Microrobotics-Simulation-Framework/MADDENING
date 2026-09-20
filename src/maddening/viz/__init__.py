"""
MADDENING visualization subsystem.

Provides a modular, thread-safe pipeline for real-time visualization of
running simulations.
"""

from typing import TYPE_CHECKING, Any

from maddening.viz.renderer import Renderer, GraphInfo
from maddening.viz.relay import StateRelay
from maddening.viz.runner import RealtimeRunner

if TYPE_CHECKING:
    # PEP 562: the names below are resolved lazily by the module
    # `__getattr__`, which a type checker cannot see through -- without
    # these re-exports it reports every one of them as missing from
    # `__all__`, and a downstream `from maddening.viz import X` is an error.
    # Nothing here runs at import time; the lazy table stays the only
    # runtime path.
    from maddening.viz.backends.pygfx_viewer import GPUHistoryViewer
    from maddening.viz.backends.pyvista_live import PyVistaLiveRenderer
    from maddening.viz.history_viewer import HistoryViewer3D


__all__ = [
    "Renderer",
    "GraphInfo",
    "StateRelay",
    "RealtimeRunner",
    "HistoryViewer3D",
    "GPUHistoryViewer",
    "PyVistaLiveRenderer",
]

# Network transport (ZMQ) -- imported explicitly to avoid hard dep on pyzmq
# when only local visualization is needed:
#   from maddening.viz.network import NetworkRelay, NetworkReceiver

# Backends are imported explicitly from maddening.viz.backends to avoid
# pulling in matplotlib/terminal deps when only the core viz API is needed.

# 3D history viewer (PyVista) -- imported explicitly to avoid hard dep:
#   from maddening.viz.history_viewer import HistoryViewer3D


_INSTALL_HINTS = {
    "HistoryViewer3D": "viz3d",
    "PyVistaLiveRenderer": "viz3d",
    "GPUHistoryViewer": "gpu-viz",
    "viewer_from_usd": "usd",
    "viewer_from_usd_with_geometry": "usd",
    "render_usd_frame": "usd",
}


def __getattr__(name: str) -> Any:
    _lazy = {
        "HistoryViewer3D": ("maddening.viz.history_viewer", "HistoryViewer3D"),
        "GPUHistoryViewer": ("maddening.viz.backends.pygfx_viewer", "GPUHistoryViewer"),
        "viewer_from_usd": ("maddening.viz.usd_viewer", "viewer_from_usd"),
        "viewer_from_usd_with_geometry": ("maddening.viz.usd_viewer", "viewer_from_usd_with_geometry"),
        "render_usd_frame": ("maddening.viz.usd_viewer", "render_usd_frame"),
        "PyVistaLiveRenderer": ("maddening.viz.backends.pyvista_live", "PyVistaLiveRenderer"),
    }
    if name in _lazy:
        mod_path, attr = _lazy[name]
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            return getattr(mod, attr)
        except ImportError as exc:
            extra = _INSTALL_HINTS.get(name, "")
            raise ImportError(
                f"'{name}' requires additional dependencies. "
                f"Install with:  pip install maddening[{extra}]"
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
