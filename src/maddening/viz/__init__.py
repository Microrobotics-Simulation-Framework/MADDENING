"""
MADDENING visualization subsystem.

Provides a modular, thread-safe pipeline for real-time visualization of
running simulations.
"""

from maddening.viz.renderer import Renderer, GraphInfo
from maddening.viz.relay import StateRelay
from maddening.viz.runner import RealtimeRunner

__all__ = [
    "Renderer",
    "GraphInfo",
    "StateRelay",
    "RealtimeRunner",
    "HistoryViewer3D",
    "GPUHistoryViewer",
]

# Network transport (ZMQ) -- imported explicitly to avoid hard dep on pyzmq
# when only local visualization is needed:
#   from maddening.viz.network import NetworkRelay, NetworkReceiver

# Backends are imported explicitly from maddening.viz.backends to avoid
# pulling in matplotlib/terminal deps when only the core viz API is needed.

# 3D history viewer (PyVista) -- imported explicitly to avoid hard dep:
#   from maddening.viz.history_viewer import HistoryViewer3D


def __getattr__(name):
    if name == "HistoryViewer3D":
        from maddening.viz.history_viewer import HistoryViewer3D
        return HistoryViewer3D
    if name == "GPUHistoryViewer":
        from maddening.viz.backends.pygfx_viewer import GPUHistoryViewer
        return GPUHistoryViewer
    if name == "viewer_from_usd":
        from maddening.viz.usd_viewer import viewer_from_usd
        return viewer_from_usd
    if name == "viewer_from_usd_with_geometry":
        from maddening.viz.usd_viewer import viewer_from_usd_with_geometry
        return viewer_from_usd_with_geometry
    if name == "render_usd_frame":
        from maddening.viz.usd_viewer import render_usd_frame
        return render_usd_frame
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
