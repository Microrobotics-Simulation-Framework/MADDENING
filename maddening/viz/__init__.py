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
]

# Network transport (ZMQ) -- imported explicitly to avoid hard dep on pyzmq
# when only local visualization is needed:
#   from maddening.viz.network import NetworkRelay, NetworkReceiver

# Backends are imported explicitly from maddening.viz.backends to avoid
# pulling in matplotlib/terminal deps when only the core viz API is needed.
