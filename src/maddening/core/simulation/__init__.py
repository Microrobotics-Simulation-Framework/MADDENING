"""Simulation utilities — adaptive timestepping, checkpointing, integrators, etc.

Re-exports for backward compatibility::

    from maddening.core.simulation.adaptive import AdaptiveConfig   # new path
    from maddening.core.simulation.adaptive import AdaptiveConfig              # still works via core.__init__
"""

from maddening.core.simulation.adaptive import AdaptiveConfig
from maddening.core.simulation.checkpoint import save_state, load_state
from maddening.core.simulation.history_logger import HistoryLogger

__all__ = ["AdaptiveConfig", "save_state", "load_state", "HistoryLogger"]
