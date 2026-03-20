"""MADDENING core — graph manager, nodes, edges, coupling, simulation utilities.

All public names are re-exported here for backward compatibility.
New code should prefer the subpackage paths::

    from maddening.core.coupling.group import CouplingGroup
    from maddening.core.simulation.adaptive import AdaptiveConfig
    from maddening.core.compliance.metadata import NodeMeta

But the flat imports still work::

    from maddening.core import CouplingGroup, AdaptiveConfig
"""

from maddening.core.simulation.adaptive import AdaptiveConfig
from maddening.core.simulation.checkpoint import save_state, load_state
from maddening.core.coupling import CouplingGroup
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.simulation.history_logger import HistoryLogger
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.transforms import register_transform, resolve_transform
from maddening.core.schedule import (
    detect_cycles,
    find_strongly_connected_components,
    topological_sort,
)

__all__ = [
    "AdaptiveConfig",
    "BoundaryInputSpec",
    "CouplingGroup",
    "EdgeSpec",
    "GraphManager",
    "HistoryLogger",
    "SimulationNode",
    "detect_cycles",
    "find_strongly_connected_components",
    "load_state",
    "register_transform",
    "resolve_transform",
    "save_state",
    "topological_sort",
]
