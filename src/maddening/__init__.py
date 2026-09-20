"""
MADDENING - Modular Automatic Differentiation and Data Enhanced
             Neural-network INteracting Graph

A JAX-based modular simulation framework for multi-physics.

Install extras for optional features::

    pip install maddening[viz]       # matplotlib renderers
    pip install maddening[terminal]  # rich terminal renderer
    pip install maddening[network]   # ZMQ remote transport
    pip install maddening[all]       # everything
    pip install maddening[client]    # viz-only (no JAX needed)
"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # PEP 562: the names below are resolved lazily by the module
    # `__getattr__`, which a type checker cannot see through -- without
    # these re-exports it reports every one of them as missing from
    # `__all__`, and a downstream `from maddening import X` is an error.
    # Nothing here runs at import time; the lazy table stays the only
    # runtime path.
    from maddening.cloud.session import CloudConfig, CloudSession
    from maddening.core.coupling import CouplingGroup
    from maddening.core.edge import EdgeSpec
    from maddening.core.graph_manager import GraphManager
    from maddening.core.node import SimulationNode
    from maddening.core.simulation.adaptive import AdaptiveConfig
    from maddening.core.simulation.history_logger import HistoryLogger
    from maddening.surrogates.architecture import SurrogateArchitecture
    from maddening.surrogates.node import SurrogateNode


try:
    __version__ = _pkg_version("maddening")
except PackageNotFoundError:  # source tree without install metadata
    __version__ = "0.4.0.dev0"


def __getattr__(name: str) -> Any:
    """Lazy imports so that ``maddening.viz`` can be used without JAX."""
    _lazy = {
        "AdaptiveConfig": "maddening.core.simulation.adaptive",
        "CouplingGroup": "maddening.core.coupling",
        "EdgeSpec": "maddening.core.edge",
        "GraphManager": "maddening.core.graph_manager",
        "HistoryLogger": "maddening.core.simulation.history_logger",
        "SimulationNode": "maddening.core.node",
        "SurrogateNode": "maddening.surrogates.node",
        "SurrogateArchitecture": "maddening.surrogates.architecture",
        "CloudSession": "maddening.cloud.session",
        "CloudConfig": "maddening.cloud.session",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'maddening' has no attribute {name!r}")


__all__ = [
    "AdaptiveConfig",
    "CouplingGroup",
    "EdgeSpec",
    "GraphManager",
    "HistoryLogger",
    "SimulationNode",
    "SurrogateNode",
    "SurrogateArchitecture",
    "CloudSession",
    "CloudConfig",
]
