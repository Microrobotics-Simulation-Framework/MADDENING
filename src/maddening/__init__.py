"""
MADDENING - Modular Automatic Differentiation and Data-Enhanced
             Neural-network INteracting Graph

A JAX-based modular simulation framework for multi-physics.

Install extras for optional features::

    pip install maddening[viz]       # matplotlib renderers
    pip install maddening[terminal]  # rich terminal renderer
    pip install maddening[network]   # ZMQ remote transport
    pip install maddening[all]       # everything
    pip install maddening[client]    # viz-only (no JAX needed)
"""


def __getattr__(name: str):
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
