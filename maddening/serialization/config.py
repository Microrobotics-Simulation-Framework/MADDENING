"""
Serialization helpers for graph structure.

Produces / consumes plain dicts that are JSON / YAML compatible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maddening.core.graph_manager import GraphManager


def to_dict(graph_manager: "GraphManager") -> dict:
    """Serialise *graph_manager* to a JSON-compatible dict.

    This captures the graph *structure* (node descriptors + edges),
    **not** runtime state.
    """
    return graph_manager.to_dict()


def from_dict(
    config: dict,
    node_registry: dict[str, type],
) -> "GraphManager":
    """Reconstruct a :class:`GraphManager` from a serialised config.

    *node_registry* maps type-name strings (e.g. ``"BallNode"``)
    to the corresponding Python class.
    """
    from maddening.core.graph_manager import GraphManager
    return GraphManager.from_dict(config, node_registry)
