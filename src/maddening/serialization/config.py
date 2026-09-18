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

    This captures the graph *structure* (node descriptors + edges,
    edge mappings as their ``MappingSpec``, coupling groups with every
    field of their solver configuration), **not** runtime state: node
    states and mapping weights go in checkpoints.
    """
    return graph_manager.to_dict()


def from_dict(
    config: dict,
    node_registry: dict[str, type],
    *,
    base_dir=None,
) -> "GraphManager":
    """Reconstruct a :class:`GraphManager` from a serialised config.

    *node_registry* maps type-name strings (e.g. ``"BallNode"``)
    to the corresponding Python class.  *base_dir* is the directory
    the config was read from: edge mappings saved with
    ``{"asset": "<file>.npy"}`` point references load their arrays
    relative to it (see :mod:`maddening.core.coupling.mapping_spec`).

    A config written before a given key existed loads as though the
    graph had none of it, so an older file is still a valid input.
    """
    from maddening.core.graph_manager import GraphManager
    return GraphManager.from_dict(config, node_registry, base_dir=base_dir)
