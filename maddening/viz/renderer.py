"""
Renderer ABC and GraphInfo dataclass.

``GraphInfo`` is a read-only snapshot of the simulation graph metadata,
passed to renderers so they can set up axes, labels, etc. without
holding a reference to the mutable ``GraphManager``.

``Renderer`` defines the interface every visualization backend must
implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class GraphInfo:
    """Read-only metadata about the simulation graph, passed to renderers."""

    node_names: list[str]
    node_params: dict[str, dict]
    node_state_fields: dict[str, list[str]]
    edges: list[dict]
    timestep: float

    @classmethod
    def from_graph_manager(cls, gm) -> "GraphInfo":
        """Build a ``GraphInfo`` from a live ``GraphManager`` instance."""
        node_names = list(gm._nodes.keys())
        node_params = {
            name: dict(spec.node.params)
            for name, spec in gm._nodes.items()
        }
        node_state_fields = {
            name: list(gm._state[name].keys())
            for name in gm._nodes
        }
        edges = [e.to_dict() for e in gm._edges]
        timestep = gm.timestep
        return cls(
            node_names=node_names,
            node_params=node_params,
            node_state_fields=node_state_fields,
            edges=edges,
            timestep=timestep,
        )


class Renderer(ABC):
    """Abstract base for all visualization backends."""

    @abstractmethod
    def setup(self, graph_info: GraphInfo) -> None:
        """Initialize given graph metadata.  Called once before sim starts."""
        ...

    @abstractmethod
    def update(self, sim_time: float, state: dict[str, dict]) -> None:
        """Process a new state snapshot.  Called at display rate."""
        ...

    @abstractmethod
    def teardown(self) -> None:
        """Release resources."""
        ...

    def requested_fields(self) -> Optional[dict[str, list[str]]]:
        """Optional: declare which (node, fields) this renderer needs.

        Returns ``{node_name: [field1, ...]}`` or ``None`` for everything.
        """
        return None
