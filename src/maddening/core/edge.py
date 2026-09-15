"""
EdgeSpec -- an immutable description of a data dependency between two nodes.

An edge says: "before updating *target_node*, copy
*source_node.state[source_field]* into boundary_inputs[target_field],
optionally applying *transform* first."

``transform``, if provided, must be a JAX-traceable pure function.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability


@stability(StabilityLevel.STABLE)
@dataclass(frozen=True)
class EdgeSpec:
    source_node: str
    target_node: str
    source_field: str
    target_field: str
    transform: Optional[Callable] = None
    additive: bool = False  # If True, ADD to existing boundary_input value
    source_units: Optional[str] = None  # Physical units of the source field
    target_units: Optional[str] = None  # Physical units after transform
    # Interface mapping (``maddening.core.coupling.mapping.Mapping``)
    # applied before ``transform``; its weights live in
    # ``GraphManager.params["mappings"][edge.key]``.
    mapping: Optional[Any] = None

    @property
    def key(self) -> str:
        """Stable identifier: ``"<src>.<field>-><tgt>.<field>"``."""
        return (f"{self.source_node}.{self.source_field}->"
                f"{self.target_node}.{self.target_field}")

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "source_node": self.source_node,
            "target_node": self.target_node,
            "source_field": self.source_field,
            "target_field": self.target_field,
        }
        if self.mapping is not None:
            describe = getattr(self.mapping, "describe", None)
            d["mapping"] = describe() if callable(describe) else {
                "kind": getattr(self.mapping, "kind", type(self.mapping).__name__),
            }
        if self.transform is not None:
            d["transform"] = self.transform.__qualname__
        if self.additive:
            d["additive"] = True
        if self.source_units is not None:
            d["source_units"] = self.source_units
        if self.target_units is not None:
            d["target_units"] = self.target_units
        return d

    def __repr__(self) -> str:
        arrow = f"{self.source_node}.{self.source_field} -> {self.target_node}.{self.target_field}"
        if self.mapping is not None:
            arrow += f"  (mapping {self.mapping!r})"
        if self.transform is not None:
            arrow += f"  (via {self.transform.__qualname__})"
        if self.source_units or self.target_units:
            arrow += f"  [{self.source_units or '?'} -> {self.target_units or '?'}]"
        return f"EdgeSpec({arrow})"
