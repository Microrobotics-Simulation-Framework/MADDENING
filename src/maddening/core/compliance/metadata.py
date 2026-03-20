"""
NodeMeta and related schema types for compliance and documentation.

All types in this module are pure Python dataclasses/enums with no JAX
dependency, so they can be imported from ``maddening.compliance`` without
installing JAX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StabilityLevel(Enum):
    """API stability classification for nodes and public surfaces."""
    EXPERIMENTAL = "experimental"
    PROVISIONAL = "provisional"
    STABLE = "stable"
    DEPRECATED = "deprecated"


class UQReadiness(Enum):
    """Uncertainty quantification readiness level."""
    NOT_READY = "not_ready"
    PARAMETER_SWEEP = "parameter_sweep"
    FULL = "full"


@dataclass(frozen=True)
class Reference:
    """A bibliographic reference (BibTeX key + human-readable description)."""
    key: str
    description: str = ""


@dataclass(frozen=True)
class ValidatedRegime:
    """A quantitative parameter regime within which the node has been verified.

    These are parameter-bound, quantitative risks: operating outside a
    validated regime means the model's behaviour is uncharacterised.
    """
    parameter: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    units: str = ""
    notes: str = ""


@dataclass(frozen=True)
class EdgeMeta:
    """Metadata for an edge (data coupling between nodes)."""
    description: str = ""
    units: str = ""
    physical_quantity: str = ""


@dataclass(frozen=True)
class NodeMeta:
    """Structured metadata for a SimulationNode.

    Provides the machine-readable information needed for IEC 62304 SOUP
    assessment, ISO 14971 hazard identification, algorithm documentation,
    and downstream compliance tooling.
    """
    # Identity
    algorithm_id: str = ""
    algorithm_version: str = "0.0.0"
    stability: StabilityLevel = StabilityLevel.EXPERIMENTAL

    # Documentation
    description: str = ""
    governing_equations: str = ""
    discretization: str = ""

    # Assumptions and limitations
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    # Validation
    validated_regimes: tuple[ValidatedRegime, ...] = ()

    # References
    references: tuple[Reference, ...] = ()

    # UQ
    uq_readiness: UQReadiness = UQReadiness.NOT_READY

    # Deprecation
    deprecation_notice: str = ""

    # ISO 14971 risk management (Section 9.8)
    hazard_hints: tuple[str, ...] = ()
    """Technical hazard hints for downstream ISO 14971 hazard identification.

    Each string describes a **technical condition** — numerical instability,
    unvalidated parameter regime, algorithmic limitation — that a risk
    manager should consider as a potential hazard contributor.  These are
    strictly technical hazard hints, NOT clinical risk assessments.

    MADDENING provides: technical conditions (e.g., "CFL > 1 causes
    numerical instability", "behaviour uncharacterised at Re > 100").

    MADDENING does NOT provide: clinical risk assessments.
    """

    # Implementation mapping (Phase 3 — Section 3)
    implementation_map: dict[str, str] = field(default_factory=dict)
    """Machine-readable mapping from equation term descriptions to Python
    function qualified names.  Used by Sphinx build verification and
    ``scripts/check_impl_mapping.py`` to detect documentation rot.
    """


# ---------------------------------------------------------------------------
# Harvesting utilities
# ---------------------------------------------------------------------------

def collect_node_metadata() -> dict[str, NodeMeta]:
    """Collect NodeMeta from all SimulationNode subclasses in the process."""
    from maddening.core.node import SimulationNode  # deferred to avoid cycle

    result = {}
    for cls in SimulationNode.__subclasses__():
        meta = getattr(cls, "meta", None)
        if meta is not None:
            result[cls.__name__] = meta
    return result


def collect_hazard_hints() -> dict[str, list[str]]:
    """Collect hazard_hints across all nodes for risk management input."""
    from maddening.core.node import SimulationNode

    result = {}
    for cls in SimulationNode.__subclasses__():
        meta = getattr(cls, "meta", None)
        if meta is not None and meta.hazard_hints:
            result[cls.__name__] = list(meta.hazard_hints)
    return result
