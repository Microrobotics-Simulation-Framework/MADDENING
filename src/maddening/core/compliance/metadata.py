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
    """API stability classification for nodes and public surfaces.

    Levels (per ``plans/MADDENING_v0.3.0_PLAN.md`` §A2):

    - ``STABLE``: locked at v1.0.0; backwards-incompatible changes require
      a major version bump.
    - ``EVOLVING``: settled wire format / signature, but additions are
      possible without breaking existing callers.
    - ``PROVISIONAL``: synonym for ``EVOLVING`` retained for back-compat
      with pre-v0.3.0 tagged surfaces.
    - ``EXPERIMENTAL``: may break in any minor release. Opt-in only.
    - ``INTERNAL``: not part of the public API. Implementation detail.
    - ``DEPRECATED``: scheduled for removal in a future release.
    """
    EXPERIMENTAL = "experimental"
    PROVISIONAL = "provisional"
    EVOLVING = "evolving"
    STABLE = "stable"
    INTERNAL = "internal"
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
class DiscretizationOrder:
    """The order of accuracy a node's scheme is *claimed* to achieve.

    This is the theoretical order the discretisation was designed for,
    not a measured one.  Declaring it is what lets
    :func:`maddening.testing.mms.verify_node_order` turn a per-node
    convergence study into a parametrised test: the harness refines the
    grid (or the timestep), measures the rate at which the error falls,
    and fails when the measurement falls short of what is declared
    here.  A node that declares nothing is *skipped* explicitly by that
    harness rather than silently passing it.

    Order is the claim a verification measurement can actually falsify.
    An absolute error tolerance cannot: a wrong stencil weight, a
    mishandled boundary or an off-by-one in a flux usually leaves the
    error looking perfectly acceptable on the one grid a threshold test
    runs on, and shows up only as order 1 where order 2 was claimed.

    Attributes
    ----------
    spatial : float or None
        Order in the grid spacing ``h``: the error should fall like
        ``O(h**spatial)`` under spatial refinement at fixed timestep.
        ``None`` for a node with no spatial discretisation (an ODE node
        such as :class:`~maddening.nodes.ball.BallNode`).
    temporal : float or None
        Order in the timestep ``dt``.  ``None`` if the node does not
        integrate in time.
    notes : str
        Where the claim comes from, and any caveat that bounds it: a
        boundary closure of lower order than the interior stencil, a
        regime in which the order degrades, the reference the scheme is
        taken from.

    Examples
    --------
    >>> DiscretizationOrder(spatial=2.0, temporal=1.0, notes="central FD, forward Euler")
    DiscretizationOrder(spatial=2.0, temporal=1.0, notes='central FD, forward Euler')

    A node whose order depends on how it was constructed declares the
    default here and overrides the instance hook the harness prefers::

        def discretization_order(self):
            return DiscretizationOrder(
                spatial=float(self.params["stencil_order"]), temporal=1.0,
            )
    """
    spatial: Optional[float] = None
    temporal: Optional[float] = None
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

    # Order of accuracy the scheme claims (prose lives in
    # ``discretization``; this is the machine-readable claim the MMS
    # harness measures against).  ``None`` means the node has not
    # declared one, and ``maddening.testing.mms`` skips it explicitly
    # rather than passing it silently.
    discretization_order: Optional[DiscretizationOrder] = None

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
