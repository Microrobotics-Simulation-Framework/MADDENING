"""
Uncertainty quantification interface hooks (Section 9.4).

Provides ``UncertaintySpec`` and ``UncertainParameter`` dataclasses
that nodes can use to declare their UQ capabilities.

Pure Python — no JAX dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DistributionType(Enum):
    """Supported probability distribution types for UQ."""
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOG_NORMAL = "log_normal"
    TRUNCATED_NORMAL = "truncated_normal"


@dataclass(frozen=True)
class UncertainParameter:
    """Declares that a node parameter supports uncertainty quantification.

    Attributes
    ----------
    name : str
        Parameter name (must match a key in ``node.params``).
    distribution : DistributionType
        Default distribution type for sampling.
    nominal : float
        Nominal (baseline) value.
    lower_bound : float or None
        Lower bound for sampling.
    upper_bound : float or None
        Upper bound for sampling.
    units : str
        Physical units.
    description : str
        Human-readable description.
    """
    name: str
    distribution: DistributionType = DistributionType.UNIFORM
    nominal: float = 0.0
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    units: str = ""
    description: str = ""


@dataclass(frozen=True)
class UncertaintySpec:
    """UQ specification for a node, returned by ``node.uncertainty_spec()``.

    Attributes
    ----------
    parameters : tuple of UncertainParameter
        Parameters that support UQ.
    notes : str
        Any caveats about UQ support for this node.
    """
    parameters: tuple[UncertainParameter, ...] = ()
    notes: str = ""
