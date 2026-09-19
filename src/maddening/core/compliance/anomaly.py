"""
Anomaly management schema types (Section 9.7).

Pure Python — no JAX dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AnomalySeverity(Enum):
    """Anomaly severity classification."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    ENHANCEMENT = "enhancement"


class SafetyRelevance(Enum):
    """Safety relevance assessment for an anomaly."""
    SAFETY_RELEVANT = "safety_relevant"
    NOT_SAFETY_RELEVANT = "not_safety_relevant"
    CONTEXT_DEPENDENT = "context_dependent"


class ResolutionStatus(Enum):
    """Resolution status for an anomaly.

    ``PARTIALLY_RESOLVED`` is a real IEC 62304 state rather than a hedge:
    the defect is fixed in the released version but a bounded residual
    risk remains, recorded in the entry's ``residual_risk`` field.
    MADD-ANO-005 is one.  Until v0.4.0 the enum could not represent the
    registry MADDENING itself ships.
    """
    OPEN = "open"
    RESOLVED = "resolved"
    PARTIALLY_RESOLVED = "partially_resolved"
    WONT_FIX = "wont_fix"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class AnomalyRecord:
    """A single anomaly entry matching known_anomalies.yaml schema."""
    anomaly_id: str
    title: str
    description: str
    severity: AnomalySeverity
    safety_relevance: SafetyRelevance
    safety_relevance_rationale: str
    affected_components: tuple[str, ...] = ()
    affected_versions: str = ""
    workaround: str = ""
    resolution_status: ResolutionStatus = ResolutionStatus.OPEN
    resolution_version: str = ""
    github_issue: Optional[str] = None
