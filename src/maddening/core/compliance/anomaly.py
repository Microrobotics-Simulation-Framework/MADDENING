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
    a fix has landed, and it does not cover the whole defect.  Part of it
    is still reachable in the version the registry describes, and the
    entry's ``residual_risk`` field says which part and why.  So for every
    purpose that asks whether a user can meet the defect it counts as
    reachable, exactly as ``OPEN`` does: the SOUP package's reachable-defect
    count includes it, and ``scripts/check_anomalies.py`` requires its
    ``affected_versions`` range to admit the current version.  What sets it
    apart from ``OPEN`` is that the fix exists and its tests are cited under
    ``verification``.  MADD-ANO-005 (the criterion falls back to the
    pre-0.4.0 test on a reachable path), MADD-ANO-014 (the documentation is
    fixed, the default behaviour is not) and MADD-ANO-016 (the port is
    signature-correct and unobserved against a live provider) are examples.
    Until v0.4.0 the enum could not represent the registry MADDENING
    itself ships.
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
