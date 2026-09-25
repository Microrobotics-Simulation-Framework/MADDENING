"""
Anomaly management schema types (Section 9.7).

Pure Python — no JAX dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AnomalySeverity(Enum):
    """Anomaly severity classification.

    This is the one definition of the four levels.  ``known_anomalies.yaml``,
    the SOUP package and the ``anomaly:*`` issue labels all use it, and
    DOCUMENTATION_ARCHITECTURE.md section 9.7 repeats it.  Severity rates
    the defect: a ``resolved`` entry keeps the severity of what it records,
    and it is independent of ``safety_relevance``, which rates how much
    the defect matters in a given context of use.

    Two terms carry the definitions.  A *wrong result* is a value the
    library returns, reports or saves that describes the simulated system
    or vouches for it -- a state, an output, a gradient, a convergence flag
    or error estimate, the reported status of an operation it performed, a
    saved or reloaded configuration -- and that differs from what the
    library documents by more than the accuracy the library states for it
    (a scheme's declared order, a solve's tolerance).  A result is *silent*
    when the library raised no exception and emitted no warning when it
    produced it.  A field that later goes non-finite was still silent: its
    finite values before that were wrong and nothing said so.

    **The rule for silent wrong results: a silent wrong result is never
    minor**, whatever its size, however narrow the configuration that
    reaches it, and whether or not a built-in node reaches it.  Size, reach
    and detectability belong in the entry's description and
    ``safety_relevance_rationale``, where a reader can weigh them; they do
    not lower the severity below ``major``.

    *Unauthorised access* is a party the user did not authorise reading or
    changing the simulation, its inputs or its files, or running code, by a
    defect of the library.

    CRITICAL
        A silent wrong result that a shipped default configuration reaches,
        that the user cannot detect from anything the library returns, and
        that no workaround avoids short of not using the feature.
        Unauthorised access through a surface the library opens (a
        listening socket, an HTTP route) that a shipped default
        configuration exposes to other hosts.  Code execution by an
        unauthorised party through such a surface, in any configuration.
    MAJOR
        Any other silent wrong result.  Any other unauthorised access: a
        surface that only a configuration the user chose exposes to other
        hosts (a loopback default is not such an exposure), or an API that
        executes or trusts what it is handed while presenting itself as
        safe for data.  And a loud failure -- an exception, a refusal, a
        crash -- whose only workaround is not to use the feature.
    MINOR
        A loud failure with a workaround that keeps the feature usable.  A
        wrong statement in documentation or declared metadata where the
        computed values still meet the accuracy the library states for
        them.  A defect in performance, placement or a count of work done
        (iterations, passes, where a graph runs) that changes no value
        describing or vouching for the solution.  A difference that appears
        only under ``jax_enable_x64`` and leaves the result no less accurate
        than the framework's default float32 would compute it.  A cosmetic
        defect.
    ENHANCEMENT
        Not a defect: a request for behaviour the library never claimed.
    """
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
