"""
Compliance schema and tooling for MADDENING and downstream libraries.

This module re-exports all data structures, decorators, and utilities
that downstream libraries need to participate in MADDENING's compliance
architecture.  Import from here — not from maddening.core internals.

Usage (from a downstream library like MIME)::

    from maddening.compliance import (
        # Metadata schemas
        NodeMeta, EdgeMeta, ValidatedRegime, Reference,
        StabilityLevel, UQReadiness,
        # Anomaly management
        AnomalyRecord, AnomalySeverity, SafetyRelevance, ResolutionStatus,
        # Validation infrastructure
        ValidationBenchmark, BenchmarkType, verification_benchmark,
        # Stability
        stability,
        # Harvesting
        collect_node_metadata, collect_hazard_hints,
        # Registry validation
        validate_anomaly_registry,
    )

No JAX dependency — everything here is pure Python.
"""

# Metadata schemas
from maddening.core.metadata import (
    NodeMeta,
    EdgeMeta,
    ValidatedRegime,
    Reference,
    StabilityLevel,
    UQReadiness,
    collect_node_metadata,
    collect_hazard_hints,
)

# Anomaly management
from maddening.core.anomaly import (
    AnomalyRecord,
    AnomalySeverity,
    SafetyRelevance,
    ResolutionStatus,
)

# Validation infrastructure
from maddening.core.validation import (
    ValidationBenchmark,
    BenchmarkType,
    verification_benchmark,
    get_benchmark_registry,
)

# Stability — identity decorator in Phase 0-3, functional in Phase 4.
# See DOCUMENTATION_ARCHITECTURE.md Section 9.5 and Appendix B item 30.
from maddening.core.stability import stability, generate_stability_report

# Registry validation
from maddening.compliance._validate import validate_anomaly_registry

__all__ = [
    # Metadata
    "NodeMeta", "EdgeMeta", "ValidatedRegime", "Reference",
    "StabilityLevel", "UQReadiness",
    # Anomaly
    "AnomalyRecord", "AnomalySeverity", "SafetyRelevance", "ResolutionStatus",
    # Validation
    "ValidationBenchmark", "BenchmarkType", "verification_benchmark",
    "get_benchmark_registry",
    # Stability
    "stability", "generate_stability_report",
    # Harvesting
    "collect_node_metadata", "collect_hazard_hints",
    # Registry validation
    "validate_anomaly_registry",
]
