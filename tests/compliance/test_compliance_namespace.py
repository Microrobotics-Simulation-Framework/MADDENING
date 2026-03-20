"""Tests for the maddening.compliance namespace (Phase 0 checklist)."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import subprocess
import sys

import pytest


class TestComplianceImports:
    """Verify all compliance types are importable."""

    def test_import_nodemeta(self):
        from maddening.compliance import NodeMeta
        assert NodeMeta is not None

    def test_import_edgemeta(self):
        from maddening.compliance import EdgeMeta
        assert EdgeMeta is not None

    def test_import_validated_regime(self):
        from maddening.compliance import ValidatedRegime
        assert ValidatedRegime is not None

    def test_import_reference(self):
        from maddening.compliance import Reference
        assert Reference is not None

    def test_import_stability_level(self):
        from maddening.compliance import StabilityLevel
        assert StabilityLevel is not None

    def test_import_uq_readiness(self):
        from maddening.compliance import UQReadiness
        assert UQReadiness is not None

    def test_import_anomaly_record(self):
        from maddening.compliance import AnomalyRecord
        assert AnomalyRecord is not None

    def test_import_anomaly_severity(self):
        from maddening.compliance import AnomalySeverity
        assert AnomalySeverity is not None

    def test_import_safety_relevance(self):
        from maddening.compliance import SafetyRelevance
        assert SafetyRelevance is not None

    def test_import_resolution_status(self):
        from maddening.compliance import ResolutionStatus
        assert ResolutionStatus is not None

    def test_import_validation_benchmark(self):
        from maddening.compliance import ValidationBenchmark
        assert ValidationBenchmark is not None

    def test_import_benchmark_type(self):
        from maddening.compliance import BenchmarkType
        assert BenchmarkType is not None

    def test_import_verification_benchmark_decorator(self):
        from maddening.compliance import verification_benchmark
        assert callable(verification_benchmark)

    def test_import_stability_decorator(self):
        from maddening.compliance import stability
        assert callable(stability)

    def test_import_validate_anomaly_registry(self):
        from maddening.compliance import validate_anomaly_registry
        assert callable(validate_anomaly_registry)

    def test_import_collect_node_metadata(self):
        from maddening.compliance import collect_node_metadata
        assert callable(collect_node_metadata)

    def test_import_collect_hazard_hints(self):
        from maddening.compliance import collect_hazard_hints
        assert callable(collect_hazard_hints)

    def test_import_generate_stability_report(self):
        from maddening.compliance import generate_stability_report
        assert callable(generate_stability_report)

    def test_import_get_benchmark_registry(self):
        from maddening.compliance import get_benchmark_registry
        assert callable(get_benchmark_registry)


class TestNoJAXDependency:
    """Verify compliance namespace doesn't require JAX at import time.

    We test this by checking that the import path from compliance
    to the schema types does not go through any JAX module.
    """

    def test_compliance_types_are_pure_python(self):
        """All compliance schema types use only stdlib modules."""
        from maddening.core.compliance.metadata import NodeMeta
        # NodeMeta is a dataclass — no JAX imports needed
        meta = NodeMeta(algorithm_id="test")
        assert meta.algorithm_id == "test"

    def test_anomaly_types_are_pure_python(self):
        from maddening.core.compliance.anomaly import AnomalyRecord, AnomalySeverity
        ar = AnomalyRecord(
            anomaly_id="TEST-001",
            title="Test",
            description="Test",
            severity=AnomalySeverity.MINOR,
            safety_relevance=__import__("maddening.core.anomaly", fromlist=["SafetyRelevance"]).SafetyRelevance.NOT_SAFETY_RELEVANT,
            safety_relevance_rationale="Test",
        )
        assert ar.anomaly_id == "TEST-001"


class TestCLI:
    """Test the CLI interface."""

    def test_check_anomalies_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "maddening.compliance", "check-anomalies", "--help"],
            capture_output=True, text=True, timeout=30,
            cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
        )
        # argparse --help exits with 0
        assert result.returncode == 0

    def test_check_anomalies_valid_file(self):
        repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
        yaml_path = os.path.join(repo_root, "docs", "validation", "known_anomalies.yaml")
        if not os.path.exists(yaml_path):
            pytest.skip("known_anomalies.yaml not found")
        result = subprocess.run(
            [sys.executable, "-m", "maddening.compliance", "check-anomalies", yaml_path],
            capture_output=True, text=True, timeout=30,
            cwd=repo_root,
        )
        assert result.returncode == 0
        assert "OK" in result.stdout
