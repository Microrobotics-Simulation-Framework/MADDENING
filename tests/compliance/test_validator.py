"""Tests for the anomaly registry validator."""

import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import yaml

from maddening.compliance._validate import validate_anomaly_registry


@pytest.fixture
def valid_registry(tmp_path):
    """Create a minimal valid known_anomalies.yaml."""
    data = {
        "schema_version": "1.0",
        "maddening_version": "0.1.0",
        "generated_date": "2026-03-12",
        "anomalies": [],
    }
    path = tmp_path / "known_anomalies.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


@pytest.fixture
def registry_with_anomalies(tmp_path):
    """Create a registry with two valid anomalies."""
    data = {
        "schema_version": "1.0",
        "maddening_version": "0.1.0",
        "generated_date": "2026-03-12",
        "anomalies": [
            {
                "anomaly_id": "MADD-ANO-001",
                "title": "Test anomaly 1",
                "description": "A test anomaly",
                "severity": "major",
                "safety_relevance": "context_dependent",
                "safety_relevance_rationale": "Depends on context",
            },
            {
                "anomaly_id": "MADD-ANO-002",
                "title": "Test anomaly 2",
                "description": "Another test anomaly",
                "severity": "minor",
                "safety_relevance": "not_safety_relevant",
                "safety_relevance_rationale": "Not safety relevant",
            },
        ],
    }
    path = tmp_path / "known_anomalies.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


class TestValidRegistry:
    def test_empty_anomalies_list(self, valid_registry):
        errors = validate_anomaly_registry(valid_registry)
        assert errors == []

    def test_with_anomalies(self, registry_with_anomalies):
        errors = validate_anomaly_registry(registry_with_anomalies)
        assert errors == []


class TestMissingFields:
    def test_missing_schema_version(self, tmp_path):
        data = {"generated_date": "2026-03-12", "anomalies": []}
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        errors = validate_anomaly_registry(str(path))
        assert any("schema_version" in e for e in errors)

    def test_missing_generated_date(self, tmp_path):
        data = {"schema_version": "1.0", "anomalies": []}
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        errors = validate_anomaly_registry(str(path))
        assert any("generated_date" in e for e in errors)

    def test_missing_anomalies_key(self, tmp_path):
        data = {"schema_version": "1.0", "generated_date": "2026-03-12"}
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        errors = validate_anomaly_registry(str(path))
        assert any("anomalies" in e for e in errors)


class TestAnomalyValidation:
    def test_missing_required_field(self, tmp_path):
        data = {
            "schema_version": "1.0",
            "generated_date": "2026-03-12",
            "anomalies": [{
                "anomaly_id": "MADD-ANO-001",
                "title": "Test",
                # missing description, severity, safety_relevance, safety_relevance_rationale
            }],
        }
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        errors = validate_anomaly_registry(str(path))
        assert len(errors) >= 3  # description, severity, safety_relevance, rationale

    def test_invalid_severity(self, tmp_path):
        data = {
            "schema_version": "1.0",
            "generated_date": "2026-03-12",
            "anomalies": [{
                "anomaly_id": "MADD-ANO-001",
                "title": "Test",
                "description": "Test",
                "severity": "invalid_value",
                "safety_relevance": "context_dependent",
                "safety_relevance_rationale": "Test",
            }],
        }
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        errors = validate_anomaly_registry(str(path))
        assert any("invalid severity" in e for e in errors)

    def test_invalid_safety_relevance(self, tmp_path):
        data = {
            "schema_version": "1.0",
            "generated_date": "2026-03-12",
            "anomalies": [{
                "anomaly_id": "MADD-ANO-001",
                "title": "Test",
                "description": "Test",
                "severity": "major",
                "safety_relevance": "invalid_value",
                "safety_relevance_rationale": "Test",
            }],
        }
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        errors = validate_anomaly_registry(str(path))
        assert any("invalid safety_relevance" in e for e in errors)

    def test_duplicate_anomaly_id(self, tmp_path):
        anomaly = {
            "anomaly_id": "MADD-ANO-001",
            "title": "Test",
            "description": "Test",
            "severity": "major",
            "safety_relevance": "context_dependent",
            "safety_relevance_rationale": "Test",
        }
        data = {
            "schema_version": "1.0",
            "generated_date": "2026-03-12",
            "anomalies": [anomaly, anomaly],
        }
        path = tmp_path / "bad.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        errors = validate_anomaly_registry(str(path))
        assert any("Duplicate" in e for e in errors)


class TestPrefixEnforcement:
    def test_matching_prefix(self, registry_with_anomalies):
        errors = validate_anomaly_registry(registry_with_anomalies, prefix="MADD-ANO-")
        assert errors == []

    def test_wrong_prefix(self, registry_with_anomalies):
        errors = validate_anomaly_registry(registry_with_anomalies, prefix="MIME-ANO-")
        assert len(errors) == 2  # Both anomalies fail prefix check

    def test_no_prefix_accepts_all(self, registry_with_anomalies):
        errors = validate_anomaly_registry(registry_with_anomalies, prefix="")
        assert errors == []


class TestActualRegistry:
    """Validate the actual known_anomalies.yaml in the repo."""

    def test_repo_registry_is_valid(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "docs", "validation", "known_anomalies.yaml"
        )
        if not os.path.exists(path):
            pytest.skip("known_anomalies.yaml not found")
        errors = validate_anomaly_registry(path)
        assert errors == [], f"Registry validation errors: {errors}"

    def test_repo_registry_uses_madd_prefix(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "docs", "validation", "known_anomalies.yaml"
        )
        if not os.path.exists(path):
            pytest.skip("known_anomalies.yaml not found")
        errors = validate_anomaly_registry(path, prefix="MADD-ANO-")
        assert errors == [], f"Prefix errors: {errors}"
