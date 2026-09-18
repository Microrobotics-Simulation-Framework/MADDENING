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
                "resolution_status": "open",
            },
            {
                "anomaly_id": "MADD-ANO-002",
                "title": "Test anomaly 2",
                "description": "Another test anomaly",
                "severity": "minor",
                "safety_relevance": "not_safety_relevant",
                "safety_relevance_rationale": "Not safety relevant",
                "resolution_status": "resolved",
                "resolution_version": "0.4.0",
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
                "resolution_status": "open",
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
                "resolution_status": "open",
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
            "resolution_status": "open",
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
        assert len(errors) == 2, errors  # Both anomalies fail prefix check

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


class TestResolutionStatus:
    """The one field an IEC 62304 known-anomalies list is read for."""

    def _one(self, tmp_path, **overrides):
        anomaly = {
            "anomaly_id": "MADD-ANO-001",
            "title": "Test",
            "description": "Test",
            "severity": "major",
            "safety_relevance": "context_dependent",
            "safety_relevance_rationale": "Test",
            "resolution_status": "open",
        }
        anomaly.update(overrides)
        data = {
            "schema_version": "1.0",
            "generated_date": "2026-03-12",
            "anomalies": [anomaly],
        }
        path = tmp_path / "registry.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        return str(path)

    def test_a_free_text_resolution_status_is_rejected(self, tmp_path):
        path = self._one(tmp_path, resolution_status="probably fine tbh")
        errors = validate_anomaly_registry(path)
        assert any("invalid resolution_status" in e for e in errors), errors

    def test_every_resolution_status_enum_value_is_accepted(self, tmp_path):
        """The enum and the YAML schema must not drift apart."""
        from maddening.core.compliance.anomaly import ResolutionStatus

        for member in ResolutionStatus:
            path = self._one(tmp_path, resolution_status=member.value)
            assert validate_anomaly_registry(path) == [], member

    def test_an_unknown_field_is_rejected(self, tmp_path):
        path = self._one(tmp_path, workaroud="dropped by a typo")
        errors = validate_anomaly_registry(path)
        assert any("workaroud" in e for e in errors), errors

    def test_an_unknown_top_level_field_is_rejected(self, tmp_path):
        data = {
            "schema_version": "1.0",
            "generated_date": "2026-03-12",
            "anomalis": [],
            "anomalies": [],
        }
        path = tmp_path / "registry.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        errors = validate_anomaly_registry(str(path))
        assert any("anomalis" in e for e in errors), errors

    def test_resolve_references_can_be_turned_off_for_a_foreign_registry(
        self, tmp_path
    ):
        path = self._one(
            tmp_path, affected_components=["not_a_real_package.Thing"]
        )
        assert validate_anomaly_registry(path) != []
        assert validate_anomaly_registry(path, resolve_references=False) == []


class TestActualRegistryReferences:
    """Every symbol and test path the repo registry names must still exist."""

    REPO_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )
    REGISTRY = os.path.join(
        REPO_ROOT, "docs", "validation", "known_anomalies.yaml"
    )

    def test_repo_registry_components_and_verification_resolve(self):
        if not os.path.exists(self.REGISTRY):
            pytest.skip("known_anomalies.yaml not found")
        errors = validate_anomaly_registry(
            self.REGISTRY, prefix="MADD-ANO-", repo_root=self.REPO_ROOT
        )
        assert errors == [], f"Registry reference errors: {errors}"

    def test_the_registry_actually_names_components_to_resolve(self):
        """Guards against the gate going green because it checks nothing."""
        if not os.path.exists(self.REGISTRY):
            pytest.skip("known_anomalies.yaml not found")
        with open(self.REGISTRY) as f:
            data = yaml.safe_load(f)
        components = [
            c for a in data["anomalies"] for c in (a.get("affected_components") or [])
        ]
        verification = [
            v for a in data["anomalies"] for v in (a.get("verification") or [])
        ]
        assert len(components) >= 5
        assert len(verification) >= 5


class TestSharedResolver:
    """check_impl_mapping.py and the registry share this resolver."""

    def test_an_inherited_attribute_is_refused_when_own_is_required(self):
        from maddening.compliance._validate import resolve_dotted_name

        qname = "maddening.nodes.heat.HeatNode.to_dict"
        assert resolve_dotted_name(qname).ok
        strict = resolve_dotted_name(qname, require_own=True)
        assert not strict.ok
        assert strict.inherited_from == "SimulationNode"

    def test_a_non_callable_is_refused_when_callable_is_required(self):
        from maddening.compliance._validate import resolve_dotted_name

        qname = "maddening.nodes.heat.HeatNode.__doc__"
        assert resolve_dotted_name(qname).ok
        assert not resolve_dotted_name(qname, require_callable=True).ok

    def test_a_renamed_test_is_caught_even_though_the_file_exists(self):
        from maddening.compliance._validate import resolve_test_reference

        root = TestActualRegistryReferences.REPO_ROOT
        real = "tests/compliance/test_validator.py::test_repo_registry_is_valid"
        assert resolve_test_reference(real, root) is None
        renamed = "tests/compliance/test_validator.py::test_renamed_away"
        assert resolve_test_reference(renamed, root) is not None
