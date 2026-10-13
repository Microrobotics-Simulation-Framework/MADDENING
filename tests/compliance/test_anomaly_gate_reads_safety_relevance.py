"""``scripts/check_anomalies.py`` checks every entry's ``safety_relevance``.

The gate ran the compliance validator, which compares the field against a set
it spells out for itself; only a unit test of the validator pinned that set to
:class:`~maddening.core.compliance.anomaly.SafetyRelevance`, and a YAML list in
the field crashed the gate with a ``TypeError`` traceback (and the validator
with it) instead of naming the entry.  The gate now reads the field against
the enum itself, and the validator reports an unhashable value instead of
raising.  Each test plants a bad value and asserts the gate fails on it.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "scripts" / "check_anomalies.py"
REGISTRY = REPO_ROOT / "docs" / "validation" / "known_anomalies.yaml"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("_gate_check_anomalies_sr", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(value, aid="MADD-ANO-001"):
    e = {"anomaly_id": aid}
    if value is not _MISSING:
        e["safety_relevance"] = value
    return e


_MISSING = object()


@pytest.mark.parametrize("value", ["safety_relevant", "not_safety_relevant", "context_dependent"])
def test_every_enum_value_passes(gate, value):
    assert gate.safety_relevance_errors([_entry(value)]) == []


@pytest.mark.parametrize("value", ["contextual", "Context_Dependent", "", ["context_dependent"],
                                   {"v": 1}, True, 1, None, _MISSING])
def test_anything_else_is_a_finding_that_names_the_entry(gate, value):
    errors = gate.safety_relevance_errors([_entry(value, "MADD-ANO-042")])
    assert len(errors) == 1 and errors[0].startswith("MADD-ANO-042: safety_relevance")


def test_the_gate_reads_the_enum_not_the_validators_copy(gate, monkeypatch):
    """A value the validator's spelled-out set has drifted to admit is still refused."""
    from maddening.compliance import _validate

    monkeypatch.setattr(_validate, "_VALID_SAFETY_RELEVANCES",
                        _validate._VALID_SAFETY_RELEVANCES | {"contextual"})
    assert gate.safety_relevance_errors([_entry("contextual")]) != []


def _registry_with(tmp_path, value):
    """The shipped registry with one entry's ``safety_relevance`` replaced."""
    text = REGISTRY.read_text(encoding="utf-8")
    old = 'safety_relevance: "context_dependent"'
    assert old in text
    path = tmp_path / "known_anomalies.yaml"
    path.write_text(text.replace(old, f"safety_relevance: {value}", 1), encoding="utf-8")
    return path


def _run_gate(path):
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"), JAX_PLATFORMS="cpu")
    return subprocess.run(
        [sys.executable, str(GATE), str(path), "--repo-root", str(REPO_ROOT), "--no-resolve"],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT))


@pytest.mark.parametrize("value", ['"contextual"', '["context_dependent"]'])
def test_the_gate_fails_on_a_planted_value_and_says_which_entry(tmp_path, value):
    result = _run_gate(_registry_with(tmp_path, value))
    out = result.stdout + result.stderr
    assert result.returncode == 1, out
    assert "is not one of the SafetyRelevance values" in out
    assert "Traceback" not in out


def test_the_validator_reports_an_unhashable_enum_value_instead_of_raising(tmp_path):
    from maddening.compliance._validate import validate_anomaly_registry

    doc = {"schema_version": "1.0", "anomalies": [{
        "anomaly_id": "MADD-ANO-001", "title": "t", "description": "d",
        "severity": ["minor"], "safety_relevance": ["context_dependent"],
        "safety_relevance_rationale": "r", "resolution_status": {"s": 1}}]}
    path = tmp_path / "r.yaml"
    path.write_text(yaml.safe_dump(doc))
    errors = validate_anomaly_registry(str(path), resolve_references=False)
    for field in ("severity", "safety_relevance", "resolution_status"):
        assert any(f"invalid {field}" in e for e in errors), (field, errors)
