"""A registry's top level carries exactly one project version field.

The rule (``maddening.compliance._validate``): besides ``schema_version``,
``generated_date`` and ``anomalies``, a registry carries exactly one key named
``<project>_version`` -- the whole key matching ``[a-z][a-z0-9_]*_version``
and not ``schema_version`` -- whose value is a non-empty string.
``version_field=`` (``--version-field NAME`` on the CLI) pins the exact name.

The registry schema is shared with downstream libraries, each of which keeps
its own registry under its own version field, so the validator must accept
any project's field and still refuse every key that is not one.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import maddening
from maddening.compliance._validate import (
    _validate_registry_document,
    validate_anomaly_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
#: The ``src`` directory of the package under test, so the CLI subprocess
#: imports this tree and not whatever else is installed.
_SRC = str(Path(maddening.__file__).resolve().parents[1])

_ANOMALY = {
    "anomaly_id": "LIB-ANO-001",
    "title": "T",
    "description": "D",
    "severity": "minor",
    "safety_relevance": "not_safety_relevant",
    "safety_relevance_rationale": "R",
    "resolution_status": "open",
}


def _registry(tmp_path, **top):
    """A registry whose top level is the three fixed keys plus ``top``."""
    data = {
        "schema_version": "1.0",
        "generated_date": "2026-03-12",
        "anomalies": [dict(_ANOMALY)],
    }
    data.update(top)
    path = tmp_path / "known_anomalies.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def _raw(tmp_path, text):
    path = tmp_path / "known_anomalies.yaml"
    path.write_text(text)
    return str(path)


def _validate(path, **kwargs):
    return validate_anomaly_registry(path, resolve_references=False, **kwargs)


class TestTheVersionFieldNamesAnyProject:
    @pytest.mark.parametrize("key", [
        "maddening_version",
        "mime_version",
        "my_lib_version",    # a two-word project
        "lib2_version",      # digits after the first letter
        "a_version",         # the shortest project name
    ])
    def test_one_well_named_field_is_accepted(self, tmp_path, key):
        assert _validate(_registry(tmp_path, **{key: "0.1.0"})) == []

    @pytest.mark.parametrize("key", [
        "maddening_version", "mime_version", "my_lib_version",
    ])
    def test_the_same_field_is_accepted_when_it_is_required_by_name(
        self, tmp_path, key
    ):
        path = _registry(tmp_path, **{key: "0.1.0"})
        assert _validate(path, version_field=key) == []

    def test_the_default_is_any_one_field(self, tmp_path):
        """Today's callers pass no ``version_field``; their registries pass."""
        path = _registry(tmp_path, maddening_version="0.4.0.dev0")
        assert _validate(path) == _validate(path, version_field=None) == []


class TestExactlyOne:
    def test_no_version_field_is_an_error_that_says_what_to_add(self, tmp_path):
        errors = _validate(_registry(tmp_path))
        assert len(errors) == 1, errors
        assert errors[0].startswith("Missing project version field"), errors
        assert "<project>_version" in errors[0]
        assert "[a-z][a-z0-9_]*" in errors[0]
        assert "add one" in errors[0]

    def test_schema_version_alone_is_not_a_project_version_field(
        self, tmp_path
    ):
        """``schema_version`` matches the pattern and is excluded by name."""
        errors = _validate(_registry(tmp_path))
        assert any(e.startswith("Missing project version field")
                   for e in errors), errors
        assert not any("Unknown top-level field" in e for e in errors), errors

    def test_two_version_fields_are_an_error_naming_both(self, tmp_path):
        path = _registry(tmp_path, maddening_version="0.4.0",
                         my_lib_version="0.1.0")
        errors = _validate(path)
        assert len(errors) == 1, errors
        assert "2 project version fields found" in errors[0]
        assert "(maddening_version, my_lib_version)" in errors[0]
        assert "exactly one" in errors[0]

    def test_three_version_fields_are_all_named(self, tmp_path):
        path = _registry(tmp_path, a_version="1", b_version="2",
                         c_version="3")
        errors = _validate(path)
        assert len(errors) == 1, errors
        assert "3 project version fields found (a_version, b_version, " \
               "c_version)" in errors[0]

    def test_two_fields_are_refused_even_when_one_is_the_required_one(
        self, tmp_path
    ):
        path = _registry(tmp_path, maddening_version="0.4.0",
                         my_lib_version="0.1.0")
        errors = _validate(path, version_field="my_lib_version")
        assert len(errors) == 1, errors
        assert "2 project version fields found" in errors[0]
        assert "'my_lib_version' is required here" in errors[0]


class TestTheNameIsChecked:
    @pytest.mark.parametrize("key", [
        "_version",          # no project
        "version",           # no project and no separator
        "Mime_version",      # uppercase
        "MIME_VERSION",
        "mime_Version",
        "my-lib_version",    # hyphenated
        "9lib_version",      # does not start with a letter
        "my_lib_version2",   # the suffix must end the key
        "my_lib_versions",
        "mylibversion",
    ])
    def test_a_malformed_name_is_an_unknown_field(self, tmp_path, key):
        errors = _validate(_registry(tmp_path, **{key: "0.1.0"}))
        unknown = [e for e in errors if "Unknown top-level field" in e]
        assert len(unknown) == 1 and f"field: {key}" in unknown[0], errors
        # ... and it does not stand in for the field the registry lacks.
        assert any(e.startswith("Missing project version field")
                   for e in errors), errors

    def test_a_malformed_name_beside_a_good_one_is_only_unknown(
        self, tmp_path
    ):
        """It is not counted as a second version field."""
        path = _registry(tmp_path, maddening_version="0.4.0",
                         **{"My-Lib_version": "0.1.0"})
        errors = _validate(path)
        assert len(errors) == 1, errors
        assert "Unknown top-level field: My-Lib_version" in errors[0]
        assert "not a valid project version field name" in errors[0]

    def test_an_unknown_key_is_refused_with_the_known_keys_and_the_rule(
        self, tmp_path
    ):
        path = _registry(tmp_path, maddening_version="0.4.0", anomalis=[])
        errors = _validate(path)
        assert len(errors) == 1, errors
        assert errors[0].startswith("Unknown top-level field: anomalis "), errors
        assert "anomalies, generated_date, schema_version" in errors[0]
        assert "exactly one project version field" in errors[0]
        assert "'<project>_version'" in errors[0]
        assert "not a valid project version field name" not in errors[0]

    def test_a_non_string_key_is_reported_not_raised(self, tmp_path):
        """Sorting a mix of key types raised ``TypeError``."""
        path = _raw(tmp_path, (
            'schema_version: "1.0"\nmy_lib_version: "0.1.0"\n'
            'generated_date: "2026-03-12"\n1: x\nnull: y\nfoo: z\n'
            'anomalies:\n'
            '  - anomaly_id: LIB-ANO-001\n    title: T\n    description: D\n'
            '    severity: minor\n    safety_relevance: not_safety_relevant\n'
            '    safety_relevance_rationale: R\n    resolution_status: open\n'
            '    2: stray\n    extra: stray\n'
        ))
        errors = _validate(path)
        assert sorted(e.split(" (known")[0] for e in errors
                      if "top-level" in e) == [
            "Unknown top-level field: 1",
            "Unknown top-level field: None",
            "Unknown top-level field: foo",
        ], errors
        assert "LIB-ANO-001: unknown field '2'" in errors, errors
        assert "LIB-ANO-001: unknown field 'extra'" in errors, errors


    @pytest.mark.parametrize("raw_id", ["[LIB-ANO-001]", "{id: LIB-ANO-001}"])
    def test_an_unhashable_anomaly_id_is_reported_not_raised(
        self, tmp_path, raw_id
    ):
        """A list or mapping id raised ``TypeError`` at the duplicate check."""
        path = _raw(tmp_path, (
            'schema_version: "1.0"\nmy_lib_version: "0.1.0"\n'
            'generated_date: "2026-03-12"\nanomalies:\n'
            f'  - anomaly_id: {raw_id}\n    title: T\n    description: D\n'
            '    severity: minor\n    safety_relevance: not_safety_relevant\n'
            '    safety_relevance_rationale: R\n    resolution_status: open\n'
        ))
        errors = _validate(path, prefix="LIB-ANO-")
        assert any(e.startswith("anomaly_id must be a string, found ")
                   for e in errors), errors
        assert any("does not match required prefix" in e
                   for e in errors), errors


class TestVersionFieldOption:
    def test_a_different_field_is_an_error_naming_both(self, tmp_path):
        path = _registry(tmp_path, mime_version="0.1.0")
        errors = _validate(path, version_field="maddening_version")
        assert errors == [
            "Project version field is 'mime_version' but 'maddening_version' "
            "is required (version_field / --version-field): rename the key, "
            "or require the field this registry is meant to carry"
        ]

    def test_a_missing_field_names_the_required_one(self, tmp_path):
        errors = _validate(_registry(tmp_path), version_field="my_lib_version")
        assert len(errors) == 1, errors
        assert "'my_lib_version' is required here" in errors[0]
        assert "my_lib_version: \"0.1.0\"" in errors[0]

    @pytest.mark.parametrize("name", [
        "schema_version", "version", "_version", "Mime_version",
        "my-lib_version", "", "my_lib_version2",
    ])
    def test_a_malformed_required_name_is_refused_loudly(self, tmp_path, name):
        path = _registry(tmp_path, my_lib_version="0.1.0")
        with pytest.raises(ValueError, match="not a project version field"):
            _validate(path, version_field=name)


class TestTheValueIsANonEmptyString:
    @pytest.mark.parametrize("value", [
        0.1, 1, None, "", "   ", True, ["0.1.0"], {"v": "0.1.0"},
    ])
    def test_a_non_string_or_empty_value_is_an_error(self, tmp_path, value):
        errors = _validate(_registry(tmp_path, my_lib_version=value))
        assert len(errors) == 1, errors
        assert errors[0].startswith(
            "Project version field 'my_lib_version' must be a non-empty "
            "string"), errors

    def test_an_unquoted_version_is_caught_and_the_fix_is_named(
        self, tmp_path
    ):
        """YAML reads ``0.10`` as the float ``0.1``: the version is lost."""
        path = _raw(tmp_path, (
            'schema_version: "1.0"\nmy_lib_version: 0.10\n'
            'generated_date: "2026-03-12"\nanomalies: []\n'
        ))
        errors = _validate(path)
        assert len(errors) == 1, errors
        assert "found 0.1 (float)" in errors[0]
        assert "quote it" in errors[0]

    def test_the_value_is_checked_when_the_name_is_required_too(
        self, tmp_path
    ):
        path = _registry(tmp_path, my_lib_version=3)
        errors = _validate(path, version_field="my_lib_version")
        assert len(errors) == 1 and "non-empty string" in errors[0], errors

    def test_a_version_that_is_not_pep_440_is_accepted_here(self, tmp_path):
        """PEP 440 is held for MADDENING's own registry by its own gate."""
        path = _registry(tmp_path, my_lib_version="2026.10-rc1+local build")
        assert _validate(path) == []


class TestMaddeningsOwnRegistry:
    def test_the_shipped_registry_carries_maddening_version(self):
        path = REPO_ROOT / "docs" / "validation" / "known_anomalies.yaml"
        assert _validate(str(path), prefix="MADD-ANO-",
                         version_field="maddening_version") == []

    def test_the_gate_requires_maddening_version(self, tmp_path, monkeypatch):
        """``scripts/check_anomalies.py`` passes the name explicitly."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_gate_check_anomalies_version_field",
            REPO_ROOT / "scripts" / "check_anomalies.py",
        )
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        seen = []

        def recorder(path, document, **kwargs):
            seen.append(kwargs.get("version_field"))
            return _validate_registry_document(path, document, **kwargs)

        # The gate loads the registry once and hands the document to the
        # validator's document entry point (``validate_anomaly_registry``
        # minus the load).
        monkeypatch.setattr(gate, "_validate_registry_document", recorder)
        path = _registry(tmp_path, my_lib_version="0.1.0")
        assert gate.main([path, "--no-resolve"]) == 1
        assert seen == ["maddening_version"]


def _cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [_SRC] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return subprocess.run(
        [sys.executable, "-m", "maddening.compliance", "check-anomalies",
         *args],
        capture_output=True, text=True, timeout=60, env=env,
    )


class TestTheCommandLine:
    """``python -m maddening.compliance check-anomalies``, as a user runs it.

    The registries here name no components or tests, so nothing is resolved.
    """

    def test_a_downstream_registry_passes_without_the_option(self, tmp_path):
        result = _cli(_registry(tmp_path, my_lib_version="0.1.0"),
                      "--prefix", "LIB-ANO-")
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_the_option_accepts_the_field_the_registry_carries(self, tmp_path):
        result = _cli(_registry(tmp_path, my_lib_version="0.1.0"),
                      "--prefix", "LIB-ANO-", "--version-field",
                      "my_lib_version")
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout

    def test_the_option_refuses_another_field(self, tmp_path):
        result = _cli(_registry(tmp_path, my_lib_version="0.1.0"),
                      "--version-field", "maddening_version")
        assert result.returncode == 1, result.stdout
        assert ("ERROR: Project version field is 'my_lib_version' but "
                "'maddening_version' is required") in result.stderr
        assert "Traceback" not in result.stderr

    def test_a_registry_without_a_version_field_fails(self, tmp_path):
        result = _cli(_registry(tmp_path))
        assert result.returncode == 1, result.stdout
        assert "ERROR: Missing project version field" in result.stderr

    def test_two_version_fields_fail(self, tmp_path):
        result = _cli(_registry(tmp_path, maddening_version="0.4.0",
                                my_lib_version="0.1.0"))
        assert result.returncode == 1, result.stdout
        assert "2 project version fields found" in result.stderr

    @pytest.mark.parametrize("name", ["schema_version", "Lib_version"])
    def test_a_malformed_option_is_a_usage_error(self, tmp_path, name):
        result = _cli(_registry(tmp_path, my_lib_version="0.1.0"),
                      "--version-field", name)
        assert result.returncode == 2, result.stdout + result.stderr
        assert "argument --version-field" in result.stderr
        assert "not a project version field name" in result.stderr
        assert "Traceback" not in result.stderr

    def test_help_documents_the_option_and_the_rule(self):
        result = _cli("--help")
        assert result.returncode == 0, result.stderr
        text = " ".join(result.stdout.split())
        assert "--version-field NAME" in text
        assert "<project>_version" in text
