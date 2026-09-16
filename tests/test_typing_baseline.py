"""scripts/typing_baseline.py summarises a pyright --outputjson document
correctly without pyright being installed."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "typing_baseline.py"


@pytest.fixture(scope="module")
def tb():
    spec = importlib.util.spec_from_file_location("typing_baseline", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves the defining module through sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _diag(file, severity, rule, message="m"):
    d = {"file": file, "severity": severity, "message": message,
         "range": {"start": {"line": 0, "character": 0},
                   "end": {"line": 0, "character": 1}}}
    if rule is not None:
        d["rule"] = rule
    return d


@pytest.fixture
def report(tb):
    root = tb.REPO_ROOT
    a = str(root / "src" / "maddening" / "a.py")
    b = str(root / "src" / "maddening" / "sub" / "b.py")
    return {
        "version": "1.1.414",
        "generalDiagnostics": [
            _diag(a, "error", "reportAttributeAccessIssue"),
            _diag(a, "error", "reportAttributeAccessIssue"),
            _diag(a, "warning", "reportMissingImports"),
            _diag(b, "error", "reportCallIssue"),
            _diag(b, "error", None),
            _diag("/elsewhere/c.py", "warning", "reportMissingImports"),
        ],
        "summary": {"filesAnalyzed": 3, "errorCount": 4, "warningCount": 2,
                    "informationCount": 0, "timeInSec": 0.1},
    }


def test_totals_and_rule_counts_match_diagnostics(tb, report):
    s = tb.summarise(report)
    assert s.files_analyzed == 3
    assert s.error_count == 4
    assert s.warning_count == 2
    assert s.by_rule[("error", "reportAttributeAccessIssue")] == 2
    assert s.by_rule[("warning", "reportMissingImports")] == 2
    assert s.by_rule[("error", "reportCallIssue")] == 1
    assert s.by_rule[("error", "<no rule>")] == 1
    assert s.source_files_with_diagnostics == 3


def test_per_file_errors_are_relative_to_repo_root(tb, report):
    s = tb.summarise(report)
    assert s.errors_by_file == {"src/maddening/a.py": 2,
                                "src/maddening/sub/b.py": 2}
    # A path outside the root is kept as-is rather than raising.
    outside = tb.summarise({"generalDiagnostics": [
        _diag("/elsewhere/c.py", "error", "r")], "summary": {}})
    assert outside.errors_by_file == {"/elsewhere/c.py": 1}


def test_render_sorts_rules_by_count_and_limits_files(tb, report):
    text = tb.render(tb.summarise(report), top=1)
    rules = [l for l in text.splitlines() if l.startswith(("error", "warning"))]
    assert rules[0].split() == ["error", "reportAttributeAccessIssue", "2"]
    assert "src/maddening/a.py" in text
    assert "src/maddening/sub/b.py" not in text  # top=1 keeps a.py only (ties broken by name)


def test_markdown_output_uses_tables(tb, report):
    md = tb.render(tb.summarise(report), markdown=True)
    assert "### pyright summary" in md
    assert "| severity | rule | count |" in md
    assert "| 2 | src/maddening/a.py |" in md


def test_main_reads_stored_json_and_fail_flag_sets_exit_status(tb, report, tmp_path, capsys):
    path = tmp_path / "out.json"
    path.write_text(json.dumps(report))
    assert tb.main(["--json", str(path)]) == 0
    assert "errors: 4" in capsys.readouterr().out
    assert tb.main(["--json", str(path), "--fail-on-errors"]) == 1
    path.write_text(json.dumps({"generalDiagnostics": [], "summary": {}}))
    assert tb.main(["--json", str(path), "--fail-on-errors"]) == 0
