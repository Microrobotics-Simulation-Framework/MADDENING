"""scripts/typing_baseline.py summarises a pyright --outputjson document
correctly without pyright being installed, and refuses a run whose numbers
cannot be trusted (exit 2) instead of reporting a misleading count.

The failure paths are exercised through ``main`` against a fake pyright:
a tiny Python script written to ``tmp_path`` that prints canned JSON /
stderr and exits with a chosen code."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
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


def _missing(file, module):
    return _diag(file, "warning", "reportMissingImports",
                 f'Import "{module}" could not be resolved')


def _summary(files=3, errors=0, warnings=0):
    return {"filesAnalyzed": files, "errorCount": errors,
            "warningCount": warnings, "informationCount": 0, "timeInSec": 0.1}


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
            _missing(a, "pygfx"),
            _diag(b, "error", "reportCallIssue"),
            _diag(b, "error", None),
            _missing("/elsewhere/c.py", "gi.repository"),
        ],
        "summary": _summary(files=3, errors=4, warnings=2),
    }


def fake_pyright(tmp_path, stdout, stderr="", code=0, name="pyright"):
    """A `--pyright` command that prints canned output and exits `code`.

    It also records the arguments it was called with in ``<name>.argv``.
    """
    script = tmp_path / f"{name}.py"
    script.write_text(
        "import json, sys\n"
        f"open({str(tmp_path / (name + '.argv'))!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
        f"sys.stdout.write({stdout!r})\n"
        f"sys.stderr.write({stderr!r})\n"
        f"sys.exit({code})\n"
    )
    return f"{sys.executable} {script}"


def _argv(tmp_path, name="pyright"):
    return json.loads((tmp_path / f"{name}.argv").read_text())


# --- summarise / render ----------------------------------------------------


def test_totals_and_rule_counts_match_diagnostics(tb, report):
    s = tb.summarise(report)
    assert s.files_analyzed == 3
    assert s.has_summary_block
    assert s.error_count == 4
    assert s.warning_count == 2
    assert s.by_rule[("error", "reportAttributeAccessIssue")] == 2
    assert s.by_rule[("warning", "reportMissingImports")] == 2
    assert s.by_rule[("error", "reportCallIssue")] == 1
    assert s.by_rule[("error", "<no rule>")] == 1
    assert s.source_files_with_diagnostics == 3
    assert s.missing_imports == {"pygfx": 1, "gi.repository": 1}


def test_per_file_errors_are_relative_to_repo_root(tb, report):
    s = tb.summarise(report)
    assert s.errors_by_file == {"src/maddening/a.py": 2,
                                "src/maddening/sub/b.py": 2}
    # A path outside the root is kept as-is rather than raising.
    outside = tb.summarise({"generalDiagnostics": [
        _diag("/elsewhere/c.py", "error", "r")], "summary": {}})
    assert outside.errors_by_file == {"/elsewhere/c.py": 1}
    assert not outside.has_summary_block


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
    assert "| 2 | `src/maddening/a.py` |" in md


def test_markdown_file_cells_are_code_spans(tb):
    """`__init__.py` must not render as **init**.py and a pipe must not
    split the cell (S1)."""
    root = tb.REPO_ROOT
    s = tb.summarise({"generalDiagnostics": [
        _diag(str(root / "src/maddening/usd/__init__.py"), "error", "r"),
        _diag(str(root / "src/maddening/odd|name.py"), "error", "r"),
    ], "summary": _summary()})
    md = tb.render(s, markdown=True)
    assert "| 1 | `src/maddening/usd/__init__.py` |" in md
    assert "| 1 | `src/maddening/odd\\|name.py` |" in md
    text = tb.render(s)
    assert "`" not in text  # plain text is untouched


# --- main with a stored run -------------------------------------------------


def test_main_reads_stored_json_and_fail_flag_sets_exit_status(tb, report, tmp_path, capsys):
    path = tmp_path / "out.json"
    path.write_text(json.dumps(report))
    assert tb.main(["--json", str(path)]) == 0
    assert "errors: 4" in capsys.readouterr().out
    assert tb.main(["--json", str(path), "--fail-on-errors"]) == 1
    path.write_text(json.dumps({"generalDiagnostics": [], "summary": _summary()}))
    assert tb.main(["--json", str(path), "--fail-on-errors"]) == 0


def test_zero_files_analysed_is_an_infrastructure_failure_not_a_pass(tb, tmp_path, capsys):
    """pyright emits valid JSON with filesAnalyzed 0 and exit 0 when the
    include path does not exist; that must never read as "0 errors" (T1)."""
    empty = json.dumps({"version": "1.1.414", "generalDiagnostics": [],
                        "summary": _summary(files=0)})
    cmd = fake_pyright(tmp_path, empty, stderr='File or directory "src/missing" does not exist.\n')
    for flags in ([], ["--fail-on-errors"], ["--markdown", "--top", "15", "--fail-on-errors"]):
        assert tb.main(["--pyright", cmd, *flags]) == tb.EXIT_INFRASTRUCTURE
        out, err = capsys.readouterr()
        assert out == ""  # no table that could be mistaken for a result
        assert "analysed 0 files" in err
        assert "does not exist" in err  # pyright's own stderr is forwarded

    # The same document through --json, and a document without a summary block.
    path = tmp_path / "out.json"
    path.write_text(empty)
    assert tb.main(["--json", str(path)]) == tb.EXIT_INFRASTRUCTURE
    path.write_text(json.dumps({"generalDiagnostics": []}))
    assert tb.main(["--json", str(path), "--fail-on-errors"]) == tb.EXIT_INFRASTRUCTURE
    assert "no `summary` block" in capsys.readouterr().err

    # --min-files raises the floor; a checkout with fewer files is refused.
    path.write_text(json.dumps({"generalDiagnostics": [], "summary": _summary(files=3)}))
    assert tb.main(["--json", str(path), "--min-files", "100"]) == tb.EXIT_INFRASTRUCTURE
    assert tb.main(["--json", str(path), "--min-files", "3"]) == 0


def test_pyright_stderr_is_forwarded_and_config_error_exit_code_is_not_flattened(
        tb, report, tmp_path, capsys):
    """Valid JSON on stdout must not hide pyright's stderr; exit 3 (config
    could not be parsed) is an infrastructure failure, exit 1 is "errors
    found" (T2)."""
    doc = json.dumps(report)
    noise = 'Config file "pyrightconfig.json" could not be parsed.'
    cmd = fake_pyright(tmp_path, doc, stderr=noise, code=3)
    assert tb.main(["--pyright", cmd, "--fail-on-errors"]) == tb.EXIT_INFRASTRUCTURE
    out, err = capsys.readouterr()
    assert out == ""
    assert noise in err
    assert "exited 3" in err

    # exit 1 with a warning on stderr: analysis completed, errors counted,
    # stderr still visible.
    cmd = fake_pyright(tmp_path, doc, stderr="WARNING: something on stderr\n", code=1)
    assert tb.main(["--pyright", cmd, "--fail-on-errors"]) == tb.EXIT_ERRORS
    out, err = capsys.readouterr()
    assert "errors: 4" in out
    assert "WARNING: something on stderr" in err

    # exit 0 with no errors: the plain success path.
    clean = json.dumps({"generalDiagnostics": [], "summary": _summary()})
    cmd = fake_pyright(tmp_path, clean, code=0)
    assert tb.main(["--pyright", cmd, "--fail-on-errors"]) == tb.EXIT_OK

    # Non-JSON stdout (e.g. an npm bootstrap message) is infrastructure too.
    cmd = fake_pyright(tmp_path, "Installing pyright...\n", code=0)
    assert tb.main(["--pyright", cmd]) == tb.EXIT_INFRASTRUCTURE
    assert "did not produce JSON" in capsys.readouterr().err


def test_missing_import_surge_is_reported_as_environment_failure(tb, tmp_path, capsys):
    """An interpreter without the dependencies makes the error count *drop*
    (395 -> 230 in the audit) with empty stderr; a core import that does
    not resolve, or more missing imports than the ceiling, is exit 2 (T3)."""
    root = tb.REPO_ROOT
    f = str(root / "src" / "maddening" / "a.py")
    path = tmp_path / "out.json"

    def run(diags, *flags):
        path.write_text(json.dumps({"generalDiagnostics": diags, "summary": _summary()}))
        return tb.main(["--json", str(path), *flags])

    # The baseline shape: optional extras missing, under the ceiling.
    optional = [_missing(f, m) for m in ("pygfx", "gi.repository", "cupy", "fsspec")] * 4
    assert len(optional) == 16
    assert run(optional) == 0

    # One core dependency unresolved is enough, whatever the count.
    for mod in ("jax", "jax.numpy", "numpy", "yaml"):
        assert run([_missing(f, mod)]) == tb.EXIT_INFRASTRUCTURE
        err = capsys.readouterr().err
        assert "core imports" in err and mod in err and "--pythonpath" in err

    # Many unknown modules over the documented ceiling.
    flood = [_missing(f, f"ext{i}") for i in range(tb.DEFAULT_MAX_MISSING_IMPORTS + 1)]
    assert run(flood) == tb.EXIT_INFRASTRUCTURE
    assert "exceed --max-missing-imports 40" in capsys.readouterr().err
    assert run(flood[:-1]) == 0  # exactly at the ceiling is still accepted
    assert run(flood, "--max-missing-imports", "100") == 0

    # reportMissingModuleSource (stubs only) is not a resolution failure.
    stubs = [_diag(f, "warning", "reportMissingModuleSource",
                   'Import "yaml" could not be resolved from source')] * 50
    assert run(stubs) == 0


def test_pythonpath_interpreter_is_validated_before_pyright_runs(tb, report, tmp_path, capsys):
    """pyright silently accepts a `--pythonpath` that does not exist or has
    no numpy; the script checks it first and forwards it after `--outputjson`
    when it is good (T3)."""
    cmd = fake_pyright(tmp_path, json.dumps(report), code=1)
    marker = tmp_path / "pyright.argv"

    assert tb.main(["--pyright", cmd, "--", "--pythonpath", "/nonexistent/python"]) == tb.EXIT_INFRASTRUCTURE
    assert "no such interpreter" in capsys.readouterr().err
    assert not marker.exists()  # pyright was never invoked

    # Exists, runs, but cannot import numpy.
    bare = tmp_path / "bare_python"
    bare.write_text("#!/bin/sh\necho 'ModuleNotFoundError: numpy' >&2\nexit 1\n")
    bare.chmod(bare.stat().st_mode | stat.S_IXUSR)
    assert tb.main(["--pyright", cmd, "--", f"--pythonpath={bare}"]) == tb.EXIT_INFRASTRUCTURE
    err = capsys.readouterr().err
    assert "`import numpy` fails" in err and "ModuleNotFoundError" in err
    assert not marker.exists()

    # The real interpreter passes and the flag reaches pyright.
    assert tb.main(["--pyright", cmd, "--fail-on-errors", "--",
                    "--pythonpath", sys.executable]) == tb.EXIT_ERRORS
    assert _argv(tmp_path) == ["--outputjson", "--pythonpath", sys.executable]


def test_missing_pyright_command_gives_actionable_message_not_traceback(tb, tmp_path, capsys):
    """An absent command is the advertised hint and exit 2, not a
    FileNotFoundError traceback (S3)."""
    assert tb.main(["--pyright", "/nonexistent/pyright", "--fail-on-errors"]) == tb.EXIT_INFRASTRUCTURE
    err = capsys.readouterr().err
    assert "install it with `pip install pyright` or pass --pyright" in err
    assert "Traceback" not in err

    # The environment variable is the same path.
    old = os.environ.get("PYRIGHT")
    os.environ["PYRIGHT"] = str(tmp_path / "missing")
    try:
        assert tb.main([]) == tb.EXIT_INFRASTRUCTURE
    finally:
        if old is None:
            del os.environ["PYRIGHT"]
        else:
            os.environ["PYRIGHT"] = old

    # `--help` tells the reader that pyright's own flags go after `--`.
    with pytest.raises(SystemExit) as exc:
        tb.main(["--help"])
    assert exc.value.code == 0
    assert "after `--`" in capsys.readouterr().out


def test_ci_invocation_markdown_top_fail_on_errors_reports_the_count(tb, report, tmp_path, capsys):
    """The exact CI command line, end to end through main with a fake
    pyright: exit 1 with the Markdown tables on stdout (X1)."""
    cmd = fake_pyright(tmp_path, json.dumps(report), code=1)
    assert tb.main(["--pyright", cmd, "--markdown", "--top", "15",
                    "--fail-on-errors"]) == tb.EXIT_ERRORS
    out, err = capsys.readouterr()
    assert err == ""
    assert out.startswith("### pyright summary\n")
    assert "files analysed: 3; files with diagnostics: 3; errors: 4; warnings: 2; informations: 0" in out
    assert "### top 15 files by error count" in out
    assert "| 2 | `src/maddening/a.py` |" in out
    assert _argv(tmp_path) == ["--outputjson"]
    # The count line is what the CI report step parses.
    import re
    assert re.search(r"^files analysed:.*; errors: (\d+);", out, re.M).group(1) == "4"
