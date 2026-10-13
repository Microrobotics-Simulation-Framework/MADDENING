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


# ---------------------------------------------------------------------
# Tier gate (phase 2): typing_tiers.json, per-package ceilings, and the
# refusal to compare counts measured with a different pyright.
#
# Each of these asserts on the *message*, not only on the exit status:
# exit 1 is returned for "a package is over its ceiling" and exit 2 for
# four different infrastructure faults, so a test that checked `rc`
# alone could not tell them apart and would pass for the wrong reason.
# ---------------------------------------------------------------------

TIERS_DOC = {
    "pyright_version": "1.1.414",
    "environment": "a fixture",
    "tiers": {
        "clean": {"description": "zero", "max_errors": {"alpha": 0}},
        "ratchet": {"description": "capped", "max_errors": {"beta": 2}},
    },
}


@pytest.fixture
def tiers_file(tmp_path):
    def write(doc):
        p = tmp_path / "typing_tiers.json"
        p.write_text(json.dumps(doc))
        return p
    return write


def _pkg_report(tb, counts, version="1.1.414"):
    """A report with *counts* errors in each named top-level package."""
    root = tb.REPO_ROOT / "src" / "maddening"
    diags = []
    for pkg, n in counts.items():
        f = str(root / pkg / "mod.py") if "." not in pkg else str(root / pkg)
        diags += [_diag(f, "error", "reportArgumentType") for _ in range(n)]
    return {"version": version, "generalDiagnostics": diags,
            "summary": _summary(files=9, errors=sum(counts.values()))}


def test_errors_are_counted_per_top_level_package(tb):
    s = tb.summarise(_pkg_report(tb, {"alpha": 3, "beta": 1}))
    assert s.errors_by_package == {"alpha": 3, "beta": 1}


def test_a_module_directly_in_the_package_is_its_own_entry(tb):
    s = tb.summarise(_pkg_report(tb, {"sysid.py": 2}))
    assert s.errors_by_package == {"sysid.py": 2}


def test_summarise_can_restrict_to_one_tiers_packages(tb):
    report = _pkg_report(tb, {"alpha": 3, "beta": 1})
    assert tb.summarise(report, packages=("beta",)).error_count == 1
    # files_analyzed still describes the whole run, so the environment
    # checks keep seeing the truth.
    assert tb.summarise(report, packages=("beta",)).files_analyzed == 9


def test_gate_passes_at_the_ceiling(tb, tiers_file):
    tier = tb.load_tier("ratchet", tiers_file(TIERS_DOC))
    ok, verdict = tb.gate(tb.summarise(_pkg_report(tb, {"beta": 2})), tier)
    assert ok
    assert any("at the ceiling" in line for line in verdict)


def test_gate_fails_and_names_the_package_that_grew(tb, tiers_file):
    tier = tb.load_tier("ratchet", tiers_file(TIERS_DOC))
    ok, verdict = tb.gate(tb.summarise(_pkg_report(tb, {"beta": 5})), tier)
    assert not ok
    assert any(line.startswith("FAIL  beta:") and "+3" in line
               for line in verdict), verdict


def test_gate_reports_slack_so_a_ratchet_can_be_tightened(tb, tiers_file):
    tier = tb.load_tier("ratchet", tiers_file(TIERS_DOC))
    ok, verdict = tb.gate(tb.summarise(_pkg_report(tb, {"beta": 0})), tier)
    assert ok
    assert any("lower the ceiling to 0" in line for line in verdict), verdict


def test_a_single_new_error_fails_the_zero_tier(tb, tiers_file):
    tier = tb.load_tier("clean", tiers_file(TIERS_DOC))
    assert tb.gate(tb.summarise(_pkg_report(tb, {"alpha": 0})), tier)[0]
    ok, verdict = tb.gate(tb.summarise(_pkg_report(tb, {"alpha": 1})), tier)
    assert not ok
    assert any("FAIL  alpha: 1 errors, ceiling 0" in line for line in verdict)


def test_a_missing_tier_file_is_an_infrastructure_failure(tb, tmp_path):
    with pytest.raises(tb.InfrastructureFailure, match="cannot read"):
        tb.load_tier("clean", tmp_path / "nope.json")


def test_an_unknown_tier_name_is_an_infrastructure_failure(tb, tiers_file):
    with pytest.raises(tb.InfrastructureFailure, match="defines no tier"):
        tb.load_tier("nosuch", tiers_file(TIERS_DOC))


def test_an_empty_ceiling_map_is_refused_rather_than_passing(tb, tiers_file):
    """A gate over nothing would pass on anything -- fail closed."""
    doc = {"pyright_version": "1.1.414",
           "tiers": {"clean": {"max_errors": {}}}}
    with pytest.raises(tb.InfrastructureFailure, match="no `max_errors`"):
        tb.load_tier("clean", tiers_file(doc))


def test_a_non_count_ceiling_is_refused(tb, tiers_file):
    doc = {"pyright_version": "1.1.414",
           "tiers": {"clean": {"max_errors": {"alpha": True}}}}
    with pytest.raises(tb.InfrastructureFailure, match="not a count"):
        tb.load_tier("clean", tiers_file(doc))


def test_ceilings_from_another_pyright_release_are_refused(tb, tiers_file):
    """Every release changes diagnostics, so the comparison is void."""
    tier = tb.load_tier("clean", tiers_file(TIERS_DOC))
    tb.check_tier_environment(tier, "1.1.414")          # the recorded one
    with pytest.raises(tb.InfrastructureFailure, match="1.1.999"):
        tb.check_tier_environment(tier, "1.1.999")


def test_a_tier_file_without_a_recorded_version_is_refused(tb, tiers_file):
    doc = {"tiers": {"clean": {"max_errors": {"alpha": 0}}}}
    tier = tb.load_tier("clean", tiers_file(doc))
    with pytest.raises(tb.InfrastructureFailure, match="no `pyright_version`"):
        tb.check_tier_environment(tier, "1.1.414")


def test_main_exits_1_and_says_which_tier_when_a_ceiling_is_exceeded(
        tb, tmp_path, tiers_file, capsys):
    run = tmp_path / "run.json"
    run.write_text(json.dumps(_pkg_report(tb, {"alpha": 4})))
    rc = tb.main(["--json", str(run), "--tier", "clean",
                  "--tiers-file", str(tiers_file(TIERS_DOC))])
    err = capsys.readouterr().err
    assert rc == tb.EXIT_ERRORS
    assert "tier clean is over its committed ceiling" in err
    assert "FAIL  alpha: 4 errors, ceiling 0" in err


def test_main_exits_2_not_1_when_the_pyright_release_does_not_match(
        tb, tmp_path, tiers_file, capsys):
    """The distinction matters: exit 1 means 'you broke it', exit 2 means
    'nobody can tell'.  Both would be non-zero to a shell."""
    run = tmp_path / "run.json"
    run.write_text(json.dumps(_pkg_report(tb, {"alpha": 0},
                                          version="1.1.999")))
    rc = tb.main(["--json", str(run), "--tier", "clean",
                  "--tiers-file", str(tiers_file(TIERS_DOC))])
    err = capsys.readouterr().err
    assert rc == tb.EXIT_INFRASTRUCTURE
    assert "infrastructure failure" in err
    assert "1.1.999" in err


def test_the_committed_tier_file_is_loadable_and_covers_every_package(tb):
    """The real typing_tiers.json must name every package under
    ``src/maddening``, or a package could drift with no ceiling at all.

    ``src/maddening/__init__.py`` is included, and used to be excluded
    here.  It is the one module the exclusion left with no ceiling, and
    it is the module every ``from maddening import X`` resolves through
    now that the wheel ships ``py.typed``: its ``if TYPE_CHECKING:``
    re-export block is what a downstream checker reads, and
    ``tests/test_lazy_reexports.py`` is deliberately ``ast``-based, so
    pyright is the only thing that can see a stale name in it.
    """
    tier1 = tb.load_tier("tier1")
    tier2 = tb.load_tier("tier2")
    covered = set(tier1.packages) | set(tier2.packages)
    root = tb.REPO_ROOT / "src" / "maddening"
    on_disk = {
        p.name for p in root.iterdir()
        if (p.is_dir() and (p / "__init__.py").exists()
            and p.name not in ("examples", "__pycache__"))
        or (p.is_file() and p.suffix == ".py")
    }
    assert not on_disk - covered, (
        f"{sorted(on_disk - covered)} are under src/maddening but in no "
        "tier, so nothing gates them; add them to typing_tiers.json"
    )
    assert not covered - on_disk, (
        f"{sorted(covered - on_disk)} are in typing_tiers.json but not on "
        "disk; a ceiling on a package that no longer exists can never fail"
    )


def test_tier1_is_committed_at_zero(tb):
    """The policy's own claim, as a test: tier 1 is not merely small."""
    assert set(tb.load_tier("tier1").max_errors.values()) == {0}


def test_the_package_root_module_maps_to_a_key_the_tiers_actually_name(tb):
    """``src/maddening/__init__.py`` is gated, not silently out of scope.

    ``_package_of`` gives a module directly under the package its own
    key, so the package root module is ``"__init__.py"``.  With
    ``--tier``, ``summarise`` filters the diagnostics to the tier's
    packages *before* ``gate`` runs, so a key in neither tier is not
    reported as uncovered -- it simply disappears, and every CI typing
    step passes however many errors the file has.  The pairing of this
    assertion with the coverage test above is what closes that.
    """
    key = tb._package_of("src/maddening/__init__.py")
    assert key == "__init__.py"
    assert key in tb.load_tier("tier1").max_errors


def test_an_error_in_the_package_root_module_fails_the_real_tier1_gate(
        tb, tmp_path, capsys):
    """End to end against the committed typing_tiers.json, as CI runs it.

    Measured 2026-09-20 with pyright 1.1.414: the file has **zero**
    errors, so the ceiling is a ratchet at its floor rather than an
    accommodation of an existing count.
    """
    run = tmp_path / "run.json"
    run.write_text(json.dumps(_pkg_report(tb, {"__init__.py": 1})))
    rc = tb.main(["--json", str(run), "--tier", "tier1"])
    err = capsys.readouterr().err
    assert rc == tb.EXIT_ERRORS
    # Not just the exit code: the verdict has to name the file, or a
    # maintainer cannot act on it and an unrelated guard could be what
    # turned the run red.
    assert "FAIL  __init__.py: 1 errors, ceiling 0" in err
