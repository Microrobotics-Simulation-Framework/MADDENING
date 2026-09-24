"""The CI test-time budget must fail on the slow test it exists to catch.

``scripts/report_test_durations.py`` reads pytest's JUnit XML and fails the
job on an unlisted test over the hard line.  Like every gate, it is only
evidence if it can fail: these tests plant a slow test and require a
non-zero exit, and plant an empty or unreadable report and require that it
does not pass either.

Everything runs in-process on hand-written XML -- no JAX, no subprocess --
so the file stays well inside the budget it tests.
"""

import importlib.util
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "report_test_durations.py"
ALLOWLIST = REPO_ROOT / "tests" / "duration_allowlist.txt"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("_gate_report_test_durations", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(file, name, seconds, cls="", line=10, extra=""):
    module = file[:-3].replace("/", ".")
    classname = f"{module}.{cls}" if cls else module
    return (f'<testcase classname="{classname}" name="{name}" file="{file}" '
            f'line="{line}" time="{seconds}">{extra}</testcase>')


def _report(tmp_path, *cases, name="results.xml"):
    path = tmp_path / name
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">'
        + "".join(cases) + "</testsuite></testsuites>", encoding="utf-8")
    return path


def _run(gate, capsys, *argv):
    code = gate.main([*map(str, argv), "--markdown", str(argv[0]) + ".md"])
    return code, capsys.readouterr().out


def test_an_unlisted_test_over_the_hard_line_fails_the_job(gate, tmp_path, capsys):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_slow", 25.0),
                     _case("tests/a/test_x.py", "test_fast", 0.2))
    code, out = _run(gate, capsys, report)
    assert code == 1
    assert "::error file=tests/a/test_x.py,line=11," in out
    assert "tests/a/test_x.py::test_slow took 25.0 s" in out


def test_the_same_test_on_the_allowlist_passes(gate, tmp_path, capsys):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_slow", 25.0))
    allow = tmp_path / "allow.txt"
    allow.write_text("# header\n\ntests/a/test_x.py::test_slow # kept: the only end-to-end run\n")
    code, out = _run(gate, capsys, report, "--allowlist", allow)
    assert code == 0
    assert "::error" not in out and "::warning" not in out


def test_an_unlisted_test_over_the_policy_line_warns_without_failing(gate, tmp_path, capsys):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_middling", 7.0, cls="TestA"))
    code, out = _run(gate, capsys, report)
    assert code == 0
    assert "::warning file=tests/a/test_x.py,line=11,title=Test over 5 s::" in out
    assert "tests/a/test_x.py::TestA::test_middling" in out


def test_fail_over_zero_reports_without_gating(gate, tmp_path, capsys):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_slow", 900.0))
    code, out = _run(gate, capsys, report, "--fail-over", "0")
    assert code == 0
    assert "::error" not in out


def test_a_report_with_no_test_cases_does_not_pass(gate, tmp_path, capsys):
    code, out = _run(gate, capsys, _report(tmp_path))
    assert code == 2
    assert "no test cases" in out


def test_a_missing_report_does_not_pass(gate, tmp_path, capsys):
    code, out = _run(gate, capsys, tmp_path / "never_written.xml")
    assert code == 2
    assert "did pytest run?" in out


def test_an_unparsable_report_does_not_pass(gate, tmp_path, capsys):
    bad = tmp_path / "bad.xml"
    bad.write_text("<testsuites><testcase")
    code, out = _run(gate, capsys, bad)
    assert code == 2
    assert "not valid XML" in out


def test_an_xunit2_report_is_refused_rather_than_misread(gate, tmp_path, capsys):
    # xunit2 (pytest's default family) drops the file attribute, without
    # which neither the node id nor the annotation's location is known.
    xml = tmp_path / "x2.xml"
    xml.write_text('<testsuites><testsuite><testcase classname="tests.a.test_x" '
                   'name="test_slow" time="30"/></testsuite></testsuites>')
    code, out = _run(gate, capsys, xml)
    assert code == 2
    assert "junit_family=xunit1" in out


def test_an_allowlist_entry_without_a_reason_is_refused(gate, tmp_path, capsys):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_slow", 25.0))
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/a/test_x.py::test_slow\n")
    code, out = _run(gate, capsys, report, "--allowlist", allow)
    assert code == 2
    assert "has no reason" in out


def test_node_ids_are_rebuilt_exactly(gate, tmp_path):
    report = _report(
        tmp_path,
        _case("tests/core/test_x.py", "test_fn[a-b.c]", 1.0),
        _case("tests/core/test_x.py", "test_m", 1.0, cls="TestOuter.TestInner"),
        _case("tests/core/test_x.py", "test_s", 1.0, cls="TestA", extra="<skipped/>"),
    )
    got = {t.nodeid: t.outcome for t in gate.read_report(report)}
    assert got == {
        "tests/core/test_x.py::test_fn[a-b.c]": "passed",
        "tests/core/test_x.py::TestOuter::TestInner::test_m": "passed",
        "tests/core/test_x.py::TestA::test_s": "skipped",
    }


def test_two_lanes_are_judged_on_each_tests_slower_time(gate, tmp_path, capsys):
    fast = _report(tmp_path, _case("tests/a/test_x.py", "test_t", 2.0), name="a.xml")
    slow = _report(tmp_path, _case("tests/a/test_x.py", "test_t", 30.0), name="b.xml")
    assert _run(gate, capsys, fast, slow)[0] == 1
    assert _run(gate, capsys, slow, fast)[0] == 1


def test_the_summary_names_the_bands_and_the_removable_entries(gate, tmp_path, capsys):
    report = _report(tmp_path,
                     _case("tests/a/test_x.py", "test_fine", 0.5),
                     _case("tests/a/test_x.py", "test_watch", 3.0),
                     _case("tests/a/test_x.py", "test_now_fast", 0.4))
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/a/test_x.py::test_now_fast # pending triage\n"
                     "tests/a/test_gone.py::test_renamed # pending triage\n")
    code, _ = _run(gate, capsys, report, "--allowlist", allow)
    md = Path(str(report) + ".md").read_text()
    assert code == 0
    assert re.search(r"\| up to 1 s: fine \| 2 \|", md)
    assert re.search(r"\| 1-5 s: watch, optimise \| 1 \|", md)
    assert "`tests/a/test_x.py::test_now_fast`" in md.split("may be removable")[1]
    assert "`tests/a/test_gone.py::test_renamed`" in md.split("may be removable")[1]


def test_the_shipped_allowlist_names_real_tests_with_reasons(gate):
    # A stale entry exempts nothing, but a typo'd one would silently fail to
    # exempt the test it meant -- and a list nobody can check only grows.
    entries = gate.read_allowlist(ALLOWLIST)
    assert entries, "the allowlist parsed empty"
    for nodeid, reason in entries.items():
        assert reason.startswith(("pending triage", "kept: ")), (nodeid, reason)
        file, *_, name = nodeid.split("::")
        path = REPO_ROOT / file
        assert path.is_file(), f"{nodeid}: {file} does not exist"
        func = name.split("[", 1)[0]
        assert re.search(rf"def {re.escape(func)}\b", path.read_text(encoding="utf-8")), (
            f"{nodeid}: no `def {func}` in {file}")


def test_ci_runs_the_budget_on_the_default_lane():
    # The gate is only a gate while CI calls it with the allowlist and feeds
    # it the XML it needs.
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--junitxml=test-results.xml -o junit_family=xunit1" in ci
    assert re.search(r"report_test_durations\.py test-results\.xml\s*\\\s*\n\s*--allowlist tests/duration_allowlist\.txt", ci)
