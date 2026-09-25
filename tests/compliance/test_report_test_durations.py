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
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "report_test_durations.py"
ALLOWLIST = REPO_ROOT / "tests" / "duration_allowlist.txt"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("_gate_report_test_durations", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(file, name, seconds, cls="", line=10, extra="", jax=None):
    """A ``<testcase>``; ``jax`` overrides the timing properties (``None``: none).

    A ``subprocesses`` key in ``jax`` is written only when given, as in a
    report from before the process count was recorded.
    """
    module = file[:-3].replace("/", ".")
    classname = f"{module}.{cls}" if cls else module
    if jax is not None:
        props = {"jax_trace_s": 0, "jax_lower_s": 0, "jax_compile_s": 0,
                 "jax_cache_read_s": 0, "jax_cache_hits": 0, "jax_cache_misses": 0, **jax}
        extra += "<properties>" + "".join(
            f'<property name="{k}" value="{v}"/>' for k, v in props.items()) + "</properties>"
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


def test_an_allowlist_entry_exempts_exactly_its_node_id(gate, tmp_path, capsys):
    # Not a prefix: a base id does not exempt its parametrisations, and a
    # file does not exempt its tests.  One entry must not quietly cover
    # tests nobody decided to keep.
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_p[a]", 30.0),
                     _case("tests/a/test_x.py", "test_q", 30.0),
                     _case("tests/a/test_x.py", "test_kept[b]", 30.0))
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/a/test_x.py::test_p # kept: only the base id is listed\n"
                     "tests/a/test_x.py # kept: only the file is listed\n"
                     "tests/a/test_x.py::test_kept[b] # kept: the exact id\n")
    code, out = _run(gate, capsys, report, "--allowlist", allow)
    assert code == 1
    assert "tests/a/test_x.py::test_p[a] took 30.0 s" in out
    assert "tests/a/test_x.py::test_q took 30.0 s" in out
    assert "test_kept[b] took" not in out


def test_the_shipped_allowlist_names_real_tests_with_reasons(gate):
    # A stale entry exempts nothing, but a typo'd one would silently fail to
    # exempt the test it meant -- and a list nobody can check only grows.
    entries = gate.read_allowlist(ALLOWLIST)
    assert entries, "the allowlist parsed empty"
    for nodeid, reason in entries.items():
        # Every entry is a decision: the `pending triage` backlog the gate
        # started with is gone, and a new one would be a list nobody decided.
        assert reason.startswith("kept: "), (nodeid, reason)
        file, *_, name = nodeid.split("::")
        path = REPO_ROOT / file
        assert path.is_file(), f"{nodeid}: {file} does not exist"
        func = name.split("[", 1)[0]
        assert re.search(rf"def {re.escape(func)}\b", path.read_text(encoding="utf-8")), (
            f"{nodeid}: no `def {func}` in {file}")


def test_each_slow_test_is_diagnosed_by_where_its_time_went(gate):
    # The measured shapes of three real slow tests (2026-09-24 probe).
    loop = gate.TestTime("t.py::loop", "t.py", 1, 170.0, "passed",
                         {"jax_trace_s": 0.0, "jax_lower_s": 0.2, "jax_compile_s": 0.6,
                          "jax_cache_read_s": 0, "jax_cache_hits": 0, "jax_cache_misses": 0})
    retrace = gate.TestTime("t.py::retrace", "t.py", 1, 100.0, "passed",
                            {"jax_trace_s": 60.0, "jax_lower_s": 10.0, "jax_compile_s": 28.0,
                             "jax_cache_read_s": 0, "jax_cache_hits": 0, "jax_cache_misses": 0})
    compile_ = gate.TestTime("t.py::compile", "t.py", 1, 135.0, "passed",
                             {"jax_trace_s": 0.7, "jax_lower_s": 34.8, "jax_compile_s": 83.5,
                              "jax_cache_read_s": 0, "jax_cache_hits": 3, "jax_cache_misses": 12})
    assert gate.diagnose(loop) == "running 100%: un-jitted loop, heavy compute, or a subprocess"
    assert gate.diagnose(retrace).startswith("tracing/lowering 70%")
    assert gate.diagnose(compile_) == "compiling 62%: 12 cache misses"
    assert loop.uncacheable == pytest.approx(169.4)
    assert compile_.running == pytest.approx(135.0 - 83.5 - 35.5)
    # no timing recorded: no diagnosis, and nothing is assumed cacheable
    bare = gate.TestTime("t.py::bare", "t.py", 1, 9.0, "passed")
    assert gate.diagnose(bare) == "" and bare.uncacheable == 9.0


def test_a_test_slow_without_its_compile_is_listed_as_slow_even_when_warm(gate, tmp_path, capsys):
    report = _report(
        tmp_path,
        # 9 s, 1 s of it compiling: still 8 s with a perfect cache
        _case("tests/a/test_x.py", "test_loop", 9.0, jax={"jax_compile_s": 1.0}),
        # 9 s, 8 s compiling: a warm cache brings it to 1 s
        _case("tests/a/test_x.py", "test_compiles", 9.0, jax={"jax_compile_s": 8.0}),
    )
    code, out = _run(gate, capsys, report, "--cache-mode", "cold")
    md = Path(str(report) + ".md").read_text()
    warm = md.split("### Slow even with a warm cache (1)")[1].split("###")[0]
    assert "`tests/a/test_x.py::test_loop`" in warm
    assert "test_compiles" not in warm
    assert "**Compilation cache: cold**" in md
    assert ("running 9.0 s, tracing/lowering 0.0 s, XLA compile 9.0 s (cache reads excluded), "
            "cache reads 0.0 s") in md
    # "%" is escaped as %25 in a workflow command
    assert "(running 89%25: un-jitted loop, heavy compute, or a subprocess)" in out


def test_a_warm_run_never_calls_an_allowlist_entry_removable(gate, tmp_path, capsys):
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_cached", 0.4,
                                     jax={"jax_cache_hits": 5}))
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/a/test_x.py::test_cached # pending triage\n")
    for mode, listed in (("warm", False), ("mixed", False), ("cold", True), ("off", True)):
        _run(gate, capsys, report, "--allowlist", allow, "--cache-mode", mode)
        md = Path(str(report) + ".md").read_text().split("## Test durations")[-1]
        assert ("may be removable" in md) is listed, mode
        assert ("not judged removable on a run that restored a cache" in md) is (not listed), mode


def test_the_timing_plugin_attaches_its_properties_before_the_teardown_report():
    from types import SimpleNamespace
    from tests import _jax_timing as jt

    timing = jt.JaxTiming()
    plugin = jt.Plugin(timing)
    item = SimpleNamespace(user_properties=[])
    plugin.pytest_runtest_setup(item)
    timing.on_duration("/jax/core/compile/backend_compile_duration", 2.5)
    timing.on_duration("/jax/core/compile/backend_compile_duration", 0.5)
    timing.on_duration("/jax/some/other_duration", 99.0)   # ignored
    timing.on_event("/jax/compilation_cache/cache_hits")
    for when in ("setup", "call"):
        plugin.pytest_runtest_makereport(item, SimpleNamespace(when=when))
    assert item.user_properties == []
    plugin.pytest_runtest_makereport(item, SimpleNamespace(when="teardown"))
    props = dict(item.user_properties)
    assert set(props) == set(jt.PROPERTIES)
    assert props["jax_compile_s"] == 3.0 and props["jax_cache_hits"] == 1
    assert props["jax_trace_s"] == 0
    # the next test starts from zero
    plugin.pytest_runtest_setup(item)
    assert timing.properties() == [(p, 0) for p in jt.PROPERTIES]


def test_jax_still_emits_the_events_the_timing_plugin_listens_for():
    # If JAX renames an event, the plugin would report zero for it and every
    # slow test would read as "running".  Check the names against the
    # installed JAX: the compile-pipeline durations by listening to a real
    # compile, the cache events by their presence in JAX's compiler module
    # (they only fire with a cache configured).
    import inspect
    import jax
    import jax.numpy as jnp
    from jax import monitoring
    from jax._src import compiler, compilation_cache
    from tests import _jax_timing as jt

    seen, listening = set(), [True]

    def listener(event, secs, **_):
        if listening[0]:
            seen.add(event)

    # Switched off rather than unregistered: the unregister API differs
    # across the JAX versions CI runs, and a listener that ignores every
    # event costs nothing.
    monitoring.register_event_duration_secs_listener(listener)
    try:
        # A constant no other test uses, so this is a fresh trace and compile.
        jax.jit(lambda x: x * 1.2345678901 + 0.987654321)(jnp.ones(3)).block_until_ready()
    finally:
        listening[0] = False
    pipeline = [e for e in jt.DURATION_EVENTS if e.startswith("/jax/core/compile/")]
    assert pipeline and set(pipeline) <= seen, set(pipeline) - seen
    source = inspect.getsource(compiler) + inspect.getsource(compilation_cache)
    cache_events = [e for e in [*jt.DURATION_EVENTS, *jt.COUNT_EVENTS]
                    if e.startswith("/jax/compilation_cache/")]
    assert len(cache_events) == 3
    for event in cache_events:
        assert event in source, f"JAX no longer emits {event!r}"


def test_ci_runs_the_budget_on_the_default_lane():
    # The gate is only a gate while CI calls it with the allowlist and feeds
    # it the XML it needs.
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--junitxml=test-results-shard${{ matrix.shard }}.xml -o junit_family=xunit1" in ci
    assert re.search(r"report_test_durations\.py test-results-shard\$\{\{ matrix\.shard \}\}\.xml\s*\\\s*\n"
                     r"\s*--allowlist tests/duration_allowlist\.txt", ci)
    # ... with the per-test JAX split recorded, and the cache mode stated
    assert 'MADDENING_TEST_JAX_TIMING: "1"' in ci
    assert "--cache-mode" in ci
    # ...one cache per shard, so shard i of a PR reads shard i of the base
    assert re.search(r"key=jaxcc-v1-.*-shard\$\{\{ matrix\.shard \}\}of4", ci)
    # ...and without a size cap: with one, every cache write rescans the
    # whole directory, quadratic over a cold run (measured: a lane past
    # 95 minutes).
    assert "JAX_COMPILATION_CACHE_MAX_SIZE:" not in ci


def test_a_partial_lane_lists_no_allowlist_entry_as_removable(gate, tmp_path, capsys):
    # With one shard's report missing, an allowlisted test that is absent
    # may simply be on that shard.
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_fast", 0.2))
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/b/test_y.py::test_elsewhere # pending triage\n")
    _run(gate, capsys, report, "--allowlist", allow)
    assert "may be removable" in Path(str(report) + ".md").read_text()
    Path(str(report) + ".md").unlink()
    _run(gate, capsys, report, "--allowlist", allow, "--no-removable")
    assert "may be removable" not in Path(str(report) + ".md").read_text()


def test_every_shipped_allowlist_entry_is_a_node_id_pytest_collects(gate):
    """An entry is matched exactly, so a typo in it exempts nothing, silently.

    The check above finds the file and the function; this one asks pytest
    for the real node ids, so a wrong parameter id or class name fails too.
    Slow-marked tests are collected as well: an entry for one is stale,
    which the lane summary reports, not a typo.
    """
    entries = gate.read_allowlist(ALLOWLIST)
    files = sorted({nodeid.split("::", 1)[0] for nodeid in entries})
    # Not this job's shard (which would deselect other shards' files), nor
    # this job's timing plugin.
    env = {k: v for k, v in os.environ.items()
           if k not in ("MADDENING_TEST_SHARD", "MADDENING_TEST_JAX_TIMING", "PYTEST_ADDOPTS")}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         "-m", "slow or not slow", *files],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    collected = {line.strip() for line in proc.stdout.splitlines() if "::" in line}
    missing = sorted(set(entries) - collected)
    assert not missing, f"allowlist entries pytest does not collect: {missing}"


def test_a_cache_read_is_counted_once_and_reported_on_its_own(gate, tmp_path, capsys):
    """JAX times the whole compile-or-read-the-cache call as the compile.

    ``backend_compile_duration`` wraps ``compile_or_get_cached()``, and the
    cache read (``cache_retrieval_time_sec``) is recorded inside it, so on
    a hit the "compile" *is* the read.  The shape below is a measured warm
    run of one program (jax 0.11.0, 2026-09-25): compile event 0.365 s of
    which 0.348 s was the read.
    """
    warm = {"jax_trace_s": 0.208, "jax_lower_s": 0.242, "jax_compile_s": 0.365,
            "jax_cache_read_s": 0.348, "jax_cache_hits": 3, "jax_cache_misses": 0}
    t = gate.TestTime("t.py::warm", "t.py", 1, 1.748, "passed", warm)
    assert t.compiling == pytest.approx(0.017)
    assert t.cache_read == pytest.approx(0.348)
    # Each part subtracted once: running is wall - compile event - build.
    assert t.running == pytest.approx(1.748 - 0.365 - 0.450)
    # A warm run is the run that pays the read, so it is not removable.
    assert t.uncacheable == pytest.approx(1.748 - 0.017)
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_warm", 6.0,
                                     jax={**warm, "jax_compile_s": 1.5, "jax_cache_read_s": 1.0}))
    _run(gate, capsys, report, "--cache-mode", "warm")
    md = Path(str(report) + ".md").read_text()
    assert "XLA compile 0.5 s (cache reads excluded), cache reads 1.0 s" in md
    assert "running 4.0 s" in md                   # 6.0 - 1.5 - 0.45


def test_a_test_whose_work_runs_in_a_subprocess_is_not_called_slow_even_when_warm(
        gate, tmp_path, capsys):
    """A child's compiles are invisible here, and it inherits the cache directory.

    Measured: a test whose JAX work runs in a child recorded 0 s of
    compile in the pytest process on both CI lanes, was listed as "slow
    even with a warm cache", and ran at 0.53x on the next warm run.
    """
    report = _report(
        tmp_path,
        # started a process and recorded no JAX of its own
        _case("tests/a/test_x.py", "test_child", 9.0, jax={"subprocesses": 1}),
        # started a process and compiled in-process too
        _case("tests/a/test_x.py", "test_both", 9.0,
              jax={"subprocesses": 2, "jax_compile_s": 1.0}),
        # a report from before the count: no JAX activity at all here
        _case("tests/a/test_x.py", "test_old_report", 9.0, jax={}),
        # plain Python, no process started: a cache cannot help it
        _case("tests/a/test_x.py", "test_sleeps", 9.0, jax={"subprocesses": 0}),
    )
    code, out = _run(gate, capsys, report, "--cache-mode", "cold")
    md = Path(str(report) + ".md").read_text()
    warm = md.split("### Slow even with a warm cache (1)")[1].split("###")[0]
    assert "test_sleeps" in warm
    assert not any(name in warm for name in ("test_child", "test_both", "test_old_report"))
    sub = md.split("### Work in a subprocess (not measured here) (3)")[1].split("###")[0]
    assert all(f"`tests/a/test_x.py::{name}`" in sub
               for name in ("test_child", "test_both", "test_old_report"))
    assert "| not recorded |" in sub
    assert "running 100%: work in a subprocess (not measured here)" in md
    assert "1 slow even with a warm cache; 3 with work in a subprocess" in out


def test_a_split_that_adds_up_to_more_than_the_wall_time_is_capped_and_flagged(gate):
    # Measured on jax 0.10.2 in CI: a 0.28 s test whose tracing events
    # summed to 0.33 s, printed as "tracing/lowering 118%".
    over = gate.TestTime("t.py::over", "t.py", 1, 0.28, "passed",
                         {"jax_trace_s": 0.30, "jax_lower_s": 0.03, "jax_compile_s": 0.0,
                          "jax_cache_read_s": 0, "jax_cache_hits": 0, "jax_cache_misses": 0})
    why = gate.diagnose(over)
    assert why.startswith("tracing/lowering 100%: ")
    assert "more than the wall time" in why
    assert not re.search(r"\b(10[1-9]|1[1-9][0-9]|[2-9][0-9]{2,})%", why)
    within = over._replace(seconds=1.0)
    assert "more than the wall time" not in gate.diagnose(within)


def test_a_report_holding_only_a_collection_error_does_not_pass(gate, tmp_path, capsys):
    """A file that fails to import stops pytest before any test runs.

    Every shard collects the whole suite, so one broken import leaves every
    shard's report with a single collection-error entry.  That is a run in
    which no test ran, which the gate must not pass -- and a lane summary
    built from it must not call allowlist entries removable.  The report
    here is real pytest output, from this interpreter's pytest.
    """
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (project / "tests" / "test_broken.py").write_text("import a_module_that_does_not_exist\n")
    (project / "tests" / "test_fine.py").write_text("def test_fine():\n    pass\n")
    env = {k: v for k, v in os.environ.items()
           if k not in ("MADDENING_TEST_SHARD", "MADDENING_TEST_JAX_TIMING", "PYTEST_ADDOPTS")}
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
                           "--junitxml=broken.xml", "-o", "junit_family=xunit1"],
                          cwd=project, env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2, proc.stdout
    broken = project / "broken.xml"
    (case,) = gate.read_report(broken)
    assert case.collection_error and case.outcome == "error"
    code, out = _run(gate, capsys, broken)
    assert code == 2 and "collection error" in out
    # In a lane with another shard's real report: that report is judged,
    # the broken one left out, and nothing is called removable.
    fine = _report(tmp_path, _case("tests/a/test_x.py", "test_fast", 0.2), name="fine.xml")
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/b/test_y.py::test_elsewhere # pending triage\n")
    code, out = _run(gate, capsys, broken, fine, "--allowlist", allow, "--cache-mode", "cold")
    assert code == 0
    assert "::warning title=Test durations::left out" in out and "broken.xml" in out
    assert "may be removable" not in Path(str(broken) + ".md").read_text()


def test_timing_plugin_counts_the_processes_a_test_starts():
    from types import SimpleNamespace
    from tests import _jax_timing as jt

    timing = jt.JaxTiming()
    plugin = jt.Plugin(timing)
    item = SimpleNamespace(user_properties=[])
    plugin.pytest_runtest_setup(item)
    timing.on_audit("subprocess.Popen", ("python", ["python", "-c", ""], None, None))
    timing.on_audit("os.fork", ())
    timing.on_audit("open", ("f", "r", 0))            # not a process
    plugin.pytest_runtest_makereport(item, SimpleNamespace(when="teardown"))
    assert dict(item.user_properties)["subprocesses"] == 2


def test_registering_the_timing_plugin_starts_counting_processes():
    # What conftest calls when MADDENING_TEST_JAX_TIMING=1.  In a child:
    # neither the audit hook nor JAX's listeners can be removed once added.
    # JAX's listener API is stubbed (its events are checked against the real
    # JAX above), which keeps the child to a fraction of a second.
    probe = (
        "import subprocess, sys\n"
        "from types import ModuleType, SimpleNamespace\n"
        "jax = ModuleType('jax')\n"
        "jax.monitoring = SimpleNamespace(register_event_duration_secs_listener=lambda f: None,\n"
        "                                 register_event_listener=lambda f: None)\n"
        "sys.modules['jax'] = jax\n"
        "from tests import _jax_timing as jt\n"
        "names = []\n"
        "config = SimpleNamespace(pluginmanager=SimpleNamespace(\n"
        "    register=lambda plugin, name: names.append(name)))\n"
        "timing = jt.register(config)\n"
        "subprocess.run([sys.executable, '-c', ''], check=True)\n"
        "print(timing.totals['subprocesses'], names)\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONSAFEPATH"}
    proc = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, env=env,
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.split(None, 1) == ["1", "['maddening-jax-timing']\n"], proc.stdout


def test_python_still_raises_the_audit_events_the_process_count_listens_for():
    # If Python renamed one, the count would read zero and a subprocess
    # test would be called uncacheable again.  Checked in a child, since an
    # audit hook cannot be removed once added.
    from tests import _jax_timing as jt

    probe = (
        "import os, subprocess, sys\n"
        "seen = set()\n"
        "sys.addaudithook(lambda e, a: seen.add(e))\n"
        "subprocess.run([sys.executable, '-c', ''], check=True)\n"
        "os.system('true')\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os._exit(0)\n"
        "os.waitpid(pid, 0)\n"
        "os.waitpid(os.posix_spawn('/bin/true', ['true'], dict(os.environ)), 0)\n"
        "print(' '.join(sorted(e for e in seen if e.startswith(('os.', 'subprocess.')))))\n"
    )
    proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    seen = set(proc.stdout.split())
    # (os.spawn* forks and execs on POSIX, raising os.fork; its own event is Windows-only.)
    for event in ("subprocess.Popen", "os.system", "os.fork", "os.posix_spawn"):
        assert event in seen, f"Python no longer raises {event!r}"
        assert event in jt.SPAWN_EVENTS


def _ci():
    return yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))


def _budget_step():
    (step,) = [s for s in _ci()["jobs"]["test"]["steps"] if s.get("name") == "Test time budget"]
    return step


def _commands(script):
    joined = re.sub(r"\\\n", " ", script)
    return [ln.strip() for ln in joined.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def test_the_budget_step_can_fail_the_job():
    """The gate is a gate only while its exit code is the step's.

    ``--fail-over 0`` switches the hard line off, ``continue-on-error``
    turns a red step green, and ``|| true`` (or ``set +e``) swallows the
    exit code -- each a one-line change that leaves every other check here
    passing.
    """
    job = _ci()["jobs"]["test"]
    step = _budget_step()
    assert "continue-on-error" not in job and "continue-on-error" not in step
    # GitHub's default for `run:` is `bash -e {0}`; a custom shell could drop -e.
    assert "shell" not in step
    assert step["if"] == "${{ !cancelled() }}", "the budget must run after a red test step too"
    # The `${{ }}` expressions are GitHub's, evaluated before the shell runs.
    script = re.sub(r"\$\{\{.*?\}\}", "EXPR", step["run"])
    for undo in ("--fail-over", "||", "set +e"):
        assert undo not in script, f"the budget step contains {undo!r}"
    # `${{ matrix.shard }}` -> `{{matrix.shard}}`, one shell word.
    last = shlex.split(re.sub(r"\$\{\{\s*(.*?)\s*\}\}", r"{{\1}}", _commands(step["run"])[-1]))
    assert last[:3] == ["python", "scripts/report_test_durations.py",
                        "test-results-shard{{matrix.shard}}.xml"], last
    assert last[last.index("--allowlist") + 1] == "tests/duration_allowlist.txt"


def _render(text, values):
    def sub(m):
        expr = m.group(1).strip()
        assert expr in values, f"the test does not know the expression {expr!r}; add it"
        return values[expr]
    return re.sub(r"\$\{\{(.*?)\}\}", sub, text)


def _shim(tmp_path):
    """A PATH on which `python` and `python3` run this interpreter."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("python", "python3"):
        (bin_dir / name).write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
        (bin_dir / name).chmod(0o755)
    return f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


def test_the_budget_step_records_the_cache_mode_even_when_the_gate_fails(tmp_path):
    """The step runs under ``bash -e``: a line after a failing gate never runs.

    The mode file used to be written after the gate, so an over-budget
    shard left none, and the lane summary labelled the lane from the other
    three shards -- "cold" for a lane with a warm shard, whose allowlisted
    tests were then offered as removable.
    """
    work = tmp_path / "work"
    (work / "tests").mkdir(parents=True)
    (work / "scripts").symlink_to(REPO_ROOT / "scripts")
    (work / "tests" / "duration_allowlist.txt").write_text("# empty\n")
    _report(work, _case("tests/a/test_x.py", "test_slow", 25.0), name="test-results-shard1.xml")
    step = _budget_step()
    script = _render(step["run"], {
        "matrix.shard": "1",
        "(steps.cc.outputs.mode == 'warm' && steps.cc-restore.outputs.cache-matched-key != '') "
        "&& 'warm' || 'cold'": "warm",
    })
    proc = subprocess.run(["bash", "-e", "-c", script], cwd=work, capture_output=True, text=True,
                          env={**os.environ, "PATH": _shim(tmp_path)}, timeout=120)
    assert proc.returncode == 1, proc.stdout + proc.stderr     # the gate failed the step
    assert (work / "cache-mode-shard1.txt").read_text().strip() == "warm"


def test_a_lane_missing_a_shards_cache_mode_is_labelled_mixed(tmp_path):
    (step,) = [s for s in _ci()["jobs"]["test-durations"]["steps"]
               if s.get("name") == "Summarise the lane"]
    work = tmp_path / "work"
    (work / "results").mkdir(parents=True)
    (work / "tests").mkdir()
    (work / "scripts").symlink_to(REPO_ROOT / "scripts")
    (work / "tests" / "duration_allowlist.txt").write_text(
        "tests/a/test_x.py::test_cached # pending triage\n")
    for shard in (1, 2, 3, 4):
        _report(work / "results", _case("tests/a/test_x.py", f"test_{shard}", 0.2),
                name=f"test-results-shard{shard}.xml")
    for shard in (1, 2, 3):                         # shard 4 wrote no mode file
        (work / "results" / f"cache-mode-shard{shard}.txt").write_text("warm\n")
    summary = tmp_path / "summary.md"
    script = _render(step["run"], {"matrix.jax-version": "0.10.2"})
    proc = subprocess.run(["bash", "-e", "-c", script], cwd=work, capture_output=True, text=True,
                          env={**os.environ, "PATH": _shim(tmp_path),
                               "GITHUB_STEP_SUMMARY": str(summary)}, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    md = summary.read_text()
    assert "**Compilation cache: mixed**" in md
    assert "may be removable" not in md


# ---------------------------------------------------------------------------
# What each shard's compilation cache was, from what its report recorded
# ---------------------------------------------------------------------------


def _lookups(hits, misses, n_tests=4):
    """``n_tests`` cases in one file sharing ``hits`` / ``misses``."""
    per = [(hits // n_tests + (i < hits % n_tests), misses // n_tests + (i < misses % n_tests))
           for i in range(n_tests)]
    return [_case("tests/a/test_x.py", f"test_{i}", 0.1,
                  jax={"jax_cache_hits": h, "jax_cache_misses": m}) for i, (h, m) in enumerate(per)]


def _shard(tmp_path, n, hits, misses, mode):
    report = _report(tmp_path, *_lookups(hits, misses), name=f"test-results-shard{n}.xml")
    if mode is not None:
        (tmp_path / f"cache-mode-shard{n}.txt").write_text(mode + "\n")
    return report


def test_a_restored_cache_that_most_lookups_missed_is_not_called_warm(gate, tmp_path, capsys):
    """The label follows the hits, not the restore.

    On CI about one PR shard in four restored a cache and then hit 17-20% of
    its lookups -- a cold run's rate, from programs it compiled itself --
    because JAX's key includes the host's CPU features and runners with one
    CPU model name can differ in them.  Those shards used to be labelled
    warm, with "these times are lower than a cold run's".
    """
    reports = [_shard(tmp_path, 1, 3856, 521, "warm"), _shard(tmp_path, 2, 625, 2916, "warm")]
    code, out = _run(gate, capsys, *reports, "--cache-mode", "warm")
    md = Path(str(reports[0]) + ".md").read_text()
    assert code == 0
    assert "**Compilation cache: mixed**" in md and "**Compilation cache: warm**" not in md
    assert "shard 1 warm, 3856 of 4377 lookups hit (88%)" in md
    assert "shard 2 restored but unused (cold), 625 of 3541 lookups hit (18%)" in md
    assert "compilation cache: shard 1 warm" in out            # the step log says it too


def test_a_lane_whose_restored_caches_all_went_unused_says_so(gate, tmp_path, capsys):
    reports = [_shard(tmp_path, n, 600, 2900, "warm") for n in (1, 2)]
    allow = tmp_path / "allow.txt"
    allow.write_text("tests/b/test_y.py::test_elsewhere # kept: x\n")
    _run(gate, capsys, *reports, "--cache-mode", "warm", "--allowlist", allow)
    md = Path(str(reports[0]) + ".md").read_text()
    assert "**Compilation cache: restored but unused (cold)**" in md
    # A shard that restored a cache still lists nothing as removable: a low
    # hit rate can also be a change that touched most programs.
    assert "may be removable" not in md
    assert "not judged removable on a run that restored a cache" in md


@pytest.mark.parametrize("hits, label", [(51, "warm"), (50, "restored but unused (cold)"),
                                         (0, "restored but unused (cold)")])
def test_warm_needs_more_than_half_of_the_lookups_to_hit(gate, tmp_path, capsys, hits, label):
    report = _shard(tmp_path, 1, hits, 100 - hits, "warm")
    _run(gate, capsys, report)
    md = Path(str(report) + ".md").read_text()
    assert f"shard 1 {label}, {hits} of 100 lookups hit" in md


def test_a_warm_run_that_recorded_no_lookups_keeps_its_claim(gate, tmp_path, capsys):
    # Without the timing plugin there is nothing to judge the claim by.
    report = _report(tmp_path, _case("tests/a/test_x.py", "test_t", 0.1),
                     name="test-results-shard1.xml")
    (tmp_path / "cache-mode-shard1.txt").write_text("warm\n")
    _run(gate, capsys, report)
    assert "**Compilation cache: warm**" in Path(str(report) + ".md").read_text()


def test_each_shards_mode_file_decides_its_label(gate, tmp_path, capsys):
    # --cache-mode is the workflow's summary of the same files; a file beside
    # a report is that shard's own record, and wins.
    cold = _shard(tmp_path, 1, 20, 80, "cold")
    warm = _shard(tmp_path, 2, 99, 1, "warm")
    _run(gate, capsys, cold, warm, "--cache-mode", "warm")
    md = Path(str(cold) + ".md").read_text()
    assert "shard 1 cold, 20 of 100" in md and "shard 2 warm, 99 of 100" in md
    assert "**Compilation cache: mixed**" in md


@pytest.mark.parametrize("content", ["", "warmish", "WARM"])
def test_an_unreadable_mode_file_is_unknown_not_guessed(gate, tmp_path, capsys, content):
    report = _shard(tmp_path, 1, 99, 1, None)
    (tmp_path / "cache-mode-shard1.txt").write_text(content)
    other = _shard(tmp_path, 2, 99, 1, "warm")
    _run(gate, capsys, report, other, "--cache-mode", "warm")
    md = Path(str(report) + ".md").read_text()
    assert "shard 1 unknown" in md and "**Compilation cache: mixed**" in md


def test_a_cold_run_is_not_said_to_pay_every_compile_in_full(gate, tmp_path, capsys):
    # A cold CI shard reads back 17-25% of its lookups from programs it
    # compiled earlier in the same run.
    report = _shard(tmp_path, 1, 20, 80, "cold")
    _run(gate, capsys, report)
    md = Path(str(report) + ".md").read_text()
    assert "**Compilation cache: cold**" in md
    assert "Every compile is paid in full" not in md
    assert "reads it back from the cache the run is writing" in md


def _off_report(tmp_path, files):
    cases = [_case("tests/a/test_first.py", "test_before", 0.1, jax={})]
    cases += [_case(f, f"test_{i}", 0.1, jax={"jax_cache_hits": 1, "jax_cache_misses": 2})
              for i, f in enumerate(files)]
    return _report(tmp_path, *cases, name="test-results-shard4.xml")


def test_an_off_run_whose_later_tests_hit_a_cache_is_flagged(gate, tmp_path, capsys):
    """The slow lane claims no cache; a test that leaves one on makes that false.

    ``tests/core/test_compile_cache.py`` switched a persistent cache on and
    never off, and 858 of one slow shard's 1503 tests then ran with it,
    under a summary that said "Compilation cache: off".
    """
    report = _off_report(tmp_path, ["tests/core/test_leaks.py", "tests/nodes/test_after.py",
                                    "tests/nodes/test_after.py"])
    _, out = _run(gate, capsys, report, "--cache-mode", "off", "--fail-over", "0")
    md = Path(str(report) + ".md").read_text()
    warning = [ln for ln in out.splitlines() if ln.startswith("::warning title=Compilation cache::")]
    assert len(warning) == 1, out
    assert "3 tests recorded persistent-cache lookups (hits 3, misses 6)" in warning[0]
    assert "from tests/core/test_leaks.py::test_0 on" in warning[0]
    assert "cache claimed off, but 3 tests recorded persistent-cache lookups" in md
    assert "left it on" in md


def test_an_off_run_with_lookups_in_one_file_is_noted_without_a_warning(gate, tmp_path, capsys):
    # A file whose tests configure a cache of their own records lookups; a
    # cache it left on would show in the files after it.
    report = _off_report(tmp_path, ["tests/core/test_own_cache.py"] * 2)
    _, out = _run(gate, capsys, report, "--cache-mode", "off", "--fail-over", "0")
    md = Path(str(report) + ".md").read_text()
    assert "::warning title=Compilation cache::" not in out
    assert "cache claimed off, but 2 tests recorded" in md and "All in one file" in md


def test_an_off_run_without_lookups_is_not_flagged(gate, tmp_path, capsys):
    report = _off_report(tmp_path, [])
    _, out = _run(gate, capsys, report, "--cache-mode", "off", "--fail-over", "0")
    assert "Compilation cache::" not in out
    assert "cache claimed off, but" not in Path(str(report) + ".md").read_text()


def test_a_skipped_or_failed_allowlisted_test_is_not_called_removable(gate, tmp_path, capsys):
    # Fast because it did not run, or stopped early: that says nothing about
    # what it costs when it passes.
    report = _report(tmp_path,
                     _case("tests/a/test_x.py", "test_passed_fast", 0.2),
                     _case("tests/a/test_x.py", "test_skipped", 0.0,
                           extra='<skipped type="pytest.skip" message="no tool"/>'),
                     _case("tests/a/test_x.py", "test_failed", 0.3,
                           extra='<failure message="assert 0">assert 0</failure>'),
                     _case("tests/a/test_x.py", "test_errored", 0.1,
                           extra='<error message="fixture failed">boom</error>'))
    allow = tmp_path / "allow.txt"
    allow.write_text("".join(f"tests/a/test_x.py::{n} # kept: x\n"
                             for n in ("test_passed_fast", "test_skipped", "test_failed",
                                       "test_errored")))
    _run(gate, capsys, report, "--allowlist", allow, "--cache-mode", "cold")
    removable = Path(str(report) + ".md").read_text().split("may be removable")[1]
    assert "`tests/a/test_x.py::test_passed_fast`" in removable
    for name in ("test_skipped", "test_failed", "test_errored"):
        assert name not in removable, name
