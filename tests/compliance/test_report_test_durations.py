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


def _case(file, name, seconds, cls="", line=10, extra="", jax=None):
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
    assert "running 9.0 s, tracing/lowering 0.0 s, XLA compile 9.0 s" in md
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
        assert ("not judged removable on a warm run" in md) is (not listed), mode


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
