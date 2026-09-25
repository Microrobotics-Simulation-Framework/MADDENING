#!/usr/bin/env python
"""Report per-test wall-clock from CI, and stop new slow tests landing.

The default CI lane (everything not marked ``@pytest.mark.slow``) grew to
55-78 minutes per lane, and nothing said where the time went: pytest's
``-v`` output carries no durations, and reconstructing them meant scraping
timestamps out of an hour of log.  This script reads the JUnit XML that
pytest writes (``--junitxml``) and puts the answer on the run page.

The time budget it enforces (see ``docs/developer_guide/testing_standards.md``):

under ``--watch-over`` (1 s)
    Fine.
between ``--watch-over`` and ``--slow-over`` (1-5 s)
    The watch list: shown in the summary, worth optimising, not an error.
over ``--slow-over`` (5 s)
    Belongs under ``@pytest.mark.slow`` -- which still runs in
    ``slow-tests.yml``: on a schedule three times a week on ``main`` only
    (GitHub runs a scheduled workflow on the default branch), and on a
    release branch only when someone dispatches it by hand -- unless it
    can be made much faster, or it guards something important enough to
    pay for on every push.  A kept test goes on the allowlist *with the
    reason*.  Unlisted tests over this line get a warning annotation on
    the run.
over ``--fail-over`` (20 s)
    An unlisted test this slow fails the job.

Why the hard line is four times the policy line
-----------------------------------------------
Measured on eight green lane-runs of the same tree (2026-09-24): a test
whose median was over 2 s varied between runs by 1.7x at the median and
3.1x at the 90th percentile (2.5x even after normalising each run by its
total).  Runners differ, and a test that compiles something first pays for
every later test that reuses it, so moving one test moves another's time.
A hard gate at 5 s would fail on noise every day.  At 20 s it fires on the
tests that actually drag the lane: the ten slowest tests took 17 minutes.

Why a test is slow
------------------
When the run sets ``MADDENING_TEST_JAX_TIMING=1``, ``tests/_jax_timing.py``
records each test's JAX tracing, lowering, XLA compile and cache-read time,
and how many processes it started, as JUnit properties, and the summary
splits every slow test into

* **compiling** -- XLA backend compilation, the only part a persistent
  compilation cache removes.  JAX's compile timer wraps the whole
  compile-or-read-the-cache call, so on a cache hit what it times *is*
  the cache read; the read is subtracted here and shown on its own;
* **cache read** -- reading compiled programs back from the persistent
  cache, which a warm run pays and a cold run does not;
* **tracing / lowering** -- building the program in Python; no cache
  removes it;
* **running** -- the rest: executing, Python overhead, I/O, and anything a
  subprocess does (its JAX events are not seen here).
  An un-jitted ``for`` loop over ``node.update`` lands here.

JAX's timers can overlap (on jax 0.10.2 a test's tracing has summed to
more than its wall time), so a share is capped at 100% and the diagnosis
says when the parts add up to more than the whole.

Subtracting the compile time gives the time a warm cache cannot remove, on
any run, cold or warm.  A test still over the policy line after that is
listed under *Slow even with a warm cache*: it needs a code change, not a
cache -- unless it started a process, whose compiles are invisible here
and which inherits the cache directory.  Those are listed separately as
*work in a subprocess (not measured here)*.  (A report written before the
process count was recorded cannot tell; there, a test with no JAX activity
at all in the pytest process is put in that list.)

Warm and cold runs
------------------
``--cache-mode`` says what the run's compilation cache was: ``off`` (none),
``cold`` (started empty), ``warm`` (restored from an earlier run), or
``mixed`` (a lane whose shards differed, or one where a shard's mode was
not recorded).  A
warm run's times are not comparable with a cold run's, so the header says
which it was, and a warm run does not list allowlist entries as removable
-- a test that is only fast because its compile was cached is still slow.

The allowlist
-------------
``tests/duration_allowlist.txt``: one pytest node id per line, then
`` # `` and the reason it is there.  It holds two kinds of entry:

* ``pending triage`` -- tests that were already over the policy line when
  this gate arrived.  Each is to be marked slow, made faster, or kept with
  a reason; the list only shrinks.
* ``kept: <why>`` -- a deliberate decision that a slow test must run on
  every push.

Listed tests that no longer run in this lane (marked slow, renamed,
deleted) or now finish under the policy line are reported as removable;
that is advisory, because one fast run is not proof.  A single shard's
report is not the lane, so the per-shard gate writes no summary and the
lane summary passes ``--no-removable`` when a shard's report is missing.

Usage
-----
::

    MADDENING_TEST_JAX_TIMING=1 python -m pytest ... \\
        --junitxml=test-results.xml -o junit_family=xunit1
    python scripts/report_test_durations.py test-results.xml \\
        --allowlist tests/duration_allowlist.txt --cache-mode cold

Writes Markdown to ``--markdown`` (default ``$GITHUB_STEP_SUMMARY`` when
set), GitHub annotations to stdout, and exits

* 0 -- within budget,
* 1 -- an unlisted test over ``--fail-over``,
* 2 -- nothing to judge: a report missing, unreadable, with no test
  cases, or holding only collection errors (a test file that fails to
  import stops pytest before any test runs).  A gate with nothing to read
  does not pass.  When only some of several reports are collection
  errors, those are left out, a warning names them, and no allowlist entry
  is listed as removable.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

#: GitHub renders at most ten warning annotations per step; the rest are
#: dropped silently, so the summary table is the complete list.
ANNOTATION_LIMIT = 10

#: The properties ``tests/_jax_timing.py`` writes.
JAX_PROPERTIES = ("jax_trace_s", "jax_lower_s", "jax_compile_s", "jax_cache_read_s",
                  "jax_cache_hits", "jax_cache_misses")
#: Processes the test started (``tests/_jax_timing.py``); absent from
#: reports written before it was recorded, which is kept as ``None``.
SUBPROCESS_PROPERTY = "subprocesses"

#: The ``message`` pytest's JUnit writer gives a file that failed to import.
COLLECTION_FAILURE = "collection failure"

CACHE_MODES = ("off", "cold", "warm", "mixed")


class TestTime(NamedTuple):
    nodeid: str
    file: str
    line: int | None
    seconds: float
    outcome: str
    #: ``{property: value}`` from ``tests/_jax_timing.py``; ``None`` when
    #: the run did not record them.
    jax: dict | None = None
    #: The entry pytest writes for a test file that failed to import.
    collection_error: bool = False

    @property
    def cache_read(self) -> float:
        """Reading programs back from the persistent cache; a warm run pays it."""
        return self.jax["jax_cache_read_s"] if self.jax else 0.0

    @property
    def compiling(self) -> float:
        """XLA compilation proper: what a warm cache removes.

        JAX times ``backend_compile_duration`` around the whole
        compile-or-read-the-cache call, and records the cache read
        (``cache_retrieval_time_sec``) *inside* it, so on a hit the
        "compile" is the read.  Taking the read out leaves the compile.
        """
        return max(0.0, self.jax["jax_compile_s"] - self.cache_read) if self.jax else 0.0

    @property
    def building(self) -> float:
        """Tracing plus lowering."""
        return self.jax["jax_trace_s"] + self.jax["jax_lower_s"] if self.jax else 0.0

    @property
    def running(self) -> float:
        if not self.jax:
            return self.seconds
        # The compile timer already contains the cache read: subtract each once.
        spent = self.compiling + self.cache_read + self.building
        return max(0.0, self.seconds - spent)

    @property
    def uncacheable(self) -> float:
        """What is left once a warm cache has removed the XLA compile.

        The cache read stays in: a warm run is the run that pays it.
        """
        return max(0.0, self.seconds - self.compiling)

    @property
    def overcounted(self) -> bool:
        """JAX's timers add up to more than the test's wall time.

        They can overlap (seen on jax 0.10.2: tracing alone over 100% of a
        test), so the split is then an approximation, and says so.
        """
        if not self.jax:
            return False
        return self.building + self.compiling + self.cache_read > self.seconds * 1.001 + 0.01

    @property
    def subprocess_work(self) -> bool:
        """Part of the test ran in another process, whose JAX events are not seen here.

        A child inherits the compilation cache directory, so a warm cache
        may speed it up although this process recorded no compile.  A
        report from before the process count was recorded cannot say; a
        test with no JAX activity at all in the pytest process is then
        counted here rather than called uncacheable.
        """
        if not self.jax:
            return False
        started = self.jax.get(SUBPROCESS_PROPERTY)
        if started is not None:
            return started > 0
        return not any(self.jax[k] for k in JAX_PROPERTIES)


class ReportError(Exception):
    """A report that cannot be judged (exit 2)."""


def _nodeid(case: ET.Element) -> tuple[str, str]:
    """Rebuild the pytest node id of a ``<testcase>`` and return it with its file.

    ``classname`` is the dotted module path followed by any class names
    (``tests.core.test_x.TestA``); ``file`` (``junit_family=xunit1``) is the
    path.  The class part is whatever follows the module's dotted path.
    """
    name = case.get("name", "")
    classname = case.get("classname", "")
    file = case.get("file")
    if not file:
        raise ReportError(
            "a <testcase> has no 'file' attribute; run pytest with "
            "`-o junit_family=xunit1` (xunit2 omits file and line)"
        )
    module = file[:-3].replace("/", ".").replace("\\", ".") if file.endswith(".py") else file
    classes = classname[len(module) + 1:] if classname.startswith(module + ".") else ""
    parts = [file, *[c for c in classes.split(".") if c], name]
    return "::".join(parts), file


def _jax_properties(case: ET.Element) -> dict | None:
    props = {p.get("name"): p.get("value") for p in case.iter("property")}
    if not any(k in props for k in JAX_PROPERTIES):
        return None
    try:
        out: dict = {k: float(props.get(k) or 0.0) for k in JAX_PROPERTIES}
        started = props.get(SUBPROCESS_PROPERTY)
        out[SUBPROCESS_PROPERTY] = float(started) if started is not None else None
        return out
    except ValueError as exc:
        raise ReportError(f"unreadable JAX timing property: {exc}") from exc


def read_report(path: Path) -> list[TestTime]:
    """Parse one pytest JUnit XML file into per-test times."""
    if not path.is_file():
        raise ReportError(f"{path}: no such report (did pytest run?)")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ReportError(f"{path}: not valid XML ({exc})") from exc
    out = []
    for case in root.iter("testcase"):
        nodeid, file = _nodeid(case)
        outcome, detail = "passed", None
        for tag in ("failure", "error", "skipped"):
            detail = case.find(tag)
            if detail is not None:
                outcome = tag
                break
        line = case.get("line")
        out.append(TestTime(
            nodeid=nodeid, file=file,
            line=int(line) + 1 if line is not None else None,  # 0-based in the XML
            seconds=float(case.get("time") or 0.0), outcome=outcome,
            jax=_jax_properties(case),
            collection_error=(outcome == "error" and detail is not None
                              and detail.get("message") == COLLECTION_FAILURE),
        ))
    return out


def read_allowlist(path: Path | None) -> dict[str, str]:
    """``{nodeid: reason}`` from the allowlist; ``' # '`` separates the reason."""
    if path is None:
        return {}
    entries = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        nodeid, _, reason = line.partition(" # ")
        nodeid = nodeid.strip()
        if not reason.strip():
            raise ReportError(
                f"{path}: {nodeid!r} has no reason; write `<node id> # <why>` "
                "(`pending triage` or `kept: ...`)"
            )
        entries[nodeid] = reason.strip()
    return entries


def _fmt(seconds: float) -> str:
    return f"{seconds:.1f} s" if seconds < 60 else f"{seconds / 60:.1f} min"


def _escape(text: str) -> str:
    """Escape for a GitHub workflow-command message."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def diagnose(t: TestTime) -> str:
    """Name the largest of running / tracing+lowering / compiling / cache read."""
    if not t.jax:
        return ""
    share = {"running": t.running, "tracing/lowering": t.building, "compiling": t.compiling,
             "cache read": t.cache_read}
    what = max(share, key=share.get)
    # Capped: overlapping timers can put one part over the whole.
    pct = min(100.0, 100 * share[what] / t.seconds) if t.seconds else 0
    hint = {
        "running": ("work in a subprocess (not measured here)" if t.subprocess_work
                    else "un-jitted loop, heavy compute, or a subprocess"),
        "tracing/lowering": "large or repeatedly rebuilt program; no cache removes this",
        "compiling": (f"{int(t.jax['jax_cache_misses'])} cache misses"
                      if t.jax["jax_cache_hits"] + t.jax["jax_cache_misses"]
                      else "a compilation cache would remove this"),
        "cache read": "reading compiled programs back; a warm run pays this",
    }[what]
    if t.overcounted:
        hint += "; JAX's timers add up to more than the wall time, so the split is approximate"
    return f"{what} {pct:.0f}%: {hint}"


def judge(tests, allow, *, watch_over, slow_over, fail_over):
    """Split the tests into the budget's bands; pure, so it can be tested."""
    by_id = {}
    for t in tests:  # several reports: keep each test's slowest time
        if t.nodeid not in by_id or t.seconds > by_id[t.nodeid].seconds:
            by_id[t.nodeid] = t
    ranked = sorted(by_id.values(), key=lambda t: -t.seconds)
    return {
        "ranked": ranked,
        "watch": [t for t in ranked if watch_over < t.seconds <= slow_over],
        "slow": [t for t in ranked if t.seconds > slow_over],
        "new_slow": [t for t in ranked if t.seconds > slow_over and t.nodeid not in allow],
        "failing": [t for t in ranked if t.seconds > fail_over and t.nodeid not in allow],
        "slow_warm": [t for t in ranked if t.jax and t.uncacheable > slow_over
                      and not t.subprocess_work],
        "slow_subprocess": [t for t in ranked if t.jax and t.uncacheable > slow_over
                            and t.subprocess_work],
        "allow_absent": sorted(set(allow) - set(by_id)),
        "allow_fast": [t for t in ranked if t.nodeid in allow and t.seconds <= slow_over],
    }


def markdown(verdict, allow, *, title, watch_over, slow_over, fail_over, top, cache_mode):
    ranked = verdict["ranked"]
    total = sum(t.seconds for t in ranked)
    timed = [t for t in ranked if t.jax]
    lines = [f"## {title}", ""]

    def band(sel):
        s = sum(t.seconds for t in sel)
        return f"{len(sel)} | {_fmt(s)} | {100 * s / total:.0f}%" if total else f"{len(sel)} | 0 s | -"

    fine = [t for t in ranked if t.seconds <= watch_over]
    lines += [
        f"{len(ranked)} tests, {_fmt(total)} of test time "
        f"(setup + call + teardown; collection and start-up excluded).",
        "",
    ]
    cache_line = {
        "off": "**Compilation cache: off.** Every compile is paid in full.",
        "cold": "**Compilation cache: cold** (started empty). Every compile is paid in full.",
        "warm": ("**Compilation cache: warm** (restored from an earlier run). Compiles of "
                 "unchanged programs were skipped, so these times are lower than a cold "
                 "run's; new and changed programs still compiled."),
        "mixed": ("**Compilation cache: mixed** -- some shards restored a cache and some ran "
                  "cold (a shard whose runner CPU model had no cache yet), or a shard did not "
                  "record its mode, so these times may mix warm and cold."),
    }.get(cache_mode, "**Compilation cache: not stated.**")
    if timed:
        hits = sum(t.jax["jax_cache_hits"] for t in timed)
        misses = sum(t.jax["jax_cache_misses"] for t in timed)
        split = [sum(t.running for t in timed), sum(t.building for t in timed),
                 sum(t.compiling for t in timed), sum(t.cache_read for t in timed)]
        cache_line += (f" Cache hits {int(hits)}, misses {int(misses)}. Of the test time: "
                       f"running {_fmt(split[0])}, tracing/lowering {_fmt(split[1])}, "
                       f"XLA compile {_fmt(split[2])} (cache reads excluded), "
                       f"cache reads {_fmt(split[3])}.")
    lines += [cache_line, "",
              "| band | tests | time | share |",
              "|---|---:|---:|---:|",
              f"| up to {watch_over:g} s: fine | {band(fine)} |",
              f"| {watch_over:g}-{slow_over:g} s: watch, optimise | {band(verdict['watch'])} |",
              f"| over {slow_over:g} s: mark slow, speed up, or keep with a reason | {band(verdict['slow'])} |",
              ""]
    if verdict["failing"]:
        lines += [f"### Over {fail_over:g} s and not on the allowlist -- this fails the job", ""]
        lines += [f"- `{t.nodeid}` -- {_fmt(t.seconds)} {diagnose(t)}" for t in verdict["failing"]] + [""]
    if verdict["new_slow"]:
        lines += [f"### Over {slow_over:g} s and not on the allowlist", ""]
        lines += [f"- `{t.nodeid}` -- {_fmt(t.seconds)} {diagnose(t)}" for t in verdict["new_slow"]] + [""]
    if verdict["slow_warm"]:
        lines += [f"### Slow even with a warm cache ({len(verdict['slow_warm'])})", "",
                  f"Still over {slow_over:g} s once XLA compilation is taken away. A cache "
                  "cannot fix these; they need a code change (jit or `lax.scan` a Python "
                  "loop, build the program once and reuse it) or `@pytest.mark.slow`.", "",
                  "| without compile | running | tracing/lowering | cache read | test | allowlist |",
                  "|---:|---:|---:|---:|---|---|"]
        for t in verdict["slow_warm"]:
            lines.append(f"| {_fmt(t.uncacheable)} | {_fmt(t.running)} | {_fmt(t.building)} "
                         f"| {_fmt(t.cache_read)} | `{t.nodeid}` | {allow.get(t.nodeid, '')} |")
        lines.append("")
    if verdict["slow_subprocess"]:
        lines += [f"### Work in a subprocess (not measured here) ({len(verdict['slow_subprocess'])})", "",
                  f"Over {slow_over:g} s once this process's XLA compilation is taken away, "
                  "but part of their work ran in a child process, whose tracing and compiling "
                  "this split cannot see. The child inherits the compilation cache directory, "
                  "so a warm cache may still speed them up: compare a warm run before calling "
                  "them slow even with a warm cache.", "",
                  "| without in-process compile | processes started | test | allowlist |",
                  "|---:|---:|---|---|"]
        for t in verdict["slow_subprocess"]:
            started = t.jax.get(SUBPROCESS_PROPERTY)
            lines.append(f"| {_fmt(t.uncacheable)} "
                         f"| {'not recorded' if started is None else int(started)} "
                         f"| `{t.nodeid}` | {allow.get(t.nodeid, '')} |")
        lines.append("")
    lines += [f"### Slowest {min(top, len(ranked))} tests", ""]
    if timed:
        lines += ["| time | running | tracing/lowering | compiling | cache read | why | test | allowlist |",
                  "|---:|---:|---:|---:|---:|---|---|---|"]
        for t in ranked[:top]:
            lines.append(f"| {_fmt(t.seconds)} | {_fmt(t.running)} | {_fmt(t.building)} "
                         f"| {_fmt(t.compiling)} | {_fmt(t.cache_read)} | {diagnose(t)} "
                         f"| `{t.nodeid}` | {allow.get(t.nodeid, '')} |")
    else:
        lines += ["| time | test | allowlist |", "|---:|---|---|"]
        for t in ranked[:top]:
            lines.append(f"| {_fmt(t.seconds)} | `{t.nodeid}` | {allow.get(t.nodeid, '')} |")
    files = defaultdict(lambda: [0.0, 0])
    for t in ranked:
        files[t.file][0] += t.seconds
        files[t.file][1] += 1
    lines += ["", "### Slowest files", "", "| time | tests | per test | file |", "|---:|---:|---:|---|"]
    for f, (s, n) in sorted(files.items(), key=lambda kv: -kv[1][0])[:15]:
        lines.append(f"| {_fmt(s)} | {n} | {s / n:.2f} s | `{f}` |")
    removable = verdict["allow_absent"] + [t.nodeid for t in verdict["allow_fast"]]
    if removable and cache_mode in ("warm", "mixed"):
        lines += ["", "Allowlist entries are not judged removable on a warm run: a test that "
                  "is only fast because its compile was cached is still slow. See the "
                  "after-merge (cold) runs."]
    elif removable:
        lines += ["", f"### Allowlist entries that may be removable ({len(removable)})", "",
                  "Not run in this lane (marked slow, renamed or deleted), or under "
                  f"{slow_over:g} s this run. One fast run is not proof; check the "
                  "other lane and a second run before deleting.", ""]
        lines += [f"- `{n}`" for n in removable]
    return "\n".join(lines) + "\n"


def annotations(verdict, *, slow_over, fail_over):
    out = []
    for t in verdict["failing"]:
        loc = f"file={t.file},line={t.line}," if t.line else f"file={t.file},"
        why = f" ({diagnose(t)})" if t.jax else ""
        out.append(f"::error {loc}title=Test over {fail_over:g} s::" + _escape(
            f"{t.nodeid} took {_fmt(t.seconds)}{why}. Mark it @pytest.mark.slow, make it "
            f"faster, or add it to tests/duration_allowlist.txt with the reason it "
            f"must run on every push."))
    failing = {t.nodeid for t in verdict["failing"]}
    for t in verdict["new_slow"]:
        if t.nodeid in failing:
            continue
        loc = f"file={t.file},line={t.line}," if t.line else f"file={t.file},"
        why = f" ({diagnose(t)})" if t.jax else ""
        out.append(f"::warning {loc}title=Test over {slow_over:g} s::" + _escape(
            f"{t.nodeid} took {_fmt(t.seconds)}{why}; the budget for the default lane "
            f"is {slow_over:g} s. Mark it slow or make it faster."))
    return out[:ANNOTATION_LIMIT]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("reports", nargs="+", type=Path, help="pytest --junitxml files")
    p.add_argument("--allowlist", type=Path, help="node ids exempt from the gate, with reasons")
    p.add_argument("--watch-over", type=float, default=1.0)
    p.add_argument("--slow-over", type=float, default=5.0)
    p.add_argument("--fail-over", type=float, default=20.0,
                   help="an unlisted test slower than this fails; 0 disables the gate")
    p.add_argument("--cache-mode", choices=CACHE_MODES,
                   help="the run's compilation cache: off, cold (started empty), warm, "
                        "or mixed (shards differed, or a shard's mode is unknown)")
    p.add_argument("--markdown", type=Path,
                   default=Path(os.environ["GITHUB_STEP_SUMMARY"]) if os.environ.get("GITHUB_STEP_SUMMARY") else None,
                   help="append the summary here (default: $GITHUB_STEP_SUMMARY)")
    p.add_argument("--no-removable", action="store_true",
                   help="list no allowlist entries as removable (the reports do not cover the whole lane)")
    p.add_argument("--title", default="Test durations")
    p.add_argument("--top", type=int, default=30)
    args = p.parse_args(argv)
    if not (0 < args.watch_over <= args.slow_over) or (args.fail_over and args.fail_over < args.slow_over):
        p.error("need 0 < --watch-over <= --slow-over <= --fail-over (or --fail-over 0)")

    try:
        reports = [(r, read_report(r)) for r in args.reports]
        # A file that fails to import stops pytest after collection, so its
        # report holds one collection-error entry and no test ran: it is a
        # missing report, not a fast one.
        broken = [r for r, cases in reports if cases and all(t.collection_error for t in cases)]
        tests = [t for r, cases in reports if r not in broken for t in cases]
        if not tests:
            raise ReportError(
                f"no test ran: every test case in {', '.join(map(str, broken))} is a "
                "collection error (a test file failed to import)" if broken else
                f"no test cases in {', '.join(map(str, args.reports))}")
        allow = read_allowlist(args.allowlist)
    except ReportError as exc:
        print(f"::error title=Test durations::{_escape(str(exc))}")
        print(f"report_test_durations: {exc}", file=sys.stderr)
        return 2
    if broken:
        print("::warning title=Test durations::" + _escape(
            f"left out {', '.join(map(str, broken))}: only collection errors, no test ran; "
            "no allowlist entry is listed as removable"))
        args.no_removable = True

    fail_over = args.fail_over or float("inf")
    verdict = judge(tests, allow, watch_over=args.watch_over,
                    slow_over=args.slow_over, fail_over=fail_over)
    if args.no_removable:
        # An entry absent from a partial view may simply be in the part
        # that is missing.
        verdict["allow_absent"], verdict["allow_fast"] = [], []
    md = markdown(verdict, allow, title=args.title, watch_over=args.watch_over,
                  slow_over=args.slow_over, fail_over=fail_over, top=args.top,
                  cache_mode=args.cache_mode)
    if args.markdown:
        with open(args.markdown, "a", encoding="utf-8") as fh:
            fh.write(md)
    else:
        print(md)
    for a in annotations(verdict, slow_over=args.slow_over, fail_over=fail_over):
        print(a)

    ranked = verdict["ranked"]
    print(f"{len(ranked)} tests; {len(verdict['watch'])} in the "
          f"{args.watch_over:g}-{args.slow_over:g} s watch band; "
          f"{len(verdict['slow'])} over {args.slow_over:g} s "
          f"({len(verdict['new_slow'])} not allowlisted; "
          f"{len(verdict['slow_warm'])} slow even with a warm cache; "
          f"{len(verdict['slow_subprocess'])} with work in a subprocess, not measured here); "
          + (f"{len(verdict['failing'])} unlisted over {args.fail_over:g} s."
             if args.fail_over else "hard line off (--fail-over 0)."))
    return 1 if verdict["failing"] else 0


if __name__ == "__main__":
    sys.exit(main())
