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
    Belongs under ``@pytest.mark.slow`` -- which still runs three times a
    week in ``slow-tests.yml`` -- unless it can be made much faster, or it
    guards something important enough to pay for on every push.  A kept
    test goes on the allowlist *with the reason*.  Unlisted tests over
    this line get a warning annotation on the run.
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

    python -m pytest ... --junitxml=test-results.xml -o junit_family=xunit1
    python scripts/report_test_durations.py test-results.xml \\
        --allowlist tests/duration_allowlist.txt

Writes Markdown to ``--markdown`` (default ``$GITHUB_STEP_SUMMARY`` when
set), GitHub annotations to stdout, and exits

* 0 -- within budget,
* 1 -- an unlisted test over ``--fail-over``,
* 2 -- nothing to judge: a report missing, unreadable, or with no test
  cases.  A gate with nothing to read does not pass.
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


class TestTime(NamedTuple):
    nodeid: str
    file: str
    line: int | None
    seconds: float
    outcome: str


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
        outcome = "passed"
        for tag in ("failure", "error", "skipped"):
            if case.find(tag) is not None:
                outcome = tag
                break
        line = case.get("line")
        out.append(TestTime(
            nodeid=nodeid, file=file,
            line=int(line) + 1 if line is not None else None,  # 0-based in the XML
            seconds=float(case.get("time") or 0.0), outcome=outcome,
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
        "allow_absent": sorted(set(allow) - set(by_id)),
        "allow_fast": [t for t in ranked if t.nodeid in allow and t.seconds <= slow_over],
    }


def markdown(verdict, allow, *, title, watch_over, slow_over, fail_over, top):
    ranked = verdict["ranked"]
    total = sum(t.seconds for t in ranked)
    lines = [f"## {title}", ""]

    def band(sel):
        s = sum(t.seconds for t in sel)
        return f"{len(sel)} | {_fmt(s)} | {100 * s / total:.0f}%" if total else f"{len(sel)} | 0 s | -"

    fine = [t for t in ranked if t.seconds <= watch_over]
    lines += [
        f"{len(ranked)} tests, {_fmt(total)} of test time "
        f"(setup + call + teardown; collection and start-up excluded).",
        "",
        "| band | tests | time | share |",
        "|---|---:|---:|---:|",
        f"| up to {watch_over:g} s: fine | {band(fine)} |",
        f"| {watch_over:g}-{slow_over:g} s: watch, optimise | {band(verdict['watch'])} |",
        f"| over {slow_over:g} s: mark slow, speed up, or keep with a reason | {band(verdict['slow'])} |",
        "",
    ]
    if verdict["failing"]:
        lines += [f"### Over {fail_over:g} s and not on the allowlist -- this fails the job", ""]
        lines += [f"- `{t.nodeid}` -- {_fmt(t.seconds)}" for t in verdict["failing"]] + [""]
    if verdict["new_slow"]:
        lines += [f"### Over {slow_over:g} s and not on the allowlist", ""]
        lines += [f"- `{t.nodeid}` -- {_fmt(t.seconds)}" for t in verdict["new_slow"]] + [""]
    lines += [f"### Slowest {min(top, len(ranked))} tests", "",
              "| time | test | allowlist |", "|---:|---|---|"]
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
    if removable:
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
        out.append(f"::error {loc}title=Test over {fail_over:g} s::" + _escape(
            f"{t.nodeid} took {_fmt(t.seconds)}. Mark it @pytest.mark.slow, make it "
            f"faster, or add it to tests/duration_allowlist.txt with the reason it "
            f"must run on every push."))
    failing = {t.nodeid for t in verdict["failing"]}
    for t in verdict["new_slow"]:
        if t.nodeid in failing:
            continue
        loc = f"file={t.file},line={t.line}," if t.line else f"file={t.file},"
        out.append(f"::warning {loc}title=Test over {slow_over:g} s::" + _escape(
            f"{t.nodeid} took {_fmt(t.seconds)}; the budget for the default lane "
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
        tests = [t for r in args.reports for t in read_report(r)]
        if not tests:
            raise ReportError(f"no test cases in {', '.join(map(str, args.reports))}")
        allow = read_allowlist(args.allowlist)
    except ReportError as exc:
        print(f"::error title=Test durations::{_escape(str(exc))}")
        print(f"report_test_durations: {exc}", file=sys.stderr)
        return 2

    fail_over = args.fail_over or float("inf")
    verdict = judge(tests, allow, watch_over=args.watch_over,
                    slow_over=args.slow_over, fail_over=fail_over)
    if args.no_removable:
        # An entry absent from a partial view may simply be in the part
        # that is missing.
        verdict["allow_absent"], verdict["allow_fast"] = [], []
    md = markdown(verdict, allow, title=args.title, watch_over=args.watch_over,
                  slow_over=args.slow_over, fail_over=fail_over, top=args.top)
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
          f"({len(verdict['new_slow'])} not allowlisted); "
          + (f"{len(verdict['failing'])} unlisted over {args.fail_over:g} s."
             if args.fail_over else "hard line off (--fail-over 0)."))
    return 1 if verdict["failing"] else 0


if __name__ == "__main__":
    sys.exit(main())
