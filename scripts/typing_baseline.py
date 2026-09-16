#!/usr/bin/env python
"""Summarise a pyright run over ``src/maddening`` by rule and by file.

Runs ``pyright --outputjson`` with the repository's ``pyrightconfig.json``
(or reads a stored ``--outputjson`` file with ``--json``) and prints the
severity totals, the per-rule counts and the files with the most errors.
This is the table recorded in ``docs/developer_guide/typing.md``; re-run
it to measure progress against that baseline.

The exit status is always 0 unless ``--fail-on-errors`` is given: the
type check is non-blocking in phase 1 of the typing policy (see the
developer guide).  ``--markdown`` emits GitHub-flavoured Markdown for
``$GITHUB_STEP_SUMMARY``.

Examples
--------
::

    python scripts/typing_baseline.py                 # run pyright, plain text
    python scripts/typing_baseline.py --markdown      # for a CI step summary
    python scripts/typing_baseline.py --json out.json # summarise a stored run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SEVERITIES = ("error", "warning", "information")


@dataclass
class Summary:
    """Counts extracted from one pyright ``--outputjson`` document."""

    files_analyzed: int = 0
    totals: dict[str, int] = field(default_factory=dict)
    by_rule: dict[tuple[str, str], int] = field(default_factory=dict)
    errors_by_file: dict[str, int] = field(default_factory=dict)
    source_files_with_diagnostics: int = 0

    @property
    def error_count(self) -> int:
        return self.totals.get("error", 0)

    @property
    def warning_count(self) -> int:
        return self.totals.get("warning", 0)


def summarise(report: dict, root: Path | None = None) -> Summary:
    """Aggregate a pyright ``--outputjson`` document.

    Parameters
    ----------
    report : dict
        Parsed JSON as written by ``pyright --outputjson``.  Only the
        ``summary`` and ``generalDiagnostics`` keys are read.
    root : Path, optional
        Paths in the per-file table are made relative to this directory
        when they lie under it (defaults to the repository root).

    Returns
    -------
    Summary
    """
    root = (root or REPO_ROOT).resolve()
    diags = report.get("generalDiagnostics", [])
    totals: Counter[str] = Counter()
    by_rule: Counter[tuple[str, str]] = Counter()
    by_file: Counter[str] = Counter()
    files: set[str] = set()
    for d in diags:
        sev = d.get("severity", "error")
        rule = d.get("rule") or "<no rule>"
        totals[sev] += 1
        by_rule[(sev, rule)] += 1
        path = _relative(d.get("file", "<unknown>"), root)
        files.add(path)
        if sev == "error":
            by_file[path] += 1
    summary = report.get("summary", {})
    return Summary(
        files_analyzed=int(summary.get("filesAnalyzed", 0)),
        totals=dict(totals),
        by_rule=dict(by_rule),
        errors_by_file=dict(by_file),
        source_files_with_diagnostics=len(files),
    )


def _relative(path: str, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return path


def render(summary: Summary, top: int = 15, markdown: bool = False) -> str:
    """Render a summary as plain text or Markdown tables."""
    lines: list[str] = []
    h = (lambda s: f"### {s}") if markdown else (lambda s: s.upper())
    lines.append(h("pyright summary"))
    lines.append("")
    lines.append(
        f"files analysed: {summary.files_analyzed}; "
        f"files with diagnostics: {summary.source_files_with_diagnostics}; "
        + "; ".join(f"{s}s: {summary.totals.get(s, 0)}" for s in SEVERITIES)
    )
    lines.append("")

    rows = sorted(summary.by_rule.items(), key=lambda kv: (-kv[1], kv[0]))
    lines.append(h("diagnostics by rule"))
    lines.append("")
    lines.extend(_table(["severity", "rule", "count"],
                        [[sev, rule, str(n)] for (sev, rule), n in rows],
                        markdown))
    lines.append("")

    files = sorted(summary.errors_by_file.items(),
                   key=lambda kv: (-kv[1], kv[0]))[:top]
    lines.append(h(f"top {top} files by error count"))
    lines.append("")
    lines.extend(_table(["errors", "file"],
                        [[str(n), f] for f, n in files],
                        markdown))
    return "\n".join(lines) + "\n"


def _table(header: list[str], rows: list[list[str]], markdown: bool) -> list[str]:
    if not rows:
        return ["(none)"]
    if markdown:
        out = ["| " + " | ".join(header) + " |",
               "|" + "|".join("---" for _ in header) + "|"]
        out.extend("| " + " | ".join(r) + " |" for r in rows)
        return out
    widths = [max(len(x) for x in col) for col in zip(header, *rows)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*header), fmt.format(*("-" * w for w in widths))]
    out.extend(fmt.format(*r) for r in rows)
    return out


def run_pyright(pyright: str, extra_args: list[str]) -> dict:
    """Run ``pyright --outputjson`` from the repository root and parse it.

    pyright exits 1 when it reports errors, so the exit status is ignored;
    only an unparsable stdout is treated as a failure.
    """
    cmd = [*pyright.split(), "--outputjson", *extra_args]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(
            f"pyright did not produce JSON (exit {proc.returncode}); "
            f"install it with `pip install pyright` or pass --pyright"
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", type=Path,
                    help="summarise this stored `pyright --outputjson` file "
                         "instead of running pyright")
    ap.add_argument("--pyright", default=os.environ.get("PYRIGHT", "pyright"),
                    help="pyright command (default: %(default)s; env PYRIGHT)")
    ap.add_argument("--top", type=int, default=15,
                    help="files to list in the per-file table (default 15)")
    ap.add_argument("--markdown", action="store_true",
                    help="emit Markdown tables (for $GITHUB_STEP_SUMMARY)")
    ap.add_argument("--fail-on-errors", action="store_true",
                    help="exit 1 when pyright reported any error")
    ap.add_argument("pyright_args", nargs="*",
                    help="extra arguments forwarded to pyright")
    args = ap.parse_args(argv)

    if args.json is not None:
        report = json.loads(args.json.read_text())
    else:
        report = run_pyright(args.pyright, args.pyright_args)
    summary = summarise(report)
    sys.stdout.write(render(summary, top=args.top, markdown=args.markdown))
    return 1 if (args.fail_on_errors and summary.error_count) else 0


if __name__ == "__main__":
    raise SystemExit(main())
