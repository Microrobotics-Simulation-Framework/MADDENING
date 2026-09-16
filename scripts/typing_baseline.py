#!/usr/bin/env python
"""Summarise a pyright run over ``src/maddening`` by rule and by file.

Runs ``pyright --outputjson`` with the repository's ``pyrightconfig.json``
(or reads a stored ``--outputjson`` file with ``--json``) and prints the
severity totals, the per-rule counts and the files with the most errors.
This is the table recorded in ``docs/developer_guide/typing.md``; re-run
it to measure progress against that baseline.

Exit status
-----------
0
    pyright ran to completion and reported no errors (or errors are not
    being counted because ``--fail-on-errors`` is absent).
1
    pyright ran to completion and reported errors; only with
    ``--fail-on-errors``.  Phase 1 of the typing policy is non-blocking,
    so CI keeps this exit non-fatal (see the developer guide).
2
    *Infrastructure failure*: the numbers cannot be trusted and no
    error count is reported.  The pyright command is missing, exited
    with a code other than 0/1 (3 = configuration could not be parsed),
    did not produce JSON, analysed zero files (a broken ``include`` or
    checkout), did not resolve a core import (``jax``, ``numpy``: a wrong
    interpreter turns real errors into ``Unknown`` and *lowers* the
    count), or reported more ``reportMissingImports`` diagnostics than
    ``--max-missing-imports`` allows.  Whatever pyright wrote to stderr
    is always forwarded.

``--markdown`` emits GitHub-flavoured Markdown for ``$GITHUB_STEP_SUMMARY``.
Arguments for pyright itself go after ``--``.

Examples
--------
::

    python scripts/typing_baseline.py                 # run pyright, plain text
    python scripts/typing_baseline.py --markdown      # for a CI step summary
    python scripts/typing_baseline.py --json out.json # summarise a stored run
    python scripts/typing_baseline.py --pyright "uvx pyright" -- --pythonpath .venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SEVERITIES = ("error", "warning", "information")

EXIT_OK = 0
EXIT_ERRORS = 1
EXIT_INFRASTRUCTURE = 2

#: Top-level modules that every environment pyright is meant to run in
#: resolves (the base ``dependencies`` in ``pyproject.toml``).  One
#: ``reportMissingImports`` diagnostic naming any of them means pyright
#: looked at the wrong interpreter and its error count is meaningless.
CORE_IMPORTS = ("jax", "numpy", "yaml")

#: Ceiling for ``reportMissingImports`` diagnostics before the run counts
#: as an environment failure.  The phase-1 baseline has 18 (all optional
#: extras: pygfx, gi, cupy, fsspec, rendercanvas, skimage); a run against
#: an interpreter without the dependencies had 201.
DEFAULT_MAX_MISSING_IMPORTS = 40

#: pyright exit codes that mean "analysis completed" (1 = errors found).
_PYRIGHT_COMPLETED = (0, 1)

_MISSING_IMPORT_RE = re.compile(r'Import "([^"]+)"')

_INSTALL_HINT = "install it with `pip install pyright` or pass --pyright"


class InfrastructureFailure(Exception):
    """The pyright run cannot be trusted; no error count is reported."""


@dataclass
class Summary:
    """Counts extracted from one pyright ``--outputjson`` document."""

    files_analyzed: int = 0
    totals: dict[str, int] = field(default_factory=dict)
    by_rule: dict[tuple[str, str], int] = field(default_factory=dict)
    errors_by_file: dict[str, int] = field(default_factory=dict)
    source_files_with_diagnostics: int = 0
    has_summary_block: bool = False
    missing_imports: dict[str, int] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return self.totals.get("error", 0)

    @property
    def warning_count(self) -> int:
        return self.totals.get("warning", 0)

    @property
    def missing_import_count(self) -> int:
        return sum(self.missing_imports.values())


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
    missing: Counter[str] = Counter()
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
        if rule == "reportMissingImports":
            m = _MISSING_IMPORT_RE.search(d.get("message", ""))
            missing[m.group(1) if m else "<unknown>"] += 1
    summary = report.get("summary")
    has_summary = isinstance(summary, dict) and "filesAnalyzed" in summary
    return Summary(
        files_analyzed=int(summary["filesAnalyzed"]) if has_summary else 0,
        totals=dict(totals),
        by_rule=dict(by_rule),
        errors_by_file=dict(by_file),
        source_files_with_diagnostics=len(files),
        has_summary_block=has_summary,
        missing_imports=dict(missing),
    )


def _relative(path: str, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except (ValueError, OSError):
        return path


def check_run(summary: Summary, *, min_files: int = 1,
              max_missing_imports: int = DEFAULT_MAX_MISSING_IMPORTS,
              core_imports: tuple[str, ...] = CORE_IMPORTS) -> None:
    """Refuse a run whose numbers cannot be trusted.

    Raises
    ------
    InfrastructureFailure
        When the ``summary`` block is missing, fewer than ``min_files``
        files were analysed, a core dependency did not resolve, or more
        than ``max_missing_imports`` imports did not resolve.
    """
    if not summary.has_summary_block:
        raise InfrastructureFailure(
            "pyright output has no `summary` block; the run did not complete")
    if summary.files_analyzed < min_files:
        raise InfrastructureFailure(
            f"pyright analysed {summary.files_analyzed} files (minimum "
            f"{min_files}); check `include` in pyrightconfig.json and the "
            "checkout (pyright reports a missing include path only on stderr)")
    unresolved = sorted(m for m in summary.missing_imports
                        if m.split(".")[0] in core_imports)
    if unresolved:
        raise InfrastructureFailure(
            "pyright could not resolve core imports "
            f"{', '.join(unresolved)}; it looked at the wrong interpreter "
            "(pass `-- --pythonpath /path/to/python`), so the error count "
            "would be meaningless")
    if summary.missing_import_count > max_missing_imports:
        raise InfrastructureFailure(
            f"{summary.missing_import_count} reportMissingImports diagnostics "
            f"exceed --max-missing-imports {max_missing_imports}; the "
            "environment is missing dependencies, so the error count would "
            "be meaningless")


def render(summary: Summary, top: int = 15, markdown: bool = False) -> str:
    """Render a summary as plain text or Markdown tables."""
    lines: list[str] = []
    h = (lambda s: f"### {s}") if markdown else (lambda s: s.upper())
    code = (lambda s: f"`{s}`") if markdown else (lambda s: s)
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
                        [[str(n), code(f)] for f, n in files],
                        markdown))
    return "\n".join(lines) + "\n"


def _md_cell(text: str) -> str:
    # A pipe would end the cell.  File names are code spans already (see
    # ``render``), which stops ``__init__`` from rendering as emphasis.
    return text.replace("|", "\\|")


def _table(header: list[str], rows: list[list[str]], markdown: bool) -> list[str]:
    if not rows:
        return ["(none)"]
    if markdown:
        out = ["| " + " | ".join(header) + " |",
               "|" + "|".join("---" for _ in header) + "|"]
        out.extend("| " + " | ".join(_md_cell(c) for c in r) + " |" for r in rows)
        return out
    widths = [max(len(x) for x in col) for col in zip(header, *rows)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    out = [fmt.format(*header), fmt.format(*("-" * w for w in widths))]
    out.extend(fmt.format(*r) for r in rows)
    return out


def _pythonpath_from(extra_args: list[str]) -> str | None:
    """The interpreter given as ``--pythonpath X`` / ``--pythonpath=X``."""
    for i, a in enumerate(extra_args):
        if a == "--pythonpath" and i + 1 < len(extra_args):
            return extra_args[i + 1]
        if a.startswith("--pythonpath="):
            return a.split("=", 1)[1]
    return None


def check_interpreter(python: str, module: str = "numpy") -> None:
    """Refuse a ``--pythonpath`` interpreter that pyright would accept silently.

    pyright analyses on whether or not the path exists or has the
    dependencies installed; the only symptom is a lower error count.

    Raises
    ------
    InfrastructureFailure
    """
    path = Path(python)
    if not path.exists():
        raise InfrastructureFailure(f"--pythonpath {python}: no such interpreter")
    try:
        proc = subprocess.run([str(path), "-c", f"import {module}"],
                              capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise InfrastructureFailure(
            f"--pythonpath {python}: cannot run it ({e})") from e
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise InfrastructureFailure(
            f"--pythonpath {python}: `import {module}` fails (exit "
            f"{proc.returncode}); it is not the environment the package is "
            "installed in")


def run_pyright(pyright: str, extra_args: list[str]) -> dict:
    """Run ``pyright --outputjson`` from the repository root and parse it.

    pyright's stderr is always forwarded: it is where a missing include
    path or an unparsable ``pyrightconfig.json`` is reported, and stdout
    is still valid JSON in both cases.

    Raises
    ------
    InfrastructureFailure
        When the command is missing, exits with a code other than 0/1
        (1 = errors found; 2 = fatal, 3 = config error, 4 = illegal
        arguments) or does not produce a JSON object.
    """
    cmd = [*pyright.split(), "--outputjson", *extra_args]
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    except OSError as e:
        raise InfrastructureFailure(
            f"cannot run pyright as `{pyright}` ({e}); {_INSTALL_HINT}") from e
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        if not proc.stderr.endswith("\n"):
            sys.stderr.write("\n")
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError:
        sys.stderr.write(proc.stdout)
        raise InfrastructureFailure(
            f"pyright did not produce JSON (exit {proc.returncode}); "
            f"{_INSTALL_HINT}") from None
    if proc.returncode not in _PYRIGHT_COMPLETED:
        raise InfrastructureFailure(
            f"pyright exited {proc.returncode} (3 = pyrightconfig.json could "
            "not be parsed, in which case it analysed the whole tree with the "
            "default config); see its stderr above")
    if not isinstance(report, dict):
        raise InfrastructureFailure("pyright JSON output is not an object")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Arguments for pyright itself go after `--`, e.g. "
               "`-- --pythonpath /path/to/python`.")
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
                    help=f"exit {EXIT_ERRORS} when pyright reported any error "
                         f"(infrastructure failures exit {EXIT_INFRASTRUCTURE} "
                         "regardless)")
    ap.add_argument("--min-files", type=int, default=1,
                    help="fewer analysed files is an infrastructure failure "
                         "(default %(default)s)")
    ap.add_argument("--max-missing-imports", type=int,
                    default=DEFAULT_MAX_MISSING_IMPORTS,
                    help="more reportMissingImports diagnostics is an "
                         "environment failure (default %(default)s; the "
                         "baseline has 18, all optional extras)")
    ap.add_argument("pyright_args", nargs="*",
                    help="extra arguments forwarded to pyright; put them "
                         "after `--` (e.g. `-- --pythonpath PYTHON`; that "
                         "interpreter is checked to exist and import numpy "
                         "before pyright runs)")
    args = ap.parse_args(argv)

    try:
        if args.json is not None:
            report = json.loads(args.json.read_text())
        else:
            python = _pythonpath_from(args.pyright_args)
            if python is not None:
                check_interpreter(python)
            report = run_pyright(args.pyright, args.pyright_args)
        summary = summarise(report)
        check_run(summary, min_files=args.min_files,
                  max_missing_imports=args.max_missing_imports)
    except InfrastructureFailure as e:
        sys.stderr.write(f"typing_baseline: infrastructure failure: {e}\n")
        return EXIT_INFRASTRUCTURE
    sys.stdout.write(render(summary, top=args.top, markdown=args.markdown))
    if args.fail_on_errors and summary.error_count:
        return EXIT_ERRORS
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
