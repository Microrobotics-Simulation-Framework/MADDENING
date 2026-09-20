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

``--tier NAME`` restricts every count to the top-level packages that
``typing_tiers.json`` lists for that tier, and turns the run into a
*gate*: each package's error count is compared against the ceiling
recorded there and exit 1 means at least one exceeded it.  Tier 1's
ceilings are all zero; tier 2 is ratcheted, so its ceilings are the
measured counts and may only ever be lowered.  Because the numbers only
mean anything in the environment they were taken in, the tier file also
records the pyright release, and a run under a different one is an
infrastructure failure (exit 2) rather than a comparison nobody can
trust.

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

#: Tier definitions and committed per-package error ceilings.
TIERS_FILE = REPO_ROOT / "typing_tiers.json"

#: Paths in the per-package tables are relative to this directory.
PACKAGE_ROOT = "src/maddening/"

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
    errors_by_package: dict[str, int] = field(default_factory=dict)
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


def summarise(report: dict, root: Path | None = None,
              packages: tuple[str, ...] | None = None) -> Summary:
    """Aggregate a pyright ``--outputjson`` document.

    Parameters
    ----------
    report : dict
        Parsed JSON as written by ``pyright --outputjson``.  Only the
        ``summary`` and ``generalDiagnostics`` keys are read.
    root : Path, optional
        Paths in the per-file table are made relative to this directory
        when they lie under it (defaults to the repository root).
    packages : tuple of str, optional
        Count only diagnostics in these top-level packages under
        ``src/maddening`` (a tier).  ``files_analyzed`` still describes
        the whole run, so the infrastructure checks keep seeing the
        truth; only the diagnostic counts are restricted.

    Returns
    -------
    Summary
    """
    root = (root or REPO_ROOT).resolve()
    diags = report.get("generalDiagnostics", [])
    totals: Counter[str] = Counter()
    by_rule: Counter[tuple[str, str]] = Counter()
    by_file: Counter[str] = Counter()
    by_pkg: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    files: set[str] = set()
    for d in diags:
        path = _relative(d.get("file", "<unknown>"), root)
        if packages is not None and _package_of(path) not in packages:
            continue
        sev = d.get("severity", "error")
        rule = d.get("rule") or "<no rule>"
        totals[sev] += 1
        by_rule[(sev, rule)] += 1
        files.add(path)
        if sev == "error":
            by_file[path] += 1
            pkg = _package_of(path)
            if pkg is not None:
                by_pkg[pkg] += 1
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
        errors_by_package=dict(by_pkg),
        source_files_with_diagnostics=len(files),
        has_summary_block=has_summary,
        missing_imports=dict(missing),
    )


def _package_of(relative_path: str) -> str | None:
    """The top-level package a ``src/maddening`` path belongs to.

    A module directly under the package (``sysid.py``) is its own entry,
    which is how ``typing_tiers.json`` names it.
    """
    if not relative_path.startswith(PACKAGE_ROOT):
        return None
    rest = relative_path[len(PACKAGE_ROOT):]
    return rest.split("/", 1)[0] if "/" in rest else rest


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


@dataclass
class Tier:
    """One tier's packages and their committed error ceilings."""

    name: str
    description: str
    max_errors: dict[str, int]
    pyright_version: str
    environment: str

    @property
    def packages(self) -> tuple[str, ...]:
        return tuple(sorted(self.max_errors))

    @property
    def total(self) -> int:
        return sum(self.max_errors.values())


def load_tier(name: str, path: Path = TIERS_FILE) -> Tier:
    """Read one tier out of ``typing_tiers.json``.

    Raises
    ------
    InfrastructureFailure
        The file is missing, unparsable, or does not define *name*.  A
        gate that cannot find its own ceilings must fail, not pass.
    """
    try:
        doc = json.loads(path.read_text())
    except OSError as e:
        raise InfrastructureFailure(f"cannot read {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise InfrastructureFailure(f"{path} is not valid JSON: {e}") from e
    tiers = doc.get("tiers")
    if not isinstance(tiers, dict) or name not in tiers:
        known = sorted(tiers) if isinstance(tiers, dict) else []
        raise InfrastructureFailure(
            f"{path} defines no tier {name!r} (has: {known or 'nothing'})")
    spec = tiers[name]
    ceilings = spec.get("max_errors")
    if not isinstance(ceilings, dict) or not ceilings:
        raise InfrastructureFailure(
            f"{path}: tier {name!r} has no `max_errors` mapping; an empty "
            "gate would pass on anything")
    for pkg, limit in ceilings.items():
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise InfrastructureFailure(
                f"{path}: tier {name!r} ceiling for {pkg!r} is {limit!r}, "
                "which is not a count")
    return Tier(
        name=name,
        description=str(spec.get("description", "")),
        max_errors={str(k): int(v) for k, v in ceilings.items()},
        pyright_version=str(doc.get("pyright_version", "")),
        environment=str(doc.get("environment", "")),
    )


def check_tier_environment(tier: Tier, report_version: str) -> None:
    """Refuse to compare a count against a ceiling from another pyright.

    Every pyright release changes diagnostics, so a version bump moves
    the numbers on its own.  The tier file records which release its
    ceilings were measured with; a mismatch is an infrastructure
    failure, not a gate failure -- the comparison is meaningless, and
    reporting it as a regression (or, worse, as slack) would be wrong.

    Raises
    ------
    InfrastructureFailure
    """
    want = tier.pyright_version
    if not want:
        raise InfrastructureFailure(
            f"{TIERS_FILE} records no `pyright_version`, so its ceilings "
            "cannot be tied to a measurement")
    if report_version and report_version != want:
        raise InfrastructureFailure(
            f"pyright {report_version} is running but {TIERS_FILE} records "
            f"ceilings measured with {want}; every release changes "
            "diagnostics.  Bump the `pyright==` pin in the [ci] extra, "
            "re-measure, and update typing_tiers.json and "
            "docs/developer_guide/typing.md together.")


def gate(summary: Summary, tier: Tier) -> tuple[bool, list[str]]:
    """Compare a run against a tier's ceilings.

    Returns
    -------
    (ok, lines)
        ``ok`` is false when any package exceeds its ceiling.  ``lines``
        is a human-readable verdict per package, including the slack on
        packages that are under, so a ratchet can be tightened.
    """
    ok = True
    lines = []
    for pkg in tier.packages:
        limit = tier.max_errors[pkg]
        got = summary.errors_by_package.get(pkg, 0)
        if got > limit:
            ok = False
            lines.append(f"FAIL  {pkg}: {got} errors, ceiling {limit} "
                         f"(+{got - limit})")
        elif got < limit:
            lines.append(f"ok    {pkg}: {got} errors, ceiling {limit} "
                         f"-- lower the ceiling to {got}")
        else:
            lines.append(f"ok    {pkg}: {got} errors, at the ceiling")
    unexpected = sorted(set(summary.errors_by_package) - set(tier.max_errors))
    return ok, lines + ([f"(other packages, not in this tier: "
                         f"{', '.join(unexpected)})"] if unexpected else [])


def render(summary: Summary, top: int = 15, markdown: bool = False,
           tier: "Tier | None" = None) -> str:
    """Render a summary as plain text or Markdown tables."""
    lines: list[str] = []
    h = (lambda s: f"### {s}") if markdown else (lambda s: s.upper())
    code = (lambda s: f"`{s}`") if markdown else (lambda s: s)
    lines.append(h(f"pyright summary ({tier.name})" if tier
                   else "pyright summary"))
    lines.append("")
    lines.append(
        f"files analysed: {summary.files_analyzed}; "
        f"files with diagnostics: {summary.source_files_with_diagnostics}; "
        + "; ".join(f"{s}s: {summary.totals.get(s, 0)}" for s in SEVERITIES)
    )
    lines.append("")

    if tier is not None:
        ok, verdict = gate(summary, tier)
        lines.append(h(f"{tier.name}: errors against the committed ceiling"))
        lines.append("")
        lines.extend(_table(
            ["package", "errors", "ceiling", "verdict"],
            [[code(pkg), str(summary.errors_by_package.get(pkg, 0)),
              str(tier.max_errors[pkg]),
              ("OVER" if summary.errors_by_package.get(pkg, 0)
               > tier.max_errors[pkg] else "ok")]
             for pkg in tier.packages],
            markdown))
        lines.append("")
        lines.append(f"total: {summary.error_count} errors, ceiling "
                     f"{tier.total}" + ("" if ok else "  -- GATE FAILED"))
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
    ap.add_argument("--tier",
                    help="restrict the counts to this tier's packages "
                         f"(from {TIERS_FILE.name}) and fail when any of "
                         "them exceeds its committed ceiling")
    ap.add_argument("--tiers-file", type=Path, default=TIERS_FILE,
                    help="where the tier definitions live "
                         "(default: %(default)s)")
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
        tier = load_tier(args.tier, args.tiers_file) if args.tier else None
        if args.json is not None:
            report = json.loads(args.json.read_text())
        else:
            python = _pythonpath_from(args.pyright_args)
            if python is not None:
                check_interpreter(python)
            report = run_pyright(args.pyright, args.pyright_args)
        # The infrastructure checks read the *whole* run: a tier's own
        # files can all be clean while the environment is broken.
        whole = summarise(report)
        check_run(whole, min_files=args.min_files,
                  max_missing_imports=args.max_missing_imports)
        if tier is not None:
            check_tier_environment(tier, str(report.get("version", "")))
            summary = summarise(report, packages=tier.packages)
        else:
            summary = whole
    except InfrastructureFailure as e:
        sys.stderr.write(f"typing_baseline: infrastructure failure: {e}\n")
        return EXIT_INFRASTRUCTURE
    sys.stdout.write(render(summary, top=args.top, markdown=args.markdown,
                            tier=tier))
    if tier is not None:
        ok, verdict = gate(summary, tier)
        for line in verdict:
            sys.stderr.write(line + "\n")
        if not ok:
            sys.stderr.write(
                f"typing_baseline: tier {tier.name} is over its committed "
                f"ceiling (see {args.tiers_file} and "
                "docs/developer_guide/typing.md)\n")
            return EXIT_ERRORS
        return EXIT_OK
    if args.fail_on_errors and summary.error_count:
        return EXIT_ERRORS
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
