#!/usr/bin/env python3
"""CI bridge: verify Implementation Mapping tables in algorithm guides.

Parses each algorithm guide's Implementation Mapping Markdown table,
extracts every backticked ``maddening.*`` qualified name from its rows, and
verifies that each one resolves to an existing callable that the named
class actually defines.

Three things this gate has to get right, because each was a hole:

* **Own ``__dict__``, not the MRO.**  ``getattr`` walks base classes, so a
  row naming ``HeatNode.update`` kept resolving after the concrete method
  was renamed -- ``SimulationNode.update`` answered instead.  A row may opt
  into inherited behaviour by saying "inherited" in the row; the gate then
  allows it and says which base it came from.
* **Every symbol in a row**, not just the first, and every row must carry a
  code reference at all -- dropping the backticks used to drop the row.
* **A pinned minimum per guide**, so a table that vanishes fails instead of
  quietly lowering the count.

Usage:
    python scripts/check_impl_mapping.py [docs/algorithm_guide/]

Exits 0 if all mappings resolve, nonzero if any are stale.
"""

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

# Add src to path
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from maddening.compliance._validate import resolve_dotted_name  # noqa: E402

DEFAULT_GUIDE_DIR = os.path.join("docs", "algorithm_guide")

# Minimum number of resolvable ``maddening.*`` references per guide, keyed by
# path relative to the repository root.  Pinned so that a deleted table, or a
# row that loses its backticks, fails instead of reporting a smaller "OK".
# Raise a number when a guide gains rows; never lower one to make CI pass.
MIN_MAPPINGS = {
    os.path.join("docs", "algorithm_guide", "nodes", "heat_node.md"): 5,
    os.path.join("docs", "algorithm_guide", "nodes", "adaptive_node.md"): 12,
    os.path.join(
        "docs", "algorithm_guide", "solvers", "explicit_integrators.md"
    ): 5,
    os.path.join("docs", "algorithm_guide", "nodes", "spring_node.md"): 9,
    os.path.join("docs", "algorithm_guide", "nodes", "ball_node.md"): 7,
    os.path.join("docs", "algorithm_guide", "nodes", "rigid_body_2d_node.md"): 7,
    os.path.join("docs", "algorithm_guide", "nodes", "heart_pump_node.md"): 9,
}

_QNAME = re.compile(r"`(maddening\.[^`]+)`")
_CODE_SPAN = re.compile(r"`[^`]+`")
# A Markdown table cell may contain an escaped pipe.  Splitting on a bare
# ``|`` mangled every row holding LaTeX like ``\|g\|``, which shifted the
# Implementation column out of cell 1 and made the row invisible to the gate.
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def _split_row(line: str) -> list[str]:
    """Split a Markdown table row into cells, honouring ``\\|`` escapes."""
    return [c.strip() for c in _UNESCAPED_PIPE.split(line)[1:-1]]


def extract_rows(md_path: str) -> list[list[str]]:
    """Return the data rows of the Implementation Mapping table, as cell lists."""
    with open(md_path) as f:
        content = f.read()

    section_match = re.search(
        r"## Implementation Mapping\s*\n(.*?)(?=\n## |\Z)",
        content,
        re.DOTALL,
    )
    if not section_match:
        return []

    section = section_match.group(1)

    rows = []
    for line in section.strip().split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = _split_row(line)
        if len(cells) < 2:
            continue
        # Separator rows (|---|---|---|)
        if all(set(c) <= {"-", " ", ":"} for c in cells):
            continue
        # Header row
        if (cells[0].lower().startswith("equation")
                or cells[1].lower().startswith("implementation")):
            continue
        rows.append(cells)

    return rows


def extract_qualified_names(md_path: str) -> list[tuple[str, str]]:
    """Extract ``(equation_term, qualified_name)`` pairs from a guide.

    Every backticked ``maddening.*`` span in the row is returned, not only
    the first: a row may trace one equation term to two functions.
    """
    results = []
    for cells in extract_rows(md_path):
        term = cells[0]
        for match in _QNAME.finditer(" | ".join(cells)):
            results.append((term, match.group(1).rstrip("`).,( ")))
    return results


def check_guide(
    md_path: str, relpath: str
) -> tuple[int, list[str], list[str], list[str]]:
    """Check one guide.  Returns ``(n_checked, errors, notes, skipped)``.

    ``n_checked`` counts only references this environment could actually
    resolve; one that needs an uninstalled optional subpackage lands in
    ``skipped`` and is not counted as verified.
    """
    errors: list[str] = []
    notes: list[str] = []
    skipped: list[str] = []
    checked = 0

    for cells in extract_rows(md_path):
        term = cells[0]
        impl = cells[1]
        # A row whose Implementation cell carries no code span at all traces
        # its equation term to nothing.  Dropping the backticks used to make
        # the row invisible to this gate.
        if not _CODE_SPAN.search(impl):
            errors.append(
                f"{relpath}: row '{term}' has no code reference in its "
                f"Implementation column"
            )
            continue

        row_text = " | ".join(cells)
        allow_inherited = "inherited" in row_text.lower()
        for match in _QNAME.finditer(row_text):
            qname = match.group(1).rstrip("`).,( ")
            checked += 1
            res = resolve_dotted_name(
                qname,
                require_own=not allow_inherited,
                require_callable=True,
            )
            if res.unavailable:
                # An optional subpackage this environment cannot import.
                # Not checked is not the same as not there; saying "stale"
                # here would make a guide for a USD or viz node impossible
                # to keep green in a CI that installs only [ci].
                checked -= 1
                skipped.append(
                    f"{relpath}: '{qname}' (for term '{term}') was NOT "
                    f"checked -- {res.reason}"
                )
            elif not res.ok:
                errors.append(
                    f"{relpath}: '{qname}' (for term '{term}') does not "
                    f"resolve: {res.reason}"
                )
            elif res.inherited_from:
                notes.append(
                    f"{relpath}: '{qname}' (for term '{term}') is inherited "
                    f"from {res.inherited_from}, as the row states"
                )

    return checked, errors, notes, skipped


def check_pinned(
    per_file: dict[str, int],
    min_mappings: dict[str, int],
    repo_root: str,
) -> list[str]:
    """Check the pinned per-guide minimums against the repository.

    Runs regardless of the directory that was scanned, so that a guide
    deleted outright still fails rather than dropping out of the count.
    """
    errors: list[str] = []
    for pinned, minimum in sorted(min_mappings.items()):
        abspath = os.path.join(repo_root, pinned)
        if not os.path.isfile(abspath):
            errors.append(
                f"{pinned}: pinned in MIN_MAPPINGS but the file does not exist"
            )
            continue
        found = per_file.get(pinned)
        if found is None:
            found, errs, _notes, _skipped = check_guide(abspath, pinned)
            errors.extend(errs)
        if found < minimum:
            errors.append(
                f"{pinned}: {found} implementation mapping(s) found, at least "
                f"{minimum} expected -- a row or the whole table has gone "
                f"missing (update MIN_MAPPINGS only if the guide legitimately "
                f"shrank)"
            )
    return errors


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    guide_dir = argv[0] if argv else os.path.join(_REPO_ROOT, DEFAULT_GUIDE_DIR)

    if not os.path.isdir(guide_dir):
        print(f"Directory not found: {guide_dir}", file=sys.stderr)
        return 1

    errors: list[str] = []
    notes: list[str] = []
    skipped: list[str] = []
    checked = 0
    per_file: dict[str, int] = {}

    # Recursive: the scope used to be a non-recursive listdir of
    # docs/algorithm_guide/nodes/, so a guide in any other subdirectory was
    # invisible to the gate.
    for root, _dirs, files in os.walk(guide_dir):
        for fname in sorted(files):
            if not fname.endswith(".md") or fname.startswith("_"):
                continue
            fpath = os.path.join(root, fname)
            relpath = os.path.relpath(fpath, _REPO_ROOT)
            n, errs, ns, sk = check_guide(fpath, relpath)
            checked += n
            per_file[relpath] = n
            errors.extend(errs)
            notes.extend(ns)
            skipped.extend(sk)

    errors.extend(check_pinned(per_file, MIN_MAPPINGS, _REPO_ROOT))

    for n in notes:
        print(f"NOTE: {n}")
    for sk in skipped:
        print(f"NOTE: {sk}")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(
            f"\n{len(errors)} stale mapping(s) found out of {checked} checked",
            file=sys.stderr,
        )
        return 1

    suffix = f", {len(skipped)} not checked" if skipped else ""
    print(
        f"OK: {checked} implementation mapping(s) verified{suffix} "
        f"across {guide_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
