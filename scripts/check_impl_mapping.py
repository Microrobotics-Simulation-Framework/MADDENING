#!/usr/bin/env python3
"""CI bridge: verify Implementation Mapping tables in algorithm guides.

Parses each algorithm guide's Implementation Mapping Markdown table,
extracts function qualified names from the "Implementation" column,
and verifies via importlib + getattr() that each name resolves to an
existing callable.

Usage:
    python scripts/check_impl_mapping.py [docs/algorithm_guide/nodes/]

Exits 0 if all mappings resolve, nonzero if any are stale.
"""

import importlib
import os
import re
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def extract_qualified_names(md_path: str) -> list[tuple[str, str]]:
    """Extract (equation_term, qualified_name) pairs from a Markdown table.

    Looks for a section called "Implementation Mapping" and parses the
    table rows. Extracts qualified names matching the pattern
    ``module.path.ClassName.method_name`` from the Implementation column.
    """
    with open(md_path) as f:
        content = f.read()

    # Find the Implementation Mapping section
    section_match = re.search(
        r"## Implementation Mapping\s*\n(.*?)(?=\n## |\Z)",
        content,
        re.DOTALL,
    )
    if not section_match:
        return []

    section = section_match.group(1)

    # Parse table rows (skip header and separator)
    rows = []
    for line in section.strip().split("\n"):
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) >= 2:
            rows.append(cells)

    # Skip header row and separator
    data_rows = []
    for row in rows:
        # Skip separator rows (|---|---|---|)
        if all(set(c) <= {"-", " ", ":"} for c in row):
            continue
        # Skip header row
        if row[0].lower().startswith("equation") or row[1].lower().startswith("implementation"):
            continue
        data_rows.append(row)

    # Extract qualified names from the Implementation column
    # Pattern: module.path.ClassName.method or module.path.ClassName
    qname_pattern = re.compile(r"`(maddening\.[^`]+)`")

    results = []
    for row in data_rows:
        term = row[0]
        impl = row[1]
        match = qname_pattern.search(impl)
        if match:
            qname = match.group(1).rstrip("`).,(")
            results.append((term, qname))

    return results


def resolve_qualified_name(qname: str) -> bool:
    """Check if a qualified name resolves to an existing callable."""
    parts = qname.split(".")

    # Try progressively longer module paths
    for i in range(len(parts) - 1, 0, -1):
        module_path = ".".join(parts[:i])
        attr_path = parts[i:]
        try:
            mod = importlib.import_module(module_path)
            obj = mod
            for attr in attr_path:
                obj = getattr(obj, attr)
            return True
        except (ImportError, AttributeError):
            continue

    return False


def main():
    guide_dir = sys.argv[1] if len(sys.argv) > 1 else "docs/algorithm_guide/nodes/"

    if not os.path.isdir(guide_dir):
        print(f"Directory not found: {guide_dir}")
        sys.exit(1)

    errors = []
    checked = 0

    for fname in sorted(os.listdir(guide_dir)):
        if not fname.endswith(".md") or fname.startswith("_"):
            continue

        fpath = os.path.join(guide_dir, fname)
        mappings = extract_qualified_names(fpath)

        for term, qname in mappings:
            checked += 1
            if not resolve_qualified_name(qname):
                errors.append(f"{fname}: '{qname}' (for term '{term}') does not resolve")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(f"\n{len(errors)} stale mapping(s) found out of {checked} checked", file=sys.stderr)
        sys.exit(1)

    print(f"OK: {checked} implementation mapping(s) verified across {guide_dir}")


if __name__ == "__main__":
    main()
