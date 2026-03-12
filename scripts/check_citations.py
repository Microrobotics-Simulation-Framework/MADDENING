#!/usr/bin/env python3
"""CI bridge: verify bibliography citations in algorithm guide documents.

Parses docs/bibliography.bib for valid BibTeX keys and scans algorithm
guide Markdown files for Pandoc-style [@Key] citations.  Verifies that
every cited key exists in the bibliography.

Also reports unused bibliography entries as warnings (non-blocking).

Usage:
    python scripts/check_citations.py [docs/algorithm_guide/]

Exits 0 if all citations resolve, nonzero if any are dangling.
"""

import os
import re
import sys


def parse_bib_keys(bib_path: str) -> set[str]:
    """Extract all BibTeX entry keys from a .bib file.

    Matches lines like: @article{Crank1975,
    """
    with open(bib_path) as f:
        content = f.read()

    # Match @type{key, patterns
    return set(re.findall(r"@\w+\{(\w+)\s*,", content))


def extract_citations(md_path: str) -> list[tuple[int, str]]:
    """Extract all [@Key] citations from a Markdown file.

    Returns (line_number, key) pairs.  Handles both single citations
    [@Key] and multiple citations [@Key1; @Key2].
    """
    citations = []
    with open(md_path) as f:
        for lineno, line in enumerate(f, 1):
            # Match [@Key] or [@Key1; @Key2; ...]
            for bracket_match in re.finditer(r"\[([^\]]*@[^\]]+)\]", line):
                bracket_content = bracket_match.group(1)
                for key_match in re.finditer(r"@(\w+)", bracket_content):
                    citations.append((lineno, key_match.group(1)))
    return citations


def scan_directory(guide_dir: str) -> list[tuple[str, int, str]]:
    """Recursively find all [@Key] citations in .md files under guide_dir.

    Returns (filepath, line_number, key) triples.
    """
    results = []
    for root, _dirs, files in os.walk(guide_dir):
        for fname in sorted(files):
            if not fname.endswith(".md") or fname.startswith("_"):
                continue
            fpath = os.path.join(root, fname)
            for lineno, key in extract_citations(fpath):
                results.append((fpath, lineno, key))
    return results


def main():
    guide_dir = sys.argv[1] if len(sys.argv) > 1 else "docs/algorithm_guide/"
    bib_path = os.environ.get("BIB_PATH", "docs/bibliography.bib")

    if not os.path.isfile(bib_path):
        print(f"Bibliography not found: {bib_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(guide_dir):
        print(f"Algorithm guide directory not found: {guide_dir}", file=sys.stderr)
        sys.exit(1)

    bib_keys = parse_bib_keys(bib_path)
    if not bib_keys:
        print(f"WARNING: No BibTeX entries found in {bib_path}", file=sys.stderr)

    citations = scan_directory(guide_dir)

    # Check for dangling citations (cited but not in bib)
    errors = []
    cited_keys = set()
    for fpath, lineno, key in citations:
        cited_keys.add(key)
        if key not in bib_keys:
            relpath = os.path.relpath(fpath)
            errors.append(f"{relpath}:{lineno}: [@{key}] not found in {bib_path}")

    # Report unused bib entries (warning, non-blocking)
    unused = bib_keys - cited_keys
    if unused:
        for key in sorted(unused):
            print(f"WARNING: {bib_path} entry '{key}' is not cited by any algorithm guide")

    # Report results
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(
            f"\n{len(errors)} dangling citation(s) found "
            f"({len(cited_keys)} unique keys cited, {len(bib_keys)} bib entries)",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"OK: {len(citations)} citation(s) verified "
        f"({len(cited_keys)} unique keys, {len(bib_keys)} bib entries)"
    )


if __name__ == "__main__":
    main()
