#!/usr/bin/env python3
"""Print the outline of a Python file: every top-level def and class and every
method, with its line range and length, so you can read the right hundred
lines instead of the whole file.

    outline.py FILE [--min N] [--grep TEXT]

FILE is relative to --root (the repository holding this script by default),
or absolute.  --min: only definitions of at least N lines.  --grep: only names
containing TEXT.  Standard library only.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def outline(source: str, minimum: int = 0, grep: str | None = None) -> list[str]:
    tree = ast.parse(source)
    out = []

    def show(node, indent=""):
        n = node.end_lineno - node.lineno + 1
        kind = "class" if isinstance(node, ast.ClassDef) else "def"
        if n >= minimum and (not grep or grep.lower() in node.name.lower()):
            out.append(f"{node.lineno:>6}-{node.end_lineno:<6} {n:>5}  {indent}{kind} {node.name}")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            show(node)
            if isinstance(node, ast.ClassDef):
                for m in node.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        show(m, "    ")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--root", type=Path, default=REPO_ROOT,
                    help="what FILE is relative to (default: the repository holding this script)")
    ap.add_argument("--min", type=int, default=0, help="only definitions of at least N lines")
    ap.add_argument("--grep", help="only names containing this text")
    a = ap.parse_args(argv)
    path = Path(a.file)
    if not path.is_absolute():
        path = a.root / path
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 2
    print(f"{a.file}: {len(source.splitlines())} lines")
    for row in outline(source, a.min, a.grep):
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
