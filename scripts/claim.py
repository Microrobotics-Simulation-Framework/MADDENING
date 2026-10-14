#!/usr/bin/env python3
"""Look up claims-inventory rows without opening the inventories (thousands of lines each).

    claim.py show CPL-088 [SYS-071 ...]     the full rows
    claim.py list INVENTORY [--grep TEXT]   id, status and the first 110 characters
                                            of each claim; INVENTORY is the part of
                                            docs/validation/<name>_claims.yaml before
                                            _claims (claim.py inventories lists them)
    claim.py next SYS                       the next free number for a prefix
    claim.py inventories                    the inventories this tree has

Standard library only.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_ROW = re.compile(r"\s*-\s+id:\s*\"?([A-Z]+-\d+)")


def inventories(root: Path) -> dict[str, Path]:
    return {p.name[:-len("_claims.yaml")]: p
            for p in sorted((root / "docs" / "validation").glob("*_claims.yaml"))}


def rows(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [i for i, x in enumerate(lines) if _ROW.match(x)]
    for p, q in zip(starts, starts[1:] + [len(lines)]):
        yield _ROW.match(lines[p]).group(1), p + 1, lines[p:q]


def one(block, name):
    for i, x in enumerate(block):
        m = re.match(rf"\s+{name}:\s*(.*)", x)
        if m:
            v = m.group(1).strip()
            if v in ("", ">", "|", ">-", "|-"):
                v = " ".join(y.strip() for y in block[i + 1:i + 4])
            return v.strip("\"'")
    return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT,
                    help="the repository to read (default: the one holding this script)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show", help="the full rows")
    s.add_argument("ids", nargs="+")
    ls = sub.add_parser("list", help="id, status and claim of each row of one inventory")
    ls.add_argument("inventory")
    ls.add_argument("--grep", help="text that must appear in the row")
    n = sub.add_parser("next", help="the next free number for a prefix")
    n.add_argument("prefix")
    sub.add_parser("inventories", help="the inventories this tree has")
    a = ap.parse_args(argv)
    files = inventories(a.root.resolve())
    if a.cmd == "inventories":
        for name, path in files.items():
            print(f"{name:<14} {path.name}")
        return 0
    if a.cmd == "show":
        status = 0
        for want in a.ids:
            found = False
            for path in files.values():
                for rid, line, block in rows(path):
                    if rid == want:
                        print(f"# {path.name} line {line}")
                        print("\n".join(block).rstrip())
                        print()
                        found = True
            if not found:
                print(f"{want}: not found")
                status = 1
        return status
    if a.cmd == "list":
        match = [p for name, p in files.items() if name.startswith(a.inventory)]
        if not match:
            print(f"no inventory named {a.inventory}; this tree has {sorted(files)}",
                  file=sys.stderr)
            return 2
        for rid, _, block in rows(match[0]):
            if a.grep and a.grep.lower() not in "\n".join(block).lower():
                continue
            print(f"{rid:<9} {one(block, 'status'):<10} {one(block, 'claim')[:110]}")
        return 0
    nums = [int(rid.split("-")[1]) for p in files.values() for rid, _, _ in rows(p)
            if rid.startswith(a.prefix + "-")]
    top = max(nums, default=0)
    print(f"{a.prefix}: {len(nums)} rows, highest {top}, next free {a.prefix}-{top + 1:03d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
