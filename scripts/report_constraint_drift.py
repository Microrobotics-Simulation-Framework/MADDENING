#!/usr/bin/env python3
"""Say which installed versions differ from a constraints file.

The weekly ``scheduled-checks.yml`` installs the CI extras at the newest
versions their ranges allow and runs the tests.  When that goes red, the
first question is what is newer than what per-push CI installs, so this
prints it as a Markdown table (to ``--markdown``, default
``$GITHUB_STEP_SUMMARY``, else stdout)::

    python scripts/report_constraint_drift.py ci/constraints.txt

``pip list`` is read from the running interpreter.  A distribution that is
installed and not in the file (jax and jaxlib, which each lane pins itself,
aside) is listed as *new*; one in the file and not installed is not drift (the
job installs fewer extras than the file covers) and is not printed.  The
project itself is ignored.  Exits 0 whatever it
finds: it reports, the tests decide.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_ci_constraints import LANE_PINNED, canonical  # noqa: E402

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)$")


def read_constraints(text: str) -> dict[str, str]:
    """``{canonical name: version}`` from a constraints file; comments and blanks are skipped."""
    pins = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PIN.match(line)
        if not m:
            raise SystemExit(f"not a name==version pin: {line!r}")
        pins[canonical(m.group(1))] = m.group(2)
    return pins


def installed() -> dict[str, str]:
    out = subprocess.run([sys.executable, "-m", "pip", "list", "--format=json"],
                         capture_output=True, text=True, check=True).stdout
    return {canonical(d["name"]): d["version"] for d in json.loads(out)}


def drift(pinned: dict[str, str], have: dict[str, str]) -> dict[str, list]:
    """``{"changed": [(name, pinned, have)], "new": [(name, have)], "absent": [(name, pinned)]}``."""
    # The project, and what each lane pins itself (jax, jaxlib: not in the file).
    have = {k: v for k, v in have.items() if k != "maddening" and k not in LANE_PINNED}
    return {
        "changed": [(n, pinned[n], have[n]) for n in sorted(pinned) if n in have and have[n] != pinned[n]],
        "new": [(n, have[n]) for n in sorted(have) if n not in pinned],
        "absent": [(n, pinned[n]) for n in sorted(pinned) if n not in have],
    }


def markdown(d: dict[str, list], source: str) -> str:
    lines = [f"### Installed versions against `{source}`", ""]
    if not (d["changed"] or d["new"]):
        return "\n".join(lines + ["Nothing differs: the newest versions are the pinned ones.", ""])
    if d["changed"]:
        lines += ["| distribution | per-push CI | installed here |", "|---|---|---|"]
        lines += [f"| {n} | {a} | {b} |" for n, a, b in d["changed"]]
        lines.append("")
    if d["new"]:
        lines += ["Installed here, not in the file: " + ", ".join(f"{n} {v}" for n, v in d["new"]), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("constraints", type=Path)
    ap.add_argument("--markdown", type=Path,
                    default=Path(os.environ["GITHUB_STEP_SUMMARY"]) if os.environ.get("GITHUB_STEP_SUMMARY") else None)
    args = ap.parse_args(argv)
    text = markdown(drift(read_constraints(args.constraints.read_text(encoding="utf-8")), installed()),
                    args.constraints.as_posix())
    if args.markdown:
        with open(args.markdown, "a", encoding="utf-8") as fh:
            fh.write(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
