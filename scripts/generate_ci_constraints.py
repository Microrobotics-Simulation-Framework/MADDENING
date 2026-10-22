#!/usr/bin/env python3
"""Write the constraints files CI installs against, from a fresh resolution.

``ci.yml`` installs the package with ``pip install -e ".[ci]"`` and friends,
which resolves every dependency to the newest release the declared ranges
allow, on the day.  A release of any of them can turn a green tree red
(lineax 0.1.1 did, in 0.4.0) and a red run cannot be reproduced a week
later.  So every install in ``ci.yml`` reads ``PIP_CONSTRAINT``, one
committed file::

    ci/constraints.txt

holding ``name==version`` for every distribution a resolution of the extras
CI installs (``ci``, ``usd``) pulls in.  JAX and jaxlib are not
in it: each lane pins its own, and the file is resolved against every JAX
version of the test matrix (``JAX_VERSIONS``) and refused unless they agree
on every other distribution (0.10.2 and 0.11.2 do).
Constraints restrict versions and install nothing, so a job that installs
fewer extras is held to the same versions for the part it installs.

The weekly ``scheduled-checks.yml`` installs the same extras *without* the
file, at the newest versions the ranges allow, and runs the tests: that is
what keeps catching what the newest dependencies break.  When it goes red
for a release, the answer is a ``pyproject.toml`` bound or a fix; when a
release is wanted in per-push CI, regenerate the files::

    python scripts/generate_ci_constraints.py

A file starts with the sha256 of the requirement strings it was resolved
from (the base dependencies and the three extras, as ``pyproject.toml``
spells them): ``tests/compliance/test_ci_constraints.py`` recomputes it, so
an edit to a declared range that is not followed by a regeneration fails
there, and so does a pinned version outside a range.  It cannot say that a
transitive dependency has a newer release: that is the scheduled job's.

The resolution is ``pip install --dry-run --ignore-installed --report``:
nothing is installed, and it runs wherever there is network, on the CI
platform (CPython 3.12, Linux x86-64) for the pins to be the ones CI gets.
Distributions that exist only for another platform are not in the report,
and so not in the file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONSTRAINTS_DIR = REPO_ROOT / "ci"

#: The extras CI installs anywhere (``ci.yml``): the test lanes and the
#: type check install ``ci``, the compliance job and ``test-usd`` add
#: ``usd``.
EXTRAS = ("ci", "usd")
#: The JAX versions of the test matrix, which the one file must hold for.
#: ``tests/compliance/test_ci_constraints.py`` holds this against the matrix.
JAX_VERSIONS = ("0.10.2", "0.11.2")
#: Distributions the lanes pin themselves, left out of the file.
LANE_PINNED = ("jax", "jaxlib")

DIGEST_PREFIX = "# requirements-digest: "
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


CONSTRAINTS_FILE = CONSTRAINTS_DIR / "constraints.txt"


def canonical(name: str) -> str:
    """PEP 503 name: runs of ``-_.`` become one ``-``, lower case."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_strings(pyproject: dict) -> list[str]:
    """The base dependencies and the extras of ``EXTRAS`` as written, sorted, with their group."""
    project = pyproject["project"]
    out = [f"dependencies: {r}" for r in project["dependencies"]]
    for extra in EXTRAS:
        out += [f"{extra}: {r}" for r in project["optional-dependencies"][extra]]
    return sorted(out)


def requirements_digest(pyproject: dict) -> str:
    """sha256 of the requirement strings and of the JAX versions the file was resolved for."""
    text = "\n".join([*requirement_strings(pyproject), "jax-versions: " + ",".join(JAX_VERSIONS)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pins_from_report(report: dict) -> dict[str, str]:
    """``{canonical name: version}`` for everything a ``pip --report`` installs but the project itself."""
    pins: dict[str, str] = {}
    for item in report["install"]:
        meta = item["metadata"]
        name = canonical(meta["name"])
        if name == "maddening":
            continue
        if name in pins and pins[name] != meta["version"]:
            raise SystemExit(f"{name} is resolved to both {pins[name]} and {meta['version']}")
        pins[name] = meta["version"]
    return pins


def render(pins: dict[str, str], digest: str, resolved_on: str) -> str:
    for name, version in pins.items():
        if not _NAME.fullmatch(name) or not re.fullmatch(r"[0-9A-Za-z.!+_-]+", version):
            raise SystemExit(f"cannot write {name}=={version} as a pin")
    header = (
        "# The versions CI installs against.  Written by\n"
        "# scripts/generate_ci_constraints.py from fresh resolutions of the base\n"
        f"# dependencies with the extras {', '.join(EXTRAS)} on JAX {' and '.join(JAX_VERSIONS)};\n"
        "# do not edit by hand.  jax and jaxlib are pinned by each lane, not here.\n"
        "# ci.yml reads this file through PIP_CONSTRAINT; the weekly\n"
        "# scheduled-checks.yml installs the same extras without it.  Resolved on\n"
        f"# CPython 3.12, Linux x86-64, {resolved_on}.\n"
        f"{DIGEST_PREFIX}{digest}\n"
    )
    return header + "".join(f"{name}=={pins[name]}\n" for name in sorted(pins))


def agree(resolutions: dict[str, dict[str, str]]) -> dict[str, str]:
    """One set of pins from the per-JAX-version resolutions, JAX itself left out; a disagreement is an error."""
    kept = {v: {n: ver for n, ver in pins.items() if n not in LANE_PINNED} for v, pins in resolutions.items()}
    first_version, first = next(iter(kept.items()))
    for version, pins in kept.items():
        differing = sorted(n for n in first.keys() | pins.keys() if first.get(n) != pins.get(n))
        if differing:
            raise SystemExit(
                f"JAX {first_version} and JAX {version} resolve to different versions of "
                + ", ".join(f"{n} ({first.get(n)} / {pins.get(n)})" for n in differing)
                + ": one constraints file cannot hold both; split it per JAX version")
    return first


def resolve(jax_version: str, python: str = sys.executable) -> dict[str, str]:
    """Resolve ``jax``, ``jaxlib`` and the project's extras; nothing is installed."""
    extras = ",".join(EXTRAS)
    with tempfile.TemporaryDirectory(prefix="ci-constraints-") as tmp:
        report = Path(tmp) / "report.json"
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("PIP_") and k not in ("PYTHONPATH", "VIRTUAL_ENV")}
        cmd = [python, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--quiet",
               "--report", str(report), f"jax=={jax_version}", f"jaxlib=={jax_version}",
               f"{REPO_ROOT}[{extras}]"]
        proc = subprocess.run(cmd, env=env, cwd=tmp, capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit(f"pip could not resolve JAX {jax_version}:\n{proc.stdout}{proc.stderr}")
        return pins_from_report(json.loads(report.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="write nothing; exit 1 if the file's digest is not pyproject.toml's")
    args = ap.parse_args(argv)
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    digest = requirements_digest(pyproject)
    if args.check:
        have = None
        if CONSTRAINTS_FILE.is_file():
            have = next((ln[len(DIGEST_PREFIX):].strip() for ln in CONSTRAINTS_FILE.read_text(encoding="utf-8").splitlines()
                         if ln.startswith(DIGEST_PREFIX)), None)
        if have != digest:
            print(f"{CONSTRAINTS_FILE.name} was resolved from other requirements than pyproject.toml's; "
                  "run scripts/generate_ci_constraints.py")
            return 1
        return 0
    pins = agree({v: resolve(v) for v in JAX_VERSIONS})
    CONSTRAINTS_DIR.mkdir(exist_ok=True)
    CONSTRAINTS_FILE.write_text(render(pins, digest, date.today().isoformat()), encoding="utf-8")
    print(f"wrote {CONSTRAINTS_FILE.relative_to(REPO_ROOT)}: {len(pins)} pins")
    return 0


if __name__ == "__main__":
    sys.exit(main())
