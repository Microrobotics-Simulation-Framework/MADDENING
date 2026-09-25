#!/usr/bin/env python3
"""Generate MADDENING's CycloneDX SBOMs from a clean install of the wheel.

One SBOM per install in ``check_sbom.SBOM_INSTALLS`` -- ``core`` (the base
dependencies, what ``pip install maddening`` gives every user) and the
extras argued for in ``docs/validation/soup_package.md`` §6 -- written to
``docs/validation/sbom/maddening-<version>-<install>.cdx.json``.

For each install this

1. builds the wheel from this tree (``uv build --wheel``);
2. creates a fresh, isolated virtual environment (``uv venv``, no seed
   packages, no system site-packages, ``PYTHONPATH`` and friends scrubbed)
   and installs the wheel into it with the install's extra;
3. runs ``cyclonedx-py environment`` against that environment.  The tool
   lives in a *separate* tool venv, so neither it nor its dependencies
   appear in the SBOM;
4. checks that the components are exactly the distributions installed in
   the environment, less MADDENING itself (which is the root component);
5. finalises the document (:func:`finalise`): the root component gets its
   purl and a dependency edge to each *declared* direct dependency
   (cyclonedx-py also links the packages an unselected extra names), the
   resolution environment is recorded, and the result is sorted and sealed;
6. validates it against the CycloneDX schema, writes it, and runs
   ``check_sbom.py`` over the output directory.

Determinism
-----------
The resolution is pinned with uv's ``--exclude-newer``: only
distributions uploaded before the cutoff are considered.  The cutoff
defaults to 00:00 UTC on the day of generation, so two runs on one day
resolve alike, and it is recorded in the SBOM
(``maddening:sbom:exclude-newer``); pass ``--exclude-newer`` to reproduce
an earlier SBOM.  ``metadata.timestamp`` is the cutoff too, unless
``SOURCE_DATE_EPOCH`` is set, in which case it is that.  ``serialNumber``
is derived from the content (``check_sbom.content_serial``), components
and the dependency graph are sorted, and keys are written sorted -- so a
regeneration that resolves the same environment writes the same bytes.
``python scripts/check_sbom.py --normalise FILE`` strips the three
date-dependent fields, for comparing SBOMs generated on different days.

The Python and platform matter as much as the date: jaxlib, numpy, scipy
and most of the server stack ship per-platform wheels, and a marker can
select a different dependency set.  Each SBOM records the PEP 508 marker
variables and the wheel platform tag it was resolved on.

Usage
-----
::

    python scripts/generate_sbom.py                       # every covered install
    python scripts/generate_sbom.py --install core        # one of them
    python scripts/generate_sbom.py --extra cuda12 --output-dir /tmp/sbom
    python scripts/generate_sbom.py --exclude-newer 2026-09-25

Needs ``uv`` and network access to the package index.  It never touches
the environment it runs in: every environment it installs into is created
under a temporary directory and deleted afterwards (``--keep-work-dir``
keeps them).  At release time this runs on the release commit, after the
version bump and before the tag; see ``docs/validation/soup_package.md``
§6.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_sbom  # noqa: E402  (after sys.path setup)

from packaging.utils import canonicalize_name  # noqa: E402

REPO_ROOT = check_sbom.REPO_ROOT

#: The cyclonedx-bom release the SBOMs are generated with.  Pinned exactly,
#: like pyright in the ``ci`` extra: the tool writes its own name and
#: version into ``metadata.tools``, so a floating tool would make two
#: generations of the same environment differ.  Bump it deliberately and
#: regenerate.
CYCLONEDX_BOM = "cyclonedx-bom==7.4.0"

#: CycloneDX specification version written.
SPEC_VERSION = "1.6"

DEFAULT_INDEX = "https://pypi.org/simple"

#: Environment variables that could put packages into, or pull
#: configuration into, an environment this script means to be clean.
_SCRUBBED_ENV = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE",
                 "PYTHONSAFEPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "PIP_CONFIG_FILE")

#: Run inside the target environment to record where it resolved.
_PROBE = r"""
import json, platform, sys, sysconfig
from importlib import metadata
env = {
    "implementation_name": sys.implementation.name,
    "os_name": __import__("os").name,
    "platform_machine": platform.machine(),
    "platform_python_implementation": platform.python_implementation(),
    "platform_system": platform.system(),
    "python_full_version": platform.python_version(),
    "python_version": ".".join(platform.python_version_tuple()[:2]),
    "sys_platform": sys.platform,
}
libc = " ".join(p for p in platform.libc_ver() if p)
dists = sorted({(d.metadata["Name"], d.version) for d in metadata.distributions()})
license_fields = {d.metadata["Name"]: d.metadata["License"]
                  for d in metadata.distributions() if d.metadata["License"]}
print(json.dumps({"marker": env, "platform": sysconfig.get_platform(),
                  "libc": libc or "unknown", "distributions": dists,
                  "license_fields": license_fields}))
"""

#: Longest legacy ``License`` field read as a licence *name*.  Longer, or
#: multi-line, it is licence text, which cyclonedx-py would embed only
#: with ``--gather-license-texts`` (every licence file of every package).
_LICENSE_NAME_MAX = 200


# --------------------------------------------------------------------
# The pure half: turning cyclonedx-py's output into the committed SBOM
# --------------------------------------------------------------------


def _timestamp(exclude_newer: _dt.datetime) -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch:
        moment = _dt.datetime.fromtimestamp(int(epoch), tz=_dt.timezone.utc)
    else:
        moment = exclude_newer
    return moment.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def finalise(raw: dict, *, pyproject: dict, install: str, probe: dict,
             provenance: dict[str, str], timestamp: str) -> dict:
    """Turn a ``cyclonedx-py environment`` document into the committed SBOM.

    Pure: no files, no network.  ``probe`` is what :data:`_PROBE` printed
    in the target environment; ``provenance`` is extra
    ``maddening:sbom:<key>`` properties (resolver, index, cutoff).  Raises
    ``ValueError`` if a declared direct dependency is not in the
    environment -- the install itself would then be wrong.
    """
    sbom = json.loads(json.dumps(raw))            # deep copy, JSON types only
    version = str(pyproject["project"]["version"])
    meta = sbom.setdefault("metadata", {})
    root = meta.setdefault("component", {})
    old_ref = root.get("bom-ref")
    new_ref = f"{check_sbom.ROOT_NAME}=={version}"
    root["bom-ref"] = new_ref
    root["purl"] = check_sbom.root_purl(version)
    root.setdefault("name", check_sbom.ROOT_NAME)
    root.setdefault("version", version)

    by_name = {canonicalize_name(c["name"]): c for c in sbom.get("components", [])}
    env = dict(probe["marker"])

    # cyclonedx-py reads the legacy ``License`` field only when it is an
    # SPDX identifier on the SPDX list, and otherwise treats it as licence
    # text, which it drops unless asked to embed every licence text.  So a
    # package whose metadata names its licence any other way got none:
    # usd-core declares ``License: LicenseRef-TOST-1.0``.  A short,
    # one-line field is a name; carry it.
    fields = {canonicalize_name(k): v for k, v in probe.get("license_fields", {}).items()}
    for key, comp in by_name.items():
        value = (fields.get(key) or "").strip()
        if (not comp.get("licenses") and value and "\n" not in value
                and len(value) <= _LICENSE_NAME_MAX):
            comp["licenses"] = [{"license": {"acknowledgement": "declared",
                                             "name": value}}]
    edges = []
    for req in check_sbom.declared_requirements(pyproject, install):
        if req.marker is not None and not req.marker.evaluate({**env, "extra": ""}):
            continue
        comp = by_name.get(canonicalize_name(req.name))
        if comp is None:
            raise ValueError(f"declared direct dependency {req} is not installed in "
                             f"the {install} environment")
        edges.append(comp["bom-ref"])

    deps = []
    root_seen = False
    for dep in sbom.get("dependencies", []):
        dep = dict(dep)
        if dep.get("ref") in (old_ref, new_ref):
            # cyclonedx-py links the root to every installed package its
            # Requires-Dist names, extras included: in a core install that
            # made equinox (installed for lineax, named by [surrogates]) a
            # "direct" dependency.  The declared set is the truth.
            dep = {"ref": new_ref, "dependsOn": sorted(set(edges))}
            root_seen = True
        elif old_ref and old_ref in dep.get("dependsOn", []):
            dep["dependsOn"] = [new_ref if r == old_ref else r for r in dep["dependsOn"]]
        deps.append(dep)
    if not root_seen:
        deps.append({"ref": new_ref, "dependsOn": sorted(set(edges))})
    sbom["dependencies"] = deps

    requirement = check_sbom.install_requirement(install)
    props = [p for p in meta.get("properties", [])
             if not str(p.get("name", "")).startswith(check_sbom.PROP)]
    props += [
        {"name": check_sbom.PROP_INSTALL, "value": install},
        {"name": check_sbom.PROP + "requirement", "value": requirement},
        {"name": check_sbom.PROP + "platform", "value": probe["platform"]},
        {"name": check_sbom.PROP + "libc", "value": probe["libc"]},
    ]
    props += [{"name": check_sbom.PROP_MARKER + k, "value": str(v)}
              for k, v in env.items()]
    props += [{"name": check_sbom.PROP + k, "value": v} for k, v in provenance.items()]
    meta["properties"] = props
    meta["timestamp"] = timestamp
    return check_sbom.seal(sbom)


def installed_mismatch(sbom: dict, distributions: list[list[str]]) -> list[str]:
    """What differs between the SBOM's components and the environment.

    ``distributions`` is ``[name, version]`` for every distribution the
    target environment's ``importlib.metadata`` sees.  MADDENING is the
    root and must not be a component; everything else must match one for
    one.  This is the check that nothing leaked in (a tool, a seed
    package, a ``PYTHONPATH`` entry) and nothing was dropped.
    """
    installed = {canonicalize_name(n): v for n, v in distributions}
    root_version = installed.pop(check_sbom.ROOT_NAME, None)
    problems = []
    if root_version != sbom["metadata"]["component"]["version"]:
        problems.append(f"the environment holds maddening {root_version}, the SBOM's "
                        f"root is {sbom['metadata']['component']['version']}")
    listed = {canonicalize_name(c["name"]): c["version"] for c in sbom["components"]}
    for name in sorted(set(installed) | set(listed)):
        if installed.get(name) != listed.get(name):
            problems.append(f"{name}: installed {installed.get(name)}, in the SBOM "
                            f"{listed.get(name)}")
    return problems


# --------------------------------------------------------------------
# The impure half
# --------------------------------------------------------------------


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if k not in _SCRUBBED_ENV and not k.startswith(("PIP_", "UV_"))}
    if "UV_CACHE_DIR" in os.environ:
        env["UV_CACHE_DIR"] = os.environ["UV_CACHE_DIR"]
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    shown = [("<script>" if "\n" in part else part) for part in cmd]
    print("+ " + " ".join(shown), file=sys.stderr)
    return subprocess.run(cmd, check=True, env=_clean_env(), **kwargs)


def _find_uv(explicit: str | None) -> str:
    for candidate in (explicit, os.environ.get("UV"), shutil.which("uv"),
                      str(Path.home() / ".local" / "bin" / "uv")):
        if candidate and Path(candidate).is_file():
            return candidate
    raise SystemExit("uv not found: install it (https://docs.astral.sh/uv/) or pass "
                     "--uv /path/to/uv")


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _parse_cutoff(text: str | None) -> _dt.datetime:
    if not text:
        today = _dt.datetime.now(_dt.timezone.utc).date()
        return _dt.datetime(today.year, today.month, today.day, tzinfo=_dt.timezone.utc)
    if len(text) == 10:                                  # YYYY-MM-DD
        day = _dt.date.fromisoformat(text)
        return _dt.datetime(day.year, day.month, day.day, tzinfo=_dt.timezone.utc)
    moment = _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=_dt.timezone.utc)
    return moment.astimezone(_dt.timezone.utc)


def _warn_if_dirty() -> None:
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", "src", "pyproject.toml"],
                             cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return
    if out.strip():
        print("WARNING: src/ or pyproject.toml has uncommitted changes, and the wheel "
              "is built from the working tree; a release SBOM must come from a clean "
              "tree", file=sys.stderr)


def generate(install: str, *, uv: str, wheel: Path, python: str, work: Path,
             tool_python: Path, cutoff: _dt.datetime, index: str,
             pyproject: dict) -> dict:
    """Install ``install`` into a fresh venv and return its finalised SBOM."""
    cutoff_text = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    venv = work / f"env-{install}"
    _run([uv, "venv", "--no-config", "--python", python, str(venv)])
    target = _venv_python(venv)
    spec = f"{check_sbom.install_requirement(install)} @ {wheel.as_uri()}"
    _run([uv, "pip", "install", "--no-config", "--python", str(target),
          "--default-index", index, "--exclude-newer", cutoff_text, spec])

    probe = json.loads(_run([str(target), "-I", "-c", _PROBE],
                            capture_output=True, text=True).stdout)
    raw_path = work / f"raw-{install}.json"
    _run([str(tool_python), "-m", "cyclonedx_py", "environment",
          "--pyproject", str(REPO_ROOT / "pyproject.toml"),
          "--mc-type", "library", "--sv", SPEC_VERSION, "--output-reproducible",
          "--of", "JSON", "-o", str(raw_path), str(target)])
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    uv_version = _run([uv, "--version"], capture_output=True, text=True).stdout.strip()
    sbom = finalise(
        raw, pyproject=pyproject, install=install, probe=probe,
        provenance={"exclude-newer": cutoff_text, "index": index,
                    "resolver": uv_version,
                    "generator": "scripts/generate_sbom.py"},
        timestamp=_timestamp(cutoff),
    )
    problems = installed_mismatch(sbom, probe["distributions"])
    if problems:
        raise SystemExit(f"the {install} SBOM does not match its environment:\n  "
                         + "\n  ".join(problems))
    return sbom


def _validate(tool_python: Path, path: Path) -> None:
    code = (
        "import sys\n"
        "from cyclonedx.schema import SchemaVersion\n"
        "from cyclonedx.validation.json import JsonStrictValidator\n"
        f"v = JsonStrictValidator(SchemaVersion.V{SPEC_VERSION.replace('.', '_')})\n"
        "e = v.validate_str(open(sys.argv[1], encoding='utf-8').read())\n"
        "sys.exit(f'invalid CycloneDX: {e}' if e else 0)\n"
    )
    _run([str(tool_python), "-I", "-c", code, str(path)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--install", action="append", choices=check_sbom.SBOM_INSTALLS,
        help="generate only this covered install (repeatable; default: all)")
    parser.add_argument(
        "--extra", action="append", default=[],
        help="also generate an SBOM for this extra, which is not committed or "
             "checked (e.g. cuda12, or cuda12+server for two at once); needs "
             "an --output-dir other than the default")
    parser.add_argument("--output-dir", type=Path, default=check_sbom.SBOM_DIR)
    parser.add_argument("--python", default="3.12",
                        help="interpreter uv resolves and installs for (default 3.12, "
                             "the CI version)")
    parser.add_argument("--exclude-newer", default=None,
                        help="resolution cutoff, YYYY-MM-DD or RFC 3339 (default: "
                             "00:00 UTC today)")
    parser.add_argument("--index-url", default=DEFAULT_INDEX)
    parser.add_argument("--uv", default=None, help="path to uv")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="where to build the wheel and the venvs (default: a "
                             "temporary directory)")
    parser.add_argument("--keep-work-dir", action="store_true")
    args = parser.parse_args(argv)

    default_out = args.output_dir.resolve() == check_sbom.SBOM_DIR.resolve()
    if args.extra and default_out:
        parser.error("--extra writes an SBOM this repository does not commit; pass "
                     "--output-dir somewhere else")
    installs = list(args.install or ([] if args.extra else check_sbom.SBOM_INSTALLS))
    installs += args.extra

    uv = _find_uv(args.uv)
    cutoff = _parse_cutoff(args.exclude_newer)
    cutoff_text = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
    pyproject = check_sbom.load_pyproject()
    version = str(pyproject["project"]["version"])
    _warn_if_dirty()

    work = Path(tempfile.mkdtemp(prefix="maddening-sbom-", dir=args.work_dir))
    try:
        dist = work / "dist"
        _run([uv, "build", "--no-config", "--wheel", "--python", args.python,
              "--default-index", args.index_url, "--exclude-newer", cutoff_text,
              "--out-dir", str(dist), str(REPO_ROOT)])
        wheels = sorted(dist.glob(f"maddening-{version}-*.whl"))
        if len(wheels) != 1:
            raise SystemExit(f"expected one maddening {version} wheel, built {wheels}")

        tool = work / "tool-venv"
        _run([uv, "venv", "--no-config", "--python", args.python, str(tool)])
        _run([uv, "pip", "install", "--no-config", "--python", str(_venv_python(tool)),
              "--default-index", args.index_url, "--exclude-newer", cutoff_text,
              CYCLONEDX_BOM])

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for install in installs:
            sbom = generate(install, uv=uv, wheel=wheels[0], python=args.python,
                            work=work, tool_python=_venv_python(tool), cutoff=cutoff,
                            index=args.index_url, pyproject=pyproject)
            out = args.output_dir / check_sbom.sbom_filename(version, install)
            out.write_text(check_sbom.dumps(sbom), encoding="utf-8")
            _validate(_venv_python(tool), out)
            print(f"wrote {out} ({len(sbom['components'])} components)")
            # A previous version's SBOM for the same install is stale by
            # definition; the check would refuse it, so remove it here.
            for old in args.output_dir.glob(f"maddening-*-{install}.cdx.json"):
                if old != out:
                    old.unlink()
                    print(f"removed {old} (a previous version)")
    finally:
        if args.keep_work_dir:
            print(f"kept {work}", file=sys.stderr)
        else:
            shutil.rmtree(work, ignore_errors=True)

    if default_out:
        return check_sbom.main([])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
