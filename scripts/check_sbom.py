#!/usr/bin/env python3
"""Check the committed CycloneDX SBOMs against ``pyproject.toml`` and the SOUP package.

``docs/validation/sbom/`` holds one CycloneDX SBOM per install that
``scripts/generate_sbom.py`` covers: ``maddening-<version>-<install>.cdx.json``,
where ``<install>`` is ``core`` (``pip install maddening``) or an extra
(``pip install maddening[<extra>]``).  Each one is the environment a clean
install of the wheel resolved to, on the Python and platform its metadata
records.  This script needs no network: it reads the committed files and
checks that they still describe the package ``pyproject.toml`` declares.

What it checks, and names on failure
------------------------------------
Per directory:

* one SBOM for every install in :data:`SBOM_INSTALLS`, at the version
  ``pyproject.toml`` declares -- a version bump without regenerating fails
  here, which is what makes regeneration a release step that cannot be
  skipped;
* no other ``*.cdx.json`` (a previous version's file, an install nobody
  covers);
* ``soup_package.md`` names every SBOM file, so the SOUP document and the
  directory cannot drift apart.

Per SBOM:

* the root component is ``maddening`` at the declared version, with its
  purl and the declared licence;
* the metadata records which install it is and the environment it was
  resolved in (the PEP 508 marker variables), since the transitive
  versions depend on both;
* every component has a name, a version and a ``pkg:pypi`` purl that
  agrees with both, and no package appears twice;
* every direct dependency ``pyproject.toml`` declares for that install
  (the base dependencies, plus the extra's) is present, at a version its
  specifier admits, and is a direct edge of the root component in the
  dependency graph -- and nothing else is;
* every SOUP item ``soup_package.md`` lists (the Base Dependencies row of
  its Software Identification table) is present, is a base dependency in
  ``pyproject.toml``, and is at a version the ``pyproject.toml`` range
  admits;
* every ``dependsOn`` reference resolves;
* components and dependencies are in canonical order, and the
  ``serialNumber`` is the one the content implies (:func:`seal`), so a
  hand edit -- bumping a version in the JSON to make this check pass, say
  -- is refused rather than trusted.

Usage
-----
::

    python scripts/check_sbom.py                 # check docs/validation/sbom/
    python scripts/check_sbom.py --normalise F   # print F without its volatile fields

``--normalise`` drops ``serialNumber``, ``metadata.timestamp`` and the
resolution cutoff property, the three fields that differ between two
generations of the same environment on different days, so two SBOMs can be
compared with ``diff``.  ``tests/compliance/test_sbom_check.py`` runs the
check on the committed files, so CI fails on a stale SBOM.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import tomllib
import uuid
from pathlib import Path
from typing import Any, Iterable

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
SOUP_PACKAGE = REPO_ROOT / "docs" / "validation" / "soup_package.md"
SBOM_DIR = REPO_ROOT / "docs" / "validation" / "sbom"

#: The installs that get a committed SBOM.  ``core`` is the base install,
#: what every user gets; the rest are extras.  Why these and not others is
#: argued in ``soup_package.md`` §6: ``server`` is the network-facing
#: bundle (and a superset of ``api``, ``network``, ``terminal``, ``viz``
#: and ``compression``), ``surrogates`` puts a trained network in the
#: computed result, and ``usd`` reads geometry into a graph.
SBOM_INSTALLS: tuple[str, ...] = ("core", "server", "surrogates", "usd")

#: Namespace of the properties ``generate_sbom.py`` writes into
#: ``metadata.properties``.  ``cdx:`` is reserved by CycloneDX.
PROP = "maddening:sbom:"
PROP_INSTALL = PROP + "install"
PROP_EXCLUDE_NEWER = PROP + "exclude-newer"
PROP_MARKER = PROP + "marker:"

#: The PEP 508 marker variables an SBOM must record: enough to evaluate
#: any marker ``pyproject.toml`` could put on a dependency, and none of
#: the host-identifying ones (``platform_release``, ``platform_version``).
MARKER_VARIABLES: tuple[str, ...] = (
    "implementation_name",
    "os_name",
    "platform_machine",
    "platform_python_implementation",
    "platform_system",
    "python_full_version",
    "python_version",
    "sys_platform",
)

ROOT_NAME = "maddening"

#: Fixed namespace for the content-derived serial number.  Any constant
#: works; changing it changes every serial, so it never changes.
_SERIAL_NAMESPACE = uuid.UUID("8f0cf2a4-5d0e-4c55-9d8c-6a0a2f1f0c11")

_FILE_RE = re.compile(r"^maddening-(?P<version>.+)-(?P<install>[a-z0-9+-]+)\.cdx\.json$")


# --------------------------------------------------------------------
# Names, files, sealing
# --------------------------------------------------------------------


def sbom_filename(version: str, install: str) -> str:
    """The committed file name of the SBOM for ``install`` at ``version``."""
    return f"maddening-{version}-{install}.cdx.json"


def root_purl(version: str) -> str:
    return f"pkg:pypi/{ROOT_NAME}@{version}"


def _component_sort_key(component: dict) -> tuple[str, str, str]:
    return (canonicalize_name(str(component.get("name", ""))),
            str(component.get("version", "")),
            str(component.get("bom-ref", "")))


def canonical_order(sbom: dict) -> dict:
    """Return a copy with components, dependencies and properties sorted.

    Components by canonical package name then version, the dependency
    graph by ``ref`` with each ``dependsOn`` sorted, and
    ``metadata.properties`` by name.  Everything else keeps its order:
    JSON object keys are sorted when the file is written.
    """
    out = copy.deepcopy(sbom)
    out["components"] = sorted(out.get("components", []), key=_component_sort_key)
    deps = []
    for dep in out.get("dependencies", []):
        dep = dict(dep)
        if "dependsOn" in dep:
            dep["dependsOn"] = sorted(dep["dependsOn"])
        deps.append(dep)
    out["dependencies"] = sorted(deps, key=lambda d: str(d.get("ref", "")))
    meta = out.get("metadata", {})
    if "properties" in meta:
        meta["properties"] = sorted(
            meta["properties"], key=lambda p: (str(p.get("name")), str(p.get("value"))))
    return out


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_serial(sbom: dict) -> str:
    """The serial number the content of ``sbom`` implies.

    A UUIDv5 over the canonical JSON of everything except
    ``serialNumber`` itself.  Two generations of the same environment
    with the same timestamp get the same serial, so the committed file is
    reproducible; any change of content gets a different one, as
    CycloneDX asks of a serial; and a file edited after generation no
    longer matches its own serial, which :func:`check_sbom` refuses.
    """
    body = {k: v for k, v in sbom.items() if k != "serialNumber"}
    digest = hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()
    return f"urn:uuid:{uuid.uuid5(_SERIAL_NAMESPACE, digest)}"


def seal(sbom: dict) -> dict:
    """Put ``sbom`` in canonical order and stamp its content serial."""
    out = canonical_order(sbom)
    out.pop("serialNumber", None)
    out["serialNumber"] = content_serial(out)
    return out


def dumps(sbom: dict) -> str:
    """The on-disk form: two-space indent, sorted keys, trailing newline."""
    return json.dumps(sbom, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def normalise(sbom: dict) -> dict:
    """``sbom`` without the fields that differ between two generation days.

    Drops ``serialNumber`` (it hashes the timestamp), ``metadata.timestamp``
    and the ``maddening:sbom:exclude-newer`` resolution cutoff.  What is
    left is the resolved environment: two normalised SBOMs are equal
    exactly when the same packages, at the same versions, were resolved
    for the same install on the same Python and platform.
    """
    out = canonical_order(sbom)
    out.pop("serialNumber", None)
    meta = out.get("metadata", {})
    meta.pop("timestamp", None)
    if "properties" in meta:
        meta["properties"] = [p for p in meta["properties"]
                              if p.get("name") != PROP_EXCLUDE_NEWER]
    return out


# --------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------


def load_pyproject(path: Path = PYPROJECT) -> dict:
    with Path(path).open("rb") as fh:
        return tomllib.load(fh)


def install_extras(install: str) -> list[str]:
    """The extras an install name selects: none for ``core``, else ``+``-joined."""
    return [] if install == "core" else install.split("+")


def install_requirement(install: str) -> str:
    """The requirement string that installs ``install``: ``maddening[a,b]``."""
    extras = install_extras(install)
    return ROOT_NAME + (f"[{','.join(extras)}]" if extras else "")


def declared_requirements(pyproject: dict, install: str) -> list[Requirement]:
    """The direct dependencies ``pip install maddening[<install>]`` asks for.

    The base dependencies, plus each selected extra's own list when
    ``install`` is not ``core`` (``cuda12+server`` selects two).  A
    self-reference (``maddening[all]`` in ``dev``) is expanded into the
    extras it names.  Raises ``KeyError`` for an extra ``pyproject.toml``
    does not declare.
    """
    project = pyproject["project"]
    extras = project.get("optional-dependencies", {})
    reqs = [Requirement(r) for r in project.get("dependencies", [])]
    pending = install_extras(install)
    seen: set[str] = set()
    while pending:
        extra = pending.pop()
        if extra in seen:
            continue
        seen.add(extra)
        if extra not in extras:
            raise KeyError(extra)
        for spec in extras[extra]:
            req = Requirement(spec)
            if canonicalize_name(req.name) == ROOT_NAME:
                pending.extend(sorted(req.extras))
            else:
                reqs.append(req)
    return reqs


def _generated_block(text: str, name: str) -> str | None:
    match = re.search(
        r"<!-- BEGIN GENERATED: " + re.escape(name) + r"(?![\w-])[^>]*-->\n(.*?)"
        r"<!-- END GENERATED: " + re.escape(name) + r" -->",
        text, re.DOTALL)
    return match.group(1) if match else None


def soup_items(soup_text: str) -> list[str]:
    """The requirement strings ``soup_package.md`` lists as SOUP.

    Read from the ``Base Dependencies`` row of the generated Software
    Identification table.  Raises ``ValueError`` when the row cannot be
    found or is empty: a check that finds nothing to check must not pass.
    """
    block = _generated_block(soup_text, "software-identification")
    if block is None:
        raise ValueError("soup_package.md has no generated software-identification "
                         "block, so the SOUP items it lists cannot be read")
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[0] == "Base Dependencies":
            raw = cells[1]
            break
    else:
        raise ValueError("soup_package.md's software-identification table has no "
                         "'Base Dependencies' row, so the SOUP items it lists "
                         "cannot be read")
    # The generator joins requirements with ", ".  A specifier may itself
    # hold a comma, with or without a space ("jax>=0.10, <0.13"), so a
    # piece that starts with an operator belongs to the one before it.
    items: list[str] = []
    for piece in (p.strip() for p in raw.split(",")):
        if not piece:
            continue
        if items and piece[0] in "<>=!~":
            items[-1] = f"{items[-1]},{piece}"
        else:
            items.append(piece)
    if not items:
        raise ValueError("soup_package.md's 'Base Dependencies' row is empty")
    return items


# --------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------


def _properties(sbom: dict) -> dict[str, str]:
    return {str(p.get("name")): str(p.get("value"))
            for p in sbom.get("metadata", {}).get("properties", [])}


def marker_environment(sbom: dict) -> dict[str, str]:
    """The PEP 508 marker variables the SBOM records (possibly incomplete)."""
    props = _properties(sbom)
    return {var: props[PROP_MARKER + var]
            for var in MARKER_VARIABLES if PROP_MARKER + var in props}


def _purl_parts(purl: str) -> tuple[str, str, str] | None:
    """``(type, name, version)`` of a purl, or ``None`` if it is not one."""
    if not purl.startswith("pkg:"):
        return None
    body = purl[4:].split("#", 1)[0].split("?", 1)[0]
    if "/" not in body or "@" not in body:
        return None
    ptype, rest = body.split("/", 1)
    name, _, version = rest.rpartition("@")
    return ptype, name, version


def _license_values(component: dict) -> set[str]:
    values = set()
    for entry in component.get("licenses", []) or []:
        if "expression" in entry:
            values.add(str(entry["expression"]))
        lic = entry.get("license", {})
        for key in ("id", "name"):
            if key in lic:
                values.add(str(lic[key]))
    return values


def _declared_license(pyproject: dict) -> str | None:
    lic = pyproject["project"].get("license")
    if isinstance(lic, str):
        return lic
    if isinstance(lic, dict):
        return lic.get("text")
    return None


def _admits(req: Requirement, version: str) -> bool:
    try:
        # prereleases=True: what ``pip check`` asks -- a resolved
        # pre-release inside the numeric range satisfies the requirement.
        return req.specifier.contains(Version(version), prereleases=True)
    except InvalidVersion:
        return False


def check_sbom(sbom: dict, *, pyproject: dict, install: str,
               soup_requirements: Iterable[str], source: str = "SBOM") -> list[str]:
    """Every discrepancy between one SBOM and its sources, each named.

    ``install`` is the install the SBOM is expected to describe
    (``"core"`` or an extra).  ``soup_requirements`` are the requirement
    strings the SOUP document lists (:func:`soup_items`).  Returns an
    empty list when the SBOM is consistent.
    """
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(f"{source}: {msg}")

    version = str(pyproject["project"]["version"])

    if sbom.get("bomFormat") != "CycloneDX":
        err(f"bomFormat is {sbom.get('bomFormat')!r}, not 'CycloneDX'")
    if not sbom.get("specVersion"):
        err("has no specVersion")

    # -- sealing and order -------------------------------------------
    serial = sbom.get("serialNumber")
    if serial is None:
        err("has no serialNumber; regenerate it with scripts/generate_sbom.py")
    elif serial != content_serial(sbom):
        err("serialNumber does not match the content, so the file was changed "
            "after generation (a hand edit, or a merge); regenerate it with "
            "scripts/generate_sbom.py rather than editing it")
    if sbom.get("components", []) != canonical_order(sbom)["components"]:
        err("components are not in canonical order (by package name, then "
            "version); regenerate it with scripts/generate_sbom.py")
    if sbom.get("dependencies", []) != canonical_order(sbom)["dependencies"]:
        err("the dependency graph is not in canonical order; regenerate it with "
            "scripts/generate_sbom.py")

    # -- root component ----------------------------------------------
    root = sbom.get("metadata", {}).get("component") or {}
    root_ref = root.get("bom-ref")
    if root.get("name") != ROOT_NAME:
        err(f"root component is {root.get('name')!r}, not {ROOT_NAME!r}")
    if root.get("version") != version:
        err(f"root component is maddening {root.get('version')}, but "
            f"pyproject.toml says {version}; regenerate the SBOMs for this version")
    if root.get("purl") != root_purl(version):
        err(f"root component purl is {root.get('purl')!r}, expected "
            f"{root_purl(version)!r}")
    if not root_ref:
        err("root component has no bom-ref, so the dependency graph cannot "
            "name it")
    declared_lic = _declared_license(pyproject)
    if declared_lic and declared_lic not in _license_values(root):
        err(f"root component licence is {sorted(_license_values(root))}, but "
            f"pyproject.toml declares {declared_lic!r}")

    # -- provenance --------------------------------------------------
    props = _properties(sbom)
    recorded_install = props.get(PROP_INSTALL)
    if recorded_install != install:
        err(f"records install {recorded_install!r} in {PROP_INSTALL}, "
            f"expected {install!r}")
    env = marker_environment(sbom)
    missing_env = [v for v in MARKER_VARIABLES if v not in env]
    if missing_env:
        err(f"does not record the environment it was resolved in (missing "
            f"{', '.join(PROP_MARKER + v for v in missing_env)}); the resolved "
            f"versions depend on the Python and platform, and the dependency "
            f"markers cannot be evaluated without them")

    # -- components --------------------------------------------------
    components = sbom.get("components", [])
    by_name: dict[str, dict] = {}
    refs: set[str] = {root_ref} if root_ref else set()
    for i, comp in enumerate(components):
        name = comp.get("name")
        cversion = comp.get("version")
        label = f"component {name or f'#{i}'}"
        if not name:
            err(f"component #{i} has no name")
            continue
        if not cversion:
            err(f"{label} has no version")
        purl = comp.get("purl")
        if not purl:
            err(f"{label} has no purl")
        else:
            parts = _purl_parts(purl)
            if parts is None or parts[0] != "pypi":
                err(f"{label} purl {purl!r} is not a pkg:pypi purl")
            else:
                if canonicalize_name(parts[1]) != canonicalize_name(name):
                    err(f"{label} purl {purl!r} names a different package")
                if cversion and parts[2] != cversion:
                    err(f"{label} purl {purl!r} names version {parts[2]!r}, "
                        f"the component says {cversion!r}")
        key = canonicalize_name(name)
        if key == ROOT_NAME:
            err("maddening is listed as a component as well as the root: the "
                "environment scan picked up the installed wheel")
        if key in by_name:
            err(f"{label} appears twice ({by_name[key].get('version')} and "
                f"{cversion}); one environment holds one version of a package")
        by_name[key] = comp
        if comp.get("bom-ref"):
            refs.add(comp["bom-ref"])

    # -- declared direct dependencies ---------------------------------
    try:
        direct = declared_requirements(pyproject, install)
    except KeyError as exc:
        err(f"pyproject.toml declares no extra {exc.args[0]!r}")
        direct = []
    expected_edges: set[str] = set()
    for req in direct:
        if req.marker is not None:
            if missing_env:
                err(f"cannot evaluate the marker of direct dependency {req} "
                    f"without the recorded environment")
                continue
            if not req.marker.evaluate({**env, "extra": ""}):
                continue
        comp = by_name.get(canonicalize_name(req.name))
        if comp is None:
            err(f"direct dependency {req} declared in pyproject.toml "
                f"({'base' if install == 'core' else f'base or [{install}]'}) "
                f"is missing from the SBOM")
            continue
        if not _admits(req, str(comp.get("version", ""))):
            err(f"direct dependency {req.name} is at {comp.get('version')} in the "
                f"SBOM, outside the range pyproject.toml declares ({req})")
        if comp.get("bom-ref"):
            expected_edges.add(comp["bom-ref"])

    # -- SOUP items ---------------------------------------------------
    base = {canonicalize_name(r.name): r
            for r in (Requirement(s) for s in pyproject["project"].get("dependencies", []))}
    for item in soup_requirements:
        try:
            listed = Requirement(item)
        except InvalidRequirement:
            err(f"SOUP item {item!r} in soup_package.md is not a requirement string")
            continue
        key = canonicalize_name(listed.name)
        declared = base.get(key)
        if declared is None:
            err(f"SOUP item {listed.name} is listed in soup_package.md, but "
                f"pyproject.toml declares no base dependency of that name")
        comp = by_name.get(key)
        if comp is None:
            err(f"SOUP item {listed.name} listed in soup_package.md is not in the SBOM")
            continue
        if declared is not None and not _admits(declared, str(comp.get("version", ""))):
            err(f"SOUP item {listed.name} is at {comp.get('version')} in the SBOM, "
                f"outside the range pyproject.toml declares ({declared})")

    # -- dependency graph ---------------------------------------------
    root_edges: set[str] | None = None
    for dep in sbom.get("dependencies", []):
        ref = dep.get("ref")
        if ref not in refs:
            err(f"dependency graph entry {ref!r} names no component")
        for target in dep.get("dependsOn", []):
            if target not in refs:
                err(f"dependency graph: {ref!r} depends on {target!r}, which names "
                    f"no component")
        if root_ref and ref == root_ref:
            root_edges = set(dep.get("dependsOn", []))
    if root_ref:
        if root_edges is None:
            err("the dependency graph has no entry for the root component")
        else:
            for ref in sorted(expected_edges - root_edges):
                err(f"the root component does not depend on {ref!r}, a declared "
                    f"direct dependency")
            # An edge to a ref that names no component is reported above
            # as dangling; calling it "undeclared" too would be wrong when
            # it is a declared dependency whose component is missing.
            for ref in sorted((root_edges - expected_edges) & refs):
                err(f"the root component depends on {ref!r}, which pyproject.toml "
                    f"does not declare as a direct dependency of this install")

    return errors


def check_directory(sbom_dir: Path, *, pyproject: dict, soup_text: str,
                    installs: Iterable[str] = SBOM_INSTALLS) -> tuple[list[str], list[dict]]:
    """Check every committed SBOM; return ``(errors, the SBOMs read)``."""
    errors: list[str] = []
    version = str(pyproject["project"]["version"])
    installs = tuple(installs)
    try:
        soup_reqs = soup_items(soup_text)
    except ValueError as exc:
        errors.append(f"soup_package.md: {exc}")
        soup_reqs = []

    expected = {sbom_filename(version, i): i for i in installs}
    present = {p.name for p in Path(sbom_dir).glob("*.cdx.json")} if Path(sbom_dir).is_dir() else set()
    for name in sorted(present - set(expected)):
        match = _FILE_RE.match(name)
        why = ("a previous version's file" if match and match["version"] != version
               else "an install this check does not cover")
        errors.append(f"{Path(sbom_dir).name}/{name}: unexpected SBOM ({why}); "
                      f"delete it, or add its install to SBOM_INSTALLS")
    sboms = []
    for name, install in expected.items():
        if name not in soup_text:
            errors.append(f"soup_package.md does not name {name}, the {install} "
                          f"SBOM; list it in §6")
        path = Path(sbom_dir) / name
        if name not in present:
            errors.append(f"{Path(sbom_dir).name}/{name}: missing -- the {install} "
                          f"SBOM for maddening {version}.  Run "
                          f"`python scripts/generate_sbom.py` (it needs network "
                          f"access and uv)")
            continue
        try:
            sbom = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{Path(sbom_dir).name}/{name}: unreadable ({exc})")
            continue
        sboms.append(sbom)
        errors += check_sbom(sbom, pyproject=pyproject, install=install,
                             soup_requirements=soup_reqs,
                             source=f"{Path(sbom_dir).name}/{name}")
    return errors, sboms


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sbom-dir", type=Path, default=SBOM_DIR)
    parser.add_argument("--pyproject", type=Path, default=PYPROJECT)
    parser.add_argument("--soup", type=Path, default=SOUP_PACKAGE)
    parser.add_argument(
        "--normalise", type=Path, metavar="FILE",
        help="print FILE without serialNumber, timestamp and resolution cutoff, "
             "for diffing two SBOMs; checks nothing")
    args = parser.parse_args(argv)

    if args.normalise:
        sbom = json.loads(args.normalise.read_text(encoding="utf-8"))
        sys.stdout.write(dumps(normalise(sbom)))
        return 0

    pyproject = load_pyproject(args.pyproject)
    errors, sboms = check_directory(
        args.sbom_dir, pyproject=pyproject,
        soup_text=args.soup.read_text(encoding="utf-8"))
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(f"FAILED: {len(errors)} discrepancy(ies) between the SBOMs and "
              f"their sources", file=sys.stderr)
        return 1
    counts = ", ".join(
        f"{_properties(s).get(PROP_INSTALL)} {len(s.get('components', []))}"
        for s in sboms)
    print(f"OK: {len(sboms)} SBOMs for maddening {pyproject['project']['version']} "
          f"match pyproject.toml and soup_package.md (components: {counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
