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
* every component has a name, a version, a ``pkg:pypi`` purl that agrees
  with both, and a licence (``soup_package.md`` §6: "the licence its
  metadata declares"), and no package appears twice;
* every component is reachable from the root component through the
  dependency graph: an SBOM is what one install resolved to, and a
  package nothing in that install depends on is not part of it;
* the Python the SBOM records (``python_full_version`` and
  ``python_version``) is one ``requires-python`` admits: an environment on
  any other Python is not one ``pip install maddening`` can produce;
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
from packaging.specifiers import InvalidSpecifier, SpecifierSet
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
#: Per component: one property per line of the package's own
#: ``Requires-Dist`` metadata, verbatim, and how many lines there were.
PROP_REQUIRES_DIST = PROP + "requires-dist"
PROP_REQUIRES_DIST_COUNT = PROP + "requires-dist-count"

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
    ``metadata.properties`` and each component's ``properties`` by name
    then value.  Everything else keeps its order:
    JSON object keys are sorted when the file is written.
    """
    out = copy.deepcopy(sbom)
    out["components"] = sorted(out.get("components", []), key=_component_sort_key)
    for comp in out["components"]:
        if isinstance(comp, dict) and "properties" in comp:
            comp["properties"] = sorted(
                comp["properties"],
                key=lambda p: (str(p.get("name")), str(p.get("value"))))
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
    """The licence identifiers, names and expressions a component carries.

    Blank ones are not licences: ``{"license": {"name": ""}}`` names
    nothing, and passed as "carries a licence" until audit_040_p4_4 (M4,
    G4) re-sealed one onto numpy.
    """
    values = set()
    for entry in component.get("licenses", []) or []:
        if not isinstance(entry, dict):
            continue
        candidates = [entry.get("expression")]
        lic = entry.get("license", {})
        if isinstance(lic, dict):
            candidates += [lic.get("id"), lic.get("name")]
        values.update(str(v).strip() for v in candidates
                      if v is not None and str(v).strip())
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


#: The cutoff's one spelling, as ``generate_sbom.py`` writes it.
_CUTOFF_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

#: Marker variables an SBOM does not record, because they identify the
#: host.  A ``Requires-Dist`` marker that reads one cannot be evaluated
#: from the SBOM alone, so it is reported rather than evaluated against
#: whatever machine runs this check.
_UNRECORDED_MARKER_VARIABLES = ("platform_release", "platform_version",
                                "implementation_version")


def recorded_requirements(component: dict) -> tuple[list[str] | None, str | None]:
    """``(requirement strings, problem)``: the ``Requires-Dist`` a component records.

    ``generate_sbom.py`` writes one ``maddening:sbom:requires-dist``
    property per line of the package's metadata and a
    ``maddening:sbom:requires-dist-count``, so that a package that
    requires nothing (``[]``) is told apart from a record that was
    removed (``None``, with the problem).
    """
    props = [p for p in component.get("properties", []) or [] if isinstance(p, dict)]
    reqs = [str(p.get("value")) for p in props if p.get("name") == PROP_REQUIRES_DIST]
    counts = [str(p.get("value")) for p in props
              if p.get("name") == PROP_REQUIRES_DIST_COUNT]
    if not counts:
        return None, (f"does not record its Requires-Dist ({PROP_REQUIRES_DIST_COUNT} "
                      f"is missing), so whether the install is complete cannot be "
                      f"checked; regenerate it with scripts/generate_sbom.py")
    if len(counts) > 1 or counts[0] != str(len(reqs)):
        return None, (f"records {counts} as its {PROP_REQUIRES_DIST_COUNT} but "
                      f"carries {len(reqs)} {PROP_REQUIRES_DIST} line(s)")
    return reqs, None


def _requirement_closure_errors(components: list[dict], direct: list[Requirement],
                                env: dict[str, str], graph: dict[str, set],
                                root_ref: str | None, install: str) -> list[str]:
    """What the components' own ``Requires-Dist`` says the SBOM lacks.

    The install's requirements are followed from the root's declared
    direct dependencies through each component's recorded
    ``Requires-Dist``, with each marker evaluated in the recorded
    environment and with the extras its dependants asked for
    (``uvicorn[standard]`` turns on uvicorn's ``extra == "standard"``
    lines).  Then:

    * every requirement the install turns on names a component, at a
      version the requirement admits -- a dropped transitive dependency
      (scipy, which jax and jaxlib require) and one rewritten to a version
      its dependants refuse (ml-dtypes 0.0.1 under jax's ``>=0.5.0``)
      both passed when re-sealed (audit_040_p4_4, M4, G1 and G2);
    * each such requirement is an edge of the dependency graph, and each
      edge is a requirement the component records (cyclonedx-py also
      links the installed packages an unselected extra names, so an edge
      may be a requirement whose marker is off -- but never one the
      package does not declare at all);
    * every component is reached by the requirements the install turns
      on: one reached only through edges, or not at all, is not part of
      the install.
    """
    errors: list[str] = []
    by_key = {canonicalize_name(str(c.get("name"))): c for c in components
              if c.get("name")}
    parsed: dict[str, list[Requirement]] = {}
    for key, comp in by_key.items():
        reqs, problem = recorded_requirements(comp)
        if problem:
            errors.append(f"component {comp.get('name')} {problem}")
            continue
        parsed[key] = []
        for text in reqs or []:
            try:
                req = Requirement(text)
            except InvalidRequirement:
                errors.append(f"component {comp.get('name')} records "
                              f"{PROP_REQUIRES_DIST} {text!r}, which is not a "
                              f"requirement")
                continue
            if req.marker is not None and any(
                    v in str(req.marker) for v in _UNRECORDED_MARKER_VARIABLES):
                errors.append(f"component {comp.get('name')} requires {text!r}, "
                              f"whose marker reads a variable this SBOM does not "
                              f"record ({', '.join(_UNRECORDED_MARKER_VARIABLES)})")
                continue
            parsed[key].append(req)
    if errors:
        return errors

    def on(req: Requirement, extras: set[str]) -> bool:
        if req.marker is None:
            return True
        return any(req.marker.evaluate({**env, "extra": e})
                   for e in ({""} | extras))

    # Extras each component is asked for, to a fixed point.
    extras: dict[str, set[str]] = {key: set() for key in by_key}
    reached: set[str] = set()
    todo: list[str] = []
    for req in direct:
        if not on(req, set()):
            continue
        key = canonicalize_name(req.name)
        if key in by_key:
            extras[key] |= {canonicalize_name(e) for e in req.extras}
            todo.append(key)
    while todo:
        key = todo.pop()
        reached.add(key)
        for req in parsed.get(key, []):
            if not on(req, extras[key]):
                continue
            target = canonicalize_name(req.name)
            if target not in by_key:
                continue
            new = {canonicalize_name(e) for e in req.extras} - extras[target]
            if new or target not in reached:
                extras[target] |= new
                todo.append(target)

    key_of_ref = {c.get("bom-ref"): key for key, c in by_key.items()
                  if c.get("bom-ref")}
    for key in sorted(reached):
        comp = by_key[key]
        name, ref = comp.get("name"), comp.get("bom-ref")
        edge_keys = {key_of_ref[r] for r in graph.get(ref, set()) if r in key_of_ref}
        for req in parsed.get(key, []):
            if not on(req, extras[key]):
                continue
            target = by_key.get(canonicalize_name(req.name))
            if target is None:
                errors.append(
                    f"component {name} requires {req} (its recorded Requires-Dist, "
                    f"on in this environment), but the SBOM holds no "
                    f"{req.name}: `pip install {install_requirement(install)}` "
                    f"cannot have resolved without it")
                continue
            if not _admits(req, str(target.get("version", ""))):
                errors.append(
                    f"component {name} requires {req}, but the SBOM holds "
                    f"{target.get('name')} {target.get('version')}, which that "
                    f"refuses: no resolver installs the two together")
            if canonicalize_name(req.name) not in edge_keys:
                errors.append(
                    f"dependency graph: {ref!r} has no edge to "
                    f"{target.get('bom-ref')!r}, although {name} requires {req}")
        named = {canonicalize_name(r.name) for r in parsed.get(key, [])}
        for target_key in sorted(edge_keys - named):
            errors.append(
                f"dependency graph: {ref!r} depends on {by_key[target_key].get('bom-ref')!r}, "
                f"but {name}'s recorded Requires-Dist names no {target_key}")
    for key in sorted(set(by_key) - reached):
        comp = by_key[key]
        errors.append(
            f"component {comp.get('name')} is required by nothing "
            f"`pip install {install_requirement(install)}` turns on (following "
            f"the declared direct dependencies and each component's recorded "
            f"Requires-Dist), so it is not part of this install")
    return errors


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
    # The cutoff is a third of what decides the transitive versions, and
    # soup_package.md §6 says every file records it.  Removing it passed
    # when re-sealed (audit_040_p4_4, M4, G3).
    cutoff = props.get(PROP_EXCLUDE_NEWER)
    if cutoff is None:
        err(f"does not record its resolution cutoff ({PROP_EXCLUDE_NEWER}); "
            f"the transitive versions depend on it, and soup_package.md §6 says "
            f"every SBOM records it.  Regenerate it with scripts/generate_sbom.py")
    elif not _CUTOFF_RE.fullmatch(cutoff):
        err(f"records {PROP_EXCLUDE_NEWER} {cutoff!r}, not a UTC timestamp "
            f"(YYYY-MM-DDTHH:MM:SSZ) as scripts/generate_sbom.py writes it")
    env = marker_environment(sbom)
    missing_env = [v for v in MARKER_VARIABLES if v not in env]
    if missing_env:
        err(f"does not record the environment it was resolved in (missing "
            f"{', '.join(PROP_MARKER + v for v in missing_env)}); the resolved "
            f"versions depend on the Python and platform, and the dependency "
            f"markers cannot be evaluated without them")
    # An SBOM resolved on a Python requires-python refuses describes no
    # install anybody can make (audit_040_p4_2, S3: 3.11 recorded against
    # ">=3.12", resealed, passed).  No requires-python is no constraint.
    requires_python = pyproject["project"].get("requires-python")
    if requires_python:
        try:
            python_spec = SpecifierSet(str(requires_python))
        except InvalidSpecifier:
            err(f"pyproject.toml's requires-python {requires_python!r} is not a "
                f"version specifier, so the recorded Python cannot be checked")
            python_spec = None
        for var in ("python_full_version", "python_version"):
            value = env.get(var)
            if python_spec is None or value is None:
                continue                 # a missing variable is reported above
            try:
                admitted = python_spec.contains(Version(value), prereleases=True)
            except InvalidVersion:
                err(f"records {PROP_MARKER}{var} {value!r}, which is not a "
                    f"version")
                continue
            if not admitted:
                err(f"records {var} {value}, outside requires-python "
                    f"{requires_python!r} in pyproject.toml: no install of "
                    f"this package resolves on that Python; regenerate on a "
                    f"supported one")

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
        if not _license_values(comp):
            err(f"{label} carries no licence; soup_package.md §6 says every "
                f"component carries the licence its metadata declares.  "
                f"Regenerate it with scripts/generate_sbom.py; if the "
                f"package's metadata names no licence, say so in §6 rather "
                f"than dropping the field")
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

    # -- the components are closed under their own requirements ----------
    if not missing_env:
        graph_edges: dict[str, set] = {}
        for dep in sbom.get("dependencies", []):
            graph_edges.setdefault(dep.get("ref"), set()).update(dep.get("dependsOn", []))
        for e in _requirement_closure_errors(
                [c for c in components if isinstance(c, dict)],
                [r for r in direct], env, graph_edges, root_ref, install):
            err(e)

    # -- every component is something the install brings in -------------
    # An orphan -- a component no path from the root reaches -- passed when
    # it was resealed (audit_040_p4_2, S2), although §6 says each SBOM is
    # exactly the environment one install resolved to.
    if root_ref and root_edges is not None:
        graph: dict[str, list] = {}
        for dep in sbom.get("dependencies", []):
            graph.setdefault(dep.get("ref"), []).extend(dep.get("dependsOn", []))
        reached: set[str] = set()
        todo = [root_ref]
        while todo:
            ref = todo.pop()
            if ref in reached:
                continue
            reached.add(ref)
            todo.extend(graph.get(ref, []))
        for i, comp in enumerate(components):
            name = comp.get("name") or f"#{i}"
            ref = comp.get("bom-ref")
            if not ref:
                err(f"component {name} has no bom-ref, so the dependency graph "
                    f"cannot reach it from the root")
            elif ref not in reached:
                err(f"component {name} ({ref!r}) is reached by no path from the "
                    f"root component in the dependency graph: nothing "
                    f"`pip install {install_requirement(install)}` resolves "
                    f"depends on it, so it is not part of this install")

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
    recorded: dict[str, dict[str, str]] = {}
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
        recorded[name] = _properties(sbom)
    errors += _shared_resolution_errors(recorded, Path(sbom_dir).name)
    return errors, sboms


#: The properties every SBOM in the directory must agree on: one
#: resolution -- one cutoff, one index, one resolver, one Python and
#: platform -- of several installs.
SHARED_PROPERTIES: tuple[str, ...] = (
    PROP_EXCLUDE_NEWER, PROP + "index", PROP + "resolver", PROP + "platform",
    PROP + "libc", *(PROP_MARKER + v for v in MARKER_VARIABLES))


def _shared_resolution_errors(recorded: dict[str, dict[str, str]],
                              where: str) -> list[str]:
    """Refuse a directory whose SBOMs were not resolved together.

    Each file is checked alone by :func:`check_sbom`, so a server SBOM
    resolved on another day, with another numpy, than the core one passed
    (audit_040_p4_4, M4, G5) -- and the SOUP package's table, which reads
    the four as one record of one release, would have described two
    environments as one.  A property missing from a file is reported per
    file; here it simply has no value to agree on.
    """
    errors = []
    for prop in SHARED_PROPERTIES:
        values: dict[str, list[str]] = {}
        for name, props in sorted(recorded.items()):
            if prop in props:
                values.setdefault(props[prop], []).append(name)
        if len(values) > 1:
            listing = "; ".join(f"{v!r} in {', '.join(names)}"
                                for v, names in sorted(values.items()))
            errors.append(
                f"{where}: the SBOMs disagree about {prop} ({listing}).  They "
                f"are one record of one release, resolved together: regenerate "
                f"them all in one run of scripts/generate_sbom.py")
    return errors


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
