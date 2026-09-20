#!/usr/bin/env python
"""Guard the signatures of every ``STABLE`` public surface.

A surface tagged ``@stability(StabilityLevel.STABLE)`` promises that its
signature will not change incompatibly before the next major version (see
``docs/developer_guide/deprecation_policy.md``).  Nothing enforced that
promise: a keyword could be renamed, a default flipped or a positional
parameter inserted and every test would still pass, because the tests call
the surface the new way.

This script records the signature of every ``STABLE`` surface in a committed
snapshot (``docs/developer_guide/stable_api.json``) and compares the tree
against it.  A changed or removed signature fails; an added one fails too,
but with a different message, because adding a surface is not a break — the
snapshot is merely out of date.

The snapshot covers, for each tagged surface:

* functions — parameters (name, kind, default, annotation) and the return
  annotation;
* classes — the constructor, plus every public method and property the class
  *resolves* (inherited members included, because
  ``BallNode().update(...)`` is the promise, not ``BallNode.update``'s
  definition site), restricted to members defined inside ``maddening``.

Accepting an intended change::

    python scripts/check_stable_signatures.py --update

and say in the commit message why the change is compatible, or which major
release carries it.  ``--update`` is the only supported way to move the
snapshot; hand-editing it is how a break sneaks through.

Exit codes:
    0 -- every recorded signature still matches
    1 -- at least one signature changed, was removed, or is unrecorded
    2 -- the check could not be trusted (a module carrying a recorded
         surface could not be imported, or a default does not round-trip)
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

#: Committed snapshot.  Lives beside the generated stability report so the
#: two travel together in review.
DEFAULT_SNAPSHOT = REPO_ROOT / "docs" / "developer_guide" / "stable_api.json"

#: Bumped when the *shape* of a snapshot record changes, so an old snapshot
#: is regenerated rather than silently mis-compared.
SNAPSHOT_FORMAT = 1

_ADDRESS_RE = re.compile(r" at 0x[0-9a-fA-F]+")
_WS_RE = re.compile(r"\s+")
#: ``typing.Optional`` / ``collections.abc.Callable`` render the same as the
#: bare names a PEP 563 module writes.
_TYPING_PREFIX_RE = re.compile(r"\b(?:typing|collections\.abc)\.")
#: ``Optional[ForwardRef('X')]`` is how ``typing`` prints a string annotation
#: nested in a subscript; write it the way the source did.
_FORWARDREF_RE = re.compile(r"ForwardRef\('([^']+)'[^)]*\)")


# --------------------------------------------------------------------------
# Loading the tagged surfaces
# --------------------------------------------------------------------------

def load_registry() -> tuple[dict[str, Any], dict[str, str]]:
    """Import every ``@stability``-tagged module; return the registry.

    Reuses ``scripts/generate_stability_report.py`` so the two never drift
    apart: a module missing from ``STABILITY_MODULES`` is invisible to both,
    and ``tests/compliance/test_stability.py`` already fails on that.

    Only keys under ``maddening.`` are returned.  The registry is a process
    global, and ``@stability`` fires wherever it is applied — including on
    the throwaway classes ``tests/compliance/test_stability.py`` defines
    inside its test functions.  Run in a fresh interpreter that never
    happens, but this function is also called in-process from the tests,
    and a surface named ``tests.compliance.<...>.<locals>.MyClass`` is not
    something the package promises.

    Returns
    -------
    tuple of (dict, dict)
        The ``{qualified name: StabilityLevel}`` registry, and the
        ``{module: reason}`` map of modules skipped for a missing optional
        dependency.
    """
    gen_path = REPO_ROOT / "scripts" / "generate_stability_report.py"
    spec = importlib.util.spec_from_file_location("_stability_gen", gen_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)          # imports the tagged modules
    from maddening.core.compliance.stability import (  # noqa: PLC0415
        _STABILITY_REGISTRY,
    )
    registry = {name: level for name, level in _STABILITY_REGISTRY.items()
                if name.startswith("maddening.")}
    return registry, dict(module.SKIPPED_MODULES)


def resolve(full_name: str) -> Any:
    """Resolve a registry key (``pkg.mod.Qual.Name``) to the live object."""
    parts = full_name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        mod = sys.modules.get(".".join(parts[:i]))
        if mod is None:
            continue
        obj: Any = mod
        try:
            for attr in parts[i:]:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        return obj
    raise LookupError(f"cannot resolve {full_name!r} from the imported modules")


# --------------------------------------------------------------------------
# Rendering a signature
# --------------------------------------------------------------------------

def _annotation(value: Any) -> str | None:
    """Render an annotation as stable text (``None`` when absent).

    A module with ``from __future__ import annotations`` hands us the source
    text; one without hands us the object.  Both are rendered the same way —
    ``Optional[Callable]``, not ``typing.Optional[typing.Callable]`` in one
    file and ``Optional`` in the other — so turning PEP 563 on or off in a
    module is not mistaken for a signature change.
    """
    if value is inspect.Signature.empty:
        return None
    if isinstance(value, str):                       # PEP 563 module
        text = value
    elif inspect.isclass(value):                     # str, bool, SimulationNode
        text = value.__name__
    else:                                            # Optional[...], dict[...]
        text = str(value)
    text = _WS_RE.sub(" ", text.strip().strip("'\""))
    text = _FORWARDREF_RE.sub(r"'\1'", text)
    return _TYPING_PREFIX_RE.sub("", text)


def _default(value: Any) -> str | None:
    """Render a default as stable text (``None`` when the parameter has none).

    A default whose ``repr`` embeds an object address (a bare ``object()``
    sentinel, a lambda) cannot be compared across runs; such a surface is an
    infrastructure error rather than a silent pass.
    """
    if value is inspect.Signature.empty:
        return None
    text = repr(value)
    if _ADDRESS_RE.search(text):
        raise ValueError(
            f"default {text!r} embeds an object address and cannot be "
            "snapshotted; give the surface a comparable default (a module-level "
            "sentinel with a __repr__, or None)"
        )
    return _WS_RE.sub(" ", text)


def signature_record(obj: Any, *, drop_self: bool = False) -> dict[str, Any]:
    """Render one callable's signature as a snapshot record."""
    sig = inspect.signature(obj)
    params = []
    for i, (name, p) in enumerate(sig.parameters.items()):
        if drop_self and i == 0 and name in ("self", "cls"):
            continue
        entry: dict[str, Any] = {"name": name, "kind": p.kind.name}
        annotation = _annotation(p.annotation)
        if annotation is not None:
            entry["annotation"] = annotation
        default = _default(p.default)
        if default is not None:
            entry["default"] = default
        params.append(entry)
    record: dict[str, Any] = {"parameters": params}
    returns = _annotation(sig.return_annotation)
    if returns is not None:
        record["returns"] = returns
    return record


def _owned_by_maddening(value: Any) -> bool:
    """Is this member defined inside the package (rather than inherited from
    ``object``, ``abc`` or a third-party base)?"""
    target = value
    if isinstance(value, property):
        target = value.fget
    target = inspect.unwrap(target) if callable(target) else target
    module = getattr(target, "__module__", "") or ""
    return module == "maddening" or module.startswith("maddening.")


def class_members(cls: type) -> dict[str, dict[str, Any]]:
    """Public methods and properties the class resolves, as snapshot records.

    Inherited members are included on purpose: the promise a ``STABLE`` class
    makes is about what an instance answers to, so a change to
    ``SimulationNode.update`` correctly shows up against every tagged node
    that inherits it.
    """
    members: dict[str, dict[str, Any]] = {}
    for name in dir(cls):
        if name.startswith("_"):
            continue
        static = inspect.getattr_static(cls, name, None)
        if static is None or not _owned_by_maddening(static):
            continue
        if isinstance(static, property):
            if static.fget is None:
                continue
            record = signature_record(static.fget, drop_self=True)
            record["kind"] = "property"
            record["settable"] = static.fset is not None
            members[name] = record
            continue
        attr = getattr(cls, name, None)
        if not callable(attr):
            continue
        kind = ("classmethod" if isinstance(static, classmethod)
                else "staticmethod" if isinstance(static, staticmethod)
                else "method")
        try:
            record = signature_record(attr, drop_self=(kind == "method"))
        except (TypeError, ValueError) as exc:       # builtin / uninspectable
            raise ValueError(f"{cls.__qualname__}.{name}: {exc}") from exc
        record["kind"] = kind
        members[name] = record
    return members


def build_snapshot(registry: dict[str, Any]) -> dict[str, Any]:
    """Render the whole ``STABLE`` surface as a snapshot document."""
    from maddening.core.compliance.metadata import StabilityLevel  # noqa: PLC0415

    surfaces: dict[str, Any] = {}
    for full_name, level in sorted(registry.items()):
        if level is not StabilityLevel.STABLE:
            continue
        obj = resolve(full_name)
        module = getattr(obj, "__module__", full_name.rsplit(".", 1)[0])
        try:
            if inspect.isclass(obj):
                record = signature_record(obj)
                record["kind"] = "class"
                record["members"] = class_members(obj)
            else:
                record = signature_record(obj)
                record["kind"] = "function"
        except ValueError as exc:
            raise ValueError(f"{full_name}: {exc}") from exc
        record["module"] = module
        surfaces[full_name] = record
    return {"format": SNAPSHOT_FORMAT, "surfaces": surfaces}


def write_snapshot(snapshot: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------

def _flatten(snapshot: dict[str, Any]) -> dict[str, Any]:
    """``{"surface" or "surface.member": record}`` for a readable diff.

    A method can be **both** a member of a tagged class and a tagged surface
    in its own right: ``SimulationNode.static_data_deps`` carries its own
    ``@stability``, and so does ``invalidate_static_cache`` on three classes.
    Those two renderings land on the same key and are not byte-identical --
    the surface is resolved from the unbound function and keeps ``self``, the
    member record drops it -- so one silently overwrote the other, and the
    "N member(s) unchanged" line understated the snapshot by exactly the
    number of such surfaces (4 of 243 today).

    The registered surface wins, because that is the thing the registry
    promises, and the duplicate member is not emitted at all rather than
    written and clobbered.  Surfaces are laid down first so the outcome does
    not depend on dict order.
    """
    flat: dict[str, Any] = {}
    for name, record in snapshot.get("surfaces", {}).items():
        flat[name] = {k: v for k, v in record.items() if k != "members"}
    for name, record in snapshot.get("surfaces", {}).items():
        for member, mrecord in record.get("members", {}).items():
            key = f"{name}.{member}"
            if key in flat:                  # also tagged in its own right
                continue
            flat[key] = mrecord
    return flat


def _counts(snapshot: dict[str, Any]) -> tuple[int, int, int]:
    """``(surfaces, distinct members, members that are also surfaces)``.

    Reported by both ``--update`` and the passing path, from one place, so
    the two can no longer print different totals for the same snapshot.
    """
    surfaces = snapshot.get("surfaces", {})
    n_surfaces = len(surfaces)
    n_flat = len(_flatten(snapshot))
    n_recorded = sum(len(r.get("members", {})) for r in surfaces.values())
    n_members = n_flat - n_surfaces
    return n_surfaces, n_members, n_recorded - n_members


def _count_line(snapshot: dict[str, Any]) -> str:
    n_surfaces, n_members, n_dual = _counts(snapshot)
    line = f"{n_surfaces} STABLE surface(s), {n_members} member(s)"
    if n_dual:
        line += f" ({n_dual} of them tagged in their own right)"
    return line


def _render(record: dict[str, Any]) -> str:
    """One-line rendering of a record, for the failure message."""
    parts = []
    for p in record.get("parameters", []):
        text = p["name"]
        if p["kind"] == "VAR_POSITIONAL":
            text = "*" + text
        elif p["kind"] == "VAR_KEYWORD":
            text = "**" + text
        if "annotation" in p:
            text += f": {p['annotation']}"
        if "default" in p:
            text += f" = {p['default']}"
        parts.append(text)
    rendered = f"({', '.join(parts)})"
    if "returns" in record:
        rendered += f" -> {record['returns']}"
    return rendered


def compare(recorded: dict[str, Any], current: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Compare two snapshots.

    Returns
    -------
    tuple of (list of str, list of str)
        ``(breaking, additions)`` — human-readable lines.  A *breaking* line
        is a signature that changed or a surface that vanished; an *addition*
        is a surface or member the snapshot does not record yet.
    """
    old, new = _flatten(recorded), _flatten(current)
    breaking, additions = [], []
    for name in sorted(set(old) | set(new)):
        if name not in new:
            breaking.append(f"  {name}: REMOVED (was {_render(old[name])})")
        elif name not in old:
            additions.append(f"  {name}: new {_render(new[name])}")
        elif old[name] != new[name]:
            breaking.append(
                f"  {name}: CHANGED\n"
                f"      recorded: {_render(old[name])}\n"
                f"      current : {_render(new[name])}"
            )
    return breaking, additions


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT,
                    help="path to the committed snapshot (default: %(default)s)")
    ap.add_argument("--update", action="store_true",
                    help="accept the current tree and rewrite the snapshot "
                         "(a STABLE signature change also needs a major "
                         "version bump and a CHANGELOG entry)")
    args = ap.parse_args(argv)

    registry, skipped = load_registry()
    try:
        current = build_snapshot(registry)
    except (ValueError, LookupError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.update:
        write_snapshot(current, args.snapshot)
        print(f"Wrote {args.snapshot}: {_count_line(current)}")
        return 0

    if not args.snapshot.exists():
        print(f"FAIL: no snapshot at {args.snapshot}.\n"
              "Fix: python scripts/check_stable_signatures.py --update",
              file=sys.stderr)
        return 1

    recorded = json.loads(args.snapshot.read_text())
    if recorded.get("format") != SNAPSHOT_FORMAT:
        print(f"FAIL: {args.snapshot} is format {recorded.get('format')!r}, "
              f"this check speaks format {SNAPSHOT_FORMAT}.\n"
              "Fix: python scripts/check_stable_signatures.py --update",
              file=sys.stderr)
        return 1

    # A module that failed to import for a missing optional dependency drops
    # its surfaces from the registry, which would read as a removal.  Only
    # fail hard when such a module actually carries a recorded surface.
    if skipped:
        affected = sorted(
            name for name, record in recorded.get("surfaces", {}).items()
            if any(record.get("module", "") == m
                   or record.get("module", "").startswith(m + ".")
                   for m in skipped)
        )
        for module, why in skipped.items():
            print(f"note: {module} not imported: {why}", file=sys.stderr)
        if affected:
            print("ERROR: these STABLE surfaces live in a module that could not "
                  "be imported, so the check cannot be trusted; install the "
                  f"optional extras and re-run:\n  " + "\n  ".join(affected),
                  file=sys.stderr)
            return 2

    breaking, additions = compare(recorded, current)
    if breaking:
        print(f"FAIL: {len(breaking)} STABLE signature change(s):")
        for line in breaking:
            print(line)
        print(
            "\nA STABLE surface's signature is frozen until the next major "
            "version (docs/developer_guide/deprecation_policy.md).  Either:\n"
            "  - revert the change, or keep it additive (a new keyword-only "
            "parameter with a default is compatible; renaming, reordering or "
            "removing one is not); or\n"
            "  - land it behind a major version bump, then accept it with\n"
            "      python scripts/check_stable_signatures.py --update\n"
            "    and record the break under '### Removed' in CHANGELOG.md."
        )
        if additions:
            print(f"\nAlso {len(additions)} unrecorded addition(s):")
            for line in additions:
                print(line)
        return 1

    if additions:
        print(f"FAIL: {len(additions)} STABLE surface(s) not in the snapshot:")
        for line in additions:
            print(line)
        print(
            "\nAdding a STABLE surface is not a break, but it must be recorded "
            "so the next change to it is caught:\n"
            "      python scripts/check_stable_signatures.py --update"
        )
        return 1

    print(f"OK: {_count_line(current)} unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
