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

**A removal needs more than ``--update``.**  A surface or member that leaves
the ``STABLE`` set -- deleted, renamed, made private, or demoted to a lower
level -- withdraws a promise callers already rely on, which the deprecation
policy treats as a breaking change in its own right.  ``--update`` refuses to
drop one from the snapshot unless ``--accept-removal`` is also given, and
lists what it would drop either way, so the withdrawal is a decision somebody
typed rather than a side effect of regenerating a file.  Additions and
compatible widenings still need only ``--update``.

**An empty ``STABLE`` set fails**, in both modes.  With every tag demoted and
the snapshot regenerated, this script used to print ``OK: 0 STABLE
surface(s), 0 member(s) unchanged`` and exit 0 -- a guard with nothing left
to guard, reporting success (audit_040_phase3_wave_d, G9).  ``--update``
will not write an empty snapshot either.

Exit codes:
    0 -- every recorded signature still matches
    1 -- at least one signature changed, was removed, or is unrecorded; the
         tree or the snapshot has no STABLE surface at all; or ``--update``
         would drop a surface without ``--accept-removal``
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
    number of such surfaces: 4 of the 243 recorded member records when this
    was found (2026-09-20).  On 2026-09-25 it is still 4, of 252 records,
    and the passing line reads "17 STABLE surface(s), 248 member(s) (4 of
    them tagged in their own right)".  That line (:func:`_count_line`) is
    the live count; a figure written here goes stale.

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


#: Parameter kinds an existing caller supplies positionally.
_POSITIONAL = ("POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD")


def is_additive(recorded: dict[str, Any], current: dict[str, Any]) -> bool:
    """Is the change from *recorded* to *current* one no caller can observe?

    The failure message has always claimed that "a new keyword-only parameter
    with a default is compatible; renaming, reordering or removing one is
    not", and the comparison did not make that distinction: any difference at
    all was reported as a break.  A guard that cries breach over the three
    additive `params=None` parameters this release added is a guard somebody
    switches off, so the classification has to match the promise.

    A change is additive when **every recorded parameter survives byte for
    byte, in the same relative order, with its positional index intact**, the
    return annotation and the kind are unchanged, and **every new parameter is
    one an existing call can omit** — it has a default, or it is ``*args`` /
    ``**kwargs``.  Anything else is breaking, including a widened annotation:
    the annotation is part of what a type checker holds callers to, and
    deciding which widenings are safe is not something this can do from text.
    """
    if recorded.get("kind") != current.get("kind"):
        return False
    if recorded.get("returns") != current.get("returns"):
        return False
    if recorded.get("settable") != current.get("settable"):
        return False

    old_params = recorded.get("parameters", [])
    new_params = current.get("parameters", [])
    new_by_name = {p["name"]: p for p in new_params}

    # every recorded parameter is still there, unchanged
    for parameter in old_params:
        if new_by_name.get(parameter["name"]) != parameter:
            return False

    # ... in the same relative order
    recorded_names = [p["name"] for p in old_params]
    kept = [p["name"] for p in new_params if p["name"] in set(recorded_names)]
    if kept != recorded_names:
        return False

    # ... and a positional parameter keeps its index, so nothing is inserted
    # in front of one an existing caller passes by position
    old_positional = [p["name"] for p in old_params if p["kind"] in _POSITIONAL]
    new_positional = [p["name"] for p in new_params if p["kind"] in _POSITIONAL]
    if new_positional[:len(old_positional)] != old_positional:
        return False

    # every new parameter is one an existing call can leave out
    for parameter in new_params:
        if parameter["name"] in set(recorded_names):
            continue
        if parameter["kind"] in ("VAR_POSITIONAL", "VAR_KEYWORD"):
            continue
        if "default" not in parameter:
            return False
    return True


def compare(
    recorded: dict[str, Any], current: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    """Compare two snapshots.

    Returns
    -------
    tuple of (list of str, list of str, list of str)
        ``(breaking, compatible, additions)`` — human-readable lines.

        A *breaking* line is a signature that changed in a way a caller can
        observe, or a surface that vanished.  A *compatible* line is a change
        :func:`is_additive` accepts: the snapshot is out of date, the contract
        is not.  An *addition* is a surface or member the snapshot does not
        record yet.

        All three still fail the check, because the snapshot has to move or
        the next change is measured against a stale baseline.  They fail with
        different messages, and the caller must read the classification rather
        than the exit code: two different defects with the same ``rc`` make a
        check nothing can be asserted about.
    """
    old, new = _flatten(recorded), _flatten(current)
    breaking, compatible, additions = [], [], []
    for name in sorted(set(old) | set(new)):
        if name not in new:
            breaking.append(f"  {name}: REMOVED (was {_render(old[name])})")
        elif name not in old:
            additions.append(f"  {name}: new {_render(new[name])}")
        elif old[name] != new[name]:
            line = (
                f"  {name}: {{verdict}}\n"
                f"      recorded: {_render(old[name])}\n"
                f"      current : {_render(new[name])}"
            )
            if is_additive(old[name], new[name]):
                compatible.append(line.format(verdict="WIDENED"))
            else:
                breaking.append(line.format(verdict="CHANGED"))
    return breaking, compatible, additions


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
    ap.add_argument("--accept-removal", action="store_true",
                    help="with --update: also drop surfaces and members that "
                         "have left the STABLE set.  A removal is a breaking "
                         "change under docs/developer_guide/"
                         "deprecation_policy.md; record it under "
                         "'### Removed' in CHANGELOG.md")
    args = ap.parse_args(argv)
    if args.accept_removal and not args.update:
        ap.error("--accept-removal only means something with --update")

    registry, skipped = load_registry()
    try:
        current = build_snapshot(registry)
    except (ValueError, LookupError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # The floor.  A tree with no STABLE surface has nothing for this guard to
    # guard; "OK: 0 STABLE surface(s)" is an empty scope reporting success.
    if not current["surfaces"]:
        print("FAIL: the tree has no @stability(StabilityLevel.STABLE) "
              "surface at all, so there is nothing to check.  Every STABLE "
              "tag has been demoted or the stability report's module list "
              "has lost them; either is a finding, not a pass.",
              file=sys.stderr)
        return 1

    recorded: dict[str, Any] | None = None
    if args.snapshot.exists():
        recorded = json.loads(args.snapshot.read_text())
        if recorded.get("format") != SNAPSHOT_FORMAT:
            if not args.update:
                print(f"FAIL: {args.snapshot} is format "
                      f"{recorded.get('format')!r}, this check speaks format "
                      f"{SNAPSHOT_FORMAT}.\n"
                      "Fix: python scripts/check_stable_signatures.py --update",
                      file=sys.stderr)
                return 1
            recorded = None                  # a format bump regenerates
    elif not args.update:
        print(f"FAIL: no snapshot at {args.snapshot}.\n"
              "Fix: python scripts/check_stable_signatures.py --update",
              file=sys.stderr)
        return 1

    # A module that failed to import for a missing optional dependency drops
    # its surfaces from the registry, which would read as a removal.  Only
    # fail hard when such a module actually carries a recorded surface --
    # and in --update too, which would otherwise quietly write the partial
    # tree over the full snapshot.
    if skipped and recorded is not None:
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

    if args.update:
        removed = []
        if recorded is not None:
            removed = sorted(set(_flatten(recorded)) - set(_flatten(current)))
        if removed:
            verb = "Dropping" if args.accept_removal else "REFUSED: would drop"
            print(f"{verb} {len(removed)} surface(s)/member(s) that have left "
                  f"the STABLE set:")
            for name in removed:
                print(f"  {name}")
        if removed and not args.accept_removal:
            print(
                "\nRemoving a surface from the STABLE set -- deleting, "
                "renaming, making private or demoting it -- is a breaking "
                "change (docs/developer_guide/deprecation_policy.md), not a "
                "snapshot refresh.  If it is intended and ships in a major "
                "release after its deprecation period, re-run with\n"
                "      python scripts/check_stable_signatures.py --update "
                "--accept-removal\n"
                "and record it under '### Removed' in CHANGELOG.md.",
            )
            return 1
        write_snapshot(current, args.snapshot)
        print(f"Wrote {args.snapshot}: {_count_line(current)}")
        return 0

    assert recorded is not None
    if not recorded.get("surfaces"):
        # The tree has surfaces (checked above), so compare() would report
        # every one as an addition and fail anyway -- but say what is
        # actually wrong: the baseline itself is empty.
        print(f"FAIL: {args.snapshot} records no STABLE surface; an empty "
              f"baseline guards nothing.\n"
              "Fix: python scripts/check_stable_signatures.py --update",
              file=sys.stderr)
        return 1

    breaking, compatible, additions = compare(recorded, current)

    def _also(label: str, lines: list[str]) -> None:
        if lines:
            print(f"\nAlso {len(lines)} {label}:")
            for line in lines:
                print(line)

    if breaking:
        print(f"FAIL: {len(breaking)} BREAKING STABLE signature change(s):")
        for line in breaking:
            print(line)
        print(
            "\nA STABLE surface's signature is frozen until the next major "
            "version (docs/developer_guide/deprecation_policy.md).  Either:\n"
            "  - revert the change, or keep it additive (a new parameter with "
            "a default, added after the existing ones, is compatible; "
            "renaming, reordering or removing one is not); or\n"
            "  - land it behind a major version bump, then accept it with\n"
            "      python scripts/check_stable_signatures.py --update\n"
            "    and record the break under '### Removed' in CHANGELOG.md."
        )
        _also("compatible change(s)", compatible)
        _also("unrecorded addition(s)", additions)
        return 1

    if compatible:
        print(f"FAIL: {len(compatible)} COMPATIBLE STABLE signature change(s):")
        for line in compatible:
            print(line)
        print(
            "\nEvery recorded parameter survives unchanged and each new one "
            "has a default, so no existing call is affected: this is the "
            "snapshot being out of date, NOT a break of the contract.  "
            "Record it:\n"
            "      python scripts/check_stable_signatures.py --update"
        )
        _also("unrecorded addition(s)", additions)
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
