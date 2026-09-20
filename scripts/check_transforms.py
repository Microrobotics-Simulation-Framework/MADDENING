#!/usr/bin/env python
"""Validate that every string edge-transform reference resolves.

Scans Python files under ``src/maddening/`` and ``tests/`` for calls to
``add_edge(..., transform="name")`` and ``EdgeSpec(..., transform="name")``,
and verifies that each string reference resolves to a registered
transform.

A name counts as registered if it is in the global ``TransformRegistry``
at import time *or* if the same file registers it itself with
``@register_transform("name")`` -- test modules and examples legitimately
register their own transforms at import, and those are as real as the
built-ins.

This is a CI gate for USD serialization compatibility: a transform
referenced by string name must be importable and registered, because a
stage records the *name* and resolves it on load.  ``PLAN_accuracy_and_usd.md``
names the USD serializer as the reason this gate exists, so the scan has to
cover the whole package, not the three subpackages it started with.

The gate fails if it finds nothing to check: a scope that has silently
narrowed to zero references reports ``OK`` forever, which is worse than no
gate at all because it is cited as delivered coverage.

Usage:
    python scripts/check_transforms.py [ROOT ...]

Exit codes:
    0 -- all referenced transforms are valid
    1 -- at least one unresolvable transform found, or nothing was checked
"""

import ast
import importlib
import importlib.util
import sys
from pathlib import Path


_EDGE_CALLS = {"add_edge", "EdgeSpec"}

#: ``transform`` is the 5th positional parameter of
#: ``GraphManager.add_edge(source, target, source_field, target_field,
#: transform, ...)`` and the 5th field of ``EdgeSpec``, so a string there is
#: resolved at runtime exactly as ``transform="name"`` is.  The gate read
#: ``node.keywords`` only, so the positional form was invisible
#: (audit_040_r2/gates, finding G2).
#: ``tests/compliance/test_gate_scripts.py::TestTransformPositionalForm``
#: pins this index against both real signatures.
_TRANSFORM_POSITION = 4

# Default scan roots, relative to the project root.  ``tests`` is in scope
# deliberately: every string-literal edge transform in the repository lives
# there, so excluding it left the gate with nothing to verify.
_DEFAULT_ROOTS = ("src/maddening", "tests")

# Deliberate negative tests: a call site that names a transform which must
# *not* resolve, because the test asserts that the lookup raises.  Keyed by
# (path relative to the project root, transform name) so that a typo
# anywhere else in the same file is still caught, with a one-line reason so
# a deliberate exemption stays distinguishable from an accumulated one.
#
# Add an entry only for a test whose *subject* is the failure -- never to
# quiet a name that should have been registered.  `tests/compliance/
# test_gate_scripts.py::TestTransformAllowlist` caps the size, requires the
# reason, and fails an entry whose file no longer names that transform.
_ALLOWED_UNRESOLVABLE = {
    ("tests/core/test_transforms.py", "this_does_not_exist"):
        "asserts GraphManager.add_edge raises KeyError on an unknown name",
    ("tests/core/test_graph_mutation_atomicity.py", "no_such_transform"):
        "asserts add_edge fails atomically on an unknown name; the entry is "
        "inert until fix/adaptive-contract lands the file",
}
# A ceiling, not a target.  Two entries is the natural number of "prove the
# lookup raises" tests; a third is plausible, a tenth means the gate is
# being worked around rather than the code fixed.
_MAX_ALLOWED_UNRESOLVABLE = 5


def _call_name(func: ast.expr) -> str | None:
    """Return the trailing identifier of a call target, if it has one."""
    return getattr(func, "attr", None) or getattr(func, "id", None)


def _module_level_string_constants(tree: ast.AST) -> dict[str, str]:
    """Module-level ``NAME = "literal"`` bindings.

    ``transform=EXTRACT_LAST`` is as much a string reference as
    ``transform="extract_last"``; binding the name to a constant first used
    to hide it from the scan.
    """
    constants: dict[str, str] = {}
    body = getattr(tree, "body", [])
    for stmt in body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not (isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                constants[target.id] = stmt.value.value
    return constants


def find_transform_string_refs(tree: ast.AST) -> list[tuple[int, str]]:
    """Find string references used as ``transform=`` arguments on edge calls.

    Returns a list of ``(line_number, string_value)`` pairs.  Only edge
    constructors are considered: ``ParamSpec(transform="log")`` is a
    parameter reparametrisation, not an edge transform.
    """
    constants = _module_level_string_constants(tree)
    results = []

    def record(value: ast.expr) -> None:
        if isinstance(value, ast.Constant):
            if isinstance(value.value, str):
                results.append((value.lineno, value.value))
        elif isinstance(value, ast.Name) and value.id in constants:
            results.append((value.lineno, constants[value.id]))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) not in _EDGE_CALLS:
            continue
        keyworded = False
        for kw in node.keywords:
            if kw.arg != "transform":
                continue
            keyworded = True
            record(kw.value)
        # The positional form means the same thing and resolves the same way.
        if not keyworded and len(node.args) > _TRANSFORM_POSITION:
            record(node.args[_TRANSFORM_POSITION])
    return results


def find_local_registrations(tree: ast.AST) -> set[str]:
    """Find names this module registers itself via ``register_transform``.

    Covers both the decorator form ``@register_transform("name")`` and the
    direct call ``register_transform("name")(fn)``.
    """
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) != "register_transform":
            continue
        if node.args and isinstance(node.args[0], ast.Constant):
            if isinstance(node.args[0].value, str):
                names.add(node.args[0].value)
    return names


def scan_file(filepath: Path) -> tuple[list[tuple[int, str]], set[str]]:
    """Return ``(string refs, locally registered names)`` for one file."""
    try:
        tree = ast.parse(filepath.read_text(), filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return [], set()
    return find_transform_string_refs(tree), find_local_registrations(tree)


def registry_after_importing(
    filepath: Path, project_root: Path
) -> tuple[set[str] | None, str | None]:
    """Import ``filepath`` and return the live registry's names.

    ``find_local_registrations`` is a *lexical* check: it finds the
    ``register_transform("name")`` call expression anywhere in the file,
    including inside a function nobody calls.  Such a registration never
    executes, so the name is absent from the registry after import and
    ``add_edge`` raises ``KeyError`` at runtime -- while the gate said the
    reference was verified (audit_040_r2/gates, finding G2b).  The realistic
    shape is a helper a fixture forgot to call, or one behind a
    ``try/except ImportError`` fallback.

    Returns ``(names, None)`` on success and ``(None, reason)`` when the
    module cannot be imported.  A module needing an optional extra this
    environment does not have is *unchecked*, not broken -- the same
    degradation ``resolve_dotted_name``'s ``unavailable`` handling makes,
    and for the same reason: this gate must stay usable in a CI that
    installs only ``[ci]``.
    """
    from maddening.core.transforms import _TRANSFORM_REGISTRY

    try:
        rel = filepath.resolve().relative_to(project_root.resolve())
    except ValueError:
        rel = None

    try:
        if rel is not None:
            parts = list(rel.with_suffix("").parts)
            if parts and parts[0] == "src":
                parts = parts[1:]
            root = str(project_root)
            if root not in sys.path:
                sys.path.insert(0, root)
            importlib.import_module(".".join(parts))
        else:
            # A scan root outside the repository (a temporary directory in
            # the gate's own tests).  Load it by path, without giving it a
            # place in sys.modules it could collide in.
            spec = importlib.util.spec_from_file_location(
                f"_check_transforms_probe_{filepath.stem}", filepath
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - any import failure degrades
        return None, f"{type(exc).__name__}: {exc}"

    return set(_TRANSFORM_REGISTRY), None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    project_root = Path(__file__).parent.parent

    sys.path.insert(0, str(project_root / "src"))
    from maddening.core.transforms import _TRANSFORM_REGISTRY

    roots = [Path(a) for a in argv] or [project_root / r for r in _DEFAULT_ROOTS]

    # Snapshot the registry before anything is imported.  The live check
    # below imports modules that register transforms, and those
    # registrations are global: without a snapshot, a name registered by one
    # test module would start satisfying a reference in another, which is
    # exactly what "registered in another file does not count" forbids.
    builtin_names = set(_TRANSFORM_REGISTRY)

    errors = []
    notes = []
    n_checked = 0
    n_allowlisted = 0
    n_unconfirmed = 0
    scanned_roots = []
    # (file, [(lineno, name)]) for references resolved only by a lexical
    # local registration -- the ones the live check has to confirm.
    credited: list[tuple[Path, str, list[tuple[int, str]]]] = []

    for root in roots:
        if not root.exists():
            continue
        scanned_roots.append(str(root))
        for pyfile in sorted(root.rglob("*.py")):
            refs, local = scan_file(pyfile)
            if not refs:
                continue
            try:
                rel = str(pyfile.relative_to(project_root))
            except ValueError:
                rel = str(pyfile)
            local_refs: list[tuple[int, str]] = []
            for lineno, name in refs:
                if (rel, name) in _ALLOWED_UNRESOLVABLE:
                    n_allowlisted += 1
                    continue
                n_checked += 1
                if name in builtin_names:
                    continue
                if name in local:
                    local_refs.append((lineno, name))
                    continue
                errors.append(
                    f"  {rel}:{lineno}: transform '{name}' is neither in "
                    f"the TransformRegistry nor registered in this file"
                )
            if local_refs:
                credited.append((pyfile, rel, local_refs))

    # A lexical @register_transform is a claim, not a registration.  Import
    # the module and check the registry actually gained the name.
    for pyfile, rel, local_refs in credited:
        live, reason = registry_after_importing(pyfile, project_root)
        if live is None:
            n_unconfirmed += len(local_refs)
            names = ", ".join(sorted({n for _lineno, n in local_refs}))
            notes.append(
                f"{rel}: {len(local_refs)} reference(s) credited to a local "
                f"@register_transform ({names}) were NOT confirmed against "
                f"the live registry -- the module could not be imported "
                f"here ({reason})"
            )
            continue
        for lineno, name in local_refs:
            if name not in live:
                errors.append(
                    f"  {rel}:{lineno}: transform '{name}' is registered by "
                    f"a @register_transform in this file, but is absent from "
                    f"the TransformRegistry after the module is imported -- "
                    f"the registration never executes, and add_edge would "
                    f"raise KeyError on this name at runtime"
                )

    for note in notes:
        print(f"NOTE: {note}")

    if errors:
        print(f"FAIL: {len(errors)} unresolvable transform reference(s):")
        for err in errors:
            print(err)
        print(
            "\nFix: register each transform with "
            "@register_transform('name') from "
            "maddening.core.transforms, at module level so the registration "
            "runs on import"
        )
        return 1

    if n_checked == 0:
        print(
            f"FAIL: 0 string transform reference(s) found in {scanned_roots}.\n"
            "A gate that verifies nothing cannot fail.  Either the scan roots "
            "are wrong or every reference is allowlisted; fix the scope rather "
            "than trusting the OK.",
            file=sys.stderr,
        )
        return 1

    # Verified, declined and unconfirmed are three numbers, not one.  A
    # single headline that folds in what the gate skipped is how "50
    # citations verified" came to mean 45.
    extra = ""
    if n_allowlisted:
        extra += f", {n_allowlisted} allowlisted and not checked"
    if n_unconfirmed:
        extra += f", {n_unconfirmed} not confirmed against the live registry"
    print(
        f"OK: {n_checked} string transform reference(s) verified{extra} "
        f"({len(builtin_names)} transforms in registry)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
