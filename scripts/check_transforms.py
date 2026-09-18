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
import sys
from pathlib import Path


_EDGE_CALLS = {"add_edge", "EdgeSpec"}

# Default scan roots, relative to the project root.  ``tests`` is in scope
# deliberately: every string-literal edge transform in the repository lives
# there, so excluding it left the gate with nothing to verify.
_DEFAULT_ROOTS = ("src/maddening", "tests")

# Deliberate negative tests: a call site that names a transform which must
# *not* resolve, because the test asserts that the lookup raises.  Keyed by
# (path relative to the project root, transform name) so that a typo
# anywhere else in the same file is still caught.
_ALLOWED_UNRESOLVABLE = {
    # Asserts that GraphManager.add_edge raises KeyError for an unknown
    # transform name -- the name is required to be absent from the registry.
    ("tests/core/test_transforms.py", "this_does_not_exist"),
}


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
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _call_name(node.func) not in _EDGE_CALLS:
            continue
        for kw in node.keywords:
            if kw.arg != "transform":
                continue
            if isinstance(kw.value, ast.Constant):
                if isinstance(kw.value.value, str):
                    results.append((kw.value.lineno, kw.value.value))
            elif isinstance(kw.value, ast.Name) and kw.value.id in constants:
                results.append((kw.value.lineno, constants[kw.value.id]))
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


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    project_root = Path(__file__).parent.parent

    sys.path.insert(0, str(project_root / "src"))
    from maddening.core.transforms import _TRANSFORM_REGISTRY

    roots = [Path(a) for a in argv] or [project_root / r for r in _DEFAULT_ROOTS]

    errors = []
    n_checked = 0
    scanned_roots = []

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
            for lineno, name in refs:
                if (rel, name) in _ALLOWED_UNRESOLVABLE:
                    continue
                n_checked += 1
                if name not in _TRANSFORM_REGISTRY and name not in local:
                    errors.append(
                        f"  {rel}:{lineno}: transform '{name}' is neither in "
                        f"the TransformRegistry nor registered in this file"
                    )

    if errors:
        print(f"FAIL: {len(errors)} unresolvable transform reference(s):")
        for err in errors:
            print(err)
        print(
            "\nFix: register each transform with "
            "@register_transform('name') from "
            "maddening.core.transforms"
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

    print(
        f"OK: {n_checked} string transform reference(s) verified "
        f"({len(_TRANSFORM_REGISTRY)} transforms in registry)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
