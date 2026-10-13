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

The gate fails if it *verified* nothing.  A scope that has silently
narrowed to zero references reports ``OK`` forever, which is worse than no
gate at all because it is cited as delivered coverage -- and a scope whose
every reference is allowlisted or unconfirmed has narrowed to zero just as
surely as an empty one.  The floor is on verified references, not on
references found.

A reference credited to a local ``@register_transform`` is confirmed by
importing the module and reading the live registry.  When that import
fails, the reason decides the outcome:

* a **missing third-party package** (an optional extra such as ``usd-core``)
  leaves the reference *unconfirmed*.  Unconfirmed is not verified, and by
  default it fails the gate with exit 2, "could not be trusted": the CI job
  that runs this gate installs the extras precisely so that nothing is
  unconfirmed, so an unconfirmed reference there means the gate has quietly
  started verifying less.  ``--allow-missing-optional`` accepts them -- for a
  contributor or a test lane without the extras -- and still reports each
  one, and still fails if nothing at all was verified.  This is the contract
  ``scripts/generate_stability_report.py --allow-missing-optional`` and
  ``scripts/check_stable_signatures.py``'s exit 2 already have;
* **anything else** -- the module raises, names a first-party module that
  does not exist, or a registration collides -- is a broken module, and a
  hard failure: its registrations do not run anywhere.

What the scan sees, and what it does not
----------------------------------------
Seen: a string literal passed as ``transform=`` or as the fifth positional
argument, a name bound to a string literal at module level or in an
enclosing function (``name = "x"``, ``for name in ("x", "y")``), and a
``transform`` key in a literal ``**{...}`` or ``**dict(...)`` splat.

Not seen, and therefore not verified -- review has to catch these:
a name that arrives as a function parameter (including a
``pytest.mark.parametrize`` value), an attribute (``cfg.transform``), a
computed string (an f-string, a concatenation, a call), a ``**kwargs``
mapping held in a variable, and a ``*args`` positional splat.  The gate
cannot tell a string from a callable in those positions, and a callable is
the ordinary, unregistered-by-design case.

Usage:
    python scripts/check_transforms.py [--allow-missing-optional] [ROOT ...]

Exit codes:
    0 -- every reference in scope was verified (or, with
         ``--allow-missing-optional``, verified or reported as unconfirmed)
    1 -- an unresolvable transform, a module that fails to import for a
         reason other than a missing optional package, a scan root that does
         not exist, or nothing verified
    2 -- the check could not be trusted: a reference could not be confirmed
         because an optional package is missing, and
         ``--allow-missing-optional`` was not given
"""

import argparse
import ast
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import NamedTuple


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


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _string_literals(value: ast.expr) -> list[str] | None:
    """The string(s) ``value`` is literally, or ``None`` if it is not one.

    A string constant is one string; a literal tuple or list of string
    constants is each of them (the iterable of a ``for`` loop).
    """
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return [value.value]
    if isinstance(value, (ast.Tuple, ast.List)) and value.elts:
        if all(isinstance(e, ast.Constant) and isinstance(e.value, str)
               for e in value.elts):
            return [e.value for e in value.elts]
    return None


def _scope_string_bindings(scope: ast.AST) -> dict[str, list[str]]:
    """``NAME -> [literal, ...]`` for names one scope binds to string literals.

    ``transform=EXTRACT_LAST`` is as much a string reference as
    ``transform="extract_last"``, and so is ``name = "extract_last"`` inside
    the test function that then passes ``transform=name`` -- binding the
    name first used to hide it from the scan, at module level first and in
    a function body until audit_040_phase3_wave_d (T6).

    Covers ``NAME = "x"``, ``NAME: str = "x"`` and ``for NAME in ("x",
    "y")`` anywhere in the scope's own body, including inside ``if`` /
    ``with`` / ``try`` blocks, but not inside a nested function or class --
    those are scopes of their own.  Every literal a name is bound to is
    kept, because any of them can reach the call.

    A name the scope binds to something *else* -- a parameter, a call
    result -- maps to an empty list.  It is still local, so it shadows an
    outer string constant of the same name instead of being mistaken for it.
    """
    bindings: dict[str, list[str]] = {}

    def bind(target: ast.expr, literals: list[str] | None) -> None:
        if isinstance(target, ast.Name):
            bindings.setdefault(target.id, []).extend(literals or [])
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                bind(element, None)

    def single(value: ast.expr) -> list[str] | None:
        literals = _string_literals(value)
        return literals if literals is not None and len(literals) == 1 else None

    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = scope.args
        for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs,
                    *(x for x in (a.vararg, a.kwarg) if x is not None)):
            bindings.setdefault(arg.arg, [])

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _SCOPES):
                continue
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    bind(target, single(child.value))
            elif isinstance(child, ast.AnnAssign) and child.value is not None:
                bind(child.target, single(child.value))
            elif isinstance(child, (ast.For, ast.AsyncFor)):
                bind(child.target, _string_literals(child.iter))
            visit(child)

    visit(scope)
    return bindings


def find_transform_string_refs(tree: ast.AST) -> list[tuple[int, str]]:
    """Find string references used as ``transform=`` arguments on edge calls.

    Returns a list of ``(line_number, string_value)`` pairs.  Only edge
    constructors are considered: ``ParamSpec(transform="log")`` is a
    parameter reparametrisation, not an edge transform.

    A bare name is looked up in the innermost enclosing scope that binds it
    to a string literal, then outwards to module level -- see
    :func:`_scope_string_bindings`.  The module docstring lists what the
    scan cannot see.
    """
    results: list[tuple[int, str]] = []

    def record(value: ast.expr, stack: list[dict[str, list[str]]]) -> None:
        if isinstance(value, ast.Constant):
            if isinstance(value.value, str):
                results.append((value.lineno, value.value))
            return
        if isinstance(value, ast.Name):
            for bindings in reversed(stack):
                if value.id in bindings:
                    for literal in bindings[value.id]:
                        results.append((value.lineno, literal))
                    return

    def splatted(kw: ast.keyword) -> list[ast.expr]:
        """``transform`` values inside a literal ``**{...}`` / ``**dict(...)``.

        ``add_edge(..., **{"transform": "x"})`` resolves ``"x"`` exactly as
        the keyword does, and was invisible (audit_040_phase3_wave_d, T7).
        """
        value = kw.value
        if isinstance(value, ast.Dict):
            return [v for k, v in zip(value.keys, value.values)
                    if isinstance(k, ast.Constant) and k.value == "transform"]
        if (isinstance(value, ast.Call) and _call_name(value.func) == "dict"
                and not value.args):
            return [k.value for k in value.keywords if k.arg == "transform"]
        return []

    def visit(node: ast.AST, stack: list[dict[str, list[str]]]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.Lambda)):
            stack = [*stack, _scope_string_bindings(node)]
        if (isinstance(node, ast.Call)
                and _call_name(node.func) in _EDGE_CALLS):
            keyworded = False
            for kw in node.keywords:
                if kw.arg == "transform":
                    keyworded = True
                    record(kw.value, stack)
                elif kw.arg is None:
                    for value in splatted(kw):
                        keyworded = True
                        record(value, stack)
            # The positional form means the same thing and resolves the
            # same way.
            if not keyworded and len(node.args) > _TRANSFORM_POSITION:
                record(node.args[_TRANSFORM_POSITION], stack)
        for child in ast.iter_child_nodes(node):
            visit(child, stack)

    # A class body is not an enclosing scope for the methods inside it, so
    # only functions push bindings; the module is the outermost scope.
    visit(tree, [_scope_string_bindings(tree)])
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


class LiveRegistry(NamedTuple):
    """What importing one module told us about the live registry.

    Exactly one of the three fields is set.
    """

    #: The registry's names after the import succeeded.
    names: set[str] | None = None
    #: Set when the import failed for a missing *third-party* package: the
    #: module's registrations could not be confirmed in this environment.
    unconfirmed: str | None = None
    #: Set when the import failed for any other reason: the module is broken
    #: and its registrations run nowhere.
    broken: str | None = None


def _first_party_roots(project_root: Path) -> set[str]:
    """Top-level import names that belong to this repository."""
    roots = {"maddening"}
    for base in (project_root, project_root / "src"):
        if base.is_dir():
            roots.update(
                p.name for p in base.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
            roots.update(p.stem for p in base.glob("*.py"))
    return roots


def missing_optional_package(
    exc: BaseException, project_root: Path
) -> str | None:
    """The third-party package whose absence caused ``exc``, if that is what did.

    Walks the exception chain, because a subpackage that refuses to import
    without its extra re-raises: ``maddening.usd`` turns
    ``ModuleNotFoundError('pxr')`` into an ``ImportError`` naming the extra,
    with the original as ``__cause__``.  A ``ModuleNotFoundError`` naming a
    *first-party* module (``maddening.<typo>``, ``tests.<gone>``) is not a
    missing extra -- it is a broken import, and stays one.
    """
    first_party = _first_party_roots(project_root)
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModuleNotFoundError) and current.name:
            top = current.name.split(".")[0]
            if top not in first_party:
                return top
        current = current.__cause__ or current.__context__
    return None


def registry_after_importing(filepath: Path, project_root: Path) -> LiveRegistry:
    """Import ``filepath`` and report the live registry's names.

    ``find_local_registrations`` is a *lexical* check: it finds the
    ``register_transform("name")`` call expression anywhere in the file,
    including inside a function nobody calls.  Such a registration never
    executes, so the name is absent from the registry after import and
    ``add_edge`` raises ``KeyError`` at runtime -- while the gate said the
    reference was verified (audit_040_r2/gates, finding G2b).  The realistic
    shape is a helper a fixture forgot to call, or one behind a
    ``try/except ImportError`` fallback.

    An import failure used to degrade to "unconfirmed" whatever raised it,
    so a module that raised at import -- whose registrations therefore run
    nowhere -- passed as merely unchecked.  Only a missing third-party
    package is an environment's limitation (:func:`missing_optional_package`);
    everything else is reported as ``broken``.  A module-level
    ``pytest.skip`` / ``importorskip`` is the module declaring that it
    cannot run here, and counts as unconfirmed.
    """
    from maddening.core.transforms import _TRANSFORM_REGISTRY

    try:
        rel = filepath.resolve().relative_to(project_root.resolve())
    except ValueError:
        rel = None

    snapshot = None
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
            # place in sys.modules it could collide in -- and put the
            # registry back afterwards: each load re-executes the module,
            # so a second load in the same process would re-register every
            # name to a new function object and ``register_transform`` would
            # raise, turning a correct probe into a "broken" one.
            snapshot = dict(_TRANSFORM_REGISTRY)
            spec = importlib.util.spec_from_file_location(
                f"_check_transforms_probe_{filepath.stem}", filepath
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: BLE001 - classified below
        reason = f"{type(exc).__name__}: {exc}"
        if snapshot is not None:
            _TRANSFORM_REGISTRY.clear()
            _TRANSFORM_REGISTRY.update(snapshot)
        missing = missing_optional_package(exc, project_root)
        if missing is not None:
            return LiveRegistry(
                unconfirmed=f"optional package {missing!r} is not installed "
                            f"({reason})"
            )
        if type(exc).__name__ == "Skipped":        # pytest.skip at import
            return LiveRegistry(unconfirmed=f"skipped at import ({reason})")
        return LiveRegistry(broken=reason)

    names = set(_TRANSFORM_REGISTRY)
    if snapshot is not None:
        _TRANSFORM_REGISTRY.clear()
        _TRANSFORM_REGISTRY.update(snapshot)
    return LiveRegistry(names=names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "roots", nargs="*", type=Path,
        help="directories to scan (default: src/maddening and tests)",
    )
    parser.add_argument(
        "--allow-missing-optional", action="store_true",
        help="accept references that cannot be confirmed because an "
             "optional package is missing (they are still reported, and a "
             "scope in which nothing was verified still fails)",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    project_root = Path(__file__).parent.parent

    sys.path.insert(0, str(project_root / "src"))
    from maddening.core.transforms import _TRANSFORM_REGISTRY

    roots = args.roots or [project_root / r for r in _DEFAULT_ROOTS]

    # A root that does not exist is a typo in the command line or a moved
    # directory, not an empty scope to be reported as "0 found in []".
    absent = [str(r) for r in roots if not r.exists()]
    if absent:
        print(f"FAIL: scan root(s) do not exist: {absent}", file=sys.stderr)
        return 1

    # Snapshot the registry before anything is imported.  The live check
    # below imports modules that register transforms, and those
    # registrations are global: without a snapshot, a name registered by one
    # test module would start satisfying a reference in another, which is
    # exactly what "registered in another file does not count" forbids.
    builtin_names = set(_TRANSFORM_REGISTRY)

    errors = []
    notes = []
    n_in_scope = 0
    n_allowlisted = 0
    n_unconfirmed = 0
    scanned_roots = [str(r) for r in roots]
    # (file, [(lineno, name)]) for references resolved only by a lexical
    # local registration -- the ones the live check has to confirm.
    credited: list[tuple[Path, str, list[tuple[int, str]]]] = []

    for root in roots:
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
                # References the gate takes responsibility for.  Not the
                # same number as the ones it ends up verifying: the
                # live-registry loop below can leave some unconfirmed,
                # and `n_verified` subtracts those back out.
                n_in_scope += 1
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
        live = registry_after_importing(pyfile, project_root)
        names = ", ".join(sorted({n for _lineno, n in local_refs}))
        if live.broken is not None:
            errors.append(
                f"  {rel}: {len(local_refs)} reference(s) credited to a local "
                f"@register_transform ({names}), but the module fails to "
                f"import -- {live.broken}.  That is not a missing optional "
                f"package, so the registration runs nowhere"
            )
            continue
        if live.unconfirmed is not None:
            n_unconfirmed += len(local_refs)
            notes.append(
                f"{rel}: {len(local_refs)} reference(s) credited to a local "
                f"@register_transform ({names}) were NOT confirmed against "
                f"the live registry -- {live.unconfirmed}"
            )
            continue
        for lineno, name in local_refs:
            if name not in live.names:
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

    # Verified, declined and unconfirmed are three numbers, not one.  A
    # single headline that folds in what the gate skipped is how "50
    # citations verified" came to mean 45.
    #
    # The unconfirmed are subtracted here rather than never counted,
    # because which references they are is only known after the
    # live-registry loop above.  The allowlisted ones `continue` before
    # they are ever counted, and the two used to be handled differently:
    # a scope with two references, one credited to a local
    # `@register_transform` in a module this environment cannot import,
    # printed "2 ... verified, 1 not confirmed" (audit_040_r3).  One was
    # verified, not two -- and the counter was introduced by 44250c3,
    # the fix for this very defect class.
    n_verified = n_in_scope - n_unconfirmed
    extra = ""
    if n_allowlisted:
        extra += f", {n_allowlisted} allowlisted and not checked"
    if n_unconfirmed:
        extra += f", {n_unconfirmed} not confirmed against the live registry"

    # The floor is on what was *verified*.  It used to be on what was in
    # scope, so a scope whose only reference could not be confirmed
    # printed "OK: 0 string transform reference(s) verified, 1 not
    # confirmed" and exited 0 (audit_040_phase3_wave_d, T5) -- the one
    # gate of seven without a verified-count floor.
    if n_verified == 0:
        print(
            f"FAIL: 0 string transform reference(s) verified in "
            f"{scanned_roots}{extra}.\n"
            "A gate that verifies nothing cannot fail.  Either the scan roots "
            "are wrong, or every reference is allowlisted or could not be "
            "confirmed here; fix the scope rather than trusting the OK.",
            file=sys.stderr,
        )
        return 1

    if n_unconfirmed and not args.allow_missing_optional:
        print(
            f"ERROR: {n_unconfirmed} string transform reference(s) could not "
            f"be confirmed against the live registry, because a module "
            f"needs an optional package this environment lacks (see the "
            f"NOTE lines above); {n_verified} were verified.  The check "
            f"cannot be trusted as a whole: install the extras (the CI "
            f"compliance job installs .[ci,usd]) and re-run, or pass "
            f"--allow-missing-optional to accept a partial check.",
            file=sys.stderr,
        )
        return 2

    print(
        f"OK: {n_verified} string transform reference(s) verified{extra} "
        f"({len(builtin_names)} transforms in registry)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
