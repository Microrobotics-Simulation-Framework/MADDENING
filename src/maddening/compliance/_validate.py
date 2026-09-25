"""
Anomaly registry validation (Section 9.7 / Section 16).

Validates a ``known_anomalies.yaml`` file against the MADDENING anomaly schema.
Usable both as a library function and via the CLI
(``python -m maddening.compliance check-anomalies``).

Also hosts :func:`resolve_dotted_name`, the shared symbol resolver used both
here (for ``affected_components``) and by ``scripts/check_impl_mapping.py``
(for algorithm-guide Implementation Mapping tables), so that the two gates
cannot drift apart on what "this symbol exists" means.
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import os
from types import ModuleType
from typing import NamedTuple, Optional

import yaml


_VALID_SEVERITIES = {"critical", "major", "minor", "enhancement"}
_VALID_SAFETY_RELEVANCES = {
    "safety_relevant", "not_safety_relevant", "context_dependent",
}
# These three mirror the enums in ``maddening.core.compliance.anomaly``.
# They are spelled out rather than imported so this module stays importable
# on its own, and pinned to be *exactly* equal to the enums by
# ``tests/compliance/test_validator.py::TestSchemaEnumsMatchTheDataclass``.
# That pin is how ``partially_resolved`` -- used by MADD-ANO-005, missing
# from ``ResolutionStatus`` until v0.4.0 -- was caught.
_VALID_RESOLUTION_STATUSES = {
    "open", "resolved", "partially_resolved", "wont_fix", "duplicate",
}
_REQUIRED_ANOMALY_FIELDS = (
    "anomaly_id", "title", "description",
    "severity", "safety_relevance", "safety_relevance_rationale",
    # An IEC 62304 known-anomalies list is read for this field above all
    # others: it is what tells a reader whether the defect is still live.
    "resolution_status",
)
# ``affected_versions`` is not checked here.  MADDENING's own
# registry holds it to a PEP 440 convention against ``maddening_version``
# in ``scripts/check_anomalies.py`` (``version_range_errors``); that rule is
# not applied to other registries validated through this function.
_OPTIONAL_ANOMALY_FIELDS = (
    "affected_components", "affected_versions", "workaround",
    "resolution_version", "verification", "residual_risk", "github_issue",
)
_KNOWN_ANOMALY_FIELDS = frozenset(
    _REQUIRED_ANOMALY_FIELDS + _OPTIONAL_ANOMALY_FIELDS
)
_KNOWN_TOPLEVEL_FIELDS = frozenset(
    {"schema_version", "maddening_version", "generated_date", "anomalies"}
)


# ---------------------------------------------------------------------------
# Shared symbol resolution
# ---------------------------------------------------------------------------

class Resolution(NamedTuple):
    """Outcome of resolving a dotted name."""

    ok: bool
    reason: Optional[str] = None
    #: Name of the base class an attribute was actually found on, when it is
    #: not defined on the class the dotted name names.
    inherited_from: Optional[str] = None
    #: Top-level package whose absence stopped the name being checked at all,
    #: e.g. ``"pxr"`` for a ``maddening.usd.*`` symbol in an environment
    #: without the ``usd`` extra.  When this is set, ``ok`` is false but the
    #: reference is *unverified*, not known-stale: the caller must not report
    #: it as broken.
    unavailable: Optional[str] = None


class _PrefixImport(NamedTuple):
    """Result of walking a dotted name's module prefixes."""

    module: Optional[ModuleType] = None
    attrs: tuple[str, ...] = ()
    #: A real import failure seen at a *longer* prefix than the one that
    #: imported.  Kept even when a shorter prefix succeeds.
    error: Optional[str] = None
    #: Top-level package named by that failure, when it was a missing import.
    missing_dependency: Optional[str] = None


def _first_party_import_failure(
    exc: BaseException, first_party: str
) -> Optional[str]:
    """The first-party import that failed somewhere in ``exc``'s chain.

    Walks ``__cause__`` / ``__context__``, because an optional-extra guard
    re-raises: ``maddening.viz``'s lazy loader turns *any* ``ImportError``
    into one naming the extra, so a first-party failure can sit one link
    down.  Returns the failed module's name when an ``ImportError`` in the
    chain names a module of the ``first_party`` package, else ``None``.
    """
    seen: set[int] = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = getattr(current, "name", None)
        if (isinstance(current, ImportError) and name
                and name.split(".")[0] == first_party):
            return f"{type(current).__name__}: {current}"
        current = current.__cause__ or current.__context__
    return None


def _import_longest_prefix(parts: list[str]) -> _PrefixImport:
    """Import the longest importable module prefix of a dotted name.

    An import can fail for three very different reasons, and telling them
    apart is the whole point of this function:

    * ``import maddening.nodes.heat.HeatNode`` fails because the last
      component is an attribute, not a module.  Expected -- keep shortening,
      and say nothing.
    * ``import maddening.usd.serialization`` fails because the ``usd`` extra
      is not installed.  The name cannot be *checked* here; it is not stale.
      This arrives in two shapes: a bare ``ModuleNotFoundError(name='pxr')``,
      and the ``ImportError("maddening.usd requires 'usd-core' ...")`` that
      ``maddening/usd/__init__.py`` raises in its place -- which carries no
      ``name`` at all, so both have to be handled.
    * anything else raised during import is a broken module, and stays a
      hard error.  That includes an ``ImportError`` naming a *first-party*
      module anywhere in its chain -- ``cannot import name 'x' from
      'maddening.core.solver_utils'`` is a stale internal rename, not an
      extra somebody forgot to install.  It used to be labelled with its
      top-level package, ``maddening``, and reported "not checked (missing
      optional extra)", so every reference behind the broken module went
      unverified and the anomaly gate exited 0 (audit_040_p4_1, A25).
      First-party is the name's own top-level package (``parts[0]``).

    A shorter prefix almost always imports after the second kind of failure
    (``maddening`` itself always does), and the accumulated error used to be
    thrown away at that point -- so a missing optional dependency was
    reported as ``'maddening' has no attribute 'usd'``, a stale-reference
    message about a symbol that exists.  The error and the package that
    caused it now travel back with the successful shorter prefix.
    """
    error = None
    missing = None
    for i in range(len(parts), 0, -1):
        modpath = ".".join(parts[:i])
        try:
            mod = importlib.import_module(modpath)
        except ImportError as exc:
            name = getattr(exc, "name", None)
            # "No module named 'maddening.nodes.heat.HeatNode'" -- the tail is
            # an attribute.  Keep shortening; this is not a failure.
            if (isinstance(exc, ModuleNotFoundError)
                    and name and modpath.startswith(name)):
                continue
            broken = _first_party_import_failure(exc, parts[0])
            if broken is not None:
                # Not an optional extra: the package's own code is broken.
                # No ``missing`` label, so callers report it as an error.
                error = (f"importing {modpath} failed on a broken first-party "
                         f"import ({broken}), not a missing optional extra")
                missing = None
                continue
            # Overwrite rather than keep the first: prefixes are tried
            # longest-first, so the last real failure is the shortest one --
            # ``maddening.usd`` rather than ``maddening.usd.serialization.f``.
            # That is the subpackage boundary, and the message it raises is
            # the one that names the extra to install.
            error = f"importing {modpath} failed: {exc}"
            # The third-party package when we know it, else the module that
            # refused to import.  Either way it is a label for "unavailable
            # here", not a claim about the reference.
            missing = name.split(".")[0] if name else modpath
        except Exception as exc:  # pragma: no cover - defensive
            error = f"importing {modpath} raised {type(exc).__name__}: {exc}"
        else:
            return _PrefixImport(mod, tuple(parts[i:]), error, missing)
    return _PrefixImport(None, (), error, missing)


def resolve_dotted_name(
    qname: str,
    *,
    require_own: bool = False,
    require_callable: bool = False,
    require_routine: bool = False,
) -> Resolution:
    """Resolve ``module.path.Class.attr`` and report why it failed.

    Parameters
    ----------
    qname : str
        Dotted name to resolve.
    require_own : bool, optional
        When true, an attribute looked up on a class must be defined in
        that class's own ``__dict__``.  ``getattr`` walks the MRO, so
        without this a documentation row naming ``HeatNode.update`` keeps
        resolving after the concrete method is deleted or renamed, because
        ``SimulationNode.update`` answers in its place.
    require_callable : bool, optional
        When true, the resolved object must be callable.
    require_routine : bool, optional
        When true, the resolved object must be a function, a method or a
        property -- code an equation term can be traced to.  A class is
        callable, so ``require_callable`` alone accepted one, and a row
        re-pointed from ``HeatNode.update`` to ``HeatNode`` kept resolving
        (audit_040_p4_2, M5).  A wrapper that exposes ``__wrapped__``
        (``functools.wraps``, ``jax.jit``) is judged by what it wraps.

    Returns
    -------
    Resolution
        ``ok`` is false with a ``reason`` when the name does not resolve.
        ``inherited_from`` names the defining base class when an attribute
        was found only through the MRO, whether or not that was allowed.
        ``unavailable`` names a missing third-party package when the name
        could not be *checked* here at all -- that is not a stale reference
        and callers must not report it as one.
    """
    parts = qname.split(".")
    if len(parts) < 2:
        return Resolution(False, f"'{qname}' is not a dotted name")

    found = _import_longest_prefix(parts)
    mod, attrs = found.module, found.attrs

    def _unverified() -> Resolution:
        return Resolution(
            False,
            f"'{qname}' could not be checked in this environment: "
            f"{found.error}",
            None,
            found.missing_dependency,
        )

    if mod is None:
        if found.missing_dependency:
            return _unverified()
        return Resolution(
            False, found.error or f"no importable module in '{qname}'"
        )

    obj = mod
    inherited_from = None
    seen = mod.__name__
    parent = None
    attr = None
    for attr in attrs:
        parent = obj
        if not hasattr(parent, attr):
            # A longer prefix failed to import, and the walk from the shorter
            # one cannot get past it: a submodule is not an attribute of its
            # package until it has been imported.  Whatever stopped that
            # import is the real answer -- "has no attribute" describes the
            # symptom and blames the wrong thing.
            if found.missing_dependency:
                return _unverified()
            if found.error:
                # A module that exists and raises: a defect, and a hard
                # failure, but report what actually happened.
                return Resolution(
                    False, f"'{qname}' could not be resolved: {found.error}"
                )
            return Resolution(False, f"'{seen}' has no attribute '{attr}'")
        if isinstance(parent, type) and attr not in parent.__dict__:
            owner = next(
                (k.__name__ for k in parent.__mro__ if attr in k.__dict__),
                None,
            )
            inherited_from = owner
            if require_own:
                return Resolution(
                    False,
                    f"'{attr}' is not defined on {parent.__name__}; it "
                    f"resolves only through base class {owner}",
                    owner,
                )
        obj = getattr(parent, attr)
        seen = f"{seen}.{attr}"

    if require_routine:
        # A property read off its class is the property object, which is
        # not callable; it is still code a row can trace a term to.
        static = None
        if isinstance(parent, type) and attr is not None:
            try:
                static = inspect.getattr_static(parent, attr)
            except AttributeError:
                static = None
        if isinstance(static, (property, functools.cached_property)):
            return Resolution(True, None, inherited_from)
        if isinstance(obj, type):
            return Resolution(
                False,
                f"'{qname}' resolves to the class {obj.__name__}, not to a "
                f"function, method or property",
                inherited_from,
            )
        try:
            target = inspect.unwrap(obj)
        except ValueError:               # a __wrapped__ cycle
            target = obj
        if not (inspect.isroutine(obj) or inspect.isroutine(target)):
            return Resolution(
                False,
                f"'{qname}' resolves to a {type(obj).__name__}, not to a "
                f"function, method or property",
                inherited_from,
            )

    if require_callable and not callable(obj):
        return Resolution(
            False,
            f"'{qname}' resolves to a {type(obj).__name__}, which is not "
            f"callable",
            inherited_from,
        )

    return Resolution(True, None, inherited_from)


_DEFINITION = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _parse(py_path: str) -> Optional[ast.Module]:
    try:
        with open(py_path) as f:
            return ast.parse(f.read(), filename=py_path)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None


def _defined_in(body: list) -> dict:
    """The functions and classes defined directly in ``body``, by name."""
    return {node.name: node for node in body if isinstance(node, _DEFINITION)}


def _member(
    cls: ast.ClassDef, name: str, module: dict, seen: frozenset = frozenset()
) -> Optional[ast.AST]:
    """``name`` defined on ``cls`` or inherited from a base in the same file.

    pytest collects a test method a class inherits, under the subclass's
    node id, so an inherited method is a real node id.  A base that is not
    a class defined in this file cannot be looked into and yields None:
    the caller fails closed rather than trusting it.
    """
    own = _defined_in(cls.body).get(name)
    if own is not None:
        return own
    for base in cls.bases:
        base_def = module.get(base.id) if isinstance(base, ast.Name) else None
        if isinstance(base_def, ast.ClassDef) and base_def.name not in seen:
            found = _member(base_def, name, module, seen | {cls.name})
            if found is not None:
                return found
    return None


def resolve_test_reference(ref: str, repo_root: str) -> Optional[str]:
    """Check a ``path/to/test.py::test_name`` reference; return a reason or None.

    The ``::`` part is optional.  After the path, the components are
    resolved the way pytest reads a node id: the first must be defined at
    module level, and each further one must be defined in the body of the
    class the previous one names, or inherited from a base class defined
    in the same file.  A function has no children in a node id.

    This used to accept ``File::Class::method`` whenever both names were
    defined *anywhere* in the file, so a verification entry naming a
    module-level test under a class it is not in -- a node id pytest
    reports as "not found" -- passed the registry gate.
    """
    parts = str(ref).split("::")
    rel = parts[0]
    abspath = os.path.join(repo_root, rel)
    if not os.path.isfile(abspath):
        return f"test file '{rel}' does not exist"
    if len(parts) == 1:
        return None
    tree = _parse(abspath)
    if tree is None:
        return f"'{rel}' could not be parsed, so {ref!r} cannot be resolved"
    module = _defined_in(tree.body)
    node: Optional[ast.AST] = None
    for parent, name in zip([None, *parts[1:-1]], parts[1:]):
        if node is None:
            found, where = module.get(name), "at module level"
        elif isinstance(node, ast.ClassDef):
            found, where = _member(node, name, module), f"in class {node.name}"
        else:
            return (f"'{rel}' defines no {name!r} inside {parent!r}, which is "
                    f"a function, not a class")
        if found is None:
            return f"'{rel}' defines no {name!r} {where}"
        node = found
    return None


def _find_repo_root(path: str) -> Optional[str]:
    """Walk up from ``path`` looking for the directory with pyproject.toml."""
    here = os.path.dirname(os.path.abspath(path))
    while True:
        if os.path.isfile(os.path.join(here, "pyproject.toml")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


# ---------------------------------------------------------------------------
# Registry validation
# ---------------------------------------------------------------------------

def validate_anomaly_registry(
    path: str,
    *,
    prefix: str = "",
    repo_root: Optional[str] = None,
    resolve_references: bool = True,
    notes: Optional[list[str]] = None,
) -> list[str]:
    """Validate a known_anomalies.yaml file against the anomaly schema.

    Parameters
    ----------
    path : str
        Path to the YAML file to validate.
    prefix : str, optional
        Expected anomaly ID prefix (e.g., ``"MIME-ANO-"``).  If provided,
        all anomaly IDs must start with this prefix.  If empty, any
        prefix is accepted.
    repo_root : str, optional
        Directory that ``verification`` test paths are resolved against.
        Defaults to the nearest ancestor of ``path`` holding a
        ``pyproject.toml``.
    resolve_references : bool, optional
        When true (the default), every ``affected_components`` symbol must
        import and every ``verification`` entry must name a real test file
        and a real ``def``.  A registry pointing at code that no longer
        exists is the failure this list is kept to prevent.
    notes : list of str, optional
        Appended to, never read.  A reference that could not be *checked*
        here -- a ``maddening.usd`` symbol in an environment without the
        ``usd`` extra -- is recorded here rather than in the returned
        errors, because "I could not look" is not "this is broken".
        Callers that care about coverage should surface these.

    Returns
    -------
    list[str]
        List of validation errors.  Empty list means the file is valid.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    errors: list[str] = []

    # Top-level structure
    if not isinstance(data, dict):
        return ["File must contain a YAML mapping"]

    for required in ("schema_version", "generated_date"):
        if required not in data:
            errors.append(f"Missing top-level field: {required}")

    for key in sorted(set(data) - _KNOWN_TOPLEVEL_FIELDS):
        errors.append(
            f"Unknown top-level field: {key} "
            f"(known: {', '.join(sorted(_KNOWN_TOPLEVEL_FIELDS))})"
        )

    if "anomalies" not in data:
        errors.append("Missing top-level field: anomalies (must be a list, may be empty)")
        return errors

    if not isinstance(data["anomalies"], list):
        errors.append("'anomalies' must be a list")
        return errors

    if repo_root is None:
        repo_root = _find_repo_root(path)

    # Anomaly entries
    ids_seen: set[str] = set()
    for a in data.get("anomalies", []):
        if not isinstance(a, dict):
            errors.append(f"Anomaly entry is not a mapping: {a!r}")
            continue

        aid = a.get("anomaly_id", "<missing>")

        # Uniqueness
        if aid in ids_seen:
            errors.append(f"Duplicate anomaly_id: {aid}")
        ids_seen.add(aid)

        # Prefix enforcement
        if prefix and not str(aid).startswith(prefix):
            errors.append(f"{aid}: does not match required prefix '{prefix}'")

        # Required fields
        for fld in _REQUIRED_ANOMALY_FIELDS:
            if not a.get(fld):
                errors.append(f"{aid}: missing required field '{fld}'")

        # Unknown fields.  A typo like ``workaroud:`` silently drops the
        # workaround from an anomaly a reader is relying on.
        for key in sorted(set(a) - _KNOWN_ANOMALY_FIELDS):
            errors.append(f"{aid}: unknown field '{key}'")

        # Valid enums
        sev = a.get("severity")
        if sev is not None and sev not in _VALID_SEVERITIES:
            errors.append(f"{aid}: invalid severity '{sev}'")

        sr = a.get("safety_relevance")
        if sr is not None and sr not in _VALID_SAFETY_RELEVANCES:
            errors.append(f"{aid}: invalid safety_relevance '{sr}'")

        rs = a.get("resolution_status")
        if rs is not None and rs not in _VALID_RESOLUTION_STATUSES:
            errors.append(f"{aid}: invalid resolution_status '{rs}'")

        if not resolve_references:
            continue

        # affected_components must name importable symbols.  They are not
        # required to be callable: several name a dataclass field.
        components = a.get("affected_components") or []
        if isinstance(components, str):
            components = [components]
        for comp in components:
            res = resolve_dotted_name(str(comp))
            if res.ok:
                continue
            if res.unavailable:
                # An optional subpackage this environment cannot import.
                # Asserting the reference is stale would be a false
                # statement in a compliance artefact, and would make every
                # anomaly touching maddening.usd or maddening.viz
                # unrecordable in a CI that installs only [ci].
                if notes is not None:
                    notes.append(
                        f"{aid}: affected_components entry '{comp}' was NOT "
                        f"checked -- {res.reason}"
                    )
                continue
            errors.append(
                f"{aid}: affected_components entry '{comp}' does not "
                f"resolve ({res.reason})"
            )

        # verification entries must name a real test file and a real def.
        verification = a.get("verification") or []
        if isinstance(verification, str):
            verification = [verification]
        if verification and repo_root is None:
            errors.append(
                f"{aid}: cannot resolve verification entries -- no repository "
                f"root found above {path}; pass repo_root explicitly"
            )
        elif verification and repo_root is not None:
            for ref in verification:
                reason = resolve_test_reference(str(ref), repo_root)
                if reason:
                    errors.append(f"{aid}: verification entry '{ref}': {reason}")

    return errors
