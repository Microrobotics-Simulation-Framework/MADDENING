"""Every name a lazy package exports is visible to a type checker.

Several packages list their public names in ``__all__`` and resolve them
through a `PEP 562 <https://peps.python.org/pep-0562/>`_ module
``__getattr__`` so that importing the package never pulls in an optional
dependency.  A type checker cannot execute that table, so unless the same
names are also imported inside an ``if TYPE_CHECKING:`` block, a
downstream ``from maddening.cloud import CloudSession`` is an error for
the consumer -- which matters now that the distribution ships
``py.typed`` and its annotations are believed.

These checks are static (``ast``), for two reasons: importing the lazy
names would install-gate the test on ``pygfx``/``skypilot``/``equinox``,
and the property being checked is precisely what a checker sees *without
running anything*.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "maddening"

#: Packages whose ``__all__`` is served by a lazy ``__getattr__``.
LAZY_PACKAGES = (
    "__init__.py",
    "cloud/__init__.py",
    "surrogates/__init__.py",
    "surrogates/architectures/__init__.py",
    "surrogates/training/__init__.py",
    "viz/__init__.py",
    "viz/backends/__init__.py",
)


def _module(rel: str) -> ast.Module:
    return ast.parse((PACKAGE_ROOT / rel).read_text())


def _dunder_all(tree: ast.Module) -> list[str]:
    """``__all__`` as a list of names, or ``[]`` when it is not a literal."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == "__all__" for t in node.targets):
            continue
        if not isinstance(node.value, ast.List):
            return []
        return [e.value for e in node.value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def _is_type_checking(test: ast.expr) -> bool:
    """``if TYPE_CHECKING:`` / ``if typing.TYPE_CHECKING:``."""
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _bound_names(tree: ast.Module, *, type_checking: bool = True) -> set[str]:
    """Names a checker can see: real imports, defs, classes, assignments.

    Includes the contents of ``if TYPE_CHECKING:`` blocks, which is the
    whole point -- they are invisible at runtime and visible to a checker.
    With ``type_checking=False`` those blocks are skipped, which gives the
    opposite view: the names that exist when the module is *executed*.
    """
    names: set[str] = set()

    def walk(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(a.asname or a.name.split(".")[0] for a in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.If):
                if type_checking or not _is_type_checking(node.test):
                    walk(node.body)
                walk(node.orelse)
            elif isinstance(node, ast.Try):
                walk(node.body)
                for h in node.handlers:
                    walk(h.body)
                walk(node.orelse)
                walk(node.finalbody)

    walk(tree.body)
    return names


def _lazy_table_keys(tree: ast.Module) -> set[str]:
    """The string keys of the module's PEP 562 lazy table.

    Spelled differently across the seven packages -- a module-level
    ``_LAZY`` (annotated or not) in some, a ``_lazy`` local inside
    ``__getattr__`` in others, values that are a module path in some and a
    ``(module, attribute)`` pair in ``viz`` -- so this matches on the name
    and reads only the keys, anywhere in the module.
    """
    keys: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if not any(isinstance(t, ast.Name) and t.id.lower() == "_lazy"
                   for t in targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        keys.update(k.value for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str))
    return keys


@pytest.mark.parametrize("rel", LAZY_PACKAGES)
def test_dunder_all_is_a_literal_list(rel: str) -> None:
    """A computed ``__all__`` is opaque to a checker, so it is spelled out."""
    assert _dunder_all(_module(rel)), (
        f"{rel}: __all__ is missing or is not a list literal; a checker "
        "cannot evaluate `list(_LAZY.keys())` and reports every downstream "
        "`from ... import X` as an error"
    )


@pytest.mark.parametrize("rel", LAZY_PACKAGES)
def test_every_exported_name_is_visible_to_a_checker(rel: str) -> None:
    """No ``__all__`` entry is reachable only through ``__getattr__``."""
    tree = _module(rel)
    missing = sorted(set(_dunder_all(tree)) - _bound_names(tree))
    assert not missing, (
        f"{rel}: {missing} are in __all__ but bound nowhere a type checker "
        "can see them.  Add them to the `if TYPE_CHECKING:` re-export block "
        "next to the lazy table (they stay lazy at runtime)."
    )


def test_training_dunder_all_matches_its_lazy_table() -> None:
    """The spelled-out ``__all__`` and ``_LAZY`` cannot drift apart.

    ``maddening.surrogates.training`` used to derive one from the other.
    Spelling ``__all__`` out for the checker means a name added to
    ``_LAZY`` alone would silently stop being exported.
    """
    from maddening.surrogates.training import _LAZY, __all__ as exported
    assert list(exported) == list(_LAZY), (
        "maddening/surrogates/training/__init__.py: __all__ and _LAZY have "
        "drifted; they were one expression until __all__ was spelled out "
        "for the type checker"
    )


@pytest.mark.parametrize("rel", LAZY_PACKAGES)
def test_every_exported_name_also_resolves_at_runtime(rel: str) -> None:
    """The direction that hurts a ``py.typed`` consumer.

    ``test_every_exported_name_is_visible_to_a_checker`` checks
    ``__all__`` minus what a checker can see.  This is the opposite
    subtraction: a name in ``__all__`` and in the ``if TYPE_CHECKING:``
    block but absent from the runtime lazy table type-checks perfectly
    and raises ``AttributeError`` on access -- and with the marker
    shipped, the checker's word is what a consumer acts on, so the
    mistake surfaces as a crash in their code rather than an error in
    ours.  Both subtractions are needed; neither implies the other
    (audit_040_r3, L2: all 16 tests passed on a seeded instance).

    Static for the same reason as its counterpart: importing the lazy
    names would install-gate this on pygfx / skypilot / equinox.
    """
    tree = _module(rel)
    reachable = (_bound_names(tree, type_checking=False)
                 | _lazy_table_keys(tree))
    unresolvable = sorted(set(_dunder_all(tree)) - reachable)
    assert not unresolvable, (
        f"{rel}: {unresolvable} are in __all__ and visible to a type "
        "checker, but are neither imported eagerly nor keys of the lazy "
        "table, so `from ... import X` type-checks and raises "
        "AttributeError at runtime.  Add them to the lazy table."
    )


def test_the_lazy_table_reader_reads_the_table_and_not_dunder_all() -> None:
    """The check above is only as good as this helper.

    A helper that quietly returned ``__all__`` would make every package
    look consistent -- the failure mode mutation testing cannot reach
    from outside, because it is in the test's own machinery.
    ``maddening.viz`` is the one package where the two sets differ in
    both directions: four names are imported eagerly and are in
    ``__all__`` but not in the table, and three USD helpers are in the
    table but not in ``__all__``.  Asserting the exact set therefore
    pins that the reader read the dict.
    """
    tree = _module("viz/__init__.py")
    assert _lazy_table_keys(tree) == {
        "HistoryViewer3D", "GPUHistoryViewer", "PyVistaLiveRenderer",
        "viewer_from_usd", "viewer_from_usd_with_geometry",
        "render_usd_frame",
    }
    assert "Renderer" in _dunder_all(tree)          # eager, not in the table
    assert "Renderer" not in _lazy_table_keys(tree)
    assert "render_usd_frame" not in _dunder_all(tree)  # table, not exported


@pytest.mark.parametrize("rel", LAZY_PACKAGES)
def test_the_lazy_table_is_found_where_each_package_spells_it(rel: str) -> None:
    """The check above passes vacuously if the table cannot be located.

    ``_lazy_table_keys`` matches on a name, so a package that renamed its
    table would silently contribute nothing and every lazy export would
    look eager-or-missing.  It fails closed in that direction -- the names
    would be reported as unresolvable -- but only as long as some name is
    actually lazy, which is what this pins.
    """
    assert _lazy_table_keys(_module(rel)), (
        f"{rel}: no lazy table found.  It is matched by the name `_LAZY` / "
        "`_lazy` bound to a dict literal; if this package spells it "
        "differently, teach _lazy_table_keys about it rather than leaving "
        "the runtime check with nothing to compare against."
    )
