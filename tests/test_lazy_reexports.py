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


def _bound_names(tree: ast.Module) -> set[str]:
    """Names a checker can see: real imports, defs, classes, assignments.

    Includes the contents of ``if TYPE_CHECKING:`` blocks, which is the
    whole point -- they are invisible at runtime and visible to a checker.
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
