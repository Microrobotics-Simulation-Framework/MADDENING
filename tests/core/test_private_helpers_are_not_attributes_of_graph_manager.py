"""The machinery ``GraphManager`` runs is private to the modules that define it.

``maddening.core.graph_manager`` holds the class.  The module-level
helpers it runs (the coupled block, the fixed-point loop, the IFT solve,
the bounds, the parameter probes, the bookkeeping structs) live in
private modules, and ``graph_manager`` reads them through those modules
instead of importing them by name.  So none of them is an attribute of
``graph_manager``: a reference to ``graph_manager._helper``, and above
all a ``monkeypatch.setattr(graph_manager, "_helper", ...)``, fails
instead of naming a copy that nothing reads.
"""

from __future__ import annotations

import ast
import importlib
import types
from pathlib import Path

import pytest

from maddening.core import graph_manager

#: The private modules that hold what used to be ``graph_manager``'s
#: module-level helpers.
HELPER_MODULES = (
    "maddening.core._adaptive_scan",
    "maddening.core._graph_specs",
    "maddening.core._param_probes",
    "maddening.core.coupling._bounds",
    "maddening.core.coupling._coupled_block",
    "maddening.core.coupling._fixed_point",
    "maddening.core.coupling._group_layout",
    "maddening.core.coupling._ift",
    "maddening.core.coupling._interface_plan",
    "maddening.core.coupling._reports",
)


def _private_definitions(module_name: str) -> list[str]:
    """The private names *module_name* defines at module level (not the ones it imports)."""
    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names += [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return [n for n in names if n.startswith("_") and not n.startswith("__")]


_CASES = [(m, n) for m in HELPER_MODULES for n in _private_definitions(m)]


def test_the_helper_modules_define_private_names():
    """The premise: the scan below is over something (more than a hundred names)."""
    assert len(_CASES) > 100, len(_CASES)
    assert {m for m, _ in _CASES} == set(HELPER_MODULES)


@pytest.mark.parametrize("module_name, name", _CASES, ids=[f"{m.rsplit('.', 1)[1]}.{n}" for m, n in _CASES])
def test_a_private_helper_is_not_an_attribute_of_graph_manager(module_name, name):
    assert not hasattr(graph_manager, name), (
        f"maddening.core.graph_manager.{name} resolves: {module_name} defines it, and "
        f"graph_manager must read it as {module_name.rsplit('.', 1)[1]}.{name}")


def test_a_stale_import_or_patch_fails_loudly(monkeypatch):
    with pytest.raises(ImportError):
        from maddening.core.graph_manager import _META_KEY  # noqa: F401, PLC0415
    with pytest.raises(AttributeError):
        monkeypatch.setattr(graph_manager, "_group_evaluations", lambda *a: None)


def test_every_private_module_graph_manager_reads_is_listed():
    """A helper module added later is scanned too: the list above is the modules
    ``graph_manager`` holds, and the two the others reach only through each other.
    A module is private by its own name, whatever name ``graph_manager`` binds it
    to: ``reason_codes``, public constants held as ``_reason_codes``, is not one."""
    held = {value.__name__ for value in vars(graph_manager).values()
            if isinstance(value, types.ModuleType)
            and value.__name__.startswith("maddening.")
            and value.__name__.rsplit(".", 1)[1].startswith("_")}
    assert held <= set(HELPER_MODULES), sorted(held - set(HELPER_MODULES))
    assert set(HELPER_MODULES) - held == {"maddening.core.coupling._fixed_point",
                                          "maddening.core.coupling._ift"}


def test_the_public_names_are_still_importable_from_graph_manager():
    from maddening.core.coupling import _bounds  # noqa: PLC0415
    from maddening.core.graph_manager import (  # noqa: F401, PLC0415
        EVENT_COMPILED, EVENT_EDGE_ADDED, EVENT_EDGE_REMOVED, EVENT_FIT_PROGRESS,
        EVENT_NODE_ADDED, EVENT_NODE_REMOVED, EVENT_STEP, GRADIENT_PROBE_ENTRY_LIMIT,
        ExternalInputSpec, GraphManager, ShardingIssue,
    )
    assert GRADIENT_PROBE_ENTRY_LIMIT is _bounds.GRADIENT_PROBE_ENTRY_LIMIT
    for public in (ExternalInputSpec, GraphManager, ShardingIssue):
        assert public.__module__ == "maddening.core.graph_manager"
