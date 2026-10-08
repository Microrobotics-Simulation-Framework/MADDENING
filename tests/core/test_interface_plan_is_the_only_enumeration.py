"""Only the group's description reads an edge's ends.

``core/coupling/_interface_plan.py`` describes a coupling group's edges
once: which are internal, in what order, what each delivers and which
fields are read.  The defects it was written to end all began the same
way: a function that needed "the group's internal edges" or "the fields
the norm reads" wrote its own loop over the edges, and its answer drifted
from the others'.

This scan fails when a function in the coupling package or in
``graph_manager.py`` reads an edge's ends or its geometry
(``.source_node``, ``.target_node``, ``.source_field``, ``.target_field``,
``.geometry``) and is neither the description nor on the list below.
The list is the functions that *apply* edges or report on the graph
(boundary resolution, validation, serialisation, messages), each with its
reason.  A new entry is a decision: the right change is nearly always to
read a view of the plan, or to add one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import maddening

#: The package under test, wherever it was imported from.
SRC = Path(maddening.__file__).resolve().parent
#: The attributes whose reading is "looking at an edge's ends".
EDGE_ATTRIBUTES = ("source_node", "target_node", "source_field", "target_field", "geometry")
#: The one module that may read them freely.
DESCRIPTION = "core/coupling/_interface_plan.py"

#: ``file::function`` -> why it may read an edge's ends.  Innermost function,
#: by its qualified name.
ALLOWED = {
    # -- the step applies edges: one node's boundary inputs at a time ------
    "core/coupling/_coupled_block.py::_run_coupled_block_impl._resolve_value":
        "reads one edge's source value, from the state and then the pass's fluxes",
    "core/coupling/_coupled_block.py::_run_coupled_block_impl._resolve_boundary":
        "writes each incoming edge's delivered value into the target's inputs",
    "core/coupling/_coupled_block.py::_run_coupled_block_impl._resolve_boundary_interpolated":
        "the same for a sub-cycled member, interpolating the source and its geometry",
    "core/graph_manager.py::GraphManager._build_step_fn":
        "indexes the graph's edges by target, and the nodes a target anchor reads",
    "core/graph_manager.py::GraphManager._build_step_fn._resolve_and_update_node":
        "the uncoupled step's boundary resolution",
    "core/graph_manager.py::GraphManager._build_dt_step_fn":
        "indexes the graph's edges by target",
    "core/graph_manager.py::GraphManager._build_dt_step_fn._resolve_and_update":
        "the adaptive step's boundary resolution",
    "core/graph_manager.py::GraphManager._boundary_inputs_from":
        "resolve_boundary_inputs: one node's inputs from a state, outside the step",
    "core/coupling/helpers.py::check_conservation.fluxes_of":
        "the conservation diagnostic rebuilds a node's upstream fluxes",
    # -- the graph as a whole: no group's interface ------------------------
    "core/graph_manager.py::GraphManager.validate":
        "checks every edge's endpoints, fields, shapes and units",
    "core/graph_manager.py::GraphManager._remove_node":
        "drops the edges of a removed node",
    "core/graph_manager.py::GraphManager.remove_edge":
        "finds the edge to remove by its ends",
    "core/graph_manager.py::GraphManager._coupling_group_advisories":
        "counts what feeds each input, over edges and external inputs",
    "core/coupling/_group_layout.py::_loop_through_outside_nodes":
        "names the staggered edges of a feedback loop in a warning",
    "core/coupling/_group_layout.py::_staggered_across_components":
        "names a back edge between two components in a warning",
    # -- external inputs, which share the attribute names ------------------
    "core/graph_manager.py::ExternalInputSpec.to_dict": "an external input's own fields",
    "core/graph_manager.py::GraphManager.compile": "an external input's own fields",
    "core/graph_manager.py::GraphManager.from_dict": "an external input's own fields",
    "core/graph_manager.py::GraphManager._default_external_inputs":
        "an external input's own fields",
    "core/graph_manager.py::GraphManager._resolve_external_inputs":
        "an external input's own fields",
    "core/graph_manager.py::GraphManager._warn_of_input_casts":
        "an external input's own fields",
}


def scanned_files() -> list:
    """The modules the rule covers: the coupling package and the graph manager."""
    return [*sorted((SRC / "core" / "coupling").glob("*.py")), SRC / "core" / "graph_manager.py"]


def edge_reads(source: str) -> dict:
    """``{qualified function name: {attribute, ...}}`` for every read of an
    edge attribute in *source*, attributed to the innermost function (or
    class) that contains it; ``"<module>"`` at module level."""
    found: dict[str, set] = {}

    def walk(node, qualified):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(child, [*qualified, child.name])
                continue
            if (isinstance(child, ast.Attribute) and child.attr in EDGE_ATTRIBUTES
                    and isinstance(child.ctx, ast.Load)):
                found.setdefault(".".join(qualified) or "<module>", set()).add(child.attr)
            walk(child, qualified)

    walk(ast.parse(source), [])
    return found


def readers() -> dict:
    """``{"file::function": {attribute, ...}}`` over :func:`scanned_files`."""
    out = {}
    for path in scanned_files():
        rel = path.relative_to(SRC).as_posix()
        for name, attrs in edge_reads(path.read_text(encoding="utf-8")).items():
            out[f"{rel}::{name}"] = attrs
    return out


def problems(found: dict, allowed: dict) -> list:
    """What breaks the rule: a reader that is not allowed, and an allowance
    nothing uses any more."""
    out = [f"{where} reads an edge's {sorted(attrs)} itself"
           for where, attrs in sorted(found.items())
           if not where.startswith(f"{DESCRIPTION}::") and where not in allowed]
    out += [f"{where} is allowed to read an edge's ends and no longer does: drop its entry"
            for where in sorted(allowed) if where not in found]
    return out


def test_no_function_enumerates_a_groups_edges_outside_the_description():
    broken = problems(readers(), ALLOWED)
    assert broken == [], (
        "a coupling group's edges are described once, in "
        f"maddening/{DESCRIPTION}: read a view of the group's plan (or add one there) "
        "instead of reading the edges' ends:\n  " + "\n  ".join(broken))


def test_the_scan_reads_the_modules_the_rule_is_about():
    """The premise: the scope exists, the description is in it and reads
    edges, and so do the functions the list names in both scanned places."""
    files = [p.relative_to(SRC).as_posix() for p in scanned_files()]
    assert len(files) > 15, files
    for rel in (DESCRIPTION, "core/coupling/_coupled_block.py",
                "core/coupling/_group_layout.py", "core/coupling/acceleration.py",
                "core/graph_manager.py"):
        assert rel in files, rel
    found = readers()
    assert f"{DESCRIPTION}::_edge_record" in found
    assert found["core/graph_manager.py::GraphManager._boundary_inputs_from"] >= {
        "source_node", "source_field", "target_node", "target_field"}
    assert "geometry" in found[
        "core/coupling/_coupled_block.py::_run_coupled_block_impl._resolve_boundary_interpolated"]
    assert all(reason.strip() for reason in ALLOWED.values())


def test_the_scan_finds_a_new_enumeration_and_nothing_else():
    """The scan can fail: each way the interface has been enumerated by hand
    is found, in the function that does it, and a reader of the plan is not."""
    by_hand = {
        "def _group_reads(group, edges):\n    return [e for e in edges\n"
        "            if e.source_node in group.nodes and e.target_node in group.nodes]\n":
            {"_group_reads": {"source_node", "target_node"}},
        "def block(group, edges):\n    def _read_fields(s):\n"
        "        return {(e.source_node, e.source_field) for e in edges}\n    return _read_fields\n":
            {"block._read_fields": {"source_node", "source_field"}},
        "def iqn(edges):\n    return [e.geometry[1] for e in edges if e.geometry is not None]\n":
            {"iqn": {"geometry"}},
        "class G:\n    def coupled(self, edges):\n        return {e.target_field for e in edges}\n":
            {"G.coupled": {"target_field"}},
        "KEYS = [e.source_field for e in EDGES]\n": {"<module>": {"source_field"}},
    }
    for source, want in by_hand.items():
        assert edge_reads(source) == want, source
    harmless = (
        "def weights(plan):\n    return {r.source for r in plan.internal if r.mapping is not None}\n",
        "def fed(plan, nn):\n    return plan.coupled_inputs().get(nn)\n",
        "def holder(record):\n    return record.anchor[1], record.target[0]\n",
        "def shape(mapping):\n    return mapping.geometry_shape\n",
        "def make(spec):\n    spec.geometry = None\n",
        "def call(gm):\n    return gm.add_edge(source_node='a', target_node='b')\n",
    )
    for source in harmless:
        assert edge_reads(source) == {}, source


def test_a_reader_off_the_list_and_a_stale_allowance_are_both_reported():
    """The rule itself: the description is free, the list is exact."""
    found = {
        f"{DESCRIPTION}::_edge_record": {"source_node"},
        "core/coupling/_group_layout.py::_new_loop": {"source_field", "source_node"},
        "core/graph_manager.py::GraphManager.validate": {"target_node"},
    }
    allowed = {"core/graph_manager.py::GraphManager.validate": "checks every edge",
               "core/graph_manager.py::GraphManager.gone": "no longer there"}
    assert problems(found, allowed) == [
        "core/coupling/_group_layout.py::_new_loop reads an edge's "
        "['source_field', 'source_node'] itself",
        "core/graph_manager.py::GraphManager.gone is allowed to read an edge's ends and no "
        "longer does: drop its entry",
    ]
    assert problems({f"{DESCRIPTION}::anything": {"geometry"}}, {}) == []
    # A file's name is not an allowance for everything in it.
    assert problems({"core/graph_manager.py::GraphManager._new": {"geometry"}}, allowed)[0].startswith(
        "core/graph_manager.py::GraphManager._new reads")
