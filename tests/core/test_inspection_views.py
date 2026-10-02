"""What the inspection views say, on every kind of graph.

``format_graph``, ``to_mermaid`` / ``to_dot``, ``state_summary``,
``params_table``, ``coupling_report``, ``memory_estimate`` and the
``InspectionTable`` they return.  That they change nothing is
``test_inspection_read_only.py``; this file checks that what they report
is true, that the text formats are well formed and deterministic, and
pins the exact text on a small graph (golden outputs).
"""

from __future__ import annotations

import io
import math
import os
import re
import shutil
import subprocess
import textwrap

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core import inspection
from maddening.core.graph_manager import GraphManager
from maddening.core.inspection import InspectionTable
from maddening.core.node import SimulationNode
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from tests.core.inspection_graphs import BUILDERS, build, graph


# ----------------------------------------------------------------------
# format_graph
# ----------------------------------------------------------------------

def test_graph_text_names_every_node_edge_group_and_input():
    text = graph("mapped").format_graph()
    for needle in ("rod_a  HeatNode", "rod_b  HeatNode", "temperature float32[6]",
                   "rod_a.temperature -> rod_b.heat_source",
                   "mapping StaticLinearMapping 6->12", "additive",
                   "rod_a.right_heat_flux -> rod_b.left_temperature", "flux; transform negate",
                   "transform _mapped.<locals>.<lambda>",
                   "back edge: reads the previous step's value",
                   "External inputs (1)", "rod_a.left_temperature", "float32[]",
                   "1. rod_a", "2. rod_b"):
        assert needle in text, needle


def test_graph_text_reports_the_coupling_group_settings():
    text = graph("coupled_diagnostics").format_graph()
    for needle in ("Coupling groups (1)", "members: a, b",
                   "solver ift, acceleration none, mode gauss-seidel",
                   "norm l2 (tolerance 1e-06), max_iterations 8",
                   "diagnostics on, strict off", "[a+b] coupled block: a, b",
                   "iterated inside its coupling group"):
        assert needle in text, needle


def test_graph_text_shows_the_norm_s_live_tolerances():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01))
    gm.add_node(SpringDamperNode("b", 0.01))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], convergence_norm="mixed", rtol=1e-4, atol=1e-8,
                          acceleration="aitken", strict_convergence=True)
    text = gm.format_graph()
    assert "norm mixed (atol 1e-08, rtol 0.0001)" in text
    assert "acceleration aitken" in text and "strict on" in text
    assert "tolerance" not in text.split("Coupling groups")[1].split("External")[0]


def test_graph_text_reports_rate_dividers_and_firing_on_a_multirate_graph():
    text = graph("multirate").format_graph()
    assert "(multi-rate)" in text
    assert "timestep 0.02, rate divider 2" in text
    assert "2. slow\n      every 2 base steps" in text


def test_graph_text_reports_sub_cycling_and_the_true_base_step():
    text = graph("subcycled").format_graph()
    assert "sub-cycled x2 per coupling pass" in text
    assert "sub-cycling: coarse x1, fine x2 evaluations per pass" in text
    # one step of a sub-cycling group is its largest member timestep
    assert "base timestep: 0.02" in text
    assert "gm.timestep reads 0.01" in text


def test_graph_text_on_an_uncompiled_graph_reports_what_compile_would_decide_as_unknown():
    text = graph("single_uncompiled").format_graph()
    assert "status: not compiled" in text
    assert "rate divider: not compiled" in text
    assert "Execution order (not compiled)\n  unknown until compile()" in text


def test_graph_text_on_a_modified_graph_marks_the_schedule_stale():
    text = graph("stale").format_graph()
    assert "modified since the last compile" in text
    assert "Execution order (last compile; stale)" in text
    assert "t  SpringDamperNode" in text
    assert "rate divider: not compiled (added since the last compile)" in text


def test_graph_text_on_a_traced_graph_reads_shapes_from_the_tracers():
    text = graph("after_grad").format_graph()
    assert "state holds JAX tracers" in text
    assert "state: position float32[], velocity float32[]" in text


def test_graph_text_on_an_empty_graph():
    text = GraphManager().format_graph()
    assert text.startswith("GraphManager: 0 nodes, 0 edges")
    assert "base timestep" not in text


@pytest.mark.parametrize("width", [40, 72, 100])
def test_long_names_wrap_at_the_width_and_are_never_split(width):
    long = "a_node_with_an_exceptionally_long_descriptive_name_" + "x" * 30
    gm = GraphManager()
    gm.add_node(SpringDamperNode(long, 0.01))
    gm.add_node(SpringDamperNode("b", 0.01))
    gm.add_edge(long, "b", "position", "anchor_position")
    text = gm.format_graph(width=width)
    assert long in text                       # whole, on one line
    for line in text.splitlines():
        if len(line) > width:
            # only a single unbreakable token may overflow, alone after its indent
            assert len(line.split()) <= 3 and any(long in tok for tok in line.split()), line


def test_graph_text_is_deterministic_across_builds():
    one, two = build("coupled"), build("coupled")
    assert one.format_graph() == two.format_graph()
    assert one.to_mermaid() == two.to_mermaid() and one.to_dot() == two.to_dot()
    assert str(one.params_table()) == str(two.params_table())


def test_print_graph_writes_format_graph_to_the_file():
    gm = graph("coupled")
    buf = io.StringIO()
    gm.print_graph(file=buf, width=60)
    assert buf.getvalue() == gm.format_graph(width=60)


def test_print_graph_rich_renders_the_same_names():
    pytest.importorskip("rich", reason="rich is an optional extra (maddening[terminal])")
    buf = io.StringIO()
    graph("mapped").print_graph(file=buf, rich=True)
    out = buf.getvalue()
    assert "rod_a.temperature -> rod_b.heat_source" in out and "External inputs (1)" in out


def test_rich_without_the_package_names_the_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_rich(name, *args, **kwargs):
        if name == "rich" or name.startswith("rich."):
            raise ImportError("No module named 'rich'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_rich)
    with pytest.raises(ImportError, match=r"maddening\[terminal\]"):
        graph("single").print_graph(file=io.StringIO(), rich=True)
    with pytest.raises(ImportError, match=r"maddening\[terminal\]"):
        graph("single").print_state_summary(file=io.StringIO(), rich=True)
    # plain text needs nothing
    graph("single").print_graph(file=io.StringIO())


# ----------------------------------------------------------------------
# Mermaid and DOT: well formed, escaped, plain text
# ----------------------------------------------------------------------

_MID = r"[A-Za-z][A-Za-z0-9_]*"
_MLABEL = r'"[^"]*"'
_MERMAID_LINES = {
    "node": re.compile(rf"^({_MID})\[({_MLABEL})\]$"),
    "input": re.compile(rf"^({_MID})\[/({_MLABEL})/\]$"),
    "subgraph": re.compile(rf"^subgraph ({_MID})\[({_MLABEL})\]$"),
    "end": re.compile(r"^end$"),
    "edge": re.compile(rf"^({_MID}) (-->|-\.->)\|({_MLABEL})\| ({_MID})$"),
    "plain_edge": re.compile(rf"^({_MID}) -\.-> ({_MID})$"),
}


def check_mermaid(text: str) -> dict:
    """Structural well-formedness of the flowchart :func:`to_mermaid` writes.

    Every line is one statement of a known form; labels are quoted and
    hold no raw quote, backtick, ampersand or angle bracket except the
    ``<br/>`` line breaks; ``#`` only opens an entity code; subgraphs
    balance; every edge endpoint is declared.  Returns the declared ids.
    """
    lines = text.rstrip("\n").split("\n")
    assert re.fullmatch(r"flowchart (LR|RL|TB|BT)", lines[0]), lines[0]
    declared, edges, depth = set(), [], 0
    for raw in lines[1:]:
        line = raw.strip()
        for kind, pattern in _MERMAID_LINES.items():
            m = pattern.match(line)
            if m:
                break
        else:
            raise AssertionError(f"unrecognised Mermaid line: {raw!r}")
        labels = [g for g in m.groups() if g and g.startswith('"')]
        for label in labels:
            body = label[1:-1].replace("<br/>", "")
            assert not re.search(r"[<>`&\"]", body), label
            assert not re.search(r"#(?!\w+;)", body), label
        if kind == "subgraph":
            assert depth == 0, "nested subgraph"
            depth += 1
        elif kind == "end":
            assert depth == 1, "unbalanced end"
            depth -= 1
        elif kind in ("node", "input"):
            assert m.group(1) not in declared, f"{m.group(1)} declared twice"
            declared.add(m.group(1))
        elif kind == "edge":
            edges.append((m.group(1), m.group(4)))
        else:
            edges.append((m.group(1), m.group(2)))
    assert depth == 0, "unclosed subgraph"
    for a, b in edges:
        assert a in declared and b in declared, (a, b)
    return {"declared": declared, "edges": edges}


_DOT_STRING = r'"(?:[^"\\]|\\.)*"'


def check_dot(text: str) -> None:
    """DOT that a parser accepts: Graphviz's own ``dot`` when installed,
    else ``pydot``, and in every case a structural check."""
    lines = text.rstrip("\n").split("\n")
    assert lines[0] == "digraph maddening {" and lines[-1] == "}"
    depth = 0
    for raw in lines:
        line = raw.strip()
        stripped = re.sub(_DOT_STRING, '""', line)
        assert '"' not in stripped.replace('""', ""), f"unbalanced quote: {raw!r}"
        depth += stripped.count("{") - stripped.count("}")
        assert depth >= 0
        if not (line.endswith("{") or line == "}"):
            assert line.endswith(";"), raw
    assert depth == 0
    exe = shutil.which("dot")
    if exe is not None:
        proc = subprocess.run([exe, "-Tcanon"], input=text, capture_output=True, text=True,
                              timeout=60)
        assert proc.returncode == 0, proc.stderr
    try:
        import pydot
    except ImportError:
        pydot = None
    if pydot is not None:
        assert pydot.graph_from_dot_data(text), "pydot could not parse the DOT"


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_mermaid_is_well_formed_for_every_graph(kind):
    gm = graph(kind)
    parsed = check_mermaid(gm.to_mermaid())
    node_ids = {f"n{i}" for i in range(len(gm.node_names))}
    assert node_ids <= parsed["declared"]
    assert len([e for e in parsed["edges"] if not e[0].startswith("x")]) == len(gm.edges)


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_dot_is_well_formed_for_every_graph(kind):
    check_dot(graph(kind).to_dot())


def test_coupling_groups_are_subgraphs_and_edges_are_labelled_field_to_input():
    gm = graph("coupled")
    mermaid = gm.to_mermaid()
    assert 'subgraph g0["coupling group a+b"]' in mermaid
    assert '|"position→anchor_position"|' in mermaid
    dot = gm.to_dot()
    assert "subgraph cluster_g0 {" in dot and 'label="coupling group a+b";' in dot
    assert '[label="position→anchor_position"]' in dot


def test_flux_edges_and_external_inputs_are_drawn_dashed():
    mermaid = graph("mapped").to_mermaid()
    assert re.search(r"n0 -\.->\|\"right_heat_flux→left_temperature \(negate\)\"\| n1", mermaid)
    assert 'x0[/"external: left_temperature"/]' in mermaid and "x0 -.-> n0" in mermaid
    dot = graph("mapped").to_dot()
    assert "style=dashed" in dot and "shape=parallelogram" in dot


def _hostile_graph() -> GraphManager:
    name = 'we"ird <b>&amp; `tick` naïve\nsecond line'
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name, 0.01))
    gm.add_node(SpringDamperNode("plain", 0.01))
    gm.add_edge(name, "plain", "position", "anchor_position")
    gm.add_edge("plain", name, "position", "anchor_position")
    gm.add_coupling_group([name, "plain"])
    return gm


def test_names_are_escaped_so_they_cannot_break_the_chart():
    gm = _hostile_graph()
    mermaid = gm.to_mermaid()
    check_mermaid(mermaid)
    assert "we#quot;ird #lt;b#gt;#38;amp; #96;tick#96; naïve<br/>second line" in mermaid
    dot = gm.to_dot()
    check_dot(dot)
    assert 'we\\"ird <b>&amp; `tick` naïve\\nsecond line' in dot


def test_direction_is_validated():
    gm = graph("single")
    assert gm.to_mermaid(direction="TB").startswith("flowchart TB\n")
    assert "rankdir=BT;" in gm.to_dot(rankdir="BT")
    with pytest.raises(ValueError, match="direction"):
        gm.to_mermaid(direction="sideways")
    with pytest.raises(ValueError, match="direction"):
        gm.to_dot(rankdir="LR; evil")


# ----------------------------------------------------------------------
# state_summary
# ----------------------------------------------------------------------

def test_state_summary_rows_hold_the_state_s_values():
    gm = graph("mapped")
    table = gm.state_summary()
    rows = {(r["node"], r["field"]): r for r in table}
    assert set(rows) == {("rod_a", "temperature"), ("rod_b", "temperature")}
    for (node, field), row in rows.items():
        arr = np.asarray(gm.get_node_state(node)[field])
        assert row["shape"] == arr.shape and row["dtype"] == "float32"
        assert row["min"] == float(arr.min()) and row["max"] == float(arr.max())
        assert math.isclose(row["mean"], float(arr.astype(np.float64).mean()), rel_tol=1e-12)
        assert row["nan"] == 0 and row["inf"] == 0 and row["bytes"] == arr.nbytes
        assert row["flags"] == ()


def test_state_summary_counts_and_flags_non_finite_entries():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01))
    gm.set_node_state("s", {"position": jnp.asarray(jnp.nan, jnp.float32),
                            "velocity": jnp.asarray(jnp.inf, jnp.float32)})
    rows = {r["field"]: r for r in gm.state_summary()}
    assert rows["position"]["nan"] == 1 and rows["position"]["min"] is None
    assert rows["velocity"]["inf"] == 1
    assert rows["position"]["flags"] == ("non-finite values: 1 NaN, 0 inf",)
    assert "! s.position: non-finite values: 1 NaN, 0 inf" in str(gm.state_summary())


def test_state_summary_includes_meta_only_when_asked():
    gm = graph("coupled")
    assert {r["node"] for r in gm.state_summary()} == {"a", "b"}
    meta = [r for r in gm.state_summary(include_meta=True) if r["node"] == "_meta"]
    assert {r["field"] for r in meta} == {"coupling_a+b_iterations", "coupling_a+b_residual",
                                          "coupling_a+b_amplification"}
    iters = next(r for r in meta if r["field"] == "coupling_a+b_iterations")
    assert iters["dtype"] == "int32" and iters["min"] == iters["max"] >= 1


def test_state_summary_on_a_traced_graph_has_shapes_but_no_values():
    table = graph("after_grad").state_summary()
    assert len(table) == 4
    for row in table:
        assert row["shape"] == () and row["dtype"] == "float32" and row["bytes"] == 4
        assert row["min"] is None and row["nan"] is None
    assert any("holds JAX tracers" in n for n in table.notes)


def test_state_summary_on_an_uncompiled_graph_says_so():
    table = graph("single_uncompiled").state_summary(include_meta=True)
    assert {r["field"] for r in table} == {"position", "velocity"}
    assert table[0]["min"] is not None
    assert any("not compiled" in n for n in table.notes)
    assert any("no _meta" in n for n in table.notes)


def test_stats_read_bool_int_bfloat16_and_skip_prng_keys():
    import jax
    assert inspection._stats(jnp.asarray([True, False, True]))["mean"] == pytest.approx(2 / 3)
    assert inspection._stats(jnp.arange(4, dtype=jnp.int32))["max"] == 3
    bf = inspection._stats(jnp.asarray([1.0, jnp.nan], jnp.bfloat16))
    assert bf["min"] == 1.0 and bf["nan"] == 1
    assert inspection._stats(jax.random.key(0)) is None
    leaf = inspection._leaf(jax.random.split(jax.random.key(0), 3))
    assert leaf.shape == (3,) and leaf.nbytes == 24


# ----------------------------------------------------------------------
# params_table
# ----------------------------------------------------------------------

def test_params_table_has_a_row_per_leaf_with_its_spec():
    gm = graph("single")
    rows = {r["param"]: r for r in gm.params_table()}
    assert set(rows) == set(gm.params["nodes"]["s"])
    stiff = rows["stiffness"]
    spec = gm.param_specs()["nodes"]["s"]["stiffness"]
    assert stiff["value"] == 30.0 and stiff["shape"] == () and stiff["dtype"] == "float32"
    assert stiff["trainable"] is spec.trainable and stiff["bounds"] == spec.bounds
    assert stiff["transform"] == spec.transform and stiff["out_of_bounds"] is False
    assert rows["initial_position"]["trainable"] is False


def test_params_table_flags_a_value_outside_its_bounds():
    gm = build("single")
    gm.set_param_spec("s", "damping", ParamSpec(bounds=(0.0, 1.0)))
    rows = {r["param"]: r for r in gm.params_table()}
    assert rows["damping"]["bounds"] == (0.0, 1.0)
    assert rows["damping"]["out_of_bounds"] is True
    assert rows["damping"]["flags"] == ("out of bounds: above the upper bound 1",)
    gm.params["nodes"]["s"]["stiffness"] = jnp.float32(-1.0)      # log transform: must be > 0
    rows = {r["param"]: r for r in gm.params_table()}
    assert rows["stiffness"]["out_of_bounds"] is True
    gm.params["nodes"]["s"]["mass"] = jnp.float32(jnp.nan)
    assert {r["param"]: r for r in gm.params_table()}["mass"]["flags"] == (
        "out of bounds: not finite",)


def test_params_table_agrees_with_check_params_on_out_of_bounds():
    gm = build("single")
    for value, ok in ((5.0, True), (0.0, False), (-2.0, False)):
        gm.params["nodes"]["s"]["stiffness"] = jnp.float32(value)
        row = next(r for r in gm.params_table() if r["param"] == "stiffness")
        try:
            gm.check_params()
            passed = True
        except ValueError:
            passed = False
        assert passed is ok and row["out_of_bounds"] is (not ok), value


def test_params_table_reads_a_python_float_without_coercing_it():
    gm = build("single")
    gm.params["nodes"]["s"]["damping"] = 3.0
    row = next(r for r in gm.params_table() if r["param"] == "damping")
    assert row["value"] == 3.0
    assert type(gm.params["nodes"]["s"]["damping"]) is float


def test_params_table_lists_mapping_weights_as_frozen_arrays():
    rows = [r for r in graph("mapped").params_table() if r["section"] == "mappings"]
    assert len(rows) == 1
    (row,) = rows
    assert row["owner"] == "rod_a.temperature->rod_b.heat_source" and row["param"] == "H"
    assert row["value"] is None and row["shape"] == (12, 6) and row["trainable"] is False


def test_params_table_on_an_uncompiled_graph_shows_what_compile_would_take():
    gm = graph("single_uncompiled")
    table = gm.params_table()
    rows = {r["param"]: r for r in table}
    assert rows["stiffness"]["value"] == 30.0
    assert any("not compiled" in n for n in table.notes)
    assert gm.params == {"nodes": {}, "mappings": {}}


def test_params_table_on_a_traced_graph_reads_the_concrete_params():
    rows = graph("after_grad").params_table()
    assert all(r["value"] is not None for r in rows)


# ----------------------------------------------------------------------
# coupling_report
# ----------------------------------------------------------------------

def test_coupling_report_matches_coupling_diagnostics():
    gm = graph("coupled_diagnostics")
    diag = gm.coupling_diagnostics()["a+b"]
    (row,) = gm.coupling_report()
    for key in inspection._REPORT_KEYS:
        want, got = diag[key], row[key]
        if isinstance(want, float) and math.isnan(want):
            assert math.isnan(got), key
        else:
            assert got == want, key
    assert row["max_iterations"] == 8 and row["solver"] == "ift"
    assert not math.isnan(row["rho_spectral"])      # diagnostics=True computes it


def test_coupling_report_flags_the_cap_the_fallback_and_non_convergence():
    (row,) = graph("coupled_capped").coupling_report()
    flags = " | ".join(row["flags"])
    assert row["iterations"] == 1 and row["ratio_usable"] is False
    assert "hit max_iterations (1)" in flags
    assert "fell back to the raw residual test" in flags
    assert "converged=False" in flags


def test_coupling_report_flags_precision_limited():
    (row,) = graph("coupled").coupling_report()
    assert row["precision_limited"] is True
    assert any(f.startswith("precision_limited=True") for f in row["flags"])


def test_coupling_report_explains_a_missing_report():
    (row,) = graph("coupled_fori").coupling_report()
    assert row["iterations"] is None
    assert row["flags"] == ("no report: solver='fori' records diagnostics only with "
                            "diagnostics=True",)
    fresh = _fresh_pair()
    fresh.compile()
    (row,) = fresh.coupling_report()
    assert row["flags"] == ("no report yet: no step has run since compile() or reset_state()",)


def _fresh_pair() -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, initial_position=1.0))
    gm.add_node(SpringDamperNode("b", 0.01))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"])
    return gm


@pytest.mark.parametrize("kind, needle", [
    ("single", "no coupling groups"),
    ("multirate", "no coupling groups"),
    ("after_grad", "holds JAX tracers"),
])
def test_coupling_report_says_why_there_is_nothing_to_report(kind, needle):
    table = graph(kind).coupling_report()
    assert any(needle in n for n in table.notes)


def test_coupling_report_on_an_uncompiled_graph():
    table = _fresh_pair().coupling_report()
    assert len(table) == 1 and table[0]["iterations"] is None
    assert any("not compiled" in n for n in table.notes)


def test_coupling_report_text_prints_flags_visibly():
    text = str(graph("coupled_capped").coupling_report())
    assert "! hit max_iterations (1)" in text and "! ratio_usable=False" in text


# ----------------------------------------------------------------------
# memory_estimate
# ----------------------------------------------------------------------

@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_memory_estimate_adds_up_the_state_s_bytes(kind):
    gm = graph(kind)
    table = gm.memory_estimate()
    expected = {}
    for node, fields in gm._state.items():
        expected[node] = sum(int(np.prod(np.shape(v))) * jnp.dtype(v.dtype).itemsize
                             for v in fields.values())
    assert {r["node"]: r["bytes"] for r in table} == expected
    s = table.summary
    assert s["total_bytes"] == sum(expected.values())
    assert s["meta_bytes"] == expected.get("_meta", 0)
    assert s["state_bytes"] == s["total_bytes"] - s["meta_bytes"]
    assert any(n.startswith("state memory only") for n in table.notes)


def test_memory_estimate_has_no_meta_row_before_compile():
    table = graph("single_uncompiled").memory_estimate()
    assert [r["node"] for r in table] == ["s"]
    assert table.summary["meta_bytes"] == 0


# ----------------------------------------------------------------------
# InspectionTable
# ----------------------------------------------------------------------

def _table(**kwargs) -> InspectionTable:
    rows = [{"name": "alpha", "n": 1, "x": 0.5, "flags": ()},
            {"name": "beta", "n": 22, "x": None, "flags": ("look here",)}]
    return InspectionTable("Demo: 2 rows", ("name", "n", "x", "flags"), rows,
                           label_columns=("name",), **kwargs)


def test_table_is_a_sequence_of_row_dicts():
    t = _table()
    assert len(t) == 2 and t[0]["name"] == "alpha" and [r["n"] for r in t] == [1, 22]
    assert t[0:1] == ({"name": "alpha", "n": 1, "x": 0.5, "flags": ()},)
    assert t.columns == ("name", "n", "x") and t.title == "Demo: 2 rows"
    assert repr(t) == "InspectionTable('Demo: 2 rows', 2 rows)"


def test_table_text_grid_and_its_record_fallback():
    t = _table(notes=("a note",), summary={"total_bytes": 2048})
    assert t.to_text() == textwrap.dedent("""\
        Demo: 2 rows
          name    n    x
          -----  --  ---
          alpha   1  0.5
          beta   22    -

        flags:
          ! beta: look here

        total_bytes: 2048 (2.0 KiB)

        notes:
          - a note
        """)
    assert t.to_text(width=10) == textwrap.dedent("""\
        Demo: 2 rows
          alpha
              n 1, x 0.5
          beta
              n 22, x -
              ! look here

        total_bytes: 2048 (2.0 KiB)

        notes:
          - a note
        """)
    assert str(t) == t.to_text()


def test_table_print_writes_its_text():
    buf = io.StringIO()
    _table().print(buf, width=30)
    assert buf.getvalue() == _table().to_text(width=30)


# ----------------------------------------------------------------------
# Golden outputs on a small graph
# ----------------------------------------------------------------------

class Mass(SimulationNode):
    """Three masses on springs; ``k`` read from ``params``."""

    def initial_state(self):
        return {"x": jnp.asarray([0.0, 1.0, 2.0], jnp.float32),
                "v": jnp.zeros(3, jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        k = self.params["k"] if params is None else params["k"]
        force = boundary_inputs.get("force", 0.0)
        v = state["v"] + dt * (force - k * state["x"])
        return {"x": state["x"] + dt * v, "v": v}

    def param_specs(self):
        return {"k": ParamSpec(bounds=(0.0, 10.0))}


class Probe(SimulationNode):
    def initial_state(self):
        return {"reading": jnp.float32(0.0)}

    def update(self, state, boundary_inputs, dt):
        return {"reading": jnp.sum(boundary_inputs.get("signal", jnp.zeros(3)))}


def _golden() -> GraphManager:
    gm = GraphManager()
    gm.add_node(Mass("m1", 0.01, k=2.0))
    gm.add_node(Mass("m2", 0.01, k=3.0))
    gm.add_node(Probe("probe", 0.02))
    gm.add_edge("m1", "m2", "x", "force")
    gm.add_edge("m2", "m1", "x", "force", additive=True)
    gm.add_edge("m2", "probe", "v", "signal", transform="negate")
    gm.add_coupling_group(["m1", "m2"], max_iterations=5, tolerance=1e-5)
    gm.add_external_input("probe", "offset")
    gm.compile()
    return gm


GOLDEN_GRAPH = """\
GraphManager: 3 nodes, 3 edges, 1 coupling group, 1 external input
status: compiled
base timestep: 0.01 (multi-rate)

Nodes (3)
  m1  Mass
      timestep 0.01, rate divider 1
      coupling group m1+m2
      state: v float32[3], x float32[3]
  m2  Mass
      timestep 0.01, rate divider 1
      coupling group m1+m2
      state: v float32[3], x float32[3]
  probe  Probe
      timestep 0.02, rate divider 2
      state: reading float32[]

Edges (3)
  m1.x -> m2.force
      state; iterated inside its coupling group
  m2.x -> m1.force
      state; additive; iterated inside its coupling group
  m2.v -> probe.signal
      state; transform negate

Coupling groups (1)
  m1+m2
      members: m1, m2
      solver ift, acceleration none, mode gauss-seidel
      norm l2 (tolerance 1e-05), max_iterations 5
      diagnostics off, strict off

External inputs (1)
  probe.offset
      float32[]

Execution order
  1. [m1+m2] coupled block: m1, m2
      every base step
  2. probe
      every 2 base steps
"""

GOLDEN_MERMAID = """\
flowchart LR
    subgraph g0["coupling group m1+m2"]
        n0["m1<br/>Mass"]
        n1["m2<br/>Mass"]
    end
    n2["probe<br/>Probe"]
    n0 -->|"x→force"| n1
    n1 -->|"x→force (additive)"| n0
    n1 -->|"v→signal (negate)"| n2
    x0[/"external: offset"/]
    x0 -.-> n2
"""

GOLDEN_DOT = """\
digraph maddening {
    rankdir=LR;
    node [shape=box];
    subgraph cluster_g0 {
        label="coupling group m1+m2";
        style=rounded;
        n0 [label="m1\\nMass"];
        n1 [label="m2\\nMass"];
    }
    n2 [label="probe\\nProbe"];
    n0 -> n1 [label="x→force"];
    n1 -> n0 [label="x→force (additive)"];
    n1 -> n2 [label="v→signal (negate)"];
    x0 [label="external: offset", shape=parallelogram];
    x0 -> n2 [style=dashed];
}
"""

GOLDEN_STATE = """\
State summary: 5 fields in 3 nodes
  node   field    shape  dtype    min  max  mean  nan  inf  bytes
  -----  -------  -----  -------  ---  ---  ----  ---  ---  -----
  m1     v        (3,)   float32    0    0     0    0    0     12
  m1     x        (3,)   float32    0    2     1    0    0     12
  m2     v        (3,)   float32    0    0     0    0    0     12
  m2     x        (3,)   float32    0    2     1    0    0     12
  probe  reading  ()     float32    0    0     0    0    0      4

notes:
  - min, max and mean are taken over the finite entries; nan and inf count the rest
"""

GOLDEN_PARAMS = """\
Parameters: 2 leaves
  owner  param  value  shape  dtype    trainable  bounds   transform  out_of_bounds
  -----  -----  -----  -----  -------  ---------  -------  ---------  -------------
  m1     k          2  ()     float32  yes        (0, 10)  -          no
  m2     k          3  ()     float32  yes        (0, 10)  -          no

notes:
  - nodes whose update() takes no params keyword (constants baked into the step, not in this table):
    probe
  - value is the scalar for a one-element leaf; an array leaf shows its shape. out_of_bounds applies
    ParamSpec.check()'s rule (non-finite values count)
"""

GOLDEN_MEMORY = """\
State memory estimate: 68 B in 4 entries
  node   fields  bytes  per_device_bytes  devices  sharding
  -----  ------  -----  ----------------  -------  --------
  m1          2     24                24        1  -
  m2          2     24                24        1  -
  probe       1      4                 4        1  -
  _meta       4     16                16        1  -

state_bytes: 52 (52 B)
meta_bytes: 16 (16 B)
total_bytes: 68 (68 B)
total_per_device_bytes: 68 (68 B)

notes:
  - state memory only, computed from shapes and dtypes: XLA workspace, compiled programs, the copies
    a step makes, scan histories, params and external inputs are not included
  - bytes is a node's global (logical) size; per_device_bytes is what one device holds of it (a
    sharded field's shard, a replicated or unsharded field in full); total_per_device_bytes adds
    those, the most one device holds if every node's share lands on it
"""

GOLDEN_COUPLING = """\
Coupling report: 1 group
  m1+m2
      iterations -, total_iterations -, max_iterations 5, converged -, residual -, error_estimate -,
        amplification -, ratio_usable -, precision_limited -, rho_spectral -,
        spectral_error_bound -, spectral_usable -
      ! no report yet: no step has run since compile() or reset_state()
"""


@pytest.fixture(scope="module")
def golden():
    return _golden()


def test_golden_graph_text(golden):
    assert golden.format_graph() == GOLDEN_GRAPH


def test_golden_mermaid(golden):
    assert golden.to_mermaid() == GOLDEN_MERMAID


def test_golden_dot(golden):
    assert golden.to_dot() == GOLDEN_DOT


def test_golden_state_summary(golden):
    assert str(golden.state_summary()) == GOLDEN_STATE


def test_golden_params_table(golden):
    assert str(golden.params_table()) == GOLDEN_PARAMS


def test_golden_memory_estimate(golden):
    assert str(golden.memory_estimate()) == GOLDEN_MEMORY


def test_golden_coupling_report_before_the_first_step(golden):
    assert str(golden.coupling_report()) == GOLDEN_COUPLING


def test_coupling_report_judges_a_replaced_group_under_the_one_that_ran():
    """Until the next compile the report is the last step's, judged (and
    capped) under the group that step ran, as coupling_diagnostics() does."""
    gm = build("coupled")
    gm.remove_coupling_group(["a", "b"])
    gm.add_coupling_group(["a", "b"], max_iterations=3, tolerance=1e-6)
    (row,) = gm.coupling_report()
    assert row["max_iterations"] == 8
    assert row["iterations"] == gm.coupling_diagnostics()["a+b"]["iterations"]
    assert any("modified since the last compile" in n for n in gm.coupling_report().notes)


def test_rich_tables_stay_readable_when_wide():
    """A table too wide for a grid renders one key/value table per row in
    ``rich`` too, with flags printed whole beneath it."""
    pytest.importorskip("rich", reason="rich is an optional extra (maddening[terminal])")
    buf = io.StringIO()
    graph("coupled_capped").print_coupling_report(file=buf, rich=True, width=90)
    out = buf.getvalue()
    assert "spectral_error_bound" in out and "…" not in out
    assert "! hit max_iterations (1): the solve stopped on its budget" in out
    buf = io.StringIO()
    graph("coupled").print_memory_estimate(file=buf, rich=True, width=90)
    assert "per_device_bytes" in buf.getvalue() and "total_bytes: 28" in buf.getvalue()
