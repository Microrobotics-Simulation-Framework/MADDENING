"""``GraphManager.print_graph_diagram``: the graph drawn as boxes and arrows.

The drawing is termaid's, from a flowchart in the dialect termaid parses
(``inspection._termaid_mermaid``).  This file checks that the drawing shows
what the graph holds (names, types, the coupling group's title, the edge
labels, the external input), that the dialect is ``to_mermaid``'s
flowchart with only the two changes termaid needs, that every theme
colours and a bad one is refused, that names termaid's parser would
choke on still draw, and that a missing ``termaid`` (or ``rich``, for a
theme) is a clear ``ImportError``.  That the method changes nothing on
every kind of graph is ``test_inspection_read_only.py``
(``INSPECTION_CALLS``); the checks here that compare the graph before and
after name the attributes a reader would look at first.

``termaid`` and ``rich`` are the optional ``terminal`` extra.  CI installs
both through the ``ci`` extra, so every test here runs there; the tests
that hide a package by patching the import run everywhere.
"""

from __future__ import annotations

import builtins
import copy
import io
import os
import re

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core import inspection
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.heat import HeatNode
from tests.core.inspection_graphs import BUILDERS, build, graph
from tests.core.inspection_guard_support import HAS_RICH, HAS_TERMAID, compile_events

needs_termaid = pytest.mark.skipif(
    not HAS_TERMAID, reason="termaid is an optional extra (maddening[terminal]); "
                            "CI installs it through the ci extra")
needs_rich = pytest.mark.skipif(
    not HAS_RICH, reason="rich is an optional extra (maddening[terminal]); "
                         "CI installs it through the ci extra")

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture
def plain_env(monkeypatch):
    """No environment variable that makes rich colour a non-terminal file."""
    for var in ("FORCE_COLOR", "TTY_COMPATIBLE", "NO_COLOR"):
        monkeypatch.delenv(var, raising=False)


def _hide(monkeypatch, *packages: str) -> None:
    """Make ``import <package>`` (and its submodules) fail, as if absent."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if any(name == p or name.startswith(p + ".") for p in packages):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _draw(gm: GraphManager, **kwargs) -> str:
    buf = io.StringIO()
    gm.print_graph_diagram(file=buf, **kwargs)
    return buf.getvalue()


def _rods() -> GraphManager:
    """A coupled pair joined by a flux edge and a state edge, a third node
    downstream on a plain edge, and an external input.  Never compiled."""
    gm = GraphManager()
    for name, t0 in (("rod_a", 300.0), ("rod_b", 350.0), ("rod_c", 320.0)):
        gm.add_node(HeatNode(name, 1e-4, n_cells=4, thermal_diffusivity=0.1,
                             initial_temperature=t0))
    gm.add_edge("rod_a", "rod_b", "right_heat_flux", "left_temperature", transform="negate")
    gm.add_edge("rod_b", "rod_a", "temperature", "heat_source")
    gm.add_edge("rod_b", "rod_c", "temperature", "heat_source")
    gm.add_coupling_group(["rod_a", "rod_b"], max_iterations=5, tolerance=1e-5)
    gm.add_external_input("rod_a", "left_temperature")
    return gm


# ----------------------------------------------------------------------
# What the drawing shows
# ----------------------------------------------------------------------

@needs_termaid
def test_drawing_shows_names_types_the_group_title_edges_and_the_external_input(plain_env):
    out = _draw(_rods())
    for text in ("rod_a :: HeatNode", "rod_b :: HeatNode", "rod_c :: HeatNode",
                 "coupling group rod_a+rod_b",
                 "right_heat_flux→left_temperature (negate)", "temperature→heat_source",
                 "external:", "left_temperature"):
        assert text in out, (text, out)
    # The Mermaid syntax termaid would print verbatim if it failed to parse it.
    for raw in ("g0[", "subgraph", '["', "<br/>", "n0", "x0", "-->", "-.->"):
        assert raw not in out, (raw, out)
    # A box per node and the external input's parallelogram.
    assert out.count("┌") >= 4 and "/──" in out
    # Dotted lines are drawn: the flux edge and the external input's edge.
    assert "┄" in out


def test_termaid_dialect_marks_the_flux_edge_dotted_and_the_state_edges_solid():
    src = inspection._termaid_mermaid(_rods())
    assert re.search(r'n0 -\.->\|"right_heat_flux→left_temperature \(negate\)"\| n1', src)
    assert 'n1 -->|"temperature→heat_source"| n0' in src
    assert 'n1 -->|"temperature→heat_source"| n2' in src
    assert 'x0[/"external: left_temperature"/]' in src and "x0 -.-> n0" in src
    assert 'subgraph g0 ["coupling group rod_a+rod_b"]' in src


_ENTITIES = (("#quot;", '"'), ("#lt;", "<"), ("#gt;", ">"), ("#38;", "&"),
             ("#96;", "`"), ("#35;", "#"))


#: The kinds that differ in structure (none, one node uncompiled, a node
#: added since the compile, a coupled pair, a sub-cycled pair, a mapping
#: with transforms, a flux edge and an external input).  The rest differ in
#: state or solver settings, which neither export draws, and cost seconds
#: to build.
_STRUCTURAL_KINDS = ("empty", "single_uncompiled", "stale", "coupled", "subcycled", "mapped")


@pytest.mark.parametrize("kind", _STRUCTURAL_KINDS)
def test_termaid_dialect_is_to_mermaid_with_only_the_two_changes_termaid_needs(kind):
    """Same nodes, groups, edges, inputs and ids, in the same order.  The
    differences are the subgraph-title form, the name/type separator and
    escaping (none here: termaid decodes no entity codes)."""
    assert kind in BUILDERS
    gm = graph(kind)
    expected = re.sub(r"^(    subgraph g\d+)\[", r"\1 [", gm.to_mermaid(), flags=re.M)
    expected = expected.replace("<br/>", " :: ")
    for code, char in _ENTITIES:
        expected = expected.replace(code, char)
    assert inspection._termaid_mermaid(gm) == expected


@needs_termaid
def test_drawing_reads_the_same_with_or_without_rich_on_a_non_terminal(plain_env, monkeypatch):
    gm = _rods()
    with_rich = _draw(gm)
    _hide(monkeypatch, "rich")
    without = _draw(gm)
    assert with_rich == without
    # The drawing is wider than rich's 80-column default: rich must not
    # wrap or crop it (soft_wrap), or the two would differ.
    assert max(len(line) for line in without.splitlines()) > 80


@needs_termaid
def test_direction_and_ascii(plain_env):
    gm = _rods()
    lr, tb = _draw(gm), _draw(gm, direction="TB")
    assert lr != tb and "rod_c :: HeatNode" in tb
    ascii_out = _draw(gm, use_ascii=True)
    assert ascii_out.isascii(), sorted({c for c in ascii_out if not c.isascii()})
    assert "right_heat_flux->left_temperature (negate)" in ascii_out
    with pytest.raises(ValueError, match="direction"):
        _draw(gm, direction="sideways")


@needs_termaid
def test_an_empty_graph_says_so():
    assert _draw(GraphManager()) == "(empty graph: no nodes to draw)\n"


class Tiny(SimulationNode):
    def initial_state(self):
        return {"x": jnp.float32(0.0)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] + dt}


#: Names carrying each sequence termaid's parser acts on inside a quoted
#: label, and the look-alike the drawing shows instead.
_AWKWARD = {
    'we"ird': "we'ird",                 # a quote would end the label
    "50%%done": "50% %done",            # %% would start a comment
    "a:::b": "a: ::b",                  # ::: would start a class suffix
    "`tick`": "'tick'",                 # "`...`" would be a Markdown label
    "C:\\new": "C:\\ new",              # \n would break the label in two
    "two\nlines": "two lines",          # a line break would end the statement
    "[bold]x :ok:": "[bold]x :ok:",     # rich markup and emoji stay literal
    "a;b & c": "a;b & c",               # ; and & are safe inside quotes
}


@needs_termaid
@pytest.mark.parametrize("use_rich", [False, True], ids=["plain", "rich"])
def test_awkward_names_draw_as_look_alikes_and_never_break_the_drawing(
        use_rich, plain_env, monkeypatch):
    if use_rich and not HAS_RICH:
        pytest.skip("rich is an optional extra (maddening[terminal]); "
                    "CI installs it through the ci extra")
    gm = GraphManager()
    names = list(_AWKWARD)
    for name in names:
        gm.add_node(Tiny(name, 0.01))
    for src, dst in zip(names, names[1:]):
        gm.add_edge(src, dst, "x", "x_in")
    gm.add_coupling_group(names[:2])
    if not use_rich:
        _hide(monkeypatch, "rich")
    out = _draw(gm)
    for name, shown in _AWKWARD.items():
        assert f"{shown} :: Tiny" in out, (name, shown, out)
    assert "coupling group 50% %done+we'ird" in out
    assert "x→x_in" in out
    assert "g0[" not in out and '["' not in out


def test_look_alikes_leave_no_sequence_termaid_acts_on():
    for name in _AWKWARD:
        text = inspection._termaid_text(name * 3)
        for bad in ('"', "`", "%%", ":::", "\\n", "\n", "\r"):
            assert bad not in text, (name, bad, text)


# ----------------------------------------------------------------------
# Themes
# ----------------------------------------------------------------------

@needs_termaid
def test_theme_list_is_termaid_s():
    from termaid.renderer.themes import THEMES  # noqa: PLC0415
    assert inspection._DIAGRAM_THEMES == tuple(THEMES)


@needs_termaid
@needs_rich
@pytest.mark.parametrize("theme", inspection._DIAGRAM_THEMES)
def test_every_theme_colours_the_same_drawing(theme, plain_env, monkeypatch):
    gm = _rods()
    plain = _draw(gm)
    monkeypatch.setenv("FORCE_COLOR", "1")          # rich treats the file as a terminal
    coloured = _draw(gm, theme=theme)
    assert _ANSI.search(coloured), f"theme {theme!r} wrote no colour"
    assert _ANSI.sub("", coloured) == plain
    if theme != "default":
        assert coloured != _draw(gm, theme="default"), f"theme {theme!r} is the default's"


@needs_termaid
def test_an_unknown_theme_is_refused_not_silently_replaced():
    with pytest.raises(ValueError, match=r"theme must be None or one of .*'phosphor'"):
        _draw(_rods(), theme="momo")


@needs_termaid
def test_a_theme_without_rich_names_the_extra_and_no_theme_still_draws(plain_env, monkeypatch):
    gm = _rods()
    _hide(monkeypatch, "rich")
    with pytest.raises(ImportError, match=r'pip install "maddening\[terminal\]"'):
        _draw(gm, theme="amber")
    assert "rod_a :: HeatNode" in _draw(gm)


# ----------------------------------------------------------------------
# termaid missing
# ----------------------------------------------------------------------

def test_missing_termaid_names_the_extra(monkeypatch):
    gm = _rods()
    before = gm.to_dict()
    _hide(monkeypatch, "termaid")
    with pytest.raises(ImportError, match=r'needs the optional \'termaid\' package: '
                                          r'pip install "maddening\[terminal\]"'):
        gm.print_graph_diagram(file=io.StringIO())
    # Refused before drawing anything, and the rest of the inspection API
    # needs nothing.
    assert gm.to_dict() == before
    gm.print_graph(file=io.StringIO())
    gm.to_mermaid()


# ----------------------------------------------------------------------
# Read-only
# ----------------------------------------------------------------------

def _leaves(tree):
    return [np.asarray(leaf).copy() for leaf in jax.tree_util.tree_leaves(tree)]


def _snapshot(gm: GraphManager) -> dict:
    # strict_mappings=False: the "mapped" kind's explicit matrix has no
    # recorded point sets, which only a config writer needs.
    return {"to_dict": copy.deepcopy(gm.to_dict(strict_mappings=False)),
            "params_tree": jax.tree_util.tree_structure(gm.params),
            "params": _leaves(gm.params),
            "state": _leaves(gm._state),                      # noqa: SLF001
            "dirty": gm._dirty,                               # noqa: SLF001
            "generation": gm._compile_generation,             # noqa: SLF001
            "compiled_step": gm._compiled_step,               # noqa: SLF001
            "trace_count": gm.trace_count,
            "scan_trace_count": gm.scan_trace_count}


def _assert_same(before: dict, after: dict) -> None:
    for key in before:
        b, a = before[key], after[key]
        if key in ("params", "state"):
            assert len(a) == len(b) and all(
                x.dtype == y.dtype and x.shape == y.shape and np.array_equal(x, y, equal_nan=True)
                for x, y in zip(a, b)), key
        elif key == "compiled_step":
            assert a is b, key
        else:
            assert a == b, key


@needs_termaid
@pytest.mark.parametrize("kind", ["coupled", "mapped", "stale"])
def test_drawing_a_graph_changes_nothing_on_it(kind):
    gm = build(kind)
    before = _snapshot(gm)
    with compile_events() as events:
        _draw(gm)
        if HAS_RICH:
            _draw(gm, theme="terra", direction="BT", use_ascii=True)
    _assert_same(before, _snapshot(gm))
    assert not events, sorted(set(events))


@needs_termaid
def test_drawing_does_not_compile_an_uncompiled_graph(monkeypatch):
    gm = _rods()
    assert gm._compiled_step is None and gm._dirty            # noqa: SLF001
    before = _snapshot(gm)
    calls = []
    monkeypatch.setattr(gm, "compile", lambda *a, **k: calls.append("compile"))
    with compile_events() as events:
        _draw(gm)
    assert not calls and not events
    _assert_same(before, _snapshot(gm))
    assert gm._compiled_step is None and gm._dirty            # noqa: SLF001
    assert gm.params == {"nodes": {}, "mappings": {}}
