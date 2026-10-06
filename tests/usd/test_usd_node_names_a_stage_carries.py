"""A node's name is restored from a USD stage as it was, and a stage cannot
bring in a name the graph refuses.

``save_graph_to_usd`` mangles a node's name into a prim name and writes the
name itself to ``maddening:nodeName``.  A name with a NUL in it was written
and reloaded as its first half (the string ends at the NUL), so the stage
reloaded a different graph without a word.  ``GraphManager.add_node`` now
refuses a name holding a control character, a surrogate or U+FFFE / U+FFFF
(``tests/core/test_a_name_is_refused_where_no_carrier_could_hold_it.py``);
this file holds the stage to the other half: every name ``add_node`` takes
comes back from a stage, and a stage edited to hold a refused name is
refused when it is loaded.
"""

from __future__ import annotations

import pytest

pxr = pytest.importorskip("pxr", reason="usd-core not installed")

from maddening.core.graph_manager import GraphManager  # noqa: E402
from maddening.nodes import BallNode, SpringDamperNode  # noqa: E402
from maddening.usd.serialization import load_graph_from_usd, save_graph_to_usd  # noqa: E402

REGISTRY = {"BallNode": BallNode, "SpringDamperNode": SpringDamperNode}
#: The names of the core file's ``ACCEPTED``: lookalikes of what is refused,
#: the characters an edge's key is made with (but ``#``), other scripts.
ACCEPTED = ["a b", "a.b", "a:b", 'a"b', "a\\b", "a<&>b", "a%b", "a$b{}~=", "é", "名",
            "😀", "a\u2028b", "a\u00a0b", "a\u200bb", "a\ufeffb", "a\ufffdb", "nan",
            "x00", "meta"]


def _graph() -> GraphManager:
    gm = GraphManager()
    for i, name in enumerate(ACCEPTED):
        gm.add_node(BallNode(name, 0.01, initial_position=1.0 + i))
    gm.add_node(SpringDamperNode("spring", 0.01))
    gm.add_edge(ACCEPTED[1], "spring", "position", "anchor_position")
    return gm


def _through_text(stage):
    """The stage as its ``.usda`` text reads back: what a file holds."""
    reread = pxr.Usd.Stage.CreateInMemory()
    reread.GetRootLayer().ImportFromString(stage.GetRootLayer().ExportToString())
    return reread


def test_the_names_add_node_takes_reload_from_a_usd_stage():
    gm = _graph()
    stage = pxr.Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    again = load_graph_from_usd(_through_text(stage), node_registry=REGISTRY)
    assert sorted(again._nodes) == sorted(gm._nodes)   # noqa: SLF001
    assert [(e.source_node, e.target_node) for e in again._edges] == [   # noqa: SLF001
        (ACCEPTED[1], "spring")]


@pytest.mark.parametrize("bad", ["a\nb", "a\x1bb", "a\x7fb", "a\ufffeb", "NaN", "_meta", "a/b"],
                         ids=["a line feed", "an escape", "a delete", "U+FFFE",
                              "a non-finite token", "a reserved key", "a separator"])
def test_a_stage_that_names_a_node_the_graph_refuses_is_refused_when_it_is_loaded(bad):
    """The stage is a door like any other: ``load_graph_from_usd`` adds
    through ``add_node``, so a ``maddening:nodeName`` written by hand to a
    name the graph could not save again is its ``ValueError``."""
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01))
    stage = pxr.Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    named = [prim.GetAttribute("maddening:nodeName") for prim in stage.Traverse()
             if prim.GetAttribute("maddening:nodeName").IsValid()]
    assert len(named) == 1 and named[0].Get() == "ball"
    named[0].Set(bad)
    with pytest.raises(ValueError, match="is invalid"):
        load_graph_from_usd(stage, node_registry=REGISTRY)
    named[0].Set("ball again")
    assert list(load_graph_from_usd(stage, node_registry=REGISTRY)._nodes) == [   # noqa: SLF001
        "ball again"]
