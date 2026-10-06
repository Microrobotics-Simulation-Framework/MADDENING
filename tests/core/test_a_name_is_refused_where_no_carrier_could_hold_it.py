"""A name the graph takes is one every place a name is written can hold.

A node's name, and the fields an edge or an external input names, are
written to a checkpoint's member names, to the config (``to_dict``), to a
USD stage, to an FMU's model description (XML 1.0) and to log lines.  The
graph used to take any text but ``/``, ``#``, ``->`` and its reserved
names, and three of those places then failed, each without a word at the
moment of writing:

* a node named with a NUL: its checkpoint was written and did not load
  (a member's name ends at the NUL), and a USD stage reloaded the node
  under the name's first half;
* a control character XML 1.0 cannot spell (U+0001, U+001F, ...: all of
  U+0000 to U+001F but tab, line feed and carriage return) or U+FFFE /
  U+FFFF: the model description was written and no parser read it;
* a surrogate: the graph could not compile, and no file could hold it.

And an edge's field names were not asked anything at all:

* a target field named ``NaN``, ``Infinity`` or ``-Infinity`` -- how the
  config writes a non-finite float -- was taken, and ``to_dict`` raised for
  as long as the edge was there;
* a ``#`` in a field's name, which ends an edge's key and starts a mapped
  edge's ordinal: two mapped edges on one field pair got one key and one
  slot of weights between them, and ``remove_edge`` left the slot behind.

``add_node``, ``add_edge`` and ``add_external_input`` refuse each where the
name is introduced, and what they take is shown to reload from every
carrier (the USD stage in ``tests/usd/test_usd_node_names_a_stage_carries.py``,
which the job that installs ``usd-core`` runs).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.mapping import StaticLinearMapping
from maddening.core.graph_manager import GraphManager
from maddening.core._graph_specs import (
    _field_name_refusal,
    _node_name_refusal,
    _uncarriable_characters,
)
from maddening.nodes import BallNode, HeatNode, SpringDamperNode
from maddening.serialization.json_codec import NON_FINITE_TOKENS

REGISTRY = {"BallNode": BallNode, "SpringDamperNode": SpringDamperNode,
            "HeatNode": HeatNode}

#: What some carrier cannot hold, by what it is.
REFUSED = {
    "a NUL": "\x00", "U+0001": "\x01", "a backspace": "\x08", "a vertical tab": "\x0b",
    "an escape": "\x1b", "U+001F": "\x1f", "a high surrogate": "\ud800",
    "a low surrogate": "\udfff", "U+FFFE": "\ufffe", "U+FFFF": "\uffff",
}
#: Names every carrier holds: lookalikes of what is refused, the
#: characters an edge's key is made with (but ``#``), and other scripts.
ACCEPTED = ["a b", "a.b", "a:b", 'a"b', "a\\b", "a<&>b", "a%b", "a$b{}~=", "é", "名",
            "😀", "a\u2028b", "a\u00a0b", "a\u200bb", "a\ufeffb", "a\ufffdb", "nan",
            "x00", "meta",
            # Unusual, and carried by every one of them: a name is not
            # refused for that.  The two with a line break are the names
            # the diagram and chart tests draw (test_inspection_diagram.py,
            # test_inspection_views.py).
            "a\tb", "two\nlines", 'we"ird <b>&amp; `tick` naïve\nsecond line', "a\rb",
            "a\r\nb", "a\x7fb", "a\x85b", "a\x9fb"]


def _names(char: str) -> list[str]:
    """*char* at the start, in the middle and at the end of a name."""
    return [char + "b", "a" + char + "b", "a" + char]


# ---------------------------------------------------------------------------
# The alphabet
# ---------------------------------------------------------------------------

def test_what_a_name_cannot_hold_is_what_xml_1_0_has_no_character_for():
    """Against the ``Char`` production of XML 1.0, the narrowest carrier:
    ``#x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]``
    is taken and nothing else is refused -- not a tab or a line break, not
    U+007F to U+009F.  Every code point of the Basic Multilingual Plane,
    and one in 257 of the other planes with their first and last."""
    points = list(range(0x10000)) + [p for plane in range(1, 17) for p in (
        *range(plane << 16, (plane + 1) << 16, 257), (plane << 16) + 0xFFFE,
        (plane << 16) + 0xFFFF)]
    wrong = []
    for point in points:
        char = chr(point)
        xml_char = (point in (0x9, 0xA, 0xD) or 0x20 <= point <= 0xD7FF
                    or 0xE000 <= point <= 0xFFFD or 0x10000 <= point <= 0x10FFFF)
        if bool(_uncarriable_characters(char)) == xml_char:
            wrong.append(hex(point))
    assert not wrong, wrong[:20]


def test_every_character_a_name_may_hold_is_one_xml_and_utf8_carry():
    """The two carriers that decide the set, asked directly: each character
    a name may hold (the whole Basic Multilingual Plane, and the ends of
    the other planes) is written to an XML attribute, encoded as UTF-8,
    parsed, and read back as itself."""
    points = list(range(0x10000)) + [p for plane in range(1, 17)
                                     for p in (plane << 16, (plane << 16) + 0xFFFF)]
    taken = [chr(p) for p in points if not _uncarriable_characters(chr(p))]
    assert len(taken) > 63_000
    root = ET.Element("names")
    for start in range(0, len(taken), 512):
        ET.SubElement(root, "n", attrib={"v": "".join(taken[start:start + 512])})
    read = ET.fromstring(ET.tostring(root, encoding="unicode").encode("utf-8"))
    assert "".join(el.get("v") for el in read) == "".join(taken)


@pytest.mark.parametrize("char", list(REFUSED.values()), ids=lambda c: f"U+{ord(c):04X}")
def test_what_is_refused_is_what_xml_or_utf8_cannot_carry(char):
    """The other direction: each refused character is one XML 1.0 or UTF-8
    has no way to write.  A document holding it is not read back."""
    root = ET.Element("names", attrib={"v": f"a{char}b"})
    with pytest.raises((ET.ParseError, UnicodeEncodeError)):
        ET.fromstring(ET.tostring(root, encoding="unicode").encode("utf-8"))
    assert _uncarriable_characters(char)


def test_the_refusal_names_each_code_point_once_in_the_order_met():
    assert _uncarriable_characters("a\x01b\x00c\x01\ud800") == ["U+0001", "U+0000", "U+D800"]
    assert _uncarriable_characters("a\tb\nc\rd\x7fe\x85f\x9f") == []
    assert _uncarriable_characters("") == []
    assert _uncarriable_characters("a b.c->d/e#f") == []


# ---------------------------------------------------------------------------
# A node's name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("what", sorted(REFUSED))
def test_add_node_refuses_a_name_some_carrier_cannot_hold(what):
    char = REFUSED[what]
    for name in _names(char):
        gm = GraphManager()
        with pytest.raises(ValueError, match="is invalid") as refused:
            gm.add_node(BallNode(name, 0.01))
        said = str(refused.value)
        assert f"U+{ord(char):04X}" in said and "must not contain U+0000 to U+001F" in said
        # The message is itself text every reader can carry.
        assert not _uncarriable_characters(said), said
        said.encode("utf-8")
        assert not gm._nodes and not gm._state, name   # noqa: SLF001


def test_a_name_that_is_not_text_is_refused_as_a_name():
    gm = GraphManager()
    with pytest.raises(ValueError, match="a name is a string, not int"):
        gm.add_node(BallNode(5, 0.01))   # pyright: ignore[reportArgumentType]
    assert not gm._nodes   # noqa: SLF001


def test_the_names_refused_before_are_refused_in_the_same_words():
    """One rule, asked by ``add_node`` and by ``POST /graph/nodes``: the
    separators, the empty name, the graph's own keys and the non-finite
    tokens keep the words they had."""
    assert "must be non-empty and must not contain ['/']" in _node_name_refusal("a/b")
    assert "must be non-empty and must not contain ['/', '#', '->']" in _node_name_refusal("")
    assert "reserves for its own state and checkpoints" in _node_name_refusal("_meta")
    assert "it spells a non-finite JSON token" in _node_name_refusal("NaN")
    for name in ACCEPTED:
        assert _node_name_refusal(name) is None, name


def _graph_of(names: list[str]) -> GraphManager:
    gm = GraphManager()
    for i, name in enumerate(names):
        gm.add_node(BallNode(name, 0.01, initial_position=1.0 + i))
    gm.add_node(SpringDamperNode("spring", 0.01))
    gm.add_edge(names[1], "spring", "position", "anchor_position")
    return gm


@pytest.fixture(scope="module")
def stepped():
    gm = _graph_of(ACCEPTED)
    gm.compile()
    gm.step()
    return gm


def test_the_names_add_node_takes_reload_from_a_checkpoint(stepped, tmp_path):
    path = stepped.save_state(tmp_path / "names.npz")
    again = _graph_of(ACCEPTED)
    again.compile()
    again.load_state(path)
    for name in ACCEPTED:
        for field, value in stepped._state[name].items():   # noqa: SLF001
            np.testing.assert_array_equal(np.asarray(again._state[name][field]),   # noqa: SLF001
                                          np.asarray(value), err_msg=f"{name!r}.{field}")


def test_the_names_add_node_takes_reload_from_a_config_file(stepped, tmp_path):
    """Written as UTF-8 without escapes -- the strictest way to write one."""
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(stepped.to_dict(), ensure_ascii=False), encoding="utf-8")
    again = GraphManager.from_dict(json.loads(path.read_text(encoding="utf-8")), REGISTRY)
    assert list(again._nodes) == list(stepped._nodes)   # noqa: SLF001
    assert [e.key for e in again._edges] == [e.key for e in stepped._edges]   # noqa: SLF001
    again.compile()
    again.step()


def test_the_names_add_node_takes_are_read_back_from_an_fmu_model_description(stepped):
    from maddening.fmi.model_description import build_model_description

    description = build_model_description(stepped, model_name="names", include_evolving=True)
    root = ET.fromstring(description.to_xml().encode("utf-8"))
    variables = {el.get("name") for el in root.find("ModelVariables")}
    for name in ACCEPTED:
        assert f"{name}.position" in variables, name


def test_from_dict_refuses_a_config_naming_what_the_graph_could_not_save_again():
    """``from_dict`` adds through ``add_node`` and ``add_edge``: a config
    written by hand comes in by the same doors."""
    config = json.loads(json.dumps(_graph_of(["a", "b"]).to_dict()))
    bad_node = json.loads(json.dumps(config))
    bad_node["nodes"][0]["name"] = "a\x00b"
    with pytest.raises(ValueError, match="U\\+0000"):
        GraphManager.from_dict(bad_node, REGISTRY)
    bad_edge = json.loads(json.dumps(config))
    bad_edge["edges"][0]["target_field"] = "a#b"
    with pytest.raises(ValueError, match="'#'"):
        GraphManager.from_dict(bad_edge, REGISTRY)
    GraphManager.from_dict(config, REGISTRY)


# ---------------------------------------------------------------------------
# An edge's fields, and an external input's
# ---------------------------------------------------------------------------

BAD_FIELDS = sorted(NON_FINITE_TOKENS) + ["a#b", "#", "a\x00b", "a\x1bb", "a\ud800b", "a\ufffeb"]
GOOD_FIELDS = ["anchor_position", "undeclared", "", "nan", "infinity", "a.b", "a->b", "a/b",
               "a b", "é", "two\nlines", "a\tb"]


def _pair() -> GraphManager:
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=3.0))
    gm.add_node(SpringDamperNode("spring", 0.01))
    return gm


@pytest.mark.parametrize("which", ["source_field", "target_field"])
@pytest.mark.parametrize("bad", BAD_FIELDS, ids=[repr(b).encode("ascii", "replace").decode()
                                                 for b in BAD_FIELDS])
def test_add_edge_refuses_a_field_name_the_config_or_an_edge_key_cannot_hold(which, bad):
    gm = _pair()
    fields = {"source_field": "position", "target_field": "anchor_position", which: bad}
    with pytest.raises(ValueError, match="is invalid") as refused:
        gm.add_edge("ball", "spring", **fields)
    said = str(refused.value)
    assert which.split("_")[0] + " field" in said and repr(bad) in said
    assert not _uncarriable_characters(said), said
    assert not gm._edges   # noqa: SLF001
    assert gm.to_dict()["edges"] == []
    assert _field_name_refusal(bad) is not None


@pytest.mark.parametrize("target_field", GOOD_FIELDS)
def test_add_edge_takes_any_other_target_field_and_the_graph_is_written_and_reloads(target_field):
    """Also one the target node does not declare: ``validate`` passes it,
    because a node may read an input it does not declare.  The config is
    written, reloads with the same edge, and the graph steps."""
    gm = _pair()
    gm.add_edge("ball", "spring", "position", target_field)
    assert _field_name_refusal(target_field) is None
    config = json.loads(json.dumps(gm.to_dict(), ensure_ascii=False))
    again = GraphManager.from_dict(config, REGISTRY)
    assert [(e.source_field, e.target_field) for e in again._edges] == [   # noqa: SLF001
        ("position", target_field)]
    assert not [issue for issue in gm.validate() if issue.startswith("ERROR")]
    gm.compile()
    gm.step()


def _mapped_rods(target_field: str) -> GraphManager:
    gm = GraphManager()
    for name in ("a", "b"):
        gm.add_node(HeatNode(name, 0.01, n_cells=4, length=1.0, thermal_diffusivity=0.01))
    gm.add_edge("a", "b", "temperature", target_field, additive=True,
                mapping=StaticLinearMapping(jnp.eye(4, dtype=jnp.float32)))
    gm.add_edge("a", "b", "temperature", target_field, additive=True,
                mapping=StaticLinearMapping(2.0 * jnp.eye(4, dtype=jnp.float32)))
    return gm


def test_two_mapped_edges_on_one_field_pair_keep_a_slot_each_and_a_hash_cannot_merge_them():
    """What the ``#`` rule protects.  With ``s#x`` as the target field the
    two edges below got the same key (the ordinal is read after the first
    ``#``) and so one slot of weights between them, and ``remove_edge``
    left that slot in ``params["mappings"]``."""
    gm = _mapped_rods("source")
    keys = [e.key for e in gm._edges]   # noqa: SLF001
    assert len(set(keys)) == 2 and keys[1] == keys[0] + "#1"
    gm.compile()
    assert sorted(gm.params["mappings"]) == sorted(keys)
    scales = sorted(float(np.asarray(next(iter(gm.params["mappings"][key].values())))[0, 0])
                    for key in keys)
    assert scales == [1.0, 2.0]
    gm.remove_edge("a", "b", "temperature", "source")
    assert not gm._edges and not gm.params.get("mappings")   # noqa: SLF001
    with pytest.raises(ValueError, match="'#'"):
        _mapped_rods("s#x")


@pytest.mark.parametrize("bad", BAD_FIELDS, ids=[repr(b).encode("ascii", "replace").decode()
                                                 for b in BAD_FIELDS])
def test_add_external_input_refuses_the_field_names_an_edge_refuses(bad):
    gm = _pair()
    with pytest.raises(ValueError, match="is invalid") as refused:
        gm.add_external_input("spring", bad)
    assert repr(bad) in str(refused.value)
    assert not gm._external_inputs   # noqa: SLF001
    assert gm.to_dict()["external_inputs"] == []
    gm.add_external_input("spring", "anchor_position")
    assert len(gm.to_dict()["external_inputs"]) == 1
