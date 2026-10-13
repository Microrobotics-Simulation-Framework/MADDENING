"""A name or a text the graph could not save again is a 4xx at the request
that introduces it, and nothing is changed.

Three requests were each answered 2xx (or 500 after the change was made)
and left a server that could not report, save or restore its graph:

* ``POST /graph/nodes`` with a NUL in the name: 201; ``POST
  /checkpoint/save`` then 200, and ``POST /checkpoint/load`` of that file
  400.  With a surrogate in the name: 500, *and the node was added* --
  ``GET /graph`` and ``POST /graph/compile`` were then 500 and no URL can
  spell the name to delete it.
* ``POST /graph/edges`` with the target field ``NaN``: 201, and ``GET
  /graph`` then 500 until the edge was deleted.
* ``POST /graph/nodes`` of a registered class with a text parameter given
  ``"NaN"``: 500, the node added, ``GET /graph`` 500 until it was deleted.

The rule is ``GraphManager``'s (``tests/core/
test_a_name_is_refused_where_no_carrier_could_hold_it.py``); these are its
REST doors, and the checkpoint's name, which the same rule now covers.
"""

from __future__ import annotations

import json

import pytest

from maddening.nodes import BallNode
from tests.property import rest_oracle as O

JSON = {"content-type": "application/json"}
#: A character of each kind a name cannot hold.
UNCARRIABLE = {"a NUL": "\x00", "U+0001": "\x01", "a vertical tab": "\x0b", "an escape": "\x1b",
               "U+001F": "\x1f", "a surrogate": "\ud800", "U+FFFE": "\ufffe",
               "U+FFFF": "\uffff"}
TOKENS = ["NaN", "Infinity", "-Infinity"]


def strict(body: bytes):
    """*body* as a strict parser reads it: UTF-8, and no bare ``NaN``."""
    def refuse(token):
        raise AssertionError(f"the reply holds the bare token {token}")
    return json.loads(body.decode("utf-8"), parse_constant=refuse)


class CountedBall(BallNode):
    """A ball that counts its constructions."""

    built = 0

    def __init__(self, name, timestep, **params):
        type(self).built += 1
        super().__init__(name, timestep, **params)


class LabelledBall(BallNode):
    """A ball with two text parameters, as a registered class may have.
    Its step reads the label (so ``PUT /graph/params`` rebuilds the node
    for a new one) and not the tags."""

    def __init__(self, name, timestep, label="x", tags=(), **params):
        super().__init__(name, timestep, **params)
        self.params["label"] = label
        self.params["tags"] = list(tags)

    def update(self, state, boundary_inputs, dt):
        new = super().update(state, boundary_inputs, dt)
        return {**new, "position": new["position"] + 0.5 * len(self.params["label"])}


@pytest.fixture
def served():
    from maddening.core.graph_manager import GraphManager
    from maddening.nodes import SpringDamperNode

    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=3.0))
    gm.add_node(SpringDamperNode("spring", 0.01))
    gm.compile()
    registry = dict(O.REGISTRY, CountedBall=CountedBall, LabelledBall=LabelledBall)
    served = O.serve(gm, registry=registry)
    CountedBall.built = 0
    yield served
    served.close()


def post(served, url, body):
    """*body* as a client that escapes non-ASCII writes it: a surrogate is
    ``\\udXXX`` on the wire, which the server's parser reads."""
    return served.client.post(url, content=json.dumps(body), headers=JSON)


def refused_and_nothing_changed(served, before, resp, status=400):
    assert resp.status_code == status, (resp.status_code, resp.text[:300])
    detail = strict(resp.content)["detail"]
    assert not O.differences(before, O.snapshot(served)), detail
    assert served.client.get("/graph").status_code == 200
    return detail


# ---------------------------------------------------------------------------
# POST /graph/nodes: the name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("what", sorted(UNCARRIABLE))
def test_a_node_name_some_carrier_cannot_hold_is_a_400_before_anything_is_built(served, what):
    char = UNCARRIABLE[what]
    for name in (f"a{char}b", char, f"{char}b"):
        before = O.snapshot(served)
        body = {"type": "CountedBall", "name": name, "timestep": 0.01, "params": {}}
        detail = refused_and_nothing_changed(served, before, post(served, "/graph/nodes", body))
        assert f"U+{ord(char):04X}" in detail and "is invalid" in detail
        # "a repeat POST ... gets the same 400, not 409"
        assert strict(post(served, "/graph/nodes", body).content)["detail"] == detail
    assert CountedBall.built == 0


def test_a_refused_name_is_refused_before_its_type_is_looked_up(served):
    """So no reply quotes a name a reply could not carry: an unknown type
    with such a name is the name's 400."""
    before = O.snapshot(served)
    resp = post(served, "/graph/nodes", {"type": "NoSuchNode", "name": "a\x00b",
                                         "timestep": 0.01})
    assert "U+0000" in refused_and_nothing_changed(served, before, resp)


@pytest.mark.parametrize("name", ["a b", "a.b", 'a"b', "é", "名", "😀", "a\u2028b", "nan",
                                  "x00", "two\nlines", "a\tb", "a\x7fb"],
                         ids=lambda n: repr(n).encode("ascii", "replace").decode())
def test_a_node_name_every_carrier_holds_is_taken_and_its_checkpoint_loads(served, name):
    resp = post(served, "/graph/nodes", {"type": "BallNode", "name": name, "timestep": 0.01})
    assert resp.status_code == 201, resp.text
    assert name in [node["name"] for node in served.client.get("/graph").json()["nodes"]]
    assert served.client.post("/sim/step").status_code == 200
    saved = served.client.post("/checkpoint/save", params={"path": "named.npz"})
    assert saved.status_code == 200, saved.text
    loaded = served.client.post("/checkpoint/load", params={"path": "named.npz"})
    assert loaded.status_code == 200, loaded.text
    assert name in loaded.json()["state"]
    O.check_accepted_graph(served, f"a node named {name!r}", step=False)


# ---------------------------------------------------------------------------
# POST /graph/edges: the fields
# ---------------------------------------------------------------------------

EDGE = {"source_node": "ball", "target_node": "spring", "source_field": "position",
        "target_field": "anchor_position"}
BAD_FIELDS = TOKENS + ["a#b", "a\x00b", "a\x1bb", "a\ud800b", "a\ufffeb"]


@pytest.mark.parametrize("field", ["source_field", "target_field"])
@pytest.mark.parametrize("bad", BAD_FIELDS, ids=[repr(b).encode("ascii", "replace").decode()
                                                 for b in BAD_FIELDS])
def test_an_edge_field_the_config_could_not_carry_is_a_400(served, field, bad):
    """The target field is the caller's to choose, so it is where the
    rule bites; a source field must exist, which refuses these first."""
    before = O.snapshot(served)
    detail = refused_and_nothing_changed(
        served, before, post(served, "/graph/edges", {**EDGE, field: bad}))
    if field == "target_field":
        assert "its target field" in detail and "is invalid" in detail
    assert not served.gm._edges   # noqa: SLF001


@pytest.mark.parametrize("target_field", ["undeclared", "nan", "a.b", "a->b", "a/b", "a b",
                                          "two\nlines"])
def test_an_edge_to_a_field_the_target_does_not_declare_is_taken_and_the_graph_reloads(
        served, target_field):
    """What ``POST /graph/edges`` documents is that the nodes and the
    source field exist.  A target field the node does not declare is
    taken, as ``GraphManager.validate`` passes it (a node may read an input
    it does not declare); the graph is reported, saved and reloaded, and
    the edge is removed by the same four names."""
    edge = {**EDGE, "target_field": target_field}
    assert post(served, "/graph/edges", edge).status_code == 201
    shown = served.client.get("/graph")
    assert shown.status_code == 200
    assert [e["target_field"] for e in shown.json()["edges"]] == [target_field]
    O.check_accepted_graph(served, f"an edge to {target_field!r}", step=False,
                           shapes_may_differ=True)
    assert served.client.post("/sim/step").status_code == 200
    removed = served.client.request("DELETE", "/graph/edges", content=json.dumps(edge),
                                    headers=JSON)
    assert removed.status_code == 200 and not served.gm._edges   # noqa: SLF001


# ---------------------------------------------------------------------------
# A text parameter's value
# ---------------------------------------------------------------------------

BAD_TEXT = TOKENS + ["a\ud800b"]


@pytest.mark.parametrize("params", [{"label": bad} for bad in BAD_TEXT]
                         + [{"tags": ["ok", bad]} for bad in BAD_TEXT]
                         + [{"tags": [["ok"], {"deep": "NaN"}]}],
                         ids=lambda p: repr(p).encode("ascii", "replace").decode())
def test_a_new_node_with_text_the_config_could_not_carry_is_a_400_and_no_node(served, params):
    before = O.snapshot(served)
    resp = post(served, "/graph/nodes", {"type": "LabelledBall", "name": "new",
                                         "timestep": 0.01, "params": params})
    detail = refused_and_nothing_changed(served, before, resp)
    assert detail.startswith(f"params.{next(iter(params))}: the text ")
    assert "new" not in served.gm._nodes   # noqa: SLF001


def test_text_that_only_looks_like_a_token_is_taken_and_reported(served):
    resp = post(served, "/graph/nodes", {
        "type": "LabelledBall", "name": "new", "timestep": 0.01,
        "params": {"label": "nan", "tags": ["Infinity ", "NaNs", "-infinity", "é"]}})
    assert resp.status_code == 201, resp.text
    assert strict(resp.content)["node"]["params"]["label"] == "nan"
    params = served.client.get("/graph/params/new")
    assert params.status_code == 200
    assert strict(params.content)["tags"] == ["Infinity ", "NaNs", "-infinity", "é"]
    assert served.client.get("/graph").status_code == 200
    O.check_accepted_graph(served, "a node with text parameters", step=False)


@pytest.mark.parametrize("bad", BAD_TEXT, ids=[repr(b).encode("ascii", "replace").decode()
                                               for b in BAD_TEXT])
def test_a_text_parameter_cannot_be_written_to_text_the_config_could_not_carry(served, bad):
    assert post(served, "/graph/nodes", {"type": "LabelledBall", "name": "new",
                                         "timestep": 0.01}).status_code == 201
    before = O.snapshot(served)
    for params in ({"label": bad}, {"tags": ["ok", bad]}):
        resp = served.client.put("/graph/params/new", content=json.dumps({"params": params}),
                                 headers=JSON)
        detail = refused_and_nothing_changed(served, before, resp)
        assert detail.startswith(f"{next(iter(params))}: the text "), detail
    shown = served.client.get("/graph/params/new")
    assert shown.status_code == 200 and strict(shown.content)["label"] == "x"
    # A lookalike is written, reported and saved.
    resp = served.client.put("/graph/params/new", json={"params": {"label": "nan"}})
    assert resp.status_code == 200, resp.text
    assert strict(served.client.get("/graph/params/new").content)["label"] == "nan"
    O.check_accepted_graph(served, "a text parameter written", step=False)


# ---------------------------------------------------------------------------
# A checkpoint's name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("what", sorted(set(UNCARRIABLE) - {"a surrogate"}))
@pytest.mark.parametrize("route", ["/checkpoint/save", "/checkpoint/load"])
def test_a_checkpoint_name_some_carrier_cannot_hold_is_a_400_and_no_file(served, route, what):
    """A surrogate cannot arrive in a URL; every other character can."""
    char = UNCARRIABLE[what]
    before = O.snapshot(served)
    files = sorted(p.name for p in served.root.rglob("*"))
    resp = served.client.post(route, params={"path": f"a{char}b.npz"})
    detail = refused_and_nothing_changed(served, before, resp)
    assert "checkpoint path" in detail
    assert sorted(p.name for p in served.root.rglob("*")) == files


@pytest.mark.parametrize("name", ["a b.npz", "a.b", "é名.npz", "NaN", "a#b.npz", "two\nlines.npz"])
def test_a_checkpoint_name_every_carrier_holds_is_saved_and_loaded(served, name):
    saved = served.client.post("/checkpoint/save", params={"path": name})
    assert saved.status_code == 200, saved.text
    strict(saved.content)
    loaded = served.client.post("/checkpoint/load", params={"path": name})
    assert loaded.status_code == 200, loaded.text
