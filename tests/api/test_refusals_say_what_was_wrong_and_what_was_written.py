"""Small refusals that said the wrong thing, or cost too much to reach.

* ``POST /checkpoint/load`` into a graph that cannot compile answered
  "could not load checkpoint" of a file nothing was wrong with; it names
  the graph's reason now, as ``POST /sim/step`` does.
* ``POST /checkpoint/save`` moves two files into place, and what its 400
  says was written depends on which move failed: "nothing was written"
  (and the directories it made are removed) for the checkpoint's own move,
  "was written, but its manifest could not be" for the manifest's.
* The value count of a request's params counted numbers only, so a body of
  empty lists passed it and was refused for its shape after the whole of it
  had been converted, inside the graph's lock.  Lists and objects are
  counted too; a state write is held to the lists its field's shape has.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from tests._loopback_client import LoopbackTestClient as TestClient

import maddening.api.server as server_module
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes import HeatNode, SpringDamperNode

DT = 0.01


class Grid(SimulationNode):
    """A field of two axes."""

    def initial_state(self):
        return {"g": jnp.zeros((2, 3), jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return state


REGISTRY = {"SpringDamperNode": SpringDamperNode, "HeatNode": HeatNode}


@pytest.fixture()
def served(tmp_path):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.add_node(HeatNode("rod", DT, n_cells=4, thermal_diffusivity=0.01))
    gm.add_node(Grid("grid", DT))
    gm.compile()
    root = tmp_path / "root"
    root.mkdir()
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm,
                              checkpoint_root=str(root))
    with TestClient(server.create_app(), raise_server_exceptions=False) as client:
        yield client, gm, root


def test_a_load_into_a_graph_that_cannot_compile_names_the_graphs_reason(served):
    client, gm, root = served
    assert client.post("/checkpoint/save?path=good.npz").status_code == 200
    # An edge between fields of different shapes is accepted, and the graph
    # then cannot compile until it is removed.
    edge = {"source_node": "rod", "target_node": "s", "source_field": "temperature",
            "target_field": "anchor_position"}
    assert client.post("/graph/edges", json=edge).status_code == 201
    step = client.post("/sim/step")
    load = client.post("/checkpoint/load?path=good.npz")
    assert step.status_code == load.status_code == 400
    detail = load.json()["detail"]
    assert detail.startswith("could not load checkpoint 'good.npz': the graph cannot "
                             "compile with its current configuration (")
    assert "ShapeMismatchError" in detail and detail.endswith("nothing was loaded")
    # The reason is the one the step gives, and no server path is in it.
    assert "ShapeMismatchError" in step.json()["detail"] and str(root) not in detail
    assert client.request("DELETE", "/graph/edges", json=edge).status_code == 200
    assert client.post("/checkpoint/load?path=good.npz").status_code == 200


def test_a_file_that_is_not_a_checkpoint_is_still_answered_as_one(served):
    client, gm, root = served
    (root / "junk.npz").write_bytes(b"not an archive")
    resp = client.post("/checkpoint/load?path=junk.npz")
    assert resp.status_code == 400 and "it is not an .npz archive" in resp.json()["detail"]


def _failing_replace(monkeypatch, name: str) -> list:
    """Make the server's move of a file into place as *name* raise; the
    moves it made, by destination name."""
    real, calls = os.replace, []

    def replace(src, dst, *args, **kwargs):
        calls.append(os.path.basename(str(dst)))
        if calls[-1] == name:
            raise OSError(28, "No space left on device")
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(server_module.os, "replace", replace)
    return calls


def test_a_save_whose_first_move_fails_says_nothing_was_written_and_leaves_nothing(
        served, monkeypatch):
    client, gm, root = served
    calls = _failing_replace(monkeypatch, "a.npz")
    resp = client.post("/checkpoint/save?path=made/deeper/a.npz")
    assert resp.status_code == 400 and calls[-1] == "a.npz"
    detail = resp.json()["detail"]
    assert detail.startswith("could not save checkpoint") and "nothing was written" in detail
    assert "was written, but" not in detail
    # ... and nothing was: no file, no temporary, none of the directories.
    assert list(root.iterdir()) == []
    assert client.post("/checkpoint/load?path=made/deeper/a.npz").status_code == 404


def test_a_save_whose_manifest_move_fails_says_the_checkpoint_was_written(served, monkeypatch):
    client, gm, root = served
    calls = _failing_replace(monkeypatch, "a.npz.manifest.json")
    resp = client.post("/checkpoint/save?path=made/a.npz")
    assert resp.status_code == 400 and calls[-2:] == ["a.npz", "a.npz.manifest.json"]
    assert "was written, but its manifest could not be" in resp.json()["detail"]
    assert sorted(p.name for p in (root / "made").iterdir()) == ["a.npz"]
    loaded = client.post("/checkpoint/load?path=made/a.npz")
    assert loaded.status_code == 200 and loaded.json()["sim_time_from_checkpoint"] is False


#: The bound, lowered for these tests: a million-member body takes seconds
#: to encode and parse, and the count is the same code at any bound.
_BOUND = 2_000


@pytest.fixture()
def low_bound(monkeypatch):
    monkeypatch.setattr(server_module, "MAX_NODE_PARAM_ELEMENTS", _BOUND)


_TOO_MANY = _BOUND + 1


@pytest.mark.parametrize("value", [
    pytest.param([[] for _ in range(_TOO_MANY)], id="empty lists"),
    pytest.param([{} for _ in range(_TOO_MANY)], id="empty objects"),
    pytest.param([[[]] for _ in range(_TOO_MANY // 2 + 1)], id="nested empty lists"),
])
def test_lists_and_objects_are_counted_like_values_by_every_params_model(
        served, low_bound, value):
    client, gm, root = served
    before = client.get("/graph").json()
    for method, url, body in (
            ("POST", "/graph/nodes", {"type": "SpringDamperNode", "name": "n", "timestep": DT,
                                      "params": {"stiffness": value}}),
            ("PUT", "/graph/params/s", {"params": {"stiffness": value}}),
            ("PUT", "/graph/params/s", {"params": {"unknown": {"deep": value}}})):
        resp = client.request(method, url, json=body)
        assert resp.status_code == 422, (method, url, resp.status_code, resp.text[:200])
        assert f"at most {_BOUND} lists and objects" in resp.text
    assert client.get("/graph").json() == before


def test_as_many_lists_as_the_bound_are_not_refused_for_their_count(served, low_bound):
    client, gm, root = served
    # One list of the bound's numbers, and the bound's worth of lists (the
    # outer one included): neither is a 422 of the count.
    for value in ([0.0] * _BOUND, [[] for _ in range(_BOUND - 2)]):
        resp = client.put("/graph/params/s", json={"params": {"stiffness": value}})
        assert resp.status_code == 400 and "expected shape ()" in resp.text, resp.text[:200]


@pytest.mark.parametrize("value, needle", [
    ([[] for _ in range(1_000)] + [1.0], "nested in 1001 list(s)"),
    ([[1.0]], "nested in 2 list(s)"),
    ([], "expected 1 value(s)"),
    ([1.0, 2.0], "expected 1 value(s)"),
])
def test_a_state_value_nested_in_other_lists_than_its_shape_is_refused_uncounted(
        served, value, needle):
    client, gm, root = served
    before = client.get("/graph/state/s").json()
    resp = client.put("/graph/state/s", json={"state": {"position": value, "velocity": 0.0}})
    assert resp.status_code == 400 and needle in resp.json()["detail"], resp.text[:300]
    assert client.get("/graph/state/s").json() == before


def test_a_state_value_spelled_with_its_shapes_lists_is_written(served):
    client, gm, root = served
    assert client.put("/graph/state/s", json={"state": {"position": 2.0, "velocity": 0.0}}
                      ).status_code == 200
    assert client.put("/graph/state/rod", json={"state": {"temperature": [1.0, 2.0, 3.0, 4.0]}}
                      ).status_code == 200
    assert client.get("/graph/state/rod").json()["temperature"] == [1.0, 2.0, 3.0, 4.0]


def test_a_field_of_two_axes_is_held_to_one_list_per_row(served):
    client, gm, root = served
    rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert client.put("/graph/state/grid", json={"state": {"g": rows}}).status_code == 200
    for value, lists in (([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1),
                         ([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], 4),
                         ([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], 5)):
        resp = client.put("/graph/state/grid", json={"state": {"g": value}})
        if lists == 4:
            # As many lists as the shape has, in another arrangement: the
            # shape check's refusal, as before.
            assert resp.status_code == 400 and "shape" in resp.json()["detail"]
        else:
            assert resp.status_code == 400, resp.text
            assert f"nested in {lists} list(s)" in resp.json()["detail"]
    assert client.get("/graph/state/grid").json()["g"] == rows
    assert [server_module._lists_in_shape(shape) for shape in
            ((), (4,), (2, 3), (2, 3, 1), (0, 5), (5, 0))] == [0, 1, 3, 9, 1, 6]
