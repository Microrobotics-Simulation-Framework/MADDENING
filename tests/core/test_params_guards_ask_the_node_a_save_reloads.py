"""A params write is judged beside the values the graph runs with, whatever
door they came in by.

A fit, a ``gm.params`` write and a checkpoint load move a live leaf and
leave the value the node was *constructed* with; ``PUT /graph/params``
moves both.  So two graphs can hold the same live values -- and save the
same config -- with different values in ``node.params``.  A guard that asks
what a *save of the graph would reload* has to ask it of the live values:
``from_dict`` builds the node from those.

Two guards asked it of the constructed values (MADD-ANO-215):

* the check that a write does not move the points an interface mapping was
  built from built the rod with the new ``length`` beside the diffusivity
  it was constructed with, and took the constructor refusing that pair as
  "cannot tell".  After a load had lowered the diffusivity, a length that
  is unstable at the constructed one and stable at the live one was
  refused by nothing: 200, a mapping on the old grid, a save that did not
  load.  One request carrying the length and the lower diffusivity
  together did the same with no load at all.
* the check that the constructor takes a structural value asked it beside
  the constructed leaves, so a ``stencil_order`` the live diffusivity
  allows was refused after a load and taken after a ``PUT`` of the same
  value.

Every test below reaches one set of live values through each door and asks
for one verdict.
"""

from __future__ import annotations

import json
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from tests._loopback_client import LoopbackTestClient as TestClient

from maddening.api.server import SimulationServer
from maddening.core.coupling.mapping import nearest_neighbor_mapping
from maddening.core.coupling.sparse_mapping import sparse_nearest_neighbor_mapping
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode

REGISTRY = {"HeatNode": HeatNode}
DT = 0.01  # units: s
#: Rod ``a`` has six cells, so its Fourier number is ``36 * DT * alpha /
#: length**2``.  Built at ``HIGH`` it is 0.216 at length 1 and 0.864 at
#: length 0.5 (past the limit of 1/2); at ``LOW`` it is 0.007 at either.
HIGH, LOW = 0.6, 0.005  # units: m^2/s
#: ... and at ``STEEP`` it is 0.45: inside the 2nd-order limit, past the
#: 4th-order one (5/16).
STEEP = 1.25  # units: m^2/s
SHORT = 0.5  # units: m

_MAPPINGS = {"dense": nearest_neighbor_mapping, "sparse": sparse_nearest_neighbor_mapping}

#: How the live diffusivity of rod ``a`` gets to ``LOW``.  After ``put``
#: the node's own value is ``LOW`` too; after the others it is still the
#: value the rod was built with.
DOORS = ("put", "fit", "load", "load_state")


def _rods(alpha: float, kind: str | None) -> GraphManager:
    a = HeatNode("a", DT, n_cells=6, length=1.0, thermal_diffusivity=alpha,
                 initial_temperature=np.linspace(1.0, 2.0, 6).tolist())
    b = HeatNode("b", DT, n_cells=5, length=1.0, thermal_diffusivity=LOW,
                 initial_temperature=0.5)
    gm = GraphManager()
    gm.add_node(a)
    gm.add_node(b)
    if kind is not None:
        gm.add_edge("a", "b", "temperature", "heat_source", mapping=_MAPPINGS[kind](
            np.asarray(a.static_data["grid_x"].value), np.asarray(b.static_data["grid_x"].value),
            source_ref={"node": "a", "field": "grid_x"},
            target_ref={"node": "b", "field": "grid_x"}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


def _client(gm: GraphManager, root) -> TestClient:
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm,
                              checkpoint_root=str(root))
    return TestClient(server.create_app(), raise_server_exceptions=False)


def _put(client, params: dict):
    return client.put("/graph/params/a", json={"params": params})


def _lowered(door: str, root, *, alpha: float = HIGH, kind: str | None = "dense"):
    """``(gm, client)``: rod ``a`` built at *alpha*, its live diffusivity
    brought to ``LOW`` through *door*."""
    gm = _rods(alpha, kind)
    client = _client(gm, root)
    if door == "put":
        assert _put(client, {"thermal_diffusivity": LOW}).status_code == 200
    elif door == "fit":
        gm.params["nodes"]["a"]["thermal_diffusivity"] = jnp.asarray(LOW, jnp.float32)
    elif door == "load":
        assert _put(client, {"thermal_diffusivity": LOW}).status_code == 200
        assert client.post("/checkpoint/save", params={"path": "low.npz"}).status_code == 200
        assert _put(client, {"thermal_diffusivity": alpha}).status_code == 200
        resp = client.post("/checkpoint/load", params={"path": "low.npz"})
        assert resp.status_code == 200, resp.text
    else:
        other = _rods(alpha, kind)
        other.params["nodes"]["a"]["thermal_diffusivity"] = jnp.asarray(LOW, jnp.float32)
        other.save_state(root / "low_state.npz")
        gm.load_state(root / "low_state.npz")
    assert float(gm.params["nodes"]["a"]["thermal_diffusivity"]) == pytest.approx(LOW)
    own = gm.get_node("a").params["thermal_diffusivity"]
    assert own == pytest.approx(LOW if door == "put" else alpha)
    return gm, client


def _reloads_and_agrees(gm: GraphManager) -> None:
    """The graph's config, through JSON, reloads and steps as the graph does."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = GraphManager.from_dict(json.loads(json.dumps(gm.to_dict())), REGISTRY)
        fresh.compile()
        for name in gm.node_names:
            fresh.set_node_state(name, gm.get_node_state(name))
        gm.run(3)
        fresh.run(3)
    for name in gm.node_names:
        for field, value in gm.get_node_state(name).items():
            np.testing.assert_array_equal(np.asarray(value),
                                          np.asarray(fresh.get_node_state(name)[field]))


# ---------------------------------------------------------------------------
# The points a mapping was built from
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", sorted(_MAPPINGS))
@pytest.mark.parametrize("door", DOORS)
def test_the_route_refuses_a_length_under_a_mapping_whatever_brought_the_other_leaf(
        door, kind, tmp_path):
    gm, client = _lowered(door, tmp_path, kind=kind)
    before = client.get("/graph/params/a").json()
    resp = _put(client, {"length": SHORT})
    assert resp.status_code == 400, resp.text
    assert "was built from a.grid_x" in resp.json()["detail"], resp.text
    assert client.get("/graph/params/a").json() == before
    assert gm.get_node("a").params["length"] == 1.0
    _reloads_and_agrees(gm)


@pytest.mark.parametrize("kind", sorted(_MAPPINGS))
def test_the_route_refuses_a_length_written_with_the_diffusivity_that_allows_it(kind, tmp_path):
    """One request, no load: the length alone is refused by the constructor
    (and so, before, "not told" by the mapping check), the pair is taken by
    it, and the pair was asked of no mapping check."""
    gm = _rods(HIGH, kind)
    client = _client(gm, tmp_path)
    resp = _put(client, {"length": SHORT, "thermal_diffusivity": LOW})
    assert resp.status_code == 400, resp.text
    assert "was built from a.grid_x" in resp.json()["detail"], resp.text
    assert gm.get_node("a").params["thermal_diffusivity"] == HIGH
    assert float(gm.params["nodes"]["a"]["thermal_diffusivity"]) == pytest.approx(HIGH)
    _reloads_and_agrees(gm)


@pytest.mark.parametrize("door", DOORS)
def test_a_length_in_gm_params_under_a_mapping_is_refused_whatever_brought_the_other_leaf(
        door, tmp_path):
    gm, _client_unused = _lowered(door, tmp_path)
    gm.params["nodes"]["a"]["length"] = jnp.asarray(SHORT, jnp.float32)
    for call in (lambda: gm.run(1), gm.to_dict, lambda: gm.save_state(tmp_path / "x.npz")):
        with pytest.raises(ValueError, match="interface mapping on edge"):
            call()
    gm.params["nodes"]["a"]["length"] = jnp.asarray(1.0, jnp.float32)
    _reloads_and_agrees(gm)


def test_a_length_the_constructor_refuses_is_still_refused_for_the_mapping_in_gm_params():
    """No fit and no load: the new length is unstable beside every value
    the graph holds, so the rod cannot be built to read its grid.
    ``gm.params`` has no check that asks the constructor, and ran the
    write; the leaf is now asked whether any value of it moves the points."""
    gm = _rods(HIGH, "dense")
    gm.params["nodes"]["a"]["length"] = jnp.asarray(SHORT, jnp.float32)
    with pytest.raises(ValueError, match="interface mapping on edge"):
        gm.run(1)


@pytest.mark.parametrize("door", DOORS)
def test_a_checkpoint_carrying_a_length_is_refused_whatever_brought_the_other_leaf(
        door, tmp_path):
    """The checkpoint route shares the decision: a file with the short
    length and the low diffusivity, loaded over each graph."""
    source = _rods(HIGH, None)
    source.params["nodes"]["a"]["thermal_diffusivity"] = jnp.asarray(LOW, jnp.float32)
    source.params["nodes"]["a"]["length"] = jnp.asarray(SHORT, jnp.float32)
    source.save_state(tmp_path / "short.npz")
    gm, client = _lowered(door, tmp_path)
    resp = client.post("/checkpoint/load", params={"path": "short.npz"})
    assert resp.status_code == 400, resp.text
    assert "interface mapping on edge" in resp.json()["detail"], resp.text
    assert float(gm.params["nodes"]["a"]["length"]) == 1.0
    _reloads_and_agrees(gm)


def test_a_checkpoint_carrying_a_length_and_the_diffusivity_that_allows_it_is_refused(tmp_path):
    """... and over the graph as it was built, where the file's length is
    unstable beside the live diffusivity and stable beside the file's."""
    source = _rods(HIGH, None)
    source.params["nodes"]["a"]["thermal_diffusivity"] = jnp.asarray(LOW, jnp.float32)
    source.params["nodes"]["a"]["length"] = jnp.asarray(SHORT, jnp.float32)
    source.save_state(tmp_path / "short.npz")
    gm = _rods(HIGH, "sparse")
    resp = _client(gm, tmp_path).post("/checkpoint/load", params={"path": "short.npz"})
    assert resp.status_code == 400, resp.text
    assert "interface mapping on edge" in resp.json()["detail"], resp.text
    assert float(gm.params["nodes"]["a"]["thermal_diffusivity"]) == pytest.approx(HIGH)


# ---------------------------------------------------------------------------
# What the guards must go on taking
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("door", DOORS)
def test_a_calibration_of_the_mapped_rod_is_taken_after_every_door(door, tmp_path):
    gm, client = _lowered(door, tmp_path, kind="sparse")
    resp = _put(client, {"thermal_diffusivity": 2 * LOW})
    assert resp.status_code == 200, resp.text
    _reloads_and_agrees(gm)


@pytest.mark.parametrize("door", DOORS)
def test_a_length_on_an_unmapped_rod_is_taken_after_every_door(door, tmp_path):
    """The same write with no mapping to hold it: stable at the live
    diffusivity, taken, and the save reloads the rod that runs."""
    gm, client = _lowered(door, tmp_path, kind=None)
    resp = _put(client, {"length": SHORT})
    assert resp.status_code == 200, resp.text
    _reloads_and_agrees(gm)


# ---------------------------------------------------------------------------
# The constructor, asked of a structural value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("door", DOORS)
def test_a_structural_value_the_live_leaves_allow_is_taken_after_every_door(door, tmp_path):
    """``stencil_order: 4`` is past its limit at the diffusivity the rod was
    built with and well inside it at the live one.  It was a 400 after a
    fit or a load and a 200 after a ``PUT`` of the same diffusivity."""
    gm, client = _lowered(door, tmp_path, alpha=STEEP, kind=None)
    resp = _put(client, {"stencil_order": 4})
    assert resp.status_code == 200, resp.text
    _reloads_and_agrees(gm)


def test_a_structural_value_the_live_leaves_refuse_is_still_refused(tmp_path):
    gm = _rods(STEEP, None)
    resp = _put(_client(gm, tmp_path), {"stencil_order": 4})
    assert resp.status_code == 400, resp.text
    assert "constructor refuses it" in resp.json()["detail"], resp.text
    assert gm.get_node("a").params["stencil_order"] == 2
