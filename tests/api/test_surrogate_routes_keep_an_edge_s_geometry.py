"""The surrogate routes refuse to drop the field an edge's geometry is read from.

An edge with a geometry-dependent mapping (experimental) reads a state
field of its source or target node.  ``POST /surrogate/activate`` puts a
trained node in a node's place and ``POST /surrogate/deactivate`` puts the
original back; either would leave such an edge reading a field that is no
longer there, or has another shape or dtype.  Both ask ``replace_node``'s
own rule before they change anything and answer a 409 that says which
edge and which field, with the graph, the server's record of the
surrogates and the streams exactly as they were.  Left to
``replace_node`` inside the route's transaction the graph was put back
too, but the client was told a generic 500.

``GET /graph`` shows an edge's geometry (it is the config's key).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.core import geometry_surface_graphs as G
from tests.property import rest_oracle as O
from tests.property.differential import no_cloud_launch, quiet
from tests.property.graph_fingerprint import assert_exactly_as_it_was, served_fingerprint


@pytest.fixture(scope="module", autouse=True)
def _no_cloud():
    with no_cloud_launch():
        yield


@pytest.fixture
def served():
    with quiet():
        gm = G.graph()
        gm.step()
    s = O.serve(gm, registry=G.REGISTRY)
    with quiet():
        yield s
    s.close()


class _Trained:
    """What ``POST /surrogate/activate`` reads of a finished job's result."""

    def __init__(self, node_type, state_spec):
        self.node_type, self.state_spec = node_type, state_spec

    def to_node(self, name, timestep, initial_values):
        return self.node_type(name, timestep, speed=0.25)


class WideMarkers(G.Markers):
    """``pos`` with two columns: the field is there, with another shape."""

    def initial_state(self):
        state = super().initial_state()
        return {**state, "pos": np.repeat(np.asarray(state["pos"]), 2, axis=1)}


class HalfMarkers(G.Markers):
    """``pos`` as float16: the field is there, with a dtype a geometry cannot have."""

    def initial_state(self):
        state = super().initial_state()
        return {**state, "pos": state["pos"].astype("float16")}


def _job(served, node_type, state_spec=None):
    spec = {"x": (G.N_MARKERS,), "pos": (G.N_MARKERS, 1)} if state_spec is None else state_spec
    served.server._surrogate_jobs["job"] = {  # noqa: SLF001
        "status": "done", "node_name": "markers", "result": _Trained(node_type, spec)}


def _geometries(served):
    return {e.key: e.geometry for e in served.gm.edges}


BOTH = {G.GATHER: ("target", "pos"), G.SCATTER: ("source", "pos")}


def test_get_graph_shows_each_edge_s_geometry(served):
    reply = served.client.get("/graph")
    assert reply.status_code == 200, reply.text
    shown = {(e["source_node"], e["target_node"]): e.get("geometry")
             for e in reply.json()["edges"]}
    assert shown == {("grid", "markers"): {"anchor": "target", "field": "pos"},
                     ("markers", "grid"): {"anchor": "source", "field": "pos"}}


@pytest.mark.parametrize("node_type", [G.StillMarkers, WideMarkers, HalfMarkers],
                         ids=["no field", "another shape", "float16"])
def test_activating_a_surrogate_that_drops_a_geometry_is_a_409_and_changes_nothing(
        served, node_type):
    _job(served, node_type, {"x": (G.N_MARKERS,)})
    original = served.gm._nodes["markers"].node  # noqa: SLF001
    before = served_fingerprint(served)
    reply = served.client.post("/surrogate/activate/job")
    assert reply.status_code == 409, reply.text
    body = reply.json()
    detail = body["detail"]
    assert "reads its geometry from markers.pos" in detail and node_type.__name__ in detail
    assert G.GATHER in detail or G.SCATTER in detail
    assert "Nothing was activated; the graph is as it was." in detail
    assert body["was_running"] is False
    assert_exactly_as_it_was(before, served_fingerprint(served), "a refused activation")
    assert served.gm._nodes["markers"].node is original  # noqa: SLF001
    assert served.server._original_nodes == {}  # noqa: SLF001
    assert served.server._active_surrogates == set()  # noqa: SLF001
    assert _geometries(served) == BOTH
    assert served.client.post("/sim/step").status_code == 200


def test_a_surrogate_that_holds_the_geometry_is_activated_and_reverted_with_the_edges_as_they_were(
        served):
    """The control: the same routes, with a replacement that holds ``pos``."""
    _job(served, G.Markers)
    original = served.gm._nodes["markers"].node  # noqa: SLF001
    reply = served.client.post("/surrogate/activate/job")
    assert reply.status_code == 200, reply.text
    assert served.gm._nodes["markers"].node is not original  # noqa: SLF001
    assert _geometries(served) == BOTH
    assert served.client.post("/sim/step").status_code == 200
    # The surrogate's markers drift at its own speed: the edges read *its* field.
    moved = np.asarray(served.gm.get_node_state("markers")["pos"]).ravel()
    assert moved[0] == pytest.approx(0.3 + 0.25 * G.DT, rel=1e-6)

    reply = served.client.post("/surrogate/deactivate/markers")
    assert reply.status_code == 200, reply.text
    assert served.gm._nodes["markers"].node is original  # noqa: SLF001
    assert _geometries(served) == BOTH
    assert served.client.post("/sim/step").status_code == 200


def test_reverting_to_an_original_that_does_not_hold_a_geometry_is_a_409_and_changes_nothing(
        served):
    """The record of what to put back names a node without ``pos`` (the
    edges were given their geometry while the surrogate was active)."""
    server = served.server
    edges = list(served.gm.edges)
    server._original_nodes["markers"] = (G.StillMarkers("markers", G.DT), edges, [])  # noqa: SLF001
    server._active_surrogates.add("markers")  # noqa: SLF001
    live = served.gm._nodes["markers"].node  # noqa: SLF001
    before = served_fingerprint(served)
    reply = served.client.post("/surrogate/deactivate/markers")
    assert reply.status_code == 409, reply.text
    detail = reply.json()["detail"]
    assert "reads its geometry from markers.pos" in detail and "StillMarkers" in detail
    assert "Nothing was deactivated; the graph is as it was." in detail
    assert_exactly_as_it_was(before, served_fingerprint(served), "a refused revert")
    assert served.gm._nodes["markers"].node is live  # noqa: SLF001
    assert _geometries(served) == BOTH
    assert served.client.post("/sim/step").status_code == 200


def test_replacing_the_other_end_of_a_geometry_edge_is_not_refused(served):
    """Only the node the geometry is read from is held to the rule."""
    served.server._surrogate_jobs["job"] = {  # noqa: SLF001
        "status": "done", "node_name": "grid",
        "result": _TrainedGrid()}
    reply = served.client.post("/surrogate/activate/job")
    assert reply.status_code == 200, reply.text
    assert _geometries(served) == BOTH
    assert served.client.post("/sim/step").status_code == 200


class _TrainedGrid:
    state_spec = {"x": (G.N_GRID,)}

    def to_node(self, name, timestep, initial_values):
        return G.GridField(name, timestep, decay=0.25)
