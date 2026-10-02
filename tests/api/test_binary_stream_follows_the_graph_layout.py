"""Every ``/ws/state/binary`` frame is laid out by the last schema sent.

The server built its ``BinaryStateEncoder`` once and dropped it only on a
reset.  A node replaced over REST (``DELETE`` then ``POST`` with another
``n_cells``) left the cached encoder in place, so a new client was sent a
schema the graph no longer had; and ``encode`` wrote the new values into
the old slot by ``bytearray`` slice assignment, which *resizes* the
payload: six values were cut to four, and two values were read back as
four, two of them phantom zeros, in a frame exactly ``frame_bytes`` long.

The invariants: the decoded frame equals ``GET /graph/state`` after any
structural edit, for a new client and for one connected throughout; the
cached encoder is dropped when a node is added or removed or the graph is
compiled; and ``encode`` refuses a field of another size.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api.binary_encoder import BinaryStateEncoder, decode_frame
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode, HeatNode


def _server():
    gm = GraphManager()
    gm.add_node(HeatNode("a_rod", timestep=0.01, n_cells=4, initial_temperature=300.0))
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=5.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({"HeatNode": HeatNode}, graph_manager=gm)
    return gm, server, TestClient(server.create_app(), raise_server_exceptions=False)


def _replace_rod(client, n_cells: int) -> None:
    assert client.delete("/graph/nodes/a_rod").status_code == 200
    resp = client.post("/graph/nodes", json={
        "type": "HeatNode", "name": "a_rod", "timestep": 0.01,
        "params": {"n_cells": n_cells, "initial_temperature": 300.0}})
    assert resp.status_code == 201, resp.text


def _assert_frame_is_the_state(frame: bytes, schema: dict, truth: dict) -> None:
    assert len(frame) == schema["frame_bytes"]
    _, values = decode_frame(frame, schema)
    assert values.size == schema["total_floats"]
    fields = {(f["node"], f["field"]) for f in schema["fields"]}
    want_fields = {(n, f) for n, d in truth.items() if n != "_meta" for f in d}
    assert fields == want_fields
    for f in schema["fields"]:
        got = values[f["offset"]: f["offset"] + f["count"]]
        want = np.atleast_1d(np.asarray(truth[f["node"]][f["field"]], dtype=np.float32))
        assert got.size == want.size, (f, got, want)
        assert np.array_equal(got, want.ravel()), (f, got, want)


@pytest.mark.parametrize("n_cells", [6, 2])
def test_a_new_client_after_a_node_is_replaced_decodes_the_graphs_state(n_cells):
    gm, server, client = _server()
    with client.websocket_connect("/ws/state/binary") as ws:   # builds the encoder
        ws.receive_json()
    _replace_rod(client, n_cells)
    with client.websocket_connect("/ws/state/binary") as ws:
        schema = ws.receive_json()
        assert client.post("/sim/step").status_code == 200
        message = ws.receive()
        while "text" in message:                 # a schema for the new layout
            schema = __import__("json").loads(message["text"])
            message = ws.receive()
        frame = message["bytes"]
    rod = next(f for f in schema["fields"] if f["node"] == "a_rod")
    assert rod["shape"] == [n_cells]
    _assert_frame_is_the_state(frame, schema, client.get("/graph/state").json())


@pytest.mark.parametrize("n_cells", [6, 2])
def test_a_client_connected_through_a_replacement_gets_the_new_schema_first(n_cells):
    import json

    gm, server, client = _server()
    with client.websocket_connect("/ws/state/binary") as ws:
        schema = ws.receive_json()
        assert client.post("/sim/step").status_code == 200
        _assert_frame_is_the_state(ws.receive_bytes(), schema,
                                   client.get("/graph/state").json())
        _replace_rod(client, n_cells)
        assert client.post("/sim/step").status_code == 200
        truth = client.get("/graph/state").json()
        # The snapshot from before the replacement was sent already, so the
        # first frame now is the new step's: laid out by the schema sent
        # last before it, which must be the new layout's.
        message = ws.receive()
        while "text" in message:
            schema = json.loads(message["text"])
            message = ws.receive()
        rod = next(f for f in schema["fields"] if f["node"] == "a_rod")
        assert rod["shape"] == [n_cells], "the frame came under the old schema"
        _assert_frame_is_the_state(message["bytes"], schema, truth)


@pytest.mark.parametrize("edit", ["add", "remove", "compile"])
def test_the_cached_encoder_is_dropped_by_a_structural_change(edit):
    gm, server, client = _server()
    with client.websocket_connect("/ws/state/binary") as ws:
        ws.receive_json()
    assert server._binary_encoder is not None
    generation = server._layout_generation
    if edit == "add":
        assert client.post("/graph/nodes", json={
            "type": "HeatNode", "name": "z_rod", "timestep": 0.01,
            "params": {"n_cells": 3}}).status_code == 201
    elif edit == "remove":
        assert client.delete("/graph/nodes/a_rod").status_code == 200
    else:
        assert client.post("/graph/compile").status_code == 200
    assert server._binary_encoder is None
    assert server._layout_generation > generation


@pytest.mark.parametrize("values", [np.full(6, 1.0), np.full(2, 1.0)])
def test_encode_refuses_a_field_of_another_size(values):
    encoder = BinaryStateEncoder({"rod": {"temperature": np.zeros(4)}})
    with pytest.raises(ValueError, match="rod.temperature has .* the schema 4"):
        encoder.encode(0.0, {"rod": {"temperature": values}})


def test_encode_of_the_schemas_layout_is_unchanged():
    encoder = BinaryStateEncoder({"b": {"p": np.float32(0.0)},
                                  "rod": {"temperature": np.zeros(4)}})
    frame = encoder.encode(1.5, {"b": {"p": np.float32(2.0)},
                                 "rod": {"temperature": np.arange(4.0)}})
    assert len(frame) == encoder.frame_bytes
    sim_time, values = decode_frame(frame, encoder.schema())
    assert sim_time == 1.5 and values.tolist() == [2.0, 0.0, 1.0, 2.0, 3.0]
