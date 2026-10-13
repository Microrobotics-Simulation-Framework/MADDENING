"""Every route that changes the state without a step publishes it to the
streams, at the right time.

``StateRelay`` publishes on the steps it observes, and only
``POST /checkpoint/load`` published otherwise: after ``POST /sim/reset``
the streams sent nothing (a client kept the pre-reset state), after
``PUT /graph/state`` they served the old value (1.0 against 123), and after
``DELETE /graph/nodes`` they served the removed node -- the binary stream
re-sending a schema that re-added it.  Each such route now publishes the
graph's state: a reset (and a surrogate swap, which resets) at step 0 and
time 0, the others at the clock the streams had.

The graph is a counter (zero gravity, unit velocity, dt = 1/64), so a
frame whose clock agrees with its state carries ``position == sim_time``.
All three streams are read after each route: ``/ws/state``,
``/ws/state/binary`` and ``/ws/render``.
"""

from __future__ import annotations

import json
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from tests._loopback_client import LoopbackTestClient as TestClient

from maddening.api.binary_encoder import decode_frame
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode

DT = 1.0 / 64.0


class _RecordingRenderer:
    width, height, fmt, content_type = 4, 4, "png", "image/png"

    def __init__(self) -> None:
        self.calls: list = []

    def set_format(self, fmt, quality=None) -> None:
        pass

    def reset(self) -> None:
        pass

    def render(self, sim_time, state) -> bytes:
        self.calls.append((sim_time, {n: {f: np.asarray(v).tolist() for f, v in d.items()}
                                      for n, d in state.items()}))
        return b"frame"


def _counter(name="c") -> BallNode:
    return BallNode(name, timestep=DT, initial_position=0.0, initial_velocity=1.0,
                    gravity=0.0)


def _server(*nodes):
    gm = GraphManager()
    for node in nodes or (_counter(),):
        gm.add_node(node)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    renderer = _RecordingRenderer()
    server = SimulationServer({"BallNode": BallNode}, graph_manager=gm,
                              frame_renderer=renderer)
    return gm, server, TestClient(server.create_app(), raise_server_exceptions=False), renderer


def _receive(ws, timeout: float = 10.0) -> dict:
    """``ws.receive()``, failing the test after *timeout* seconds: a stream
    that publishes nothing must fail here, not hang the run (the session's
    close, on the way out, releases the waiting thread)."""
    import threading

    box: list = []

    def wait() -> None:
        try:
            box.append(ws.receive())
        except Exception as exc:  # noqa: BLE001 - the session closed under it
            box.append(exc)

    waiter = threading.Thread(target=wait, daemon=True)
    waiter.start()
    waiter.join(timeout)
    if not box:
        pytest.fail(f"the stream sent nothing for {timeout:g} s")
    if isinstance(box[0], Exception):
        raise box[0]
    return box[0]


def _receive_json(ws) -> dict:
    return json.loads(_receive(ws)["text"])


def _streams(client, renderer) -> dict:
    """What a fresh client of each stream is sent first: ``{stream:
    (sim_time, {node: {field: value}})}``."""
    out = {}
    with client.websocket_connect("/ws/state") as ws:
        frame = _receive_json(ws)
    out["state"] = (frame["sim_time"], frame["state"])
    with client.websocket_connect("/ws/state/binary") as ws:
        schema = _receive_json(ws)
        message = _receive(ws)
        while "text" in message:
            schema = json.loads(message["text"])
            message = _receive(ws)
    sim_time, values = decode_frame(message["bytes"], schema)
    state: dict = {}
    for f in schema["fields"]:
        v = values[f["offset"]: f["offset"] + f["count"]]
        state.setdefault(f["node"], {})[f["field"]] = float(v[0]) if not f["shape"] \
            else v.tolist()
    out["binary"] = (sim_time, state)
    with client.websocket_connect("/ws/render") as ws:
        _receive_json(ws)
        assert _receive(ws)["bytes"] == b"frame"
    out["render"] = renderer.calls[-1]
    return out


def _assert_every_stream_shows(client, renderer, sim_time: float) -> dict:
    truth = {n: d for n, d in client.get("/graph/state").json().items() if n != "_meta"}
    for name, (t, state) in _streams(client, renderer).items():
        assert t == sim_time, (name, t, sim_time)
        assert set(state) == set(truth), (name, set(state), set(truth))
        for node, fields in truth.items():
            for field, value in fields.items():
                assert np.allclose(state[node][field], value, rtol=0, atol=0), \
                    (name, node, field, state[node][field], value)
    return truth


def test_a_reset_is_published_at_step_zero():
    gm, server, client, renderer = _server()
    assert client.post("/sim/run", params={"n_steps": 64}).status_code == 200
    _assert_every_stream_shows(client, renderer, 1.0)
    assert client.post("/sim/reset").status_code == 200
    truth = _assert_every_stream_shows(client, renderer, 0.0)
    assert truth["c"]["position"] == 0.0
    assert server.relay.step_count == 0


def test_a_state_write_is_published_at_the_clock_it_was_written():
    gm, server, client, renderer = _server()
    assert client.post("/sim/run", params={"n_steps": 64}).status_code == 200
    resp = client.put("/graph/state/c", json={"state": {"position": 123.0, "velocity": 1.0}})
    assert resp.status_code == 200, resp.text
    truth = _assert_every_stream_shows(client, renderer, 1.0)
    assert truth["c"]["position"] == 123.0
    assert server.relay.step_count == 64


def test_a_removed_node_leaves_every_stream():
    gm, server, client, renderer = _server(_counter("c"), _counter("x"))
    assert client.post("/sim/run", params={"n_steps": 32}).status_code == 200
    assert client.delete("/graph/nodes/x").status_code == 200
    truth = _assert_every_stream_shows(client, renderer, 0.5)
    assert set(truth) == {"c"}


def test_an_added_node_joins_every_stream():
    gm, server, client, renderer = _server()
    assert client.post("/sim/run", params={"n_steps": 32}).status_code == 200
    resp = client.post("/graph/nodes", json={"type": "BallNode", "name": "y",
                                             "timestep": DT, "params": {}})
    assert resp.status_code == 201, resp.text
    truth = _assert_every_stream_shows(client, renderer, 0.5)
    assert set(truth) == {"c", "y"}


def test_a_connected_client_is_sent_the_written_state():
    gm, server, client, renderer = _server()
    assert client.post("/sim/step").status_code == 200
    with client.websocket_connect("/ws/state") as ws:
        assert ws.receive_json()["state"]["c"]["position"] == DT
        resp = client.put("/graph/state/c", json={"state": {"position": 7.0, "velocity": 1.0}})
        assert resp.status_code == 200, resp.text
        frame = _receive_json(ws)
    assert frame == {"sim_time": DT, "state": {"c": {"position": 7.0, "velocity": 1.0}}}


def test_a_surrogate_swap_is_published_at_step_zero_both_ways():
    pytest.importorskip("equinox", reason="surrogate training needs the surrogates extra")
    pytest.importorskip("optax", reason="surrogate training needs the surrogates extra")
    import time

    gm, server, client, renderer = _server()
    resp = client.post("/surrogate/train", json={"node_name": "c", "n_epochs": 1,
                                                 "n_data_steps": 5, "hidden_sizes": [8],
                                                 "batch_size": 16})
    assert resp.status_code == 200, resp.text
    job = resp.json()["job_id"]
    end = time.monotonic() + 120
    while client.get(f"/surrogate/status/{job}").json()["status"] == "running":
        assert time.monotonic() < end
        time.sleep(0.05)
    assert client.post("/sim/run", params={"n_steps": 64}).status_code == 200
    resp = client.post(f"/surrogate/activate/{job}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["was_running"] is False
    _assert_every_stream_shows(client, renderer, 0.0)
    assert client.post("/sim/run", params={"n_steps": 8}).status_code == 200
    resp = client.post("/surrogate/deactivate/c")
    assert resp.status_code == 200, resp.text
    truth = _assert_every_stream_shows(client, renderer, 0.0)
    assert truth["c"]["position"] == 0.0
