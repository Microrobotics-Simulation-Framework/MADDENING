"""After ``POST /checkpoint/load`` the streams serve the loaded state, at
the checkpoint's time.

``StateRelay`` counts the steps it observes, and a load is not a step: the
streams went on serving the snapshot from before the load until the next
step, and their ``sim_time`` went on counting from the step before it -- a
graph loaded back to t = 1 s streamed t = 2 s, then 2.01 s.  The load now
publishes the loaded state and sets the relay's clock to the checkpoint's:
the time this server recorded in the manifest it wrote beside the
checkpoint (``<path>.manifest.json``), or zero -- counted from the load --
for a file without one, or whose manifest does not hash to it.

All three streams read the relay: ``/ws/state``, ``/ws/state/binary`` and
``/ws/render`` are each checked.
"""

from __future__ import annotations

import json
import os
import shutil
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api.binary_encoder import decode_frame
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode


class _RecordingRenderer:
    """A ``/ws/render`` renderer that records what it was asked to draw."""

    width, height, fmt, content_type = 4, 4, "png", "image/png"

    def __init__(self) -> None:
        self.calls: list = []

    def set_format(self, fmt, quality=None) -> None:
        pass

    def reset(self) -> None:
        pass

    def render(self, sim_time, state) -> bytes:
        self.calls.append((sim_time, float(state["ball"]["position"])))
        return b"frame"


def _server(tmp_path, renderer=None):
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=100.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({}, graph_manager=gm, checkpoint_root=str(tmp_path),
                              frame_renderer=renderer)
    return gm, server, TestClient(server.create_app(), raise_server_exceptions=False)


def _saved_at_one_second_then_run_on(client, server) -> tuple[float, float]:
    """Run to t = 1 s, save, run to t = 2 s; returns the time and position
    the checkpoint holds."""
    assert client.post("/sim/run", params={"n_steps": 100}).status_code == 200
    saved_time = server.relay.latest_snapshot()[0]
    saved_position = client.get("/graph/state/ball").json()["position"]
    resp = client.post("/checkpoint/save", params={"path": "t1.npz"})
    assert resp.status_code == 200, resp.text
    assert client.post("/sim/run", params={"n_steps": 100}).status_code == 200
    return saved_time, saved_position


def test_ws_state_serves_the_loaded_state_at_the_checkpoints_time(tmp_path):
    gm, server, client = _server(tmp_path)
    saved_time, saved_position = _saved_at_one_second_then_run_on(client, server)
    resp = client.post("/checkpoint/load", params={"path": "t1.npz"})
    assert resp.status_code == 200, resp.text
    assert client.get("/graph/state/ball").json()["position"] == saved_position
    with client.websocket_connect("/ws/state") as ws:
        frame = ws.receive_json()
    assert frame["state"]["ball"]["position"] == saved_position
    assert frame["sim_time"] == saved_time
    assert resp.json()["sim_time"] == saved_time
    assert resp.json()["sim_time_from_checkpoint"] is True
    # ... and the clock goes on from there.
    assert client.post("/sim/step").status_code == 200
    with client.websocket_connect("/ws/state") as ws:
        assert ws.receive_json()["sim_time"] == saved_time + 0.01
    assert server.relay.step_count == 101


def test_ws_state_binary_serves_the_loaded_state_at_the_checkpoints_time(tmp_path):
    gm, server, client = _server(tmp_path)
    saved_time, saved_position = _saved_at_one_second_then_run_on(client, server)
    assert client.post("/checkpoint/load", params={"path": "t1.npz"}).status_code == 200
    with client.websocket_connect("/ws/state/binary") as ws:
        schema = ws.receive_json()
        sim_time, values = decode_frame(ws.receive_bytes(), schema)
    position = next(f for f in schema["fields"] if f["field"] == "position")
    assert sim_time == saved_time
    assert values[position["offset"]] == np.float32(saved_position)


def test_ws_render_draws_the_loaded_state_at_the_checkpoints_time(tmp_path):
    renderer = _RecordingRenderer()
    gm, server, client = _server(tmp_path, renderer)
    saved_time, saved_position = _saved_at_one_second_then_run_on(client, server)
    assert client.post("/checkpoint/load", params={"path": "t1.npz"}).status_code == 200
    with client.websocket_connect("/ws/render") as ws:
        ws.receive_json()               # the renderer's config
        assert ws.receive_bytes() == b"frame"
    assert renderer.calls[-1] == (saved_time, pytest.approx(saved_position))


def test_a_checkpoint_without_the_servers_manifest_restarts_the_clock_at_the_load(tmp_path):
    gm, server, client = _server(tmp_path)
    saved_time, saved_position = _saved_at_one_second_then_run_on(client, server)
    (tmp_path / "t1.npz.manifest.json").unlink()
    resp = client.post("/checkpoint/load", params={"path": "t1.npz"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["sim_time"] == 0.0
    assert resp.json()["sim_time_from_checkpoint"] is False
    with client.websocket_connect("/ws/state") as ws:
        frame = ws.receive_json()
    assert frame == {"sim_time": 0.0, "state": {"ball": {"position": saved_position,
                                                         "velocity": frame["state"]["ball"]["velocity"]}}}


def test_a_manifest_that_does_not_hash_to_its_checkpoint_is_not_believed(tmp_path):
    """A checkpoint replaced since its manifest was written (another tool
    saved over it) is loaded, but the recorded time is not its time."""
    gm, server, client = _server(tmp_path)
    _saved_at_one_second_then_run_on(client, server)
    assert client.post("/checkpoint/save", params={"path": "t2.npz"}).status_code == 200
    shutil.copy(tmp_path / "t2.npz", tmp_path / "t1.npz")      # t1's manifest is stale
    resp = client.post("/checkpoint/load", params={"path": "t1.npz"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["sim_time_from_checkpoint"] is False
    assert resp.json()["sim_time"] == 0.0


def test_the_manifest_records_the_hash_and_the_streams_clock(tmp_path):
    from maddening.core.simulation.checkpoint import verify_manifest

    gm, server, client = _server(tmp_path)
    assert client.post("/sim/run", params={"n_steps": 7}).status_code == 200
    resp = client.post("/checkpoint/save", params={"path": "c.npz"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["sim_time"] == server.relay.elapsed
    manifest = json.loads((tmp_path / "c.npz.manifest.json").read_text())
    verify_manifest(tmp_path / "c.npz", manifest)       # the hash is the file's
    assert manifest["extra"]["server_clock"] == {"sim_time": server.relay.elapsed,
                                                 "step_count": 7}


def test_a_runner_started_after_a_load_counts_on_from_the_checkpoints_time(tmp_path):
    gm, server, client = _server(tmp_path)
    saved_time, _ = _saved_at_one_second_then_run_on(client, server)
    assert client.post("/checkpoint/load", params={"path": "t1.npz"}).status_code == 200
    assert client.post("/sim/start").status_code == 200
    try:
        import time
        end = time.monotonic() + 10
        while server.relay.step_count < 105 and time.monotonic() < end:
            time.sleep(0.01)
    finally:
        assert client.post("/sim/stop").status_code == 200
    steps = server.relay.step_count - 100
    assert steps >= 5
    assert server.relay.elapsed == pytest.approx(saved_time + steps * 0.01, rel=1e-12)
