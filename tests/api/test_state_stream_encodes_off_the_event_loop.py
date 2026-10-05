"""``/ws/state`` encodes its frames off the event loop, once per snapshot,
for a bounded number of clients.

The stream built each frame's JSON synchronously in its async handler, on
the event loop every other request needs: with one client on a 256 x 256
lattice (15.6 MiB frames) ``GET /healthz`` took up to 0.6 s, and 2.7 s at
512 x 512 -- per client, since each client encoded the same snapshot
again, and nothing bounded the clients.  A frame is now encoded in a
worker thread, a frame of the whole state is encoded once and sent to
every client of it, and past ``MAX_STREAM_CONNECTIONS`` streams a
handshake is closed with 1013.
"""

from __future__ import annotations

import asyncio
import os
import threading
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from tests._loopback_client import LoopbackTestClient as TestClient
from starlette.websockets import WebSocketDisconnect

from maddening.api import server as server_module
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode


def _client():
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=3.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({}, graph_manager=gm)
    return gm, server, TestClient(server.create_app(), raise_server_exceptions=False)


def test_state_frames_are_encoded_off_the_event_loop(monkeypatch):
    real = server_module._json_reply
    on_loop = []

    def watching(value):
        try:
            asyncio.get_running_loop()
            on_loop.append(threading.current_thread().name)
        except RuntimeError:
            pass
        return real(value)

    monkeypatch.setattr(server_module, "_json_reply", watching)
    gm, server, client = _client()
    assert client.post("/sim/step").status_code == 200
    with client.websocket_connect("/ws/state") as ws:
        frame = ws.receive_json()
    assert frame["state"]["ball"]["position"] == pytest.approx(3.0, abs=1e-2)
    assert on_loop == [], f"a frame was encoded on the event loop ({on_loop})"


def test_a_frame_of_the_whole_state_is_encoded_once_for_every_client(monkeypatch):
    real = server_module._encode_state_frame
    encoded = []

    def counting(sim_time, state, fields):
        encoded.append(fields)
        return real(sim_time, state, fields)

    monkeypatch.setattr(server_module, "_encode_state_frame", counting)
    gm, server, client = _client()
    assert client.post("/sim/step").status_code == 200
    with client.websocket_connect("/ws/state") as a:
        first = a.receive_json()
        with client.websocket_connect("/ws/state") as b:
            assert b.receive_json() == first
    assert encoded == [None]
    # A client of part of the state gets its own frame.
    with client.websocket_connect("/ws/state") as c:
        c.receive_json()
        c.send_json({"type": "subscribe", "fields": {"ball": ["position"]}})
        assert set(c.receive_json()["state"]["ball"]) == {"position"}
    assert encoded == [None, {"ball": ["position"]}]


@pytest.mark.parametrize("third", ["/ws/state", "/ws/state/binary"])
def test_streams_past_the_cap_are_refused_with_1013(monkeypatch, third):
    monkeypatch.setattr(server_module, "MAX_STREAM_CONNECTIONS", 2)
    gm, server, client = _client()
    with client.websocket_connect("/ws/state") as a, \
            client.websocket_connect("/ws/state/binary") as b:
        b.receive_json()
        # Accepted, then closed with 1013: a real client sees the close
        # code (closed before the accept it saw HTTP 403, the answer to an
        # Origin or token refusal).
        accepted = False
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect(third) as c:
                accepted = True
                c.receive_json()
        assert accepted, "closed before the accept: a real client is sent HTTP 403"
        assert refused.value.code == 1013
    # Closing a stream frees its place.
    with client.websocket_connect(third) as c:
        if third.endswith("binary"):
            assert c.receive_json()["type"] == "schema"
    assert server._stream_connections == 0


_CAPPED_SERVER = """
import sys, types, warnings
warnings.simplefilter("ignore")
stub = types.ModuleType("maddening.cloud.session")
def _refuse(*a, **k):
    raise RuntimeError("cloud stubbed out in this test")
stub.CloudSession = stub.CloudConfig = _refuse
sys.modules["maddening.cloud.session"] = stub
import uvicorn
from maddening.api import server as server_module
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode
server_module.MAX_STREAM_CONNECTIONS = 1
gm = GraphManager()
gm.add_node(BallNode("c", timestep=1.0 / 64.0, initial_velocity=1.0, gravity=0.0))
gm.compile()
server = server_module.SimulationServer({}, graph_manager=gm, bind_host="127.0.0.1")
uvicorn.run(server.create_app(), host="127.0.0.1", port=int(sys.argv[1]),
            log_level="warning")
"""


# Per push: tests/api/test_state_stream_encodes_off_the_event_loop.py::test_streams_past_the_cap_are_refused_with_1013
@pytest.mark.slow  # starts a uvicorn server process: about 5 s
def test_a_real_client_past_the_cap_gets_close_code_1013_not_http_403(tmp_path):
    """Under uvicorn, a handshake closed before it is accepted is answered
    HTTP 403 -- the answer to an Origin or token refusal -- with no close
    code; only Starlette's test client reported 1013."""
    import socket
    import subprocess
    import sys
    import time
    from pathlib import Path

    import httpx
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect

    import maddening

    script = tmp_path / "serve.py"
    script.write_text(_CAPPED_SERVER)
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(("RUNPOD_", "AWS_", "GOOGLE_", "GCLOUD_", "SKY",
                                        "LAMBDA_", "AZURE_"))}
    env.update(HOME=str(home), JAX_PLATFORMS="cpu",
               PYTHONPATH=str(Path(maddening.__file__).resolve().parents[1])
               + os.pathsep + env.get("PYTHONPATH", ""))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([sys.executable, str(script), str(port)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1)
                break
            except httpx.HTTPError:
                assert proc.poll() is None, proc.stdout.read().decode()[-2000:]
                time.sleep(0.2)
        url = f"ws://127.0.0.1:{port}/ws/state"
        with connect(url, open_timeout=30):
            with connect(url, open_timeout=30) as refused:
                with pytest.raises(ConnectionClosed) as closed:
                    refused.recv(timeout=30)
        assert closed.value.rcvd is not None and closed.value.rcvd.code == 1013, closed.value
    finally:
        if proc.poll() is None:              # only the process this test started
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
