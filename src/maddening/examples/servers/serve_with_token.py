#!/usr/bin/env python3
"""Serve the REST API with its bearer token: refused without it, served with it.

A server reachable from another machine asks every request for
``Authorization: Bearer <token>``.  This example runs that configuration
end to end without publishing anything: it serves a small graph with
uvicorn **in this process, on a loopback port the OS picks**, and tells
the server the address a deployment would bind (``bind_host="0.0.0.0"``),
which is what switches the token on.  The socket itself stays on
``127.0.0.1``, so nothing is reachable from another machine.  Then:

1. ``GET /healthz`` answers without a token (a container probe holds no
   credential).
2. ``GET /graph`` without a token is **refused** (401, with the reason),
   and so is a wrong token.
3. The same request with ``Authorization: Bearer <token>`` is **served**,
   and so are a run and a checkpoint save.
4. A checkpoint path is relative to ``checkpoint_root``; one that climbs
   out of it is refused (400) and nothing is written.
5. With no ``MADDENING_API_TOKEN`` set, the server generates a token and
   logs it once at start-up (``server.auth.announce(port)``); that token
   is the one it accepts.

The token in part 1 comes from ``MADDENING_API_TOKEN``, set inside this
process only to an obviously fake value.  The client is the standard
library's ``urllib``.  The guide page is
``docs/user_guide/serving_the_rest_api.md``.  This example never requests
a ``/cloud``, ``/surrogate`` or ``/ws`` route.

Usage
-----
    pip install "maddening[api]"   # fastapi, uvicorn
    python -m maddening.examples.servers.serve_with_token
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import uvicorn

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode, TableNode

REGISTRY = {"BallNode": BallNode, "TableNode": TableNode}

#: An obviously fake token, assembled at run time.  Use a long random one
#: in a deployment (``python -c "import secrets; print(secrets.token_urlsafe(32))"``).
FAKE_TOKEN = "-".join(("example", "token", "not", "a", "secret"))


def build_graph() -> GraphManager:
    """A ball bouncing on a table."""
    gm = GraphManager()
    gm.add_node(TableNode("table", 0.01, position=0.0))
    gm.add_node(BallNode("ball", 0.01, initial_position=2.0, elasticity=0.7))
    gm.add_edge("table", "ball", source_field="position", target_field="table_position")
    gm.compile()
    return gm


class LoopbackServer:
    """uvicorn serving *app* in a background thread on 127.0.0.1:<free port>."""

    def __init__(self, server: SimulationServer) -> None:
        self.server = server
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        # proxy_headers=False: never let a forwarded header stand in for
        # the peer address (the library's own launch paths do the same).
        config = uvicorn.Config(server.create_app(), log_level="warning",
                                proxy_headers=False)
        self.uvicorn = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.uvicorn.run,
                                       kwargs={"sockets": [self.sock]}, daemon=True)

    def __enter__(self) -> "LoopbackServer":
        self.thread.start()
        deadline = time.monotonic() + 30.0
        while not self.uvicorn.started:
            if not self.thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("uvicorn did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc) -> None:
        # A server run in another thread asks its own long work to stop
        # first, then tells uvicorn to exit.
        self.server.request_shutdown()
        self.uvicorn.should_exit = True
        self.thread.join(timeout=30.0)
        self.sock.close()
        assert not self.thread.is_alive(), "uvicorn did not stop"


def request(url: str, method: str = "GET", token: str | None = None) -> tuple[int, dict]:
    """One HTTP request with the standard library; returns (status, JSON body)."""
    req = urllib.request.Request(url, method=method)
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def part_one(checkpoint_root: str) -> None:
    previous = os.environ.get("MADDENING_API_TOKEN")
    os.environ["MADDENING_API_TOKEN"] = FAKE_TOKEN       # this process only
    try:
        server = SimulationServer(node_registry=REGISTRY, graph_manager=build_graph(),
                                  checkpoint_root=checkpoint_root, bind_host="0.0.0.0")
    finally:
        if previous is None:
            del os.environ["MADDENING_API_TOKEN"]
        else:
            os.environ["MADDENING_API_TOKEN"] = previous
    assert server.auth.enforced and not server.auth.generated
    token = server.auth.token

    with LoopbackServer(server) as live:
        section(f"1. Serving on {live.url} (told bind_host='0.0.0.0', so the token is on)")
        status, body = request(f"{live.url}/healthz")
        print(f"GET /healthz, no token        -> {status} {body}")
        assert status == 200 and body["status"] == "ok"

        section("2. Without the token, or with a wrong one, a route is refused")
        status, body = request(f"{live.url}/graph")
        print(f"GET /graph, no token          -> {status} {body['detail']}")
        assert status == 401 and "Authorization: Bearer" in body["detail"]
        status, _ = request(f"{live.url}/graph", token="not-the-token")
        print(f"GET /graph, a wrong token     -> {status}")
        assert status == 401

        section("3. With 'Authorization: Bearer <token>' it is served")
        status, body = request(f"{live.url}/graph", token=token)
        names = sorted(n["name"] for n in body["nodes"])
        print(f"GET /graph, the token         -> {status}, nodes {names}")
        assert status == 200 and names == ["ball", "table"]
        status, body = request(f"{live.url}/sim/run?n_steps=10", "POST", token=token)
        print(f"POST /sim/run?n_steps=10      -> {status}, ball at "
              f"{body['ball']['position']:.4f} m")
        assert status == 200

        section("4. Checkpoint paths stay under checkpoint_root")
        status, body = request(f"{live.url}/checkpoint/save?path=run1.npz", "POST",
                               token=token)
        print(f"POST /checkpoint/save?path=run1.npz      -> {status} {body['status']}")
        assert status == 200 and (Path(checkpoint_root) / "run1.npz").is_file()
        status, body = request(f"{live.url}/checkpoint/save?path=../escape.npz", "POST",
                               token=token)
        print(f"POST /checkpoint/save?path=../escape.npz -> {status} {body['detail']}")
        assert status == 400
        assert not (Path(checkpoint_root).parent / "escape.npz").exists()
        print(f"  files under the root: {sorted(p.name for p in Path(checkpoint_root).iterdir())}")


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def part_two(checkpoint_root: str) -> None:
    section("5. No MADDENING_API_TOKEN: a generated token, logged once at start-up")
    previous = os.environ.pop("MADDENING_API_TOKEN", None)
    try:
        server = SimulationServer(node_registry=REGISTRY, graph_manager=build_graph(),
                                  checkpoint_root=checkpoint_root, bind_host="0.0.0.0")
    finally:
        if previous is not None:
            os.environ["MADDENING_API_TOKEN"] = previous
    assert server.auth.generated

    capture = _Capture()
    auth_log = logging.getLogger("maddening.api.auth")
    auth_log.addHandler(capture)
    propagate, auth_log.propagate = auth_log.propagate, False   # keep it off the console
    try:
        with LoopbackServer(server) as live:
            assert server.auth.announce(live.port)            # what a deployment calls
            assert not server.auth.announce(live.port)        # once only
            logged = "\n".join(capture.messages)
            # The operator copies the token from that log line; it is the
            # value server.auth.token holds.
            assert server.auth.token in logged
            assert f"curl -H 'Authorization: Bearer {server.auth.token}'" in logged
            print(f"  the start-up log holds a {len(server.auth.token)}-character token "
                  f"({server.auth.token[:4]}...) and a curl line that uses it")
            status, _ = request(f"{live.url}/graph")
            print(f"GET /graph, no token             -> {status}")
            assert status == 401
            status, _ = request(f"{live.url}/graph", token=server.auth.token)
            print(f"GET /graph, the logged token     -> {status}")
            assert status == 200
    finally:
        auth_log.removeHandler(capture)
        auth_log.propagate = propagate


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="maddening_token_demo_") as tmp:
        root = Path(tmp) / "checkpoints"
        root.mkdir()
        part_one(str(root))
        part_two(str(root))
    print()
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
