"""A web page the developer visits is not an authorised caller.

On a loopback bind the API is unauthenticated "exactly as it always
was", and the documented threat model is "the caller is anyone with a
shell on this box".  That omits every page the developer's browser
loads.  A cross-origin **simple** request -- ``POST`` with
``Content-Type: text/plain``, which needs no preflight -- reaches the
handler from any origin, and the peer backstop cannot help because the
peer *is* 127.0.0.1.  Measured before the fix: ``/sim/reset``,
``/sim/stop``, ``/graph/compile`` and ``/sim/run`` all answered 200, and
``POST /checkpoint/save?path=csrf.npz`` really wrote a file.  With DNS
rebinding the same is reachable from an arbitrary remote attacker.

The attacker cannot read the responses -- there is no
``Access-Control-Allow-Origin`` -- so the HTTP half is write-only.  The
WebSocket half is not: WebSocket is exempt from the same-origin policy,
so a page could open ``/ws/state`` on a loopback bind and *read* the
full simulation state.

The rule these pin: a request carrying an ``Origin`` that is not this
server's own origin is refused if it changes state or opens a
WebSocket.  A loopback-bound API has no legitimate cross-origin caller;
an embedder that serves a UI from another port names it in
``allowed_origins``.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

REGISTRY = {"SpringDamperNode": SpringDamperNode}
EVIL = "https://evil.example"

#: A cross-origin POST a browser will send with no preflight at all.
SIMPLE_POST = {"Content-Type": "text/plain;charset=UTF-8", "Origin": EVIL}

#: Routes a page could usefully attack: they change state or spend time.
STATE_CHANGING = [
    "/sim/reset", "/sim/start", "/sim/stop", "/graph/compile", "/sim/step",
]


def _graph():
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="spring", timestep=0.01))
    return gm


def _client(bind_host="127.0.0.1", checkpoint_root=None, allowed_origins=None):
    server = SimulationServer(
        node_registry=REGISTRY,
        graph_manager=_graph(),
        bind_host=bind_host,
        checkpoint_root=checkpoint_root,
        allowed_origins=allowed_origins,
    )
    return TestClient(server.create_app())


@pytest.mark.parametrize("path", STATE_CHANGING)
def test_a_cross_origin_state_change_is_refused(path):
    """The gate.  ``text/plain`` is what makes this need no preflight."""
    client = _client()

    response = client.post(path, headers=SIMPLE_POST)

    assert response.status_code == 403, path


def test_a_cross_origin_checkpoint_write_never_reaches_the_filesystem(tmp_path):
    """The one that wrote a real file during the audit."""
    client = _client(checkpoint_root=str(tmp_path))

    response = client.post("/checkpoint/save?path=csrf.npz", headers=SIMPLE_POST)

    assert response.status_code == 403
    assert list(tmp_path.glob("*")) == []


def test_the_same_request_from_no_origin_at_all_is_served(tmp_path):
    """The control.

    Everything above proves nothing if the route is simply broken.  A
    caller with no ``Origin`` header -- curl, a script, the local
    development flow -- is unaffected, and the checkpoint really is
    written.
    """
    client = _client(checkpoint_root=str(tmp_path))

    assert client.post("/sim/reset").status_code == 200
    assert client.post(
        "/checkpoint/save?path=allowed.npz",
        headers={"Content-Type": "text/plain;charset=UTF-8"},
    ).status_code == 200
    assert (tmp_path / "allowed.npz").exists()


def test_the_bundled_pages_own_origin_is_served():
    """A same-origin POST carries an ``Origin`` in Chrome; it must pass."""
    client = _client()

    response = client.post(
        "/sim/reset",
        headers={"Origin": "http://testserver", "Host": "testserver"},
    )

    assert response.status_code == 200


def test_a_cross_origin_read_is_left_alone():
    """Scope, stated on purpose.

    A cross-origin ``GET`` cannot be read back without
    ``Access-Control-Allow-Origin``, which this API never sends, so it is
    not an exfiltration path -- and refusing it would break embedding the
    viz pages.  The rule is about state changes and WebSockets.
    """
    client = _client()

    assert client.get("/graph", headers={"Origin": EVIL}).status_code == 200
    assert "access-control-allow-origin" not in {
        k.lower() for k in client.get("/graph", headers={"Origin": EVIL}).headers
    }


def test_an_embedder_can_name_the_origin_it_serves_its_ui_from():
    client = _client(allowed_origins=["https://ui.example"])

    assert client.post(
        "/sim/reset", headers={"Origin": "https://ui.example"},
    ).status_code == 200
    assert client.post("/sim/reset", headers={"Origin": EVIL}).status_code == 403


@pytest.mark.parametrize("origin", [
    "null", "not a url", "https://", "//evil.example", "evil.example",
    "https://evil.example@testserver", "https://testserver.evil.example",
])
def test_an_origin_that_cannot_be_matched_fails_closed(origin):
    """An ``Origin`` we cannot resolve to this host is not this host.

    ``null`` is what a sandboxed iframe or a ``file://`` page sends, and
    a userinfo-prefixed authority is the classic way to make a foreign
    origin read like a local one.
    """
    client = _client()

    assert client.post("/sim/reset", headers={"Origin": origin}).status_code == 403


def test_the_refusal_says_what_to_do():
    client = _client()

    detail = client.post("/sim/reset", headers={"Origin": EVIL}).json()["detail"]

    assert "Origin" in detail and "allowed_origins" in detail


# ----------------------------------------------------------------------
# WebSockets: the read half
# ----------------------------------------------------------------------

def test_a_cross_origin_page_cannot_open_the_state_stream():
    """WebSocket is exempt from the same-origin policy.

    On a loopback bind no token is required, so before this a page on any
    origin could open ``ws://127.0.0.1:8000/ws/state`` and *read* the
    full simulation state -- a disclosure, not merely a write.
    """
    client = _client()

    with pytest.raises(Exception):
        with client.websocket_connect("/ws/state", headers={"Origin": EVIL}):
            pass


def test_a_same_origin_page_can_still_open_the_state_stream():
    """The control: the refusal above is the origin, not a broken route."""
    client = _client()

    with client.websocket_connect(
        "/ws/state", headers={"Origin": "http://testserver", "Host": "testserver"},
    ) as ws:
        assert ws.accepted_subprotocol is None


def test_a_websocket_with_no_origin_header_is_unaffected():
    """Non-browser clients send no ``Origin``; the local flow is unchanged."""
    client = _client()

    with client.websocket_connect("/ws/state") as ws:
        assert ws.accepted_subprotocol is None
