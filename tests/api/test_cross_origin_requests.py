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
from tests._loopback_client import LoopbackTestClient as TestClient

from maddening.api.server import SimulationServer, _host_name
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
        headers={"Origin": "http://127.0.0.1", "Host": "127.0.0.1"},
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
        "/ws/state", headers={"Origin": "http://127.0.0.1", "Host": "127.0.0.1"},
    ) as ws:
        assert ws.accepted_subprotocol is None


def test_a_websocket_with_no_origin_header_is_unaffected():
    """Non-browser clients send no ``Origin``; the local flow is unchanged."""
    client = _client()

    with client.websocket_connect("/ws/state") as ws:
        assert ws.accepted_subprotocol is None


# ----------------------------------------------------------------------
# What the Origin check cannot see: DNS rebinding (MADD-ANO-076)
# ----------------------------------------------------------------------
#
# A page served from ``http://attacker.example:8000`` whose name the
# attacker then re-points at 127.0.0.1 is same-origin in the browser's
# eyes: it sends ``Host`` and ``Origin`` that both name its own domain, so
# the comparison above passes it, and a loopback bind demands no token.
# Until the Host allowlist, ``POST /sim/reset`` and ``/checkpoint/save``
# answered 200 (and wrote the file), ``GET /graph`` 200, and ``/ws/state``
# accepted the handshake.  A loopback-bound server now answers only to the
# names this machine is reached by.  These requests come from a loopback
# TCP peer, as a browser's do (``client=``).  The check keys on the Host,
# never on the peer: Starlette's in-process ``"testclient"`` peer is asked
# too (and, not being a loopback address, must present the token).

REBOUND = "http://attacker.example:8000"
LOOPBACK_PEER = ("127.0.0.1", 51234)


def _rebound_client(tmp_path, *, base_url=REBOUND, **kwargs):
    server = SimulationServer(
        node_registry=REGISTRY, graph_manager=_graph(),
        bind_host="127.0.0.1", checkpoint_root=str(tmp_path), **kwargs,
    )
    return TestClient(server.create_app(), base_url=base_url, client=LOOPBACK_PEER,
                      raise_server_exceptions=False)


def test_a_dns_rebound_page_is_refused_by_its_host(tmp_path):
    client = _rebound_client(tmp_path)
    rebound = {"Origin": REBOUND, "Content-Type": "text/plain;charset=UTF-8"}

    for method, path in [("POST", "/sim/reset"), ("POST", "/checkpoint/save?path=rebound.npz"),
                         ("GET", "/graph"), ("GET", "/graph/state"), ("GET", "/healthz")]:
        response = client.request(method, path, headers=rebound)
        assert response.status_code == 403, (path, response.text)
        assert "allowed_hosts" in response.json()["detail"]
    assert not (tmp_path / "rebound.npz").exists()
    with pytest.raises(Exception):
        with client.websocket_connect(
            "/ws/state", headers={"Origin": REBOUND, "Host": "attacker.example:8000"},
        ):
            pass


def test_a_dns_rebound_page_cannot_reach_the_cloud_launch_route(tmp_path, monkeypatch):
    """The route that provisions paid instances, with the launcher stubbed
    to raise if it were ever called and an empty HOME."""
    import maddening.cloud.session as cloud_session

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for name in list(os.environ):
        if name.upper().startswith(("RUNPOD_", "AWS_", "GOOGLE_", "SKYPILOT_", "LAMBDA_")):
            monkeypatch.delenv(name, raising=False)
    launched = []

    class NoLaunch:
        def __init__(self, *args, **kwargs):
            launched.append(args)
            raise AssertionError("the cloud launcher was reached")

    monkeypatch.setattr(cloud_session, "CloudSession", NoLaunch)
    client = _rebound_client(tmp_path)
    response = client.post("/cloud/launch", json={},
                           headers={"Origin": REBOUND})
    assert response.status_code == 403, response.text
    assert launched == []


@pytest.mark.parametrize("base_url", [
    "http://127.0.0.1:8000", "http://localhost:8000",
    "http://localhost", "http://127.0.0.2:9000", "http://LOCALHOST.:8000",
])
def test_the_loopback_names_are_served(tmp_path, base_url):
    origin = base_url.lower().rstrip(".")
    client = _rebound_client(tmp_path, base_url=base_url)
    assert client.get("/graph").status_code == 200
    response = client.post("/sim/reset", headers={"Origin": base_url})
    assert response.status_code == 200, (base_url, origin, response.text)


def test_an_operator_can_name_more_hosts(tmp_path):
    client = _rebound_client(tmp_path, base_url="http://sim.lab.example:8443",
                             allowed_hosts=["sim.lab.example"])
    assert client.get("/graph").status_code == 200
    assert _rebound_client(tmp_path, base_url="http://other.example:8443",
                           allowed_hosts=["sim.lab.example"]).get("/graph").status_code == 403


@pytest.mark.parametrize("host", ["attacker.example", "127.0.0.1.attacker.example",
                                  "localhost.attacker.example", "localhost:http", "", " "])
def test_a_host_that_is_not_a_loopback_name_fails_closed(tmp_path, host):
    client = _rebound_client(tmp_path, base_url="http://127.0.0.1:8000")
    assert client.get("/graph", headers={"Host": host}).status_code == 403


@pytest.mark.parametrize("header, name", [
    ("localhost:8000", "localhost"), ("LocalHost.", "localhost"),
    ("[::1]:8000", "::1"), ("[::1]", "::1"), ("127.0.0.1", "127.0.0.1"),
    # not a valid host[:port]: None, which is refused
    ("::1", None), ("[::1]x", None), ("[::1", None), ("a:b:c", None),
    ("host:", None), ("host:80a", None), ("", None),
])
def test_how_a_host_header_is_read(header, name):
    """IPv6 literals are read here directly: Starlette's ``TestClient``
    cannot carry one in its URL."""
    assert _host_name(header) == name


def test_an_allowed_hosts_entry_that_is_not_a_host_is_refused():
    with pytest.raises(ValueError, match="not a host name"):
        SimulationServer(node_registry=REGISTRY, allowed_hosts=["a:b:c"])


def test_the_token_a_non_loopback_bind_demands_refuses_a_rebound_page():
    """The boundary of that hole: off loopback the bearer token is
    demanded (MADD-ANO-051), and a rebound page does not hold it."""
    server = SimulationServer(
        node_registry=REGISTRY, graph_manager=_graph(),
        bind_host="0.0.0.0", api_token="s3cret-for-this-test",
    )
    client = TestClient(server.create_app(), base_url=REBOUND)

    response = client.post(
        "/sim/reset",
        headers={"Origin": REBOUND, "Content-Type": "text/plain;charset=UTF-8"},
    )

    assert response.status_code == 401


# ----------------------------------------------------------------------
# The IPv6 loopback, and a Host that is not one
# ----------------------------------------------------------------------
#
# ``urlsplit`` reads an origin's IPv6 literal without its brackets and the
# ``Host`` header carries them, so a page served from ``http://[::1]:8000``
# was refused (403) as cross-origin by its own server.  Both sides are now
# read as (host, port) the same way; every refusal above still holds.  And
# a ``Host`` that is not a valid ``host[:port]`` (``[::1]x``) raised inside
# the authentication middleware, where ``request.url`` is built from it: a
# 500 where the refusal is a 403 (401 where a token is demanded).


def test_a_page_served_from_the_ipv6_loopback_can_change_state(tmp_path):
    """TestClient cannot carry an IPv6 literal in its URL, so the page's
    ``Host`` is set directly."""
    client = _rebound_client(tmp_path, base_url="http://127.0.0.1:8000")
    for origin in (None, "http://[::1]:8000"):
        headers = {"Host": "[::1]:8000", **({"Origin": origin} if origin else {})}
        response = client.post("/sim/reset", headers=headers)
        assert response.status_code == 200, (origin, response.text)


def test_a_page_served_from_the_ipv6_loopback_can_open_the_state_stream(tmp_path):
    client = _rebound_client(tmp_path, base_url="http://127.0.0.1:8000")
    with client.websocket_connect("/ws/state", headers={
            "Host": "[::1]:8000", "Origin": "http://[::1]:8000"}):
        pass


@pytest.mark.parametrize("origin, host, same", [
    ("http://[::1]:8000", "[::1]:8000", True),
    ("http://[::1]", "[::1]", True),
    ("http://[::ffff:127.0.0.1]:8000", "[::ffff:127.0.0.1]:8000", True),
    ("https://[::1]:8000", "[::1]:8000", True),           # the scheme is ignored, as before
    ("http://127.0.0.1:08000", "127.0.0.1:8000", True),   # as before
    ("http://[::1]:8000", "[::1]:8001", False),
    ("http://[::1]:8000", "127.0.0.1:8000", False),
    ("http://127.0.0.1:8000", "[::1]:8000", False),
    ("http://[::1]:8000", "::1:8000", False),              # a bare IPv6 Host is invalid
    ("http://[::1]:8000", "[::1]x", False),
    ("http://[::1", "[::1]", False),                       # urlsplit raises on it
    ("http://127.0.0.1", "127.0.0.1:8000", False),
    ("http://localhost.:8000", "localhost:8000", False),
    ("http://evil.example@[::1]:8000", "[::1]:8000", False),
])
def test_an_origin_and_a_host_compare_as_host_and_port(origin, host, same):
    from maddening.api.server import origin_is_same_site

    assert origin_is_same_site(origin, host) is same


@pytest.mark.parametrize("host", ["[::1]x", "[::1", "[]:80", "[::1]:http"])
def test_a_malformed_host_on_a_loopback_bind_is_a_403_not_a_500(tmp_path, host):
    client = _rebound_client(tmp_path, base_url="http://127.0.0.1:8000")
    for method, path in [("GET", "/graph"), ("POST", "/sim/reset")]:
        response = client.request(method, path, headers={"Host": host})
        assert response.status_code == 403, (host, path, response.text)


@pytest.mark.parametrize("host", ["[::1]x", "[::1"])
def test_a_malformed_host_where_a_token_is_demanded_is_a_401_not_a_500(host):
    server = SimulationServer(node_registry=REGISTRY, graph_manager=_graph(),
                              bind_host="0.0.0.0", api_token="s3cret-for-this-test")
    client = TestClient(server.create_app(), raise_server_exceptions=False)
    response = client.get("/graph", headers={"Host": host})
    assert response.status_code == 401, response.text
    with pytest.raises(Exception):
        with client.websocket_connect("/ws/state", headers={"Host": host}):
            pass


@pytest.mark.parametrize("method, path, body", [
    ("DELETE", "/graph/nodes/spring", None),
    ("DELETE", "/graph/edges", {"source_node": "spring", "target_node": "spring",
                                "source_field": "position", "target_field": "anchor_position"}),
])
def test_a_cross_origin_delete_is_refused_and_removes_nothing(method, path, body):
    """DELETE is one of the state-changing methods the Origin check covers:
    a page on another origin cannot remove a node or an edge."""
    server = SimulationServer(node_registry=REGISTRY, graph_manager=_graph(),
                              bind_host="127.0.0.1")
    client = TestClient(server.create_app(), raise_server_exceptions=False)
    nodes, edges = list(server.gm._nodes), list(server.gm._edges)
    response = client.request(method, path, json=body, headers={"Origin": EVIL})
    assert response.status_code == 403, response.text
    assert list(server.gm._nodes) == nodes and list(server.gm._edges) == edges
    # The same request with no Origin (a script, not a page) goes through.
    assert client.request("DELETE", "/graph/nodes/spring").status_code == 200
