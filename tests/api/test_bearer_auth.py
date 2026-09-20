"""A non-loopback bind must never serve without a bearer token.

The API provisions paid cloud GPUs (``POST /cloud/launch``) and rewrites
the simulation graph, and every shipped container path binds it to
``0.0.0.0``.  These tests pin both halves of the rule:

* a loopback bind is served exactly as it always was -- no token, no
  header, nothing to configure.  That is the compatibility constraint;
  breaking it breaks every local development flow.
* anything else demands ``Authorization: Bearer <token>`` on *every*
  route but the handful that are exempt on purpose.

``test_every_route_refuses_an_anonymous_caller`` enumerates the app's
routes rather than listing them, so a route added later is covered
without anybody remembering to add it here.
"""

import logging
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from fastapi.testclient import TestClient
from starlette.routing import Route

from maddening.api.auth import (
    TOKEN_ENV,
    UNAUTHENTICATED_PATHS,
    WS_SUBPROTOCOL,
    APIAuth,
    encode_ws_bearer,
    is_loopback,
    is_routable_peer,
    websocket_credentials,
)
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

TOKEN = "test-token-not-a-real-credential"
REGISTRY = {"SpringDamperNode": SpringDamperNode}

#: A peer address that is not this machine.  TEST-NET-3 (RFC 5737).
REMOTE_PEER = ("203.0.113.5", 44321)


def _graph():
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="spring", timestep=0.01))
    return gm


def _server(bind_host, token=TOKEN, frame_renderer=None):
    return SimulationServer(
        node_registry=REGISTRY,
        graph_manager=_graph(),
        bind_host=bind_host,
        api_token=token,
        frame_renderer=frame_renderer,
    )


@pytest.fixture
def public_client():
    """A client of a server that believes it is bound to 0.0.0.0."""
    return TestClient(_server("0.0.0.0").create_app())


@pytest.fixture
def loopback_client():
    """A client of a server bound to 127.0.0.1: no token anywhere."""
    return TestClient(_server("127.0.0.1").create_app())


def _bearer(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


# ----------------------------------------------------------------------
# The compatibility constraint
# ----------------------------------------------------------------------

def test_loopback_bind_serves_every_route_without_a_token(loopback_client):
    """Local development is unchanged: no header, no 401, anywhere."""
    assert loopback_client.get("/graph").status_code == 200
    assert loopback_client.get("/graph/state").status_code == 200
    assert loopback_client.post("/sim/step").status_code == 200
    assert loopback_client.get("/viz/app").status_code == 200
    # The interactive docs are a local-development tool and stay on.
    assert loopback_client.get("/docs").status_code == 200
    assert loopback_client.get("/openapi.json").status_code == 200


def test_loopback_bind_streams_state_without_a_token(loopback_client):
    with loopback_client.websocket_connect("/ws/state") as ws:
        assert ws.accepted_subprotocol is None


def test_loopback_bind_does_not_reject_a_client_that_offers_a_token(loopback_client):
    """A client that always sends the header is not punished for it."""
    assert loopback_client.get("/graph", headers=_bearer("anything")).status_code == 200


# ----------------------------------------------------------------------
# The gate
# ----------------------------------------------------------------------

def _http_routes(app):
    """(method, concrete path) for every HTTP route the app serves."""
    out = []
    for route in app.routes:
        if not isinstance(route, Route):
            continue
        path = route.path
        for name in getattr(route, "param_convertors", {}) or {}:
            path = path.replace("{" + name + "}", "probe")
        for method in sorted(route.methods or set()):
            if method in {"HEAD", "OPTIONS"}:
                continue
            out.append((method, path, route.path))
    return out


def test_every_route_refuses_an_anonymous_caller(public_client):
    """The gate.  Enumerated, so a new route is covered automatically.

    A route that answers anything but 401 here is a route an anonymous
    caller on the internet can reach, because every shipped container
    path binds 0.0.0.0.  The only permitted exceptions are the paths
    :data:`UNAUTHENTICATED_PATHS` names on purpose.
    """
    routes = _http_routes(public_client.app)
    assert len(routes) > 30, f"route enumeration found only {len(routes)}"
    served = []
    for method, path, template in routes:
        if template in UNAUTHENTICATED_PATHS:
            continue
        response = public_client.request(method, path)
        if response.status_code != 401:
            served.append((method, template, response.status_code))
    assert not served, (
        "these routes answered an anonymous caller on a non-loopback bind: "
        f"{served}"
    )


def test_the_exempt_paths_are_exactly_these(public_client):
    """Adding an exemption has to be a deliberate, visible act."""
    assert UNAUTHENTICATED_PATHS == frozenset({
        "/healthz", "/viz/app", "/viz/graph", "/viz/render", "/viz/auth.js",
    })
    for path in UNAUTHENTICATED_PATHS:
        assert public_client.get(path).status_code == 200, path


def test_an_anonymous_caller_cannot_launch_cloud_instances(public_client):
    """The route the whole exercise is about."""
    response = public_client.post("/cloud/launch", json={})
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_the_token_unlocks_the_routes(public_client):
    assert public_client.get("/graph", headers=_bearer()).status_code == 200
    assert public_client.post("/sim/step", headers=_bearer()).status_code == 200


@pytest.mark.parametrize("presented", [
    "", "wrong", TOKEN + "x", TOKEN[:-1], TOKEN.upper(), TOKEN + " x",
])
def test_a_wrong_token_is_refused(public_client, presented):
    assert public_client.get("/graph", headers=_bearer(presented)).status_code == 401


def test_extra_whitespace_after_the_scheme_is_tolerated(public_client):
    """RFC 7235 allows bad whitespace between the scheme and the value.

    A non-ASCII token cannot be exercised through a client -- httpx
    refuses to encode the header -- so that case lives in
    ``test_verify_survives_anything_a_client_can_send``.
    """
    response = public_client.get(
        "/graph", headers={"Authorization": f"Bearer   {TOKEN}  "},
    )
    assert response.status_code == 200


@pytest.mark.parametrize("header", [
    f"Basic {TOKEN}", TOKEN, f"bearer{TOKEN}", "Bearer",
])
def test_a_malformed_authorization_header_is_refused(public_client, header):
    response = public_client.get("/graph", headers={"Authorization": header})
    assert response.status_code == 401


def test_a_query_string_token_does_not_authenticate(public_client):
    """No authenticated HTTP route accepts ``?token=``.

    Query strings reach access logs, proxy logs and Referer headers.  The
    only place a token legitimately appears in a URL is the *page* URL of
    an exempt ``/viz/*`` route, which the page then scrubs.
    """
    assert public_client.get(f"/graph?token={TOKEN}").status_code == 401


def test_interactive_docs_are_not_served_when_the_token_is_enforced(public_client):
    """Swagger UI fetches its own schema with no Authorization header.

    The anonymous answer is 401 rather than 404 because the check runs
    before routing -- which is also why it does not disclose whether the
    route exists.  With the token, the route really is gone.
    """
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert public_client.get(path).status_code == 401, path
        assert public_client.get(path, headers=_bearer()).status_code == 404, path


def test_healthz_says_nothing_about_the_graph(public_client):
    body = public_client.get("/healthz").json()
    assert body["status"] == "ok"
    assert set(body) == {"status", "version"}


def test_the_viz_pages_never_contain_the_token(public_client):
    """The pages are exempt *because* they hold no secret."""
    for path in ("/viz/app", "/viz/graph", "/viz/render", "/viz/auth.js"):
        assert TOKEN not in public_client.get(path).text, path


# ----------------------------------------------------------------------
# The peer backstop
# ----------------------------------------------------------------------

def test_a_routable_peer_is_challenged_even_on_a_believed_loopback_bind():
    """The app cannot see the socket uvicorn binds.

    ``uvicorn.run(app, host="0.0.0.0")`` never tells the app, so a server
    that was told nothing would serve the world.  The peer address is the
    second, independent reason to demand a token.
    """
    client = TestClient(_server("127.0.0.1").create_app(), client=REMOTE_PEER)
    assert client.get("/graph").status_code == 401
    assert client.get("/graph", headers=_bearer()).status_code == 200


def test_a_loopback_peer_is_not_challenged_on_a_loopback_bind():
    client = TestClient(_server("127.0.0.1").create_app(), client=("127.0.0.1", 5))
    assert client.get("/graph").status_code == 200


def test_the_backstop_explains_the_misconfiguration():
    client = TestClient(_server("127.0.0.1").create_app(), client=REMOTE_PEER)
    detail = client.get("/graph").json()["detail"]
    assert "bind_host" in detail and "MADDENING_HOST" in detail


# ----------------------------------------------------------------------
# WebSockets
# ----------------------------------------------------------------------

WS_PATHS = ["/ws/state", "/ws/state/binary"]


@pytest.mark.parametrize("path", WS_PATHS)
def test_a_websocket_without_a_credential_is_refused(public_client, path):
    with pytest.raises(Exception) as excinfo:
        with public_client.websocket_connect(path):
            pass
    assert "1008" in str(excinfo.value) or "Disconnect" in type(excinfo.value).__name__


@pytest.mark.parametrize("path", WS_PATHS)
def test_a_websocket_authenticates_with_the_authorization_header(public_client, path):
    """The carrier a non-browser client should use."""
    with public_client.websocket_connect(path, headers=_bearer()) as ws:
        assert ws.accepted_subprotocol is None


@pytest.mark.parametrize("path", WS_PATHS)
def test_a_websocket_authenticates_with_the_subprotocol(public_client, path):
    """The carrier a browser must use: it cannot set a header.

    RFC 6455 requires the server to select one of the offered names, so
    the client offers a second, real one and the server echoes it.  A
    browser aborts the connection when the server selects none.
    """
    with public_client.websocket_connect(
        path, subprotocols=websocket_credentials(TOKEN),
    ) as ws:
        assert ws.accepted_subprotocol == WS_SUBPROTOCOL


@pytest.mark.parametrize("path", WS_PATHS)
def test_a_websocket_with_a_wrong_subprotocol_token_is_refused(public_client, path):
    with pytest.raises(Exception):
        with public_client.websocket_connect(
            path, subprotocols=[encode_ws_bearer("wrong"), WS_SUBPROTOCOL],
        ):
            pass


@pytest.mark.parametrize("path", WS_PATHS)
def test_a_websocket_subprotocol_that_is_not_base64_is_refused(public_client, path):
    """A client must not be able to raise inside the check."""
    with pytest.raises(Exception):
        with public_client.websocket_connect(
            path, subprotocols=["maddening.bearer.!!!not!base64!!!", WS_SUBPROTOCOL],
        ):
            pass


def test_the_render_websocket_authenticates_before_it_admits_it_has_no_renderer():
    """Whether a renderer is configured is not anonymous-readable."""
    client = TestClient(_server("0.0.0.0", frame_renderer=None).create_app())
    with pytest.raises(Exception) as excinfo:
        with client.websocket_connect("/ws/render"):
            pass
    assert "renderer" not in str(excinfo.value).lower()


def test_a_loopback_websocket_still_gets_its_subprotocol_echoed(loopback_client):
    """The handshake contract does not depend on whether auth is on."""
    with loopback_client.websocket_connect(
        "/ws/state", subprotocols=[WS_SUBPROTOCOL],
    ) as ws:
        assert ws.accepted_subprotocol == WS_SUBPROTOCOL


# ----------------------------------------------------------------------
# Token sourcing
# ----------------------------------------------------------------------

def test_the_token_comes_from_the_environment():
    auth = APIAuth(bind_host="0.0.0.0", environ={TOKEN_ENV: "from-env"})
    assert auth.token == "from-env" and auth.generated is False


def test_a_generated_token_has_real_entropy():
    tokens = {APIAuth(bind_host="0.0.0.0", environ={}).token for _ in range(8)}
    assert len(tokens) == 8
    assert all(len(t) >= 40 for t in tokens)


@pytest.mark.parametrize("blank", ["", " ", "\t", "\n"])
def test_a_blank_token_refuses_to_start(blank):
    """``MADDENING_API_TOKEN=$UNSET_VARIABLE`` is a mistake, not an opt-out.

    Reading a blank value as "authentication off" would rebuild exactly
    the hole this module closes, silently, in the configuration most
    likely to be public.
    """
    with pytest.raises(ValueError, match="blank"):
        APIAuth(bind_host="0.0.0.0", environ={TOKEN_ENV: blank})
    with pytest.raises(ValueError, match="blank"):
        APIAuth(bind_host="127.0.0.1", environ={TOKEN_ENV: blank})
    with pytest.raises(ValueError, match="blank"):
        SimulationServer(node_registry={}, bind_host="0.0.0.0", api_token=blank)


def test_a_generated_token_is_logged_once_and_only_when_it_is_needed(caplog):
    public = APIAuth(bind_host="0.0.0.0", environ={})
    with caplog.at_level(logging.WARNING, logger="maddening.api.auth"):
        assert public.announce(8000) is True
        assert public.announce(8000) is False
    assert public.token in caplog.text

    caplog.clear()
    local = APIAuth(bind_host="127.0.0.1", environ={})
    with caplog.at_level(logging.WARNING, logger="maddening.api.auth"):
        assert local.announce(8000) is False
    assert local.token not in caplog.text


def test_a_configured_token_is_not_echoed_into_the_log(caplog):
    """Announcing a secret the operator already holds only spreads it."""
    auth = APIAuth(bind_host="0.0.0.0", environ={TOKEN_ENV: "operator-chose-this"})
    with caplog.at_level(logging.WARNING, logger="maddening.api.auth"):
        auth.announce(8000)
    assert "operator-chose-this" not in caplog.text
    assert TOKEN_ENV in caplog.text


def test_a_generated_token_can_be_written_to_a_file(tmp_path, monkeypatch):
    """For a container whose log nobody reads."""
    import stat as _stat

    path = tmp_path / "token"
    monkeypatch.setenv("MADDENING_API_TOKEN_FILE", str(path))
    auth = APIAuth(bind_host="0.0.0.0", environ={})
    auth.announce(8000)
    assert path.read_text().strip() == auth.token
    assert _stat.S_IMODE(path.stat().st_mode) == 0o600


def test_an_unwritable_token_file_does_not_stop_the_server(tmp_path, monkeypatch):
    monkeypatch.setenv("MADDENING_API_TOKEN_FILE", str(tmp_path / "no" / "such" / "dir"))
    auth = APIAuth(bind_host="0.0.0.0", environ={})
    assert auth.announce(8000) is True


def test_the_bind_host_falls_back_to_the_environment():
    assert APIAuth(environ={"MADDENING_HOST": "0.0.0.0"}).enforced is True
    assert APIAuth(environ={"MADDENING_HOST": "127.0.0.1"}).enforced is False
    # An embedder who set nothing gets the safe-for-them loopback answer,
    # which the peer backstop then covers.
    assert APIAuth(environ={}).enforced is False


# ----------------------------------------------------------------------
# The comparison itself
# ----------------------------------------------------------------------

def test_the_comparison_is_constant_time():
    """``==`` on a secret leaks its prefix to a timing attack."""
    import inspect

    source = inspect.getsource(APIAuth.verify)
    assert "hmac.compare_digest" in source
    assert "==" not in source


def test_verify_survives_anything_a_client_can_send():
    auth = APIAuth(bind_host="0.0.0.0", token=TOKEN, environ={})
    for presented in ["", None, "éè", "\ud800", "\x00", TOKEN + "\x00"]:
        assert auth.verify(presented) is False
    assert auth.verify(TOKEN) is True


# ----------------------------------------------------------------------
# Address classification
# ----------------------------------------------------------------------

@pytest.mark.parametrize("host,loopback", [
    ("127.0.0.1", True), ("127.1.2.3", True), ("localhost", True),
    ("::1", True), ("[::1]", True), ("::ffff:127.0.0.1", True),
    ("LOCALHOST", True), (" 127.0.0.1 ", True),
    ("0.0.0.0", False), ("::", False), ("192.168.1.10", False),
    ("10.0.0.1", False), ("8.8.8.8", False), ("example.com", False),
    ("", False), (None, False), ("127.0.0.1.evil.com", False),
])
def test_loopback_classification(host, loopback):
    assert is_loopback(host) is loopback


@pytest.mark.parametrize("peer,routable", [
    ("203.0.113.5", True), ("10.0.0.1", True), ("2001:db8::1", True),
    ("127.0.0.1", False), ("::1", False),
    # Not IPs: an in-process or Unix-socket transport, not a remote peer.
    ("testclient", False), ("", False), (None, False),
])
def test_routable_peer_classification(peer, routable):
    assert is_routable_peer(peer) is routable
