"""The signaling WebSocket admits only clients that present the session token.

These tests drive a real ``websockets`` client against a real server on a
loopback port, using the very handler ``SelkiesSession`` installs, because
the bug they pin was invisible to any test that mocked the connection: the
handler validated the *server's own* token instead of the client's, so the
rejection branch was unreachable and an anonymous client was relayed.
"""

import asyncio
import json

import pytest

websockets = pytest.importorskip(
    "websockets",
    reason="signaling needs the optional 'websockets' dependency "
           "(maddening[streaming] / maddening[api])",
)

from maddening.cloud._auth import generate_session_token  # noqa: E402
from maddening.cloud.selkies_session import (  # noqa: E402
    _client_token,
    _make_signaling_handler,
)

SESSION_ID = "0123456789ab"
SECRET = "shared-secret"
OTHER_SESSION_ID = "ba9876543210"
VALID_TOKEN = generate_session_token(SESSION_ID, SECRET)

#: Long enough that a loopback round trip cannot lose a race, short enough
#: that a hung handler fails the test instead of the suite.
TIMEOUT_S = 5.0


async def _connect(url: str, headers: dict | None = None):
    """``websockets.connect`` with the header kwarg this release accepts."""
    if not headers:
        return websockets.connect(url)
    try:
        return websockets.connect(url, additional_headers=headers)
    except TypeError:  # websockets < 14
        return websockets.connect(url, extra_headers=headers)


async def _attempt(
    *,
    token: str | None = None,
    in_header: bool = False,
    session_in_url: str = SESSION_ID,
    query: str | None = None,
) -> tuple[int | None, list]:
    """Run one client connection against a freshly served handler.

    Returns the close code the client observed (``None`` while the
    connection is still open, i.e. the client was admitted) and the
    messages the server relayed to its input handler.
    """
    relayed: list = []
    handler = _make_signaling_handler(
        SESSION_ID, SECRET, lambda: relayed.append,
    )
    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        url = f"ws://127.0.0.1:{port}/signaling/{session_in_url}"
        headers = None
        if query is not None:
            url += f"?{query}"
        elif token is not None and in_header:
            headers = {"Authorization": f"Bearer {token}"}
        elif token is not None:
            url += f"?token={token}"

        ws = await (await _connect(url, headers))
        try:
            try:
                await ws.send(json.dumps({"type": "offer", "sdp": "v=0"}))
            except websockets.exceptions.ConnectionClosed:
                # A rejected client may lose the race to send at all.
                pass
            deadline = asyncio.get_event_loop().time() + TIMEOUT_S
            while not relayed and ws.close_code is None:
                if asyncio.get_event_loop().time() > deadline:
                    raise AssertionError("handler neither relayed nor closed")
                await asyncio.sleep(0.01)
            return ws.close_code, relayed
        finally:
            await ws.close()
    finally:
        server.close()
        await server.wait_closed()


def _run(**kwargs) -> tuple[int | None, list]:
    return asyncio.run(asyncio.wait_for(_attempt(**kwargs), TIMEOUT_S * 2))


class TestSignalingClientAuthentication:
    """One real client per case, against a real server on 127.0.0.1."""

    def test_valid_token_in_query_is_admitted_and_relayed(self):
        close_code, relayed = _run(token=VALID_TOKEN)
        assert close_code is None
        assert relayed == [{"type": "offer", "sdp": "v=0"}]

    def test_valid_token_in_authorization_header_is_admitted(self):
        close_code, relayed = _run(token=VALID_TOKEN, in_header=True)
        assert close_code is None
        assert relayed == [{"type": "offer", "sdp": "v=0"}]

    def test_no_token_is_rejected(self):
        close_code, relayed = _run(token=None)
        assert close_code == 1008
        assert relayed == []

    def test_wrong_token_is_rejected(self):
        close_code, relayed = _run(token="f" * 64)
        assert close_code == 1008
        assert relayed == []

    def test_token_for_another_session_is_rejected(self):
        # The exact bypass: a token that is internally valid, just not for
        # this session.  The server's own token used to be validated here.
        other = generate_session_token(OTHER_SESSION_ID, SECRET)
        close_code, relayed = _run(token=other, session_in_url=OTHER_SESSION_ID)
        assert close_code == 1008
        assert relayed == []

    def test_token_from_another_secret_is_rejected(self):
        close_code, relayed = _run(
            token=generate_session_token(SESSION_ID, "not-the-secret"),
        )
        assert close_code == 1008
        assert relayed == []

    def test_empty_token_parameter_is_rejected(self):
        close_code, relayed = _run(query="token=")
        assert close_code == 1008
        assert relayed == []

    def test_duplicate_token_parameters_are_rejected(self):
        # Two values are a smuggling attempt, not a credential: neither is
        # picked.
        close_code, relayed = _run(query=f"token=wrong&token={VALID_TOKEN}")
        assert close_code == 1008
        assert relayed == []

    def test_non_ascii_token_is_rejected_not_crashed(self):
        # hmac.compare_digest refuses non-ASCII str, so an unguarded
        # comparison would raise inside the auth check and close 1011.
        close_code, relayed = _run(query="token=%C3%B6")
        assert close_code == 1008
        assert relayed == []


class TestClientTokenExtraction:
    """Carrier parsing, away from the socket."""

    class _Request:
        def __init__(self, path="", headers=None):
            self.path = path
            self.headers = headers or {}

    def test_query_parameter(self):
        req = self._Request(path=f"/signaling/x?token={VALID_TOKEN}")
        assert _client_token(req) == VALID_TOKEN

    def test_bearer_header_wins_over_query(self):
        req = self._Request(
            path="/signaling/x?token=from-query",
            headers={"Authorization": f"Bearer {VALID_TOKEN}"},
        )
        assert _client_token(req) == VALID_TOKEN

    def test_bearer_scheme_is_case_insensitive(self):
        req = self._Request(headers={"Authorization": f"bearer {VALID_TOKEN}"})
        assert _client_token(req) == VALID_TOKEN

    def test_non_bearer_authorization_falls_back_to_query(self):
        req = self._Request(
            path="/signaling/x?token=q",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        assert _client_token(req) == "q"

    def test_absent_everything(self):
        assert _client_token(self._Request(path="/signaling/x")) == ""
        assert _client_token(None) == ""

    def test_legacy_path_argument(self):
        # websockets < 13 passes the request target to the handler rather
        # than exposing a request object.
        assert _client_token(None, f"/signaling/x?token={VALID_TOKEN}") == VALID_TOKEN
