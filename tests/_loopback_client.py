"""An in-process client that reaches a SimulationServer as a local process does.

Starlette's ``TestClient`` reports the peer ``"testclient"`` and sends the
Host ``testserver``.  Neither names this machine, so a loopback-bound
``SimulationServer`` demands the bearer token of it -- a peer that is not a
loopback IP literal is unknown, because under uvicorn's ``proxy_headers``
any request arriving over loopback can name its own peer -- and, without
a valid token, refuses its Host (the DNS-rebinding check).  No allowance
for the in-process client exists in the library, so none can be reached by
a request that forges its peer or its Host.

The tests that drive the API as a local developer does construct this
client instead: Host ``127.0.0.1`` and peer ``("127.0.0.1", 50000)``,
unless the test passes its own ``base_url`` or ``client``.  A test of the
other cases (a routable peer, a foreign Host, Starlette's own defaults)
passes them explicitly.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence
from urllib.parse import urljoin

from fastapi.testclient import TestClient as _StarletteTestClient

#: The base URL a local client uses: its Host header is ``127.0.0.1``.
LOOPBACK_BASE_URL = "http://127.0.0.1"

#: The peer a direct loopback connection reports.
LOOPBACK_CLIENT = ("127.0.0.1", 50000)

#: Starlette's own defaults, for a test of the in-process client as such.
STARLETTE_BASE_URL = "http://testserver"
STARLETTE_CLIENT = ("testclient", 50000)


class LoopbackTestClient(_StarletteTestClient):
    """Starlette's ``TestClient`` with a loopback Host and peer by default."""

    def __init__(self, app: Any, base_url: str = LOOPBACK_BASE_URL, *args: Any,
                 **kwargs: Any) -> None:
        kwargs.setdefault("client", LOOPBACK_CLIENT)
        super().__init__(app, base_url, *args, **kwargs)

    def websocket_connect(self, url: str, subprotocols: Optional[Sequence[str]] = None,
                          **kwargs: Any):
        """Starlette joins a relative WebSocket URL to ``ws://testserver``
        whatever the base URL; this joins it to the client's own host."""
        if "://" not in url:
            scheme = "wss" if self.base_url.scheme == "https" else "ws"
            url = urljoin(f"{scheme}://{self.base_url.netloc.decode('ascii')}", url)
        return super().websocket_connect(url, subprotocols, **kwargs)
