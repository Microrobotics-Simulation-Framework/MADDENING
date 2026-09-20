"""Bearer-token authentication for the MADDENING HTTP/WebSocket API.

The API provisions paid cloud GPUs (``POST /cloud/launch``) and rewrites
the simulation graph, and every shipped container path binds it to
``0.0.0.0``.  This module supplies the credential that closes that hole
without changing the local development flow.

The rule
--------
**A loopback bind is unauthenticated; anything else demands a token.**
Binding ``127.0.0.1`` (or ``::1``, or ``localhost``) behaves exactly as
it did before this module existed: no token, no header, nothing to
configure.  Binding any other address turns on a bearer token on every
route except the handful listed in :data:`UNAUTHENTICATED_PATHS`.

Two independent facts can each turn the check on, and only one of them
has to be right:

1. the bind address given to :class:`APIAuth` is not loopback, which is
   the configured, whole-server answer; and
2. the peer address of the individual request is a routable IP, which
   catches a caller who binds ``0.0.0.0`` through
   ``uvicorn.run(app, host="0.0.0.0")`` without telling the app.

(2) exists because the app cannot see the socket uvicorn binds.  Without
it, the security of the whole server would rest on a string the caller
remembered to pass.

Where the token comes from
--------------------------
``MADDENING_API_TOKEN`` if it is set, otherwise a fresh
:func:`secrets.token_urlsafe` value generated at start-up and logged
once (the Jupyter pattern).  ``MADDENING_API_TOKEN`` **set to an empty
or blank string is a configuration error and raises**: it is what
``MADDENING_API_TOKEN=$UNSET_VARIABLE`` produces, and reading it as
"authentication off" would resurrect exactly the failure this module
exists to remove.  Unset the variable if you want a generated token.

A generated token is only useful to somebody who can read the log.
When nothing can — a detached container, a job whose stdout goes
nowhere — set ``MADDENING_API_TOKEN`` yourself, or set
``MADDENING_API_TOKEN_FILE`` to a path the generated token is written
to with mode ``0600``.  The server does not fail if it cannot write
that file; it logs the failure and carries on, because a server that
refuses to start is worse than one whose token you have to recover from
the log.

How a client presents it
------------------------
* HTTP: ``Authorization: Bearer <token>``, and nothing else.  No
  authenticated HTTP route accepts a token in the query string, so a
  credential never reaches an access log through this API's own routes.
* WebSocket: ``Authorization: Bearer <token>`` for clients that can set
  headers, or -- for browsers, which cannot -- the subprotocol pair
  ``["maddening.bearer.<base64url(token)>", "maddening.v1"]``.  The
  server selects ``maddening.v1`` in the handshake response, as RFC 6455
  requires when the client offered any subprotocol at all.  See
  :func:`websocket_credentials`.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import ipaddress
import logging
import os
import secrets
import stat
from typing import Iterable, Mapping, Optional

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

logger = logging.getLogger(__name__)

#: Environment variable holding an operator-chosen API token.
TOKEN_ENV = "MADDENING_API_TOKEN"

#: Environment variable naming a file a *generated* token is written to.
TOKEN_FILE_ENV = "MADDENING_API_TOKEN_FILE"

#: Subprotocol the server selects in the WebSocket handshake response.
WS_SUBPROTOCOL = "maddening.v1"

#: Prefix of the pseudo-subprotocol that carries a browser's token.
WS_BEARER_PREFIX = "maddening.bearer."

#: Bytes of entropy in a generated token (32 -> 43 base64url characters).
TOKEN_BYTES = 32

#: Hostnames that name this machine and nothing else.
LOOPBACK_HOSTS = frozenset({
    "127.0.0.1", "localhost", "::1", "[::1]", "0:0:0:0:0:0:0:1",
})

#: Paths served without a token even when authentication is on.
#:
#: ``/healthz`` carries no information about the graph and exists so a
#: container probe does not need a credential.  The ``/viz/*`` pages are
#: static assets shipped in the wheel: they contain no secret (the token
#: is never templated into them -- see ``static/_auth.js``), so refusing
#: them would only stop the page that asks for the token from loading.
UNAUTHENTICATED_PATHS = frozenset({
    "/healthz",
    "/viz/app",
    "/viz/graph",
    "/viz/render",
    "/viz/auth.js",
})


@stability(StabilityLevel.EVOLVING)
def is_loopback(host: Optional[str]) -> bool:
    """Whether *host* names an address only this machine can reach.

    Parameters
    ----------
    host : str or None
        A hostname or IP literal, with or without brackets.  ``None`` and
        ``""`` are **not** loopback: an unknown address is treated as
        reachable so the unknown case fails closed.

    Returns
    -------
    bool
        ``True`` for ``127.0.0.0/8``, ``::1``, their IPv4-mapped form and
        the literal name ``localhost``; ``False`` for everything else,
        including the wildcard binds ``0.0.0.0`` and ``::``.

    Examples
    --------
    >>> is_loopback("127.0.0.1"), is_loopback("0.0.0.0")
    (True, False)
    >>> is_loopback("::ffff:127.0.0.1")
    True
    """
    normalised = (host or "").strip().lower()
    if not normalised:
        return False
    if normalised in LOOPBACK_HOSTS:
        return True
    candidate = normalised[1:-1] if normalised.startswith("[") and normalised.endswith("]") else normalised
    # Strip a zone index (fe80::1%eth0) before parsing.
    candidate = candidate.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return bool(address.is_loopback)


@stability(StabilityLevel.EVOLVING)
def is_routable_peer(host: Optional[str]) -> bool:
    """Whether *host* is an IP address that is not loopback.

    The peer backstop (rule 2 in the module docstring) uses this rather
    than ``not is_loopback(...)`` because ASGI transports that are not
    TCP put a non-address in ``scope["client"]`` -- Starlette's
    ``TestClient`` uses the string ``"testclient"``, and a Unix socket
    leaves the field empty.  Those are in-process or same-host by
    construction, so they are not treated as remote.

    Parameters
    ----------
    host : str or None
        The value of ``scope["client"][0]``, or ``None``.

    Returns
    -------
    bool
        ``True`` only when *host* parses as an IP address and that
        address is not loopback.
    """
    normalised = (host or "").strip()
    if not normalised:
        return False
    try:
        ipaddress.ip_address(normalised.split("%", 1)[0])
    except ValueError:
        return False
    return not is_loopback(normalised)


def encode_ws_bearer(token: str) -> str:
    """The subprotocol value a browser offers to present *token*.

    RFC 6455 subprotocol names are HTTP tokens: no commas, spaces or
    quotes.  An operator-chosen ``MADDENING_API_TOKEN`` may contain any
    of those, so the token travels base64url-encoded with the padding
    stripped, which is always a legal subprotocol name.

    Parameters
    ----------
    token : str
        The bearer token.

    Returns
    -------
    str
        ``"maddening.bearer." + base64url(token)``, unpadded.
    """
    packed = base64.urlsafe_b64encode(token.encode("utf-8")).decode("ascii")
    return WS_BEARER_PREFIX + packed.rstrip("=")


def decode_ws_bearer(value: str) -> str:
    """The token inside a ``maddening.bearer.*`` subprotocol value.

    Parameters
    ----------
    value : str
        One offered subprotocol name.

    Returns
    -------
    str
        The decoded token, or ``""`` when *value* is not a bearer
        subprotocol or does not decode.  A client must not be able to
        raise inside an authentication check by offering garbage.
    """
    if not value.startswith(WS_BEARER_PREFIX):
        return ""
    packed = value[len(WS_BEARER_PREFIX):]
    padding = "=" * (-len(packed) % 4)
    try:
        return base64.urlsafe_b64decode(packed + padding).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return ""


def websocket_credentials(token: str) -> list[str]:
    """The subprotocol list a browser passes to ``new WebSocket``.

    Parameters
    ----------
    token : str
        The bearer token.

    Returns
    -------
    list of str
        ``[bearer-carrier, "maddening.v1"]``.  The server echoes the
        second entry; a browser aborts the connection if the server
        selects none of the offered names.

    Examples
    --------
    >>> websocket_credentials("abc")
    ['maddening.bearer.YWJj', 'maddening.v1']
    """
    return [encode_ws_bearer(token), WS_SUBPROTOCOL]


def bearer_from_headers(headers: Mapping[str, str]) -> str:
    """The token in an ``Authorization: Bearer`` header, or ``""``.

    Parameters
    ----------
    headers : mapping
        Case-insensitive header mapping (Starlette's ``request.headers``).

    Returns
    -------
    str
        The token, or ``""`` when the header is absent or is not a
        ``Bearer`` credential.
    """
    raw = headers.get("authorization") or ""
    scheme, _, credential = raw.partition(" ")
    if scheme.strip().lower() != "bearer":
        return ""
    return credential.strip()


def bearer_from_subprotocols(offered: Iterable[str]) -> str:
    """The token carried by an offered WebSocket subprotocol, or ``""``.

    Parameters
    ----------
    offered : iterable of str
        The subprotocol names the client offered.

    Returns
    -------
    str
        The first decodable ``maddening.bearer.*`` value, else ``""``.
    """
    for name in offered:
        token = decode_ws_bearer(name.strip())
        if token:
            return token
    return ""


@stability(StabilityLevel.EVOLVING)
class APIAuth:
    """The token the API demands, and the rule for when it demands it.

    Parameters
    ----------
    bind_host : str, optional
        The address the server will be bound to.  ``None`` reads
        ``MADDENING_HOST`` from *environ* and falls back to
        ``"127.0.0.1"``, which is what an embedder who never set it gets.
        A non-loopback value turns authentication on for every request.
    token : str, optional
        An explicit token, overriding the environment.  Must be a
        non-blank string.
    environ : mapping, optional
        Environment to read; defaults to :data:`os.environ`.

    Attributes
    ----------
    token : str
        The credential clients must present.  Always populated -- even
        on a loopback bind, so that the peer backstop has something a
        remote caller could in principle present.
    generated : bool
        ``True`` when the token was generated rather than configured.
    bind_is_loopback : bool
        Whether *bind_host* named a loopback address.
    enforced : bool
        Whether every request needs a token.  Equivalent to
        ``not bind_is_loopback``.

    Raises
    ------
    ValueError
        If ``MADDENING_API_TOKEN`` (or *token*) is set but blank.  See
        the module docstring: an empty token is a configuration mistake,
        and silently reading it as "no authentication" is the defect
        this class exists to fix.

    Examples
    --------
    >>> auth = APIAuth(bind_host="127.0.0.1", token="s3cret", environ={})
    >>> auth.enforced
    False
    >>> auth.required_for_peer("10.0.0.4")
    True
    >>> auth.verify("s3cret"), auth.verify("wrong")
    (True, False)
    """

    def __init__(
        self,
        bind_host: Optional[str] = None,
        token: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        env = os.environ if environ is None else environ
        if bind_host is None:
            bind_host = env.get("MADDENING_HOST", "127.0.0.1")
        self.bind_host = bind_host
        self.bind_is_loopback = is_loopback(bind_host)

        source = token if token is not None else env.get(TOKEN_ENV)
        if source is not None and not source.strip():
            origin = "the token= argument" if token is not None else TOKEN_ENV
            raise ValueError(
                f"{origin} is set but blank. A blank token is a configuration "
                f"error, not a request to disable authentication -- it is what "
                f"{TOKEN_ENV}=$UNSET_VARIABLE produces. Unset {TOKEN_ENV} to "
                f"have the server generate a token and log it once, or set it "
                f"to a value."
            )
        self.generated = source is None
        self.token = secrets.token_urlsafe(TOKEN_BYTES) if source is None else source
        self._announced = False

    @property
    def enforced(self) -> bool:
        """Whether the bind address alone makes a token mandatory."""
        return not self.bind_is_loopback

    def required_for_peer(self, peer_host: Optional[str]) -> bool:
        """Whether a request from *peer_host* must present a token.

        Parameters
        ----------
        peer_host : str or None
            ``scope["client"][0]``: the address the request came from.

        Returns
        -------
        bool
            ``True`` when the bind is non-loopback (rule 1) **or** the
            peer is a routable IP (rule 2, the backstop).
        """
        return self.enforced or is_routable_peer(peer_host)

    def verify(self, presented: Optional[str]) -> bool:
        """Whether *presented* is the expected token.

        Compared with :func:`hmac.compare_digest` on the UTF-8 bytes.
        ``compare_digest`` raises ``TypeError`` on a ``str`` holding a
        non-ASCII character, and a client must not be able to raise
        inside an authentication check by sending one, so both sides are
        encoded first.

        Parameters
        ----------
        presented : str or None
            Whatever the client sent, which may be anything at all.

        Returns
        -------
        bool
            ``True`` only on an exact match.
        """
        if not presented:
            return False
        return hmac.compare_digest(
            presented.encode("utf-8", "surrogatepass"),
            self.token.encode("utf-8", "surrogatepass"),
        )

    def announce(self, port: int = 8000) -> bool:
        """Log the generated token once, and persist it if asked.

        Called by the server at start-up.  Does nothing on a loopback
        bind (there is no credential to hand out and no reason to write
        one into a developer's log) and nothing when the operator chose
        the token themselves (they already have it; echoing a configured
        secret into the log only widens where it lives).

        Parameters
        ----------
        port : int, optional
            Bind port, for the log line only.

        Returns
        -------
        bool
            ``True`` when a token was logged by this call.
        """
        if not self.enforced:
            # uvicorn.run(app, host="0.0.0.0") without bind_host or
            # MADDENING_HOST is exactly the case the peer backstop exists
            # for: remote callers are correctly refused, with a token
            # that was never logged and never written to
            # MADDENING_API_TOKEN_FILE, because both live in the branch
            # below.  Say so once -- without the value, which is not
            # needed here and would put a live credential in a
            # developer's log on every loopback start-up.
            if self.generated and not self._announced:
                self._announced = True
                logger.info(
                    "A %s was generated at start-up. It is not needed for a "
                    "loopback bind, and it is not shown here -- but if this "
                    "server is in fact reachable from off-host, the peer "
                    "backstop will demand it and nothing will have printed "
                    "it. Pass bind_host (or set MADDENING_HOST) so it is "
                    "logged, or set %s yourself.", TOKEN_ENV, TOKEN_ENV,
                )
            return False
        if self._announced:
            return False
        self._announced = True
        if not self.generated:
            logger.warning(
                "MADDENING API is bound to %s:%s and requires "
                "'Authorization: Bearer <token>' on every route. The token "
                "came from %s; it is not repeated here.",
                self.bind_host, port, TOKEN_ENV,
            )
            return False
        self._write_token_file()
        logger.warning(
            "\n"
            "============================================================\n"
            "  MADDENING API is listening on %s:%s -- NOT loopback, so\n"
            "  every route requires a bearer token. This one was\n"
            "  generated at start-up and is shown ONCE:\n"
            "\n"
            "      %s\n"
            "\n"
            "  curl -H 'Authorization: Bearer %s' http://%s:%s/graph\n"
            "  Browser UI: http://%s:%s/viz/app#token=%s\n"
            "    (a URL fragment; it is never sent to the server, so it\n"
            "     reaches no access log. The page scrubs it immediately.)\n"
            "\n"
            "  Set %s to choose it yourself, which you must do when\n"
            "  nothing reads this log. There is still NO TLS: the token\n"
            "  crosses the network in cleartext, so keep the port off\n"
            "  the public internet and prefer an SSH tunnel.\n"
            "============================================================",
            self.bind_host, port,
            self.token, self.token, self.bind_host, port,
            self.bind_host, port, self.token,
            TOKEN_ENV,
        )
        return True

    def _write_token_file(self) -> None:
        """Write a generated token to ``$MADDENING_API_TOKEN_FILE``."""
        path = os.environ.get(TOKEN_FILE_ENV, "").strip()
        if not path:
            return
        try:
            handle = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                stat.S_IRUSR | stat.S_IWUSR,
            )
            try:
                os.write(handle, (self.token + "\n").encode("utf-8"))
            finally:
                os.close(handle)
        except OSError:
            logger.warning(
                "Could not write the generated API token to %s=%r; it is in "
                "the log line above and nowhere else.", TOKEN_FILE_ENV, path,
                exc_info=True,
            )
        else:
            logger.info("Generated API token written to %s (mode 0600)", path)
