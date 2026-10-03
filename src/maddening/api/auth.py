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
2. the individual request is not a direct connection from a loopback
   address -- its peer is not a loopback IP literal, or it carries a
   forwarding header (``X-Forwarded-For``, ``Forwarded``) -- which
   catches a caller who binds ``0.0.0.0`` through
   ``uvicorn.run(app, host="0.0.0.0")`` without telling the app.

(2) exists because the app cannot see the socket uvicorn binds.  Without
it, the security of the whole server would rest on a string the caller
remembered to pass.

The peer is ``scope["client"]`` as the ASGI server reports it, and that
is not always the socket's: under uvicorn's ``proxy_headers`` (on by
default, trusting ``127.0.0.1``) it is whatever a trusted proxy's
``X-Forwarded-For`` says, so any request arriving over loopback names
its own peer.  A peer that is not an IP literal -- Starlette's
``TestClient`` reports ``"testclient"``, a Unix socket none, and a
forged header anything at all -- is therefore unknown, and the token is
demanded of it; so is any request that carries a forwarding header.
**Never configure loopback as a trusted proxy for a loopback-bound
server** (uvicorn's ``forwarded_allow_ips``): this module no longer
depends on it, but nothing gains from it either.  The library's own
launch paths pass ``proxy_headers=False``.  An in-process client
(``TestClient``) presents :attr:`APIAuth.token`, or is constructed as a
loopback client: ``TestClient(app, base_url="http://127.0.0.1",
client=("127.0.0.1", 50000))``.

Where the token comes from
--------------------------
``MADDENING_API_TOKEN`` if it is set, otherwise a fresh
:func:`secrets.token_urlsafe` value generated at start-up and logged
once (the Jupyter pattern).  ``MADDENING_API_TOKEN`` **set to an empty
or blank string is a configuration error and raises**: it is what
``MADDENING_API_TOKEN=$UNSET_VARIABLE`` produces, and reading it as
"authentication off" would resurrect exactly the failure this module
exists to remove.  Unset the variable if you want a generated token.
A token with whitespace before or after it raises too: a client's
``Authorization`` header is read with that whitespace stripped, so such
a token could never be presented.

A generated token is only useful to somebody who can read the log.
When nothing can — a detached container, a job whose stdout goes
nowhere — set ``MADDENING_API_TOKEN`` yourself, or set
``MADDENING_API_TOKEN_FILE`` to a path the generated token is written
to with mode ``0600`` (a new file moved over the path: one already there
keeps neither its mode nor its readers).  The server does not fail if it cannot write
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
import contextlib
import hmac
import ipaddress
import logging
import os
import secrets
import tempfile
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

    A classification of an address, and no longer the peer backstop's
    rule (rule 2 in the module docstring), which demands the token of
    every peer that is not a loopback IP literal.  A value that is not
    an IP address -- Starlette's ``TestClient`` reports ``"testclient"``,
    a Unix socket leaves the field empty -- is not routable, but it is
    not known to be local either: the peer is ``scope["client"]`` as the
    ASGI server reports it, which under uvicorn's ``proxy_headers`` is a
    trusted proxy's ``X-Forwarded-For`` value, so any string at all can
    be put there by a request.

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


def _is_loopback_peer(host: Optional[str]) -> bool:
    """Whether *host* -- ``scope["client"][0]`` -- is a loopback IP
    literal (``127.0.0.0/8``, ``::1``, their IPv4-mapped form).  A name
    (``"localhost"``, ``"testclient"``), an empty peer and anything else
    that does not parse as an address are not: the peer backstop treats
    them as unknown and demands the token."""
    normalised = (host or "").strip()
    if not normalised:
        return False
    try:
        ipaddress.ip_address(normalised.split("%", 1)[0])
    except ValueError:
        return False
    return is_loopback(normalised)


#: Request headers that say a proxy forwarded the request, and that an ASGI
#: server's proxy support (uvicorn's ``proxy_headers``, hypercorn's
#: ``ProxyFixMiddleware``) reads ``scope["client"]`` from.
_FORWARDING_HEADERS = ("x-forwarded-for", "forwarded")


def _carries_forwarding_header(headers: Optional[Mapping[str, str]]) -> bool:
    """Whether *headers* (a case-insensitive mapping, or one keyed by
    lowercase names) carry ``X-Forwarded-For`` or ``Forwarded``."""
    if not headers:
        return False
    return any(headers.get(name) is not None for name in _FORWARDING_HEADERS)


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
        non-blank string without whitespace before or after it.
    environ : mapping, optional
        Environment to read (``MADDENING_HOST``, ``MADDENING_API_TOKEN``,
        ``MADDENING_API_TOKEN_FILE``); defaults to :data:`os.environ`.

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
        this class exists to fix.  Also if it has whitespace before or
        after it, which no ``Authorization`` header can present.

    Examples
    --------
    >>> auth = APIAuth(bind_host="127.0.0.1", token="s3cret", environ={})
    >>> auth.enforced
    False
    >>> auth.required_for_peer("10.0.0.4")
    True
    >>> auth.required_for_peer("127.0.0.1"), auth.required_for_peer("testclient")
    (False, True)
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
        origin = "the token= argument" if token is not None else TOKEN_ENV
        if source is not None and not source.strip():
            raise ValueError(
                f"{origin} is set but blank. A blank token is a configuration "
                f"error, not a request to disable authentication -- it is what "
                f"{TOKEN_ENV}=$UNSET_VARIABLE produces. Unset {TOKEN_ENV} to "
                f"have the server generate a token and log it once, or set it "
                f"to a value."
            )
        if source is not None and source != source.strip():
            # bearer_from_headers strips the credential a client presents,
            # so a token with whitespace around it could never be matched:
            # every client was refused, with nothing said at start-up.
            raise ValueError(
                f"{origin} has whitespace before or after it ({len(source)} "
                f"characters, {len(source.strip())} without it). An "
                "'Authorization: Bearer' header cannot carry that whitespace -- "
                "the server strips it from what a client presents -- so no "
                "client could ever present this token. Set it without the "
                "surrounding whitespace (a trailing newline from a file, say)."
            )
        self.generated = source is None
        self._environ = env
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
            ``scope["client"][0]`` as the ASGI server reports it.  That is
            the socket's peer address, except under a proxy-header
            middleware (uvicorn's ``proxy_headers``, on by default and
            trusting ``127.0.0.1``), where it is a trusted proxy's
            ``X-Forwarded-For`` value -- which any request arriving over
            loopback can set.

        Returns
        -------
        bool
            ``True`` when the bind is non-loopback (rule 1) **or** the
            peer is not a loopback IP literal (rule 2, the backstop).  A
            peer that is not an IP address at all (Starlette's
            ``TestClient`` reports ``"testclient"``, a Unix socket none, a
            forged ``X-Forwarded-For`` anything) is unknown, so the
            backstop fails closed.  Until 0.4.0's release it treated such
            a peer as local, and ``X-Forwarded-For: x`` on a request to a
            loopback-bound uvicorn made every route answer without a token.
        """
        return self.enforced or not _is_loopback_peer(peer_host)

    def _required_for_request(self, peer_host: Optional[str],
                              headers: Optional[Mapping[str, str]]) -> bool:
        """:meth:`required_for_peer`, and also for a request that carries a
        forwarding header (``X-Forwarded-For``, ``Forwarded``): a proxy
        stands between the client and this server, or the request claims
        one does, and either way the peer it reports is not a direct
        loopback connection.  The one rule the HTTP middleware, the
        WebSocket middleware and every WebSocket handler ask."""
        return self.required_for_peer(peer_host) or _carries_forwarding_header(headers)

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
        """Write a generated token to ``$MADDENING_API_TOKEN_FILE``, mode 0600.

        The token goes into a new file -- created ``0600`` by
        :func:`tempfile.mkstemp` in the destination's own directory -- which
        is then moved over the destination (:func:`os.replace`).  Whatever
        was at the path, a file of any mode included, is replaced and never
        written into.  Opening the path itself with ``O_CREAT`` and mode
        ``0600`` applied the mode only to a file it created: one already
        there (a placeholder on a volume, the last run's token) kept its
        mode, ``0644`` say, and the new token landed in a file other users
        could read.  ``chmod`` after opening would not close that either: a
        reader who had the file open before the ``chmod`` keeps reading it.
        A new inode no earlier reader can hold does.  A symbolic link at the
        path is followed, so the token lands where the link points, in a
        file that is now its own.  When the move cannot be made (a directory
        the server cannot write, a file another user owns in a sticky
        directory) nothing is written and the failure is logged.
        """
        # The environment the instance was built with (``environ=``), as
        # every other setting is read: this used to read os.environ, so
        # ``environ={"MADDENING_API_TOKEN_FILE": ...}`` wrote nothing, and
        # an explicit ``environ={}`` still wrote to the process's path.
        path = (self._environ.get(TOKEN_FILE_ENV) or "").strip()
        if not path:
            return
        target = os.path.realpath(path)
        partial = None
        try:
            handle, partial = tempfile.mkstemp(
                prefix=".maddening-token.", dir=os.path.dirname(target) or ".")
            with os.fdopen(handle, "w", encoding="utf-8") as out:
                out.write(self.token + "\n")
            os.replace(partial, target)
            partial = None
        except OSError:
            logger.warning(
                "Could not write the generated API token to %s=%r; it is in "
                "the log line above and nowhere else.", TOKEN_FILE_ENV, path,
                exc_info=True,
            )
        else:
            logger.info("Generated API token written to %s (mode 0600)", path)
        finally:
            if partial is not None:
                with contextlib.suppress(OSError):
                    os.unlink(partial)
