"""Encryption and authentication for MADDENING's ZeroMQ transports.

The ZMQ transports carry two things worth stealing and one worth
forging.  ``viz.network.NetworkRelay`` publishes the **full simulation
state** on a PUB socket; ``viz.network.CommandPublisher`` publishes
**control commands** that are fed straight into
``GraphManager.step(external_inputs=...)``; and
``cloud.multigpu.coordinator.Coordinator`` accepts **worker
registrations** on a ROUTER socket, whose ``address`` field it later
hands to other workers as the peer to subscribe to.  Before this module
all three were unauthenticated and in cleartext.

Why not a signed envelope
-------------------------
A signature proves who *sent* a frame.  It does nothing about who
*reads* one, and the confirmed exposure on the two PUB sockets is
disclosure: anyone who can reach the port subscribes with
``SUBSCRIBE ""`` and receives the state.  Only encryption closes that,
so this module uses **ZMQ CURVE**, whose crypto is libsodium inside
libzmq -- already present in the ``pyzmq`` wheel, so no new dependency.

The rule (the same one the HTTP API uses)
-----------------------------------------
**A loopback address is unencrypted; anything else demands the token.**
This is deliberately identical to :mod:`maddening.api.auth`: binding or
connecting to ``127.0.0.1`` behaves exactly as it did before, so local
development needs no keys and no configuration, while any address that
a second machine could reach turns CURVE on.

``ipc://`` and ``inproc://`` are loopback by construction -- they never
leave the host -- and are treated as such.

One token, no key files
-----------------------
CURVE normally means a keypair per peer and a public-key exchange.
That is the ceremony nobody performs, so this module does not ask for
it.  Both CURVE keypairs are **derived deterministically from a shared
secret**.  A Curve25519 secret key is 32 bytes of any origin, so the
secret scalar is a domain-separated BLAKE2b of that secret and the
public key comes from :func:`zmq.curve_public` (libsodium's basepoint
multiplication).  Both sides compute both keypairs, so neither has to
learn anything from the other.

Which secret: ``MADDENING_TRANSPORT_TOKEN`` first
-------------------------------------------------
The secret is :data:`TRANSPORT_TOKEN_ENV`
(``MADDENING_TRANSPORT_TOKEN``) when it is set, and
:data:`~maddening.api.auth.TOKEN_ENV` (``MADDENING_API_TOKEN``)
otherwise.

**Prefer the transport variable, and understand the fallback before you
rely on it.**  ``MADDENING_API_TOKEN`` is the HTTP API's bearer
credential, and there is **no TLS**: it crosses the wire in cleartext
in an ``Authorization`` header on *every* API request.  Where that
token is also the CURVE seed, anyone who can see one such request can
derive both CURVE keypairs and read the "encrypted" state and command
streams -- measured over a real socket, not inferred.  The two
variables exist so that the streams do not inherit the HTTP
credential's exposure:

* set **both** variables, to *different* values, whenever the API port
  and a ZMQ port are published on the same untrusted network;
* setting only ``MADDENING_API_TOKEN`` keeps the single-variable setup
  that earlier releases documented, and it keeps the exposure above.
  It is convenient and it is the reason the fallback is spelled out
  here rather than hidden.

Both ends of a socket must hold the same value, whichever variable it
came from -- a relay reading ``MADDENING_TRANSPORT_TOKEN`` and a
receiver reading ``MADDENING_API_TOKEN`` derive different keys and will
not talk.  Or use an SSH tunnel and set nothing at all.

What this does and does not buy
-------------------------------
It buys confidentiality and integrity against anyone who does not hold
the token, verified over real sockets (see
``tests/security/test_zmq_transport_auth.py``).  It does **not** buy
mutual distrust *between* token holders: every holder can derive both
keypairs, so any authorised subscriber could also publish.  That is the
correct trust model for a stream whose readers are the operator's own
viz clients, and it is the same trust model as a shared bearer token.
It also provides no forward secrecy for a recorded session whose token
later leaks.  Where those matter, do not publish the port: bind
loopback and forward it over SSH, which is the documented default.

Notes
-----
A ZAP handler is **required**, not optional.  CURVE without one
authenticates the *server* to the client but accepts any client key:
measured over a real socket, an attacker holding only the server's
public key and a self-generated keypair received every published
frame.  :meth:`TransportAuth.start_authenticator` installs an
allowlist of exactly the token-derived client key, which blocks it.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

from maddening.api.auth import TOKEN_ENV, is_loopback
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

logger = logging.getLogger(__name__)

__all__ = [
    "TOKEN_ENV",
    "TRANSPORT_TOKEN_ENV",
    "TransportAuth",
    "TransportAuthError",
    "address_is_loopback",
    "address_requires_security",
    "resolve_security",
]

#: Environment variable holding the secret the CURVE keys are derived
#: from.  Read in preference to :data:`~maddening.api.auth.TOKEN_ENV`,
#: which is the cleartext HTTP bearer credential; see the module
#: docstring for why sharing that one exposes the streams.
TRANSPORT_TOKEN_ENV = "MADDENING_TRANSPORT_TOKEN"

#: BLAKE2b personalisation for the key derivation (max 16 bytes).
_CURVE_PERSON = b"maddening-curve"

#: Transport schemes that cannot leave the host.
_LOCAL_SCHEMES = frozenset({"inproc", "ipc"})

#: Wildcard bind hosts.  These are reachable from off-box, so they are
#: emphatically not loopback -- ``*`` is libzmq's spelling of ``0.0.0.0``.
_WILDCARD_HOSTS = frozenset({"*", "0.0.0.0", "::", "[::]"})


@stability(StabilityLevel.EVOLVING)
class TransportAuthError(RuntimeError):
    """A ZMQ socket needs the shared token and does not have it.

    Raised instead of falling back to cleartext.  A transport that
    silently downgrades when its credential is missing is the defect
    this module exists to remove, so the failure is loud.
    """


def _split_address(address: str) -> tuple[str, str]:
    """Return ``(scheme, host)`` for a ZMQ address.

    Parameters
    ----------
    address : str
        A ZMQ endpoint such as ``"tcp://127.0.0.1:5555"``,
        ``"tcp://*:5555"`` or ``"ipc:///tmp/sock"``.

    Returns
    -------
    tuple of (str, str)
        Lowercased scheme, and the host with any port and brackets
        removed.  Both are ``""`` when *address* cannot be parsed.
    """
    if not address or "://" not in address:
        return "", ""
    scheme, _, remainder = address.partition("://")
    scheme = scheme.strip().lower()
    if scheme in _LOCAL_SCHEMES:
        return scheme, ""
    # urlsplit needs a scheme it recognises to populate .hostname, and
    # it lowercases and de-brackets the host for us.
    parts = urlsplit(f"//{remainder}")
    host = parts.hostname or ""
    if not host:
        # urlsplit gives hostname=None for "*:5555" (an invalid netloc).
        host = remainder.rsplit(":", 1)[0] if ":" in remainder else remainder
    return scheme, host.strip().lower()


@stability(StabilityLevel.EVOLVING)
def address_is_loopback(address: str) -> bool:
    """Whether *address* names an endpoint only this host can reach.

    Parameters
    ----------
    address : str
        A ZMQ endpoint string.

    Returns
    -------
    bool
        ``True`` for loopback TCP hosts and for every ``ipc://`` or
        ``inproc://`` endpoint.  ``False`` for the wildcard binds
        ``tcp://*``, ``tcp://0.0.0.0`` and ``tcp://[::]``, for any
        routable host, and for an address that cannot be parsed -- the
        unknown case fails closed, as it does in
        :func:`maddening.api.auth.is_loopback`.

    Examples
    --------
    >>> address_is_loopback("tcp://127.0.0.1:5555")
    True
    >>> address_is_loopback("tcp://*:5555"), address_is_loopback("tcp://0.0.0.0:5555")
    (False, False)
    >>> address_is_loopback("inproc://x")
    True
    """
    scheme, host = _split_address(address)
    if not scheme:
        return False
    if scheme in _LOCAL_SCHEMES:
        return True
    if host in _WILDCARD_HOSTS:
        return False
    return is_loopback(host)


@stability(StabilityLevel.EVOLVING)
def address_requires_security(address: str) -> bool:
    """Whether *address* is one CURVE must protect.

    Parameters
    ----------
    address : str
        A ZMQ endpoint string.

    Returns
    -------
    bool
        ``not address_is_loopback(address)``.
    """
    return not address_is_loopback(address)


@stability(StabilityLevel.EVOLVING)
def resolve_security(address: str, secure: Optional[bool]) -> bool:
    """Decide whether a socket on *address* runs CURVE.

    Parameters
    ----------
    address : str
        The endpoint the socket will bind or connect to.
    secure : bool or None
        ``None`` decides from *address*.  ``True`` forces CURVE on even
        for a loopback endpoint, which is what the tests use and what a
        cautious operator may want on a shared host.  ``False`` is
        accepted **only** for a loopback endpoint; asking to disable
        encryption on a routable one raises, because that is the
        configuration this module exists to prevent.

    Returns
    -------
    bool
        Whether to configure CURVE.

    Raises
    ------
    TransportAuthError
        If ``secure=False`` is passed for a non-loopback address.
    """
    if secure is None:
        return address_requires_security(address)
    if not secure and address_requires_security(address):
        raise TransportAuthError(
            f"secure=False was passed for {address!r}, which is reachable "
            f"from other hosts. Encryption cannot be switched off on a "
            f"routable endpoint: the full simulation state and the command "
            f"channel would be readable by anyone who can reach the port. "
            f"Bind a loopback address and forward it over SSH, or set "
            f"{TRANSPORT_TOKEN_ENV} (or {TOKEN_ENV}) on both sides and "
            f"leave secure unset."
        )
    return bool(secure)


@stability(StabilityLevel.EVOLVING)
class TransportAuth:
    """CURVE keys for the ZMQ transports, derived from the shared token.

    Parameters
    ----------
    token : str, optional
        The shared secret.  ``None`` reads
        :data:`TRANSPORT_TOKEN_ENV` (``MADDENING_TRANSPORT_TOKEN``)
        from *environ*, then falls back to
        :data:`~maddening.api.auth.TOKEN_ENV`
        (``MADDENING_API_TOKEN``).  The fallback keeps a
        single-variable deployment working; it also means the CURVE
        seed is the credential the HTTP API sends in cleartext on every
        request.  See the module docstring.
    environ : mapping, optional
        Environment to read; defaults to :data:`os.environ`.

    Attributes
    ----------
    token : str
        The shared secret both sides must hold.
    token_env : str or None
        Which environment variable the secret came from, or ``None``
        when it was passed as *token*.  Callers that propagate the
        secret -- the cloud launch path does -- use this to pass on the
        variable the operator actually set.

    Raises
    ------
    TransportAuthError
        If no token is configured, or it is blank.  Unlike the HTTP
        API, a token **cannot** be generated here: a generated secret
        would be known to one end of the socket only, so the peer could
        never connect.  A blank value raises for the same reason it
        does in :class:`maddening.api.auth.APIAuth` -- it is what
        ``MADDENING_TRANSPORT_TOKEN=$UNSET_VARIABLE`` produces, and
        reading it as "encryption off" would reopen the hole.  A blank
        ``MADDENING_TRANSPORT_TOKEN`` does **not** fall through to
        ``MADDENING_API_TOKEN``: a variable that is set is the
        operator's answer, and silently using a different secret than
        the one they named is how two ends stop agreeing.

    Examples
    --------
    >>> auth = TransportAuth(token="shared-secret")
    >>> pub, sec = auth.server_keypair()
    >>> len(pub), len(sec)
    (40, 40)
    >>> TransportAuth(token="shared-secret").server_keypair() == (pub, sec)
    True
    """

    def __init__(
        self,
        token: Optional[str] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        env = os.environ if environ is None else environ
        source: Optional[str] = token
        origin: Optional[str] = None
        if source is None:
            source = env.get(TRANSPORT_TOKEN_ENV)
            origin = TRANSPORT_TOKEN_ENV
        if source is None:
            source = env.get(TOKEN_ENV)
            origin = TOKEN_ENV
        if source is None:
            raise TransportAuthError(
                f"This ZeroMQ endpoint is reachable from other hosts, so it "
                f"is encrypted with ZMQ CURVE, and that needs a shared "
                f"secret: set {TRANSPORT_TOKEN_ENV} to the same value on "
                f"both sides. {TOKEN_ENV} is accepted as a fallback so a "
                f"single-variable setup keeps working, but it is the HTTP "
                f"API's bearer credential and there is no TLS, so anyone "
                f"who sees one API request can then read these streams: "
                f"prefer a separate {TRANSPORT_TOKEN_ENV}. If you did not "
                f"mean to expose the port, bind a loopback address (the "
                f"default) and forward it with "
                f"'ssh -L 5555:127.0.0.1:5555 <host>', which needs no token."
            )
        if not source.strip():
            named = origin or "the token= argument"
            raise TransportAuthError(
                f"{named} is set but blank. A blank token is a "
                f"configuration error, not a request to disable encryption "
                f"-- it is what {named}=$UNSET_VARIABLE produces. Unset "
                f"it and bind loopback, or set it to a value."
            )
        self.token = source
        self.token_env = origin

    # -- key derivation -------------------------------------------------

    def _secret(self, role: str) -> bytes:
        """The z85-encoded CURVE secret key for *role*.

        A Curve25519 secret scalar is 32 bytes of arbitrary origin, so
        a domain-separated hash of the token is a valid one.  The role
        prefix keeps the server and client keys independent, so that
        observing one published key tells an attacker nothing about the
        other.
        """
        from zmq.utils import z85

        scalar = hashlib.blake2b(
            role.encode("ascii") + b"\x00" + self.token.encode("utf-8"),
            digest_size=32,
            person=_CURVE_PERSON,
        ).digest()
        return z85.encode(scalar)

    def _keypair(self, role: str) -> tuple[bytes, bytes]:
        import zmq

        secret = self._secret(role)
        return zmq.curve_public(secret), secret

    def server_keypair(self) -> tuple[bytes, bytes]:
        """The ``(public, secret)`` z85 keypair for the binding side."""
        return self._keypair("server")

    def client_keypair(self) -> tuple[bytes, bytes]:
        """The ``(public, secret)`` z85 keypair for the connecting side."""
        return self._keypair("client")

    # -- socket configuration -------------------------------------------

    def secure_server(self, socket: Any) -> None:
        """Configure *socket* as the CURVE server before it binds.

        Parameters
        ----------
        socket : zmq.Socket
            A socket that has not yet been bound.  CURVE options must
            be set before ``bind()``.

        Notes
        -----
        This alone does not authenticate clients; pair it with
        :meth:`start_authenticator` on the socket's context.  See the
        module docstring.
        """
        public, secret = self.server_keypair()
        socket.curve_secretkey = secret
        socket.curve_publickey = public
        socket.curve_server = True

    def secure_client(self, socket: Any) -> None:
        """Configure *socket* as a CURVE client before it connects.

        Parameters
        ----------
        socket : zmq.Socket
            A socket that has not yet been connected.
        """
        server_public, _ = self.server_keypair()
        public, secret = self.client_keypair()
        socket.curve_secretkey = secret
        socket.curve_publickey = public
        socket.curve_serverkey = server_public

    def start_authenticator(self, context: Any) -> Any:
        """Install a ZAP allowlist of the token-derived client key.

        Without this, libzmq completes a CURVE handshake with **any**
        client keypair as long as the client knows the server's public
        key, which makes the encryption worthless against an attacker
        who has learned it.  The allowlist admits exactly one public
        key: the one derived from the shared token.

        Parameters
        ----------
        context : zmq.Context
            The context owning the server socket.

        Returns
        -------
        zmq.auth.thread.ThreadAuthenticator
            Already started.  The caller must ``stop()`` it at
            shutdown.
        """
        from zmq.auth.thread import ThreadAuthenticator

        expected, _ = self.client_keypair()
        authenticator = ThreadAuthenticator(context)
        authenticator.start()
        authenticator.configure_curve_callback(
            domain="*", credentials_provider=_AllowOnly(expected),
        )
        return authenticator


class _AllowOnly:
    """ZAP credentials provider admitting a single CURVE public key."""

    def __init__(self, expected: bytes) -> None:
        self._expected = expected

    def callback(self, domain: str, key: Any) -> bool:
        """Whether *key* is the one public key this endpoint accepts.

        pyzmq hands the key as z85 ``bytes``, but normalise anyway: a
        ``str`` would compare unequal to our ``bytes`` and lock out the
        legitimate client rather than the attacker, which is a failure
        that looks like a network problem.
        """
        presented = key.encode("ascii") if isinstance(key, str) else key
        if presented == self._expected:
            return True
        logger.warning(
            "Rejected a ZeroMQ peer whose CURVE public key is not the one "
            "derived from this endpoint's transport secret. Both ends must "
            "share the same value, and must read it from the same variable: "
            "%s is used when it is set, otherwise %s.",
            TRANSPORT_TOKEN_ENV, TOKEN_ENV,
        )
        return False
