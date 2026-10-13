"""Health probes with typed error attribution.

Each probe function knows which stage it belongs to, so callers
(``CloudSession.wait_ready()``) can map failures to the correct
``CloudReadyResult.error_stage``.

Every probe here talks to a **secured** endpoint.  The container's API
binds ``0.0.0.0``, which turns its bearer token on, and a relay whose
port is published is on a non-loopback address, which turns ZMQ CURVE
on.  A probe that carries no credential therefore measures the
credential, not the health: it fails identically against a container
that is up and one that never started.  :func:`probe_http` and
:func:`probe_zmq` both take the shared token for that reason, and
:func:`probe_http` reports a 401 as a credential failure rather than as
"not up yet".
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

from maddening.api.auth import TOKEN_ENV

logger = logging.getLogger(__name__)


class HealthProbeError(Exception):
    """A health probe failed, with stage and detail attribution."""

    def __init__(self, stage: str, detail: str = ""):
        super().__init__(f"{stage}: {detail}")
        self.stage = stage
        self.detail = detail


def probe_ssh(ip: str, timeout: float = 10.0) -> None:
    """Probe SSH connectivity to *ip*.

    Raises ``HealthProbeError("vm", ...)`` on failure.
    """
    import socket

    try:
        sock = socket.create_connection((ip, 22), timeout=timeout)
        sock.close()
    except (OSError, socket.timeout) as exc:
        raise HealthProbeError("vm", f"SSH probe to {ip}:22 failed: {exc}")


def probe_http(
    url: str,
    timeout: float = 10.0,
    token: Optional[str] = None,
) -> None:
    """Probe HTTP endpoint at *url*, optionally with a bearer token.

    Parameters
    ----------
    url : str
        The URL to fetch.
    timeout : float, optional
        Seconds to wait for the response.
    token : str, optional
        Bearer credential to present.  The cloud container binds
        ``0.0.0.0`` -- it must, because a container bound to loopback is
        unreachable even with a published port -- and that bind is
        exactly what turns the API's bearer token on.  Every route but
        :data:`maddening.api.auth.UNAUTHENTICATED_PATHS` therefore
        answers 401 to a probe with no credential, which is why a probe
        of an authenticated route has to carry one.

    Raises
    ------
    HealthProbeError
        With ``stage="container"``.  A 401 is reported as a *credential*
        failure rather than as "the container is not up yet": the two
        are indistinguishable from the status code alone, and retrying a
        401 for the whole timeout window is what made a launch against
        an authenticated container fail after 120 s of silence.
    """
    import urllib.request
    import urllib.error

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        req = urllib.request.Request(url, method="GET", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise HealthProbeError(
                "container",
                f"HTTP probe to {url} was refused with {exc.code}: the "
                f"container's API requires a bearer token and this probe "
                f"{'presented one that was rejected' if token else 'sent none'}"
                f". Set {TOKEN_ENV} in the launching process so the same "
                f"value reaches the container and the probe.",
            )
        raise HealthProbeError("container", f"HTTP probe to {url} failed: {exc}")
    except (urllib.error.URLError, OSError) as exc:
        raise HealthProbeError("container", f"HTTP probe to {url} failed: {exc}")


def probe_zmq(
    endpoint: str,
    timeout: float = 5.0,
    token: Optional[str] = None,
) -> None:
    """Probe a ZMQ PUB endpoint by attempting a brief SUB connect.

    Parameters
    ----------
    endpoint : str
        The ``tcp://host:port`` endpoint of the relay's PUB socket.
    timeout : float, optional
        Seconds to wait for one frame.
    token : str, optional
        The shared transport secret.  A relay whose port is published is
        on a non-loopback address, so it runs ZMQ CURVE (see
        :mod:`maddening.transport_auth`) and a *plain* SUB socket can
        never complete its handshake -- it receives nothing and the
        probe times out however healthy the relay is.  Passing the token
        configures the SUB socket as the matching CURVE client.

    Raises
    ------
    HealthProbeError
        With ``stage="data_channel"``.
    """
    try:
        import zmq
    except ImportError:
        raise HealthProbeError(
            "data_channel",
            "pyzmq not installed — cannot probe ZMQ endpoint",
        )

    ctx = zmq.Context()
    try:
        sock = ctx.socket(zmq.SUB)
        if token:
            from maddening.transport_auth import (
                TransportAuth,
                address_requires_security,
            )
            if address_requires_security(endpoint):
                TransportAuth(token=token).secure_client(sock)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVTIMEO, int(timeout * 1000))
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(endpoint)
        try:
            sock.recv()
        except zmq.Again:
            raise HealthProbeError(
                "data_channel",
                f"ZMQ probe to {endpoint} timed out (no data in {timeout}s)",
            )
    except HealthProbeError:
        raise
    except Exception as exc:
        raise HealthProbeError(
            "data_channel",
            f"ZMQ probe to {endpoint} failed: {exc}",
        )
    finally:
        sock.close()
        ctx.term()


def wait_for(
    probe_fn: Callable[[], None],
    timeout: float = 60.0,
    interval: float = 5.0,
) -> None:
    """Retry *probe_fn* until it succeeds or *timeout* expires.

    On timeout, lets the ``HealthProbeError`` from the last attempt
    propagate with its stage attribution intact.
    """
    deadline = time.monotonic() + timeout
    last_error: HealthProbeError | None = None

    while time.monotonic() < deadline:
        try:
            probe_fn()
            return  # success
        except HealthProbeError as exc:
            last_error = exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(interval, remaining))

    if last_error is not None:
        raise last_error
    raise HealthProbeError("unknown", "wait_for timed out with no probe error")
