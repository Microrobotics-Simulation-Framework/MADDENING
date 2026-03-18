"""Health probes with typed error attribution.

Each probe function knows which stage it belongs to, so callers
(``CloudSession.wait_ready()``) can map failures to the correct
``CloudReadyResult.error_stage``.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

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


def probe_http(url: str, timeout: float = 10.0) -> None:
    """Probe HTTP endpoint at *url*.

    Raises ``HealthProbeError("container", ...)`` on failure.
    """
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except (urllib.error.URLError, OSError) as exc:
        raise HealthProbeError("container", f"HTTP probe to {url} failed: {exc}")


def probe_zmq(endpoint: str, timeout: float = 5.0) -> None:
    """Probe a ZMQ PUB endpoint by attempting a brief SUB connect.

    Raises ``HealthProbeError("data_channel", ...)`` on failure.
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
