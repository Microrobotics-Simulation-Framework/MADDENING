"""``CloudSession.wait_ready()`` must succeed against a real container.

Every other ``wait_ready`` test in this package uses ``MockCloudSession``,
which overrides ``_launch_worker`` -- so the probes themselves were never
executed and the release shipped a launch path that could not work.  The
container's API binds ``0.0.0.0`` (it must: a container bound to loopback
is unreachable even with a published port), that bind turns the bearer
token on for every route but ``/healthz`` and ``/viz/*``, and the probes
sent no ``Authorization`` header.  Both of them got 401, ``wait_for``
retried for the full 120 s, and the session ended in ``CloudStage.ERROR``.

So these tests run **a real uvicorn socket serving a real
``SimulationServer`` told ``bind_host="0.0.0.0"``**, exactly as
``cloud/entrypoint.py`` builds it, and drive the real
``CloudSession._launch_worker`` against it with only SkyPilot replaced.
A probe that stops carrying a credential, or a stage pointed back at an
authenticated route, fails here.

``test_the_probes_are_not_vacuous_when_the_token_is_wrong`` is the
control: the happy path above proves nothing unless a mismatched token
is visibly refused by the same machinery.
"""

from __future__ import annotations

import os
import socket
import threading
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from maddening.cloud._health import HealthProbeError, probe_http, wait_for
from maddening.cloud.session import CloudConfig, CloudSession, CloudStage

TOKEN = "container-token-not-a-real-credential"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@pytest.fixture(scope="module")
def container():
    """A real HTTP server built the way the cloud entrypoint builds it.

    Yields ``(base_url, port)``.  ``bind_host="0.0.0.0"`` is what
    ``entrypoint.main`` passes; the socket itself stays on loopback so
    the test opens nothing on the network.
    """
    uvicorn = pytest.importorskip("uvicorn", reason="the probes need a real HTTP server")

    from maddening.api.server import SimulationServer
    from maddening.core.graph_manager import GraphManager
    from maddening.nodes.spring import SpringDamperNode

    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="spring", timestep=0.01))
    server = SimulationServer(
        node_registry={"SpringDamperNode": SpringDamperNode},
        graph_manager=gm,
        bind_host="0.0.0.0",
        api_token=TOKEN,
    )
    port = _free_port()
    config = uvicorn.Config(
        server.create_app(), host="127.0.0.1", port=port, log_level="error",
        # No WebSocket implementation: importing one drags in
        # ``websockets.legacy``, whose import-time DeprecationWarning is an
        # error under this project's ``filterwarnings = ["error"]`` and kills
        # the server thread before it binds.  Nothing here speaks WebSocket.
        ws="none",
    )
    uv = uvicorn.Server(config)
    thread = threading.Thread(target=uv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not uv.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert uv.started, "the test server did not start"
    try:
        yield f"http://127.0.0.1:{port}", port
    finally:
        uv.should_exit = True
        thread.join(timeout=10)


# ----------------------------------------------------------------------
# The probe itself
# ----------------------------------------------------------------------

def test_the_liveness_probe_needs_no_credential(container):
    """``/healthz`` is exempt on purpose and is what stage 2 uses."""
    base, _ = container
    probe_http(f"{base}/healthz", timeout=5)


def test_an_authenticated_route_refuses_a_probe_with_no_credential(container):
    """The defect, pinned: this is what stage 2 and 3 used to do."""
    base, _ = container
    with pytest.raises(HealthProbeError) as excinfo:
        probe_http(f"{base}/graph/state", timeout=5)
    assert excinfo.value.stage == "container"
    # The old message said only "failed: HTTP Error 401", which reads as
    # "not up yet" and is why this was retried for two minutes.
    assert "401" in excinfo.value.detail
    assert "sent none" in excinfo.value.detail


def test_an_authenticated_route_answers_a_probe_that_carries_the_token(container):
    base, _ = container
    probe_http(f"{base}/graph/state", timeout=5, token=TOKEN)


def test_a_probe_with_the_wrong_token_is_refused_and_says_so(container):
    base, _ = container
    with pytest.raises(HealthProbeError) as excinfo:
        probe_http(f"{base}/graph/state", timeout=5, token="wrong")
    assert "rejected" in excinfo.value.detail


def test_wait_for_does_not_burn_its_window_on_a_credential_failure(container):
    """A 401 is still retried, but it names the credential, not the container.

    ``wait_for`` retries every ``HealthProbeError``; what changed is that
    the error it finally raises says which of the two possible causes it
    was, so an operator reading the log is not told the container never
    started.
    """
    base, _ = container
    with pytest.raises(HealthProbeError) as excinfo:
        wait_for(lambda: probe_http(f"{base}/graph", timeout=2),
                 timeout=2, interval=1)
    assert "bearer token" in excinfo.value.detail


# ----------------------------------------------------------------------
# The whole launch path, with only SkyPilot replaced
# ----------------------------------------------------------------------

@pytest.fixture
def fake_skypilot(container, monkeypatch):
    """Replace provisioning; keep every health probe real.

    Records the environment ``launch_vm`` was asked to give the
    container, which is how the container and the probes come to hold
    the same token.
    """
    from maddening.cloud import _skypilot

    _, port = container
    recorded: dict = {}

    def _launch_vm(config, envs=None):
        recorded["envs"] = envs
        recorded["config"] = config
        return "127.0.0.1", "maddening-test-cluster"

    monkeypatch.setattr(_skypilot, "launch_vm", _launch_vm)
    monkeypatch.setattr(
        _skypilot, "monitor_preemption",
        lambda job_id, callback, poll_interval=5.0: None,
    )
    return recorded, port


def test_wait_ready_succeeds_against_an_authenticated_container(fake_skypilot):
    """The release blocker.  Nothing here is mocked but the VM.

    Stage 2 probes ``/healthz`` over a real socket and stage 3 probes
    ``/graph/state`` with the session's bearer token, against a server
    that really demands one.
    """
    recorded, port = fake_skypilot
    session = CloudSession(api_token=TOKEN)
    session.launch(CloudConfig(spot=False, api_port=port, ports=(port,)))
    result = session.wait_ready(timeout=90)

    assert result.error_stage is None, (
        f"{result.error_stage}: {result.error_detail}"
    )
    assert result.fully_ready is True
    assert session.stage is CloudStage.FULLY_READY


def test_the_launch_hands_the_container_the_token_the_probes_present(
    fake_skypilot,
):
    """Both halves of the fix, in one assertion each.

    The container is told the token, and the token it is told is the one
    this session will present.  Either half alone leaves the probes
    unable to authenticate.
    """
    recorded, port = fake_skypilot
    session = CloudSession(api_token=TOKEN, transport_token="a-second-secret")
    session.launch(CloudConfig(spot=False, api_port=port, ports=(port,)))
    session.wait_ready(timeout=90)

    assert recorded["envs"] == {
        "MADDENING_API_TOKEN": TOKEN,
        "MADDENING_TRANSPORT_TOKEN": "a-second-secret",
    }
    assert session.api_token == TOKEN


def test_the_probes_are_not_vacuous_when_the_token_is_wrong(fake_skypilot):
    """The control run.

    If the happy path passed with a token the container does not accept,
    it would be measuring nothing.  A session holding the wrong token
    must fail at the stage that needs the graph -- and must reach that
    stage, because stage 2 is deliberately credential-free.
    """
    recorded, port = fake_skypilot
    session = CloudSession(api_token="not-the-containers-token")
    # The production window is 60s of retries; the refusal is immediate
    # and deterministic, so there is nothing to wait for here.
    session._stage_timeouts["simulation"] = 3.0
    session.launch(CloudConfig(spot=False, api_port=port, ports=(port,)))
    result = session.wait_ready(timeout=90)

    assert result.fully_ready is False
    assert result.error_stage == "container"
    assert result.container_ready is True, (
        "stage 2 must pass without a credential; if it did not, the "
        "failure above is not the authentication failure this pins"
    )
    assert session.stage is CloudStage.ERROR


def test_a_session_generates_a_token_when_the_environment_has_none(monkeypatch):
    """A container that generates its own token is unprobeable.

    Before this, an operator who set nothing got a container whose token
    existed only in its log.  The session generates it instead, so the
    same value is in the container and in the probe -- and the caller can
    read it.
    """
    monkeypatch.delenv("MADDENING_API_TOKEN", raising=False)
    monkeypatch.delenv("MADDENING_TRANSPORT_TOKEN", raising=False)
    session = CloudSession()

    assert len(session.api_token) >= 40
    assert session.transport_token is None
    assert session.container_env() == {"MADDENING_API_TOKEN": session.api_token}


def test_a_session_propagates_whichever_token_variable_is_set(monkeypatch):
    """Item 2's requirement: the cloud path carries the transport secret."""
    monkeypatch.setenv("MADDENING_API_TOKEN", "api-secret")
    monkeypatch.setenv("MADDENING_TRANSPORT_TOKEN", "transport-secret")
    session = CloudSession()

    assert session.container_env() == {
        "MADDENING_API_TOKEN": "api-secret",
        "MADDENING_TRANSPORT_TOKEN": "transport-secret",
    }
