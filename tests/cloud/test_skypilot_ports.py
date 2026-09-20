"""``launch_vm`` publishes the ports the job config asks for, and no others.

``_skypilot.launch_vm`` used to run

    docker run --gpus all -p 8000:8000 -p 8080:8080 -p 5555:5555 -p 5556:5556

with those four ports hard-coded, never consulting ``JobConfig.ports``.
That put the HTTP API, the state stream and the command channel on the
VM's public interface on every launch, and it is why emptying
``JobConfig.ports``' default did not change what a CloudSession exposed.

These tests pin the container publish list and the provider firewall
list to ``config.ports``, so a regression to a hard-coded string fails
here rather than in a cloud bill.
"""

from __future__ import annotations

import re
import types
from dataclasses import dataclass, field

import pytest

from maddening.cloud import _skypilot


@dataclass
class _Config:
    """The attributes ``launch_vm`` reads off a job config."""

    ports: list[int] = field(default_factory=list)
    container_image: str = "example/image:latest"
    cloud: str = "runpod"
    instance_type: str = ""
    accelerator: str = ""
    spot: bool = False
    region: str = ""


class _RequestId(str):
    """A stand-in for ``sky.server.common.RequestId``.

    The real one is a ``str`` subclass.  That is precisely why the pre-port
    defect was silent: ``sky.status(...)[0]`` returned a *character*, and
    ``.get("handle", {})`` on it raised ``AttributeError`` deep inside a
    background thread.  A fake that handed back a list of dicts could not
    reproduce it, and did not -- this file's previous fake mirrored the
    pre-0.7 API and so proved the module consistent with itself.
    """


class _Handle:
    """The ``ResourceHandle`` a resolved ``sky.launch`` hands back."""

    def __init__(self, head_ip: str) -> None:
        self.head_ip = head_ip


class _ClusterRecord:
    """A ``StatusResponse``: a pydantic model with dict-like ``get``."""

    def __init__(self, **fields) -> None:
        self._fields = fields

    def get(self, key, default=None):
        return self._fields.get(key, default)


class _Recorder:
    """A stand-in for the ``sky`` module that records what it was given.

    Its call signatures are checked against the installed SkyPilot by
    ``tests/cloud/test_skypilot_api_contract.py``, so it cannot drift away
    from the real library the way its predecessor did.
    """

    #: What ``stream_and_get`` resolves a launch RequestId to.
    HEAD_IP = "203.0.113.7"

    def __init__(self) -> None:
        self.run = ""
        self.resource_ports = "unset"
        self.envs = "unset"
        self.cloud = None
        self.launch_kwargs = None
        self.down_calls = []
        self.status_calls = []
        #: RequestId -> the payload ``get`` / ``stream_and_get`` resolves.
        self._pending = {}
        self._next = 0

        recorder = self

        class Task:
            def __init__(self, run: str = "", envs=None) -> None:
                recorder.run = run
                recorder.envs = envs

            def set_resources(self, resources) -> None:
                pass

        class Resources:
            def __init__(self, **kwargs) -> None:
                recorder.resource_ports = kwargs.get("ports", "absent")
                recorder.cloud = kwargs.get("cloud", "absent")

        # ``launcher._resolve_sky_cloud_class`` finds a cloud by scanning
        # for a ``sky.clouds.Cloud`` subclass whose ``_REPR`` matches, which
        # is how the real ``sky.RunPod`` (``_REPR == "RunPod"``) is found.
        # The fake carries the same shape, so the resolution the module does
        # is the resolution the tests exercise.
        class Cloud:
            pass

        class RunPod(Cloud):
            _REPR = "RunPod"

        class GCP(Cloud):
            _REPR = "GCP"

        self.Task = Task
        self.Resources = Resources
        self.RunPod = RunPod
        self.GCP = GCP
        self.clouds = types.SimpleNamespace(Cloud=Cloud)

    def _request(self, payload) -> _RequestId:
        self._next += 1
        rid = _RequestId(f"req-{self._next}")
        self._pending[rid] = payload
        return rid

    # -- the client-server surface -------------------------------------
    # Parameter names match ``sky.launch`` / ``sky.status`` / ``sky.down``
    # / ``sky.get`` / ``sky.stream_and_get`` on the installed SkyPilot.

    def launch(self, task, cluster_name=None, **kwargs):
        self.launch_kwargs = dict(kwargs)
        return self._request((1, _Handle(self.HEAD_IP)))

    def status(self, cluster_names=None, **kwargs):
        self.status_calls.append(cluster_names)
        return self._request(
            [_ClusterRecord(handle=_Handle(self.HEAD_IP), status="UP")]
        )

    def down(self, cluster_name, purge=False, **kwargs):
        self.down_calls.append((cluster_name, purge))
        return self._request(None)

    def get(self, request_id):
        if not isinstance(request_id, _RequestId):
            raise AssertionError(
                f"sky.get was handed {request_id!r}, not a RequestId; the "
                "caller resolved something it never requested"
            )
        return self._pending[request_id]

    def stream_and_get(self, request_id=None, **kwargs):
        return self.get(request_id)


@pytest.fixture
def recorder(monkeypatch):
    fake = _Recorder()
    monkeypatch.setattr(_skypilot, "_import_sky", lambda: fake)
    return fake


def _published(run: str) -> list[int]:
    """The container ports a ``docker run`` command line publishes."""
    return [int(host) for host, container in re.findall(r"-p (\d+):(\d+)", run)]


def test_launch_vm_publishes_nothing_when_the_job_config_asks_for_nothing(
    recorder,
):
    """The shipped default: ``JobConfig.ports`` is empty, so nothing is open."""
    _skypilot.launch_vm(_Config(ports=[]))

    assert _published(recorder.run) == []
    assert "-p " not in recorder.run
    assert recorder.resource_ports is None


def test_launch_vm_publishes_only_the_ports_the_job_config_asks_for(recorder):
    """The gate: the publish list is ``config.ports``, not a constant."""
    _skypilot.launch_vm(_Config(ports=[8000]))

    assert _published(recorder.run) == [8000]
    assert recorder.resource_ports == [8000]


def test_launch_vm_never_publishes_the_zmq_ports_unasked(recorder):
    """5555 and 5556 are the state stream and the command channel.

    They carry the full simulation state and an actuation path, so a
    launch that opens them without being asked is the defect
    MADD-ANO-015 records.
    """
    _skypilot.launch_vm(_Config(ports=[8000]))

    assert 5555 not in _published(recorder.run)
    assert 5556 not in _published(recorder.run)


def test_launch_vm_never_publishes_8080(recorder):
    """Nothing in MADDENING has ever served 8080.

    The signaling server listens on 8443 and the API on 8000; the
    ``http://<vm>:8080/health`` probe in ``cloud.session`` could only
    ever time out.  Publishing it was pure attack surface.
    """
    _skypilot.launch_vm(_Config(ports=[8000, 5555, 5556]))

    assert 8080 not in _published(recorder.run)


def test_the_container_publish_and_the_firewall_agree(recorder):
    """A port open in the provider firewall but not published is a trap.

    ``sky.Resources(ports=...)`` opens the provider's firewall; the
    ``-p`` flags publish the container's port to the VM.  If those two
    lists disagree, the operator's mental model of what is reachable is
    wrong in one direction or the other.
    """
    _skypilot.launch_vm(_Config(ports=[8000, 5580]))

    assert _published(recorder.run) == list(recorder.resource_ports)


@pytest.mark.parametrize("bad", [0, -1, 65536, "8000; rm -rf /", "eight"])
def test_launch_vm_refuses_a_port_that_is_not_a_port_number(recorder, bad):
    """Ports are interpolated into a shell command line.

    They are validated as integers in range rather than quoted, so a
    value that is not a port number is refused outright.
    """
    with pytest.raises(ValueError):
        _skypilot.launch_vm(_Config(ports=[bad]))


def test_launch_vm_still_returns_the_vm_ip_and_cluster_name(recorder):
    """The port change must not disturb what launch_vm returns."""
    vm_ip, job_id = _skypilot.launch_vm(_Config(ports=[8000]))

    assert vm_ip == "203.0.113.7"
    assert job_id.startswith("maddening-")


# ----------------------------------------------------------------------
# Credentials reach the container, and only by a carrier that is not argv
# ----------------------------------------------------------------------

SECRET = "s3cret-operator-token"


def test_launch_vm_hands_the_container_the_credentials_it_was_given(recorder):
    """Without this the container generates a token nobody can present.

    ``CloudSession``'s health probes authenticate against the container,
    so the launcher and the container have to hold the same token.  A
    launch that passes none leaves ``wait_ready()`` unable to succeed.
    """
    _skypilot.launch_vm(
        _Config(ports=[8000]), envs={"MADDENING_API_TOKEN": SECRET},
    )

    assert recorder.envs == {"MADDENING_API_TOKEN": SECRET}
    assert "-e MADDENING_API_TOKEN " in recorder.run


def test_a_credential_value_never_reaches_the_docker_command_line(recorder):
    """The gate: ``/proc/<pid>/cmdline`` on the VM is world-readable.

    ``-e NAME`` makes docker read the value from its own environment;
    ``-e NAME=value`` would put the secret in the argv of every process
    in the chain.  The name is interpolated, the value is not.
    """
    _skypilot.launch_vm(
        _Config(ports=[8000]),
        envs={"MADDENING_API_TOKEN": SECRET,
              "MADDENING_TRANSPORT_TOKEN": "a-different-secret"},
    )

    assert SECRET not in recorder.run
    assert "a-different-secret" not in recorder.run
    assert "-e MADDENING_API_TOKEN=" not in recorder.run
    assert "-e MADDENING_TRANSPORT_TOKEN=" not in recorder.run


def test_launch_vm_passes_no_environment_when_it_was_given_none(recorder):
    """The pre-existing shape is unchanged for a caller that passes none."""
    _skypilot.launch_vm(_Config(ports=[8000]))

    assert recorder.envs is None
    assert "-e MADDENING_API_TOKEN" not in recorder.run
    assert "-e MADDENING_TRANSPORT_TOKEN" not in recorder.run


@pytest.mark.parametrize("bad", [
    "MADDENING API TOKEN", "TOKEN; rm -rf /", "$(id)", "", "8TOKEN",
    "TOKEN\nFOO",
])
def test_launch_vm_refuses_an_environment_name_that_is_not_an_identifier(
    recorder, bad,
):
    """Names are interpolated into a shell command line, so they fail closed."""
    with pytest.raises(ValueError):
        _skypilot.launch_vm(_Config(ports=[8000]), envs={bad: SECRET})
