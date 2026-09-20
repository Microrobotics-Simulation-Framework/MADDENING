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


class _Recorder:
    """A stand-in for the ``sky`` module that records what it was given."""

    def __init__(self) -> None:
        self.run = ""
        self.resource_ports = "unset"
        self.envs = "unset"

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

        self.Task = Task
        self.Resources = Resources
        self.RUNPOD = object()
        self.GCP = lambda: object()

    def launch(self, task, cluster_name: str, detach_run: bool = True):
        return "job-1"

    def status(self, cluster_names):
        return [{"handle": {"head_ip": "203.0.113.7"}}]


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
