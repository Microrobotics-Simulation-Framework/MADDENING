"""Credentials must not reach a command line, and demos must not listen wide.

Three small exposures the round-2 audit measured, each with a gate:

* ``CloudJob.ssh_run`` had no way to set a remote environment variable
  other than writing ``VAR=value cmd`` into the command string.  That
  string becomes one argv element of the local ``ssh`` process and then
  of the remote shell, and ``/proc/<pid>/cmdline`` is mode 0444 -- so any
  user on either machine can read the token for as long as the process
  lives.  ``shlex.quote`` prevents word splitting and does nothing about
  that.  Two shipped cloud examples did exactly this.
* ``SelkiesSession`` hardcoded ``websockets.serve(handler, "0.0.0.0",
  8443)``, so its socket could not be made loopback-only even
  deliberately.
* ``examples/servers/vessel_flow_server`` defaulted ``--host`` to
  ``0.0.0.0`` while building its own FastAPI app with no credential of
  any kind -- and every other shipped example server had moved to
  loopback in this release.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from maddening.api.auth import is_loopback

SECRET = "s3cret-operator-token"


# ----------------------------------------------------------------------
# A credential travels over ssh's stdin, never in an argv
# ----------------------------------------------------------------------

@pytest.fixture
def recorded_ssh(monkeypatch):
    """A ``CloudJob`` whose ``subprocess.run`` is recorded, not run."""
    import subprocess
    import types

    from maddening.cloud.launcher import CloudJob

    calls: list[dict] = []

    def _run(cmd, **kwargs):
        calls.append({"cmd": list(cmd), "kwargs": kwargs})
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    return CloudJob("cluster", vm_ip="203.0.113.9"), calls


def test_a_token_passed_as_env_never_appears_in_the_ssh_argv(recorded_ssh):
    """The gate.  ``/proc/<pid>/cmdline`` is world-readable."""
    job, calls = recorded_ssh

    job.ssh_run_background(
        "python3 /tmp/server.py", env={"MADDENING_API_TOKEN": SECRET},
    )

    argv = calls[0]["cmd"]
    assert not any(SECRET in part for part in argv), argv
    assert "MADDENING_API_TOKEN" in argv[-1], (
        "the variable is still exported, only its value is not on the line"
    )


def test_the_value_is_delivered_on_stdin_so_the_command_can_use_it(recorded_ssh):
    """The control: the token has to actually arrive, or this is a no-op."""
    job, calls = recorded_ssh

    job.ssh_run_background(
        "python3 /tmp/server.py", env={"MADDENING_API_TOKEN": SECRET},
    )

    payload = calls[0]["kwargs"]["input"]
    assert payload == (SECRET + "\n").encode()


def test_the_remote_shell_exports_before_the_command_runs(recorded_ssh):
    """A backgrounded process inherits it only if the export precedes it."""
    job, calls = recorded_ssh

    job.ssh_run_background(
        "python3 /tmp/server.py", env={"MADDENING_API_TOKEN": SECRET},
    )

    remote = calls[0]["cmd"][-1]
    assert remote.index("export MADDENING_API_TOKEN") < remote.index("nohup")


def test_a_command_with_no_env_is_unchanged(recorded_ssh):
    """The pre-existing shape, so nothing else has to know about this."""
    job, calls = recorded_ssh

    job.ssh_run("uptime")

    assert calls[0]["cmd"][-1] == "uptime"
    assert "input" not in calls[0]["kwargs"]


@pytest.mark.parametrize("name", ["BAD NAME", "NAME; rm -rf /", "$(id)", "1ST"])
def test_an_environment_name_that_is_not_an_identifier_is_refused(
    recorded_ssh, name,
):
    """Names *are* interpolated into the remote shell, so they fail closed."""
    job, _ = recorded_ssh

    with pytest.raises(ValueError):
        job.ssh_run("true", env={name: SECRET})


def test_a_value_containing_a_newline_is_refused(recorded_ssh):
    """The delivery is line-based; a newline would shift every later value."""
    job, _ = recorded_ssh

    with pytest.raises(ValueError):
        job.ssh_run("true", env={"TOKEN": "first\nsecond"})


def test_the_shipped_cloud_examples_no_longer_build_the_token_into_a_command():
    """The two examples the audit found, pinned at the source.

    They run on a real VM, so there is nothing to execute here; what is
    checkable is that the pattern is gone.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    examples = root / "src/maddening/examples/cloud/server"
    for path in ("04_server_test.py", "05_websocket_test.py"):
        source = (examples / path).read_text()
        assert "MADDENING_API_TOKEN={shlex.quote" not in source, path
        assert 'env={"MADDENING_API_TOKEN": API_TOKEN}' in source, path


# ----------------------------------------------------------------------
# Binds
# ----------------------------------------------------------------------

def test_the_signaling_server_bind_can_be_chosen():
    """It was hardcoded to 0.0.0.0, so it could not be narrowed at all.

    The default stays 0.0.0.0 -- the session normally runs in a
    container, where loopback is unreachable even with a published port
    -- but an operator who tunnels can now say so.  Constructing a real
    ``SelkiesSession`` needs GStreamer, so this checks the signature and
    that the bind address reaches the ``serve`` call rather than a
    literal.
    """
    import inspect

    from maddening.cloud import selkies_session

    signature = inspect.signature(selkies_session.SelkiesSession.__init__)
    assert signature.parameters["bind_host"].default == "0.0.0.0"

    source = inspect.getsource(selkies_session)
    assert 'websockets.serve(handler, "0.0.0.0"' not in source
    assert "handler, self._bind_host, self._signaling_port" in source


def _host_literals(path: pathlib.Path) -> list[str]:
    """Every bind address this module names as a literal.

    Covers ``uvicorn.run(..., host="X")`` and
    ``add_argument("--host", ..., default="X")``.  Parsed rather than
    imported: these modules pull in JAX, build LBM grids and start
    runners at import time.
    """
    tree = ast.parse(path.read_text())
    hosts = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        positional = [
            arg.value for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        ]
        is_host_argument = "--host" in positional
        for keyword in node.keywords:
            if not isinstance(keyword.value, ast.Constant):
                continue
            if not isinstance(keyword.value.value, str):
                continue
            if keyword.arg == "host" or (is_host_argument
                                         and keyword.arg == "default"):
                hosts.append(keyword.value.value)
    return hosts


def test_every_shipped_example_server_binds_loopback():
    """Enumerated, so the next example is covered without anybody asking.

    ``vessel_flow_server`` defaulted to 0.0.0.0 with no credential of any
    kind while every other example in this directory had moved to
    loopback -- a difference nobody would notice by reading one file.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    servers = sorted((root / "src/maddening/examples/servers").glob("*.py"))
    assert len(servers) >= 8, f"found only {len(servers)} example servers"

    exposed = {
        path.name: [h for h in _host_literals(path) if not is_loopback(h)]
        for path in servers
    }

    assert not {k: v for k, v in exposed.items() if v}, (
        "these shipped example servers name a non-loopback bind as a "
        f"literal or a default: { {k: v for k, v in exposed.items() if v} }"
    )


def test_the_bind_enumeration_actually_reads_the_hosts():
    """An enumeration that finds nothing passes trivially."""
    root = pathlib.Path(__file__).resolve().parents[2]
    path = root / "src/maddening/examples/servers/vessel_flow_server.py"

    assert "127.0.0.1" in _host_literals(path)
