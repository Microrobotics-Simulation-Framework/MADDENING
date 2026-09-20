"""``_skypilot`` against SkyPilot's client-server API: behaviour, not signatures.

``tests/cloud/test_skypilot_api_contract.py`` checks that the calls fit the
installed library.  This file checks what the module *does* with them, and in
particular the two failures that used to be swallowed:

* ``teardown_vm`` wrapped ``sky.down`` in ``except Exception: logger.exception``.
  A teardown that no-ops silently means a VM that keeps running and keeps
  billing, reported to the caller as a clean release.
* ``monitor_preemption`` polled ``check_status`` inside
  ``except Exception: logger.debug``.  Because ``check_status`` could not work
  at all, the preemption callback could never fire and nothing above ``DEBUG``
  said so.

None of this establishes that a real teardown releases a real VM or that a
real preemption is noticed; that needs a provider.  See MADD-ANO-016.
"""

from __future__ import annotations

import logging
import threading

import pytest

from maddening.cloud import _skypilot

from tests.cloud.test_skypilot_ports import (
    _ClusterRecord,
    _Config,
    _Handle,
    _Recorder,
    recorder,  # noqa: F401  (fixture)
)


class _Status:
    """A ``ClusterStatus``-alike: ``.value`` is ``UP``, ``str()`` is not."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:                      # pragma: no cover - clarity
        return f"ClusterStatus.{self.value}"


# ---------------------------------------------------------------------------
# launch_vm
# ---------------------------------------------------------------------------

class TestLaunchVm:
    def test_the_launch_request_is_resolved_and_the_handle_supplies_the_ip(
        self, recorder,  # noqa: F811
    ):
        """``sky.launch`` returns a RequestId; the IP comes from resolving it.

        The fake's ``get`` refuses anything that is not a RequestId it
        issued, so a module that read the RequestId directly would fail here
        rather than quietly returning a character.
        """
        vm_ip, cluster_name = _skypilot.launch_vm(_Config(ports=[8000]))

        assert vm_ip == _Recorder.HEAD_IP
        assert cluster_name.startswith("maddening-")

    def test_the_launch_call_passes_no_detach_run(self, recorder):  # noqa: F811
        _skypilot.launch_vm(_Config(ports=[8000]))

        assert recorder.launch_kwargs == {}

    def test_the_cloud_the_config_names_is_the_cloud_that_is_used(
        self, recorder,  # noqa: F811
    ):
        """``getattr(sky, "runpod".upper())`` missed and fell back to GCP, so
        every CloudSession launch went to a provider nobody asked for."""
        _skypilot.launch_vm(_Config(ports=[], cloud="runpod"))

        assert isinstance(recorder.cloud, recorder.RunPod)
        assert not isinstance(recorder.cloud, recorder.GCP)

    def test_an_unresolvable_cloud_is_refused_rather_than_redirected(
        self, recorder,  # noqa: F811
    ):
        with pytest.raises(ValueError, match="No SkyPilot cloud class"):
            _skypilot.launch_vm(_Config(ports=[], cloud="not-a-cloud"))

        assert recorder.launch_kwargs is None, "it launched anyway"

    def test_the_status_fallback_is_used_when_the_handle_has_no_ip(
        self, recorder, monkeypatch,  # noqa: F811
    ):
        """Some backends fill the IP in late.  The fallback goes through
        ``sky.get(sky.status(...))``, not through the RequestId."""
        monkeypatch.setattr(
            recorder, "launch",
            lambda task, cluster_name=None, **kw: recorder._request(
                (1, _Handle("")),
            ),
        )
        vm_ip, cluster_name = _skypilot.launch_vm(_Config(ports=[]))

        assert vm_ip == _Recorder.HEAD_IP
        assert recorder.status_calls == [[cluster_name]]

    def test_an_ip_that_never_arrives_is_reported_as_unknown(
        self, recorder, monkeypatch,  # noqa: F811
    ):
        monkeypatch.setattr(
            recorder, "launch",
            lambda task, cluster_name=None, **kw: recorder._request((1, None)),
        )
        monkeypatch.setattr(
            recorder, "status",
            lambda cluster_names=None, **kw: recorder._request([]),
        )
        vm_ip, _ = _skypilot.launch_vm(_Config(ports=[]))

        assert vm_ip == "unknown"


# ---------------------------------------------------------------------------
# check_status
# ---------------------------------------------------------------------------

class TestCheckStatus:
    def test_the_bare_status_name_is_returned_not_the_enum_repr(
        self, recorder, monkeypatch,  # noqa: F811
    ):
        """``monitor_preemption`` compares against ``"STOPPED"``.  Returning
        ``"ClusterStatus.STOPPED"`` would never match, so a preemption would
        be read as a healthy cluster."""
        monkeypatch.setattr(
            recorder, "status",
            lambda cluster_names=None, **kw: recorder._request(
                [_ClusterRecord(status=_Status("STOPPED"))],
            ),
        )
        assert _skypilot.check_status("maddening-1") == "STOPPED"

    def test_a_plain_string_status_is_passed_through(self, recorder):  # noqa: F811
        assert _skypilot.check_status("maddening-1") == "UP"

    def test_an_unknown_cluster_is_not_found(self, recorder, monkeypatch):  # noqa: F811
        monkeypatch.setattr(
            recorder, "status",
            lambda cluster_names=None, **kw: recorder._request([]),
        )
        assert _skypilot.check_status("gone") == "not_found"

    def test_a_record_without_a_status_field_is_unknown_not_a_crash(
        self, recorder, monkeypatch,  # noqa: F811
    ):
        monkeypatch.setattr(
            recorder, "status",
            lambda cluster_names=None, **kw: recorder._request(
                [_ClusterRecord()],
            ),
        )
        assert _skypilot.check_status("maddening-1") == "unknown"


# ---------------------------------------------------------------------------
# teardown_vm -- a leaked VM is not a log line
# ---------------------------------------------------------------------------

class TestTeardownVm:
    def test_a_successful_teardown_purges_and_resolves_the_request(
        self, recorder,  # noqa: F811
    ):
        _skypilot.teardown_vm("maddening-1")

        assert recorder.down_calls == [("maddening-1", True)]

    def test_a_failing_teardown_raises_rather_than_logging_and_returning(
        self, recorder, monkeypatch,  # noqa: F811
    ):
        """The defect: ``except Exception: logger.exception`` made a teardown
        that could never work indistinguishable from one that worked."""
        def boom(cluster_name, purge=False, **kw):
            raise RuntimeError("API server unreachable")

        monkeypatch.setattr(recorder, "down", boom)

        with pytest.raises(_skypilot.TeardownError) as caught:
            _skypilot.teardown_vm("maddening-1")

        message = str(caught.value)
        assert "maddening-1" in message
        assert "still be running" in message
        assert "sky down maddening-1" in message
        assert isinstance(caught.value.__cause__, RuntimeError)

    def test_a_teardown_whose_request_fails_to_resolve_also_raises(
        self, recorder, monkeypatch,  # noqa: F811
    ):
        """``sky.down`` returning a RequestId means the work happens in
        ``sky.get``; a failure there is the same leaked VM."""
        def boom(request_id):
            raise RuntimeError("request failed")

        monkeypatch.setattr(recorder, "get", boom)

        with pytest.raises(_skypilot.TeardownError, match="maddening-1"):
            _skypilot.teardown_vm("maddening-1")


# ---------------------------------------------------------------------------
# monitor_preemption -- a broken check is not silence
# ---------------------------------------------------------------------------

def _drain(thread: threading.Thread, timeout: float = 5.0) -> None:
    thread.join(timeout)
    assert not thread.is_alive(), "the monitor thread never finished"


class TestMonitorPreemption:
    @pytest.mark.parametrize("status", ["STOPPED", "not_found", "PREEMPTED"])
    def test_the_callback_fires_on_a_preemption_signal(self, monkeypatch, status):
        monkeypatch.setattr(_skypilot, "check_status", lambda job_id: status)
        fired = threading.Event()

        thread = _skypilot.monitor_preemption(
            "maddening-1", fired.set, poll_interval=0.001,
        )
        _drain(thread)

        assert fired.is_set()

    def test_a_healthy_cluster_does_not_fire_the_callback(self, monkeypatch):
        seen = []

        def status(job_id):
            seen.append(job_id)
            return "UP" if len(seen) < 3 else "STOPPED"

        monkeypatch.setattr(_skypilot, "check_status", status)
        fired = threading.Event()

        thread = _skypilot.monitor_preemption(
            "maddening-1", fired.set, poll_interval=0.001,
        )
        _drain(thread)

        assert fired.is_set()
        assert len(seen) == 3, "it stopped polling while the cluster was UP"

    def test_a_permanently_raising_status_check_gives_up_loudly(
        self, monkeypatch, caplog,
    ):
        """The defect: an ``except Exception: logger.debug`` in a ``while
        True`` meant a status check that could never succeed produced an
        infinite silent loop, and the callback could never fire."""
        calls = []

        def boom(job_id):
            calls.append(job_id)
            raise AttributeError("'str' object has no attribute 'get'")

        monkeypatch.setattr(_skypilot, "check_status", boom)
        fired = threading.Event()

        with caplog.at_level(logging.ERROR, logger=_skypilot.__name__):
            thread = _skypilot.monitor_preemption(
                "maddening-1", fired.set,
                poll_interval=0.001, max_consecutive_errors=3,
            )
            _drain(thread)

        assert len(calls) == 3, "it did not stop after the error budget"
        assert not fired.is_set(), (
            "a broken status check is not evidence of preemption; firing the "
            "callback would tear down a healthy session"
        )
        errors = [r.getMessage() for r in caplog.records
                  if r.levelno >= logging.ERROR]
        assert any("Preemption check 1/3 failed" in m for m in errors), errors
        assert any("preemption is NOT being watched" in m.replace("\n", " ")
                   or "NOT being " in m for m in errors), errors

    def test_a_transient_error_does_not_end_the_monitor(self, monkeypatch, caplog):
        """One failure is normal; the budget is for a *run* of them."""
        outcomes = [AttributeError("blip"), "UP", "STOPPED"]

        def flaky(job_id):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        monkeypatch.setattr(_skypilot, "check_status", flaky)
        fired = threading.Event()

        with caplog.at_level(logging.ERROR, logger=_skypilot.__name__):
            thread = _skypilot.monitor_preemption(
                "maddening-1", fired.set,
                poll_interval=0.001, max_consecutive_errors=2,
            )
            _drain(thread)

        assert fired.is_set(), "the counter did not reset after a success"
        assert not outcomes
        assert not any("Giving up" in r.getMessage() for r in caplog.records)
