"""``run_pod.py``'s exit status says whether a check failed, the runner
refused the run, or a goal crashed.

The runbook reads a goal's exit status as "0 = no check failed; 1 = a check
failed (a CHECK FAILED line names it); 124 = the time box ran out; anything
else = a crash" -- but a goal that raised (without ``--keep-going``) and a
refusal raised as ``SystemExit(message)`` both exited 1, with no ``CHECK
FAILED`` line.  A refusal now exits ``EXIT_REFUSED`` (2, argparse's own
status for an option it does not take) with the reason as its last line,
and a crash exits ``EXIT_CRASHED`` (5) after its traceback; the runbook says
so (``tests/compliance/test_run_pod_runbook_agrees_with_the_runner.py``).
``--summarise``'s own statuses (0, 1, 3, 4) are unchanged.  Every run here
is ``--dry-run`` or a goal replaced by a stub: nothing is launched.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import maddening
from tests.cloud.multigpu.run_pod_support import run_pod

_REPO = Path(maddening.__file__).resolve().parents[2]
_RUNNER = _REPO / "benchmarks" / "multigpu" / "run_pod.py"


@pytest.fixture(scope="module")
def rp():
    spec = importlib.util.spec_from_file_location("run_pod_exit_status", _RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_statuses_are_distinct_from_a_failed_check_and_from_the_summarys(rp):
    assert (rp.EXIT_REFUSED, rp.EXIT_CRASHED) == (2, 5)
    assert len({0, 1, rp.EXIT_REFUSED, rp.EXIT_CRASHED, 124}) == 5
    assert {rp.EXIT_REFUSED, rp.EXIT_CRASHED}.isdisjoint({0, 1, 3, 4})


def test_a_refusal_exits_refused_with_its_reason_last(rp, tmp_path, capsys, monkeypatch):
    ran = []
    monkeypatch.setattr(rp, "run_halo", lambda args, out: ran.append(args))
    status = rp.exit_status(["--goal", "halo", "--dry-run", "--n-devices", "100000",
                             "--out", str(tmp_path)])
    assert status == rp.EXIT_REFUSED
    err = capsys.readouterr().err.strip().splitlines()
    assert err and err[-1].startswith("--n-devices 100000 but only")
    assert ran == []


def test_a_goal_that_raises_exits_crashed_after_its_traceback(rp, tmp_path, capsys,
                                                              monkeypatch):
    def boom(args, out):
        raise RuntimeError("seeded crash")

    monkeypatch.setattr(rp, "run_halo", boom)
    status = rp.exit_status(["--goal", "halo", "--dry-run", "--out", str(tmp_path)])
    assert status == rp.EXIT_CRASHED
    err = capsys.readouterr().err
    assert "Traceback" in err and "RuntimeError: seeded crash" in err
    assert err.strip().splitlines()[-1].startswith("CRASHED")
    assert not (tmp_path / "halo.json").exists()


def test_with_keep_going_a_raise_is_a_failed_check_and_exits_1(rp, tmp_path, monkeypatch):
    def boom(args, out):
        raise RuntimeError("seeded crash")

    monkeypatch.setattr(rp, "run_halo", boom)
    assert rp.exit_status(["--goal", "halo", "--dry-run", "--keep-going",
                           "--out", str(tmp_path)]) == 1
    assert (tmp_path / "halo.json").exists()


def test_a_failed_check_still_exits_1_and_a_pass_0(rp, tmp_path, monkeypatch):
    for ok, want in ((True, 0), (False, 1)):
        monkeypatch.setattr(rp, "run_halo",
                            lambda args, out, ok=ok: rp.finish_checks(
                                out, [rp.check_that("seeded", ok)]))
        assert rp.exit_status(["--goal", "halo", "--dry-run",
                               "--out", str(tmp_path / str(ok))]) == want


def test_an_option_it_does_not_take_still_exits_2(rp):
    with pytest.raises(SystemExit) as raised:
        rp.exit_status(["--goal", "halo", "--dry", "--out", "x"])
    assert raised.value.code == 2


def test_the_process_exit_status_of_a_crash(tmp_path):
    """The ``__main__`` wiring, in a real process: a ``--mesh`` file that
    does not exist raises inside the goal (``FileNotFoundError``)."""
    proc = run_pod(_RUNNER, ["--goal", "forward", "--dry-run", "--out", tmp_path,
                             "--mesh", tmp_path / "nope.npz"],
                   pythonpath=str(_REPO / "src"), timeout=300)
    assert proc.returncode == 5, proc.stdout + proc.stderr
    assert "FileNotFoundError" in proc.stderr
    assert proc.stderr.strip().splitlines()[-1].startswith("CRASHED")
