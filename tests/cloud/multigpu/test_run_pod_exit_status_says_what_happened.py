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


def test_a_forward_run_too_large_to_count_exactly_exits_0_and_reads_incomplete(
        rp, tmp_path, monkeypatch, capsys):
    """At 2**24 cells or more the ``forward`` goal's exact count is a check
    not run (the stress tail's 3e7 and 1e8 cells).  Nothing about it can
    make a run exit non-zero: the goal's other checks ran and passed, so
    the run exits 0, prints ``CHECK NOT RUN`` and reads ``INCOMPLETE``.
    Emulated with the threshold lowered to the second of two dry-run
    sizes: under it the count runs, at it it does not."""
    import jax

    assert rp.EXACT_COUNT_CELLS == 2 ** 24
    monkeypatch.setattr(rp, "EXACT_COUNT_CELLS", 256)
    status = rp.exit_status(["--goal", "forward", "--dry-run", "--cells", "225", "256",
                             "--warmup", "0", "--repeats", "1", "--steps", "1",
                             "--out", str(tmp_path)])
    printed = capsys.readouterr().out
    assert status == 0, printed
    (doc,) = rp._load_results(tmp_path, "forward")
    small, large = doc["results"]
    assert (small["cells"], large["cells"]) == (225, 256)
    assert small["ones_unsharded"]["total"] == 225.0 and "ones_unsharded" not in large
    assert all("ones" in m for m in small["methods"].values())
    assert not any("ones" in m for m in large["methods"].values())
    not_run = [c for c in doc["checks"] if c.get("not_run")]
    assert [c["name"] for c in not_run] == [
        "256 cells ones: total == cells exactly, sharded and unsharded"]
    assert all(c["passed"] for c in doc["checks"] if not c.get("not_run"))
    assert doc["passed"] is False and rp.record_problems(doc) == []
    assert rp.goal_verdict([doc]) == "INCOMPLETE"
    assert "CHECK NOT RUN [forward] 256 cells ones: total == cells exactly" in printed
    assert "INCOMPLETE (checks not run; the checklist items stay open): forward" in printed
    assert "CHECK FAILED" not in printed and "FAILED:" not in printed
    assert len(jax.devices()) < 2 or doc["n_devices"] >= 2


def test_an_option_it_does_not_take_still_exits_2(rp):
    with pytest.raises(SystemExit) as raised:
        rp.exit_status(["--goal", "halo", "--dry", "--out", "x"])
    assert raised.value.code == 2


def test_the_process_exit_status_of_a_crash(tmp_path):
    """The ``__main__`` wiring, in a real process: an installation whose
    ``maddening`` cannot be imported raises when the backend is loaded.
    (A ``--mesh`` file that does not exist used to be the crash here; it
    is a refusal now.)"""
    broken = tmp_path / "broken" / "maddening"
    broken.mkdir(parents=True)
    (broken / "__init__.py").write_text("raise RuntimeError('seeded crash')\n")
    proc = run_pod(_RUNNER, ["--goal", "forward", "--dry-run", "--out", tmp_path / "out"],
                   pythonpath=str(broken.parent), timeout=300)
    assert proc.returncode == 5, proc.stdout + proc.stderr
    assert "RuntimeError: seeded crash" in proc.stderr
    assert proc.stderr.strip().splitlines()[-1].startswith("CRASHED")


# ---------------------------------------------------------------------------
# An option value no goal can use is a refusal, before anything is loaded
# ---------------------------------------------------------------------------
# ``--cells -5``, ``--repeats 0``, ``--steps 0`` / ``-3`` and a ``--mesh``
# that does not exist raised inside the first goal that read them (exit 5,
# a traceback) where the runbook says 2; ``--cells 0`` and ``--warmup -1``
# ran and wrote a passing record.

def _bad_mesh(tmp_path: Path, kind: str) -> Path:
    import numpy as np
    path = tmp_path / {"missing": "nope.npz", "a directory": "dir.npz", "not an archive": "junk.npz",
                       "no edges": "empty.npz", "edges of another shape": "flat.npy"}[kind]
    if kind == "a directory":
        path.mkdir()
    elif kind == "not an archive":
        path.write_bytes(b"not a zip archive")
    elif kind == "no edges":
        np.savez(path, partition=np.zeros(4, np.int32))
    elif kind == "edges of another shape":
        np.save(path, np.arange(6, dtype=np.int32))
    return path


_UNUSABLE = [
    (["--cells", "-5"], "--cells -5: must be an integer >= 1"),
    (["--cells", "0"], "--cells 0: must be an integer >= 1"),
    (["--cells", "256", "0", "1024"], "--cells 0: must be an integer >= 1"),
    (["--n-devices", "0"], "--n-devices 0: must be an integer >= 1"),
    (["--n-devices", "-2"], "--n-devices -2: must be an integer >= 1"),
    (["--warmup", "-1"], "--warmup -1: must be an integer >= 0"),
    (["--repeats", "0"], "--repeats 0: must be an integer >= 1"),
    (["--steps", "0"], "--steps 0: must be an integer >= 1"),
    (["--steps", "-3"], "--steps -3: must be an integer >= 1"),
    (["--grad-steps", "0"], "--grad-steps 0: must be an integer >= 1"),
    (["--cg-max-iters", "0"], "--cg-max-iters 0: must be an integer >= 1"),
    (["--fields", "0"], "--fields 0: must be an integer >= 1"),
]


def _refused_before_anything(rp, argv, tmp_path, capsys, monkeypatch) -> str:
    """Run *argv* as a dry run with the backend and every goal replaced by
    a recorder; return the last line of stderr of the refusal."""
    reached = []
    monkeypatch.setattr(rp, "_load_backend", lambda: reached.append("backend"))
    for goal in rp.ALL_GOALS:
        monkeypatch.setattr(rp, f"run_{goal}", lambda args, out, g=goal: reached.append(g))
    out = tmp_path / "out"
    status = rp.exit_status(["--goal", "all", "--dry-run", "--out", str(out), *argv])
    err = capsys.readouterr().err.strip().splitlines()
    assert status == rp.EXIT_REFUSED == 2, (status, err[-3:])
    assert reached == [], f"{argv} was refused after {reached}"
    assert not out.exists() or not list(out.iterdir())
    assert "Traceback" not in "\n".join(err)
    return err[-1]


@pytest.mark.parametrize("argv, reason", _UNUSABLE, ids=[" ".join(a) for a, _ in _UNUSABLE])
def test_a_count_no_goal_can_use_is_refused_before_the_backend_loads(
        rp, tmp_path, capsys, monkeypatch, argv, reason):
    assert _refused_before_anything(rp, argv, tmp_path, capsys, monkeypatch) == reason


@pytest.mark.parametrize("kind, reason", [
    ("missing", "no such file"), ("a directory", "no such file"),
    ("not an archive", "cannot be read as a mesh"), ("no edges", "cannot be read as a mesh"),
    ("edges of another shape", "edges must be (n_edges, 2)"),
])
def test_a_mesh_that_cannot_be_read_is_refused_before_the_backend_loads(
        rp, tmp_path, capsys, monkeypatch, kind, reason):
    mesh = _bad_mesh(tmp_path, kind)
    last = _refused_before_anything(rp, ["--mesh", str(mesh)], tmp_path, capsys, monkeypatch)
    assert last.startswith(f"--mesh {mesh}: ") and reason in last


def test_an_out_that_is_a_file_is_refused_before_the_backend_loads(rp, tmp_path, capsys,
                                                                   monkeypatch):
    taken = tmp_path / "taken"
    taken.write_text("")
    monkeypatch.setattr(rp, "_load_backend", lambda: pytest.fail("the backend was loaded"))
    assert rp.exit_status(["--goal", "halo", "--dry-run", "--out", str(taken)]) == 2
    assert capsys.readouterr().err.strip().splitlines()[-1] == (
        f"--out {taken}: exists and is not a directory")


def test_the_least_value_of_every_count_and_a_readable_mesh_are_taken(rp, tmp_path):
    import numpy as np
    mesh = tmp_path / "ring.npz"
    np.savez(mesh, edges=np.array([[0, 1], [1, 2], [2, 3], [3, 0]], np.int32))
    argv = ["--goal", "all", "--dry-run", "--out", str(tmp_path / "out"), "--mesh", str(mesh)]
    for dest, least in rp.OPTION_MINIMUMS.items():
        argv += ["--" + dest.replace("_", "-"), str(least)]
    args = rp.parse_args(argv)
    rp.check_option_values(args)          # does not raise
    assert rp._MESH_CACHE[("file", str(mesh))][0] == 4
    # ... and an option left out is not asked.
    rp.check_option_values(rp.parse_args(["--goal", "halo", "--dry-run", "--out", str(tmp_path)]))


def test_every_option_of_the_parser_has_its_values_checked(rp):
    """Fails closed on a new option: its values are argparse's (``choices``,
    a flag), a counted minimum, or one of the paths asked by name."""
    paths = {"out", "mesh", "summarise"}
    unchecked = []
    for action in rp._parser()._actions:
        if action.dest == "help" or action.choices is not None or action.nargs == 0:
            continue
        if action.dest in rp.OPTION_MINIMUMS:
            assert action.type is int, action.dest
            continue
        if action.dest not in paths:
            unchecked.append(action.dest)
    assert unchecked == []
    assert set(rp.OPTION_MINIMUMS) <= {a.dest for a in rp._parser()._actions}
