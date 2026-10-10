"""Running ``benchmarks/multigpu/run_capacity.py`` from a test, offline.

The script launches nothing but itself, and a test starts it only as a
dry run: on CPU, on four virtual devices.  Three ways in:

* :func:`run_capacity` starts the script as the session does, in a
  process of its own, with the offline environment of the runner's tests
  (an empty ``HOME``, no cloud variable);
* :func:`in_process` is a launcher the ramp can be given in place of its
  own: it runs a rung's entry point in the test process, where the rung's
  functions can be monkeypatched (a wrong halo, memory statistics as
  numbers) and nothing is compiled twice for the same shapes;
* :func:`scripted` is a launcher that ends each rung the way a test says.

Every one of them refuses an argument list that is not a dry run.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import maddening
from tests.cloud.multigpu.run_pod_support import offline_env

REPO = Path(maddening.__file__).resolve().parents[2]
SCRIPT = REPO / "benchmarks" / "multigpu" / "run_capacity.py"
RUNNER = SCRIPT.with_name("run_pod.py")

#: The tests' tile: the smallest one both meshes take (axis 0 a multiple
#: of four and at least eight; axis 1 a multiple of two and at least
#: four), no two extents alike.  One tile for every test, so the programs
#: compiled per device for one grid serve every test on that grid.
TILE = (8, 4, 6)
TILE_ARGS = ["--tile", *map(str, TILE)]

_MODULE = None


def module():
    """The script, loaded once per test process."""
    global _MODULE
    if _MODULE is None:
        spec = importlib.util.spec_from_file_location("run_capacity_under_test", SCRIPT)
        _MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MODULE)
    return _MODULE


def checked_argv(argv) -> list[str]:
    """``argv`` unchanged, or ``AssertionError`` if it would run a rung
    that is not a dry run (``--summarise`` alone reads JSON)."""
    argv = [str(a) for a in argv]
    if "--summarise" not in argv and "--dry-run" not in argv:
        raise AssertionError(f"refusing to run run_capacity.py without --dry-run: {argv}.  A "
                             "test never runs a rung for real")
    return argv


def run_capacity(argv, *, timeout: float, n_devices: int = 4) -> subprocess.CompletedProcess:
    """Run the script in a process of its own, offline; not checked.  The
    allocator's variables are dropped, so that what a run finds does not
    depend on what an earlier test of this process left set."""
    env = {k: v for k, v in offline_env(str(REPO / "src"), n_devices).items()
           if k not in module().ALLOCATOR_VARIABLES}
    return subprocess.run([sys.executable, str(SCRIPT), *checked_argv(argv)], env=env,
                          capture_output=True, text=True, timeout=timeout, check=False)


def in_process(argv, timeout_s, stderr_path):
    """A launcher for ``run_ramp``: the rung's own entry point, here."""
    rc = module()
    t0 = time.perf_counter()
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        status = rc.exit_status(checked_argv(argv))
    Path(stderr_path).write_text(err.getvalue(), encoding="utf-8")
    return rc.Launched(status, False, "\n".join(err.getvalue().splitlines()[-20:]),
                       time.perf_counter() - t0)


def scripted(steps):
    """A launcher that ends rung ``i`` as ``steps[i]`` says: ``"run"`` runs
    it here (:func:`in_process`); a ``dict`` is written as its record
    (re-labelled with the rung's own number) and the rung "exits 0"; a
    ``Launched`` is returned as it is, its ``stderr_tail`` written to the
    rung's log, with no record.  ``launch.calls`` lists the argument lists."""
    rc = module()
    calls: list = []

    def launch(argv, timeout_s, stderr_path):
        step = steps[len(calls)]
        calls.append(checked_argv(argv))
        if step == "run":
            return in_process(argv, timeout_s, stderr_path)
        if isinstance(step, dict):
            record = dict(step, rung=int(argv[argv.index("--rung-index") + 1]))
            assert [str(k) for k in record["k"]] == argv[argv.index("--k") + 1:
                                                         argv.index("--k") + 4], (record["k"], argv)
            Path(argv[argv.index("--record") + 1]).write_text(json.dumps(record),
                                                              encoding="utf-8")
            Path(stderr_path).write_text("", encoding="utf-8")
            return rc.Launched(0, False, "", 0.0)
        Path(stderr_path).write_text(step.stderr_tail, encoding="utf-8")
        return step

    launch.calls = calls
    return launch


def stub_memory(monkeypatch, *, bytes_per_cell: float, in_use_share: float = 0.25,
                limit: int | None = None, fullest: int = 2, grow: int = 0):
    """Make the rungs run here read memory statistics as numbers: on each
    device a peak of ``bytes_per_cell`` times its share of the cells from
    the first step on (a third of that after the build), a little less on
    every device but ``fullest``; ``bytes_in_use`` at ``in_use_share`` of
    the peak, plus ``grow`` bytes per reading taken.  Returns the dict the
    stub counts in (``readings``: how many were taken)."""
    rc = module()
    seen = {"cells": 0, "readings": 0, "in_pass": 0}
    real_prepare = rc.prepare

    def prepare(args, record, flush):
        problem = real_prepare(args, record, flush)
        seen["cells"] = problem.cells if problem is not None else 0
        return problem

    def read(devices):
        seen["readings"] += 1
        at = seen["in_pass"] = seen["in_pass"] % len(rc.READINGS) + 1
        share = seen["cells"] / len(devices) * bytes_per_cell
        rows = []
        for i, device in enumerate(devices):
            peak = int(share * (1.0 if i == fullest else 0.9) * (1.0 if at > 1 else 1 / 3))
            row = {"device": str(device), "measured": True, "peak_bytes_in_use": peak,
                   "bytes_in_use": int(in_use_share * peak) + grow * seen["readings"]}
            if limit is not None:
                row["bytes_limit"] = limit
            rows.append(row)
        return rows

    monkeypatch.setattr(rc, "prepare", prepare)
    monkeypatch.setattr(rc, "read_device_memory", read)
    return seen
