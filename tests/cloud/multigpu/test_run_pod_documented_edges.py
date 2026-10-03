"""``run_pod.py`` does what its docstring and its runbook say, at their edges.

The rows of ``docs/validation/rest_runpod_claims.yaml`` (``RPD-NNN``) that
``test_run_pod_dry_run.py`` and ``test_run_pod_verdict_integrity.py`` do
not already pin.  Everything here runs in-process on CPU, and every
``main()`` call carries ``--dry-run`` or ``--summarise``: no test runs a
goal for real.

* the summary's exit codes at their edge: a goal file a killed run left
  truncated, or one that is not a JSON object, must read ``INVALID`` and
  exit 3 (strict xfails: each stops the summary on a traceback, exit 1,
  which means "no goal JSON"); an older runner's file whose tables this
  runner cannot read reads ``INVALID`` and exits 3;
* the commit a file records: a repository with no commit records none,
  and a recorded commit that is not a SHA counts as none;
* the checklist's goals and the seeded faults the runbook names (the
  tests that read the runbook itself are in
  ``tests/compliance/test_run_pod_runbook_agrees_with_the_runner.py``,
  which a docs-only change runs);
* the CLI: what is required, the device count, the default sizes of a dry
  run, the order of ``--goal all``, the exit status of a goal whose every
  check was not run, the dry run's backend and its ``nvidia-smi`` skip;
* the grids, ``--mesh`` files and the partition fallback;
* the transport recommendation at its thresholds.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import maddening

_REPO = Path(maddening.__file__).resolve().parents[2]
_RUNNER = _REPO / "benchmarks" / "multigpu" / "run_pod.py"
_RECORD = Path(__file__).resolve().parent / "run_pod_record"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rp():
    return _load(_RUNNER, "run_pod_documented_edges")


def _copy_record(tmp_path: Path) -> Path:
    out = tmp_path / "results"
    shutil.copytree(_RECORD, out)
    return out


# ---------------------------------------------------------------------------
# The summary's exit codes at their edge
# ---------------------------------------------------------------------------

def test_a_truncated_goal_file_reads_invalid_and_the_summary_exits_3(rp, tmp_path):
    """The runbook runs every goal under ``timeout``, which kills the
    runner wherever it is -- inside ``json.dump`` too.  A truncated file
    is evidence of nothing, and the summary says how it reads one: a
    file it cannot read is ``INVALID`` (exit 3)."""
    directory = _copy_record(tmp_path)
    halo = directory / "halo.json"
    text = halo.read_text(encoding="utf-8")
    halo.write_text(text[: len(text) // 2], encoding="utf-8")
    assert rp.summarise(directory) == 3


def test_a_goal_file_that_is_not_an_object_reads_invalid_and_the_summary_exits_3(rp, tmp_path):
    directory = _copy_record(tmp_path)
    (directory / "halo.json").write_text("[]", encoding="utf-8")
    assert rp.summarise(directory) == 3


@pytest.mark.parametrize("content", [b"", b"\xff\xfe not utf-8", b"null", b'"halo"'],
                         ids=["empty", "not-utf8", "null", "string"])
def test_any_goal_file_that_is_not_a_json_object_reads_invalid_naming_the_file(
        rp, tmp_path, capsys, content):
    """RPD-009's neighbours: an empty file, one not in UTF-8, JSON null and a JSON
    string -- each read INVALID, naming the file and why, and the summary exits 3."""
    directory = _copy_record(tmp_path)
    (directory / "halo.json").write_bytes(content)
    assert rp.summarise(directory) == 3
    out = capsys.readouterr().out
    assert "halo.json" in out and "cannot be read as a JSON object" in out, out


def test_an_older_runners_file_reads_invalid_instead_of_stopping_the_summary(rp, tmp_path,
                                                                            capsys):
    """The runbook: "If the runner itself changed between the session and
    the summary ... the session's files no longer match it and read
    INVALID".  A schema-5 ``indivisible.json`` has no ``stencil_axis1``."""
    directory = _copy_record(tmp_path)
    path = directory / "indivisible.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["schema_version"] = 5
    for r in doc["results"]:
        r.pop("stencil_axis1", None)
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert rp.summarise(directory) == 3
    out = capsys.readouterr().out
    assert next(ln for ln in out.splitlines() if ln.startswith("indivisible ")).endswith(
        "INVALID")


def test_a_file_the_runner_can_read_but_would_not_write_exits_3_with_its_tables(rp, tmp_path,
                                                                               capsys):
    """The companion: a schema-5 file that still has every key the tables
    read is ``INVALID`` and the summary finishes (exit 3)."""
    directory = _copy_record(tmp_path)
    path = directory / "halo.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["schema_version"] = 5
    path.write_text(json.dumps(doc), encoding="utf-8")
    assert rp.summarise(directory) == 3
    out = capsys.readouterr().out
    assert "schema_version 5, not 7" in out
    assert "Halo exchange vs NumPy" in out


# ---------------------------------------------------------------------------
# The commit a file records
# ---------------------------------------------------------------------------

def _git(*args, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, timeout=60)


def test_a_repository_with_no_commit_records_no_commit(tmp_path):
    """``git rev-parse HEAD`` in a repository with no commit prints
    ``HEAD`` and exits 128 -- a tree synced without its ``.git`` and then
    ``git init``-ed.  The runbook: each file records the commit
    ``git rev-parse HEAD`` names, and a file that records none keeps every
    item open."""
    _git("init", "-q", ".", cwd=tmp_path)
    shutil.copy(_RUNNER, tmp_path / "run_pod.py")
    here = _load(tmp_path / "run_pod.py", "run_pod_in_a_repository_with_no_commit")
    assert here._git_commit() is None, here._git_commit()
    _git("add", "run_pod.py", cwd=tmp_path)
    _git("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "c",
         cwd=tmp_path)
    sha = here._git_commit()
    assert isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}([0-9a-f]{24})?", sha), sha


def test_a_recorded_commit_that_is_not_a_sha_counts_as_none(rp):
    assert rp._commit_of({"environment": {"git_commit": "0" * 40}}) == "0" * 40
    assert rp._commit_of({"environment": {"git_commit": ""}}) is None
    assert rp._commit_of({"environment": {"git_commit": "HEAD"}}) is None


# ---------------------------------------------------------------------------
# The runbook and the runner agree
# ---------------------------------------------------------------------------

def test_the_checklist_items_are_decided_by_the_runbooks_goals(rp):
    assert {i: tuple(goals) for i, (_claim, goals) in rp.CHECKLIST.items()} == {
        1: ("stencil", "forward"), 2: ("halo",), 3: ("stencil", "gradient", "coupled"),
        4: ("hybrid",), 5: ("indivisible",), 6: ("coupled",)}
    assert set(rp.GOAL_CHECKS) == set(rp.ALL_GOALS)
    assert rp.CHECKLIST_GOALS == ("indivisible", "halo", "coupled", "stencil", "hybrid")
    assert rp.ALL_GOALS == rp.CHECKLIST_GOALS + ("exchange", "forward", "gradient")
    assert rp.MIN_DECIDING_DEVICES == 4


def test_every_seeded_fault_the_runbook_names_is_a_seed():
    """RPD-024: the runner's docstring counts the seeded faults, and the count is the
    seeded test's: eight, the last of them domain integrals summed over the first mesh
    axis only (it used to count seven and leave it out).  The runbook's count is
    ``tests/compliance/test_run_pod_runbook_counts_its_seeds.py``'s, since a docs-only
    change runs only the compliance tests."""
    import re

    faults = _load(Path(__file__).resolve().parent / "test_run_pod_seeded_faults.py",
                   "run_pod_seeded_faults_for_their_names")
    words = {"seven": 7, "eight": 8, "nine": 9}
    runner = _RUNNER.read_text(encoding="utf-8")
    in_runner = re.search(r"broken in any of (\w+) ways", runner)
    assert in_runner
    assert words[in_runner.group(1)] == len(faults._SEEDS)
    assert "domain integrals summed over the first mesh axis only" in runner.replace("\n", " ")


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("argv", [[], ["--goal", "halo"], ["--out", "x"], ["--dry-run"]])
def test_goal_and_out_are_required_unless_summarising(rp, argv):
    with pytest.raises(SystemExit) as raised:
        rp.parse_args(argv)
    assert raised.value.code == 2
    assert rp.parse_args(["--summarise", "somewhere"]).goal is None


def test_more_devices_than_are_visible_are_refused_before_any_goal_runs(rp, tmp_path,
                                                                       monkeypatch):
    ran = []
    monkeypatch.setattr(rp, "run_halo", lambda args, out: ran.append(args))
    with pytest.raises(SystemExit, match=r"--n-devices 100000 but only \d+ device"):
        rp.main(["--goal", "halo", "--dry-run", "--n-devices", "100000",
                 "--out", str(tmp_path)])
    assert ran == [] and not (tmp_path / "halo.json").exists()


def _recording(rp, goals, seen, checks=None):
    def make(goal):
        def run(args, out):
            seen.append((goal, tuple(args.cells), args.warmup, args.repeats, args.steps,
                         args.cg_max_iters, args.n_devices))
            return rp.finish_checks(out, checks(rp) if checks else [rp.check_that("ok", True)])
        return run
    return {f"run_{g}": make(g) for g in goals}


def test_a_dry_run_runs_every_goal_in_order_at_small_sizes(rp, tmp_path, monkeypatch):
    import jax

    seen: list = []
    for name, fn in _recording(rp, rp.ALL_GOALS, seen).items():
        monkeypatch.setattr(rp, name, fn)
    assert rp.main(["--goal", "all", "--dry-run", "--out", str(tmp_path)]) == 0
    assert [s[0] for s in seen] == list(rp.ALL_GOALS)
    visible = len(jax.devices())
    assert {s[1:] for s in seen} == {(tuple(rp.DRY_RUN_CELLS), 1, 3, 3, 300,
                                      min(rp.MIN_DECIDING_DEVICES, visible))}
    assert rp.DRY_RUN_CELLS == (256, 1024)
    assert rp.GPU_CELLS == (100_000, 300_000, 1_000_000)
    for goal in rp.ALL_GOALS:
        assert json.loads((tmp_path / f"{goal}.json").read_text())["dry_run"] is True


def test_a_goal_whose_every_check_was_not_run_exits_1_and_one_with_a_check_run_exits_0(
        rp, tmp_path, monkeypatch):
    def none_ran(rp_):
        return [rp_.check_not_run("pencil", "needs 4 devices")]

    def one_ran(rp_):
        return [rp_.check_that("ok", True), rp_.check_not_run("pencil", "needs 4 devices")]

    for checks, want in ((one_ran, 0), (none_ran, 1)):
        seen: list = []
        for name, fn in _recording(rp, ["halo"], seen, checks).items():
            monkeypatch.setattr(rp, name, fn)
        out = tmp_path / checks.__name__
        assert rp.main(["--goal", "halo", "--dry-run", "--out", str(out)]) == want, checks
        assert json.loads((out / "halo.json").read_text())["passed"] is False


def test_a_dry_run_pins_the_cpu_backend_with_four_devices_unless_told_otherwise(rp,
                                                                              monkeypatch):
    def setup(argv, **env):
        for key in ("JAX_PLATFORMS", "XLA_FLAGS"):
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        rp._pre_import_setup(argv)
        import os
        return os.environ.get("JAX_PLATFORMS"), os.environ.get("XLA_FLAGS")

    four = "--xla_force_host_platform_device_count=4"
    assert setup(["--dry-run"]) == ("cpu", four)
    assert setup(["--dry-run"], JAX_PLATFORMS="cpu", XLA_FLAGS="--x") == ("cpu", f"--x {four}")
    kept = "--xla_force_host_platform_device_count=8"
    assert setup(["--dry-run"], XLA_FLAGS=kept) == ("cpu", kept)
    assert setup(["--dry-run"], JAX_PLATFORMS="cuda") == ("cuda", None)
    assert setup(["--goal", "halo"]) == (None, None)


def test_a_dry_run_environment_never_runs_nvidia_smi(rp, monkeypatch):
    calls = []
    real = rp.subprocess.run

    def recording(argv, *args, **kwargs):
        calls.append(list(argv))
        return real(argv, *args, **kwargs)

    monkeypatch.setattr(rp.subprocess, "run", recording)
    env = rp.environment(dry_run=True)
    assert env["nvidia_smi"] == "skipped (dry run)"
    assert not any("nvidia-smi" in a[0] for a in calls), calls
    assert env["n_devices_visible"] == len(env["devices"])


# ---------------------------------------------------------------------------
# Grids, meshes and partitions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_devices", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("cells", [16, 256, 1024, 100_000, 1_000_000])
def test_a_grid_is_never_square_and_splits_on_every_mesh(rp, n_devices, cells):
    ny, nx = rp.field_shape(cells, n_devices)
    assert nx == ny + n_devices
    assert ny % n_devices == 0 and nx % n_devices == 0
    if n_devices % 2 == 0:
        assert ny % 2 == 0 and nx % (n_devices // 2) == 0       # the 2 x D/2 pencil


def test_a_mesh_file_is_read_as_the_runner_documents(rp, tmp_path):
    n, edges = rp.grid_edges(64)
    np.save(tmp_path / "edges.npy", edges)
    assert rp.load_mesh(str(tmp_path / "edges.npy"))[0] == n
    assert rp.load_mesh(str(tmp_path / "edges.npy"))[2] is None
    pa = (np.arange(n) * 4 // n).astype(np.int64)
    np.savez(tmp_path / "with_partition.npz", edges=edges, partition=pa)
    n_got, edges_got, pa_got = rp.load_mesh(str(tmp_path / "with_partition.npz"))
    assert n_got == n and pa_got.dtype == np.int32 and pa_got.tolist() == pa.tolist()
    np.save(tmp_path / "bad.npy", edges.reshape(-1))
    with pytest.raises(ValueError, match=r"edges must be \(n_edges, 2\)"):
        rp.load_mesh(str(tmp_path / "bad.npy"))


def test_without_a_partition_the_cells_are_split_by_the_first_method_available(rp,
                                                                              monkeypatch):
    n, edges = rp.grid_edges(100)
    pa, how = rp.partition_cells(n, edges, 4, "contiguous")
    assert how == "contiguous" and pa.tolist() == (np.arange(n) * 4 // n).tolist()
    assert rp.partition_cells(n, edges, 1, "auto")[1] == "single"
    import builtins

    real_import = builtins.__import__

    def without(*names):
        def fake(name, *args, **kwargs):
            if name.split(".")[0] in names:
                raise ImportError(name)
            return real_import(name, *args, **kwargs)
        return fake

    monkeypatch.setattr(builtins, "__import__", without("pymetis"))
    pa, how = rp.partition_cells(n, edges, 4, "auto")
    assert how == "rcm" and sorted(set(pa.tolist())) == [0, 1, 2, 3]
    monkeypatch.setattr(builtins, "__import__", without("pymetis", "scipy"))
    assert rp.partition_cells(n, edges, 4, "auto")[1] == "contiguous"


# ---------------------------------------------------------------------------
# The transport recommendation at its thresholds
# ---------------------------------------------------------------------------

def _exchange_doc(rp, rows, *, n_devices=4, allow_fewer=False):
    results = []
    for cells, a2a, ppm in rows:
        results.append({
            "cells": cells, "requested_cells": cells, "n_devices": n_devices,
            "bit_identical": True, "input_presharded": True, "ppermute_speedup_median": a2a / ppm,
            "methods": {"all_to_all": {"median_ms": a2a, "min_ms": a2a, "bytes_total": 1},
                        "ppermute": {"median_ms": ppm, "min_ms": ppm, "bytes_total": 1}}})
    doc = {"schema_version": rp.SCHEMA_VERSION, "goal": "exchange", "dry_run": False,
           "allow_fewer_devices": allow_fewer, "n_devices": n_devices, "results": results,
           "environment": {"platform": "gpu", "device_kinds": ["gpu"],
                           "devices": [f"gpu:{i}" for i in range(n_devices)],
                           "n_devices_visible": n_devices, "git_commit": "0" * 40},
           "config": {"cells": [c for c, _, _ in rows], "synthetic": "ring", "mesh": None,
                      "n_devices": n_devices, "allow_fewer_devices": allow_fewer}}
    return rp.finish_checks(doc, rp.exchange_checks(results, n_devices))


@pytest.mark.parametrize("ppermute_ms, decision", [
    (1.0 / 1.05, "ppermute"),        # exactly the margin
    (1.0 / 1.0499, "tie"),           # just under it
    (1.0, "tie"),                    # equal: not faster, not slower
    (1.0001, "all_to_all"),          # slower at a deciding row
])
def test_the_recommendation_at_its_thresholds(rp, ppermute_ms, decision):
    doc = _exchange_doc(rp, [(100_000, 1.0, ppermute_ms), (1_000_000, 2.0, 1.0)])
    assert rp.recommend([doc])["decision"] == decision


@pytest.mark.parametrize("n_devices, allow_fewer, cells, deciding", [
    (4, False, 100_000, True), (4, False, 99_999, False), (3, False, 100_000, False),
    (3, True, 100_000, True), (2, True, 100_000, True), (1, True, 100_000, False),
])
def test_a_row_decides_only_at_enough_cells_and_devices(rp, n_devices, allow_fewer, cells,
                                                        deciding):
    doc = _exchange_doc(rp, [(cells, 2.0, 1.0)], n_devices=n_devices, allow_fewer=allow_fewer)
    (row,) = rp.recommend([doc])["rows"]
    assert row["deciding"] is deciding, row["excluded_because"]
    # --allow-fewer-devices never closes a checklist item.
    assert rp.closes_the_gap(copy.deepcopy(doc)) is (n_devices >= 4)
