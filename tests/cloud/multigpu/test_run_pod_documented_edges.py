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


def _goals_with_a_file(rp, directory) -> list:
    return [goal for goal in rp.ALL_GOALS if (directory / f"{goal}.json").is_file()]


def test_the_record_holds_a_file_for_every_goal_the_empty_record_rule_is_asked_of(rp, tmp_path):
    directory = _copy_record(tmp_path)
    assert set(_goals_with_a_file(rp, directory)) == set(rp.GOAL_CHECKS) == set(rp.ALL_GOALS)


@pytest.mark.parametrize("emptied", ["cells and checks", "checks", "cells"])
def test_a_goal_file_with_no_cells_or_no_checks_reads_invalid_for_every_goal(
        rp, tmp_path, capsys, emptied):
    """RPD-009 and RPD-014, one rule for every goal: a goal that did not
    raise ran at least one size and recorded at least one check ("no checks
    is not a pass").  A ``forward.json``, ``gradient.json`` or
    ``exchange.json`` with ``cells``, ``results`` and ``checks`` emptied read
    ``no checks`` and the summary exited 0, where the same edit to
    ``stencil``, ``hybrid`` or ``coupled`` read INVALID (exit 3) -- by the
    accident that their case lists take ``min()`` of the cells."""
    reference = _copy_record(tmp_path / "reference")
    assert rp.summarise(reference) in (0, 4)       # the record itself: nothing failed
    capsys.readouterr()
    goals = _goals_with_a_file(rp, reference)
    assert len(goals) >= 8
    for goal in goals:
        directory = _copy_record(tmp_path / f"{goal}-{emptied.replace(' ', '-')}")
        path = directory / f"{goal}.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["checks"] and doc["config"]["cells"] and "raised" not in doc, goal
        if "cells" in emptied:
            doc["config"]["cells"] = []
            doc["results"] = []
        if "checks" in emptied:
            doc["checks"] = []
        doc["passed"] = False
        path.write_text(json.dumps(doc), encoding="utf-8")
        problems = rp.record_problems({**doc, "_file": path.name})
        assert problems, (goal, emptied)
        if "cells" in emptied:
            assert problems[-1].startswith("its config names no cells"), (goal, problems)
        else:
            assert problems[-1].startswith("records no checks"), (goal, problems)
        assert rp.goal_verdict([doc]) in ("INVALID", "FAIL"), (goal, emptied)
        assert rp.summarise(directory) == 3, (goal, emptied)
        out = capsys.readouterr().out
        row = next(line for line in out.splitlines() if line.split()[:1] == [goal])
        assert "no checks" not in row and ("INVALID" in row or "FAIL" in row), (goal, row)


def test_a_goal_that_raised_still_records_its_one_check_and_no_cells_are_asked_of_it(rp):
    """The rule's other side: a goal that raised (``--keep-going``) holds no
    results and exactly the check its exception gives, and is not held to
    "at least one size"."""
    raised = {"type": "RuntimeError", "message": "boom"}
    doc = json.loads((_RECORD / "forward.json").read_text(encoding="utf-8"))
    doc.update(results=[], checks=[rp.goal_raised_check(raised)], raised=raised, passed=False)
    assert rp.record_problems(doc) == []
    doc["checks"] = []
    assert rp.record_problems(doc)


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


# ---------------------------------------------------------------------------
# Every size the session asks for can decide the transport
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("requested", [1, 2, 4, 5, 25, 26, 99_855, 99_856, 99_857, 100_000,
                                       100_489, 300_000, 1_000_000, 1_000_001])
def test_a_synthetic_mesh_never_has_fewer_cells_than_requested(rp, requested):
    """A grid is the smallest square holding the requested count (at least
    2 x 2), and what ``synthetic_cells`` predicts is what ``grid_edges``
    builds.  It used to round ``sqrt(n)``, and the "1e5-cell" row measured
    316**2 = 99 856 cells."""
    side = rp._grid_side(requested)
    assert side * side >= requested and side >= 2
    assert side == 2 or (side - 1) ** 2 < requested
    assert rp.synthetic_cells("grid", requested) == side * side
    assert rp.synthetic_cells("ring", requested) == requested
    if requested <= 100_489:
        n, edges = rp.grid_edges(requested)
        assert n == side * side and edges.max() == n - 1


def _relabelled_real_exchange(rp, rows, *, synthetic="grid", n_devices=4):
    """An ``exchange.json`` of a real 4-GPU run at ``rows`` = ``[(requested,
    a2a_ms, ppermute_ms)]``, measured at the size the runner builds for each
    request, with the checks the runner derives (so it reads ``PASS``)."""
    results = []
    for requested, a2a, ppm in rows:
        results.append({
            "cells": rp.synthetic_cells(synthetic, requested), "requested_cells": requested,
            "n_devices": n_devices, "bit_identical": True, "input_presharded": True,
            "ppermute_speedup_median": a2a / ppm,
            "methods": {"all_to_all": {"median_ms": a2a, "min_ms": a2a, "bytes_total": 1},
                        "ppermute": {"median_ms": ppm, "min_ms": ppm, "bytes_total": 1}}})
    doc = {"schema_version": rp.SCHEMA_VERSION, "goal": "exchange", "dry_run": False,
           "allow_fewer_devices": False, "n_devices": n_devices, "results": results,
           "environment": {"platform": "gpu", "device_kinds": ["gpu"],
                           "devices": [f"gpu:{i}" for i in range(n_devices)],
                           "n_devices_visible": n_devices, "git_commit": "0" * 40},
           "config": {"cells": [r for r, _, _ in rows], "synthetic": synthetic, "mesh": None,
                      "n_devices": n_devices, "allow_fewer_devices": False}}
    doc = rp.finish_checks(doc, rp.exchange_checks(results, n_devices))
    assert rp.record_problems(doc) == [] and rp.goal_verdict([doc]) == "PASS"
    return doc


@pytest.mark.parametrize("synthetic", ["grid", "ring"])
def test_every_default_gpu_size_decides_and_all_to_all_faster_at_1e5_keeps_the_default(
        rp, synthetic):
    """RPD-006/022/023: the session measures ``GPU_CELLS`` with the default
    ``--synthetic grid``.  Each of the three rows decides, so all_to_all
    faster at the 1e5 point keeps it the default -- the documented rule.
    Under the rounded grid the 1e5 row was excluded and the decision was
    ``ppermute`` on two of three points."""
    rows = [(n, 1.0, 1.0 / s) for n, s in zip(rp.GPU_CELLS, (0.90, 1.5, 1.5))]
    rec = rp.recommend([_relabelled_real_exchange(rp, rows, synthetic=synthetic)])
    assert [r["deciding"] for r in rec["rows"]] == [True, True, True], rec["rows"]
    assert rec["decision"] == "all_to_all" and rec["deciding_rows"] == 3, rec["reason"]
    assert [r["requested_cells"] for r in rec["rows"]] == list(rp.GPU_CELLS)
    assert all(r["cells"] >= r["requested_cells"] for r in rec["rows"])


def test_a_row_decides_on_its_requested_size_measured_at_no_fewer_cells(rp):
    """A row requested below the threshold never decides, whatever it
    measured; one measured at fewer cells than requested never decides
    either (and reads INVALID, its case not being the runner's)."""
    doc = _relabelled_real_exchange(rp, [(99_999, 2.0, 1.0)], synthetic="grid")
    assert doc["results"][0]["cells"] == 100_489             # 317**2 >= 1e5 ...
    (row,) = rp.recommend([doc])["rows"]
    assert row["deciding"] is False                          # ... but 99 999 were asked for
    assert row["excluded_because"] == "99999 cells requested < 100000"
    short = _relabelled_real_exchange(rp, [(100_000, 2.0, 1.0)], synthetic="grid")
    short["results"][0]["cells"] = 99_856                    # the old rounded grid
    assert any("lacks case(s)" in p for p in rp.record_problems(short))
    row = rp._excluded_because({**rp.recommend([short])["rows"][0], "record": "PASS"},
                               min_cells=100_000, min_devices=4)
    assert row == "measured 99856 cells, fewer than the 100000 requested"
    older = _relabelled_real_exchange(rp, [(100_000, 2.0, 1.0)])
    del older["results"][0]["requested_cells"]
    assert any("cannot be read" in p for p in rp.record_problems(older))
    (row,) = rp.recommend([older])["rows"]
    assert row["deciding"] is False


def test_the_summary_lists_every_row_that_does_not_decide(rp, tmp_path, capsys):
    """A decision on fewer rows than were measured says so: each excluded
    row is printed with its reason (``WARNING`` for a real-GPU one), and
    the recommendation line counts them."""
    doc = _relabelled_real_exchange(rp, [(50_000, 1.0, 0.5), (100_000, 1.0, 0.5),
                                         (1_000_000, 1.0, 0.5)])
    (tmp_path / "exchange.json").write_text(json.dumps(doc), encoding="utf-8")
    rp.summarise(tmp_path)
    out = capsys.readouterr().out
    assert "Rows that do not decide" in out, out
    assert ("WARNING: 50176 cells (requested 50000) on 4 GPU device(s): "
            "50000 cells requested < 100000") in out, out
    rec_line = next(ln for ln in out.splitlines() if ln.startswith("Recommendation:"))
    assert rec_line.startswith("Recommendation: ppermute") and "decided on 2 of 3 row(s)" in rec_line
    assert "1 excluded" in rec_line
    every = _relabelled_real_exchange(rp, [(100_000, 1.0, 0.5), (1_000_000, 1.0, 0.5)])
    (tmp_path / "exchange.json").write_text(json.dumps(every), encoding="utf-8")
    rp.summarise(tmp_path)
    out = capsys.readouterr().out
    rec_line = next(ln for ln in out.splitlines() if ln.startswith("Recommendation:"))
    assert "Rows that do not decide" not in out and "excluded" not in rec_line, out


@pytest.mark.parametrize("recorded, a2a, ppm, deciding, decision", [
    (3.0, 1.0, 2.0, True, "all_to_all"),          # the medians say ppermute is slower
    (float("inf"), 1.0, 0.5, True, "ppermute"),   # re-derived: 2.0
    (float("nan"), 1.0, 0.5, True, "ppermute"),
    (2.0, float("nan"), 0.5, False, "undecided"),
    (2.0, 1.0, float("inf"), False, "undecided"),
    (2.0, float("inf"), 1.0, False, "undecided"),
    (2.0, 1.0, -1.0, False, "undecided"),
    (2.0, 1e308, 1e-308, False, "undecided"),     # the ratio overflows
])
def test_the_speedup_is_rederived_from_the_medians_and_must_be_finite(
        rp, recorded, a2a, ppm, deciding, decision):
    """RPD-022's "with a finite speedup": ``recommend`` used to read the
    file's ``ppermute_speedup_median``, so an infinite one decided
    ``ppermute``, a NaN one read as a tie (``NaN >= 1.05`` and ``NaN < 1.0``
    are both false) and one its own medians contradict was believed."""
    doc = _relabelled_real_exchange(rp, [(100_000, 1.0, 0.5)])
    r = doc["results"][0]
    r["ppermute_speedup_median"] = recorded
    r["methods"]["all_to_all"]["median_ms"] = a2a
    r["methods"]["ppermute"]["median_ms"] = ppm
    rec = rp.recommend([doc])
    (row,) = rec["rows"]
    assert row["deciding"] is deciding, row["excluded_because"]
    assert rec["decision"] == decision, rec["reason"]
    if not deciding:
        assert "speedup undefined" in row["excluded_because"]


@pytest.mark.parametrize("content", [{}, {"results": []}, "stencil"],
                         ids=["empty-object", "no-goal", "another-goal"])
def test_a_goal_named_file_that_records_another_goal_reads_invalid(rp, tmp_path, capsys,
                                                                  content):
    """RPD-009's neighbour: a ``halo_rerun.json`` beside ``halo.json`` that
    holds ``{}``, an object with no ``goal``, or another goal's record is
    not evidence of anything: it reads INVALID, named, and the summary
    exits 3.  It used to be dropped in silence (exit 0)."""
    directory = _copy_record(tmp_path)
    if content == "stencil":
        content = json.loads((directory / "stencil.json").read_text(encoding="utf-8"))
    (directory / "halo_rerun.json").write_text(json.dumps(content), encoding="utf-8")
    assert rp.summarise(directory) == 3
    out = capsys.readouterr().out
    assert "halo_rerun.json" in out and "its name says goal 'halo'" in out, out
    assert rp.summarise(_copy_record(tmp_path / "control")) == 0


def _options(rp) -> list[str]:
    return [o for a in rp._parser()._actions for o in a.option_strings if o.startswith("--")]


def test_the_cli_takes_no_abbreviations(rp):
    """RPD-002: ``_pre_import_setup`` pins the CPU backend for the literal
    ``--dry-run`` only, so the parser must not accept anything else for
    it.  argparse took ``--dry`` (and every other unambiguous prefix) as
    ``--dry-run``, and such a "dry run" ran on the GPUs.  No option takes
    an abbreviation now."""
    options = _options(rp)
    assert "--dry-run" in options and "--summarise" in options
    for option in options:
        for cut in range(3, len(option)):
            prefix = option[:cut]
            if prefix in options:
                continue
            with pytest.raises(SystemExit) as raised:
                rp.parse_args([prefix, "--goal", "halo", "--out", "x"]
                              if option != "--goal" else [prefix, "halo", "--out", "x"])
            assert raised.value.code == 2, prefix
    args = rp.parse_args(["--dry-run", "--goal", "halo", "--out", "x"])
    assert args.dry_run is True


def test_a_dry_run_is_pinned_whenever_the_parser_reads_one(rp, monkeypatch):
    """The two readers of ``--dry-run`` agree on every spelling the parser
    accepts: whenever ``parse_args`` gives ``dry_run``, ``_pre_import_setup``
    has pinned the CPU backend."""
    import os

    for argv in (["--dry-run"], ["--goal", "halo", "--dry-run"], ["--dry"], ["--dry-ru"],
                 ["--dry-run", "--cells", "16"]):
        for key in ("JAX_PLATFORMS", "XLA_FLAGS"):
            monkeypatch.delenv(key, raising=False)
        rp._pre_import_setup(argv)
        pinned = os.environ.get("JAX_PLATFORMS") == "cpu"
        try:
            parsed = rp.parse_args(argv + ["--goal", "halo", "--out", "x"]).dry_run
        except SystemExit:
            parsed = False
        assert parsed <= pinned, (argv, parsed, pinned)


def test_an_abbreviated_dry_run_exits_2_before_running_anything(tmp_path):
    """The CLI itself, in a subprocess that never imports JAX: ``--dry``
    is refused with exit 2 and nothing is written."""
    import os
    import sys

    env = {k: v for k, v in os.environ.items() if k not in ("XLA_FLAGS",)}
    env.update(JAX_PLATFORMS="cpu", HOME=str(tmp_path))
    proc = subprocess.run([sys.executable, str(_RUNNER), "--dry", "--goal", "halo", "--out",
                           str(tmp_path / "out")], capture_output=True, text=True, timeout=120,
                          env=env, check=False)
    assert proc.returncode == 2, proc.stderr
    assert "unrecognized arguments: --dry" in proc.stderr
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# Guards an audit's mutants passed through
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [float("-inf"), float("inf"), float("nan")])
def test_a_non_finite_value_fails_a_less_or_equal_check(rp, value):
    """``-inf <= limit`` is true, so a ``<=`` check that asked only "not
    NaN" passed a measurement of ``-inf``; the rule is a finite number no
    greater than the limit, on the recording side and in the summary."""
    assert rp.check("x", value, 1e-5)["passed"] is False
    assert rp.expected_pass({"value": value, "limit": 1e-5, "sense": "<="}) is False
    assert rp.check_status({"value": value, "limit": 1e-5, "sense": "<=",
                            "passed": True}) == "inconsistent"


@pytest.mark.parametrize("sha, kept", [
    ("0" * 40, True), ("ab" * 32, True), ("0" * 39, False), ("0" * 41, False),
    ("0" * 12, False), ("0" * 7, False), ("0" * 63, False), ("0" * 65, False),
    ("A" * 40, False), ("g" * 40, False), (" " + "0" * 40, False),
])
def test_only_a_full_sha_counts_as_a_recorded_commit(rp, sha, kept):
    """RPD-011: a recorded commit is a full SHA-1 (40) or SHA-256 (64) in
    lowercase hex; an abbreviation names no one commit and counts as none."""
    assert (rp._commit_of({"environment": {"git_commit": sha}}) == sha) is kept


def test_a_row_measured_one_cell_short_of_its_request_never_decides(rp):
    doc = _relabelled_real_exchange(rp, [(100_000, 2.0, 1.0)], synthetic="grid")
    (row,) = rp.recommend([doc])["rows"]
    for cells, excluded in ((99_999, True), (100_000, False)):
        why = rp._excluded_because({**row, "cells": cells, "record": "PASS"},
                                   min_cells=100_000, min_devices=4)
        assert (why == f"measured {cells} cells, fewer than the 100000 requested") is excluded
