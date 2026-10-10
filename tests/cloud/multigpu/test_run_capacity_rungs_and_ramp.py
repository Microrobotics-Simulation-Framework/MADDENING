"""``run_capacity.py``: a rung, the ramp that sizes and classifies rungs, the
soak, the summary -- on CPU virtual devices, as dry runs.

The capacity test runs for the first time for real on the pod, so every
path it takes there is taken here first: the rungs run in this process
through the ramp's own launcher hook (``run_capacity_support.in_process``),
with the device memory statistics -- which CPU does not have -- stubbed to
numbers where the fill, the next rung's size or the soak's growth is the
subject.  A wrong halo and a wrong reference must fail a rung; an
out-of-memory, a kill and a time-out above a passing rung are results.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from tests.cloud.multigpu import run_capacity_support as S
from tests.cloud.multigpu.run_pod_support import run_pod

rc = S.module()
_RECORDED_GOALS = Path(__file__).resolve().parent / "run_pod_record"
#: 2**-11 GiB, half a mebibyte: a "card" a few tiles fill.
_CARD_GIB = 2.0 ** -11
_CARD = int(_CARD_GIB * rc.GIB)


@pytest.fixture(scope="module", autouse=True)
def _needs_four_devices():
    import jax

    if len(jax.devices()) < 4:
        pytest.skip("needs >= 4 devices")
    rc._load_backend()      # a test may read rc.LBMNode before any rung has run


def _ramp(tmp_path: Path, *argv, launch=S.in_process, out: str = "out"):
    """Run the ramp here: ``(exit status, out directory, summary, stdout)``."""
    directory = tmp_path / out
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        status = rc.exit_status(["--dry-run", "--out", str(directory), *S.TILE_ARGS, *argv],
                                launch)
    summary_path = directory / rc.SUMMARY_FILE
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    return status, directory, summary, text.getvalue()


def _rung(tmp_path: Path, mesh: str, k, *, steps: int = 3, name: str = "rung.json"):
    """One rung here, as the ramp starts it: ``(exit status, record)``."""
    record = tmp_path / name
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        status = rc.exit_status(["--child", "rung", "--record", str(record), "--dry-run",
                                 "--mesh", mesh, *S.TILE_ARGS, "--k", *map(str, k),
                                 "--steps", str(steps), "--rung-index", "1"])
    return status, json.loads(record.read_text())


def _summarise(directory: Path):
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        status = rc.exit_status(["--summarise", str(directory)])
    return status, text.getvalue()


def _failed(record: dict) -> list:
    return [c["name"] for c in record["checks"] if not c.get("not_run") and not c["passed"]]


# --- a ramp of explicit sizes, as the dry run of section 0 runs it -------------


@pytest.mark.parametrize("mesh", ["2x2", "4x1"])
def test_a_dry_run_ramp_over_three_sizes_passes_and_writes_what_the_summary_reads(tmp_path,
                                                                                mesh):
    status, out, summary, text = _ramp(tmp_path, "--mesh", mesh, "--cells", "200", "600", "1800")
    assert status == rc.EXIT_OK == 0, text
    assert [r["outcome"] for r in summary["rungs"]] == ["passed"] * 3
    cells = [r["cells"] for r in summary["rungs"]]
    assert cells == sorted(set(cells)) and cells[0] == 192 and cells[-1] <= 1800
    assert summary["ceiling"]["cells"] == cells[-1] and summary["stopped_by"] is None
    assert summary["gating"] is False and summary["kind"] == rc.SUMMARY_KIND
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted([rc.SUMMARY_FILE] + [f"capacity_rung_{i:02d}.{ext}"
                                                for i in (1, 2, 3)
                                                for ext in ("json", "stderr.log")])
    for i, n in enumerate(cells, start=1):
        record = json.loads((out / rc.rung_file(i)).read_text())
        assert record["kind"] == rc.RUNG_KIND and "goal" not in record
        assert record["schema_version"] == rc.CAPACITY_SCHEMA_VERSION
        assert record["cells"] == n == math.prod(record["shape"]) and record["dry_run"] is True
        assert record["mesh"]["label"] == mesh and record["lattice"] == "D3Q19"
        assert record["environment"]["platform"] == "cpu"
        assert set(record["environment"]["allocator"]) == {"found", "used"}
        assert [r["at"] for r in record["memory"]["readings"]] == list(rc.READINGS)
        # CPU keeps no memory statistics: the fill is not measured, which
        # is a check NOT RUN -- never a pass.
        assert record["memory"]["fill"] is None
        not_run = [c for c in record["checks"] if c.get("not_run")]
        assert len(not_run) == 1 and "fill" in not_run[0]["name"]
        assert not_run[0]["passed"] is False and "not measured" in not_run[0]["detail"]
        ran = [c for c in record["checks"] if not c.get("not_run")]
        assert len(ran) == 16 and all(c["passed"] for c in ran)
        assert {c["sense"] for c in ran} == {"<=", "=="}
        assert all(c["limit"] == rc.LIMIT for c in ran if c["sense"] == "<=")
        for field in rc.STATE_FIELDS:
            assert record["results"]["fields"][field]["cells_compared"] == n
        assert record["steps_per_second"] > 0 and record["seconds"]["total"] > 0
        assert record["builder_requests"] and all(
            r["cells"] == n // 4 for r in record["builder_requests"])
    assert "CEILING: rung 3" in text and "fill not measured" in text
    status, table = _summarise(out)
    assert status == rc.EXIT_OK
    assert "CEILING: rung 3" in table and "STOPPED BY: nothing" in table
    assert table.count(" passed ") == 3 and "n/m" in table


# Per push: tests/cloud/multigpu/test_run_capacity_rungs_and_ramp.py::test_one_rung_runs_in_a_process_of_its_own_as_the_session_starts_it
@pytest.mark.slow
@pytest.mark.parametrize("mesh", ["2x2", "4x1"])
def test_the_dry_run_ramp_of_the_runbook_passes_in_processes_of_its_own(tmp_path, mesh):
    """Section 0's command, on the default tile: three rungs and a soak,
    each a process of its own; then both summaries on its directory."""
    out = tmp_path / "out"
    done = S.run_capacity(["--dry-run", "--mesh", mesh, "--cells", "8000", "30000", "100000",
                           "--soak-minutes", "0.02", "--out", out], timeout=900)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    summary = json.loads((out / rc.SUMMARY_FILE).read_text())
    assert [r["outcome"] for r in summary["rungs"]] == ["passed"] * 3
    assert summary["soak"]["outcome"] == "passed" and summary["tile"] == list(rc.DEFAULT_TILE)
    for index in (1, 2, 3):
        record = json.loads((out / rc.rung_file(index)).read_text())
        assert record["environment"]["n_devices_visible"] == 4
        assert max(e["max_rel"] for e in record["results"]["fields"].values()) <= rc.LIMIT
    again = S.run_capacity(["--summarise", out], timeout=300)
    assert again.returncode == 0 and "CEILING: rung 3" in again.stdout
    runner = run_pod(S.RUNNER, ["--summarise", out], pythonpath=str(S.REPO / "src"), timeout=300)
    assert runner.returncode == 1 and "no goal JSON" in runner.stdout


def test_one_rung_runs_in_a_process_of_its_own_as_the_session_starts_it(tmp_path):
    """The script as a program: the CPU pin of ``--dry-run`` before JAX is
    imported, the rung started with ``sys.executable``, its stderr kept."""
    out = tmp_path / "out"
    done = S.run_capacity(["--dry-run", "--mesh", "2x2", *S.TILE_ARGS, "--k", "3", "3", "1",
                           "--out", out], timeout=300)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    record = json.loads((out / rc.rung_file(1)).read_text())
    env = record["environment"]
    assert env["platform"] == "cpu" and env["n_devices_visible"] == 4
    assert env["jax_platforms"] == "cpu"
    assert "--xla_force_host_platform_device_count=4" in env["xla_flags"]
    # Found unset; used with the script's defaults, set before JAX loaded.
    assert env["allocator"]["found"] == dict.fromkeys(rc.ALLOCATOR_VARIABLES)
    assert env["allocator"]["used"] == {**dict.fromkeys(rc.ALLOCATOR_VARIABLES),
                                        **rc.ALLOCATOR_DEFAULTS}
    assert record["outcome"] == "passed" and record["k"] == [3, 3, 1]
    assert record["launch"]["returncode"] == 0 and record["launch"]["timed_out"] is False
    assert record["host"]["vm_hwm_bytes"] > 0 and record["host"]["refused"] is False
    assert (out / "capacity_rung_01.stderr.log").exists()
    assert "CEILING: rung 1, 1728 cells" in done.stdout


def test_a_rung_past_its_time_box_is_killed_and_recorded_as_timed_out(tmp_path):
    """The ramp's own launcher, a real process, a time box it cannot meet."""
    out = tmp_path / "out"
    done = S.run_capacity(["--dry-run", *S.TILE_ARGS, "--k", "3", "3", "1",
                           "--rung-timeout-s", "0.05", "--out", out], timeout=300)
    assert done.returncode == rc.EXIT_NO_CEILING == 6, done.stdout[-2000:] + done.stderr[-2000:]
    record = json.loads((out / rc.rung_file(1)).read_text())
    assert record["outcome"] == "timed out" and record["launch"]["timed_out"] is True
    assert record["launch"]["returncode"] in (-9, 137)
    assert "CEILING: none" in done.stdout and "timed out" in done.stdout


def test_a_ramp_that_is_terminated_takes_its_rung_with_it(tmp_path):
    """``timeout`` around the ramp ends it with SIGTERM; the rung in flight
    is this script's own process and must not be left holding the cards."""
    import time

    out, pid = tmp_path / "out", None
    ramp = S.start_capacity(["--dry-run", *S.TILE_ARGS, "--k", "3", "3", "1", "--steps", "20000",
                             "--out", out])
    try:
        record = out / rc.rung_file(1)
        deadline = time.monotonic() + 120
        while pid is None and time.monotonic() < deadline and ramp.poll() is None:
            with contextlib.suppress(OSError, ValueError):
                pid = json.loads(record.read_text())["pid"]
            time.sleep(0.01)
        assert pid is not None and pid != ramp.pid, "the rung never wrote its record"
        assert Path(f"/proc/{pid}").exists()
        ramp.terminate()
        assert ramp.wait(timeout=60) == 128 + 15
        deadline = time.monotonic() + 30
        while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not Path(f"/proc/{pid}").exists(), "the rung outlived the ramp"
    finally:
        if ramp.poll() is None:
            ramp.kill()
            ramp.wait()
        if pid is not None and Path(f"/proc/{pid}").exists():
            import os
            import signal

            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)        # the rung this test's ramp started


# --- a wrong halo and a wrong reference must fail a rung ----------------------


def _from_the_wrong_neighbour(calls: list):
    """``halo_exchange`` with each halo taken from the neighbour on the
    other side: the right edge of shard ``d + 1`` where shard ``d - 1``'s
    belongs.  On a two-device axis both are the same device, so this is a
    fault on an axis of three devices or more."""
    from jax import lax
    import jax.numpy as jnp

    def exchange(local, *, mesh, axes, boundary):
        calls.append(tuple(axes))
        out = local
        for mesh_axis, spatial_axis, halo in axes:
            p, n = int(mesh.shape[mesh_axis]), out.shape[spatial_axis]
            left = lax.slice_in_dim(out, 0, halo, axis=spatial_axis)
            right = lax.slice_in_dim(out, n - halo, n, axis=spatial_axis)
            # SEEDED FAULT: the two permutations exchanged
            left_halo = lax.ppermute(right, mesh_axis, [(s, (s - 1) % p) for s in range(p)])
            right_halo = lax.ppermute(left, mesh_axis, [(s, (s + 1) % p) for s in range(p)])
            out = jnp.concatenate([left_halo, out, right_halo], axis=spatial_axis)
        return out

    return exchange


def _from_the_shard_itself(calls: list, only_axis: int | None = None):
    """``halo_exchange`` with a shard's halos taken from its own opposite
    edges -- periodic inside the shard, nothing exchanged -- along
    ``only_axis`` (every exchanged axis when ``None``)."""
    import jax.numpy as jnp
    from jax import lax

    from maddening.cloud.multigpu import halo as real

    def exchange(local, *, mesh, axes, boundary):
        calls.append(tuple(axes))
        out = local
        for mesh_axis, spatial_axis, halo in axes:
            if only_axis is not None and spatial_axis != only_axis:
                out = real.halo_exchange(out, mesh=mesh, axes=[(mesh_axis, spatial_axis, halo)],
                                         boundary=boundary)
                continue
            n = out.shape[spatial_axis]
            left = lax.slice_in_dim(out, 0, halo, axis=spatial_axis)
            right = lax.slice_in_dim(out, n - halo, n, axis=spatial_axis)
            out = jnp.concatenate([right, out, left], axis=spatial_axis)   # SEEDED FAULT
        return out

    return exchange


_FAULTS = {
    "wrong neighbour, 4x1": ("4x1", (3, 3, 1), lambda calls: _from_the_wrong_neighbour(calls)),
    "own edge, 4x1": ("4x1", (3, 3, 1), lambda calls: _from_the_shard_itself(calls)),
    "own edge along axis 0, 2x2": ("2x2", (3, 3, 1),
                                   lambda calls: _from_the_shard_itself(calls, 0)),
    "own edge along axis 1, 2x2": ("2x2", (3, 3, 1),
                                   lambda calls: _from_the_shard_itself(calls, 1)),
}


@pytest.mark.parametrize("fault", sorted(_FAULTS))
def test_a_wrong_halo_fails_the_tiled_comparison_with_the_check_failed_status(
        fault, tmp_path, monkeypatch):
    """The wrapper reads ``halo_exchange`` from its own module: patched
    there, the exchange delivers a wrong halo, the rung's populations are
    off the tiled reference by far more than the limit, and its status is
    the one that means a wrong number."""
    from maddening.cloud.multigpu import sharded_node

    mesh, k, make = _FAULTS[fault]
    calls: list = []
    monkeypatch.setattr(sharded_node, "halo_exchange", make(calls))
    status, record = _rung(tmp_path, mesh, k)
    assert calls, "the patched exchange was never traced: the fault was not seeded"
    assert status == rc.EXIT_CHECK_FAILED == 1
    assert record["outcome"] == "check failed"
    failed = _failed(record)
    assert "f vs the tiled reference max_rel" in failed, failed
    assert record["results"]["fields"]["f"]["max_rel"] > 100 * rc.LIMIT
    # The wall mask is zero everywhere: no halo can make it wrong.
    assert "wall_mask vs the tiled reference max_rel" not in failed
    assert rc.rederive(record) == {"outcome": "check failed", "failed": failed, "problems": [],
                                   "fill": None}


def test_the_same_rungs_pass_with_the_librarys_own_exchange(tmp_path):
    """The controls of the faults above, and of the one below."""
    for mesh in ("4x1", "2x2"):
        status, record = _rung(tmp_path, mesh, (3, 3, 1), name=f"{mesh}.json")
        assert status == rc.EXIT_OK and record["outcome"] == "passed", _failed(record)
        assert max(e["max_rel"] for e in record["results"]["fields"].values()) <= rc.LIMIT


def test_a_rung_never_has_the_lattice_build_its_whole_state_on_one_device(tmp_path, monkeypatch):
    """``ShardedStencilNode``'s constructor asks the node it wraps for its
    initial state (to read the shapes) and for its field names (whose
    default builds the initial state again): the whole grid, twice, on the
    default device.  The script's lattice answers both without building
    anything, so a rung never runs ``LBMNode.initial_state`` -- while the
    library's own node, wrapped the same way, runs it twice."""
    built: list = []
    real = rc.LBMNode.initial_state

    def counting(self):
        built.append(self._grid_shape)                               # noqa: SLF001
        return real(self)

    monkeypatch.setattr(rc.LBMNode, "initial_state", counting)
    status, record = _rung(tmp_path, "2x2", (3, 3, 1))
    assert status == rc.EXIT_OK and record["outcome"] == "passed"
    assert built == []
    assert record["on_one_device"][0]["bytes"] == record["cells"]    # the mask it does keep

    mesh, axis_map = rc.build_mesh((2, 2))
    node = rc.LBMNode("plain", 1.0, grid_shape=(24, 12, 6), lattice=rc.LATTICE)
    rc.ShardedStencilNode(node, mesh, axis_map=axis_map, boundary="periodic")
    assert built == [(24, 12, 6)] * 2
    state = real(node)
    assert all(len(a.devices()) == 1 for a in state.values())        # one device holds it all
    with pytest.raises(RuntimeError, match="built block by block"):
        rc.Lattice("ours", 1.0, grid_shape=(24, 12, 6), lattice=rc.LATTICE).initial_state()


def test_a_wrong_reference_tile_fails_the_comparison(tmp_path, monkeypatch):
    """The reference shifted by one cell along axis 0 -- still a periodic
    run of the tile, of another phase."""
    real = rc.tile_reference
    seen: list = []

    def shifted(tile, tile_state, steps):
        seen.append(steps)
        return {name: np.roll(value, 1, axis=0) for name, value in
                real(tile, tile_state, steps).items()}

    monkeypatch.setattr(rc, "tile_reference", shifted)
    status, record = _rung(tmp_path, "2x2", (3, 3, 1))
    assert seen == [3]
    assert status == rc.EXIT_CHECK_FAILED and record["outcome"] == "check failed"
    failed = _failed(record)
    assert {"f vs the tiled reference max_rel", "velocity vs the tiled reference max_rel",
            "density vs the tiled reference max_rel"} <= set(failed), failed
    # A shift moves no mass: that check, and the finiteness, still pass.
    assert "total mass vs prod(K) times the tile's rel" not in failed


def test_without_the_oddness_rule_a_halo_from_the_wrong_neighbour_passes(tmp_path, monkeypatch):
    """Why the rule exists.  With two tiles along an axis of four devices
    shards 0 and 2 hold the same rows of the tile, and so do 1 and 3: the
    right edge of the neighbour on the other side is the row that belongs,
    and a rung run on that tiling passes with the exchange broken.  The
    script refuses the tiling; with three tiles the same fault fails."""
    from maddening.cloud.multigpu import sharded_node

    assert "must be odd" in rc.tiling_problem(S.TILE, (2, 3, 1), (4, 1, 1))
    calls: list = []
    monkeypatch.setattr(sharded_node, "halo_exchange", _from_the_wrong_neighbour(calls))
    with contextlib.redirect_stderr(io.StringIO()):
        refused = rc.exit_status(["--child", "rung", "--record", str(tmp_path / "r.json"),
                                  "--dry-run", "--mesh", "4x1", *S.TILE_ARGS,
                                  "--k", "2", "3", "1"])
    assert refused == rc.EXIT_REFUSED and not (tmp_path / "r.json").exists()
    real = rc.tiling_problem
    monkeypatch.setattr(rc, "tiling_problem", lambda tile, k, per_axis: (
        None if tuple(k) == (2, 3, 1) else real(tile, k, per_axis)))
    status, record = _rung(tmp_path, "4x1", (2, 3, 1), name="even.json")
    assert calls
    assert status == rc.EXIT_OK and record["outcome"] == "passed", _failed(record)
    del calls[:]
    status, record = _rung(tmp_path, "4x1", (3, 3, 1), name="odd.json")
    assert calls and status == rc.EXIT_CHECK_FAILED


# --- the fill, the next rung's size, the last target ---------------------------


def test_the_fill_is_the_largest_peak_over_the_card_not_the_bytes_in_use(tmp_path,
                                                                        monkeypatch):
    S.stub_memory(monkeypatch, bytes_per_cell=400.0, limit=_CARD * 3 // 4)
    record_path = tmp_path / "rung.json"
    with contextlib.redirect_stdout(io.StringIO()):
        status = rc.exit_status(["--child", "rung", "--record", str(record_path), "--dry-run",
                                 *S.TILE_ARGS, "--k", "3", "3", "1", "--device-memory-gb",
                                 repr(_CARD_GIB)])
    record = json.loads(record_path.read_text())
    assert status == rc.EXIT_OK and record["outcome"] == "passed"
    memory = record["memory"]
    cells = record["cells"]
    peak = int(cells / 4 * 400.0)                       # the fullest device, from step one on
    assert memory["peak_bytes"] == peak == 172_800
    assert memory["fill"] == pytest.approx(peak / _CARD) and memory["fill"] != 0.25 * peak / _CARD
    assert memory["peak_bytes_per_cell"] == pytest.approx(400.0)
    assert memory["bytes_limit"] == _CARD * 3 // 4
    by_device = memory["fill_by_device"]
    assert len(by_device) == 4 and max(by_device.values()) == memory["fill"]
    assert sorted(by_device.values())[0] == pytest.approx(0.9 * memory["fill"], rel=1e-3)
    assert not any(c.get("not_run") for c in record["checks"])
    measured = [c for c in record["checks"] if "fill" in c["name"]]
    assert len(measured) == 1 and measured[0]["passed"] is True
    # The helper itself: the peak of ANY reading, per device.
    readings = [{"at": "a", "devices": [{"device": "d0", "peak_bytes_in_use": 10,
                                         "bytes_in_use": 90},
                                        {"device": "d1", "peak_bytes_in_use": 40,
                                         "bytes_in_use": 1}]},
                {"at": "b", "devices": [{"device": "d0", "peak_bytes_in_use": 70,
                                         "bytes_in_use": 2},
                                        {"device": "d1", "peak_bytes_in_use": 40,
                                         "bytes_in_use": 3}]}]
    got = rc.fill_of(readings, 100)
    assert got["fill"] == 0.7 and got["by_device"] == {"d0": 0.7, "d1": 0.4}
    assert got["peak_bytes"] == 70 and got["bytes_limit"] is None
    assert rc.fill_of(readings, None)["fill"] is None
    assert "--device-memory-gb" in rc.fill_of(readings, None)["why_not"]
    readings[1]["devices"][1].pop("peak_bytes_in_use")
    assert rc.fill_of(readings, 100)["fill"] is None
    assert "d1 has no peak_bytes_in_use b" in rc.fill_of(readings, 100)["why_not"]
    assert rc.fill_of([], 100)["why_not"] == "no memory reading was taken"


def test_the_ramp_sizes_each_rung_from_what_the_last_measured_and_stops_at_its_last_target(
        tmp_path, monkeypatch):
    """Memory statistics as numbers: 400 bytes per cell on the fullest
    device, where the a-priori guess is 97 x 6 = 582."""
    S.stub_memory(monkeypatch, bytes_per_cell=400.0)
    status, out, summary, text = _ramp(tmp_path, "--ramp", "0.25", "0.9",
                                       "--device-memory-gb", repr(_CARD_GIB))
    assert status == rc.EXIT_OK, text
    rungs = summary["rungs"]
    assert [r["outcome"] for r in rungs] == ["passed"] * 2
    assert [r["target_fill"] for r in rungs] == [0.25, 0.9]
    # Rung 1: the a-priori bytes per cell.
    first = rc.first_rung_cells(0.25, _CARD, 4, rc.STATE_BYTES_PER_CELL * rc.APRIORI_STEP_MULTIPLE)
    assert first == int(0.25 * _CARD * 4 / 582.0)
    assert rungs[0]["k"] == list(rc.tiling_for_cells(first, S.TILE, (2, 2, 1)))
    assert "a priori: 582 bytes per cell = 97 of state x 6" in rungs[0]["sized_from"]
    for last, rung in zip(rungs, rungs[1:]):
        assert rung["fill"] == pytest.approx(rung["cells"] / 4 * 400.0 / _CARD)
        assert rung["peak_bytes_per_cell"] == pytest.approx(400.0)
        want = rc.next_rung_cells(rung["target_fill"], last["fill"], last["cells"])
        assert want == int(last["cells"] * rung["target_fill"] / last["fill"])
        assert rung["k"] == list(rc.tiling_for_cells(want, S.TILE, (2, 2, 1)))
        assert f"rung {last['rung']} measured 400.0 bytes per cell" in rung["sized_from"]
        assert last["cells"] < rung["cells"] <= want
        assert rung["fill"] <= rung["target_fill"]
    # The first rung undershoots its target because the guess was high;
    # the last lands under 0.9 and nothing is sized beyond it.
    assert rungs[0]["fill"] < 0.25 * 400.0 / 582.0 + 1e-9
    assert 0.5 < rungs[-1]["fill"] <= 0.9 and summary["ceiling"]["rung"] == 2
    assert summary["stopped_by"] is None and "STOPPED BY: nothing" in text
    status, table = _summarise(out)
    assert status == rc.EXIT_OK and f"{rungs[-1]['fill']:.3f}" in table


def test_a_target_the_last_rung_already_reached_is_skipped_and_one_above_the_allocators_limit_is_named(
        tmp_path, monkeypatch):
    S.stub_memory(monkeypatch, bytes_per_cell=400.0, limit=int(0.6 * _CARD))
    status, out, summary, text = _ramp(tmp_path, "--ramp", "0.25", "0.251", "0.9",
                                       "--device-memory-gb", repr(_CARD_GIB),
                                       "--bytes-per-cell", "400")
    assert status == rc.EXIT_OK, text
    assert [r["target_fill"] for r in summary["rungs"]] == [0.25, 0.9]
    assert "400 bytes per cell" in summary["rungs"][0]["sized_from"]
    assert "97 of state" not in summary["rungs"][0]["sized_from"]
    notes = "\n".join(summary["notes"])
    assert "target 0.251 skipped" in notes
    assert "target 0.9 is above the allocator's own limit, 0.600 of the card" in notes
    assert "the allocator's own limit there" in text


def test_a_ramp_whose_first_fill_is_not_measured_stops_and_says_to_give_sizes(tmp_path):
    """CPU, no stub: rung 1 runs from the a-priori guess and passes; the
    next cannot be sized."""
    status, out, summary, text = _ramp(tmp_path, "--ramp", "0.25", "0.5",
                                       "--device-memory-gb", repr(_CARD_GIB))
    assert status == rc.EXIT_OK
    assert len(summary["rungs"]) == 1 and summary["rungs"][0]["fill"] is None
    assert "fill was not measured" in summary["stopped_by"]["detail"]
    assert "--cells" in text and "STOPPED BY: rung 1's fill was not measured" in text


# --- how a rung can end, and what the ramp exits with --------------------------


@pytest.fixture(scope="module")
def passing(tmp_path_factory):
    """The records of two rungs that passed (192 and 576 cells on the
    pencil: the grids ``--cells 192 576`` takes), for a scripted launcher
    to replay."""
    tmp = tmp_path_factory.mktemp("passing")
    records = {}
    for k in ((1, 1, 1), (1, 3, 1)):
        assert rc.tiling_for_cells(math.prod(rc.grid_shape(S.TILE, k)), S.TILE, (2, 2, 1)) == k
        status, record = _rung(tmp, "2x2", k, name=f"{k[1]}.json")
        assert status == rc.EXIT_OK and record["outcome"] == "passed"
        records[k] = record
    return records


def _launched(returncode, *, timed_out=False, stderr=""):
    return rc.Launched(returncode, timed_out, stderr, 0.5)


_XLA_OOM = ("jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to "
            "allocate 21474836480 bytes.")
#: (how rung 2 ends, the outcome, a phrase of the detail)
_ABOVE_A_PASSING_RUNG = {
    "killed, status 137": (_launched(137), "out of memory", "killed (status 137)"),
    "killed, SIGKILL": (_launched(-9), "out of memory", "killed (status -9)"),
    "timed out": (_launched(-9, timed_out=True), "timed out", "killed at its time box"),
    "aborted on an allocation": (_launched(-6, stderr="F external/xla/...\n" + _XLA_OOM),
                                 "out of memory", "RESOURCE_EXHAUSTED"),
    "segfault": (_launched(-11, stderr="Fatal Python error: Segmentation fault"), "crashed",
                 "Segmentation fault"),
    "exit 0 with no record": (_launched(0), "crashed", "without a record whose checks pass"),
    "exit 1 with no record": (_launched(1, stderr="Traceback ..."), "crashed", "status 1"),
}


@pytest.mark.parametrize("how", sorted(_ABOVE_A_PASSING_RUNG))
def test_a_rung_that_dies_above_a_passing_one_is_the_ceilings_reason_and_exit_0(
        how, tmp_path, passing):
    end, outcome, words = _ABOVE_A_PASSING_RUNG[how]
    launch = S.scripted([passing[(1, 1, 1)], end])
    status, out, summary, text = _ramp(tmp_path, "--cells", "192", "576", "5000", launch=launch)
    assert len(launch.calls) == 2                       # the ramp stopped: no third rung
    assert status == rc.EXIT_OK == 0, text
    assert [r["outcome"] for r in summary["rungs"]] == ["passed", outcome]
    assert summary["ceiling"]["rung"] == 1 and summary["ceiling"]["cells"] == 192
    stopped = summary["stopped_by"]
    assert stopped["rung"] == 2 and stopped["outcome"] == outcome and stopped["cells"] == 576
    assert words in stopped["detail"]
    record = json.loads((out / rc.rung_file(2)).read_text())
    assert record["outcome"] == outcome and record["cells"] == 576 and record["k"] == [1, 3, 1]
    assert record["launch"]["returncode"] == end.returncode
    assert record["launch"]["stderr_tail"] == end.stderr_tail
    assert f"STOPPED BY: rung 2 (576 cells): {outcome}" in text
    status, table = _summarise(out)
    assert status == rc.EXIT_OK and f"STOPPED BY: rung 2 (576 cells): {outcome}" in table


def test_a_rung_whose_runtime_reports_resource_exhaustion_is_out_of_memory(tmp_path, passing,
                                                                          monkeypatch):
    """The rung's own process, here: its step raises the runtime's error,
    the rung records how far it had got and exits with its own status, and
    the ramp reads an out-of-memory above the passing rung."""
    import jax

    real = rc.run_pass

    def exhausted(problem, note):
        note(rc.READINGS[0])
        raise jax.errors.JaxRuntimeError(
            "RESOURCE_EXHAUSTED: Out of memory while trying to allocate 21474836480 bytes.")

    monkeypatch.setattr(rc, "run_pass", exhausted)
    launch = S.scripted([passing[(1, 1, 1)], "run"])
    status, out, summary, text = _ramp(tmp_path, "--cells", "192", "576", launch=launch)
    monkeypatch.setattr(rc, "run_pass", real)
    assert status == rc.EXIT_OK, text
    assert [r["outcome"] for r in summary["rungs"]] == ["passed", "out of memory"]
    record = json.loads((out / rc.rung_file(2)).read_text())
    assert record["launch"]["returncode"] == rc.EXIT_RUNG_OUT_OF_MEMORY
    assert record["raised"]["type"] == "JaxRuntimeError" and record["phase"] == "running"
    assert "RESOURCE_EXHAUSTED" in record["outcome_detail"]
    assert [r["at"] for r in record["memory"]["readings"]] == [rc.READINGS[0]]
    assert "RESOURCE_EXHAUSTED" in summary["stopped_by"]["detail"]
    assert rc.is_out_of_memory(MemoryError())
    assert rc.is_out_of_memory(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY"))
    assert not rc.is_out_of_memory(ValueError("shapes do not match"))


def test_a_rung_that_raises_anything_else_is_crashed_with_its_traceback_kept(tmp_path, passing,
                                                                           monkeypatch):
    def broken(problem, note):
        raise ValueError("shapes do not match")

    monkeypatch.setattr(rc, "run_pass", broken)
    launch = S.scripted([passing[(1, 1, 1)], "run"])
    status, out, summary, _ = _ramp(tmp_path, "--cells", "192", "576", launch=launch)
    assert status == rc.EXIT_OK
    assert summary["rungs"][1]["outcome"] == "crashed"
    record = json.loads((out / rc.rung_file(2)).read_text())
    assert record["launch"]["returncode"] == rc.EXIT_CRASHED
    assert "ValueError: shapes do not match" in record["launch"]["stderr_tail"]
    assert "Traceback" in (out / "capacity_rung_02.stderr.log").read_text()


@pytest.mark.parametrize("end, outcome", [
    (_launched(137), "out of memory"), (_launched(-9, timed_out=True), "timed out"),
    (_launched(-11, stderr="Segmentation fault"), "crashed")])
def test_a_first_rung_that_cannot_run_is_its_own_exit_status(end, outcome, tmp_path):
    launch = S.scripted([end])
    status, out, summary, text = _ramp(tmp_path, "--cells", "192", "576", launch=launch)
    assert status == rc.EXIT_NO_CEILING == 6
    assert summary["ceiling"] is None and summary["rungs"][0]["outcome"] == outcome
    assert "CEILING: none" in text
    assert _summarise(out)[0] == rc.EXIT_NO_CEILING


def test_a_failed_check_stops_the_ramp_with_the_check_failed_status(tmp_path, passing,
                                                                  monkeypatch):
    """A wrong number above a passing rung is not a ceiling: exit 1."""
    real = rc.tile_reference
    monkeypatch.setattr(rc, "tile_reference", lambda tile, tile_state, steps: {
        name: np.roll(value, 1, axis=1) for name, value in real(tile, tile_state, steps).items()})
    launch = S.scripted([passing[(1, 1, 1)], "run", "run"])
    status, out, summary, text = _ramp(tmp_path, "--cells", "192", "576", "5000", launch=launch)
    assert len(launch.calls) == 2
    assert status == rc.EXIT_CHECK_FAILED == 1
    assert [r["outcome"] for r in summary["rungs"]] == ["passed", "check failed"]
    assert summary["exit_status"] == 1 and "CHECK FAILED [capacity]" in text
    assert "f vs the tiled reference max_rel" in summary["stopped_by"]["detail"]
    assert _summarise(out)[0] == rc.EXIT_CHECK_FAILED


def test_the_exit_statuses_are_the_ones_the_header_documents():
    table = dict(re.findall(r"^    (\d)   (\S.*)$", rc.__doc__, flags=re.MULTILINE))
    assert sorted(map(int, table)) == sorted({rc.EXIT_OK, rc.EXIT_CHECK_FAILED, rc.EXIT_REFUSED,
                                              rc.EXIT_RECORD_INVALID, rc.EXIT_CRASHED,
                                              rc.EXIT_NO_CEILING}) == [0, 1, 2, 3, 5, 6]
    assert table["1"].startswith("a check failed") and table["6"].startswith("no ceiling")
    rows = lambda *outcomes: [{"outcome": o} for o in outcomes]      # noqa: E731
    assert rc.ramp_status(rows("passed", "out of memory"), None) == 0
    assert rc.ramp_status(rows("passed", "passed"), {"outcome": "timed out"}) == 0
    assert rc.ramp_status(rows("passed", "passed"), {"outcome": "check failed"}) == 1
    assert rc.ramp_status(rows("check failed"), None) == 1
    assert rc.ramp_status(rows("out of memory"), None) == 6
    assert rc.ramp_status(rows(rc.HOST_GUARD), None) == 6
    assert rc.ramp_status([], None) == 6
    for outcome in rc.OUTCOMES:
        assert outcome in rc.__doc__, outcome


# --- the host guard -------------------------------------------------------------


def test_the_host_guard_refuses_a_rung_before_anything_is_built(tmp_path, monkeypatch):
    built: list = []
    monkeypatch.setattr(rc, "host_available_bytes", lambda: (30 * 1024 * 1024, "a test"))
    monkeypatch.setattr(rc, "build_state", lambda *a, **k: built.append(a))
    status, out, summary, text = _ramp(tmp_path, "--cells", "5000")
    assert not built
    assert status == rc.EXIT_NO_CEILING, text
    assert summary["rungs"][0]["outcome"] == rc.HOST_GUARD == "refused by the host guard"
    record = json.loads((out / rc.rung_file(1)).read_text())
    assert record["launch"]["returncode"] == rc.EXIT_RUNG_HOST_GUARD
    host = record["host"]
    assert host["refused"] is True and host["available_bytes"] == 30 * 1024 * 1024
    assert host["needed_bytes"] > host["available_bytes"] // 2
    assert "more than half of the 0.03 GiB it has available (a test)" in record["outcome_detail"]


def test_the_host_guard_counts_one_block_on_an_accelerator_and_the_whole_run_on_cpu():
    cells = 300_000_000
    on_card = rc.host_bytes_needed(cells, 4, devices_are_host=False)
    on_host = rc.host_bytes_needed(cells, 4, devices_are_host=True)
    assert on_card == cells // 4 * 97 + 16 * 1024 * 1024
    assert on_host == int(cells * 97 * rc.APRIORI_STEP_MULTIPLE) + 16 * 1024 * 1024
    available, source = rc.host_available_bytes()
    assert available is None or available > 0
    assert rc.host_guard(1000, 4, devices_are_host=True)["refused"] is (
        available is not None and available < 2 * rc.host_bytes_needed(1000, 4,
                                                                       devices_are_host=True))
    assert rc.vm_hwm_bytes() >= rc.vm_rss_bytes() > 0


# --- the soak --------------------------------------------------------------------


def test_the_soak_repeats_the_rung_at_a_share_of_the_ceiling_and_records_the_growth(
        tmp_path, monkeypatch):
    """Two blocks at least, the checks after each, the memory read at the
    same point of each: 1000 bytes more in use per reading taken, here."""
    S.stub_memory(monkeypatch, bytes_per_cell=400.0, grow=1000)
    status, out, summary, text = _ramp(tmp_path, "--cells", "1800", "--device-memory-gb",
                                       repr(_CARD_GIB), "--soak-minutes", "0.0001",
                                       "--soak-at", "0.75")
    assert status == rc.EXIT_OK, text
    assert summary["ceiling"]["cells"] == 1728
    soak = summary["soak"]
    assert soak["outcome"] == "passed" and soak["file"] == rc.SOAK_FILE
    assert soak["cells"] == 1152 <= 0.75 * 1728 and soak["k"] == [3, 1, 2]
    record = json.loads((out / rc.SOAK_FILE).read_text())
    assert record["kind"] == rc.SOAK_KIND and len(record["blocks"]) == 2
    for block in record["blocks"]:
        ran = [c for c in block["checks"] if not c.get("not_run")]
        assert len(ran) == 16 and all(c["passed"] for c in ran)
        assert block["host_rss_bytes"] > 0 and len(block["memory_after"]) == 4
    # Four readings per block, each 1000 bytes more in use than the last.
    growth = record["growth"]
    assert growth["blocks"] == 2
    assert set(next(iter(growth["bytes_in_use"].values()))) == {"first_to_last", "after_first"}
    assert all(g["first_to_last"] == 4000 for g in growth["bytes_in_use"].values())
    assert all(g["first_to_last"] == 0 for g in growth["peak_bytes_in_use"].values())
    assert growth["host_rss_bytes"]["first_to_last"] is not None
    assert len(record["memory"]["readings"]) == 8
    assert record["memory"]["fill"] == pytest.approx(1152 / 4 * 400.0 / _CARD)
    assert "SOAK: 1152 cells: passed" in text and "device bytes in use grew by" in text
    assert rc.rederive(record)["problems"] == []
    status, table = _summarise(out)
    assert status == rc.EXIT_OK and "SOAK: 1152 cells: passed" in table


def test_a_soak_block_that_fails_its_checks_is_the_check_failed_status(tmp_path, monkeypatch,
                                                                     passing):
    real = rc.compare_with_tiling

    def off_by_a_cell(state, reference, tile, shape, chunk_bytes):
        got = real(state, reference, tile, shape, chunk_bytes)
        got["fields"]["f"]["max_rel"] = 3e-3
        return got

    monkeypatch.setattr(rc, "compare_with_tiling", off_by_a_cell)
    launch = S.scripted([passing[(1, 3, 1)], "run"])
    status, out, summary, text = _ramp(tmp_path, "--cells", "576", "--soak-minutes", "0.0001",
                                       "--soak-at", "1.0", launch=launch)
    assert status == rc.EXIT_CHECK_FAILED, text
    assert summary["soak"]["outcome"] == "check failed"
    record = json.loads((out / rc.SOAK_FILE).read_text())
    assert len(record["blocks"]) == 1                    # it stopped at the first bad block
    assert _summarise(out)[0] == rc.EXIT_CHECK_FAILED


# --- --summarise re-derives; the runner's summary reads none of it --------------


def _replayed(tmp_path: Path, passing, *, out: str = "out"):
    launch = S.scripted([passing[(1, 1, 1)], passing[(1, 3, 1)]])
    status, directory, summary, _ = _ramp(tmp_path, "--cells", "192", "576", launch=launch,
                                          out=out)
    assert status == rc.EXIT_OK and len(summary["rungs"]) == 2
    return directory


def _edit(path: Path, change) -> None:
    record = json.loads(path.read_text())
    change(record)
    path.write_text(json.dumps(record))


def _check(record: dict, name: str) -> dict:
    return next(c for c in record["checks"] if c["name"] == name)


_F_CHECK = "f vs the tiled reference max_rel"
#: (what is changed in rung 2's record, the exit status, a phrase printed)
_TAMPERED = {
    "a value over its limit, the flag left": (
        lambda r: _check(r, _F_CHECK).update(value=4e-3), 3, "is recorded as passed, but its "
        "value 0.004 against its limit 1e-05 says failed"),
    "a flag flipped, the value left": (
        lambda r: _check(r, _F_CHECK).update(passed=False), 3, "is recorded as failed, but its "
        "value"),
    "a failed check, honestly recorded, under a passed outcome": (
        lambda r: _check(r, _F_CHECK).update(value=4e-3, passed=False), 3,
        "it is recorded as passed, but 1 check(s) fail"),
    "a failed check with its outcome": (
        lambda r: (_check(r, _F_CHECK).update(value=4e-3, passed=False),
                   r.update(outcome="check failed")), 1, "check failed"),
    "an outcome no check supports": (
        lambda r: r.update(outcome="check failed"), 3, "no check fails by its own values"),
    "a non-finite value": (
        lambda r: _check(r, _F_CHECK).update(value=float("nan")), 3, "says failed"),
    "a limit raised": (
        lambda r: _check(r, _F_CHECK).update(value=4e-3, limit=1.0), 0, "passed"),
    "a fill written in": (
        lambda r: r["memory"].update(fill=0.5), 3, "its memory readings give None"),
    "cells that are not the tiling's": (
        lambda r: r.update(cells=10 ** 9), 3, "not its tile's times its counts"),
    "no checks at all": (
        lambda r: r.pop("checks"), 3, "it records no checks"),
    "an outcome of another kind": (
        lambda r: r.update(outcome="PASS"), 3, "its outcome is 'PASS'"),
}


@pytest.mark.parametrize("what", sorted(_TAMPERED))
def test_summarise_rederives_every_verdict_from_the_values_of_a_tampered_record(
        what, tmp_path, passing):
    change, status, words = _TAMPERED[what]
    out = _replayed(tmp_path, passing)
    assert _summarise(out)[0] == rc.EXIT_OK
    _edit(out / rc.rung_file(2), change)
    got, table = _summarise(out)
    assert got == status, table
    assert words in table
    if what == "a value over its limit, the flag left":
        # The verdict printed is the re-derived one, not the stored flag.
        assert "CEILING: rung 1, 192 cells" in table
        assert "STOPPED BY: rung 2 (576 cells): check failed" in table
    if what == "a limit raised":
        # The one tampering the values cannot show: both were changed
        # together.  The limit is recorded beside it, for the reader.
        assert json.loads((out / rc.rung_file(2)).read_text())["limit"] == rc.LIMIT


@pytest.mark.parametrize("content, words", [
    ("", "cannot be read"), ("[1, 2]", "not an object"), ("{\"kind\": \"goal\"}", "its kind"),
    ("{\"kind\": \"capacity rung\", \"k\": 3}", "cannot be read")])
def test_summarise_names_a_record_it_cannot_read_and_exits_3(content, words, tmp_path, passing):
    out = _replayed(tmp_path, passing)
    (out / rc.rung_file(2)).write_text(content)
    status, table = _summarise(out)
    assert status == rc.EXIT_RECORD_INVALID == 3 and words in table


def test_summarise_of_a_directory_with_no_rung_record_says_so(tmp_path):
    status, table = _summarise(tmp_path)
    assert status == rc.EXIT_NO_CEILING and "no capacity record" in table


def test_nothing_the_script_writes_is_taken_for_a_goal_file_by_the_runners_summary(
        tmp_path, passing, monkeypatch):
    """``run_pod.py --summarise`` on the capacity test's own directory finds
    no goal file; beside a session's goal files, it prints and exits exactly
    what it does without them."""
    S.stub_memory(monkeypatch, bytes_per_cell=400.0)
    status, out, summary, _ = _ramp(tmp_path, "--cells", "192", "576", "--soak-minutes",
                                    "0.0001", "--device-memory-gb", repr(_CARD_GIB),
                                    launch=S.scripted([passing[(1, 1, 1)], passing[(1, 3, 1)],
                                                       "run"]))
    assert status == rc.EXIT_OK and summary["soak"]["outcome"] == "passed"
    written = sorted(p.name for p in out.iterdir())
    assert rc.SOAK_FILE in written and rc.SUMMARY_FILE in written and len(written) == 7
    for name in written:
        assert name.startswith("capacity_"), name
        assert not any(name.startswith(goal) for goal in rc.RUNNER_GOALS), name
    alone = run_pod(S.RUNNER, ["--summarise", out], pythonpath=str(S.REPO / "src"), timeout=300)
    assert alone.returncode == 1 and "no goal JSON" in alone.stdout

    session = tmp_path / "session"
    shutil.copytree(_RECORDED_GOALS, session)
    before = run_pod(S.RUNNER, ["--summarise", session], pythonpath=str(S.REPO / "src"),
                     timeout=300)
    for path in out.iterdir():
        shutil.copy(path, session / path.name)
    after = run_pod(S.RUNNER, ["--summarise", session], pythonpath=str(S.REPO / "src"),
                    timeout=300)
    assert "no goal JSON" not in before.stdout and before.stdout.strip()
    assert (after.returncode, after.stdout) == (before.returncode, before.stdout)


def test_an_out_directory_holding_goal_files_or_an_earlier_run_is_refused(tmp_path, passing,
                                                                        capsys):
    session = tmp_path / "session"
    shutil.copytree(_RECORDED_GOALS, session)
    launch = S.scripted([])
    assert rc.exit_status(["--dry-run", "--out", str(session), *S.TILE_ARGS, "--cells", "192"],
                          launch) == rc.EXIT_REFUSED
    assert "holds the runner's goal files" in capsys.readouterr().err
    out = _replayed(tmp_path, passing)
    assert rc.exit_status(["--dry-run", "--out", str(out), *S.TILE_ARGS, "--cells", "192"],
                          launch) == rc.EXIT_REFUSED
    assert "holds the records of an earlier run" in capsys.readouterr().err
    assert not launch.calls


# --- what the script may start, and what it may not ----------------------------


_FORBIDDEN_IMPORTS = ("sky", "skypilot", "runpod", "boto3", "requests", "urllib", "http",
                      "httpx", "aiohttp", "maddening.cloud.launcher", "maddening.cloud.session",
                      "maddening.cloud._skypilot", "maddening.api")
_FORBIDDEN_NAMES = {"CloudLauncher", "launch_vm", "teardown_vm", "CloudSession"}
_STARTS_A_PROCESS = {"run", "Popen", "call", "check_call", "check_output", "system", "popen",
                     "execv", "execve", "spawnv", "fork"}


def test_the_script_starts_only_itself_and_reaches_nothing():
    """No network or cloud library, no launcher; one process start in the
    whole file, of ``sys.executable`` on this very file; and no command
    line naming ``nvidia-smi`` (the words appear in docstrings only)."""
    source = S.SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(S.SCRIPT))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    for name in imported:
        for forbidden in _FORBIDDEN_IMPORTS:
            assert not (name == forbidden or name.startswith(forbidden + ".")), name
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not (used & _FORBIDDEN_NAMES), used & _FORBIDDEN_NAMES

    starts = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr in _STARTS_A_PROCESS
              and isinstance(n.func.value, ast.Name) and n.func.value.id in ("subprocess", "os")]
    assert len(starts) == 1 and starts[0].func.attr == "Popen"
    assert ast.unparse(starts[0].args[0]).startswith(
        "[sys.executable, str(Path(__file__).resolve()), *argv]")

    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
                  and node.body and isinstance(node.body[0], ast.Expr)
                  and isinstance(node.body[0].value, ast.Constant)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                and "nvidia" in node.value.lower():
            assert id(node) in docstrings, node.value[:80]


def test_summarise_does_not_import_jax(tmp_path):
    """The summary must run on a laptop without a working jaxlib."""
    import subprocess
    import sys

    from tests.cloud.multigpu.run_pod_support import offline_env

    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('rc', {str(S.SCRIPT)!r})\n"
        "rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)\n"
        f"status = rc.exit_status(['--summarise', {str(tmp_path)!r}])\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] in ('jax', 'jaxlib'))\n"
        "print(status, loaded)\n")
    out = subprocess.run([sys.executable, "-c", code], env=offline_env(str(S.REPO / "src")),
                         capture_output=True, text=True, timeout=300, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "6 []", out.stdout


def test_the_allocator_environment_is_recorded_as_found_and_defaulted_only_where_unset():
    """Before JAX is imported: what the operator set is kept and recorded
    as found; what is unset gets the script's default."""
    import subprocess
    import sys

    from tests.cloud.multigpu.run_pod_support import offline_env

    code = (
        "import importlib.util, json, os, sys\n"
        f"spec = importlib.util.spec_from_file_location('rc', {str(S.SCRIPT)!r})\n"
        "rc = importlib.util.module_from_spec(spec); spec.loader.exec_module(rc)\n"
        "print(json.dumps([rc._allocator_environment(), 'jax' in sys.modules,\n"
        "                  os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']]))\n")
    env = {k: v for k, v in offline_env(str(S.REPO / "src")).items()
           if k not in rc.ALLOCATOR_VARIABLES}
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         timeout=300, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    got, jax_loaded, preallocate = json.loads(out.stdout.strip().splitlines()[-1])
    assert jax_loaded is False and preallocate == "false"
    assert got["found"] == {"XLA_PYTHON_CLIENT_PREALLOCATE": None,
                            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5",
                            "XLA_PYTHON_CLIENT_ALLOCATOR": None}
    assert got["used"] == {"XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                           "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.5",
                           "XLA_PYTHON_CLIENT_ALLOCATOR": None}
    # Here JAX is loaded already: nothing is set, and "used" claims nothing.
    import os
    before = {name: os.environ.get(name) for name in rc.ALLOCATOR_VARIABLES}
    assert rc._allocator_environment()["used"] == rc._allocator_environment()["found"]
    assert {name: os.environ.get(name) for name in rc.ALLOCATOR_VARIABLES} == before


def test_the_harness_refuses_to_run_a_rung_that_is_not_a_dry_run(tmp_path):
    for argv in (["--cells", "192", "--out", "x"], ["--ramp", "0.5", "--device-memory-gb", "24",
                                                    "--out", "x"]):
        with pytest.raises(AssertionError, match="refusing to run run_capacity.py"):
            S.run_capacity(argv, timeout=1)
        with pytest.raises(AssertionError, match="refusing to run run_capacity.py"):
            S.in_process(argv, 1.0, tmp_path / "err.log")
    assert S.checked_argv(["--summarise", "x"]) == ["--summarise", "x"]
