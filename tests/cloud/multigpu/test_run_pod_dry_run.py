"""The pod-side multi-GPU runner proves itself on CPU virtual devices.

``benchmarks/multigpu/run_pod.py --dry-run`` runs each goal end-to-end
in a subprocess on four virtual host devices; the JSON it writes has the
schema the session summary depends on, the sharded results match the
unsharded references, and ``--summarise`` renders the ranking table
without pretending a CPU dry run ranks NCCL transports.  The decision
rule itself is checked on synthetic hardware-shaped results.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import maddening

_RUNNER = Path(maddening.__file__).resolve().parents[2] / "benchmarks" / "multigpu" / "run_pod.py"
_SRC = str(Path(maddening.__file__).resolve().parents[1])
_N_DEV = 4
_CELLS = 256
_METHODS = ("all_to_all", "ppermute")


def _env() -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("XLA_FLAGS", "JAX_PLATFORMS", "MADDENING_VIRTUAL_DEVICES")}
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={_N_DEV}"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run(*argv: str) -> subprocess.CompletedProcess:
    out = subprocess.run([sys.executable, str(_RUNNER), *argv], env=_env(),
                         capture_output=True, text=True, timeout=600, check=False)
    assert out.returncode == 0, f"stdout:\n{out.stdout[-3000:]}\nstderr:\n{out.stderr[-3000:]}"
    return out


def _runner_module():
    spec = importlib.util.spec_from_file_location("run_pod_under_test", _RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dry_run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("multigpu_dry_run")
    for goal in ("exchange", "forward", "gradient"):
        _run("--goal", goal, "--dry-run", "--cells", str(_CELLS), "--out", str(out))
    return out


def _load(directory: Path, goal: str) -> dict:
    with open(directory / f"{goal}.json", encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["schema_version"] == 1
    assert doc["goal"] == goal
    assert doc["dry_run"] is True
    env = doc["environment"]
    assert env["platform"] == "cpu"
    assert len(env["devices"]) == _N_DEV
    assert isinstance(env["jax"], str) and env["jax"].count(".") >= 1
    assert isinstance(env["jaxlib"], str)
    assert doc["config"]["n_devices"] == _N_DEV
    assert doc["wall_s"] > 0
    return doc


def _check_timing(t: dict):
    assert len(t["ms"]) == t["repeats"] >= 1
    assert t["warmup"] >= 1
    assert t["min_ms"] == min(t["ms"]) <= t["median_ms"] <= max(t["ms"])
    assert t["mean_ms"] > 0


@pytest.mark.slow
def test_exchange_json_ranks_both_transports_with_traffic(dry_run_dir):
    doc = _load(dry_run_dir, "exchange")
    (r,) = doc["results"]
    assert r["cells"] == _CELLS and r["n_devices"] == _N_DEV
    assert r["bit_identical"] is True
    assert set(r["methods"]) == set(_METHODS)
    traffic = r["traffic_cells_per_shard"]
    assert traffic["useful"] <= traffic["ppermute"] <= traffic["all_to_all"]
    for method in _METHODS:
        m = r["methods"][method]
        _check_timing(m)
        assert m["bytes_per_shard"] == traffic[method] * 4 * doc["config"]["fields"]
        assert m["bytes_total"] == m["bytes_per_shard"] * _N_DEV
        assert m["compile_s"] > 0
    assert r["methods"]["ppermute"]["messages"] == traffic["ppermute_messages"]
    assert r["ppermute_speedup_median"] == pytest.approx(
        r["methods"]["all_to_all"]["median_ms"] / r["methods"]["ppermute"]["median_ms"])


@pytest.mark.slow
def test_forward_json_matches_unsharded_reference_under_both_transports(dry_run_dir):
    doc = _load(dry_run_dir, "forward")
    (r,) = doc["results"]
    assert r["cells"] == _CELLS and r["steps"] == doc["config"]["steps"]
    for method in _METHODS:
        m = r["methods"][method]
        _check_timing(m["wrapper_step"])
        _check_timing(m["device_step"])
        assert m["wrapper_step"]["ms_per_step"] == pytest.approx(
            m["wrapper_step"]["median_ms"] / r["steps"])
        assert m["parity_x"]["finite"] and m["parity_x"]["max_rel"] < 1e-5
        assert m["parity_total"]["finite"] and m["parity_total"]["max_rel"] < 1e-5


@pytest.mark.slow
def test_gradient_json_reports_parity_for_rollout_and_sharded_cg(dry_run_dir):
    doc = _load(dry_run_dir, "gradient")
    (r,) = doc["results"]
    assert r["grad_steps"] == doc["config"]["grad_steps"]
    _check_timing(r["rollout"]["unsharded"]["grad"])
    for method in _METHODS:
        p = r["rollout"][method]["parity"]
        _check_timing(r["rollout"][method]["grad"])
        assert p["finite"] and p["reference_scale"] > 0 and p["max_rel"] < 1e-5
    cg = r["sharded_cg"]
    assert cg["dof"] == (_CELLS // _N_DEV) * _N_DEV
    _check_timing(cg["grad_sharded"])
    _check_timing(cg["grad_unsharded"])
    assert cg["grad_parity"]["finite"] and cg["grad_parity"]["max_rel"] < 1e-3
    assert cg["jvp_parity"]["finite"] and cg["jvp_parity"]["max_rel"] < 1e-3


@pytest.mark.slow
def test_summarise_prints_ranking_table_but_does_not_rank_a_dry_run(dry_run_dir):
    out = _run("--summarise", str(dry_run_dir)).stdout
    assert "Exchange ranking" in out
    assert f"{_CELLS:>9}    4 cpu" in out
    assert "Recommendation: undecided" in out
    assert "Forward run" in out and "Gradient parity" in out
    assert "sharded_cg grad" in out


def test_summarise_of_an_empty_directory_fails_clearly(tmp_path):
    out = subprocess.run([sys.executable, str(_RUNNER), "--summarise", str(tmp_path)],
                         env=_env(), capture_output=True, text=True, timeout=300, check=False)
    assert out.returncode == 1
    assert "no exchange/forward/gradient JSON" in out.stdout


def test_runner_makes_no_cloud_calls():
    src = _RUNNER.read_text(encoding="utf-8")
    for forbidden in ("maddening.cloud.launcher", "maddening.cloud._skypilot", "import sky",
                      "runpod", "CloudLauncher", "launch_vm"):
        assert forbidden not in src, forbidden


# --- decision rule and mesh helpers, in-process --------------------------------


def _exchange_doc(platform: str, rows: list[tuple[int, float, float]], *, dry_run=False) -> dict:
    results = []
    for cells, a2a, ppm in rows:
        results.append({
            "cells": cells, "n_devices": 4, "bit_identical": True,
            "ppermute_speedup_median": a2a / ppm,
            "methods": {
                "all_to_all": {"median_ms": a2a, "min_ms": a2a, "bytes_total": 8 * 4 * 4},
                "ppermute": {"median_ms": ppm, "min_ms": ppm, "bytes_total": 2 * 4 * 4},
            },
        })
    return {"goal": "exchange", "dry_run": dry_run, "results": results,
            "environment": {"platform": platform, "device_kinds": ["NVIDIA A100-SXM4-80GB"]}}


@pytest.mark.parametrize(
    "docs, decision",
    [
        ([_exchange_doc("gpu", [(100_000, 1.0, 0.5), (1_000_000, 4.0, 3.0)])], "ppermute"),
        ([_exchange_doc("gpu", [(100_000, 1.0, 0.5), (1_000_000, 3.0, 4.0)])], "all_to_all"),
        ([_exchange_doc("gpu", [(100_000, 1.0, 0.5), (1_000_000, 3.0, 2.95)])], "tie"),
        ([_exchange_doc("gpu", [(1_000, 1.0, 0.5)])], "undecided"),        # too small
        ([_exchange_doc("cpu", [(1_000_000, 1.0, 0.5)])], "undecided"),    # not hardware
        ([_exchange_doc("gpu", [(1_000_000, 1.0, 0.5)], dry_run=True)], "undecided"),
        ([], "undecided"),
    ],
)
def test_recommendation_rule(docs, decision):
    rec = _runner_module().recommend(docs)
    assert rec["decision"] == decision
    assert rec["reason"]


def test_neighbour_and_slab_tables_agree_with_the_layout():
    from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition

    rp = _runner_module()
    n, edges = rp.grid_edges(30)                      # 5x5 lattice
    assert n == 25
    tbl = rp.neighbour_table(n, edges)
    assert tbl.shape == (25, 4)
    assert sorted(tbl[12].tolist()) == [7, 11, 13, 17]          # interior cell
    assert sorted(tbl[0].tolist()) == [0, 0, 1, 5]              # corner: self-padded
    pa, how = rp.partition_cells(n, edges, 4, "contiguous")
    assert how == "contiguous"
    layout = build_unstructured_partition(partition_assignment=pa, edges=edges, n_devices=4)
    slab = rp.slab_table(layout, tbl)
    # every slab index resolves to the right global neighbour on the owner
    for g in range(n):
        d = int(pa[g])
        for j, s in enumerate(slab[g]):
            if s < layout.n_local_max:
                assert int(layout.local_global_ids[d][s]) == int(tbl[g, j])
            else:
                assert int(layout.ghost_global_ids[d][s - layout.n_local_max]) == int(tbl[g, j])
    ring = rp.ring_edges(8)
    assert ring.tolist()[-1] == [7, 0]
    with pytest.raises(ValueError, match="ring|grid"):
        rp.synthetic_mesh("torus", 8)
    rcm_pa, rcm_how = rp.partition_cells(n, edges, 4, "rcm")
    assert rcm_how == "rcm" and np.bincount(rcm_pa, minlength=4).min() > 0
