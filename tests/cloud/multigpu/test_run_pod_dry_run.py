"""The pod-side multi-GPU runner proves itself on CPU virtual devices.

``benchmarks/multigpu/run_pod.py --dry-run`` runs each goal end-to-end
in a subprocess on four virtual host devices; the JSON it writes has the
schema the session summary depends on, the sharded results match the
unsharded references, every timed input is pre-placed on the mesh (no
per-call reshard is charged to a transport), both sides of the gradient
timing are compiled functions with compile time reported apart, and
``--summarise`` renders the ranking table without importing JAX and
without pretending a CPU dry run ranks NCCL transports.  The decision
rule itself is checked on synthetic hardware-shaped results, including
the device-count requirement.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

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
    assert doc["schema_version"] == 2
    assert doc["goal"] == goal
    assert doc["dry_run"] is True
    assert doc["allow_fewer_devices"] is False
    env = doc["environment"]
    assert env["platform"] == "cpu"
    assert len(env["devices"]) == _N_DEV
    assert isinstance(env["jax"], str) and env["jax"].count(".") >= 1
    assert isinstance(env["jaxlib"], str)
    assert env["nvidia_smi"] == "skipped (dry run)"     # a dry run never probes the GPU
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
    assert r["input_presharded"] is True
    assert "NamedSharding" in r["input_sharding"]
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
        assert m["input_presharded"] is True
        assert m["compile_s"] > 0 and m["wrapper_first_call_s"] > 0
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
    assert cg["input_presharded"] is True
    _check_timing(cg["grad_sharded"])
    _check_timing(cg["grad_unsharded"])
    assert cg["compile_s"]["sharded"] > 0 and cg["compile_s"]["unsharded"] > 0
    assert cg["grad_parity"]["finite"] and cg["grad_parity"]["max_rel"] < 1e-3
    assert cg["jvp_parity"]["finite"] and cg["jvp_parity"]["max_rel"] < 1e-3


@pytest.mark.slow
def test_rollout_grad_timings_are_of_compiled_functions(dry_run_dir):
    """Both sides are ``jax.jit(jax.grad(...))`` with compile time reported
    apart.  A non-jitted sharded grad retraced and re-partitioned the
    statics on every call (about 1700x the unsharded median in the dry
    run); a compiled one on four virtual CPUs stays within a small factor."""
    doc = _load(dry_run_dir, "gradient")
    (r,) = doc["results"]
    ref = r["rollout"]["unsharded"]
    assert ref["compile_s"] > 0
    for method in _METHODS:
        m = r["rollout"][method]
        assert m["compile_s"] > 0
        assert m["input_presharded"] is True
        assert m["grad"]["median_ms"] < 300 * ref["grad"]["median_ms"], (method, m["grad"], ref["grad"])


@pytest.mark.slow
def test_summarise_prints_ranking_table_but_does_not_rank_a_dry_run(dry_run_dir):
    out = _run("--summarise", str(dry_run_dir)).stdout
    assert "Exchange ranking" in out
    assert f"{_CELLS:>9}    4 cpu" in out
    assert "Recommendation: undecided" in out
    assert "dry-run / CPU row" in out
    assert "Forward run" in out and "Gradient parity" in out
    assert "sharded_cg grad" in out


def test_summarise_of_an_empty_directory_fails_clearly(tmp_path):
    out = subprocess.run([sys.executable, str(_RUNNER), "--summarise", str(tmp_path)],
                         env=_env(), capture_output=True, text=True, timeout=300, check=False)
    assert out.returncode == 1
    assert "no exchange/forward/gradient JSON" in out.stdout


def test_summarise_does_not_import_jax(tmp_path):
    """The summary must run on a laptop without a working jaxlib."""
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('rp', {str(_RUNNER)!r})\n"
        "rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)\n"
        f"rc = rp.main(['--summarise', {str(tmp_path)!r}])\n"
        "loaded = sorted(m for m in sys.modules if m == 'jax' or m.startswith('jax.') "
        "or m == 'jaxlib' or m.startswith('jaxlib.'))\n"
        "print(rc, loaded)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], env=_env(),
                         capture_output=True, text=True, timeout=300, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "1 []", out.stdout    # rc 1: empty dir


_FORBIDDEN_IMPORTS = ("sky", "skypilot", "runpod", "boto3",
                      "maddening.cloud.launcher", "maddening.cloud._skypilot")
_FORBIDDEN_NAMES = {"CloudLauncher", "launch_vm", "teardown_vm", "CloudSession"}


def _imports_of(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def test_runner_makes_no_cloud_calls():
    tree = ast.parse(_RUNNER.read_text(encoding="utf-8"), filename=str(_RUNNER))
    imported = _imports_of(tree)
    for name in imported:
        for forbidden in _FORBIDDEN_IMPORTS:
            assert not (name == forbidden or name.startswith(forbidden + ".")), name
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not (used & _FORBIDDEN_NAMES), used & _FORBIDDEN_NAMES


# --- decision rule, device-count gate and mesh helpers, in-process -----------


def _exchange_doc(platform: str, rows: list[tuple[int, float, float]], *, dry_run=False,
                  n_devices: int = 4, allow_fewer_devices: bool = False) -> dict:
    results = []
    for cells, a2a, ppm in rows:
        results.append({
            "cells": cells, "n_devices": n_devices, "bit_identical": True,
            "ppermute_speedup_median": (a2a / ppm) if ppm else None,
            "methods": {
                "all_to_all": {"median_ms": a2a, "min_ms": a2a, "bytes_total": 8 * 4 * 4},
                "ppermute": {"median_ms": ppm, "min_ms": ppm, "bytes_total": 2 * 4 * 4},
            },
        })
    return {"goal": "exchange", "dry_run": dry_run, "allow_fewer_devices": allow_fewer_devices,
            "results": results,
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


def test_recommendation_ignores_single_device_hardware_rows():
    """A 1-GPU run without ``--dry-run`` looks like hardware but exchanges
    nothing; its speedup is timer noise and must not pick a transport."""
    rp = _runner_module()
    rec = rp.recommend([_exchange_doc("gpu", [(100_000, 0.020, 0.010), (1_000_000, 0.02, 0.01)],
                                      n_devices=1)])
    assert rec["decision"] == "undecided"
    assert "n_devices=1" in rec["reason"] and ">= 4 devices" in rec["reason"]
    assert all(r["deciding"] is False for r in rec["rows"])


def test_recommendation_requires_four_devices_unless_fewer_are_allowed():
    rp = _runner_module()
    rows = [(100_000, 1.0, 0.5), (1_000_000, 4.0, 3.0)]
    two = rp.recommend([_exchange_doc("gpu", rows, n_devices=2)])
    assert two["decision"] == "undecided"
    assert "n_devices=2 < 4" in two["reason"] and "--allow-fewer-devices" in two["reason"]
    allowed = rp.recommend([_exchange_doc("gpu", rows, n_devices=2, allow_fewer_devices=True)])
    assert allowed["decision"] == "ppermute" and allowed["deciding_rows"] == 2
    # the escape hatch never admits a single device
    one = rp.recommend([_exchange_doc("gpu", rows, n_devices=1, allow_fewer_devices=True)])
    assert one["decision"] == "undecided" and "no exchange happens" in one["reason"]
    # the escape hatch may also be read from the recorded CLI config
    doc = _exchange_doc("gpu", rows, n_devices=3)
    del doc["allow_fewer_devices"]
    doc["config"] = {"allow_fewer_devices": True}
    assert rp.recommend([doc])["decision"] == "ppermute"
    # a 4-device real run decides on its own
    four = rp.recommend([_exchange_doc("gpu", rows, n_devices=4)])
    assert four["decision"] == "ppermute" and four["min_speedup"] == pytest.approx(4 / 3)


def test_recommendation_uses_only_the_deciding_rows_of_mixed_docs():
    rp = _runner_module()
    docs = [
        _exchange_doc("gpu", [(1_000_000, 1.0, 2.0)], n_devices=1),          # would say a2a
        _exchange_doc("cpu", [(1_000_000, 1.0, 2.0)]),                       # would say a2a
        _exchange_doc("gpu", [(1_000_000, 1.0, 2.0)], dry_run=True),         # would say a2a
        _exchange_doc("gpu", [(100_000, 2.0, 1.0), (1_000_000, 2.0, 1.0)]),  # decides: ppermute
    ]
    rec = rp.recommend(docs)
    assert rec["decision"] == "ppermute" and rec["deciding_rows"] == 2
    assert [r["deciding"] for r in rec["rows"]] == [False, False, False, True, True]


def test_recommendation_survives_zero_median():
    rp = _runner_module()
    rec = rp.recommend([_exchange_doc("gpu", [(100_000, 0.5, 0.0)])])
    assert rec["decision"] == "undecided"
    assert "0 ms" in rec["reason"]
    # a zero next to a deciding row does not poison the decision either
    rec = rp.recommend([_exchange_doc("gpu", [(100_000, 0.5, 0.0), (1_000_000, 4.0, 3.0)])])
    assert rec["decision"] == "ppermute" and rec["deciding_rows"] == 1


def test_exchange_goal_refuses_device_counts_that_cannot_decide():
    rp = _runner_module()
    with pytest.raises(SystemExit, match="performs no exchange"):
        rp.check_exchange_device_count(1, dry_run=False, allow_fewer=False)
    with pytest.raises(SystemExit, match="performs no exchange"):
        rp.check_exchange_device_count(1, dry_run=False, allow_fewer=True)
    with pytest.raises(SystemExit, match="--allow-fewer-devices"):
        rp.check_exchange_device_count(2, dry_run=False, allow_fewer=False)
    rp.check_exchange_device_count(2, dry_run=False, allow_fewer=True)
    rp.check_exchange_device_count(4, dry_run=False, allow_fewer=False)
    rp.check_exchange_device_count(1, dry_run=True, allow_fewer=False)   # proving the script


def test_file_partition_with_empty_shards_is_an_error():
    rp = _runner_module()
    with pytest.raises(rp.MeshPartitionError, match=r"2 non-empty part\(s\) but --n-devices is 4"):
        rp.check_file_partition(np.array([0, 0, 1, 1]), 4)
    with pytest.raises(rp.MeshPartitionError, match="part ids 0..4"):
        rp.check_file_partition(np.array([0, 1, 2, 3, 4]), 4)
    with pytest.raises(rp.MeshPartitionError, match="part ids -1"):
        rp.check_file_partition(np.array([-1, 0, 1, 2]), 4)
    ok = rp.check_file_partition(np.array([3, 2, 1, 0, 0]), 4)
    assert ok.dtype == np.int32 and ok.tolist() == [3, 2, 1, 0, 0]


def test_gradient_goal_runs_a_file_mesh_once(tmp_path, monkeypatch):
    """With ``--mesh`` there is one mesh, whatever ``--cells`` says; it is
    loaded once and its partition computed once per process."""
    rp = _runner_module()
    n, edges = rp.grid_edges(64)
    path = tmp_path / "mesh.npz"
    np.savez(path, edges=edges)
    args = SimpleNamespace(mesh=str(path), cells=[100, 300, 1000], synthetic="grid",
                           partition="contiguous")
    assert rp._sizes(args) == [1000]
    assert rp._sizes(SimpleNamespace(mesh=None, cells=[100, 300])) == [100, 300]
    loads, partitions = [], []
    real_load, real_partition = rp.load_mesh, rp.partition_cells
    monkeypatch.setattr(rp, "load_mesh", lambda p: (loads.append(p), real_load(p))[1])
    monkeypatch.setattr(rp, "partition_cells",
                        lambda *a: (partitions.append(a), real_partition(*a))[1])
    for size in rp._sizes(args):
        n_got, edges_got, pa, source = rp._mesh_source(args, size)
        rp._partition_for(args, source, n_got, edges_got, pa, 4)
        rp._mesh_source(args, size)             # a second goal asking again
        rp._partition_for(args, source, n_got, edges_got, pa, 4)
    assert n_got == n and loads == [str(path)] and len(partitions) == 1


def test_exchange_timing_input_is_presharded_on_the_mesh():
    """The slab the exchange goal times carries the mesh's ``NamedSharding``
    before the first timed call; an uncommitted device-0 array (what
    ``jnp.asarray`` of host data gives) is refused, because a compiled
    ``shard_map`` would scatter it to the mesh inside every timed call."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P

    from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition

    if len(jax.devices()) < _N_DEV:
        pytest.skip(f"needs >= {_N_DEV} devices")
    rp = _runner_module()
    rp._load_backend()
    mesh = rp._mesh_for(_N_DEV)
    n, edges = rp.grid_edges(_CELLS)
    pa, _ = rp.partition_cells(n, edges, _N_DEV, "contiguous")
    layout = build_unstructured_partition(partition_assignment=pa, edges=edges, n_devices=_N_DEV)
    values = np.random.default_rng(0).standard_normal(n).astype(np.float32)
    slab = rp.exchange_input(values, layout, mesh)
    assert slab.sharding == NamedSharding(mesh, P("devices"))
    assert rp.require_presharded([slab], mesh, "test") is True
    unplaced = jnp.asarray(rp.layout_slab(values, layout))
    assert unplaced.sharding != slab.sharding
    with pytest.raises(RuntimeError, match="reshard it from device 0"):
        rp.require_presharded([unplaced], mesh, "test")
    # and the goal itself records that its timed input was pre-placed
    args = SimpleNamespace(n_devices=_N_DEV, fields=1, cells=[_CELLS], synthetic="grid",
                           partition="contiguous", warmup=1, repeats=2, mesh=None)
    (entry,) = rp.run_exchange(args, {})["results"]
    assert entry["input_presharded"] is True
    assert entry["input_sharding"] == str(slab.sharding)
    assert entry["bit_identical"] is True


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
