"""The pod-side multi-GPU runner proves itself on CPU virtual devices.

``benchmarks/multigpu/run_pod.py --goal all --dry-run`` runs every goal
end-to-end in one subprocess on four virtual host devices; the JSON it
writes has the schema the session summary depends on, every goal's
checks pass (the sharded results match their unsharded or NumPy
references, the refusals fire), every timed input is pre-placed on the
mesh (no per-call reshard is charged to a transport), both sides of the
gradient timing are compiled functions with compile time reported apart,
and ``--summarise`` renders the checklist verdict and the ranking table
without importing JAX and without pretending a CPU dry run closes the
checklist or ranks NCCL transports.  The decision rule, the verdict rule
and the stop-on-failure rule are checked in-process on synthetic results.
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
_GOALS = ("indivisible", "halo", "coupled", "stencil", "hybrid", "exchange", "forward", "gradient")


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
    """One process for every goal, as the session's own dry run does."""
    out = tmp_path_factory.mktemp("multigpu_dry_run")
    _run("--goal", "all", "--dry-run", "--cells", str(_CELLS), "--out", str(out))
    return out


def _load(directory: Path, goal: str) -> dict:
    with open(directory / f"{goal}.json", encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["schema_version"] == 3
    assert doc["goal"] == goal
    assert doc["dry_run"] is True
    assert doc["allow_fewer_devices"] is False
    assert doc["n_devices"] == _N_DEV
    env = doc["environment"]
    assert env["platform"] == "cpu"
    assert len(env["devices"]) == env["n_devices_visible"] == _N_DEV
    assert env["device_kinds"] == ["cpu"]
    # jaxlib precision is not comparable across versions: both are recorded.
    assert isinstance(env["jax"], str) and env["jax"].count(".") >= 1
    assert isinstance(env["jaxlib"], str) and env["jaxlib"].count(".") >= 1
    assert env["nvidia_smi"] == "skipped (dry run)"     # a dry run never probes the GPU
    assert doc["config"]["n_devices"] == _N_DEV
    assert doc["wall_s"] > 0
    return doc


@pytest.mark.slow
@pytest.mark.parametrize("goal", _GOALS)
def test_every_goal_records_checks_and_passes_them_in_the_dry_run(dry_run_dir, goal):
    doc = _load(dry_run_dir, goal)
    assert doc["checks"], "a goal with no checks compares nothing"
    failed = [c for c in doc["checks"] if not c["passed"]]
    assert not failed, failed
    assert doc["passed"] is True
    for c in doc["checks"]:
        assert set(c) >= {"name", "value", "limit", "passed"}


@pytest.mark.slow
def test_coupled_json_compares_forward_and_adjoint_under_both_solvers(dry_run_dir):
    doc = _load(dry_run_dir, "coupled")
    (r,) = doc["results"]
    assert r["cells"] == 16 * 16 and r["coupled_dof"] == 16 * 16 + 1
    assert r["parameters"] == ["field.diffusivity", "far.conductance", "field.exchange"]
    assert set(r["solvers"]) == {"ift", "fori"}
    for solver, sol in r["solvers"].items():
        assert sol["sharded"]["partitioned"] is True
        assert sol["unsharded"]["partitioned"] is False
        assert sol["parity_f"]["finite"] and sol["parity_f"]["max_rel"] < 1e-5
        assert sol["parity_u"] < 1e-5 and sol["parity_loss"] < 1e-5
        assert sol["parity_grad"] < (1e-4 if solver == "ift" else 1e-5)
        assert all(g != 0.0 for g in sol["unsharded"]["grad"])
        for side in ("sharded", "unsharded"):
            _check_timing(sol[side]["value_and_grad"])
            assert sol[side]["compile_s"] > 0
            assert isinstance(sol[side]["device0_pinned_ops"], int)
            # both paths against the float64 model of the coupled step
            model = sol["model"][side]
            assert model["f"]["max_rel"] < 1e-5 and model["u"] < 1e-5 and model["loss"] < 1e-5
            assert model["grad"] < 1e-4
    assert len(r["model"]["grad"]) == 3 and all(g != 0.0 for g in r["model"]["grad"])
    ift = r["solvers"]["ift"]
    assert 2 <= ift["sharded"]["last_step_iterations"] < r["max_iterations"]
    assert r["solvers"]["fori"]["sharded"]["last_step_iterations"] is None


@pytest.mark.slow
def test_halo_json_is_bit_exact_on_every_boundary_mode_width_and_mesh(dry_run_dir):
    doc = _load(dry_run_dir, "halo")
    (r,) = doc["results"]
    cases = r["stencil_cases"]
    # 3 boundary modes x 2 widths on the 1-D mesh and on the 2 x 2 pencil mesh
    assert {(c["mesh"], c["boundary"], c["halo"]) for c in cases} == {
        (m, b, h) for m in ("1d", "2d") for b in ("periodic", "edge", "zero") for h in (1, 2)}
    assert all(c["forward_max_abs"] == 0.0 and c["adjoint_max_abs"] == 0.0 for c in cases)
    assert set(r["unstructured"]["methods"]) == set(_METHODS)
    for m in r["unstructured"]["methods"].values():
        assert m["forward_max_abs"] == 0.0 and m["adjoint_max_abs"] == 0.0


@pytest.mark.slow
def test_stencil_and_hybrid_json_match_their_unsharded_nodes(dry_run_dir):
    (s,) = _load(dry_run_dir, "stencil")["results"]
    assert s["input_partitioned"] is True
    assert s["forward"]["parity_f"]["max_rel"] < 1e-5
    assert s["gradient"]["parity_grad_initial_field"]["max_rel"] < 1e-5
    assert s["gradient"]["parity_grad_diffusivity"] < 1e-5
    for side in ("sharded", "unsharded"):
        _check_timing(s["forward"][side]["rollout"])
        _check_timing(s["gradient"][side]["grad"])
    (h,) = _load(dry_run_dir, "hybrid")["results"]
    assert h["sharded"]["partitioned"] is True
    assert h["correction_rel"] > 1e-3            # the correction is part of the answer
    assert h["parity_f"]["max_rel"] < 1e-5 and h["parity_grad"] < 1e-5


@pytest.mark.slow
def test_indivisible_json_records_the_refusals_and_the_uneven_unstructured_run(dry_run_dir):
    (r,) = _load(dry_run_dir, "indivisible")["results"]
    ny, nx = r["stencil"]["shape"]
    assert ny % _N_DEV != 0
    for key in ("stencil", "pointwise", "pencil"):
        assert r[key]["raised"] == "ValueError", r[key]
    assert f"{ny} cells" in r["stencil"]["message"] and f"{_N_DEV} devices" in r["stencil"]["message"]
    assert r["stencil"]["divisible_raised"] is None
    assert "spatial axis 1" in r["pencil"]["message"]
    un = r["unstructured"]
    assert un["cells"] % _N_DEV != 0 and len(set(un["cells_per_device"])) > 1
    assert un["parity_x"]["max_rel"] < 1e-5


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


@pytest.mark.slow
def test_summarise_does_not_close_the_checklist_from_a_dry_run(dry_run_dir):
    out = _run("--summarise", str(dry_run_dir)).stdout
    lines = [line for line in out.splitlines() if line[:1].isdigit() and "[" in line]
    assert len(lines) == 6, out
    assert all("open: passed on CPU / dry run only" in line and "CLOSED" not in line
               for line in lines), lines
    for goal in _GOALS:
        assert f"{goal:<12} cpu" in out
    assert "Coupled group" in out and "Stencil wrapper" in out and "Halo exchange" in out


def test_summarise_of_an_empty_directory_fails_clearly(tmp_path):
    out = subprocess.run([sys.executable, str(_RUNNER), "--summarise", str(tmp_path)],
                         env=_env(), capture_output=True, text=True, timeout=300, check=False)
    assert out.returncode == 1
    assert "no goal JSON" in out.stdout


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


# --- checks, the checklist verdict and the stop rule, in-process -------------


def _goal_doc(goal: str, *, platform: str = "gpu", dry_run: bool = False, n_devices: int = 4,
              passed: bool = True, allow_fewer_devices: bool = False) -> dict:
    check = {"name": f"{goal} parity", "value": 1e-7 if passed else 1e-2, "limit": 1e-5,
             "passed": passed}
    return {"goal": goal, "dry_run": dry_run, "n_devices": n_devices,
            "allow_fewer_devices": allow_fewer_devices, "checks": [check], "passed": passed,
            "results": [],
            "environment": {"platform": platform, "device_kinds": ["NVIDIA A100-SXM4-80GB"],
                            "jax": "0.11.2", "jaxlib": "0.11.2"}}


def _all_goal_docs(**kw) -> dict:
    return {goal: [_goal_doc(goal, **kw)] for goal in _GOALS}


def test_check_helpers_fail_closed():
    rp = _runner_module()
    assert rp.check("x", 1e-6, 1e-5)["passed"] is True
    assert rp.check("x", 1e-4, 1e-5)["passed"] is False
    assert rp.check("x", float("nan"), 1e-5)["passed"] is False     # NaN is not "small"
    assert rp.check("x", 0.0, rp.LIMITS["exact"])["passed"] is True
    assert rp.check("x", 1e-30, rp.LIMITS["exact"])["passed"] is False
    assert rp.finish_checks({}, [])["passed"] is False               # nothing compared
    assert rp.finish_checks({}, [rp.check_that("y", True)])["passed"] is True
    assert rp.finish_checks({}, [rp.check_that("y", True), rp.check_that("z", False)])[
        "passed"] is False
    assert rp._rel_each([1.0, 2.0], [1.0, 2.0 + 2e-6]) == pytest.approx(1e-6, rel=1e-3)


def test_the_checklist_closes_only_on_a_real_gpu_run_with_enough_devices():
    rp = _runner_module()
    closed = rp.checklist_status(_all_goal_docs())
    assert [s for s, _ in closed.values()] == ["CLOSED"] * 6
    for kw in ({"dry_run": True}, {"platform": "cpu"}, {"n_devices": 2}):
        status = rp.checklist_status(_all_goal_docs(**kw))
        assert all(s.startswith("open: passed on CPU") for s, _ in status.values()), (kw, status)
    fewer = rp.checklist_status(_all_goal_docs(n_devices=2, allow_fewer_devices=True))
    assert all(s == "CLOSED" for s, _ in fewer.values())
    # one failed goal fails every item it decides, and only those
    docs = _all_goal_docs()
    docs["coupled"] = [_goal_doc("coupled", passed=False)]
    status = rp.checklist_status(docs)
    assert status[6][0] == "FAILED" and status[3][0] == "FAILED"
    assert status[1][0] == status[2][0] == status[4][0] == status[5][0] == "CLOSED"
    # a goal never run leaves its items open
    del docs["hybrid"]
    assert rp.checklist_status(docs)[4] == ("open", "hybrid not run")
    # a schema-2 file recorded no checks and cannot close anything
    docs = _all_goal_docs()
    del docs["forward"][0]["checks"]
    assert rp.checklist_status(docs)[1][0] == "open"


def test_summarise_exits_3_and_lists_a_failed_check(tmp_path, capsys):
    rp = _runner_module()
    for goal, doc in _all_goal_docs().items():
        if goal in ("exchange", "forward", "gradient"):
            continue                    # the timing tables need full results
        if goal == "halo":
            doc = [_goal_doc("halo", passed=False)]
        (tmp_path / f"{goal}.json").write_text(json.dumps(doc[0]), encoding="utf-8")
    monkey_tables = rp._print_checklist_goal_tables
    try:
        rp._print_checklist_goal_tables = lambda docs: None
        assert rp.summarise(tmp_path) == 3
    finally:
        rp._print_checklist_goal_tables = monkey_tables
    out = capsys.readouterr().out
    assert "[halo] halo parity: 1.000e-02 (limit 1e-05)" in out
    assert "2  halo exchange at the shard and global boundaries" in out and "FAILED" in out


def test_halo_reference_encodes_the_documented_boundary_fill():
    """The NumPy reference the halo goal holds the exchange to, pinned on
    a case small enough to read: 8 cells on 2 shards."""
    rp = _runner_module()
    assert rp.halo_index_map(8, 2, 1, "periodic").tolist() == [[7, 0, 1, 2, 3, 4],
                                                               [3, 4, 5, 6, 7, 0]]
    assert rp.halo_index_map(8, 2, 1, "edge").tolist() == [[0, 0, 1, 2, 3, 4],
                                                           [3, 4, 5, 6, 7, 7]]
    assert rp.halo_index_map(8, 2, 1, "zero").tolist() == [[-1, 0, 1, 2, 3, 4],
                                                           [3, 4, 5, 6, 7, -1]]
    # width 2: "edge" repeats the shard's own two outermost cells, in order
    assert rp.halo_index_map(8, 2, 2, "edge").tolist() == [[0, 1, 0, 1, 2, 3, 4, 5],
                                                           [2, 3, 4, 5, 6, 7, 6, 7]]
    a = np.arange(8, dtype=np.float32).reshape(8, 1) + 1
    rows = rp.halo_index_map(8, 2, 1, "zero")
    padded, grad = rp.halo_reference(a, rows, np.array([[0]]), np.ones((12, 1), np.float32))
    assert padded[:, 0].tolist() == [0, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 0]
    # each owned cell once, plus once more per halo slot it fills
    assert grad[:, 0].tolist() == [1, 1, 1, 2, 2, 1, 1, 1]


def test_checklist_goals_refuse_a_single_device():
    rp = _runner_module()
    with pytest.raises(SystemExit, match="on one device nothing is sharded"):
        rp.check_checklist_device_count(1)
    rp.check_checklist_device_count(2)


def test_a_multi_goal_run_stops_at_the_first_failing_goal(tmp_path, monkeypatch):
    """The session's stop condition, enforced by the runner: a failed goal
    writes its JSON, the next goal does not start, and the exit code is 1."""
    import jax

    if len(jax.devices()) < _N_DEV:
        pytest.skip(f"needs >= {_N_DEV} devices")
    rp = _runner_module()
    ran = []

    def failing(args, out):
        ran.append("indivisible")
        return rp.finish_checks(out, [rp.check("seeded", 1.0, 0.0)])

    def passing(name):
        def run(args, out):
            ran.append(name)
            return rp.finish_checks(out, [rp.check_that("fine", True)])
        return run

    monkeypatch.setattr(rp, "run_indivisible", failing)
    for name in ("halo", "coupled", "stencil", "hybrid"):
        monkeypatch.setattr(rp, f"run_{name}", passing(name))
    assert rp.main(["--goal", "checklist", "--dry-run", "--out", str(tmp_path)]) == 1
    assert ran == ["indivisible"]
    assert json.loads((tmp_path / "indivisible.json").read_text())["passed"] is False
    ran.clear()
    assert rp.main(["--goal", "checklist", "--dry-run", "--keep-going",
                    "--out", str(tmp_path / "all")]) == 1
    assert ran == ["indivisible", "halo", "coupled", "stencil", "hybrid"]
