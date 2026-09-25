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
from tests.cloud.multigpu.run_pod_support import offline_env, run_pod

_RUNNER = Path(maddening.__file__).resolve().parents[2] / "benchmarks" / "multigpu" / "run_pod.py"
_SRC = str(Path(maddening.__file__).resolve().parents[1])
_N_DEV = 4
_CELLS = 256
_METHODS = ("all_to_all", "ppermute")
_GOALS = ("indivisible", "halo", "coupled", "stencil", "hybrid", "exchange", "forward", "gradient")


def _env() -> dict:
    """Offline (``run_pod_support``): an empty ``HOME``, no cloud variables."""
    return offline_env(_SRC + os.pathsep + os.environ.get("PYTHONPATH", ""), _N_DEV)


def _run(*argv: str) -> subprocess.CompletedProcess:
    out = run_pod(_RUNNER, argv, pythonpath=_env()["PYTHONPATH"], timeout=600,
                  n_devices=_N_DEV)
    assert out.returncode == 0, f"stdout:\n{out.stdout[-3000:]}\nstderr:\n{out.stderr[-3000:]}"
    return out


def _runner_module():
    spec = importlib.util.spec_from_file_location("run_pod_under_test", _RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rules_only_runner():
    """The runner with its record-integrity rules switched off.

    The docs the verdict tests below build are synthetic -- one check, no
    results, no provenance -- and pin the other verdict rules in isolation:
    flags re-derived from value, limit and sense, the device count, the dry
    run, checks not run.  None of them is a record the runner would write,
    so each would read ``INVALID`` and hide the rule under test.  The record
    rules themselves are tested on a real runner record, in
    ``test_run_pod_verdict_integrity.py``.
    """
    rp = _runner_module()
    rp.record_problems = lambda doc: []
    rp.item_problems = lambda docs: []
    return rp


@pytest.fixture(scope="module")
def dry_run_dir(tmp_path_factory):
    """One process for every goal, as the session's own dry run does."""
    out = tmp_path_factory.mktemp("multigpu_dry_run")
    _run("--goal", "all", "--dry-run", "--cells", str(_CELLS), "--out", str(out))
    return out


def _load(directory: Path, goal: str) -> dict:
    with open(directory / f"{goal}.json", encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["schema_version"] == 5
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


# Slow: reads the all-goals dry run (dry_run_dir), one subprocess of 13-17 s on CI.
@pytest.mark.slow
@pytest.mark.parametrize("goal", _GOALS)
def test_every_goal_records_checks_and_passes_them_in_the_dry_run(dry_run_dir, goal):
    doc = _load(dry_run_dir, goal)
    assert doc["checks"], "a goal with no checks compares nothing"
    failed = [c for c in doc["checks"] if not c["passed"]]
    assert not failed, failed
    assert doc["passed"] is True
    rp = _runner_module()
    for c in doc["checks"]:
        assert set(c) >= {"name", "value", "limit", "sense", "passed"}
        assert c["sense"] in rp.SENSES
        # on four devices every case runs, and every record re-derives
        assert rp.check_status(c) == "passed", c
    # and the file is evidence --summarise accepts: the runner's own
    # derivation of its checks from its results gives exactly these
    assert rp.record_problems(doc) == [], rp.record_problems(doc)


# Slow: reads the all-goals dry run (dry_run_dir), one subprocess of 13-17 s on CI.
@pytest.mark.slow
def test_coupled_json_compares_forward_and_adjoint_under_both_solvers(dry_run_dir):
    doc = _load(dry_run_dir, "coupled")
    (r,) = doc["results"]
    # on the pencil mesh, the field, its two averages and the far field coupled
    assert (r["mesh"], r["mesh_shape"]) == ("2d", [2, 2])
    assert r["cells"] == 16 * 16 and r["coupled_dof"] == 16 * 16 + 2 + 1
    assert r["parameters"] == ["field.diffusivity", "far.conductance", "field.exchange"]
    assert set(r["solvers"]) == {"ift", "fori"}
    for solver, sol in r["solvers"].items():
        assert sol["sharded"]["partitioned"] is True
        assert sol["unsharded"]["partitioned"] is False
        assert sol["parity_f"]["finite"] and sol["parity_f"]["max_rel"] < 1e-5
        assert sol["parity_averages"]["finite"] and sol["parity_averages"]["max_rel"] < 1e-5
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
            assert model["averages"]["max_rel"] < 1e-5
            assert model["grad"] < 1e-4
    assert len(r["model"]["grad"]) == 3 and all(g != 0.0 for g in r["model"]["grad"])
    ift = r["solvers"]["ift"]
    assert 2 <= ift["sharded"]["last_step_iterations"] < r["max_iterations"]
    assert r["solvers"]["fori"]["sharded"]["last_step_iterations"] is None


# Slow: reads the all-goals dry run (dry_run_dir), one subprocess of 13-17 s on CI.
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


# Slow: reads the all-goals dry run (dry_run_dir), one subprocess of 13-17 s on CI.
@pytest.mark.slow
def test_stencil_and_hybrid_json_match_their_unsharded_nodes(dry_run_dir):
    """Periodic, edge (the wrapper's default) and Dirichlet ends of the
    field, and the D2Q9 lattice, on the 1-D and the pencil mesh, each
    forward and adjoint against the unsharded node.  Periodic alone let a
    wrapper whose unsharded halo axes always wrapped pass the goal, and the
    1-D mesh alone one that zero-filled every sharded axis but the first."""
    doc = _load(dry_run_dir, "stencil")
    cases = {(s["node"], s["mesh"], s["boundary"]): s for s in doc["results"]}
    assert set(cases) == {(node, mesh, b) for mesh in ("1d", "2d")
                          for node, b in (("field", "periodic"), ("field", "edge"),
                                          ("field", "dirichlet"), ("lbm", "periodic"))}
    assert len(doc["results"]) == 8
    for (node, mesh, boundary), s in cases.items():
        assert s["cells"] == 16 * 16
        assert s["mesh_shape"] == ([4, 1] if mesh == "1d" else [2, 2])
        assert s["input_partitioned"] is True
        fields = {"field": {"f", "averages"}, "lbm": {"f", "velocity"}}[node]
        assert set(s["forward"]["parity"]) == fields
        assert all(p["max_rel"] < 1e-5 for p in s["forward"]["parity"].values())
        assert s["gradient"]["parity_grad_initial_field"]["max_rel"] < 1e-5
        assert s["gradient"]["parity_grad_parameter"] < 1e-5
        assert s["parameter"] == {"field": "diffusivity", "lbm": "viscosity"}[node]
        for side in ("sharded", "unsharded"):
            _check_timing(s["forward"][side]["rollout"])
            _check_timing(s["gradient"][side]["grad"])
        prefix = f"{node} {mesh} {s['mesh_shape'][0]}x{s['mesh_shape'][1]} 16x16 {boundary}"
        assert sum(c["name"].startswith(prefix + " ") or c["name"].startswith(prefix + ":")
                   for c in doc["checks"]) == 10
    # the three conditions are different models: nothing here compares a
    # mode with itself under another name; and one model on both meshes
    for mesh in ("1d", "2d"):
        losses = {b: cases[("field", mesh, b)]["gradient"]["unsharded"]["loss"]
                  for b in ("periodic", "edge", "dirichlet")}
        assert len(set(losses.values())) == 3, losses
    for node, b in (("field", "periodic"), ("field", "edge"), ("field", "dirichlet"),
                    ("lbm", "periodic")):
        assert (cases[(node, "1d", b)]["gradient"]["unsharded"]["loss"]
                == cases[(node, "2d", b)]["gradient"]["unsharded"]["loss"])
    (h,) = _load(dry_run_dir, "hybrid")["results"]
    assert (h["mesh"], h["mesh_shape"]) == ("2d", [2, 2])
    assert h["sharded"]["partitioned"] is True
    assert h["correction_rel"] > 1e-3            # the correction is part of the answer
    assert h["parity_f"]["max_rel"] < 1e-5 and h["parity_grad"] < 1e-5
    assert h["parity_averages"]["max_rel"] < 1e-5


# Slow: reads the all-goals dry run (dry_run_dir), one subprocess of 13-17 s on CI.
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


# Slow: reads the all-goals dry run (dry_run_dir), one subprocess of 13-17 s on CI.
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


# Slow: reads the all-goals dry run (dry_run_dir), one subprocess of 13-17 s on CI.
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


# Slow: reads the all-goals dry run (dry_run_dir), one subprocess of 13-17 s on CI.
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


# Slow: reads the all-goals dry run (dry_run_dir), one subprocess of 13-17 s on CI.
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


# Slow: reads the all-goals dry run (dry_run_dir), one subprocess of 13-17 s on CI.
@pytest.mark.slow
def test_summarise_prints_ranking_table_but_does_not_rank_a_dry_run(dry_run_dir):
    out = _run("--summarise", str(dry_run_dir)).stdout
    assert "Exchange ranking" in out
    assert f"{_CELLS:>9}    4 cpu" in out
    assert "Recommendation: undecided" in out
    assert "dry-run / CPU row" in out
    assert "Forward run" in out and "Gradient parity" in out
    assert "sharded_cg grad" in out


# Slow: reads the all-goals dry run (dry_run_dir), one subprocess of 13-17 s on CI.
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
    assert "Checks not run" not in out and "Failed checks" not in out     # 4 devices
    assert "Records that cannot decide" not in out
    for mesh in ("1d 4x1", "2d 2x2"):
        for case in ("field", "lbm"):
            for boundary in (("periodic", "edge", "dirichlet") if case == "field"
                             else ("periodic",)):
                assert f"{case} {mesh} 16x16 {boundary}" in out, out


def test_the_stencil_goal_runs_every_boundary_and_the_lattice_on_every_mesh_at_its_smallest_size():
    rp = _runner_module()
    assert rp.STENCIL_BOUNDARIES == ("periodic", "edge", "dirichlet")
    smallest = []
    for mesh in ("1d", "2d"):
        smallest += [(256, "field", b, mesh) for b in rp.STENCIL_BOUNDARIES]
        smallest.append((256, "lbm", "periodic", mesh))
    assert rp.stencil_cases([1024, 256], 4) == [(1024, "field", "periodic", "2d"), *smallest]
    # no pencil mesh on 2 or 3 devices: the 1-D mesh carries every case
    assert rp.stencil_cases([256, 1024], 2) == [
        (256, "field", "periodic", "1d"), (256, "field", "edge", "1d"),
        (256, "field", "dirichlet", "1d"), (256, "lbm", "periodic", "1d"),
        (1024, "field", "periodic", "1d")]
    gpu = rp.stencil_cases(list(rp.GPU_CELLS), 4)
    assert gpu[:4] == [(100_000, "field", b, "1d") for b in rp.STENCIL_BOUNDARIES] + [
        (100_000, "lbm", "periodic", "1d")]
    assert gpu[-2:] == [(300_000, "field", "periodic", "2d"),
                        (1_000_000, "field", "periodic", "2d")]
    assert len(gpu) == 10
    # one grid for both meshes: each axis splits on the 1-D and the pencil mesh
    for cells in (*rp.GPU_CELLS, *rp.DRY_RUN_CELLS, 64):
        for n_dev in (2, 3, 4, 6, 8):
            ny, nx = rp.field_shape(cells, n_dev)
            assert ny % n_dev == 0 and nx % n_dev == 0


def test_summarise_of_an_empty_directory_fails_clearly(tmp_path):
    out = run_pod(_RUNNER, ["--summarise", tmp_path], pythonpath=_env()["PYTHONPATH"],
                  timeout=300, n_devices=_N_DEV)
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
    """An ``exchange.json`` as the runner writes it (a ring mesh, so the
    recorded cell counts are the configured ones), checks included: a row
    decides only from a file that reads ``PASS``."""
    rp = _runner_module()
    results = []
    for cells, a2a, ppm in rows:
        results.append({
            "cells": cells, "n_devices": n_devices, "bit_identical": True,
            "input_presharded": True,
            "ppermute_speedup_median": (a2a / ppm) if ppm else None,
            "methods": {
                "all_to_all": {"median_ms": a2a, "min_ms": a2a, "bytes_total": 8 * 4 * 4},
                "ppermute": {"median_ms": ppm, "min_ms": ppm, "bytes_total": 2 * 4 * 4},
            },
        })
    doc = {"schema_version": rp.SCHEMA_VERSION, "goal": "exchange", "dry_run": dry_run,
           "allow_fewer_devices": allow_fewer_devices, "n_devices": n_devices,
           "results": results,
           "environment": {"platform": platform, "device_kinds": ["NVIDIA A100-SXM4-80GB"],
                           "devices": [f"{platform}:{i}" for i in range(n_devices)],
                           "n_devices_visible": n_devices, "git_commit": "0" * 40},
           "config": {"cells": [cells for cells, _, _ in rows], "synthetic": "ring",
                      "mesh": None, "n_devices": n_devices,
                      "allow_fewer_devices": allow_fewer_devices}}
    return rp.finish_checks(doc, rp.exchange_checks(results, n_devices))


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
    doc["config"]["allow_fewer_devices"] = True
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
    assert rp.check("x", 1e-6, 1e-5)["sense"] == "<="
    assert rp.check_that("y", True)["sense"] == "=="
    assert rp.finish_checks({}, [])["passed"] is False               # nothing compared
    assert rp.finish_checks({}, [rp.check_that("y", True)])["passed"] is True
    assert rp.finish_checks({}, [rp.check_that("y", True), rp.check_that("z", False)])[
        "passed"] is False
    # a check not run is recorded, never passes, and makes the goal incomplete
    skipped = rp.check_not_run("pencil", "needs 4 devices")
    assert skipped["passed"] is False and skipped["not_run"] is True
    assert rp.check_status(skipped) == "not run"
    assert rp.finish_checks({}, [rp.check_that("y", True), skipped])["passed"] is False
    assert rp._rel_each([1.0, 2.0], [1.0, 2.0 + 2e-6]) == pytest.approx(1e-6, rel=1e-3)


@pytest.mark.parametrize("record, status", [
    ({"value": 0.5, "limit": 0.0, "sense": "<=", "passed": True}, "inconsistent"),
    ({"value": 0.0, "limit": 0.0, "sense": "<=", "passed": False}, "inconsistent"),
    ({"value": float("nan"), "limit": 1.0, "sense": "<=", "passed": True}, "inconsistent"),
    ({"value": False, "limit": True, "sense": "==", "passed": True}, "inconsistent"),
    ({"value": 0.5, "limit": 0.0, "sense": ">=", "passed": True}, "inconsistent"),  # unknown
    ({"value": "0.0", "limit": 0.0, "sense": "<=", "passed": True}, "inconsistent"),
    ({"value": 0.5, "limit": 0.0, "passed": True}, "inconsistent"),        # schema 3
    ({"value": 0.5, "limit": 0.0, "sense": "<=", "passed": False}, "failed"),
    ({"value": 1e-7, "limit": 1e-5, "sense": "<=", "passed": True}, "passed"),
    ({"value": 1e-7, "limit": 1e-5, "passed": True}, "passed"),            # schema 3
    ({"value": True, "limit": True, "passed": True}, "passed"),            # schema 3
    ({"value": None, "limit": None, "not_run": True, "passed": True}, "inconsistent"),
])
def test_check_status_rederives_pass_fail_from_the_value_and_limit(record, status):
    assert _runner_module().check_status({"name": "c", **record}) == status


def test_summarise_does_not_trust_a_recorded_pass(tmp_path, capsys):
    """A check with value 0.5 against limit 0.0, recorded ``passed: true``,
    read PASS / CLOSED: the summary counted the flags."""
    rp = _rules_only_runner()
    docs = _all_goal_docs()
    docs["halo"][0]["checks"][0].update(value=0.5, limit=0.0, sense="<=", passed=True)
    status = rp.checklist_status(docs)
    assert status[2][0] == "FAILED", status[2]
    assert rp.goal_verdict(docs["halo"]) == "FAIL"
    _write_checklist_docs(tmp_path, docs)
    assert _summarise_without_tables(rp, tmp_path) == 3
    out = capsys.readouterr().out
    assert ("[halo] halo parity: 5.000e-01 (limit 0.0) -- recorded passed=True, but the "
            "value fails its limit") in out
    # the file-level flag is re-derived too
    docs = _all_goal_docs()
    docs["hybrid"][0]["passed"] = False
    assert rp.checklist_status(docs)[4][0] == "FAILED"


def test_the_checklist_closes_only_on_a_real_gpu_run_with_enough_devices():
    rp = _rules_only_runner()
    closed = rp.checklist_status(_all_goal_docs())
    assert [s for s, _ in closed.values()] == ["CLOSED"] * 6
    for kw in ({"dry_run": True}, {"platform": "cpu"}):
        status = rp.checklist_status(_all_goal_docs(**kw))
        assert all(s.startswith("open: passed on CPU") for s, _ in status.values()), (kw, status)
    # --allow-fewer-devices is about the transport ranking: on 2 devices a
    # halo from the wrong neighbour passes every check, so fewer than four
    # devices never close an item, whatever the flag says
    for n_devices in (2, 3):
        for allow in (False, True):
            status = rp.checklist_status(_all_goal_docs(n_devices=n_devices,
                                                        allow_fewer_devices=allow))
            assert all(s == "open: passed on fewer than 4 devices"
                       for s, _ in status.values()), (n_devices, allow, status)
    assert not rp.closes_the_gap(_goal_doc("halo", n_devices=2, allow_fewer_devices=True))
    assert rp.closes_the_gap(_goal_doc("halo", n_devices=4))
    # a check not run keeps its items open; one that claims to have passed fails them
    docs = _all_goal_docs()
    docs["halo"][0]["checks"].append(rp.check_not_run("2d pencil", "needs 4"))
    docs["halo"][0]["passed"] = False
    assert rp.goal_verdict(docs["halo"]) == "INCOMPLETE"
    assert rp.checklist_status(docs)[2] == ("open", "halo INCOMPLETE")
    docs["halo"][0]["checks"][-1]["passed"] = True
    assert rp.checklist_status(docs)[2][0] == "FAILED"
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


def _write_checklist_docs(directory: Path, docs: dict) -> None:
    for goal, (doc, *_) in docs.items():
        if goal in ("exchange", "forward", "gradient"):
            continue                    # the timing tables need full results
        (directory / f"{goal}.json").write_text(json.dumps(doc), encoding="utf-8")


def _summarise_without_tables(rp, directory: Path) -> int:
    monkey_tables = rp._print_checklist_goal_tables
    try:
        rp._print_checklist_goal_tables = lambda docs: None
        return rp.summarise(directory)
    finally:
        rp._print_checklist_goal_tables = monkey_tables


def test_summarise_exits_3_and_lists_a_failed_check(tmp_path, capsys):
    rp = _rules_only_runner()
    docs = _all_goal_docs()
    docs["halo"] = [_goal_doc("halo", passed=False)]
    _write_checklist_docs(tmp_path, docs)
    assert _summarise_without_tables(rp, tmp_path) == 3
    out = capsys.readouterr().out
    assert "[halo] halo parity: 1.000e-02 (limit 1e-05)" in out
    assert "2  halo exchange at the shard and global boundaries" in out and "FAILED" in out


def test_summarise_lists_checks_not_run_and_exits_0(tmp_path, capsys):
    rp = _rules_only_runner()
    docs = _all_goal_docs()
    docs["indivisible"][0]["checks"].append(rp.check_not_run("pencil refusal", "needs 4"))
    docs["indivisible"][0]["passed"] = False
    _write_checklist_docs(tmp_path, docs)
    assert _summarise_without_tables(rp, tmp_path) == 0
    out = capsys.readouterr().out
    assert "Checks not run" in out and "[indivisible] pencil refusal -- needs 4" in out
    assert "5  an indivisible grid is refused" in out
    line = next(ln for ln in out.splitlines() if ln.startswith("5  "))
    assert "open  [indivisible INCOMPLETE]" in line and "CLOSED" not in line
    assert "Failed checks" not in out


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
    # width 2: "edge" repeats the outermost cell across the halo, as
    # numpy.pad(mode="edge") does -- not the two outermost cells in order
    assert rp.halo_index_map(8, 2, 2, "edge").tolist() == [[0, 0, 0, 1, 2, 3, 4, 5],
                                                           [2, 3, 4, 5, 6, 7, 7, 7]]
    assert rp.halo_index_map(8, 2, 2, "zero").tolist() == [[-1, -1, 0, 1, 2, 3, 4, 5],
                                                           [2, 3, 4, 5, 6, 7, -1, -1]]
    a = np.arange(8, dtype=np.float32).reshape(8, 1) + 1
    rows = rp.halo_index_map(8, 2, 1, "zero")
    padded, grad = rp.halo_reference(a, rows, np.array([[0]]), np.ones((12, 1), np.float32))
    assert padded[:, 0].tolist() == [0, 1, 2, 3, 4, 5, 4, 5, 6, 7, 8, 0]
    # each owned cell once, plus once more per halo slot it fills
    assert grad[:, 0].tolist() == [1, 1, 1, 2, 2, 1, 1, 1]


def _two_device_goals_record_what_they_cannot_run(goals):
    """Run ``goals`` (``(name, runner, expected not-run prefixes)``) on 2
    devices and check each records exactly those checks as not run, passes
    the rest, and makes a valid record that reads ``INCOMPLETE``."""
    rp = _runner_module()
    args = SimpleNamespace(n_devices=2, cells=[64], synthetic="grid", partition="contiguous",
                           mesh=None, steps=1, grad_steps=1, warmup=0, repeats=1)
    for goal, run_name, expected in goals:
        doc = getattr(rp, run_name)(args, {})
        not_run = [c for c in doc["checks"] if c.get("not_run")]
        assert {next(e for e in expected if c["name"].startswith(e)) for c in not_run} \
            == expected, not_run
        assert all(c["passed"] for c in doc["checks"] if not c.get("not_run"))
        assert doc["passed"] is False
        # The envelope main() writes around it, on two GPUs: the record is
        # valid -- its checks are what the runner derives on two devices,
        # not-run checks included -- and the goal reads incomplete.
        doc.update(schema_version=rp.SCHEMA_VERSION, goal=goal, n_devices=2, dry_run=False,
                   allow_fewer_devices=False,
                   environment={"platform": "gpu", "devices": ["gpu:0", "gpu:1"],
                                "n_devices_visible": 2, "git_commit": "0" * 40},
                   config={"cells": [64], "synthetic": "grid", "mesh": None, "n_devices": 2,
                           "allow_fewer_devices": False})
        assert rp.record_problems(doc) == [], rp.record_problems(doc)
        assert rp.goal_verdict([doc]) == "INCOMPLETE"


def test_a_two_device_run_records_the_cases_it_cannot_run():
    """On 2 devices the 2-D pencil cases have no mesh and a halo from the
    wrong neighbour cannot show; each is a check *not run*, so the goal
    reads incomplete instead of passing on what it could reach."""
    _two_device_goals_record_what_they_cannot_run([
        ("halo", "run_halo", {"2d pencil mesh", "1d mesh: left and right"}),
        ("indivisible", "run_indivisible", {"pencil (2-D mesh) refusal"})])


@pytest.mark.slow
def test_a_two_device_run_of_the_wrapper_goals_records_the_pencil_as_not_run():
    """The goals that run the stencil wrapper, on 2 devices: every 1-D case
    runs and passes, and the pencil mesh is one check not run.  Slow: it
    compiles four stencil cases, a hybrid graph and a coupled group's
    adjoint (43 s on three cores)."""
    _two_device_goals_record_what_they_cannot_run([
        ("stencil", "run_stencil", {"2d pencil mesh"}),
        ("hybrid", "run_hybrid", {"2d pencil mesh"}),
        ("coupled", "run_coupled", {"2d pencil mesh"})])


def test_the_stencil_goal_fails_when_unsharded_halo_axes_always_wrap(monkeypatch):
    """The fault the periodic-only goal could not see: the wrapper fills
    the halo of an axis it does not shard periodically whatever
    ``boundary`` says.  The field's ``"edge"`` case fails on it, and so
    does its Dirichlet case, which overwrites the field's halos but reads
    the grid-shaped source's, which the wrapper fills ``"edge"`` there;
    periodic wraps anyway, and so does the lattice.  (On 2 devices there
    is only the 1-D mesh, whose axis 1 is the unsharded one.)"""
    from maddening.cloud.multigpu import sharded_node

    rp = _runner_module()
    rp._load_backend()
    real = sharded_node._global_edge_halos

    def always_wrap(left, right, *, spatial_axis, halo, boundary):
        return real(left, right, spatial_axis=spatial_axis, halo=halo, boundary="periodic")

    monkeypatch.setattr(sharded_node, "_global_edge_halos", always_wrap)
    args = SimpleNamespace(n_devices=2, cells=[64], steps=2, grad_steps=2, warmup=0,
                           repeats=1)
    # The goal's field cases on the only mesh 2 devices have; its lattice
    # case is periodic, which the fault cannot move, and is left out here
    # to keep this per-push test to three small compiles.
    cases = [c for c in rp.stencil_cases(args.cells, args.n_devices) if c[1] == "field"]
    assert [c[2:] for c in cases] == [("periodic", "1d"), ("edge", "1d"), ("dirichlet", "1d")]
    checks = rp.stencil_checks([rp.run_stencil_case(c, args) for c in cases], args.n_devices)
    failed = {" ".join(c["name"].split()[:5]).rstrip(":") for c in checks
              if not c["passed"] and not c.get("not_run")}
    assert failed == {"field 1d 2x1 8x8 edge", "field 1d 2x1 8x8 dirichlet"}, [
        c["name"] for c in checks if not c["passed"]]


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
