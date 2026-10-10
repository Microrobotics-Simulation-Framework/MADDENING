"""``run_pod.py``'s checklist goals fail on a broken stencil wrapper.

A goal that compares a sharded run with an unsharded one proves the
wrapper right only if a wrong wrapper makes it fail.  Until 0.4.0 the
``stencil``, ``hybrid`` and ``coupled`` goals passed -- and a dry run
relabelled as four real GPUs closed checklist items 1, 3, 4 and 6 -- with
``ShardedStencilNode`` broken in each of the first four ways below, every
one of which its unit tests catch: the goals read a static only in the
interior, fed no grid-shaped input, declared no domain integral and ran on
a 1-D mesh only.  Then, on four devices, they passed every goal with each
of the last three -- faults confined to the second sharded axis: the
pencil was 2 x 2, so both of its axes had two devices (a shard's left and
right neighbour the same device), and every grid was square.  Each seed
here is one such fault, applied to a scratch copy of the library (never to
the tree under test), and the goals it names must fail on it while the
other checklist goals pass.

The seeds are exact text of the library.  When that code changes the
per-push test below fails, naming the seed: update the seed to the new
code rather than dropping it.

No wrapper seed reaches the ``gradient`` goal's ``sharded_cg`` part: the
operator it solves is the runner's own (a shard's two ghosts by
``ppermute``, zero at the global ends).  Its seeds (``_CG_SEEDS``) are
exact text of the runner, applied to a scratch copy of it: a ghost from the
wrong neighbour or one not zeroed at a global end must fail the part by
orders of magnitude, and the operator without its diagonal shift -- the
system the goal solved until schema 7, on which no float32 solve converges
-- must fail its convergence checks though its parity passes.
"""

from __future__ import annotations

import copy
import gc
import importlib.util
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

import maddening
from tests.cloud.multigpu.run_pod_support import checked_argv, offline_env, run_pod

_PKG = Path(maddening.__file__).resolve().parent
_RUNNER = _PKG.parents[1] / "benchmarks" / "multigpu" / "run_pod.py"
_N_DEV = 4
_WRAPPER = "cloud/multigpu/sharded_node.py"
_HALO = "cloud/multigpu/halo.py"

#: The checklist goals that run ``ShardedStencilNode``; every seed must fail
#: each of them.  ``indivisible`` builds the wrapper only to see it refuse,
#: and ``halo`` calls ``halo_exchange`` directly: a seed of the wrapper must
#: leave both passing, and a seed of ``halo_exchange`` fails ``halo`` too.
_WRAPPER_GOALS = ("stencil", "hybrid", "coupled")
_OTHER_GOALS = ("indivisible", "halo")


class Seed(NamedTuple):
    old: str
    new: str
    #: The mesh labels (``run_pod.STENCIL_MESHES``) the fault is confined to
    #: on four devices: every failed check of the ``stencil`` goal must name
    #: one of them.  Empty: any mesh.
    only_in: tuple = ()
    #: The library file the seed is applied to, under ``maddening/``.
    file: str = _WRAPPER
    #: The checklist goals it must fail; every other one must pass.
    fails: tuple = _WRAPPER_GOALS
    #: The mesh the per-push one-step test looks for the fault on.
    mesh: str = "1d"
    #: The fault makes the step raise rather than compute a wrong number
    #: (``shard_map`` refuses it): the goals must record that as a failure.
    raises: bool = False


_SEEDS = {
    "static_halos_are_nan": Seed(
        "                    padded_static[k] = halo_exchange(\n"
        "                        arr, mesh=mesh, axes=descriptors,\n"
        "                        boundary=static_boundary,\n"
        "                    )\n",
        "                    padded_static[k] = halo_exchange(\n"
        "                        arr, mesh=mesh, axes=descriptors,\n"
        "                        boundary=static_boundary,\n"
        "                    )\n"
        "                    # SEEDED FAULT: every halo cell of the static is NaN\n"
        "                    _h, _sa = descriptors[0][2], descriptors[0][1]\n"
        "                    _n = padded_static[k].shape[_sa]\n"
        "                    _i = jnp.arange(_n).reshape(\n"
        "                        [-1 if d == _sa else 1 for d in range(padded_static[k].ndim)])\n"
        "                    padded_static[k] = jnp.where((_i < _h) | (_i >= _n - _h), jnp.nan,\n"
        "                                                 padded_static[k])\n"),
    "grid_inputs_are_zeroed": Seed(
        "                k: (_pad_like_state(v) if k in grid_bi else v)\n",
        "                k: (_pad_like_state(v) * 0 if k in grid_bi else v)  # SEEDED FAULT\n"),
    "domain_integrals_are_halved": Seed(
        "                    red = lax.psum(v, axis_name=reduce_axes) if reduce_axes else v\n",
        "                    red = (lax.psum(v, axis_name=reduce_axes) if reduce_axes else v) * 0.5"
        "  # SEEDED FAULT\n"),
    # Two exchanged axes: the pencil, and the 1 x D two-axis mesh.
    "later_axes_are_zero_at_the_global_edges": Seed(
        "                arr2 = halo_exchange(\n"
        "                    arr2, mesh=mesh, axes=exchange_axes, boundary=boundary,\n"
        "                )\n",
        "                arr2 = halo_exchange(  # SEEDED FAULT: axes after the first zero-filled\n"
        "                    arr2, mesh=mesh, axes=exchange_axes,\n"
        "                    boundary={**{a[0]: 'zero' for a in exchange_axes[1:]},\n"
        "                              exchange_axes[0][0]: boundary},\n"
        "                )\n",
        only_in=("2d", "2d-flat"), mesh="2d"),
    # A domain integral summed over the first mesh axis only: on a 2-D mesh
    # the result is not replicated over the second, so shard_map refuses
    # the step and each wrapper goal raises -- which ended a --keep-going
    # run at the first of them, leaving the rest "not run".
    "domain_integrals_summed_over_the_first_mesh_axis_only": Seed(
        "                    red = lax.psum(v, axis_name=reduce_axes) if reduce_axes else v\n",
        "                    red = lax.psum(v, axis_name=reduce_axes[:1]) if reduce_axes else v"
        "  # SEEDED FAULT\n",
        mesh="2d", raises=True),
    # The three below passed every goal on four devices until schema 6.
    # Along spatial axis 1 each halo comes from the wrong neighbour: on a
    # mesh axis of two devices (both of the 2 x 2 pencil's) left and right
    # are one device, so only a mesh with all four on axis 1 shows it.
    "wrong_neighbour_along_spatial_axis_1": Seed(
        "    perm_backward = [(s, (s - 1) % p_size) for s in range(p_size)]\n",
        "    perm_backward = [(s, (s - 1) % p_size) for s in range(p_size)]\n"
        "    if spatial_axis == 1:  # SEEDED FAULT: each halo from the wrong neighbour\n"
        "        perm_forward, perm_backward = perm_backward, perm_forward\n",
        only_in=("1d-axis1", "2d-flat"), file=_HALO,
        fails=("halo", *_WRAPPER_GOALS), mesh="1d-axis1"),
    # shard_info's block extent divided by the first mesh axis's size: the
    # two axes of the 2 x 2 pencil have the same size, the 1 x D mesh's not.
    "shard_info_extent_from_the_first_mesh_axis": Seed(
        "            out[spatial_axis] = global_extent // int(self._mesh.shape[mesh_axis])\n",
        "            out[spatial_axis] = global_extent // int(  # SEEDED FAULT: first mesh axis\n"
        "                self._mesh.shape[next(iter(self._axis_map))])\n",
        only_in=("2d-flat",), mesh="2d-flat"),
    # shard_info's global extent read off spatial axis 0 for every axis:
    # invisible on a square grid, on any device count.
    "shard_info_extent_from_spatial_axis_0": Seed(
        "                f\"state field {f!r}\": int(jnp.shape(arr)[spatial_axis])\n",
        "                f\"state field {f!r}\": int(jnp.shape(arr)[0])  # SEEDED FAULT: axis 0\n",
        only_in=("1d-axis1", "2d-flat", "2d"), mesh="2d"),
}


def _runner():
    spec = importlib.util.spec_from_file_location("run_pod_seeded_under_test", _RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seeded_source(seed: Seed) -> str:
    text = (_PKG / seed.file).read_text(encoding="utf-8")
    assert text.count(seed.old) == 1, (
        f"the seed no longer matches {seed.file} exactly once ({text.count(seed.old)} "
        "matches): the library changed; update the seed to the new code")
    return text.replace(seed.old, seed.new)


# --- per push -----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_SEEDS))
def test_every_seed_applies_once_to_the_wrapper_as_it_stands(name):
    """A seed that matches nothing seeds nothing: the slow test would still
    pass its control and fail its seeded run for another reason -- or, run
    against an older wrapper, test code that no longer exists."""
    seed = _SEEDS[name]
    seeded = _seeded_source(seed)
    assert "SEEDED FAULT" in seeded
    compile(seeded, seed.file, "exec")


def test_every_seed_names_goals_meshes_and_a_mesh_the_runner_has():
    """A seed's expectations are about the runner as it stands: its goals
    are checklist goals, it fails every wrapper goal, and its meshes are
    meshes the runner builds on four devices."""
    rp = _runner()
    meshes = set(rp.meshes_that_fit(_N_DEV))
    for name, seed in _SEEDS.items():
        assert set(seed.fails) <= set(rp.CHECKLIST_GOALS), name
        assert set(_WRAPPER_GOALS) <= set(seed.fails), name
        assert set(seed.only_in) <= meshes and seed.mesh in meshes, name
        assert not seed.only_in or seed.mesh in seed.only_in, name


@pytest.mark.parametrize("argv", [
    ["--goal", "checklist", "--out", "x"], ["--goal", "stencil", "--keep-going", "--out", "x"],
    ["--summarise", "x", "--goal", "all", "--out", "x"], []])
def test_the_harness_refuses_to_run_a_goal_without_dry_run(argv):
    """No test runs a goal for real, even by mistake: without
    ``--dry-run`` the run is refused before anything starts."""
    with pytest.raises(AssertionError, match="refusing to run run_pod.py without --dry-run"):
        run_pod(_RUNNER, argv, pythonpath=str(_PKG.parent), timeout=1)
    assert checked_argv(["--summarise", "x"]) == ["--summarise", "x"]
    assert checked_argv([*argv, "--dry-run"])[-1] == "--dry-run"


def test_a_run_sees_no_cloud_credentials_and_an_empty_home(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "not-a-key")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "not-a-key")
    env = offline_env(str(_PKG.parent), _N_DEV)
    assert not [k for k in env if k.upper().startswith(("RUNPOD_", "AWS_"))]
    home = Path(env["HOME"])
    assert home != Path.home() and home.is_dir()
    assert not {p.name for p in home.iterdir()} & {".runpod", ".sky", ".aws", ".maddening"}


def test_the_seeds_must_fail_every_item_the_wrapper_decides():
    """Items 1, 3, 4 and 6 are the ones the wrapper goals decide; an item
    decided by none of them (2, the halo exchange; 5, the refusals) must
    not be reopened by a wrapper fault either."""
    rp = _runner()
    assert set(_WRAPPER_GOALS) | set(_OTHER_GOALS) == set(rp.CHECKLIST_GOALS)
    decided = {i for i, (_c, goals) in rp.CHECKLIST.items() if set(goals) & set(_WRAPPER_GOALS)}
    assert decided == {1, 3, 4, 6}


def _exec_module(name: str, source: str) -> types.ModuleType:
    module = types.ModuleType(name)
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)  # noqa: S102
    return module


@pytest.fixture(scope="module")
def seeded_wrapper_classes():
    """``{seed: ShardedStencilNode}`` built from each seeded source, in-process.

    A seed of the wrapper is the wrapper's seeded source; a seed of
    ``halo.py`` is the wrapper's own source with the seeded module's
    ``halo_exchange`` and ``_global_edge_halos`` in place of the real ones.
    Executed with ``@stability`` made a no-op, under module names outside
    the package: a seeded copy registered as a STABLE surface of the
    library is still in the process-wide registry when the stable-signature
    tests run later in the same process (it failed twelve of them on CI).
    The registry must come out exactly as it went in, and the classes are
    dropped when the module's tests end.
    """
    from maddening.core.compliance import stability as stab

    before = dict(stab._STABILITY_REGISTRY)
    real = stab.stability
    stab.stability = lambda level: (lambda obj: obj)
    out = {}
    try:
        for name, seed in _SEEDS.items():
            if seed.file == _WRAPPER:
                module = _exec_module(f"run_pod_seeded_wrapper_{name}", _seeded_source(seed))
            else:
                assert seed.file == _HALO, seed.file
                halo = _exec_module(f"run_pod_seeded_halo_{name}", _seeded_source(seed))
                module = _exec_module(f"run_pod_wrapper_for_{name}",
                                      (_PKG / _WRAPPER).read_text(encoding="utf-8"))
                module.halo_exchange = halo.halo_exchange
                module._global_edge_halos = halo._global_edge_halos
            out[name] = module.ShardedStencilNode
    finally:
        stab.stability = real
    assert stab._STABILITY_REGISTRY == before, "a seeded copy registered a stable surface"
    yield out
    out.clear()
    gc.collect()


@pytest.fixture(scope="module")
def rp_backend():
    import jax

    if len(jax.devices()) < _N_DEV:
        pytest.skip(f"needs >= {_N_DEV} devices")
    rp = _runner()
    rp._load_backend()
    return rp


def _one_step_forward(rp, wrapper_class, mesh_label: str) -> dict:
    """``{field: max_rel}`` of one step of the ``stencil`` goal's periodic
    ``Field2D`` case (8 x 12, its source fed), sharded by ``wrapper_class``
    on ``mesh_label``, against the unsharded node."""
    ny, nx = rp.field_shape(64, _N_DEV)
    mesh, axis_map, _ = rp.stencil_mesh(mesh_label, _N_DEV)
    real = rp.ShardedStencilNode
    rp.ShardedStencilNode = wrapper_class
    try:
        ref, sharded, bi, p = rp._stencil_pair("field", ny, nx, "periodic", mesh, axis_map)
    finally:
        rp.ShardedStencilNode = real
    start = dict(ref.initial_state())
    got = {label: rp._host(node.update(state, bi, ref.delta_t, params={"diffusivity": p}))
           for label, node, state in (("unsharded", ref, start),
                                      ("sharded", sharded, rp._placed_like(sharded, start)))}
    return {field: rp._diff(got["sharded"][field], got["unsharded"][field])["max_rel"]
            for field in rp.STENCIL_NODES["field"]["fields"]}


@pytest.mark.parametrize("name", [None, *sorted(_SEEDS)],
                         ids=["no_seed", *sorted(_SEEDS)])
def test_one_step_of_the_stencil_goals_node_shows_each_seeded_fault(
        name, rp_backend, seeded_wrapper_classes):
    """The per-push half of the slow test below: one step of the
    ``stencil`` goal's own node and inputs, sharded by the seeded wrapper,
    is off the unsharded step by more than the goal's forward limit -- and
    by the real wrapper, within it on every mesh.  A fault confined to some
    meshes is looked for on one of them (``Seed.mesh``)."""
    rp = rp_backend
    limit = rp.LIMITS["forward"]
    if name is None:
        for mesh_label in rp.meshes_that_fit(_N_DEV):
            rel = _one_step_forward(rp, rp.ShardedStencilNode, mesh_label)
            assert all(v <= limit for v in rel.values()), (mesh_label, rel)
    elif _SEEDS[name].raises:
        with pytest.raises(Exception):
            _one_step_forward(rp, seeded_wrapper_classes[name], _SEEDS[name].mesh)
    else:
        rel = _one_step_forward(rp, seeded_wrapper_classes[name], _SEEDS[name].mesh)
        assert any(not v <= limit for v in rel.values()), rel


# --- slow: the goals themselves, as the session runs them ---------------------


def _scratch_library(tmp: Path, seed: Seed | None) -> Path:
    src = tmp / "src"
    shutil.copytree(_PKG, src / "maddening", ignore=shutil.ignore_patterns("__pycache__"))
    if seed is not None:
        (src / "maddening" / seed.file).write_text(_seeded_source(seed), encoding="utf-8")
    found = subprocess.run(
        [sys.executable, "-c",
         "import importlib.util; print(importlib.util.find_spec('maddening').origin)"],
        env=offline_env(str(src), _N_DEV), capture_output=True, text=True, timeout=60,
        check=True).stdout.strip()
    assert Path(found).resolve() == (src / "maddening" / "__init__.py").resolve(), found
    return src


def _run_checklist(tmp: Path, src: Path) -> tuple[int, dict, str]:
    out = tmp / "out"
    proc = run_pod(_RUNNER, ["--goal", "checklist", "--dry-run", "--keep-going",
                             "--cells", "256", "--out", out],
                   pythonpath=str(src), timeout=1800, n_devices=_N_DEV)
    rp = _runner()
    docs = {goal: rp._load_results(out, goal) for goal in rp.CHECKLIST_GOALS}
    return proc.returncode, docs, proc.stdout[-4000:] + proc.stderr[-2000:]


def _as_four_gpus(docs: dict) -> dict:
    docs = copy.deepcopy(docs)
    for goal_docs in docs.values():
        for doc in goal_docs:
            doc["dry_run"] = False
            doc["environment"]["platform"] = "gpu"
            doc["environment"]["device_kinds"] = ["NVIDIA A100-SXM4-80GB"]
    return docs


# Per push: tests/cloud/multigpu/test_run_pod_seeded_faults.py::test_one_step_of_the_stencil_goals_node_shows_each_seeded_fault
@pytest.mark.slow
def test_the_checklist_goals_pass_on_the_scratch_copy_of_the_library(tmp_path):
    """The control: the scratch copy with no seed passes every goal, so a
    seeded run that fails does so because of its seed."""
    rc, docs, log = _run_checklist(tmp_path, _scratch_library(tmp_path, None))
    rp = _runner()
    assert rc == 0, log
    assert {g: rp.goal_verdict(d) for g, d in docs.items()} == {
        g: "PASS" for g in rp.CHECKLIST_GOALS}, log


# Per push: tests/cloud/multigpu/test_run_pod_seeded_faults.py::test_one_step_of_the_stencil_goals_node_shows_each_seeded_fault
@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(_SEEDS))
def test_a_seeded_wrapper_fault_fails_every_goal_that_runs_the_wrapper(name, tmp_path):
    seed = _SEEDS[name]
    rc, docs, log = _run_checklist(tmp_path, _scratch_library(tmp_path, seed))
    rp = _runner()
    verdicts = {g: rp.goal_verdict(d) for g, d in docs.items()}
    assert rc == 1, log
    assert verdicts == {g: "FAIL" if g in seed.fails else "PASS"
                        for g in rp.CHECKLIST_GOALS}, (verdicts, log)
    for goal in seed.fails:
        (doc,) = docs[goal]
        assert rp.record_problems(doc) == [], (goal, rp.record_problems(doc))
        assert ("raised" in doc) == seed.raises, (goal, doc.get("raised"))
    if seed.only_in:
        (stencil,) = docs["stencil"]
        failed = [c["name"] for c in stencil["checks"] if rp.check_status(c) == "failed"]
        assert failed and all(any(f" {m} " in c for m in seed.only_in) for c in failed), failed
    # What --summarise would say had the same checks failed the same way on
    # four GPUs: every item a failing goal decides has FAILED, the rest closed.
    status = {i: s for i, (s, _) in rp.checklist_status(_as_four_gpus(docs)).items()}
    failing = {i for i, (_c, goals) in rp.CHECKLIST.items() if set(goals) & set(seed.fails)}
    assert {1, 3, 4, 6} <= failing
    assert status == {i: "FAILED" if i in failing else "CLOSED" for i in rp.CHECKLIST}, status
    json.dumps(status)                                # (the verdicts are plain data)


# --- the gradient goal's sharded_cg part ---------------------------------------


class CgSeed(NamedTuple):
    old: str
    new: str
    #: The fault is in the sharded operator: the sharded side's checks and the
    #: parity fail, the unsharded side's pass.  ``False``: both sides solve
    #: the same wrong system, and parity cannot tell.
    sharded_only: bool = True
    #: How many times its limit each parity check must read at least.  A
    #: ghost from the wrong neighbour is wrong at every shard boundary; one
    #: kept at a global end is one wrong entry, and the measure is relative
    #: to the largest entry of the derivative.
    parity_over: float = 100.0


_CG_SEEDS = {
    "ghosts_from_the_wrong_neighbours": CgSeed(
        "        left_ghost = lax.ppermute(x[-1], MESH_AXIS, [(i, (i + 1) % D) for i in range(D)])\n"
        "        right_ghost = lax.ppermute(x[0], MESH_AXIS, [(i, (i - 1) % D) for i in range(D)])\n",
        "        # SEEDED FAULT: each ghost comes from the other neighbour\n"
        "        left_ghost = lax.ppermute(x[-1], MESH_AXIS, [(i, (i - 1) % D) for i in range(D)])\n"
        "        right_ghost = lax.ppermute(x[0], MESH_AXIS, [(i, (i + 1) % D) for i in range(D)])\n"),
    "left_global_end_keeps_the_wrapped_ghost": CgSeed(
        "        left_ghost = jnp.where(idx == 0, 0.0, left_ghost)\n",
        "        left_ghost = left_ghost  # SEEDED FAULT: not zeroed at the global end\n",
        parity_over=10.0),
    "right_global_end_keeps_the_wrapped_ghost": CgSeed(
        "        right_ghost = jnp.where(idx == D - 1, 0.0, right_ghost)\n",
        "        right_ghost = right_ghost  # SEEDED FAULT: not zeroed at the global end\n",
        parity_over=10.0),
    "the_operator_without_its_shift": CgSeed(
        "CG_SHIFT = 0.03\n",
        "CG_SHIFT = 0.0  # SEEDED FAULT: the system the goal solved until schema 7\n",
        sharded_only=False),
}

#: The dry run's larger size and its iteration cap: the unshifted operator
#: is far from converged there (at 256 unknowns CG's finite termination
#: brings it within a factor of a few of the limit).
_CG_DOF, _CG_CAP = 1024, 300


def _cg_seeded_runner(tmp: Path, name: str | None):
    """The runner, or a scratch copy of it with one seed applied, as a module."""
    if name is None:
        return _runner()
    seed = _CG_SEEDS[name]
    text = _RUNNER.read_text(encoding="utf-8")
    assert text.count(seed.old) == 1, (
        f"the seed {name!r} no longer matches run_pod.py exactly once "
        f"({text.count(seed.old)} matches): the runner changed; update the seed")
    path = tmp / f"run_pod_cg_{name}.py"
    path.write_text(text.replace(seed.old, seed.new), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"run_pod_cg_seeded_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "SEEDED FAULT" in path.read_text(encoding="utf-8")
    return module


def _cg_part(rp) -> tuple[dict, dict]:
    """``(entry, {check name: check})`` of the goal's ``sharded_cg`` part,
    run as the goal runs it, at the dry run's larger size and cap."""
    import jax

    if len(jax.devices()) < _N_DEV:
        pytest.skip(f"needs >= {_N_DEV} devices")
    rp._load_backend()
    args = SimpleNamespace(cg_max_iters=_CG_CAP, warmup=0, repeats=1)
    cg = rp.run_gradient_cg(args, rp._mesh_for(_N_DEV), _CG_DOF, np.random.default_rng(1))
    checks = rp.gradient_cg_checks(cg)
    assert len({c["name"] for c in checks}) == len(checks) == 12
    return cg, {c["name"]: c for c in checks}


def _cg_names(side: str) -> list[str]:
    prefix = f"{_CG_DOF} dof sharded_cg {side}"
    return [f"{prefix} solve stopped on its tolerance, not on the iteration cap",
            *(f"{prefix} {which} true residual" for which in ("solve", "adjoint", "tangent"))]


_CG_PARITY = [f"{_CG_DOF} dof sharded_cg {d} vs unsharded max_rel" for d in ("grad", "jvp")]


def test_the_cg_part_of_the_gradient_goal_passes_on_solves_that_converged(tmp_path):
    """The control of the seeds below, and the claim itself at the dry
    run's size: both derivatives agree with the unsharded ones to float32
    round-off -- three orders of magnitude inside the limit, where the
    check used to read within 8 % of it on solves that had not converged --
    and each side's three solves have converged with iterations to spare."""
    rp = _cg_seeded_runner(tmp_path, None)
    cg, checks = _cg_part(rp)
    assert {n: rp.check_status(c) for n, c in checks.items()} == {n: "passed" for n in checks}
    # measured 4.7e-7 (grad) and 4.0e-7 (jvp) on jax 0.10.2, 0.11.0 and 0.11.2
    assert all(checks[n]["value"] <= rp.LIMITS["krylov"] / 40 for n in _CG_PARITY), checks
    for side in ("sharded", "unsharded"):
        solve = cg["solve"][side]
        assert solve["converged"] is True
        # 56 iterations on each side: not a trivial solve, and well inside the cap
        assert 20 < solve["iterations"] < _CG_CAP // 2, solve
        # a converged solve reads just under its tolerance, never far under
        assert all(rp.CG_RTOL / 4 < r <= rp.LIMITS["krylov_residual"]
                   for r in solve["true_residual"].values()), solve
        # ... and the three are the residuals of three systems, not one reported thrice
        assert len(set(solve["true_residual"].values())) == 3, solve
    assert cg["solve"]["sharded"]["iterations"] == cg["solve"]["unsharded"]["iterations"]


def test_the_host_residual_is_that_of_the_shifted_operator_in_float64():
    """What the convergence checks measure with: ``|rhs - A x| / |rhs|`` for
    the tridiagonal ``A`` with ``2 + CG_SHIFT`` (as float32 rounds it) on
    its diagonal and -1 beside it, against a dense matrix written here."""
    rp = _runner()
    n = 9
    diagonal = float(np.float32(2 + rp.CG_SHIFT))
    a = diagonal * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    rng = np.random.default_rng(3)
    x, off = rng.standard_normal(n), rng.standard_normal(n)
    rhs = a @ x
    assert rp._cg_true_residual(x, rhs) < 1e-15
    assert rp._cg_true_residual(x.astype(np.float32), rhs) < 1e-6      # float32 in, float64 inside
    want = np.linalg.norm(off) / np.linalg.norm(rhs + off)
    assert rp._cg_true_residual(x, rhs + off) == pytest.approx(want, rel=1e-12)
    assert rp._cg_true_residual(np.zeros(n), rhs) == 1.0
    # the unshifted operator is another system: by CG_SHIFT |x| / |rhs|
    unshifted = (a - rp.CG_SHIFT * np.eye(n)) @ x
    assert rp._cg_true_residual(x, unshifted) > 1e-3


@pytest.mark.parametrize("name", sorted(_CG_SEEDS))
def test_a_seeded_fault_fails_the_cg_part_of_the_gradient_goal(name, tmp_path):
    """A wrong ghost fails both parity checks and every true residual of the
    sharded side, by orders of magnitude, and leaves the unsharded side
    passing; the unshifted operator fails the convergence checks of both
    sides by orders of magnitude with its parity inside the limit: parity
    alone passed it, at every size of the session, until schema 7.

    Measured (the same on jax 0.10.2, 0.11.0 and 0.11.2), as multiples of
    the limits: ghosts from the wrong neighbours, parity 600 and 400, the
    sharded side's worst residual 9300; a ghost kept at the left global
    end, 111 and 81, and 595; at the right one, 41 and 35, and 155; the
    unshifted operator, residuals 1300 to 39000 after all 300 iterations,
    with its parity at 7.3e-6 and 5.6e-6 (0.02 of the limit)."""
    seed = _CG_SEEDS[name]
    rp = _cg_seeded_runner(tmp_path, name)
    cg, checks = _cg_part(rp)
    status = {n: rp.check_status(c) for n, c in checks.items()}

    def over(check_name: str, limit: float, times: float) -> bool:
        value = checks[check_name]["value"]
        return status[check_name] == "failed" and not value <= times * limit   # NaN is over

    residual_limit = rp.LIMITS["krylov_residual"]
    if seed.sharded_only:
        assert all(over(n, rp.LIMITS["krylov"], seed.parity_over) for n in _CG_PARITY), checks
        _stopped, *residuals = _cg_names("sharded")
        assert all(status[n] == "failed" for n in residuals), status
        assert any(over(n, residual_limit, 100) for n in residuals), checks
        assert all(status[n] == "passed" for n in _cg_names("unsharded")), status
    else:
        for side in ("sharded", "unsharded"):
            stopped, *residuals = _cg_names(side)
            assert status[stopped] == "failed" and cg["solve"][side]["iterations"] == _CG_CAP
            assert all(over(n, residual_limit, 100) for n in residuals), checks
        # ... and nothing else would have said so
        assert all(status[n] == "passed" for n in _CG_PARITY), (status, checks)


def _run_gradient_goal(tmp: Path, runner: Path) -> tuple[int, dict, str]:
    out = tmp / "out"
    proc = run_pod(runner, ["--goal", "gradient", "--dry-run", "--out", out],
                   pythonpath=str(_PKG.parent), timeout=900, n_devices=_N_DEV)
    (doc,) = _runner()._load_results(out, "gradient")
    return proc.returncode, doc, proc.stdout[-4000:] + proc.stderr[-2000:]


# Per push: tests/cloud/multigpu/test_run_pod_seeded_faults.py::test_a_seeded_fault_fails_the_cg_part_of_the_gradient_goal
# and tests/cloud/multigpu/test_run_pod_seeded_faults.py::test_the_cg_part_of_the_gradient_goal_passes_on_solves_that_converged
@pytest.mark.slow
@pytest.mark.parametrize("name", [None, *sorted(_CG_SEEDS)], ids=["no_seed", *sorted(_CG_SEEDS)])
def test_the_gradient_goal_itself_fails_in_a_dry_run_on_each_seeded_cg_fault(name, tmp_path):
    """The goal as the session's dry run starts it (both sizes, the default
    cap), on a scratch copy of the runner: it exits 0 unseeded and 1 on
    every seed, and every failed check is one of the ``sharded_cg`` part's
    -- of the sharded side and the parity for a wrong ghost, of both sides'
    convergence and not the parity for the unshifted operator."""
    rp = _runner()
    runner = _RUNNER if name is None else Path(_cg_seeded_runner(tmp_path, name).__file__)
    rc, doc, log = _run_gradient_goal(tmp_path, runner)
    failed = [c["name"] for c in doc["checks"] if rp.check_status(c) == "failed"]
    if name is None:
        assert rc == 0 and not failed and rp.goal_verdict([doc]) == "PASS", log
        assert rp.record_problems(doc) == []
        return
    assert rc == 1, log
    assert failed and all(" dof sharded_cg " in n for n in failed), failed
    assert any(n.startswith(f"{_CG_DOF} dof") for n in failed), failed
    if _CG_SEEDS[name].sharded_only:
        assert not [n for n in failed if " sharded_cg unsharded " in n], failed
    else:
        assert not [n for n in failed if " vs unsharded " in n], failed
    assert "CHECK FAILED" in log
