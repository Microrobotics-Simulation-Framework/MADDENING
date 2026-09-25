"""``run_pod.py``'s checklist goals fail on a broken stencil wrapper.

A goal that compares a sharded run with an unsharded one proves the
wrapper right only if a wrong wrapper makes it fail.  Until 0.4.0 the
``stencil``, ``hybrid`` and ``coupled`` goals passed -- and a dry run
relabelled as four real GPUs closed checklist items 1, 3, 4 and 6 -- with
``ShardedStencilNode`` broken in each of the four ways below, every one of
which its unit tests catch: the goals read a static only in the interior,
fed no grid-shaped input, declared no domain integral and ran on a 1-D
mesh only.  Each seed here is that fault, applied to a scratch copy of the
library (never to the tree under test), and the goals that run the
wrapper must fail on it.

The seeds are exact text of ``sharded_node.py``.  When that code changes
the per-push test below fails, naming the seed: update the seed to the new
code rather than dropping it.
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
from typing import NamedTuple

import pytest

import maddening
from tests.cloud.multigpu.run_pod_support import checked_argv, offline_env, run_pod

_PKG = Path(maddening.__file__).resolve().parent
_RUNNER = _PKG.parents[1] / "benchmarks" / "multigpu" / "run_pod.py"
_N_DEV = 4
_WRAPPER = "cloud/multigpu/sharded_node.py"

#: The checklist goals that run ``ShardedStencilNode``; every seed must fail
#: each of them.  ``indivisible`` builds the wrapper only to see it refuse,
#: and ``halo`` calls ``halo_exchange`` directly: both must still pass.
_WRAPPER_GOALS = ("stencil", "hybrid", "coupled")
_OTHER_GOALS = ("indivisible", "halo")


class Seed(NamedTuple):
    old: str
    new: str
    #: A word every failed check of the ``stencil`` goal must contain, when
    #: the fault is confined to some cases (``" 2d "``: the pencil mesh).
    only_in: str = ""


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
    "later_axes_are_zero_at_the_global_edges": Seed(
        "                arr2 = halo_exchange(\n"
        "                    arr2, mesh=mesh, axes=exchange_axes, boundary=boundary,\n"
        "                )\n",
        "                arr2 = halo_exchange(  # SEEDED FAULT: axes after the first zero-filled\n"
        "                    arr2, mesh=mesh, axes=exchange_axes,\n"
        "                    boundary={**{a[0]: 'zero' for a in exchange_axes[1:]},\n"
        "                              exchange_axes[0][0]: boundary},\n"
        "                )\n",
        only_in=" 2d "),
}


def _runner():
    spec = importlib.util.spec_from_file_location("run_pod_seeded_under_test", _RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seeded_source(seed: Seed) -> str:
    text = (_PKG / _WRAPPER).read_text(encoding="utf-8")
    assert text.count(seed.old) == 1, (
        f"the seed no longer matches {_WRAPPER} exactly once ({text.count(seed.old)} "
        "matches): the wrapper changed; update the seed to the new code")
    return text.replace(seed.old, seed.new)


# --- per push -----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_SEEDS))
def test_every_seed_applies_once_to_the_wrapper_as_it_stands(name):
    """A seed that matches nothing seeds nothing: the slow test would still
    pass its control and fail its seeded run for another reason -- or, run
    against an older wrapper, test code that no longer exists."""
    seeded = _seeded_source(_SEEDS[name])
    assert "SEEDED FAULT" in seeded
    compile(seeded, _WRAPPER, "exec")


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


@pytest.fixture(scope="module")
def seeded_wrapper_classes():
    """``{seed: ShardedStencilNode}`` built from each seeded source, in-process.

    Executed with ``@stability`` made a no-op, under a module name outside
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
            module = types.ModuleType(f"run_pod_seeded_wrapper_{name}")
            exec(compile(_seeded_source(seed), f"<seeded {name}>", "exec"),  # noqa: S102
                 module.__dict__)
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
    ``Field2D`` case (8 x 8, its source fed), sharded by ``wrapper_class``
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
    by the real wrapper, within it.  The pencil-mesh fault is looked for
    on the pencil mesh, the only one it can touch."""
    rp = rp_backend
    seed = _SEEDS.get(name)
    mesh_label = "2d" if seed is not None and seed.only_in == " 2d " else "1d"
    wrapper = rp.ShardedStencilNode if name is None else seeded_wrapper_classes[name]
    rel = _one_step_forward(rp, wrapper, mesh_label)
    limit = rp.LIMITS["forward"]
    if name is None:
        assert all(v <= limit for v in rel.values()), rel
        assert all(v <= limit for v in _one_step_forward(rp, wrapper, "2d").values())
    else:
        assert any(not v <= limit for v in rel.values()), rel


# --- slow: the goals themselves, as the session runs them ---------------------


def _scratch_library(tmp: Path, seed: Seed | None) -> Path:
    src = tmp / "src"
    shutil.copytree(_PKG, src / "maddening", ignore=shutil.ignore_patterns("__pycache__"))
    if seed is not None:
        (src / "maddening" / _WRAPPER).write_text(_seeded_source(seed), encoding="utf-8")
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


@pytest.mark.slow
def test_the_checklist_goals_pass_on_the_scratch_copy_of_the_library(tmp_path):
    """The control: the scratch copy with no seed passes every goal, so a
    seeded run that fails does so because of its seed."""
    rc, docs, log = _run_checklist(tmp_path, _scratch_library(tmp_path, None))
    rp = _runner()
    assert rc == 0, log
    assert {g: rp.goal_verdict(d) for g, d in docs.items()} == {
        g: "PASS" for g in rp.CHECKLIST_GOALS}, log


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(_SEEDS))
def test_a_seeded_wrapper_fault_fails_every_goal_that_runs_the_wrapper(name, tmp_path):
    seed = _SEEDS[name]
    rc, docs, log = _run_checklist(tmp_path, _scratch_library(tmp_path, seed))
    rp = _runner()
    verdicts = {g: rp.goal_verdict(d) for g, d in docs.items()}
    assert rc == 1, log
    assert verdicts == {**{g: "FAIL" for g in _WRAPPER_GOALS},
                        **{g: "PASS" for g in _OTHER_GOALS}}, (verdicts, log)
    for goal in _WRAPPER_GOALS:
        (doc,) = docs[goal]
        assert rp.record_problems(doc) == [], (goal, rp.record_problems(doc))
    if seed.only_in:
        (stencil,) = docs["stencil"]
        failed = [c["name"] for c in stencil["checks"] if rp.check_status(c) == "failed"]
        assert failed and all(seed.only_in in c for c in failed), failed
    # What --summarise would say had the same checks failed the same way on
    # four GPUs: every item a wrapper goal decides has FAILED.
    status = {i: s for i, (s, _) in rp.checklist_status(_as_four_gpus(docs)).items()}
    assert {i: status[i] for i in (1, 3, 4, 6)} == {i: "FAILED" for i in (1, 3, 4, 6)}, status
    assert status[2] == status[5] == "CLOSED", status
    json.dumps(status)                                # (the verdicts are plain data)
