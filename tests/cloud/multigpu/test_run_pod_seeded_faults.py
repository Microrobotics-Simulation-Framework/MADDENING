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

The last three are faults of a halo two cells wide, which no node of the
goals had until schema 10 (``Field2D`` and the lattice read one cell): a
wrapper that zero-filled every wider halo at the global edges passed all
eight goals.  The ``stencil`` goal now runs ``Field2D(reach=2)`` under each
boundary on every mesh (its ``field-halo2`` cases), and each of these must
fail every one of those cases its fault can reach, and no case with a halo
of one.

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

Nor did any seed reach the ``forward`` goal, which runs
``ShardedUnstructuredNode``.  Its cross-shard sum is checked exactly, on a
field of ones the node leaves a field of ones: the step's ``total`` is then
the number of cells in any order of addition.  Its seeds (``_COUNT_SEEDS``)
are exact text of the unstructured wrapper's reduction and of the runner's
node: a shard left out of the reduction or counted twice, ghost rows
counted, the node's ``own`` mask ignored at a size that has pad rows, each
of which must read off by its number of cells; and a step that does not
keep ones ones, which must fail the precondition's own check.

And two documented paths of ``ShardedUnstructuredNode`` were on no goal's
path at all: the goals' node read no boundary input and took no ``params``,
so a wrapper that zeroed every per-cell input, or doubled every injected
parameter, passed all eight goals and closed all six items.  The node now
reads a per-cell source and takes its weight from the injected ``params``
in every goal that builds it, and the ``gradient`` goal differentiates with
respect to both.  Their seeds (``_PATH_SEEDS``) are exact text of the
wrapper: those two, an input whose ghost rows are not exchanged or whose
rows are one off, a parameter that reaches one shard only -- each of which
must fail ``indivisible``, ``forward`` and ``gradient`` -- and an input or
a parameter that arrives right and loses its cotangent, which only the
``gradient`` goal can see and must.
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
    #: The ``stencil`` goal's node (``run_pod.STENCIL_NODES``) whose cases
    #: the fault is confined to; ``"field"``: every node's.  A fault of
    #: halos two wide is ``"field-halo2"``'s alone, and the ``stencil``
    #: goal's: ``hybrid`` and ``coupled`` build ``Field2D`` with a halo of
    #: one.
    kind: str = "field"
    #: The boundaries (``run_pod.STENCIL_BOUNDARIES``) the fault is
    #: confined to (empty: any), and the one the per-push test looks under.
    only_under: tuple = ()
    boundary: str = "periodic"
    #: The fault is that node's alone: the plain ``field`` case under the
    #: same boundary still agrees.  ``False``: a fault of the boundary, which
    #: any node under it shows (the zero fill, which only ``field-shares``
    #: runs under in the goal).
    confined_to_kind: bool = True


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
    # The three below are faults of a halo two cells wide.  The first
    # passed every goal until schema 10: the state's and a grid input's
    # global-edge halos zero whenever a halo is two or more cells wide.
    # Under "dirichlet" the node overwrites the field's with the wall, and
    # the source's -- the wrapper's own fill -- shows it.
    "halos_two_wide_are_zero_at_the_global_edges": Seed(
        "                    arr2, mesh=mesh, axes=exchange_axes, boundary=boundary,\n",
        "                    arr2, mesh=mesh, axes=exchange_axes,\n"
        '                    boundary=("zero" if any(h >= 2 for _, _, h in exchange_axes) '
        "else boundary),  # SEEDED FAULT\n",
        fails=("stencil",), kind="field-halo2"),
    # The second halo cell taken from the first: every halo that comes
    # from a neighbouring shard is its innermost cell repeated.
    "outer_halo_cells_copy_the_innermost_one": Seed(
        "    right_halo = lax.ppermute(left_slice, mesh_axis, perm_backward)\n",
        "    right_halo = lax.ppermute(left_slice, mesh_axis, perm_backward)\n"
        "    if halo >= 2:  # SEEDED FAULT: each outer halo cell copies the innermost one\n"
        "        left_halo = jnp.broadcast_to(lax.slice_in_dim(\n"
        "            left_halo, halo - 1, halo, axis=spatial_axis), left_halo.shape)\n"
        "        right_halo = jnp.broadcast_to(lax.slice_in_dim(\n"
        "            right_halo, 0, 1, axis=spatial_axis), right_halo.shape)\n",
        file=_HALO, fails=("halo", "stencil"), kind="field-halo2"),
    # A two-cell halo exchanged across shards and filled by the one-cell
    # rule at a global edge: the block's two outermost cells in order, not
    # its edge cell twice (what "edge" did before 0.4.0, MADD-ANO-029).
    "wide_edge_halos_repeat_the_block_not_its_edge_cell": Seed(
        "    if halo == 1:\n        return left_slice, right_slice\n",
        "    if halo >= 1:  # SEEDED FAULT: the one-cell rule at every width\n"
        "        return left_slice, right_slice\n",
        file=_HALO, fails=("halo", "stencil"), kind="field-halo2",
        only_under=("edge", "dirichlet"), boundary="edge"),
    # The five below are on the path of the ``stencil`` goal's ``field-shares``
    # cases alone (schema 11): the wrapper's zero fill, which no goal ran,
    # and domain integrals the node asks it not to sum over every mesh axis.
    "the_zero_fill_is_the_edge_fill": Seed(
        "        self._boundary = boundary\n",
        '        self._boundary = "edge" if boundary == "zero" else boundary  # SEEDED FAULT\n',
        fails=("stencil",), kind="field-shares", only_under=("zero",), boundary="zero",
        confined_to_kind=False),
    "zero_filled_halos_are_ones": Seed(
        "        zeros = jnp.zeros_like(left_slice)\n",
        "        zeros = jnp.ones_like(left_slice)  # SEEDED FAULT\n",
        file=_HALO, fails=("halo", "stencil"), kind="field-shares", only_under=("zero",),
        boundary="zero", confined_to_kind=False),
    # An integral declared as one value a shard, summed over the mesh all the
    # same: every slot of the stack holds the total.
    "per_shard_integrals_are_summed_over_every_axis": Seed(
        "                    red = lax.psum(v, axis_name=reduce_axes) if reduce_axes else v\n",
        "                    red = lax.psum(v, axis_name=reduce_axes + unreduced)  # SEEDED FAULT\n",
        fails=("stencil",), kind="field-shares", boundary="zero"),
    # Each shard's value stacked into the next shard's slot.
    "per_shard_integrals_land_in_the_next_shards_slot": Seed(
        "                    out[k] = red[(None,) * len(unreduced)] if unreduced else red\n",
        "                    if unreduced:  # SEEDED FAULT: each value from the shard before\n"
        "                        _n = int(mesh.shape[unreduced[-1]])\n"
        "                        red = lax.ppermute(red, unreduced[-1],\n"
        "                                           [(s, (s + 1) % _n) for s in range(_n)])\n"
        "                    out[k] = red[(None,) * len(unreduced)] if unreduced else red\n",
        fails=("stencil",), kind="field-shares", boundary="zero"),
    # An integral carried in the state reaches ``update_padded`` as zeros.
    "integrals_in_the_state_are_zeroed_on_the_way_in": Seed(
        "                f: (arr if f in integrals else _pad_like_state(arr))\n",
        "                f: (arr * 0 if f in integrals else _pad_like_state(arr))  # SEEDED FAULT\n",
        fails=("stencil",), kind="field-shares", boundary="zero"),
}


def _boundaries_of(rp, kind: str) -> tuple:
    """The boundaries the ``stencil`` goal runs node ``kind`` under."""
    return tuple(dict.fromkeys(b for _n, k, b, _m in rp.stencil_cases([64], _N_DEV)
                               if k == kind))


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
    text = _RUNNER.read_text(encoding="utf-8")
    for name, seed in _SEEDS.items():
        assert set(seed.fails) <= set(rp.CHECKLIST_GOALS), name
        assert set(seed.only_in) <= meshes and seed.mesh in meshes, name
        assert not seed.only_in or seed.mesh in seed.only_in, name
        assert seed.kind in rp.STENCIL_NODES, name
        assert set(seed.only_under) <= set(_boundaries_of(rp, seed.kind)), name
        assert seed.boundary in (seed.only_under or _boundaries_of(rp, seed.kind)), name
        assert seed.confined_to_kind or seed.kind != "field", name
        if seed.kind == "field":
            assert set(_WRAPPER_GOALS) <= set(seed.fails), name
        else:
            assert set(seed.fails) - {"halo"} == {"stencil"}, name
    # ... and a halo of two is the stencil goal's alone: one call builds
    # Field2D with a reach, for the stencil goal's cases
    builds = [line for line in text.splitlines() if "reach=" in line and "``" not in line]
    assert len(builds) == 1, builds
    assert "reach=2 if kind ==" in text.split("def _stencil_pair(")[1].split("\ndef ")[0]


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


def _one_step_forward(rp, wrapper_class, mesh_label: str, kind: str = "field",
                      boundary: str = "periodic") -> dict:
    """``{field: max_rel}`` of one step of a ``Field2D`` case of the
    ``stencil`` goal (8 x 12, its source fed; ``kind`` and ``boundary`` as
    the goal names them), sharded by ``wrapper_class`` on ``mesh_label``,
    against the unsharded node."""
    ny, nx = rp.field_shape(64, _N_DEV)
    mesh, axis_map, _ = rp.stencil_mesh(mesh_label, _N_DEV)
    real = rp.ShardedStencilNode
    rp.ShardedStencilNode = wrapper_class
    try:
        ref, sharded, bi, p = rp._stencil_pair(kind, ny, nx, boundary, mesh, axis_map)
    finally:
        rp.ShardedStencilNode = real
    assert ref.halo_width() == sharded.halo_width() == dict.fromkeys(
        (0, 1), 2 if kind == "field-halo2" else 1)
    start = dict(ref.initial_state())
    got = {label: rp._host(node.update(state, bi, ref.delta_t, params={"diffusivity": p}))
           for label, node, state in (("unsharded", ref, start),
                                      ("sharded", sharded, rp._placed_like(sharded, start)))}
    return {field: rp._diff(got["sharded"][field], got["unsharded"][field])["max_rel"]
            for field in rp.STENCIL_NODES[kind]["fields"]}


@pytest.mark.parametrize("name", [None, *sorted(_SEEDS)],
                         ids=["no_seed", *sorted(_SEEDS)])
def test_one_step_of_the_stencil_goals_node_shows_each_seeded_fault(
        name, rp_backend, seeded_wrapper_classes):
    """The per-push half of the slow test below: one step of the
    ``stencil`` goal's own node and inputs, sharded by the seeded wrapper,
    is off the unsharded step by more than the goal's forward limit -- and
    by the real wrapper, within it on every mesh.  A fault confined to some
    meshes is looked for on one of them (``Seed.mesh``), and one confined
    to halos two wide in the ``field-halo2`` case, under ``Seed.boundary``
    -- where the case with a halo of one must still agree."""
    rp = rp_backend
    limit = rp.LIMITS["forward"]
    if name is None:
        for mesh_label in rp.meshes_that_fit(_N_DEV):
            rel = _one_step_forward(rp, rp.ShardedStencilNode, mesh_label)
            assert all(v <= limit for v in rel.values()), (mesh_label, rel)
        return
    seed, wrapper = _SEEDS[name], seeded_wrapper_classes[name]
    if seed.raises:
        with pytest.raises(Exception):
            _one_step_forward(rp, wrapper, seed.mesh)
        return
    rel = _one_step_forward(rp, wrapper, seed.mesh, seed.kind, seed.boundary)
    assert any(not v <= limit for v in rel.values()), rel
    if seed.kind != "field":
        rel = _one_step_forward(rp, wrapper, seed.mesh, "field", seed.boundary)
        if seed.confined_to_kind:
            assert all(v <= limit for v in rel.values()), rel
        else:                              # the boundary's fault, whatever the node
            assert any(not v <= limit for v in rel.values()), rel
        for boundary in set(rp.STENCIL_BOUNDARIES) - set(seed.only_under or
                                                         rp.STENCIL_BOUNDARIES):
            rel = _one_step_forward(rp, wrapper, seed.mesh, seed.kind, boundary)
            assert all(v <= limit for v in rel.values()), (boundary, rel)


@pytest.mark.parametrize("boundary", ["periodic", "edge", "dirichlet"])
def test_one_step_with_a_halo_of_two_agrees_on_every_mesh_under_each_boundary(
        boundary, rp_backend):
    """The control of the seeds of halos two wide: one step of the
    ``stencil`` goal's ``field-halo2`` node, sharded by the wrapper as it
    stands, is the unsharded step within the goal's forward limit on every
    mesh -- 8 x 12 cells, so on the 1-D mesh a shard's two rows are all
    halo to its neighbours."""
    rp = rp_backend
    assert boundary in rp.STENCIL_BOUNDARIES and len(rp.STENCIL_BOUNDARIES) == 3
    for mesh_label in rp.meshes_that_fit(_N_DEV):
        rel = _one_step_forward(rp, rp.ShardedStencilNode, mesh_label, "field-halo2", boundary)
        assert all(v <= rp.LIMITS["forward"] for v in rel.values()), (mesh_label, rel)


def test_one_step_under_the_zero_fill_with_integrals_a_shard_agrees_on_every_mesh(rp_backend):
    """The control of the five seeds of the ``field-shares`` cases: one step
    under the wrapper's zero fill, with one integral stacked a shard and
    one summed over the first mesh axis only, is the unsharded step slot
    by slot on every mesh -- and ``"zero"`` is the one boundary the goal
    runs that node under, on every mesh."""
    rp = rp_backend
    assert _boundaries_of(rp, "field-shares") == (rp.SHARES_BOUNDARY,) == ("zero",)
    assert {m for _n, k, _b, m in rp.stencil_cases([64], _N_DEV) if k == "field-shares"} == set(
        rp.meshes_that_fit(_N_DEV))
    for mesh_label in rp.meshes_that_fit(_N_DEV):
        rel = _one_step_forward(rp, rp.ShardedStencilNode, mesh_label, "field-shares", "zero")
        assert set(rel) == {"f", "averages", "shares", "strips"}
        assert all(v <= rp.LIMITS["forward"] for v in rel.values()), (mesh_label, rel)


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
    (stencil,) = docs["stencil"]
    failed = [c["name"] for c in stencil["checks"] if rp.check_status(c) == "failed"]
    if seed.only_in:
        assert failed and all(any(f" {m} " in c for m in seed.only_in) for c in failed), failed
    if seed.kind != "field":
        # A fault of halos two wide: every failed check is one of that
        # node's, under a boundary the fault can reach -- and every such
        # case on every mesh has one, so each of them guards the path.
        assert failed and all(c.startswith(f"{seed.kind} ") for c in failed), failed
        under = seed.only_under or _boundaries_of(rp, seed.kind)
        assert all(c.split(":")[0].split(" vs ")[0].split()[4] in under for c in failed), failed
        for mesh in seed.only_in or rp.meshes_that_fit(_N_DEV):
            for boundary in under:
                assert any(c.startswith(f"{seed.kind} {mesh} ") and f" {boundary} " in c
                           for c in failed), (mesh, boundary, failed)
    # What --summarise would say had the same checks failed the same way on
    # four GPUs: every item a failing goal decides has FAILED, the rest closed.
    status = {i: s for i, (s, _) in rp.checklist_status(_as_four_gpus(docs)).items()}
    failing = {i for i, (_c, goals) in rp.CHECKLIST.items() if set(goals) & set(seed.fails)}
    assert ({1, 3, 4, 6} if seed.kind == "field" else {1, 3}) <= failing
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
        # The flag is information (the float32 residual against ``rtol``
        # with no allowance, MADD-ANO-263): the count and the float64
        # residuals below are what the goal holds the solve to.
        assert isinstance(solve["converged"], bool)
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


# --- the forward goal's exact count of its cells -------------------------------

_UNSTRUCTURED = "cloud/multigpu/sharded_unstructured.py"
_PSUM = "                        out[k] = lax.psum(v, axis_name=mesh_axis)\n"
_TOTAL = "            total = jnp.sum(new * own)\n"
_SHARD_INFO = "                0: (idx * n_local_max, n_local_max),\n"
_OWNED = '                "n_local": jnp.asarray(n_local_per_shard)[idx],\n'
_STACKED = "                        out[k] = v[None]          # stacked along the mesh axis\n"
_RELAX = "            new = own + w * (jnp.take(x, tbl, axis=0).mean(axis=-1) - own)\n"


class CountSeed(NamedTuple):
    old: str
    new: str
    #: The library file the seed is applied to, under ``maddening/``;
    #: ``None``: the runner itself (its node under test).
    file: str | None = None


_COUNT_SEEDS = {
    "shard_1_left_out_of_the_reduction": CountSeed(
        _PSUM,
        "                        out[k] = lax.psum(jnp.where(idx == 1, 0.0, v), "
        "axis_name=mesh_axis)  # SEEDED FAULT\n", _UNSTRUCTURED),
    "shard_1_counted_twice": CountSeed(
        _PSUM,
        "                        out[k] = lax.psum(jnp.where(idx == 1, 2.0 * v, v), "
        "axis_name=mesh_axis)  # SEEDED FAULT\n", _UNSTRUCTURED),
    # The ghost tail of the slab the step was handed, summed into the total.
    "ghost_rows_counted": CountSeed(
        _TOTAL,
        '            total = (jnp.sum(new * own)\n'
        '                     + jnp.sum(state_padded["x"][n_local:]))  # SEEDED FAULT\n'),
    # Pad rows counted: a pad row starts at zero and moves towards its
    # shard's first cell by the injected weight in the step, so each reads
    # ``NODE_WEIGHT`` of a cell (a quarter).
    "own_mask_ignored": CountSeed(
        _TOTAL, "            total = jnp.sum(new)  # SEEDED FAULT\n"),
    # One ulp over 1 in every cell, on both sides: the field parity cannot
    # tell, and a limit of 1e-5 on the precondition would pass it.  (The
    # mean is 2**-21 over 1, and the injected weight is a quarter.)
    "a_step_that_does_not_keep_ones": CountSeed(
        _RELAX,
        "            new = own + w * (jnp.take(x, tbl, axis=0).mean(axis=-1) * (1.0 + 2.0 ** -21)"
        " - own)  # SEEDED FAULT\n"),
    # The four below leave the field and its total right: only the node's
    # ``per_shard`` integral, which the wrapper stacks a shard, reads them
    # (schema 11; no goal read ``shard_info``'s offset or cell count, or an
    # integral not summed over the mesh, before it).
    "shard_info_offset_is_zero_on_every_shard": CountSeed(
        _SHARD_INFO, "                0: (0 * idx, n_local_max),  # SEEDED FAULT\n", _UNSTRUCTURED),
    # What a node was told before 0.4.0: the padded block length, nothing
    # about its pad rows.  Right wherever no shard has one.
    "shard_info_n_local_is_the_padded_length": CountSeed(
        _OWNED, '                "n_local": jnp.asarray(n_local_max, jnp.int32) + 0 * idx,'
        "  # SEEDED FAULT\n", _UNSTRUCTURED),
    "unreduced_integrals_are_summed_over_the_mesh": CountSeed(
        _STACKED, "                        out[k] = lax.psum(v, axis_name=mesh_axis)[None]"
        "  # SEEDED FAULT\n", _UNSTRUCTURED),
    "unreduced_integrals_land_in_the_next_shards_slot": CountSeed(
        _STACKED, "                        out[k] = lax.ppermute(v, mesh_axis, [(s, (s + 1) % "
        "layout.n_devices) for s in range(layout.n_devices)])[None]  # SEEDED FAULT\n",
        _UNSTRUCTURED),
}

#: The seeds only the per-shard check can see, and how many of its ``2 * D``
#: entries each puts off the layout's, from the layout: every offset but
#: shard 0's; the count of each shard with a pad row (a quarter of a cell
#: over: the injected weight); every entry; every offset, and the count
#: of each shard that owns another number of cells than the one before it.
_PER_SHARD_SEEDS = {
    "shard_info_offset_is_zero_on_every_shard": lambda lay: lay.n_devices - 1,
    "shard_info_n_local_is_the_padded_length":
        lambda lay: sum(int(k) != lay.n_local_max for k in lay.n_local),
    "unreduced_integrals_are_summed_over_the_mesh": lambda lay: 2 * lay.n_devices,
    "unreduced_integrals_land_in_the_next_shards_slot":
        lambda lay: lay.n_devices + sum(int(lay.n_local[d]) != int(lay.n_local[d - 1])
                                        for d in range(lay.n_devices)),
}

#: 225 = 15 x 15 cells do not divide by four: three of the four shards
#: carry a pad row (57, 56, 56, 56 cells under reverse Cuthill-McKee).  The
#: dry run's own 256 and 1024 divide, as 300,304 and 1,000,000 do, and have
#: none: the ``own`` mask masks nothing there.
_COUNT_CELLS = (225, 256)
_ONES_UNSHARDED = "unsharded"


def _count_seeded_source(seed: CountSeed) -> str:
    path = _RUNNER if seed.file is None else _PKG / seed.file
    text = path.read_text(encoding="utf-8")
    assert text.count(seed.old) == 1, (
        f"the seed no longer matches {path.name} exactly once ({text.count(seed.old)} "
        "matches): the code changed; update the seed to the new code")
    return text.replace(seed.old, seed.new)


@pytest.mark.parametrize("name", sorted(_COUNT_SEEDS))
def test_every_count_seed_applies_once_to_the_code_as_it_stands(name):
    seeded = _count_seeded_source(_COUNT_SEEDS[name])
    assert "SEEDED FAULT" in seeded
    compile(seeded, name, "exec")


def _count_seeded_runner(tmp: Path, name: str | None):
    """The runner as a module with one seed in force: a scratch copy of it
    for a seed of its node; the runner itself, with the wrapper's seeded
    class in its namespace, for a seed of the wrapper (executed under a
    module name outside the package with ``@stability`` a no-op, as
    ``seeded_wrapper_classes`` does and for its reason)."""
    if name is None:
        return _runner()
    seed = _COUNT_SEEDS[name]
    if seed.file is None:
        path = tmp / f"run_pod_count_{name}.py"
        path.write_text(_count_seeded_source(seed), encoding="utf-8")
        spec = importlib.util.spec_from_file_location(f"run_pod_count_seeded_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return _runner_with_unstructured_wrapper(name, _count_seeded_source(seed))


def _runner_with_unstructured_wrapper(name: str, source: str):
    """The runner as a module of its own, with the ``ShardedUnstructuredNode``
    of ``source`` (the wrapper's seeded text) in its namespace."""
    from maddening.core.compliance import stability as stab

    before = dict(stab._STABILITY_REGISTRY)
    real = stab.stability
    stab.stability = lambda level: (lambda obj: obj)
    try:
        wrapper = _exec_module(f"run_pod_seeded_unstructured_{name}", source)
    finally:
        stab.stability = real
    assert stab._STABILITY_REGISTRY == before, "a seeded copy registered a stable surface"
    rp = _runner()
    rp._load_backend()
    rp.ShardedUnstructuredNode = wrapper.ShardedUnstructuredNode
    return rp


def _forward_part(rp) -> tuple[dict, dict, dict, list]:
    """``(results by size, {check name: check}, layout by size, the sharded
    nodes built)`` of the ``forward`` goal run in-process as the goal runs
    it, one step and one timed call, at ``_COUNT_CELLS``."""
    import jax

    if len(jax.devices()) < _N_DEV:
        pytest.skip(f"needs >= {_N_DEV} devices")
    rp._load_backend()
    built, layouts, real = [], {}, rp.build_pair

    def recording(n, edges, mesh, pa, method):
        ref, sharded, layout = real(n, edges, mesh, pa, method)
        built.append(sharded)
        layouts[n] = layout
        return ref, sharded, layout

    rp.build_pair = recording
    try:
        args = SimpleNamespace(mesh=None, synthetic="grid", partition="auto",
                               cells=list(_COUNT_CELLS), n_devices=_N_DEV, steps=1, warmup=0,
                               repeats=1)
        doc = rp.run_forward(args, {})
    finally:
        rp.build_pair = real
    checks = {c["name"]: c for c in doc["checks"]}
    assert len(checks) == len(doc["checks"]) == 14 * len(_COUNT_CELLS)
    return {r["cells"]: r for r in doc["results"]}, checks, layouts, built


def _per_shard_name(cells: int, method: str) -> str:
    return f"{cells} cells {method} ones: per-shard (offset, cells) entries off the layout's"


def _ones_names(cells: int, side: str) -> tuple[str, str]:
    """The names of one side's count and of its precondition."""
    return (f"{cells} cells {side} ones: |total - cells|",
            f"{cells} cells {side} ones: max |x - 1| after the step")


def test_the_forward_goal_counts_a_field_of_ones_exactly_on_both_sides(tmp_path):
    """The control of the seeds below, and the claim: one public step from
    a field of ones returns ``total`` equal to the number of cells and a
    field still all ones -- sharded under both transports and unsharded,
    at a size with pad rows and at one without -- held to a limit of
    exactly zero; and the step runs the program the goal had compiled.
    (The goal's own function, in-process on four CPU virtual devices.)
    The float total of the goal's own field is recorded and must be
    finite, and is held to no limit: its sum cancels, and the order of
    addition decides how two float32 sums of it compare."""
    rp = _count_seeded_runner(tmp_path, None)
    results, checks, layouts, built = _forward_part(rp)
    assert {n: rp.check_status(c) for n, c in checks.items()} == {n: "passed" for n in checks}
    assert rp.LIMITS["exact"] == 0.0
    exact = {"total": None, "max_abs_from_one": 0.0, "entries_not_one": 0}
    for cells in _COUNT_CELLS:
        entry = results[cells]
        assert entry["ones_unsharded"] == {**exact, "total": float(cells)}
        for method in rp.METHODS:
            m = entry["methods"][method]
            layout = layouts[cells]
            assert m["ones"] == {**exact, "total": float(cells), "per_shard": [
                [float(d * layout.n_local_max), float(k)] for d, k in enumerate(layout.n_local)]}
            c = checks[_per_shard_name(cells, method)]
            assert (c["value"], c["limit"], c["sense"]) == (0.0, 0.0, "<="), c
            assert set(m["parity_total"]) == {"max_abs", "max_rel", "reference_scale", "finite"}
            assert f"{cells} cells {method} total vs unsharded max_rel" not in checks
            assert checks[f"{cells} cells {method} total vs unsharded finite"]["value"] is True
        for side in (*rp.METHODS, _ONES_UNSHARDED):
            for name in _ones_names(cells, side):
                c = checks[name]
                assert (c["value"], c["limit"], c["sense"]) == (0.0, 0.0, "<="), c
    # the sizes are what the seeds need: pad rows at the first, none at the second
    pads = {cells: sum(layout.n_local_max - k for k in layout.n_local)
            for cells, layout in layouts.items()}
    assert pads[_COUNT_CELLS[0]] > 0, pads
    # one compiled program a wrapper, the step from ones included
    assert len(built) == len(_COUNT_CELLS) * len(rp.METHODS)
    for sharded in built:
        (step,) = sharded._sharded_cache.values()
        assert step._cache_size() == 1


@pytest.mark.parametrize("name", sorted(_COUNT_SEEDS))
def test_a_seeded_miscount_fails_the_forward_goals_exact_count(name, tmp_path):
    """Each fault reads off by its number of cells, exactly, on the sharded
    side under both transports, and nothing else of the goal says so: the
    field agrees with the unsharded one entry by entry and the float total
    is finite.  Measured (reverse Cuthill-McKee, four shards; the same on
    jax 0.10.2, 0.11.0 and 0.11.2), at 225 and 256 cells: shard 1 left out
    or counted twice, 56 and 64; ghost rows counted, 76 and 78; the mask
    ignored, 0.75 (three pad rows, a quarter of a cell each: the injected
    weight) and 0 -- at 256 cells there is no pad row and nothing is
    miscounted.  And at the session's sizes: a shard 25,122, 75,076 and
    250,000 cells; ghost rows 1,532, 2,646 and 4,830; the mask 0.75 at
    100,489 cells and 0 at 300,304 and 1,000,000.  A step that does not
    keep ones ones fails the precondition on every side, by one ulp of 1."""
    seed = _COUNT_SEEDS[name]
    rp = _count_seeded_runner(tmp_path, name)
    results, checks, layouts, _built = _forward_part(rp)
    status = {n: rp.check_status(c) for n, c in checks.items()}
    others = [n for n in checks if " ones: " not in n]
    assert others and all(status[n] == "passed" for n in others), status
    for cells in _COUNT_CELLS:
        layout = layouts[cells]
        count_un, still_un = _ones_names(cells, _ONES_UNSHARDED)
        if name in _PER_SHARD_SEEDS:
            # the field and its total are right on every side: the stacked
            # pairs alone are off, by the entries the layout says
            off = _PER_SHARD_SEEDS[name](layout)
            for side in (*rp.METHODS, _ONES_UNSHARDED):
                assert {status[n] for n in _ones_names(cells, side)} == {"passed"}
            for method in rp.METHODS:
                check = checks[_per_shard_name(cells, method)]
                assert check["value"] == off, (cells, method, check, off)
                assert status[_per_shard_name(cells, method)] == (
                    "failed" if off else "passed")
            assert off or name == "shard_info_n_local_is_the_padded_length"
            continue
        for method in rp.METHODS:              # a miscount leaves the pairs right
            if name not in ("a_step_that_does_not_keep_ones", "own_mask_ignored",
                            "ghost_rows_counted"):
                assert status[_per_shard_name(cells, method)] == "passed"
        if name == "a_step_that_does_not_keep_ones":
            for side in (*rp.METHODS, _ONES_UNSHARDED):
                _count, still = _ones_names(cells, side)
                assert status[still] == "failed" and checks[still]["value"] == 2.0 ** -23
                assert not checks[still]["value"] > 1e-5      # what a loosened limit passes
            continue
        off_by = {"shard_1_left_out_of_the_reduction": layout.n_local[1],
                  "shard_1_counted_twice": layout.n_local[1],
                  "ghost_rows_counted": sum(layout.n_ghost),
                  "own_mask_ignored": rp.NODE_WEIGHT * sum(layout.n_local_max - k
                                                           for k in layout.n_local),
                  }[name]
        assert status[count_un] == status[still_un] == "passed"
        for method in rp.METHODS:
            count, still = _ones_names(cells, method)
            assert checks[count]["value"] == off_by, (cells, method, checks[count], off_by)
            assert status[count] == ("failed" if off_by else "passed")
            assert status[still] == "passed" and checks[still]["value"] == 0.0
            total = results[cells]["methods"][method]["ones"]["total"]
            assert total == cells + (-off_by if "left_out" in name else off_by)
    if name in _PER_SHARD_SEEDS:
        # every one of them shows at the size with pad rows
        assert status[_per_shard_name(_COUNT_CELLS[0], rp.METHODS[0])] == "failed"
    elif name != "a_step_that_does_not_keep_ones":
        # every seed miscounts at the size with pad rows
        assert status[_ones_names(_COUNT_CELLS[0], rp.METHODS[0])[0]] == "failed"


def _run_forward_goal(tmp: Path, runner: Path, src: Path) -> tuple[int, dict, str]:
    out = tmp / "out"
    proc = run_pod(runner, ["--goal", "forward", "--dry-run", "--cells",
                            *map(str, _COUNT_CELLS), "--out", out],
                   pythonpath=str(src), timeout=900, n_devices=_N_DEV)
    (doc,) = _runner()._load_results(out, "forward")
    return proc.returncode, doc, proc.stdout[-4000:] + proc.stderr[-2000:]


# Per push: tests/cloud/multigpu/test_run_pod_seeded_faults.py::test_a_seeded_miscount_fails_the_forward_goals_exact_count
# and tests/cloud/multigpu/test_run_pod_seeded_faults.py::test_the_forward_goal_counts_a_field_of_ones_exactly_on_both_sides
@pytest.mark.slow
@pytest.mark.parametrize("name", [None, *sorted(_COUNT_SEEDS)],
                         ids=["no_seed", *sorted(_COUNT_SEEDS)])
def test_the_forward_goal_itself_fails_in_a_dry_run_on_each_seeded_miscount(name, tmp_path):
    """The goal as a dry run starts it, at a size with pad rows and one
    without, on a scratch copy of the runner (a seed of its node) or of the
    library (a seed of the wrapper): it exits 0 unseeded and 1 on every
    seed, and every failed check is one of the checks on the field of
    ones."""
    rp = _runner()
    seed = None if name is None else _COUNT_SEEDS[name]
    runner, src = _RUNNER, _PKG.parent
    if seed is not None and seed.file is None:
        runner = Path(_count_seeded_runner(tmp_path, name).__file__)
    elif seed is not None:
        src = _scratch_library(tmp_path, Seed(seed.old, seed.new, file=seed.file))
    rc, doc, log = _run_forward_goal(tmp_path, runner, src)
    failed = [c["name"] for c in doc["checks"] if rp.check_status(c) == "failed"]
    if name is None:
        assert rc == 0 and not failed and rp.goal_verdict([doc]) == "PASS", log
        assert rp.record_problems(doc) == []
        return
    assert rc == 1, log
    assert failed and all(" ones: " in n for n in failed), failed
    assert any(n.startswith(f"{_COUNT_CELLS[0]} cells") for n in failed), failed
    if name == "a_step_that_does_not_keep_ones":
        assert any("max |x - 1| after the step" in n for n in failed), failed
    else:
        assert not [n for n in failed if f" {_ONES_UNSHARDED} " in n], failed
    assert "CHECK FAILED" in log


# --- the unstructured wrapper's per-cell inputs and its injected parameters -----

_CELL_INPUT = ("                k: (exchange_unstructured(v, layout=layout, mesh_axis=mesh_axis, "
               "method=exchange)\n")
_INJECTION = ('            extra = {"params": local_params} if (accepts_params and local_params) '
              "else {}\n")

#: Every goal that builds ``ShardedUnstructuredNode``: each feeds the node its
#: per-cell source and injects its weight on every step.
_UNSTRUCTURED_GOALS = ("indivisible", "forward", "gradient")


class PathSeed(NamedTuple):
    old: str
    new: str
    #: The rollout derivatives (``run_pod.ROLLOUT_WRT``) the fault moves.  A
    #: fault in a value moves all three: the loss's square term puts the
    #: final field into every adjoint.
    moves: tuple = ("start", "source", "weight")
    #: The goals it must fail; every other goal that builds the wrapper
    #: must pass.  A fault that hands the node the right values and drops a
    #: cotangent is the ``gradient`` goal's alone.
    fails: tuple = _UNSTRUCTURED_GOALS


_PATH_SEEDS = {
    # The first two passed all eight goals, and closed all six items on a
    # dry run relabelled as four GPUs, until schema 10.
    "cell_inputs_are_zeroed": PathSeed(
        "                    if k in cell_bi else v)\n",
        "                    * 0.0 if k in cell_bi else v)  # SEEDED FAULT\n"),
    "injected_params_are_doubled": PathSeed(
        _INJECTION,
        '            extra = {"params": {n: 2.0 * leaf for n, leaf in local_params.items()}} '
        "if (accepts_params and local_params) else {}  # SEEDED FAULT\n"),
    # An input split over the mesh and padded, not ghost-exchanged: every
    # owned row right, every ghost row zero.
    "cell_inputs_ghost_rows_are_not_exchanged": PathSeed(
        _CELL_INPUT,
        "                k: (jnp.concatenate([v, jnp.zeros((layout.n_ghost_max,) + v.shape[1:], "
        "v.dtype)])  # SEEDED FAULT\n"),
    # Each row of a shard holds the row before it.
    "cell_inputs_are_one_row_off": PathSeed(
        _CELL_INPUT,
        "                k: (exchange_unstructured(jnp.roll(v, 1, axis=0), layout=layout, "
        "mesh_axis=mesh_axis, method=exchange)  # SEEDED FAULT\n"),
    # The injection reaches shard 0; the others run on the node's own value.
    "injected_params_reach_shard_0_only": PathSeed(
        _INJECTION,
        '            extra = {"params": {n: jnp.where(idx == 0, leaf, inner.params[n]) '
        "for n, leaf in local_params.items()}} if (accepts_params and local_params) else {}"
        "  # SEEDED FAULT\n"),
    # Right values, no cotangent: nothing a forward run compares can tell.
    "cell_inputs_lose_their_cotangent": PathSeed(
        _CELL_INPUT,
        "                k: (exchange_unstructured(lax.stop_gradient(v), layout=layout, "
        "mesh_axis=mesh_axis, method=exchange)  # SEEDED FAULT\n",
        moves=("source",), fails=("gradient",)),
    "injected_params_lose_their_cotangent": PathSeed(
        _INJECTION,
        '            extra = {"params": {n: lax.stop_gradient(leaf) for n, leaf in '
        "local_params.items()}} if (accepts_params and local_params) else {}  # SEEDED FAULT\n",
        moves=("weight",), fails=("gradient",)),
}

#: 225 cells: three shards carry a pad row and the cells stay as reverse
#: Cuthill-McKee scatters them.  256: four equal shards, which the runner
#: renumbers into partition order so that the wrapper takes a per-cell
#: input at all (``cells_in_an_order_the_wrapper_takes``).
_PATH_CELLS = (225, 256)


def _path_seeded_source(seed: PathSeed) -> str:
    text = (_PKG / _UNSTRUCTURED).read_text(encoding="utf-8")
    assert text.count(seed.old) == 1, (
        f"the seed no longer matches {_UNSTRUCTURED} exactly once ({text.count(seed.old)} "
        "matches): the wrapper changed; update the seed to the new code")
    return text.replace(seed.old, seed.new)


@pytest.mark.parametrize("name", sorted(_PATH_SEEDS))
def test_every_path_seed_applies_once_to_the_wrapper_as_it_stands(name):
    seeded = _path_seeded_source(_PATH_SEEDS[name])
    assert "SEEDED FAULT" in seeded
    compile(seeded, name, "exec")


def test_the_path_seeds_name_goals_that_build_the_wrapper_and_derivatives_it_takes():
    """The seeds' expectations are about the runner as it stands: the goals
    that build ``NeighbourMeanNode`` are the three named here (each calls
    ``build_pair`` and ``node_inputs``, and no other goal does), a seed
    fails all of them or the ``gradient`` goal alone, and the items those
    goals decide are 1, 3 and 5."""
    rp = _runner()
    text = _RUNNER.read_text(encoding="utf-8")
    def body(function: str) -> str:
        return text.split(f"def {function}(")[1].split("\ndef ")[0]

    # the gradient goal builds its pairs in the part the per-push test runs
    assert "run_gradient_rollout(" in body("run_gradient")
    parts = {**{goal: f"run_{goal}" for goal in rp.ALL_GOALS}, "gradient": "run_gradient_rollout"}
    builds = {goal for goal, fn in parts.items() if " build_pair(" in body(fn)}
    feeds = {goal for goal, fn in parts.items() if " node_inputs(" in body(fn)}
    assert builds == feeds == set(_UNSTRUCTURED_GOALS)
    for name, seed in _PATH_SEEDS.items():
        assert set(seed.moves) <= set(rp.ROLLOUT_WRT) and seed.moves, name
        assert seed.fails in (_UNSTRUCTURED_GOALS, ("gradient",)), name
        assert (seed.fails == _UNSTRUCTURED_GOALS) == (seed.moves == rp.ROLLOUT_WRT), name
    decided = {i for i, (_c, goals) in rp.CHECKLIST.items() if set(goals) & set(_UNSTRUCTURED_GOALS)}
    assert decided == {1, 3, 5}
    assert {i for i, (_c, goals) in rp.CHECKLIST.items() if "gradient" in goals} == {3}
    assert rp.NODE_WEIGHT not in (0.5, 0.0, 1.0)       # not the node's own, and a relaxation


def _paths_part(rp, cells: int) -> tuple[dict, dict, bool]:
    """``({check name: status} of the forward goal, {method: {wrt: max_rel}} of
    the gradient goal's rollout, renumbered)`` at ``cells``: the goals' own
    functions, in-process on four CPU virtual devices."""
    import jax

    if len(jax.devices()) < _N_DEV:
        pytest.skip(f"needs >= {_N_DEV} devices")
    rp._load_backend()
    # reverse Cuthill-McKee by name: what "auto" picks without PyMetis, and
    # the partition the two sizes are chosen for
    args = SimpleNamespace(mesh=None, synthetic="grid", partition="rcm", cells=[cells],
                           n_devices=_N_DEV, steps=2, grad_steps=3, warmup=0, repeats=1)
    doc = rp.run_forward(args, {})
    (entry,) = doc["results"]
    n, edges, _pa, source = rp._mesh_source(args, cells)
    pa, _how = rp._partition_for(args, source, n, edges, None, _N_DEV)
    edges, pa, renumbered = rp.cells_in_an_order_the_wrapper_takes(n, edges, pa, _N_DEV)
    assert renumbered == entry["renumbered"]
    rollout = rp.run_gradient_rollout(args, rp._mesh_for(_N_DEV), n, edges, pa)
    keys = {"start": "parity", "source": "parity_source", "weight": "parity_weight"}
    return ({c["name"]: rp.check_status(c) for c in doc["checks"]},
            {m: {wrt: rollout[m][key]["max_rel"] for wrt, key in keys.items()}
             for m in rp.METHODS}, renumbered)


@pytest.mark.parametrize("cells, renumbered", list(zip(_PATH_CELLS, (False, True))))
def test_the_unstructured_goals_pass_on_the_wrapper_as_it_stands(cells, renumbered):
    """The control of the seeds below: the ``forward`` goal and the
    ``gradient`` goal's rollout, in-process on the real wrapper, pass every
    check at a size with pad rows and at the one the runner renumbers."""
    rp = _runner()
    forward, rollout, was = _paths_part(rp, cells)
    assert was is renumbered
    assert set(forward.values()) == {"passed"}, forward
    assert all(rel <= rp.LIMITS["gradient"] for by in rollout.values()
               for rel in by.values()), rollout


@pytest.mark.parametrize("name", sorted(_PATH_SEEDS))
def test_the_unstructured_goals_show_each_seeded_fault_of_an_input_or_a_parameter(
        name, tmp_path):
    """The per-push half of the slow test below: the ``forward`` goal and
    the ``gradient`` goal's rollout, run in-process on the seeded wrapper
    at a size with pad rows.  A fault in a value fails the field's parity
    under both transports -- and nothing else of the ``forward`` goal: its
    count runs on a zero source, which every one of these faults leaves
    zero, and stays exact -- and moves all three derivatives; a dropped
    cotangent passes the whole ``forward`` goal and moves its own
    derivative only."""
    seed = _PATH_SEEDS[name]
    rp = _runner_with_unstructured_wrapper(name, _path_seeded_source(seed))
    cells = _PATH_CELLS[0]
    forward, rollout, _ = _paths_part(rp, cells)
    failed = {n for n, s in forward.items() if s != "passed"}
    field = {f"{cells} cells {m} x vs unsharded max_rel" for m in rp.METHODS}
    assert failed == (field if seed.fails == _UNSTRUCTURED_GOALS else set()), forward
    assert {n for n, s in forward.items() if s == "failed"} == failed
    for method in rp.METHODS:
        moved = {wrt for wrt, rel in rollout[method].items()
                 if not rel <= rp.LIMITS["gradient"]}
        assert moved == set(seed.moves), (method, rollout[method])
        # by orders of magnitude, not by rounding
        assert all(rollout[method][wrt] > 100 * rp.LIMITS["gradient"] for wrt in seed.moves)


def test_a_per_cell_input_is_refused_on_equal_shards_until_the_cells_are_renumbered():
    """Why the unstructured goals renumber some meshes, and that it is all
    they change.  At 256 cells reverse Cuthill-McKee gives four shards of
    64: partition layout and global order have one length, the wrapper
    cannot tell them apart and refuses the per-cell source by name.  With
    the cells in partition order it takes it.  The renumbered mesh is the
    same graph under the permutation, with each shard's cells in the order
    they had; a mesh with a pad row, or already in partition order, is
    returned untouched."""
    import jax

    if len(jax.devices()) < _N_DEV:
        pytest.skip(f"needs >= {_N_DEV} devices")
    rp = _runner()
    rp._load_backend()
    mesh = rp._mesh_for(_N_DEV)
    n, edges = rp.synthetic_mesh("grid", _PATH_CELLS[1])
    pa, how = rp.partition_cells(n, edges, _N_DEV, "rcm")
    assert how == "rcm" and set(np.bincount(pa)) == {n // _N_DEV}
    assert not np.array_equal(np.argsort(pa, kind="stable"), np.arange(n))

    _, sharded, layout = rp.build_pair(n, edges, mesh, pa, "all_to_all")
    _, bi, params = rp.node_inputs(n, layout, mesh)
    with pytest.raises(ValueError, match="does not keep cells in global order"):
        sharded.update(sharded.initial_state(), bi, 1.0, params=params)

    new_edges, new_pa, renumbered = rp.cells_in_an_order_the_wrapper_takes(n, edges, pa, _N_DEV)
    assert renumbered
    order = np.argsort(pa, kind="stable")              # new id k is old cell order[k]
    assert np.array_equal(new_pa, pa[order]) and np.all(np.diff(new_pa) >= 0)
    assert np.array_equal(order[new_edges], edges)     # the same graph, relabelled
    for d in range(_N_DEV):                            # each shard's cells in their old order
        assert np.all(np.diff(order[new_pa == d]) > 0)
    ref, sharded, layout = rp.build_pair(n, new_edges, mesh, new_pa, "all_to_all")
    bi_ref, bi, params = rp.node_inputs(n, layout, mesh)
    got = sharded.gather_global(sharded.update(sharded.initial_state(), bi, 1.0, params=params))
    want = ref.update(ref.initial_state(), bi_ref, 1.0, params=params)
    assert np.array_equal(np.asarray(got["x"]), np.asarray(want["x"]))

    for cells, method in ((_PATH_CELLS[0], "rcm"), (_PATH_CELLS[1], "contiguous")):
        n, edges = rp.synthetic_mesh("grid", cells)
        pa, _ = rp.partition_cells(n, edges, _N_DEV, method)
        same_edges, same_pa, renumbered = rp.cells_in_an_order_the_wrapper_takes(
            n, edges, pa, _N_DEV)
        assert not renumbered and same_edges is edges and np.array_equal(same_pa, pa)
    # and the same call a second time changes nothing more
    assert not rp.cells_in_an_order_the_wrapper_takes(n, new_edges, new_pa, _N_DEV)[2]


def _run_unstructured_goals(tmp: Path, src: Path) -> tuple[dict, dict, str]:
    """``({goal: exit status}, {goal: [its file]}, the logs' tails)`` of the
    three goals that build the unstructured wrapper, each as a dry run of
    its own at ``_PATH_CELLS``, on the library under ``src``."""
    rp = _runner()
    codes, docs, log = {}, {}, ""
    for goal in _UNSTRUCTURED_GOALS:
        out = tmp / "out"
        proc = run_pod(_RUNNER, ["--goal", goal, "--dry-run", "--cells",
                                 *map(str, _PATH_CELLS), "--out", out],
                       pythonpath=str(src), timeout=900, n_devices=_N_DEV)
        codes[goal], docs[goal] = proc.returncode, rp._load_results(out, goal)
        log += proc.stdout[-3000:] + proc.stderr[-2000:]
    return codes, docs, log


# Per push: tests/cloud/multigpu/test_run_pod_seeded_faults.py::test_the_unstructured_goals_show_each_seeded_fault_of_an_input_or_a_parameter
# and tests/cloud/multigpu/test_run_pod_seeded_faults.py::test_the_unstructured_goals_pass_on_the_wrapper_as_it_stands
@pytest.mark.slow
@pytest.mark.parametrize("name", [None, *sorted(_PATH_SEEDS)],
                         ids=["no_seed", *sorted(_PATH_SEEDS)])
def test_a_seeded_fault_of_an_input_or_a_parameter_fails_every_goal_on_its_path(
        name, tmp_path):
    """The goals as a dry run starts them, on a scratch copy of the library
    with one seed: each goal the seed names exits 1 with a record
    ``--summarise`` accepts, every other one exits 0, and relabelled as a
    four-GPU run every checklist item a failing goal decides has FAILED
    (1, 3 and 5; a dropped cotangent, 3).  Unseeded, all three pass."""
    rp = _runner()
    seed = None if name is None else _PATH_SEEDS[name]
    src = _scratch_library(tmp_path, None if seed is None
                           else Seed(seed.old, seed.new, file=_UNSTRUCTURED))
    codes, docs, log = _run_unstructured_goals(tmp_path, src)
    fails = () if seed is None else seed.fails
    verdicts = {g: rp.goal_verdict(d) for g, d in docs.items()}
    assert verdicts == {g: "FAIL" if g in fails else "PASS" for g in _UNSTRUCTURED_GOALS}, (
        verdicts, log)
    assert codes == {g: int(g in fails) for g in _UNSTRUCTURED_GOALS}, (codes, log)
    for goal, (doc,) in docs.items():
        assert rp.record_problems(doc) == [], (goal, rp.record_problems(doc))
        assert "raised" not in doc, (goal, doc.get("raised"))
    status = {i: s for i, (s, _) in rp.checklist_status(_as_four_gpus(docs)).items()}
    failing = {i for i, (_c, goals) in rp.CHECKLIST.items() if set(goals) & set(fails)}
    assert {i for i, s in status.items() if s == "FAILED"} == failing, status
    assert "CLOSED" not in {status[i] for i in failing}
    if seed is None:
        return
    assert failing == ({1, 3, 5} if fails == _UNSTRUCTURED_GOALS else {3})
    # what failed is the parity with the unsharded node, at both sizes --
    # the one with pad rows and the one the runner renumbers
    for goal in fails:
        (doc,) = docs[goal]
        failed = [c["name"] for c in doc["checks"] if rp.check_status(c) == "failed"]
        assert failed and all(" vs unsharded max_rel" in n for n in failed), (goal, failed)
        if goal != "indivisible":
            for cells in _PATH_CELLS:
                assert any(n.startswith(f"{cells} cells ") for n in failed), (goal, failed)
    (gradient,) = docs["gradient"]
    failed = {c["name"] for c in gradient["checks"] if rp.check_status(c) == "failed"}
    named = {"start": " grad vs unsharded", "source": " grad wrt the per-cell source vs",
             "weight": " grad wrt the injected weight vs"}
    for wrt, part in named.items():
        assert any(part in n for n in failed) == (wrt in seed.moves), (wrt, sorted(failed))
    assert not [n for n in failed if " sharded_cg " in n], failed

