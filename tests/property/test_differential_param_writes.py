"""Differential oracle 2: a ``gm.params`` (or calibration) write is a reload.

A fit hands back a parameter pytree and the caller writes it into
``gm.params``; ``to_dict()`` then saves each node's effective params (the
constructor arguments with the live leaves written over them) and
``from_dict()`` rebuilds the node from them.  So a write the graph accepts
-- one that ``run``, ``run_scan`` and ``to_dict`` do not refuse -- must be
one the saved graph reproduces.  Stated over generated writes into every
cheap built-in node and into generated multi-node graphs (coupling groups,
transforms, mappings, ParamSpec overrides), and over the output of a real
:func:`maddening.sysid.fit`.

The oracle, for every write:

* **refused** -- the documented refusal of a leaf the step cannot read is a
  ``ValueError`` from the next run *and* from ``to_dict()``, so a refused
  write can be neither computed with nor saved;
* **accepted** --

  - the config (through JSON) loads: its constructor takes the values;
  - config + a checkpoint taken right after the write restore the full
    state bit for bit and continue ``N_STEPS`` bit for bit as the written
    graph does;
  - after ``reset_state()`` on both, they step bit for bit too.

Tolerance: none (as for oracle 1).

Known disagreements, pinned as strict xfails below rather than drawn by the
properties: ``LBMPipeNode.G`` through zero (MADD-ANO-047's live residual)
and a ``HeatNode`` diffusivity past the Fourier limit (MADD-ANO-002 /
MADD-ANO-048: writes outside the REST route are not asked the
constructor's question).  The properties draw values inside every declared
bound and every constructor check; what the generator does not draw, the
oracle does not see.
"""

from __future__ import annotations

import json
import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import event, given, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager

from tests.conftest import EXAMPLES_COSTLY, EXAMPLES_STANDARD
from tests.property.differential import (
    note,
    assert_trees_identical,
    checkpoint_path,
    full_state,
    no_cloud_launch,
    reload_from_config,
    rollout,
    tmp_dir,
)
from tests.property.node_catalogue import (
    CHEAP_KINDS,
    COSTLY_KINDS,
    KINDS,
    REGISTRY,
    writes,
)

WARM_STEPS = 2
N_STEPS = 3

#: Write categories whose values lie inside every declared bound and every
#: constructor check (``node_catalogue.Kind.safe``), so the oracle owes an
#: exact agreement for each of them.
IN_BOUNDS = ("valid", "multi", "initial")


@pytest.fixture(scope="module", autouse=True)
def _no_cloud():
    with no_cloud_launch():
        yield


def _leaf_writes(gm: GraphManager, owner: str, params: dict) -> dict:
    """``{key: array}`` in the live leaves' dtypes, for every key of
    ``params`` that is a leaf of ``gm.params["nodes"][owner]``."""
    live = gm.params["nodes"].get(owner, {})
    out = {}
    for key, value in params.items():
        if key not in live:
            continue
        ref = np.asarray(live[key])
        arr = np.asarray(value, dtype=ref.dtype)
        if arr.shape != ref.shape:
            continue
        out[key] = jnp.asarray(arr)
    return out


def apply_write(gm: GraphManager, owner: str, leaves: dict, *, how: str) -> None:
    """Write ``leaves`` the way a caller does: leaf by leaf into the live
    tree, or by replacing the whole tree (what ``gm.params = fit.params``
    does)."""
    if how == "leaf":
        for key, value in leaves.items():
            gm.params["nodes"][owner][key] = value
        return
    tree = {section: {o: dict(v) for o, v in owners.items()}
            for section, owners in gm.params.items()}
    tree["nodes"][owner].update(leaves)
    gm.params = tree


def check_params_write(gm: GraphManager, registry: dict, owner: str, leaves: dict,
                       *, how: str = "leaf") -> str:
    """Hold one ``gm.params`` write to the oracle; ``"accepted"`` or ``"refused"``."""
    gm.run(WARM_STEPS)
    apply_write(gm, owner, leaves, how=how)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*live mapping weights.*")
            config = json.loads(json.dumps(gm.to_dict(), allow_nan=True))
    except ValueError as exc:
        note(f"to_dict refused: {exc}")
        before = full_state(gm)
        for label, run in (("run", lambda: gm.run(1)), ("run_scan", lambda: gm.run_scan(1))):
            try:
                run()
            except ValueError as run_exc:
                assert "differs from the node's own value" in str(run_exc), str(run_exc)
            else:
                raise AssertionError(
                    f"to_dict() refused the write ({exc}) and {label}() computed with it")
        assert_trees_identical(before, full_state(gm), what="state after a refused run")
        return "refused"
    with tmp_dir() as tmp:
        ckpt = gm.save_state(checkpoint_path(tmp))
        reloaded = reload_from_config(config, registry)
        reloaded.load_state(ckpt)
    assert_trees_identical(full_state(gm), full_state(reloaded), what="restored state")
    assert_trees_identical(rollout(gm, N_STEPS), rollout(reloaded, N_STEPS),
                           what="continued after the write")
    gm.reset_state()
    reloaded.reset_state()
    assert_trees_identical(rollout(gm, N_STEPS), rollout(reloaded, N_STEPS),
                           what="after a reset")
    return "accepted"


# ---------------------------------------------------------------------------
# Per push
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind_name", CHEAP_KINDS)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_param_write_inside_its_bounds_runs_as_its_reload(kind_name, data):
    """One built-in node, one generated in-bounds write into ``gm.params``
    (leaf by leaf, or the whole tree replaced): the saved graph reproduces
    what the written graph computes, or the write is refused everywhere."""
    kind = KINDS[kind_name]
    kwargs = data.draw(kind.kwargs, label="kwargs")
    write = data.draw(writes(kind, kwargs, categories=IN_BOUNDS), label="write")
    how = data.draw(st.sampled_from(["leaf", "tree"]), label="how")
    gm = kind.graph(kwargs)
    leaves = _leaf_writes(gm, kind.name, write.params)
    event(f"{write.category}/{how}: {'leaves' if leaves else 'no leaf'}")
    outcome = check_params_write(gm, REGISTRY, kind.name, leaves, how=how)
    event(f"{write.category}: {outcome}")
    if write.category in ("valid", "multi"):
        # Every leaf the catalogue calls ``safe`` is one the step reads.
        assert outcome == "accepted", write.params


#: ``(stiffness, damping, iterations)``.  A fixed table, not a draw: every
#: case compiles the fit's own ``value_and_grad`` (about 0.5 s), so twenty
#: draws would take a per-push test past the time budget, and what the
#: oracle asks of a fit's output does not depend on where the fit started.
_FIT_CASES = [(12.0, 0.25, 1), (45.0, 2.0, 2), (30.5, 1.0, 3)]


@pytest.mark.parametrize("stiffness, damping, iterations", _FIT_CASES)
def test_the_output_of_a_fit_written_into_the_graph_runs_as_its_reload(
        stiffness, damping, iterations):
    """``gm.params = fit(...).params`` -- the calibration write -- after a
    few Adam iterations against a target trajectory: the saved graph
    reproduces the calibrated one."""
    from maddening.nodes import SpringDamperNode
    from maddening.sysid import fit

    def build(k, c):
        gm = GraphManager()
        gm.add_node(SpringDamperNode("spring", 0.01, stiffness=k, damping=c,
                                     rest_length=0.5, initial_position=0.1))
        gm.add_external_input("spring", "anchor_position")
        gm.compile()
        return gm

    target = build(30.0, 1.0).run_scan(8)["spring"]["position"]
    gm = build(stiffness, damping)

    def loss(p):
        return jnp.sum((_positions(gm, p) - target) ** 2)

    result = fit(gm, loss, n_iter=iterations, lr=0.05)
    gm.reset_state()
    outcome = check_params_write(
        gm, REGISTRY, "spring", dict(result.params["nodes"]["spring"]), how="tree")
    assert outcome == "accepted"


def _positions(gm: GraphManager, params: dict):
    """The spring's position after 8 steps under ``params``, traced."""
    state = {n: dict(f) for n, f in gm._state.items()}  # noqa: SLF001
    ext = gm._default_external_inputs()  # noqa: SLF001
    full = gm._params_or_default(params)  # noqa: SLF001
    for _ in range(8):
        state = gm._compiled_step(state, ext, full)  # noqa: SLF001
    return state["spring"]["position"]


# ---------------------------------------------------------------------------
# Known disagreements, pinned
# ---------------------------------------------------------------------------

def _pipe(G: float):
    from maddening.nodes import LBMPipeNode

    gm = GraphManager()
    gm.add_node(LBMPipeNode("p", 1.0, nx=6, ny=6, nz=6, pipe_radius=0.8, propeller_x=2,
                            propeller_strength=0.0, G=G, rho_liquid=2.0, rho_gas=0.2,
                            fill_fraction=0.5))
    gm.compile()
    return gm


# Per push: tests/property/test_differential_param_writes.py::test_a_pipe_interaction_strength_written_within_its_branch_runs_as_its_reload
@pytest.mark.slow  # two multiphase-pipe compiles: ~6 s on 3 cores
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "differential: gm.params G=0 on a multiphase LBMPipeNode keeps the multiphase "
    "branch its constructor fixed while the saved config reloads single-phase "
    "(MADD-ANO-047 residual: writes outside the REST route are not asked); pending fix"))
def test_a_pipe_interaction_strength_written_through_zero_runs_as_its_reload():
    gm = _pipe(-5.0)
    check_params_write(gm, REGISTRY, "p", {"G": jnp.float32(0.0)})


def test_a_pipe_interaction_strength_written_within_its_branch_runs_as_its_reload():
    """The neighbour of the pinned case, which must keep passing: a write of
    ``G`` that stays on its side of zero."""
    gm = _pipe(-5.0)
    assert check_params_write(gm, REGISTRY, "p", {"G": jnp.float32(-4.5)}) == "accepted"


@pytest.mark.xfail(strict=True, raises=ValueError, reason=(
    "differential: a gm.params thermal_diffusivity past the HeatNode Fourier limit is "
    "computed with and saved, and the saved config does not load (MADD-ANO-002 / "
    "MADD-ANO-048 residual: writes outside the REST route skip the constructor); "
    "pending fix"))
def test_a_heat_diffusivity_written_past_the_fourier_limit_is_refused_or_reloads():
    from maddening.nodes import HeatNode

    gm = GraphManager()
    gm.add_node(HeatNode("rod", 1.0, n_cells=16, length=1.0,
                         thermal_diffusivity=0.2 / 256, initial_temperature=300.0))
    gm.compile()
    check_params_write(gm, REGISTRY, "rod", {"thermal_diffusivity": jnp.float32(0.6 / 256)})


@pytest.mark.parametrize("kind", ["rbf", "inverse_distance"])
def test_a_rod_length_under_a_mapping_reference_runs_as_its_reload(kind):
    """A ``gm.params`` length for a uniform rod whose ``grid_x`` an interface
    mapping references used to be computed with (the rod on the new length,
    the mapping on the old grid's weights) and saved as a config that did not
    load.  Refused at the next run and by ``to_dict()`` now -- for the
    built-in RBF mapping and for one of a registered kind alike."""
    from tests.property.test_differential_rest_params import rods_mapped_by_grid

    gm = rods_mapped_by_grid(kind)
    assert check_params_write(gm, REGISTRY, "a", {"length": jnp.float32(1.5)}) == "refused"


@pytest.mark.parametrize("kind", ["rbf", "inverse_distance"])
def test_a_calibration_of_a_mapped_rod_runs_as_its_reload(kind):
    """The write the refusal must leave alone: a constant of a mapped rod
    that moves no referenced point set is computed with, saved and
    reloaded, with the mapping's weights carried by the checkpoint."""
    from tests.property.test_differential_rest_params import rods_mapped_by_grid

    gm = rods_mapped_by_grid(kind)
    assert check_params_write(
        gm, REGISTRY, "a", {"thermal_diffusivity": jnp.float32(0.004)}) == "accepted"


# ---------------------------------------------------------------------------
# Slow lane
# ---------------------------------------------------------------------------

# Per push: tests/property/test_differential_param_writes.py::test_a_param_write_inside_its_bounds_runs_as_its_reload
@pytest.mark.slow  # an LBM / pipe / wavelet compile per example: 2-6 s each
@pytest.mark.parametrize("kind_name", COSTLY_KINDS)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_param_write_into_a_costly_node_runs_as_its_reload(kind_name, data):
    kind = KINDS[kind_name]
    kwargs = data.draw(kind.kwargs, label="kwargs")
    write = data.draw(writes(kind, kwargs, categories=IN_BOUNDS), label="write")
    gm = kind.graph(kwargs)
    check_params_write(gm, REGISTRY, kind.name, _leaf_writes(gm, kind.name, write.params))


@st.composite
def _calibrated_write(draw):
    """A generated graph and a calibration-like write: every perturbable
    leaf of up to three nodes scaled by a factor a fit could reach."""
    from tests.property.strategies import ALL_NODE_KINDS, graph_recipes

    recipe = draw(graph_recipes(kinds=ALL_NODE_KINDS, max_nodes=3), label="recipe")
    moved = {}
    for node in draw(st.lists(st.sampled_from(recipe.nodes), min_size=1, max_size=3,
                              unique_by=lambda n: n.name)):
        for key in node.perturbable:
            if draw(st.booleans()):
                moved.setdefault(node.name, {})[key] = draw(
                    st.sampled_from([0.5, 0.8, 1.25, 2.0]))
    return recipe, moved, draw(st.sampled_from(["leaf", "tree"]))


# Per push: tests/property/test_differential_param_writes.py::test_a_param_write_inside_its_bounds_runs_as_its_reload
@pytest.mark.slow  # a generated multi-node graph built and compiled per example
@settings(max_examples=EXAMPLES_STANDARD, derandomize=True)
@given(case=_calibrated_write())
def test_a_calibration_written_into_a_generated_graph_runs_as_its_reload(case):
    """Coupling groups, transforms, mappings and ParamSpec overrides around
    the leaves a calibration moved."""
    from tests.property.strategies import NODE_REGISTRY

    recipe, moved, how = case
    note(f"recipe: {recipe}; moved: {moved}; how: {how}")
    gm = recipe.build()
    gm.run(WARM_STEPS)
    for owner, factors in moved.items():
        live = gm.params["nodes"][owner]
        leaves = {k: (np.asarray(live[k]) * np.float32(f)).astype(np.asarray(live[k]).dtype)
                  for k, f in factors.items()}
        apply_write(gm, owner, {k: jnp.asarray(v) for k, v in leaves.items()}, how=how)
    gm.reset_state()
    with tmp_dir() as tmp:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*live mapping weights.*")
            config = json.loads(json.dumps(gm.to_dict(), allow_nan=True))
        ckpt = gm.save_state(checkpoint_path(tmp))
        reloaded = reload_from_config(config, dict(NODE_REGISTRY))
        reloaded.load_state(ckpt)
    assert_trees_identical(rollout(gm, N_STEPS), rollout(reloaded, N_STEPS),
                           what="calibrated graph against its reload")
