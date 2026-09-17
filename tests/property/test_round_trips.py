"""Round-trip invariants for every way a graph can leave the process.

Each test states one invariant over random *valid* graphs (see
``tests/property/strategies.py``):

* **config** -- ``to_dict`` then ``from_dict`` gives back a graph with
  the same trajectory, the same parameter pytree (structure, dtype and
  bits) and the same ``param_specs()``, including the edge-key specs
  that exist only on mapped edges;
* **idempotence** -- ``to_dict(from_dict(to_dict(g))) == to_dict(g)``;
* **USD** -- the same through ``save_graph_to_usd`` /
  ``load_graph_from_usd``;
* **checkpoint** -- ``save_state`` / ``load_state`` restores state and
  params exactly, a continued rollout agrees with an uninterrupted one,
  and a checkpoint's trained mapping weights beat the recipe a config
  rebuilds;
* **cross-format** -- config and USD reload to the same trajectory.

``pxr`` is a hard dependency of the USD tests here: it is installed in
the development environment and a skip would hide exactly the bug the
USD property is for.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, note
from hypothesis import strategies as st
from pxr import Usd

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.usd.serialization import load_graph_from_usd, save_graph_to_usd

from tests.property.invariants import (
    assert_leaf_tree_identical,
    assert_param_specs_identical,
    assert_params_identical,
    assert_states_identical,
    assert_structure_identical,
    structure,
)
from tests.property.strategies import NODE_REGISTRY, graph_recipes

#: Rollout length.  Three steps is enough to make every coupling path
#: (multi-rate dividers, additive edges, mapped interfaces) contribute
#: and short enough that the XLA compilation, not the execution,
#: dominates -- which it does either way.
N_STEPS = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reload_config(gm: GraphManager, registry: dict) -> tuple[dict, GraphManager]:
    config = gm.to_dict()
    reloaded = GraphManager.from_dict(config, registry)
    reloaded.compile()
    return config, reloaded


def _rollout(gm: GraphManager, n_steps: int) -> dict:
    """*n_steps* as a Python loop over the jitted single step, and the
    state it reaches.

    Not ``run_scan``: XLA compiles a ``lax.scan`` of length *n* as one
    program and is free to contract and reassociate across its
    iterations, so ``run_scan(3)`` and ``run_scan(2) + run_scan(1)`` can
    land one ulp apart (they do, for a two-ball chain -- see
    ``test_a_split_rollout_is_step_for_step_identical``).  That is a
    property of the compiler, not of the checkpoint, and it would turn
    the exact comparison below into a measurement of XLA's fusion
    choices.  ``run`` re-enters the *same* compiled step every time, so a
    rollout split anywhere is bit-identical to an unsplit one.
    """
    gm.run(n_steps)
    return {name: gm.get_node_state(name) for name in gm.node_names}


def _reload_usd(gm: GraphManager) -> GraphManager:
    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    reloaded = load_graph_from_usd(stage)
    reloaded.compile()
    return reloaded


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@given(recipe=graph_recipes())
def test_a_config_round_trip_preserves_trajectory_params_and_specs(recipe):
    """``from_dict(to_dict(g))`` is the same graph: same topology, same
    parameter pytree down to the dtype, same ``param_specs()``, same
    trajectory."""
    gm = recipe.build()
    note(f"recipe: {recipe}")
    config, reloaded = _reload_config(gm, recipe.registry)
    note(f"config: {config}")

    assert_structure_identical(structure(gm), structure(reloaded))
    assert_params_identical(gm.params, reloaded.params)
    assert_param_specs_identical(gm.param_specs(), reloaded.param_specs())
    assert_states_identical(gm.run_scan(N_STEPS), reloaded.run_scan(N_STEPS),
                            what="trajectory")


@given(recipe=graph_recipes(require_mapping=True, max_nodes=3))
def test_a_config_round_trip_preserves_a_mapped_edge_and_its_weights(recipe):
    """The case three audits this cycle found bugs in: an edge carrying an
    interface mapping, whose ``MappingSpec`` has to rebuild the weights
    bitwise and whose ``params["mappings"]`` slot has to keep its (possibly
    trainable) ``ParamSpec``."""
    gm = recipe.build()
    note(f"recipe: {recipe}")
    assert gm.params["mappings"], "the strategy promised a mapped edge"
    _config, reloaded = _reload_config(gm, recipe.registry)

    assert_leaf_tree_identical(gm.params["mappings"], reloaded.params["mappings"],
                               what="params['mappings']")
    assert_param_specs_identical(gm.param_specs(), reloaded.param_specs())
    assert_states_identical(gm.run_scan(N_STEPS), reloaded.run_scan(N_STEPS),
                            what="trajectory")


@given(recipe=graph_recipes())
def test_to_dict_is_idempotent_through_from_dict(recipe):
    """``to_dict(from_dict(to_dict(g))) == to_dict(g)``.  A field that
    survives the first write but not the second is a field the loader
    drops."""
    gm = recipe.build()
    note(f"recipe: {recipe}")
    once = gm.to_dict()
    twice = GraphManager.from_dict(once, recipe.registry).to_dict()
    assert twice == once


# ---------------------------------------------------------------------------
# USD
# ---------------------------------------------------------------------------

@given(recipe=graph_recipes())
def test_a_usd_round_trip_preserves_trajectory_params_and_specs(recipe):
    """The USD stage is the other serialisation of the same graph and owes
    the same guarantees as the config."""
    gm = recipe.build()
    note(f"recipe: {recipe}")
    reloaded = _reload_usd(gm)

    assert_structure_identical(structure(gm), structure(reloaded))
    assert_params_identical(gm.params, reloaded.params)
    assert_param_specs_identical(gm.param_specs(), reloaded.param_specs())
    assert_states_identical(gm.run_scan(N_STEPS), reloaded.run_scan(N_STEPS),
                            what="trajectory")


@given(recipe=graph_recipes(require_mapping=True, max_nodes=3))
def test_a_usd_round_trip_preserves_a_mapped_edge_and_its_weights(recipe):
    gm = recipe.build()
    note(f"recipe: {recipe}")
    reloaded = _reload_usd(gm)

    assert_leaf_tree_identical(gm.params["mappings"], reloaded.params["mappings"],
                               what="params['mappings']")
    assert_param_specs_identical(gm.param_specs(), reloaded.param_specs())
    assert_states_identical(gm.run_scan(N_STEPS), reloaded.run_scan(N_STEPS),
                            what="trajectory")


# -- pinned regressions: the names and fields the USD writer used to lose ---

def test_usd_keeps_a_node_name_that_is_not_a_prim_identifier():
    """``"b-1"`` is a legal node name and not a legal ``SdfPath`` element.
    The writer has to mangle the prim name; it must not mangle the *node*
    name with it, or every edge that referred to the node dangles.

    Pinned from the first failing example of
    ``test_a_usd_round_trip_preserves_trajectory_params_and_specs``.
    """
    gm = GraphManager()
    gm.add_node(TableNode("t", 0.01, position=0.25))
    gm.add_node(BallNode("b-1", 0.01, initial_position=3.0))
    gm.add_edge("t", "b-1", "position", "table_position")
    gm.compile()

    reloaded = _reload_usd(gm)
    assert reloaded.node_names == ["t", "b-1"]
    assert [(e.source_node, e.target_node) for e in reloaded.edges] == [("t", "b-1")]


def test_usd_keeps_two_node_names_that_share_one_prim_name():
    """``"a-b"`` and ``"a.b"`` are two nodes; a naive safe-name mangling
    collapses them onto the prim ``a_b`` and the reload silently loses one."""
    gm = GraphManager()
    gm.add_node(BallNode("a-b", 0.01, initial_position=1.0))
    gm.add_node(BallNode("a.b", 0.01, initial_position=2.0))
    gm.add_edge("a-b", "a.b", "position", "table_position")
    gm.compile()

    reloaded = _reload_usd(gm)
    assert sorted(reloaded.node_names) == ["a-b", "a.b"]
    assert_states_identical(gm.run_scan(N_STEPS), reloaded.run_scan(N_STEPS))


def test_usd_keeps_a_node_name_that_starts_with_a_digit():
    """``Sdf.Path.TokenizeIdentifier("1st")`` is empty, which used to leave
    the writer asking USD to define a prim at an ill-formed path."""
    gm = GraphManager()
    gm.add_node(TableNode("1st", 0.01, position=0.5))
    gm.add_node(BallNode("2nd", 0.01, initial_position=2.0))
    gm.add_edge("1st", "2nd", "position", "table_position")
    gm.compile()

    reloaded = _reload_usd(gm)
    assert reloaded.node_names == ["1st", "2nd"]
    assert_states_identical(gm.run_scan(N_STEPS), reloaded.run_scan(N_STEPS))


def test_usd_keeps_the_declared_units_of_an_edge():
    """``source_units`` / ``target_units`` are part of ``EdgeSpec`` and of
    the config; the stage has to carry them too, or a reloaded graph stops
    unit-checking the edge it was checking before."""
    gm = GraphManager()
    gm.add_node(TableNode("t", 0.01))
    gm.add_node(BallNode("b", 0.01))
    gm.add_edge("t", "b", "position", "table_position",
                source_units="m", target_units="m")
    gm.compile()

    reloaded = _reload_usd(gm)
    assert reloaded.edges[0].source_units == "m"
    assert reloaded.edges[0].target_units == "m"


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

@given(recipe=graph_recipes(), split=st.integers(min_value=1, max_value=N_STEPS - 1))
def test_a_checkpoint_restores_state_and_params_and_continues_the_same_rollout(
    recipe, split,
):
    """Stopping at step *split*, checkpointing, and resuming in a fresh
    process-equivalent graph must reach the state an uninterrupted run
    reaches -- with the parameter pytree restored bit for bit."""
    note(f"recipe: {recipe}")
    gm = recipe.build()
    gm.run(split)
    with tempfile.TemporaryDirectory() as tmp:
        path = gm.save_state(Path(tmp) / "checkpoint.npz")
        resumed = recipe.build()
        resumed.load_state(path)

        assert_params_identical(gm.params, resumed.params)
        assert_states_identical(
            {n: gm.get_node_state(n) for n in gm.node_names},
            {n: resumed.get_node_state(n) for n in resumed.node_names},
            what="restored state",
        )
        # ``gm`` carrying on past its own checkpoint *is* the
        # uninterrupted run, and costs no third graph to build.
        assert_states_identical(_rollout(gm, N_STEPS - split),
                                _rollout(resumed, N_STEPS - split),
                                what="continued trajectory")


@given(recipe=graph_recipes(train_mapping_weights=True, max_nodes=3))
def test_a_checkpoint_beats_the_config_for_trained_mapping_weights(recipe):
    """A config carries the ``MappingSpec``, not the weights.  Weights moved
    by a fit therefore come back only from a checkpoint -- and when one is
    loaded on top of a config it must win over what the spec rebuilt."""
    note(f"recipe: {recipe}")
    gm = recipe.build()
    assert gm.params["mappings"], "the strategy promised a mapped edge"

    with tempfile.TemporaryDirectory() as tmp:
        # Checkpoint before stepping, so ``gm``'s own rollout is the
        # reference the restored graph has to reproduce.
        path = gm.save_state(Path(tmp) / "trained.npz")
        reference = gm.run_scan(N_STEPS)
        # ``to_dict`` says out loud that it is about to drop these weights.
        with pytest.warns(UserWarning, match="live mapping weights"):
            config = gm.to_dict()
        rebuilt = GraphManager.from_dict(config, recipe.registry)
        rebuilt.compile()
        # The recipe rebuild is *not* the trained state: if it were, this
        # property would be vacuous.
        assert any(
            not bool((rebuilt.params["mappings"][key][w] == value).all())
            for key, slot in gm.params["mappings"].items()
            for w, value in slot.items()
        ), "mapping_weight_scale did not move the weights"

        rebuilt.load_state(path)
        assert_leaf_tree_identical(gm.params["mappings"], rebuilt.params["mappings"],
                                   what="params['mappings']")
        assert_states_identical(reference, rebuilt.run_scan(N_STEPS),
                                what="trajectory after checkpoint-over-config")


def test_a_split_rollout_is_step_for_step_identical():
    """Why the checkpoint property rolls out with ``run`` and not
    ``run_scan``.

    Pinned from the first failing example of
    ``test_a_checkpoint_restores_state_and_params_and_continues_the_same_rollout``:
    a two-ball chain whose third-step velocity came out one ulp apart
    depending on whether the three steps were one ``lax.scan`` or a
    ``lax.scan`` of two followed by one of one.  The checkpoint is exact
    -- every step-by-step spelling agrees bit for bit -- and it is the
    *fused* scan that differs, because XLA compiles each trip count as
    its own program.  Stated as equality where equality holds and as a
    tolerance where it does not, so a future XLA that fuses differently
    (or identically) still passes.
    """
    def build():
        gm = GraphManager()
        gm.add_node(BallNode("rod", 0.01, initial_position=0.0, initial_velocity=0.0,
                             elasticity=0.0, gravity=-3.0))
        gm.add_node(BallNode("node_1", 0.01, initial_position=0.0, initial_velocity=0.0,
                             elasticity=0.0, gravity=-2.181640625))
        gm.add_edge("rod", "node_1", "position", "table_position")
        gm.compile()
        return gm

    whole = _rollout(build(), 3)
    split = build()
    split.run(2)
    with tempfile.TemporaryDirectory() as tmp:
        path = split.save_state(Path(tmp) / "checkpoint.npz")
        resumed = build()
        resumed.load_state(path)
        assert_states_identical(whole, _rollout(resumed, 1), what="split rollout")

    fused = build().run_scan(3)
    assert float(fused["node_1"]["velocity"]) == pytest.approx(
        float(whole["node_1"]["velocity"]), rel=1e-6,
    )


# ---------------------------------------------------------------------------
# Cross-format
# ---------------------------------------------------------------------------

@given(recipe=graph_recipes())
def test_config_and_usd_reload_to_the_same_graph(recipe):
    """The two formats are two spellings of one graph; a difference between
    them is a bug in whichever one is younger."""
    gm = recipe.build()
    note(f"recipe: {recipe}")
    _config, from_config = _reload_config(gm, recipe.registry)
    from_usd = _reload_usd(gm)

    assert_structure_identical(structure(from_config), structure(from_usd))
    assert_params_identical(from_config.params, from_usd.params)
    assert_param_specs_identical(from_config.param_specs(), from_usd.param_specs())
    assert_states_identical(from_config.run_scan(N_STEPS), from_usd.run_scan(N_STEPS),
                            what="trajectory")
