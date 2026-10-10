"""Differential oracle 4: a serialised graph runs as the original.

``to_dict()`` -> JSON text -> ``from_dict()`` must give back a graph that
writes the same config, builds the same initial state, holds the same
parameter pytree and ``param_specs()``, and steps the same trajectory -- bit
for bit.  ``tests/property/test_round_trips.py`` states this for generated
graphs of the five scalar kinds in ``strategies.py``, and
``tests/nodes/test_builtin_node_round_trip.py`` for each built-in node with
one hand-picked set of arguments; this module states it for

* every built-in node with **generated** constructor arguments (arrays,
  grids, wall masks, constraint dicts, check dicts), per push for the cheap
  kinds and in the slow lane for the LBM and wavelet nodes;
* generated graphs over **every** node kind ``strategies`` can build (the
  heart pump and the 3-D rigid body besides the five), per push without
  coupling groups and in the slow lane with them.

The USD half of the same oracle is
``tests/usd/test_usd_differential_round_trip.py`` (the only job that installs
``usd-core`` runs it, and runs no slow test); the sharded half is
``tests/cloud/multigpu/test_property_sharded_state_io_differential.py``.

Tolerance: none.  A reload builds the same nodes from the same numbers, so
the same compiled program runs on both sides.

What it cannot see: a field both ``to_dict`` and ``from_dict`` ignore (a
constructor argument the node does not keep in ``params``, unless it changes
the trajectory, the initial state or ``param_specs()``), and a node class
with no catalogue entry (``test_every_built_in_node_has_a_catalogue_entry``
fails closed on that).
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import maddening.nodes as builtin_nodes
import maddening.nodes.adaptive as adaptive_nodes
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode

from tests.conftest import EXAMPLES_COSTLY, EXAMPLES_STANDARD
from tests.property.differential import (
    note,
    assert_trees_identical,
    full_state,
    params_tree,
    reload_from_config,
    rollout,
)
from tests.property.invariants import (
    assert_param_specs_identical,
    assert_structure_identical,
    structure,
)
from tests.property.node_catalogue import CHEAP_KINDS, COSTLY_KINDS, KINDS, REGISTRY

N_STEPS = 3


def initial_states(gm: GraphManager) -> dict:
    return {name: {k: np.asarray(v) for k, v in gm.get_node(name).initial_state().items()}
            for name in gm.node_names}


def check_config_round_trip(gm: GraphManager, registry: dict) -> GraphManager:
    """Hold ``gm`` to the oracle through ``to_dict`` -> JSON -> ``from_dict``;
    return the reload."""
    text = json.dumps(gm.to_dict(), allow_nan=True, sort_keys=True)
    reloaded = reload_from_config(json.loads(text), registry)
    assert json.dumps(reloaded.to_dict(), allow_nan=True, sort_keys=True) == text, (
        "the reload writes another config")
    assert_structure_identical(structure(gm), structure(reloaded))
    assert_trees_identical(initial_states(gm), initial_states(reloaded),
                           what="initial state")
    assert_trees_identical(full_state(gm), full_state(reloaded), what="compiled state")
    assert_trees_identical(params_tree(gm), params_tree(reloaded), what="params")
    assert_param_specs_identical(gm.param_specs(), reloaded.param_specs())
    assert_trees_identical(rollout(gm, N_STEPS), rollout(reloaded, N_STEPS),
                           what="trajectory")
    return reloaded


# ---------------------------------------------------------------------------
# Per push
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind_name", CHEAP_KINDS)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_built_in_node_with_generated_arguments_reloads_bit_for_bit(kind_name, data):
    kind = KINDS[kind_name]
    kwargs = data.draw(kind.kwargs, label="kwargs")
    check_config_round_trip(kind.graph(kwargs), REGISTRY)


@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_generated_graph_of_every_node_kind_reloads_bit_for_bit(data):
    """Transforms, additive edges, units, mappings, external inputs, ParamSpec
    overrides and calibrated leaves over all seven kinds; no coupling group
    (their compile is the slow lane's)."""
    from tests.property.strategies import ALL_NODE_KINDS, NODE_REGISTRY, graph_recipes

    recipe = data.draw(graph_recipes(kinds=ALL_NODE_KINDS, max_nodes=3,
                                     allow_coupling_groups=False), label="recipe")
    note(f"recipe: {recipe}")
    check_config_round_trip(recipe.build(), dict(NODE_REGISTRY))


@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_to_dict_of_a_generated_graph_of_every_node_kind_is_idempotent(data):
    """``to_dict(from_dict(to_dict(g))) == to_dict(g)``, with coupling groups.
    The reload is never compiled, but the recipe's own graph is (``build``
    compiles), so this is a costly tier: at the profile's depth it took 5 s
    on CI."""
    from tests.property.strategies import ALL_NODE_KINDS, NODE_REGISTRY, graph_recipes

    recipe = data.draw(graph_recipes(kinds=ALL_NODE_KINDS), label="recipe")
    gm = recipe.build()
    once = json.loads(json.dumps(gm.to_dict(), allow_nan=True))
    twice = GraphManager.from_dict(once, dict(NODE_REGISTRY)).to_dict()
    assert json.loads(json.dumps(twice, allow_nan=True)) == once


def test_every_built_in_node_has_a_catalogue_entry():
    """Fails closed: a node class exported by ``maddening.nodes`` or
    ``maddening.nodes.adaptive`` with no ``node_catalogue`` entry is a node
    the differential harness never builds."""
    exported = {
        obj for module in (builtin_nodes, adaptive_nodes) for name in module.__all__
        if inspect.isclass(obj := getattr(module, name))
        and issubclass(obj, SimulationNode) and not inspect.isabstract(obj)
    }
    assert len(exported) >= 11, sorted(c.__name__ for c in exported)
    catalogued = {kind.cls for kind in KINDS.values()}
    missing = sorted(c.__name__ for c in exported - catalogued)
    assert not missing, f"no node_catalogue entry for {missing}"


def test_a_group_that_declares_a_dead_band_on_three_members_reloads_bit_for_bit():
    """A config carries a group's ``atol``, so the reload of a group that
    declares a dead band on three or more members is advised on a second
    time when it compiles (MADD-ANO-254), and the harness expects that by
    name (``differential.reload_from_config``).  The slow test of coupled
    graphs below draws such groups, and its per-push witness draws no
    group at all: on the first slow lane after the advisory was widened
    the reload raised on three BallNodes with ``atol=1e-08``, the group
    built here."""
    from maddening.nodes.ball import BallNode

    gm = GraphManager()
    for name, gravity in (("rod", -11.0), ("node_1", -12.0), ("Ball", -12.0)):
        gm.add_node(BallNode(name, 0.01, initial_position=0.0, initial_velocity=0.0,
                             elasticity=0.0, gravity=gravity))
    gm.add_edge("rod", "node_1", "position", "table_position")
    gm.add_edge("node_1", "Ball", "position", "table_position")
    gm.add_coupling_group(["node_1", "rod", "Ball"], max_iterations=2, tolerance=1e-3,
                          convergence_norm="l2", atol=1e-8, rtol=1e-6)
    with pytest.warns(UserWarning, match="declares a dead band on 3 members"):
        gm.compile()
    check_config_round_trip(gm, {"BallNode": BallNode})


# ---------------------------------------------------------------------------
# Slow lane
# ---------------------------------------------------------------------------

# Per push: tests/property/test_differential_serialisation.py::test_a_built_in_node_with_generated_arguments_reloads_bit_for_bit
@pytest.mark.slow  # an LBM / pipe / wavelet compile per example: 2-6 s each
@pytest.mark.parametrize("kind_name", COSTLY_KINDS)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_costly_built_in_node_with_generated_arguments_reloads_bit_for_bit(kind_name, data):
    kind = KINDS[kind_name]
    kwargs = data.draw(kind.kwargs, label="kwargs")
    check_config_round_trip(kind.graph(kwargs), REGISTRY)


# Per push: tests/property/test_differential_serialisation.py::test_a_generated_graph_of_every_node_kind_reloads_bit_for_bit
@pytest.mark.slow  # coupled graphs built and compiled per example
@settings(max_examples=EXAMPLES_STANDARD, derandomize=True)
@given(data=st.data())
def test_a_generated_coupled_graph_of_every_node_kind_reloads_bit_for_bit(data):
    from tests.property.strategies import ALL_NODE_KINDS, NODE_REGISTRY, graph_recipes

    recipe = data.draw(graph_recipes(kinds=ALL_NODE_KINDS, require_coupling_group=True),
                       label="recipe")
    note(f"recipe: {recipe}")
    check_config_round_trip(recipe.build(), dict(NODE_REGISTRY))
