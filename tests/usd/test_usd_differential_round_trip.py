"""Differential oracle 4, USD half: a stage reloads to the original graph.

``save_graph_to_usd`` -> ``load_graph_from_usd`` owes what the config owes
(``tests/property/test_differential_serialisation.py``): the same structure,
initial state, parameter pytree, ``param_specs()`` and trajectory, bit for
bit -- for every cheap built-in node with generated constructor arguments,
for generated graphs over every node kind ``strategies`` can build, and
against the config reload of the same graph (two spellings of one graph).

Not slow-marked, on purpose: only the ``test-usd`` job installs
``usd-core``, it runs every file under ``tests/usd/`` and it runs no slow
test (``tests/compliance/test_ci_workflows.py`` enforces both), so a slow
mark here would run it nowhere.  The lanes that judge test time skip this
directory (``tests/usd/conftest.py``).

Tolerance: none.
"""

from __future__ import annotations

import warnings

from hypothesis import given, note, settings
from hypothesis import strategies as st
from pxr import Usd

import pytest

from maddening.core.graph_manager import GraphManager
from maddening.usd.serialization import load_graph_from_usd, save_graph_to_usd

from tests.conftest import EXAMPLES_COSTLY
from tests.property.differential import (
    assert_trees_identical,
    full_state,
    params_tree,
    rollout,
)
from tests.property.invariants import (
    assert_param_specs_identical,
    assert_structure_identical,
    structure,
)
from tests.property.node_catalogue import CHEAP_KINDS, KINDS, REGISTRY

N_STEPS = 3


def _initial_states(gm: GraphManager) -> dict:
    import numpy as np

    return {name: {k: np.asarray(v) for k, v in gm.get_node(name).initial_state().items()}
            for name in gm.node_names}


def _reload_usd(gm: GraphManager, registry: dict) -> GraphManager:
    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    from tests.property.strategies import DRAWN_DEAD_BAND

    reloaded = load_graph_from_usd(stage, node_registry=registry)
    with warnings.catch_warnings():
        # A drawn group can declare a dead band, which compile() advises
        # on for any member count (MADD-ANO-254); the round trip carries the
        # group's ``atol`` like every other field.  Nothing else is filtered.
        warnings.filterwarnings("ignore", message=DRAWN_DEAD_BAND, category=UserWarning)
        reloaded.compile()
    return reloaded


def _check(gm: GraphManager, reloaded: GraphManager, *, what: str) -> None:
    assert_structure_identical(structure(gm), structure(reloaded))
    assert_trees_identical(_initial_states(gm), _initial_states(reloaded),
                           what=f"{what}: initial state")
    assert_trees_identical(full_state(gm), full_state(reloaded), what=f"{what}: state")
    assert_trees_identical(params_tree(gm), params_tree(reloaded), what=f"{what}: params")
    assert_param_specs_identical(gm.param_specs(), reloaded.param_specs())
    assert_trees_identical(rollout(gm, N_STEPS), rollout(reloaded, N_STEPS),
                           what=f"{what}: trajectory")


@pytest.mark.parametrize("kind_name", CHEAP_KINDS)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_built_in_node_with_generated_arguments_reloads_from_usd_bit_for_bit(
        kind_name, data):
    kind = KINDS[kind_name]
    kwargs = data.draw(kind.kwargs, label="kwargs")
    gm = kind.graph(kwargs)
    _check(gm, _reload_usd(gm, REGISTRY), what="usd")


@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_generated_graph_of_every_node_kind_reloads_from_usd_bit_for_bit(data):
    """Coupling groups, mappings, transforms, ParamSpec overrides and
    calibrated leaves over every node kind ``strategies`` can build."""
    from tests.property.strategies import ALL_NODE_KINDS, NODE_REGISTRY, graph_recipes

    recipe = data.draw(graph_recipes(kinds=ALL_NODE_KINDS, max_nodes=3), label="recipe")
    note(f"recipe: {recipe}")
    # A rollout moves the graph it runs on, so the original and the reload
    # are built for this comparison alone.  Config against original is the
    # config half's (``test_differential_serialisation.py``); with this one
    # it makes the two formats agree with each other.
    _check(recipe.build(), _reload_usd(recipe.build(), dict(NODE_REGISTRY)), what="usd")
