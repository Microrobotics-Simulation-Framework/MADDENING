"""A failed graph mutation leaves the graph exactly as it was.

``GraphManager.add_node`` used to register the node's ``_NodeSpec`` and
*then* call ``initial_state()``.  Any node whose ``initial_state`` can
fail on a supported configuration therefore left a ghost behind:
``_nodes[name]`` populated, ``_state[name]`` missing.  The name was then
taken for good -- re-adding raised "already exists", ``remove_node``
raised ``KeyError`` off the missing state entry, ``compile()`` accepted
the graph, ``params["nodes"]`` carried a node that could never run, and
``step()`` died much later with a bare ``KeyError`` inside the compiled
step.

``AdaptiveNode`` is what made that pre-existing non-atomicity reachable
(it raises ``AdaptiveNodeBlindnessError`` by design at a Palais trap,
and the documented recovery is to perturb and re-add under the same
name), but the invariant is a graph invariant, so it is tested as one:
every mutator, against every way it is documented to fail.
"""

import pytest
import jax.numpy as jnp
from hypothesis import given, settings, strategies as st

from tests.conftest import EXAMPLES_COSTLY

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


class _RaisingInitialStateNode(SimulationNode):
    """A node whose ``initial_state()`` raises, as ``AdaptiveNode`` does at a
    Palais trap.  Stands in for it so the invariant is tested at the graph
    level without dragging the adaptive package into a core test."""

    class Refused(RuntimeError):
        pass

    def initial_state(self) -> dict:
        raise self.Refused("initial_state refuses at these parameters")

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        return state


class _FlakyInitialStateNode(SimulationNode):
    """Succeeds once (so it can be added) and refuses afterwards, which is
    what makes it a probe for ``reset_state``."""

    class Refused(RuntimeError):
        pass

    def __init__(self, name, timestep, **params):
        super().__init__(name, timestep, **params)
        self.armed = False

    def initial_state(self) -> dict:
        if self.armed:
            raise self.Refused("initial_state refuses on reset")
        return {"x": jnp.zeros(3)}

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        return state


def _snapshot(gm: GraphManager) -> dict:
    """Everything a caller can observe about the graph's structure.

    State arrays are compared by value, not by identity: a rollback that
    restored the keys but not the values would still be a corruption.
    """
    return {
        "nodes": list(gm._nodes),
        "specs": {n: id(s) for n, s in gm._nodes.items()},
        "state_keys": sorted(gm._state),
        "state": {
            n: {k: jnp.asarray(v).tolist() for k, v in s.items()}
            for n, s in gm._state.items()
        },
        "edges": [e.key for e in gm._edges],
        "external": [(e.target_node, e.target_field) for e in gm._external_inputs],
        "groups": [sorted(g.nodes) for g in gm._coupling_groups],
        "params": {
            n: sorted(p) for n, p in gm.params.get("nodes", {}).items()
        },
    }


def _populated_graph() -> GraphManager:
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01))
    gm.add_node(BallNode(name="ball", timestep=0.01))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.add_external_input("ball", "force")
    return gm


# --- the mutations that must fail, and the exception each owes the caller ---
#
# Each entry is ``(label, apply, expected_exception)``.  ``apply`` takes the
# graph and performs one mutation that is documented to fail.

_FAILING_MUTATIONS = [
    (
        "add_node whose initial_state raises",
        lambda gm: gm.add_node(_RaisingInitialStateNode(name="new", timestep=0.01)),
        _RaisingInitialStateNode.Refused,
    ),
    (
        "add_node under a name already taken",
        lambda gm: gm.add_node(BallNode(name="ball", timestep=0.01)),
        ValueError,
    ),
    (
        "add_node with a name containing a reserved token",
        lambda gm: gm.add_node(BallNode(name="a/b", timestep=0.01)),
        ValueError,
    ),
    (
        "add_node with an empty name",
        lambda gm: gm.add_node(BallNode(name="", timestep=0.01)),
        ValueError,
    ),
    (
        "remove_node of a name the graph does not have",
        lambda gm: gm.remove_node("absent"),
        KeyError,
    ),
    (
        "add_edge naming an unregistered transform",
        lambda gm: gm.add_edge(
            "table", "ball", "position", "table_position", transform="no_such_transform",
        ),
        KeyError,
    ),
    (
        "add_edge with an object that is not a Mapping",
        lambda gm: gm.add_edge(
            "table", "ball", "position", "table_position", mapping=object(),
        ),
        TypeError,
    ),
    (
        "add_coupling_group naming an unknown node",
        lambda gm: gm.add_coupling_group(["ball", "absent"]),
        KeyError,
    ),
    (
        "set_node_state for an unknown node",
        lambda gm: gm.set_node_state("absent", {"x": jnp.zeros(2)}),
        KeyError,
    ),
]

_MUTATION_IDS = [label for label, _, _ in _FAILING_MUTATIONS]


@pytest.mark.parametrize(
    "apply_mutation,expected",
    [(fn, exc) for _, fn, exc in _FAILING_MUTATIONS],
    ids=_MUTATION_IDS,
)
def test_a_failed_mutation_leaves_the_graph_exactly_as_it_was(apply_mutation, expected):
    gm = _populated_graph()
    before = _snapshot(gm)
    with pytest.raises(expected):
        apply_mutation(gm)
    assert _snapshot(gm) == before


@settings(max_examples=EXAMPLES_COSTLY)  # a graph is built and mutated per draw
@given(order=st.lists(st.sampled_from(range(len(_FAILING_MUTATIONS))),
                      min_size=1, max_size=6))
def test_any_sequence_of_failed_mutations_leaves_the_graph_exactly_as_it_was(order):
    """The invariant composes: no ordering of failures corrupts the graph,
    and none of them is recoverable-only-once."""
    gm = _populated_graph()
    before = _snapshot(gm)
    for i in order:
        _, apply_mutation, expected = _FAILING_MUTATIONS[i]
        with pytest.raises(expected):
            apply_mutation(gm)
        assert _snapshot(gm) == before


def test_a_name_freed_by_a_failed_add_node_can_be_used_again():
    """The recovery the developer guide documents for a Palais trap --
    perturb the parameters and re-add under the same name -- has to work."""
    gm = _populated_graph()
    with pytest.raises(_RaisingInitialStateNode.Refused):
        gm.add_node(_RaisingInitialStateNode(name="retry", timestep=0.01))
    gm.add_node(BallNode(name="retry", timestep=0.01))
    assert "retry" in gm.node_names
    assert "retry" in gm._state


def test_a_failed_add_node_leaves_no_node_the_compiled_step_cannot_run():
    """The ghost's real damage was downstream: ``compile()`` accepted it and
    ``step()`` died with a bare ``KeyError`` on the missing state entry."""
    gm = GraphManager()
    gm.add_node(BallNode(name="ball", timestep=0.01))
    with pytest.raises(_RaisingInitialStateNode.Refused):
        gm.add_node(_RaisingInitialStateNode(name="ghost", timestep=0.01))
    assert set(gm.node_names) == set(gm._state)
    gm.compile()
    assert "ghost" not in gm.params.get("nodes", {})
    gm.step()  # would have raised KeyError: 'ghost'


def test_a_failed_reset_state_leaves_every_node_at_its_previous_state():
    """``reset_state`` assigned into ``_state`` node by node, so a node whose
    ``initial_state`` refused left the graph half reset."""
    gm = GraphManager()
    gm.add_node(BallNode(name="ball", timestep=0.01))
    flaky = _FlakyInitialStateNode(name="flaky", timestep=0.01)
    gm.add_node(flaky)
    gm.set_node_state("ball", {**gm.get_node_state("ball"), "position": jnp.ones(3)})
    before = _snapshot(gm)

    flaky.armed = True
    with pytest.raises(_FlakyInitialStateNode.Refused):
        gm.reset_state()
    assert _snapshot(gm) == before


def test_a_node_whose_state_entry_is_missing_can_still_be_removed():
    """``remove_node`` used ``del self._state[name]``, so a graph that had
    somehow lost a state entry could not be repaired by removing the node."""
    gm = _populated_graph()
    del gm._state["ball"]
    gm.remove_node("ball")
    assert "ball" not in gm.node_names
    assert not [e for e in gm._edges if "ball" in (e.source_node, e.target_node)]


# ---------------------------------------------------------------- ``_meta``
#
# ``_meta`` is state too: it holds the multi-rate sub-step counter and the
# coupling predictor / IQN warm starts.  A mutator that rolls back ``_nodes``
# and ``_state`` but leaves ``_meta`` half-applied re-phases the schedule on
# the *rollback* path -- the same silent trajectory change that preserving
# ``_meta`` across a recompile was fixed to stop.  ``_snapshot`` above already
# reads ``_meta`` (it is a key of ``_state``); these are the cases that put
# something in it.


def _multirate_mid_run():
    """A multi-rate graph four base steps in, so it has a phase to lose."""
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01))
    gm.add_node(BallNode(name="ball", timestep=0.03))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    gm.run(4)
    assert int(gm._state["_meta"]["step_count"]) == 4
    return gm


def test_a_failed_reset_state_leaves_the_sub_step_phase_alone():
    gm = GraphManager()
    gm.add_node(BallNode(name="ball", timestep=0.01))
    flaky = _FlakyInitialStateNode(name="flaky", timestep=0.03)
    gm.add_node(flaky)
    gm.compile()
    gm.run(4)
    assert int(gm._state["_meta"]["step_count"]) == 4
    before = _snapshot(gm)

    flaky.armed = True
    with pytest.raises(_FlakyInitialStateNode.Refused):
        gm.reset_state()

    assert _snapshot(gm) == before
    assert int(gm._state["_meta"]["step_count"]) == 4


def test_a_failed_compile_leaves_the_sub_step_phase_alone():
    """``compile`` rebuilds ``_meta``; it must commit it only once it is done.

    The refusals after that point are real (``accelerated_fields`` naming a
    field that is not a state field, the static-data/trainable-parameter
    rule), and a graph that has to be fixed and recompiled must not have
    lost four steps of phase on the way.
    """
    gm = _multirate_mid_run()
    gm.add_coupling_group(
        ["table", "ball"],
        subcycling=True,                 # the two differ in timestep
        acceleration="iqn-ils",          # ... which accelerated_fields needs
        accelerated_fields={"ball": ["no_such_field"]},
    )
    before = _snapshot(gm)

    with pytest.raises(ValueError, match="not a state field"):
        gm.compile()

    assert _snapshot(gm) == before
    assert int(gm._state["_meta"]["step_count"]) == 4


def test_the_phase_survives_the_repair_and_the_run_continues():
    """The point of the rollback: fix the cause, recompile, carry on."""
    reference = _multirate_mid_run()
    reference.run(4)

    gm = _multirate_mid_run()
    gm.add_coupling_group(
        ["table", "ball"], subcycling=True, acceleration="iqn-ils",
        accelerated_fields={"ball": ["no_such_field"]},
    )
    with pytest.raises(ValueError):
        gm.compile()
    gm.remove_coupling_group(["table", "ball"])
    gm.run(4)

    for field, expected in reference.get_node_state("ball").items():
        assert jnp.array_equal(gm.get_node_state("ball")[field], expected), field
