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

``compile()`` is held to the same invariant, at the bottom of the file.
It derives the whole description of the step -- the schedule, the
back-edge set, the multi-rate flag and dividers, the external-input
zeros, the parameter snapshot and its dtypes and shapes -- and three
things after those writes can still raise.  A failed compile must leave
the graph describing the step it is still running, not one that was
never built.
"""

import pytest
import jax.numpy as jnp
from hypothesis import given, settings, strategies as st

from tests.conftest import EXAMPLES_COSTLY

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.params import ParamSpec
from maddening.core.static_data import StaticArray
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


def _param_values(params: dict) -> dict:
    """``gm.params`` by value, both sections.  A rollback that kept the
    leaf names but not the fitted values would still be a corruption --
    and ``compile()`` is where a leaf that no longer fits is *dropped*."""
    return {
        section: {
            owner: {k: jnp.asarray(v).tolist() for k, v in leaves.items()}
            for owner, leaves in (params.get(section) or {}).items()
        }
        for section in ("nodes", "mappings")
    }


def _step_description(gm: GraphManager) -> dict:
    """What the graph says about the step it is currently running.

    All of it is derived by ``compile()``, and each entry is read by
    something that takes it for a description of the step that was
    *built*: ``_rate_dividers`` by the next compile's sub-step phase
    check, ``_is_multirate`` by ``run_adaptive``'s refusal (which reads
    it *before* it recompiles), ``_schedule`` and ``_rate_dividers`` by
    the ``schedule`` / ``rate_dividers`` properties, ``params`` by the
    step closure's ``params=None`` default, and ``_params_shapes`` /
    ``_params_dtypes`` by the validation of an explicit ``params``.
    """
    return {
        "schedule": list(gm._schedule),
        "back_edges": sorted(e.key for e in gm._back_edges),
        "is_multirate": gm._is_multirate,
        "rate_dividers": dict(gm._rate_dividers),
        "committed_dividers": dict(gm._committed_rate_dividers),
        "ext_leaves": sorted(
            (n, f) for n, f in (getattr(gm, "_default_ext_leaves", None) or {})
        ),
        "param_values": _param_values(gm.params),
        "param_dtypes": {
            s: {o: {k: str(v) for k, v in ls.items()} for o, ls in d.items()}
            for s, d in (getattr(gm, "_params_dtypes", None) or {}).items()
        },
        "param_shapes": {
            s: {o: dict(ls) for o, ls in d.items()}
            for s, d in (getattr(gm, "_params_shapes", None) or {}).items()
        },
        "dirty": gm._dirty,
        "compile_generation": gm._compile_generation,
        "compiled_step": id(gm._compiled_step),
    }


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
        "step": _step_description(gm),
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


# -------------------------------------------------------------- ``compile``
#
# ``compile()`` is a mutator too, and the least obvious one: what it writes
# is not the graph the caller built but the *derived description of the step*
# -- the schedule, the back-edge set, the multi-rate flag and dividers, the
# external-input zeros, the parameter snapshot with its dtypes and shapes.
# Three things after those writes can still raise (the ``accelerated_fields``
# validation, the static-data/trainable-parameter refusal, and
# ``_build_step_fn`` itself), and each of the derived values is read by
# something that takes it for a description of the step that was *built*.
#
# The invariant tested here is the general one -- a failed ``compile()``
# leaves the graph bit-identical -- rather than one field at a time, because
# the damage has twice come from a field nobody had thought to check:
# ``_rate_dividers`` first (four steps of sub-step phase lost on the *repair*
# path) and then ``_is_multirate``, which ``run_adaptive`` reads to refuse a
# multi-rate graph, before it recompiles.


class _BuildRefused(RuntimeError):
    """What a ``_build_step_fn`` that cannot build the step raises."""


class _StaticFromTrainableNode(SimulationNode):
    """Declares a static derived from a trainable parameter, which
    ``compile()`` refuses outright -- after it has derived everything."""

    def __init__(self, name, timestep, **params):
        super().__init__(name, timestep, **{"gain": 1.5, **params})
        self._table = jnp.arange(2, dtype=jnp.float32)

    @property
    def static_data(self) -> dict:
        return {"table": StaticArray(self._table)}

    def static_data_deps(self) -> dict:
        return {"table": ("gain",)}

    def param_specs(self) -> dict:
        return {**super().param_specs(), "gain": ParamSpec(trainable=True)}

    def initial_state(self) -> dict:
        return {"y": jnp.zeros(())}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = {**self.params, **(params or {})}
        return {"y": state["y"] + p["gain"] * self._table[0] * dt}


def _compiled_graph(*, multirate: bool, steps: int = 0) -> GraphManager:
    """A graph that compiles, optionally already some steps into a run so it
    has a sub-step phase and warm starts to lose."""
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01))
    gm.add_node(BallNode(name="ball", timestep=0.03 if multirate else 0.01))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.add_external_input("ball", "force")
    gm.compile()
    if steps:
        gm.run(steps)
    return gm


# --- the three routes by which ``compile()`` can raise after its writes ---
#
# Each arms an already-compiled graph so the *next* ``compile()`` fails, and
# returns the exception it owes the caller.  ``subcycling`` adds a subcycling
# coupling group: that is the shape that collapses a multi-rate graph to
# uniform rate on the way through, so the failed attempt's ``_rate_dividers``
# and ``_is_multirate`` disagree with the committed ones rather than merely
# repeating them.


def _arm_accelerated_fields(gm: GraphManager, *, subcycling: bool):
    gm.add_coupling_group(
        ["table", "ball"],
        subcycling=subcycling,
        acceleration="iqn-ils",
        accelerated_fields={"ball": ["no_such_field"]},
    )
    return ValueError, "not a state field"


def _arm_static_data_refusal(gm: GraphManager, *, subcycling: bool):
    if subcycling:
        gm.add_coupling_group(["table", "ball"], subcycling=True)
    gm.add_node(_StaticFromTrainableNode(name="baked", timestep=0.01))
    return ValueError, "static_data"


def _arm_build_failure(gm: GraphManager, *, subcycling: bool):
    if subcycling:
        gm.add_coupling_group(["table", "ball"], subcycling=True)
    else:
        gm.add_external_input("table", "push")   # something to recompile for

    def _refuse(*args, **kwargs):
        raise _BuildRefused("the step could not be built")

    gm._build_step_fn = _refuse                  # noqa: SLF001
    return _BuildRefused, "could not be built"


_COMPILE_FAILURES = [
    ("accelerated_fields names a field that is not a state field",
     _arm_accelerated_fields),
    ("a static derived from a trainable parameter", _arm_static_data_refusal),
    ("_build_step_fn cannot build the step", _arm_build_failure),
]
_COMPILE_FAILURE_IDS = [label for label, _ in _COMPILE_FAILURES]

#: ``(multirate, subcycling)``.  A coupling group over nodes with different
#: timesteps *must* subcycle -- ``validate()`` refuses the alternative before
#: ``compile()`` writes anything -- so the multi-rate shape only appears with
#: the flag on.  That is also the shape that matters: subcycling collapses
#: the graph to its group's macro timestep, so the failed attempt's dividers
#: and multi-rate flag genuinely disagree with the committed ones instead of
#: repeating them.
_GRAPH_SHAPES = [
    pytest.param(False, False, id="uniform"),
    pytest.param(False, True, id="uniform+subcycling-group"),
    pytest.param(True, True, id="multirate+subcycling-group"),
]


@pytest.mark.parametrize("arm", [fn for _, fn in _COMPILE_FAILURES],
                         ids=_COMPILE_FAILURE_IDS)
@pytest.mark.parametrize("multirate,subcycling", _GRAPH_SHAPES)
def test_a_failed_compile_leaves_the_graph_exactly_as_it_was(
    arm, multirate, subcycling,
):
    gm = _compiled_graph(multirate=multirate, steps=4 if multirate else 0)
    expected, match = arm(gm, subcycling=subcycling)
    before = _snapshot(gm)

    with pytest.raises(expected, match=match):
        gm.compile()

    assert _snapshot(gm) == before


def test_a_failed_compile_leaves_a_calibrated_parameter_alone():
    """``compile()`` rewrites ``params`` from the nodes' constructor values
    with the live leaves merged back over them, and *drops* a leaf that no
    longer fits.  A failed compile that committed that rewrite would throw
    away a fit -- and the warning that says so has already been issued, so
    the repair recompile is silent about it."""
    gm = _compiled_graph(multirate=False)
    node_params = gm.params["nodes"]
    owner = next(n for n, leaves in node_params.items() if leaves)
    key = sorted(node_params[owner])[0]
    fitted = jnp.asarray(node_params[owner][key]) + 1.0
    gm.params["nodes"][owner][key] = fitted

    _arm_accelerated_fields(gm, subcycling=False)
    with pytest.raises(ValueError, match="not a state field"):
        gm.compile()

    assert jnp.array_equal(gm.params["nodes"][owner][key], fitted)


def test_a_failed_compile_does_not_defeat_the_multi_rate_refusal():
    """The stale field turning into a wrong *number*.

    ``run_adaptive`` refuses a multi-rate graph -- adaptive timestepping
    and per-node rate dividers cannot both decide when a node fires -- and
    it reads ``_is_multirate`` *before* the recompile that would correct
    it.  A failed compile of a subcycling multi-rate graph had collapsed
    the flag to ``False`` (the failed attempt demoted the graph to uniform
    rate), so once the cause was fixed the refusal no longer fired and the
    graph was integrated adaptively instead.
    """
    gm = _compiled_graph(multirate=True, steps=4)
    with pytest.raises(RuntimeError, match="multi-rate"):
        gm.run_adaptive(t_end=0.05)

    _arm_accelerated_fields(gm, subcycling=True)
    with pytest.raises(ValueError, match="not a state field"):
        gm.compile()
    assert gm.is_multirate is True

    gm.remove_coupling_group(["table", "ball"])          # the repair
    with pytest.raises(RuntimeError, match="multi-rate"):
        gm.run_adaptive(t_end=0.05)


@pytest.mark.parametrize("arm", [fn for _, fn in _COMPILE_FAILURES],
                         ids=_COMPILE_FAILURE_IDS)
def test_the_run_continues_unchanged_after_a_failed_compile_is_repaired(arm):
    """The point of the invariant: the trajectory of a graph that hit a
    failed compile and was fixed is the trajectory of one that never did."""
    reference = _compiled_graph(multirate=True, steps=4)
    reference.run(4)

    gm = _compiled_graph(multirate=True, steps=4)
    expected, match = arm(gm, subcycling=True)
    with pytest.raises(expected, match=match):
        gm.compile()
    # Undo the arming, exactly as a user fixing the reported cause would.
    gm.remove_coupling_group(["table", "ball"])
    if "baked" in gm.node_names:
        gm.remove_node("baked")
    if hasattr(gm, "_build_step_fn") and "_build_step_fn" in vars(gm):
        del gm._build_step_fn                            # noqa: SLF001
    gm.run(4)

    for field, expected_value in reference.get_node_state("ball").items():
        assert jnp.array_equal(
            gm.get_node_state("ball")[field], expected_value
        ), field


@settings(max_examples=EXAMPLES_COSTLY)  # each draw compiles and runs a graph
@given(
    route=st.sampled_from(range(len(_COMPILE_FAILURES))),
    shape=st.sampled_from([(m, s) for m, s in
                           [(p.values[0], p.values[1]) for p in _GRAPH_SHAPES]]),
    steps=st.integers(min_value=0, max_value=5),
    repeats=st.integers(min_value=1, max_value=3),
)
def test_no_sequence_of_failed_compiles_moves_the_graph(
    route, shape, steps, repeats,
):
    multirate, subcycling = shape
    """Over generated graphs and generated failure injections: a failed
    ``compile()`` is a no-op, and repeating it stays a no-op (a rollback
    that worked once and then drifted would be worse than none)."""
    gm = _compiled_graph(multirate=multirate, steps=steps)
    _, arm = _COMPILE_FAILURES[route]
    expected, match = arm(gm, subcycling=subcycling)
    before = _snapshot(gm)

    for _ in range(repeats):
        with pytest.raises(expected, match=match):
            gm.compile()
        assert _snapshot(gm) == before


def test_auto_couple_keeps_the_groups_it_cannot_replace():
    """``auto_couple`` cleared ``_coupling_groups`` and then built the new
    ones one at a time, so a knob the ``CouplingGroup`` constructor refuses
    destroyed the groups the graph already had and put nothing back."""
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01))
    gm.add_node(BallNode(name="ball", timestep=0.01))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.add_edge("ball", "table", "position", "ball_position")
    gm.add_coupling_group(["table", "ball"], max_iterations=7)
    before = _snapshot(gm)

    with pytest.raises(TypeError):
        gm.auto_couple(no_such_knob=1)

    assert _snapshot(gm) == before
    assert [g.max_iterations for g in gm._coupling_groups] == [7]
