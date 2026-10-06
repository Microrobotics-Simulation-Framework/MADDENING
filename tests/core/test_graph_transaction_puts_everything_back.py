"""A graph's transaction snapshot puts back everything the graph holds.

``GraphManager._transaction_snapshot`` / ``_transaction_restore`` are what
the REST server's write routes stand on: a request that fails is undone by
restoring the snapshot taken before it.  A restore that misses one piece of
the graph is a silent defect, so the snapshot is not a list of attributes
(``maddening.core._graph_transaction``): it records every container
reachable from the graph.  These tests hold it to that from outside --

* **every attribute, one at a time**: each attribute the graph has is
  rebound, and each container it holds is changed in place; after the
  restore the graph is object for object what it was
  (``tests/property/graph_fingerprint.py``, written apart from the
  snapshot and stricter than it).  An attribute added to ``GraphManager``
  tomorrow is in this loop without anyone listing it;
* the pieces a route actually moves: a node's own ``params`` (and the
  write counts the graph syncs from), a live leaf written in place, the
  state, a node added, an edge removed, a compile, a step;
* what it shares rather than copies: every array, so the cost does not grow
  with a field -- and every *kind* of object it shares is one this module
  has classified, so a new mutable kind fails here instead of being shared
  unnoticed.
"""

from __future__ import annotations

import dataclasses
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core._graph_transaction import _Snapshot
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes import BallNode, HeatNode, SpringDamperNode, TableNode

from tests.property.differential import quiet
from tests.property.graph_fingerprint import (
    assert_exactly_as_it_was,
    fingerprint,
    fingerprint_differences,
)

DT = 0.01


def _graph(n_cells: int = 8) -> GraphManager:
    """A compiled, stepped graph with an edge, a fitted leaf and a node
    whose params hold a list (a value that can change in place)."""
    gm = GraphManager()
    gm.add_node(HeatNode("rod", DT, n_cells=n_cells, length=1.0,
                         # the same Fourier number whatever the cell count
                         thermal_diffusivity=1e-4 * (8 / n_cells) ** 2,
                         initial_temperature=1.0 if n_cells != 8 else [0.5] * 8))
    gm.add_node(SpringDamperNode("spring", DT, stiffness=40.0, damping=0.5,
                                 initial_position=0.5))
    gm.add_node(BallNode("ball", DT, initial_position=3.0))
    gm.add_edge("spring", "ball", "position", "table_position")
    with quiet():
        gm.compile()
        gm.step()
    gm.params["nodes"]["spring"]["stiffness"] = jnp.asarray(20.0, dtype=jnp.float32)
    return gm


class _Sentinel:
    """Put where an attribute's value was."""


# ---------------------------------------------------------------------------
# Every attribute, one at a time
# ---------------------------------------------------------------------------

def _attribute_names() -> list[str]:
    return sorted(vars(_graph()))


def _change_in_place(value) -> bool:
    """Change a container in place; whether there was one to change."""
    if isinstance(value, dict):
        dict.__setitem__(value, "<added in place>", _Sentinel())
        return True
    if isinstance(value, list):
        value.append(_Sentinel())
        return True
    if isinstance(value, set):
        value.add("<added in place>")
        return True
    return False


@pytest.mark.parametrize("name", _attribute_names())
def test_an_attribute_rebound_or_changed_in_place_is_put_back(name):
    """Each attribute of the graph, by name: its container changed in place
    (when it is one), then the attribute rebound to something else, then
    deleted.  The restore puts back the object it was, holding what it
    held."""
    gm = _graph()
    before = fingerprint(gm)
    snapshot = gm._transaction_snapshot()
    held = vars(gm)[name]
    _change_in_place(held)
    if isinstance(held, dict):
        for inner in list(dict.values(held)):
            _change_in_place(inner)
    vars(gm)[name] = _Sentinel()
    assert fingerprint_differences(before, fingerprint(gm)), name
    gm._transaction_restore(snapshot)
    assert_exactly_as_it_was(before, fingerprint(gm), f"rebinding {name}")
    assert vars(gm)[name] is held
    del vars(gm)[name]
    gm._transaction_restore(snapshot)
    assert_exactly_as_it_was(before, fingerprint(gm), f"deleting {name}")


def test_the_graph_has_the_attributes_the_restore_is_documented_to_cover():
    """The table in the pull request that introduced the transaction, and
    the docstring of ``_graph_transaction``, name these pieces.  All of
    them are attributes of a compiled graph (so the loop above ran on
    each), and an attribute set after the snapshot is removed by the
    restore."""
    gm = _graph()
    named = {"_nodes", "_edges", "_state", "_params", "_coupling_groups", "_external_inputs",
             "_dirty", "_compiled_step", "_committed_rate_dividers", "_rate_dividers",
             "_scan_cache", "_schedule", "_node_writes_seen", "_param_spec_overrides",
             "_compile_generation", "_observers", "_committed_coupling_groups"}
    assert named <= set(vars(gm)), sorted(named - set(vars(gm)))
    snapshot = gm._transaction_snapshot()
    gm._made_by_a_failed_request = True
    gm._transaction_restore(snapshot)
    assert not hasattr(gm, "_made_by_a_failed_request")


# ---------------------------------------------------------------------------
# What a route moves
# ---------------------------------------------------------------------------

def _moves():
    def node_param(gm):
        gm._nodes["spring"].node.params["damping"] = 3.0

    def node_param_in_place(gm):
        gm._nodes["rod"].node.params["initial_temperature"][0] = 9.0

    def node_params_replaced(gm):
        node = gm._nodes["spring"].node
        node.params = {**node.params, "mass": 5.0}

    def node_attribute(gm):
        gm._nodes["ball"].node.some_cache = {"built": True}

    def live_leaf(gm):
        vars(gm)["_params"]["nodes"]["spring"]["stiffness"] = jnp.asarray(1.0, jnp.float32)

    def synced_write(gm):
        gm._nodes["spring"].node.params["stiffness"] = 77.0
        gm.params       # the read that takes the node write in

    def state(gm):
        gm.set_node_state("ball", {"position": jnp.asarray(9.0, jnp.float32),
                                   "velocity": jnp.asarray(1.0, jnp.float32)})

    def add_node(gm):
        gm.add_node(TableNode("table", DT))

    def remove_node(gm):
        gm.remove_node("ball")

    def remove_edge(gm):
        gm.remove_edge("spring", "ball", "position", "table_position")

    def external_input(gm):
        gm.add_external_input("spring", "external_force", (), jnp.float32)

    def structural_then_compile(gm):
        gm._nodes["rod"].node.params["stencil_order"] = 4
        with quiet():
            gm.compile()

    def step(gm):
        with quiet():
            gm.step()

    def scan(gm):
        with quiet():
            gm.run_scan(3)

    def reset(gm):
        gm.reset_state()

    def node_spec(gm):
        gm._nodes["rod"].timestep = 5.0

    return [node_param, node_param_in_place, node_params_replaced, node_attribute, live_leaf,
            synced_write, state, add_node, remove_node, remove_edge, external_input,
            structural_then_compile, step, scan, reset, node_spec]


@pytest.mark.parametrize("move", _moves(), ids=lambda f: f.__name__)
def test_what_a_write_moves_is_put_back(move):
    gm = _graph()
    before = fingerprint(gm)
    snapshot = gm._transaction_snapshot()
    move(gm)
    assert fingerprint_differences(before, fingerprint(gm)), "the move changed nothing"
    gm._transaction_restore(snapshot)
    assert_exactly_as_it_was(before, fingerprint(gm), move.__name__)


@pytest.mark.parametrize("move", _moves(), ids=lambda f: f.__name__)
def test_a_restored_graph_runs_as_one_that_was_never_touched(move):
    """Object identity is the strict half; this is the observable one.  A
    graph moved and restored, and its twin left alone, step to the same
    bits, with no recompile the twin does not make -- so the restore left
    no bookkeeping saying a write is pending."""
    gm, twin = _graph(), _graph()
    snapshot = gm._transaction_snapshot()
    move(gm)
    gm._transaction_restore(snapshot)
    traces = gm.trace_count
    with quiet():
        gm.step()
        twin.step()
    assert gm.trace_count == traces, "the restored graph recompiled"
    for name, fields in twin._state.items():
        for field, value in fields.items():
            assert np.array_equal(np.asarray(gm._state[name][field]), np.asarray(value),
                                  equal_nan=True), (name, field)
    for name, leaves in twin.params["nodes"].items():
        for key, value in leaves.items():
            assert np.array_equal(np.asarray(gm.params["nodes"][name][key]),
                                  np.asarray(value)), (name, key)


def test_a_node_write_counted_before_the_snapshot_is_still_pending_after_it():
    """The snapshot reads no property: a ``node.params`` write the graph
    has not taken in stays pending through a snapshot and a restore, and is
    taken in by the next read exactly once."""
    gm = _graph()
    gm._nodes["spring"].node.params["damping"] = 2.0
    snapshot = gm._transaction_snapshot()
    assert float(vars(gm)["_params"]["nodes"]["spring"]["damping"]) == 0.5
    gm._nodes["spring"].node.params["damping"] = 4.0
    gm.params
    gm._transaction_restore(snapshot)
    assert gm._nodes["spring"].node.params["damping"] == 2.0
    assert float(gm.params["nodes"]["spring"]["damping"]) == 2.0


def test_taking_a_snapshot_changes_nothing():
    gm = _graph()
    before = fingerprint(gm)
    gm._transaction_snapshot()
    assert_exactly_as_it_was(before, fingerprint(gm), "taking a snapshot")


def test_a_restore_is_repeatable_and_a_later_snapshot_is_independent():
    gm = _graph()
    first = gm._transaction_snapshot()
    before = fingerprint(gm)
    gm.remove_node("ball")
    second = gm._transaction_snapshot()
    after_removal = fingerprint(gm)
    gm.add_node(TableNode("table", DT))
    gm._transaction_restore(second)
    assert_exactly_as_it_was(after_removal, fingerprint(gm), "the second snapshot")
    gm._transaction_restore(first)
    gm._transaction_restore(first)
    assert_exactly_as_it_was(before, fingerprint(gm), "the first snapshot, twice")


def test_further_roots_are_recorded_with_the_graph():
    """A server's own containers about the graph (its record of the
    surrogates) are restored by the same snapshot."""
    gm = _graph()
    originals: dict = {"rod": (gm._nodes["rod"].node, [], [])}
    active = {"rod"}
    snapshot = gm._transaction_snapshot(also=(originals, active))
    originals["rod"][1].append("an edge")
    originals["spring"] = ()
    active.clear()
    gm._transaction_restore(snapshot)
    assert list(originals) == ["rod"] and originals["rod"][1] == [] and active == {"rod"}


def test_a_container_is_recorded_however_deep_and_through_whatever_holds_it():
    """A dict in a list in a tuple in a set-holding dict: each level is
    walked, whatever kind of container holds the next."""
    inner, members = {"a": 1}, {1, 2}
    held = [(inner, [members])]
    snapshot = _Snapshot([{"held": held}])
    inner["b"] = 2
    del inner["a"]
    members.add(3)
    held.append("more")
    snapshot.restore()
    assert inner == {"a": 1} and members == {1, 2} and held == [(inner, [members])]
    assert held[0][0] is inner


# ---------------------------------------------------------------------------
# What it shares
# ---------------------------------------------------------------------------

#: Every kind of object the snapshot shares instead of recording, on the
#: stock graphs: each cannot change, or is code.  ``numpy.ndarray`` is not
#: here: a node holding one makes the test below say so, and the answer is
#: in the module's docstring (shared; no route writes one in place).
SHARED_KINDS = (
    type(None), bool, int, float, str, complex, bytes,
    jax.Array,                       # immutable
    np.dtype, np.generic,            # immutable
    types.FunctionType, types.MethodType, types.BuiltinFunctionType,   # code
    type(jax.jit(lambda x: x)),      # a compiled function
    object,                          # a bare ``object()``: an identity (a params lineage)
    frozenset, range, slice,
)


def test_every_kind_of_object_the_snapshot_shares_is_classified():
    leaves: dict = {}
    gm = _graph()
    with quiet():
        gm.run_scan(2)
    gm._transaction_snapshot(leaves=leaves)
    unclassified = sorted(
        f"{kind.__module__}.{kind.__qualname__}" for kind in leaves
        if not (kind is object or (kind is not object and issubclass(kind, tuple(
            k for k in SHARED_KINDS if k is not object)))))
    assert not unclassified, (
        f"the snapshot shares {unclassified} without recording them: say in "
        "maddening/core/_graph_transaction.py whether each can change in place, "
        "and list it in SHARED_KINDS if it cannot")
    assert any(issubclass(kind, jax.Array) for kind in leaves)


def test_a_snapshot_copies_no_field_whatever_its_size():
    """A million-cell field is one reference in the snapshot: the number of
    containers recorded is that of the eight-cell graph, the arrays of the
    restored state are the objects that were there, and a NumPy array a
    node holds is shared too."""
    small, large = _graph(8), _graph(1_000_000)
    held = np.zeros(1_000_000, dtype=np.float32)
    large._nodes["rod"].node.held_by_the_node = held
    snapshot = large._transaction_snapshot()
    assert abs(len(snapshot) - len(small._transaction_snapshot())) <= 2
    arrays = {field: value for field, value in large._state["rod"].items()}
    large.set_node_state("rod", {"temperature": jnp.zeros(1_000_000, jnp.float32)})
    large._transaction_restore(snapshot)
    assert all(large._state["rod"][field] is value for field, value in arrays.items())
    assert large._nodes["rod"].node.held_by_the_node is held
    recorded = [copy for _, copy, kind in snapshot._records if kind != "tuple"]
    assert not any(copy is held or isinstance(copy, np.ndarray) for copy in recorded)


def test_an_object_from_outside_the_package_is_shared_unless_it_is_a_node_or_dataclass():
    """The boundary the module's docstring states, pinned: a plain object
    of a class defined elsewhere is a leaf (its attributes are not put
    back); a node defined elsewhere is opened."""
    class Plain:
        def __init__(self):
            self.value = 1

    class Elsewhere(SimulationNode):
        def initial_state(self):
            return {"x": jnp.zeros(())}

        def update(self, state, boundary_inputs, dt):
            return state

    @dataclasses.dataclass
    class Record:
        items: list

    plain, node, record = Plain(), Elsewhere("n", DT), Record([1])
    snapshot = _Snapshot([{"plain": plain, "node": node, "held": (record,)}])
    plain.value, node.added = 2, True
    record.items.append(2)
    record.items = [3]
    snapshot.restore()
    assert plain.value == 2
    assert not hasattr(node, "added")
    assert record.items == [1]
