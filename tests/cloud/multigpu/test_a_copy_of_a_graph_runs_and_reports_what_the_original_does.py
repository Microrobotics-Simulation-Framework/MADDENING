"""A copy of a graph, of a node or of a node's ``params`` runs and reports
what the original does, in every order of node write and calibration.

A node's ``params`` counts its writes, and a graph keeps, beside each node,
the counts it last saw.  ``copy.deepcopy(gm)`` copied the graph's record as
it was and the mapping with its counts back at zero, so the copy's first
sync read "fewer writes than I saw" as a write of that key and put the
node's value over the leaf: a graph with a node write in its past and a
calibration in ``gm.params`` after it ran the calibrated value, and its
deep copy ran, reported and saved the node's, with nothing said
(MADD-ANO-279; never released: no release has ``gm.params``).  A copy made
by ``copy.deepcopy`` or ``pickle`` now carries the counts and the lineage.
``copy.copy(gm)`` is refused: a shallow copy shared the graph's record of
the writes it had seen but not its compiled step, so after a write to a
node whose constants are baked into the step, whichever of the two graphs
stepped second ran the old value while both reported the new one.

Each case builds the same graph twice and makes the same statements on
both -- node writes and calibrations in one of the orders of ``ORDERS``,
some left pending (not yet seen by the graph) -- then takes a copy of one
of them, one of ``COPIES``.  The copy must hold the value the order leaves
in force (stated per order, so two graphs that are wrong alike do not
pass), report it in ``gm.params`` and in the saved config as the other
graph does, and run to the same state, bit for bit: the same program on
the same devices with the same constants.
"""

from __future__ import annotations

import copy
import pickle

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from tests.cloud.multigpu.test_a_params_write_on_a_wrapper_is_the_write_the_step_runs import (
    _HAS_4, CONTRACTS, MID, NEW, OLD, STEPS_AFTER, _global_x, _graph, _run,
)

pytestmark = pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")

CALIBRATED_EXTRA = 1.5


def _leaf(gm, key="rate"):
    return np.asarray(gm.params["nodes"]["n"][key]).tolist()


def _calibrate(gm, key, value):
    gm.params["nodes"]["n"][key] = jnp.asarray(value, jnp.float32)


# -- the orders -------------------------------------------------------------
# Each makes its statements and returns what is then in force:
# ``(rate, extra)``.  "Taken in" means the graph has seen the node write (a
# read of ``gm.params`` syncs); "pending" means it has not yet.

def _node_write_taken_in_then_calibration(gm, inner):
    gm.get_node("n").params["rate"] = MID
    assert _leaf(gm) == MID
    _calibrate(gm, "rate", NEW)                       # the later write: it wins
    return NEW, [0.0]


def _calibration_then_node_write_taken_in(gm, inner):
    _calibrate(gm, "rate", NEW)
    gm.get_node("n").params["rate"] = MID
    assert _leaf(gm) == MID
    return MID, [0.0]


def _calibration_then_node_write_pending(gm, inner):
    _calibrate(gm, "rate", NEW)
    gm.get_node("n").params["rate"] = MID             # not yet seen by the graph
    return MID, [0.0]


def _calibration_then_assignment_pending(gm, inner):
    _calibrate(gm, "extra", [CALIBRATED_EXTRA])
    node = gm.get_node("n")
    node.params = {**node.params, "rate": MID}        # writes rate, hands extra back
    return MID, [CALIBRATED_EXTRA]


def _the_nodes_own_value_written_back_over_a_calibration_pending(gm, inner):
    gm.get_node("n").params["rate"] = MID
    assert _leaf(gm) == MID
    _calibrate(gm, "rate", NEW)
    assert _leaf(gm) == NEW
    gm.get_node("n").params["rate"] = MID             # a write, whatever the value
    return MID, [0.0]


def _calibration_only(gm, inner):
    _calibrate(gm, "rate", NEW)
    return NEW, [0.0]


def _an_element_written_in_place_pending_beside_a_calibration(gm, inner):
    _calibrate(gm, "rate", NEW)
    gm.get_node("n").params["extra"][0] = CALIBRATED_EXTRA
    return NEW, [CALIBRATED_EXTRA]


def _two_node_writes_and_a_calibration_between(gm, inner):
    inner.params["rate"] = MID
    assert _leaf(gm) == MID
    _calibrate(gm, "rate", NEW)
    inner.params["extra"] = [CALIBRATED_EXTRA]        # another key, pending
    return NEW, [CALIBRATED_EXTRA]


ORDERS = {
    "node_write_taken_in_then_calibration": _node_write_taken_in_then_calibration,
    "calibration_then_node_write_taken_in": _calibration_then_node_write_taken_in,
    "calibration_then_node_write_pending": _calibration_then_node_write_pending,
    "calibration_then_assignment_pending": _calibration_then_assignment_pending,
    "the_nodes_own_value_written_back_pending":
        _the_nodes_own_value_written_back_over_a_calibration_pending,
    "calibration_only": _calibration_only,
    "an_element_written_in_place_pending": _an_element_written_in_place_pending_beside_a_calibration,
    "two_node_writes_and_a_calibration_between": _two_node_writes_and_a_calibration_between,
}


# -- the copies ---------------------------------------------------------------
# Each takes the graph the statements were made on and returns the graph to
# hold against the other one.

def _deep_copy_of_the_graph(gm):
    return copy.deepcopy(gm)


def _deep_copy_of_a_deep_copy(gm):
    return copy.deepcopy(copy.deepcopy(gm))


def _the_mapping_deep_copied(gm):
    node = gm.get_node("n")
    node.params = copy.deepcopy(node.params)
    return gm


def _the_mapping_pickled(gm):
    node = gm.get_node("n")
    node.params = pickle.loads(pickle.dumps(node.params))
    return gm


def _the_mapping_shallow_copied(gm):
    node = gm.get_node("n")
    node.params = copy.copy(node.params)
    return gm


COPIES = {
    "deep_copy_of_the_graph": _deep_copy_of_the_graph,
    "deep_copy_of_a_deep_copy": _deep_copy_of_a_deep_copy,
    "the_mapping_deep_copied": _the_mapping_deep_copied,
    "the_mapping_pickled": _the_mapping_pickled,
    "the_mapping_shallow_copied": _the_mapping_shallow_copied,
}

WRAPPERS = [("unsharded", 1), ("hybrid", 1), ("pointwise", 2), ("stencil", 2), ("unstructured", 2),
            ("hybrid_over_stencil", 2)]


def _check(kind, n, order, copied):
    def stated():
        gm, inner, read = _graph(kind, n, "params")
        gm.step()                                     # the step is traced before anything is written
        return gm, read, ORDERS[order](gm, inner)

    other, other_read, (rate, extra) = stated()
    original, read, _ = stated()
    subject = COPIES[copied](original)

    def reports(gm):
        saved = next(d for d in gm.to_dict()["nodes"] if d["name"] == "n")["params"]
        return _leaf(gm), _leaf(gm, "extra"), saved["rate"], saved["extra"]

    assert reports(subject) == (rate, extra, rate, extra), (
        f"{kind}-{n}, {order}, {copied}: the copy holds {reports(subject)}")
    assert reports(other) == (rate, extra, rate, extra)
    want = _run(other, other_read, STEPS_AFTER)
    got = _run(subject, read, STEPS_AFTER)
    assert np.array_equal(got, want), f"{kind}-{n}, {order}, {copied}: the copy ran another value"
    assert reports(subject) == (rate, extra, rate, extra)
    # ... and not the constructor's value: the fixture can express a lost write.
    assert rate != OLD


@pytest.mark.parametrize("copied", list(COPIES))
@pytest.mark.parametrize("order", list(ORDERS))
def test_a_copy_runs_and_reports_what_the_original_does(order, copied):
    """Every order with every copy, on a node on its own."""
    _check("unsharded", 1, order, copied)


@pytest.mark.parametrize("order", ["node_write_taken_in_then_calibration",
                                   "calibration_then_node_write_pending"])
@pytest.mark.parametrize("kind, n", WRAPPERS[1:], ids=[f"{k}-{n}" for k, n in WRAPPERS[1:]])
def test_a_deep_copy_of_a_graph_of_a_wrapped_node_runs_what_the_original_does(kind, n, order):
    """The order that was lost and one with a write pending, behind every
    wrapper."""
    _check(kind, n, order, "deep_copy_of_the_graph")


# Per push: tests/cloud/multigpu/test_a_copy_of_a_graph_runs_and_reports_what_the_original_does.py::test_a_deep_copy_of_a_graph_of_a_wrapped_node_runs_what_the_original_does
@pytest.mark.slow
@pytest.mark.parametrize("kind, n", WRAPPERS[1:], ids=[f"{k}-{n}" for k, n in WRAPPERS[1:]])
def test_every_copy_in_every_order_behind_every_wrapper(kind, n):
    for order in ORDERS:
        for copied in COPIES:
            _check(kind, n, order, copied)


@pytest.mark.parametrize("duplicate", [copy.deepcopy, copy.copy,
                                       lambda node: pickle.loads(pickle.dumps(node))],
                         ids=["deepcopy", "copy", "pickle"])
@pytest.mark.parametrize("contract", list(CONTRACTS))
def test_a_graph_built_around_a_copy_of_a_written_node_runs_the_written_value(contract, duplicate):
    """A node written and assigned before any graph holds it, then copied:
    the graph built around the copy is the graph built around the node."""
    def written():
        node = CONTRACTS[contract]()
        node.params["rate"] = MID
        node.params = {**node.params, "extra": [CALIBRATED_EXTRA]}
        return node

    def run(node):
        gm = GraphManager()
        gm.add_node(node)
        gm.compile()
        saved = gm.to_dict()["nodes"][0]["params"]
        return _run(gm, _global_x, STEPS_AFTER), saved["rate"], saved["extra"]

    want = run(written())
    got = run(duplicate(written()))
    assert np.array_equal(got[0], want[0]) and got[1:] == want[1:] == (MID, [CALIBRATED_EXTRA])


def test_a_graph_not_yet_compiled_is_pickled_with_its_writes_and_a_compiled_one_refuses():
    """``pickle`` of a graph before ``compile()`` carries the node writes
    made so far; a compiled graph holds its compiled step and cannot be
    pickled, which it says (a copy that could be wrong in silence is the
    thing to rule out)."""
    def built():
        gm = GraphManager()
        gm.add_node(CONTRACTS["params"]())
        gm.get_node("n").params["rate"] = MID
        return gm

    twin = pickle.loads(pickle.dumps(built()))
    original = built()
    for gm in (twin, original):
        gm.compile()
        _calibrate(gm, "extra", [CALIBRATED_EXTRA])
    assert _leaf(twin) == _leaf(original) == MID
    assert np.array_equal(_run(twin, _global_x, STEPS_AFTER), _run(original, _global_x, STEPS_AFTER))
    with pytest.raises((AttributeError, TypeError, pickle.PicklingError)):
        pickle.dumps(original)


def test_a_three_argument_node_write_pending_at_the_copy_is_run_by_the_copy():
    """Nothing of such a node is a leaf of ``gm.params``; a pending write
    marks the graph for a recompile, and the copy recompiles as the
    original does."""
    def stated():
        gm, inner, read = _graph("unsharded", 1, "three_argument")
        gm.step()
        gm.get_node("n").params["rate"] = MID
        return gm, read

    other, other_read = stated()
    original, read = stated()
    twin, twin_read = stated()
    assert np.array_equal(_run(copy.deepcopy(original), read, STEPS_AFTER),
                          _run(twin, twin_read, STEPS_AFTER))
    want = np.asarray(other_read(other.get_node_state("n"))) * np.float32(1.0 - 0.1 * MID) ** STEPS_AFTER
    np.testing.assert_allclose(_run(other, other_read, STEPS_AFTER), want, rtol=1e-6)


@pytest.mark.parametrize("contract", list(CONTRACTS))
@pytest.mark.parametrize("kind, n", WRAPPERS, ids=[f"{k}-{n}" for k, n in WRAPPERS])
def test_a_shallow_copy_of_a_graph_is_refused_and_names_the_deep_copy(kind, n, contract):
    """``copy.copy(gm)`` shared the nodes, the parameter tree and the
    record of the writes seen, and not the compiled step: after a write to
    a node whose constants are baked into the step (``three_argument``),
    the first graph to sync took the write in for both and only it
    recompiled, so the other ran the old value (x[0] 0.8145 against 0.4864
    after three steps in the audit's case) beside ``node.params`` and
    ``to_dict()`` saying the new one.  A record of its own per graph is not
    exact either: the second sync then writes the node's value into the
    shared tree again, over a calibration made through the first graph
    since.  So it is refused, behind every wrapper and on both contracts,
    whether the graph is compiled or not, and the graph it was asked of is
    untouched."""
    gm, _inner, read = _graph(kind, n, contract)
    for state in ("compiled", "stepped"):
        with pytest.raises(TypeError, match=r"copy\.copy\(\) of a GraphManager is refused.*"
                                            r"copy\.deepcopy\(gm\)"):
            copy.copy(gm)
        if state == "compiled":
            gm.step()
    uncompiled = GraphManager()
    uncompiled.add_node(CONTRACTS[contract]())
    with pytest.raises(TypeError, match="copy.deepcopy"):
        copy.copy(uncompiled)
    # The refusal leaves the graph as it was, and a deep copy is the way.
    gm.get_node("n").params["rate"] = MID
    twin = copy.deepcopy(gm)
    assert np.array_equal(_run(twin, read, STEPS_AFTER), _run(gm, read, STEPS_AFTER))
