"""Outside the compiled step, an edge is applied by the step's rule or not at all.

The step turns an edge into a boundary input with ``_apply_edge``: the
interface mapping, then the transform; additive edges sum; an external
input replaces what edges delivered to its field.  Three readers of the
edges rebuilt boundary inputs without it (MADD-ANO-193):

* ``check_conservation`` read the source field, applied the transform
  only, let a later edge overwrite an additive one, dropped an edge that
  reads a flux output, and read a flux name it could not find as ``0.0``;
* ``DatasetGenerator`` did the same and applied a transform to the whole
  history at once, so a transform that indexes its field indexed time;
* ``resolve_boundary_inputs`` kept an edge's value in a field an external
  input replaces.

All three now go through ``GraphManager._boundary_inputs_from``.  The
oracle throughout is a node that stores the boundary input it was fed:
after a step its state *is* what the step delivered.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.helpers import add_flux_coupling, check_conservation
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.surrogates.dataset import DatasetGenerator

F32 = jnp.float32
DISCONNECTED = "is disconnected"

#: Not the identity, and not square: source field of 3, boundary input of 2.
H = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]], np.float32)


class _Source(SimulationNode):
    """``x <- x + rate``: constant with ``rate=0``, a ramp otherwise."""

    def __init__(self, name, x0, rate=0.0):
        super().__init__(name, 0.1)
        self._x0, self._rate = np.asarray(x0, np.float32), rate

    def initial_state(self):
        return {"x": jnp.asarray(self._x0)}

    def boundary_input_spec(self):
        return {"back": BoundaryInputSpec(shape=(2,), dtype=F32, default=jnp.zeros(2, F32))}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] + jnp.float32(self._rate)}


class _Recorder(SimulationNode):
    """Its state is the boundary input it was fed, and so is its flux; the
    inputs its flux hook was called with are kept on the instance."""

    def __init__(self, name, n=2, gain=1.0):
        super().__init__(name, 0.1, gain=gain)
        self._n = n
        self.fed: list = []

    def initial_state(self):
        return {"seen": jnp.zeros(self._n, F32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._n,), dtype=F32,
                                       default=jnp.zeros(self._n, F32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"seen": boundary_inputs["u"] + 0.0 * state["seen"]}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        self.fed.append(dict(boundary_inputs))
        gain = self.params["gain"] if params is None else params["gain"]
        return {"q": gain * boundary_inputs.get("u", jnp.zeros(self._n, F32))}


def _double_reversed(v):
    """A transform that indexes its field: on a history it would index time."""
    return 2.0 * v[::-1]


#: variant -> the edges into ``tgt.u`` (source, kwargs)
EDGES = {
    "plain": [("two", {})],
    "mapping": [("three", {"mapping": lambda: matrix_mapping(H)})],
    "mapping then transform": [
        ("three", {"mapping": lambda: matrix_mapping(H), "transform": _double_reversed})],
    "mapping plus an additive edge": [
        ("three", {"mapping": lambda: matrix_mapping(H)}),
        ("two", {"additive": True})],
    "two additive mapped edges": [
        ("three", {"mapping": lambda: matrix_mapping(H)}),
        ("three", {"mapping": lambda: matrix_mapping(2.0 * H), "additive": True,
                   "transform": _double_reversed})],
    "a transform that indexes its field": [("two", {"transform": _double_reversed})],
    "additive edges": [("two", {}), ("two", {"additive": True, "transform": _double_reversed})],
    "an external input on the same field": [("two", {})],
    "a later edge that is not additive": [
        ("three", {"mapping": lambda: matrix_mapping(H)}), ("two", {"transform": _double_reversed})],
}
#: The variants the diagnostic and the dataset got wrong before the fix.
MAPPED = [v for v in EDGES if "mapp" in v]


def _graph(variant, rate=0.0, compile_it=True):
    gm = GraphManager()
    nodes = {"three": _Source("three", [1.0, 2.0, 3.0], rate),
             "two": _Source("two", [10.0, 20.0], rate), "tgt": _Recorder("tgt")}
    for node in nodes.values():
        gm.add_node(node)
    for source, kwargs in EDGES[variant]:
        kwargs = {k: (v() if k == "mapping" else v) for k, v in kwargs.items()}
        gm.add_edge(source, "tgt", "x", "u", **kwargs)
    if variant == "an external input on the same field":
        gm.add_external_input("tgt", "u", shape=(2,))
    if compile_it:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=f".*{DISCONNECTED}", category=UserWarning)
            gm.compile()
    return gm, nodes


def _expected(variant, three, two):
    """What ``tgt.u`` is by the step's rule, written out independently."""
    three, two = np.asarray(three, np.float64), np.asarray(two, np.float64)
    h = H.astype(np.float64)
    return {
        "plain": two,
        "mapping": h @ three,
        "mapping then transform": 2.0 * (h @ three)[::-1],
        "mapping plus an additive edge": h @ three + two,
        "two additive mapped edges": h @ three + 2.0 * (2.0 * h @ three)[::-1],
        "a transform that indexes its field": 2.0 * two[::-1],
        "additive edges": two + 2.0 * two[::-1],
        "an external input on the same field": np.zeros(2),
        "a later edge that is not additive": 2.0 * two[::-1],      # it replaces the first
    }[variant]


# ---------------------------------------------------------------------------
# The oracle itself, then the inspection helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", sorted(EDGES))
def test_the_step_feeds_the_node_the_written_out_rule(variant):
    """The recorder is a faithful oracle: its state after a step is the rule."""
    gm, _ = _graph(variant)
    gm.step()
    np.testing.assert_array_equal(np.asarray(gm._state["tgt"]["seen"]),
                                  _expected(variant, [1.0, 2.0, 3.0], [10.0, 20.0]))


@pytest.mark.parametrize("variant", sorted(EDGES))
def test_resolve_boundary_inputs_is_what_the_step_then_feeds(variant):
    gm, _ = _graph(variant)
    resolved = np.asarray(gm.resolve_boundary_inputs("tgt")["u"])
    gm.step()
    np.testing.assert_array_equal(resolved, np.asarray(gm._state["tgt"]["seen"]))


def test_resolve_boundary_inputs_still_raises_key_error_on_a_flux_edge():
    """Unchanged: a flux output is not in the state the helper reads."""
    gm = GraphManager()
    gm.add_node(_Recorder("a"))
    gm.add_node(_Recorder("b"))
    add_flux_coupling(gm, "a", "b", "q", "u")
    gm.compile()
    with pytest.raises(KeyError):
        gm.resolve_boundary_inputs("b")
    with pytest.raises(KeyError, match="unknown node"):
        gm.resolve_boundary_inputs("nobody")


# ---------------------------------------------------------------------------
# The conservation diagnostic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", sorted(EDGES))
def test_check_conservation_feeds_the_flux_hook_what_the_step_feeds(variant):
    gm, nodes = _graph(variant)
    state = gm._state
    nodes["tgt"].fed.clear()
    result = check_conservation(gm, state, [("tgt", "q", "tgt", "q")])
    assert result == {"tgt.q-tgt.q": 0.0}
    fed = np.asarray(nodes["tgt"].fed[0]["u"])
    gm.step()
    np.testing.assert_array_equal(fed, np.asarray(gm._state["tgt"]["seen"]))


def _mapped_pair():
    """``left`` is fed ``H @ x`` through a mapped edge, ``right`` the same
    numbers directly: the two fluxes are equal, so the imbalance is zero."""
    gm = GraphManager()
    gm.add_node(_Source("three", [1.0, 2.0, 3.0]))
    gm.add_node(_Source("mapped", H @ np.array([1.0, 2.0, 3.0], np.float32)))
    gm.add_node(_Recorder("left"))
    gm.add_node(_Recorder("right"))
    gm.add_edge("three", "left", "x", "u", mapping=matrix_mapping(H))
    gm.add_edge("mapped", "right", "x", "u")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*{DISCONNECTED}", category=UserWarning)
        gm.compile()
    return gm


def test_a_conservative_mapped_interface_reads_as_conservative():
    """The unmapped field has three entries against two: the comparison
    used to be of different quantities (here, a broadcast error)."""
    gm = _mapped_pair()
    assert check_conservation(gm, gm._state, [("left", "q", "right", "q")]) == \
        {"left.q-right.q": 0.0}


def test_check_conservation_reads_the_live_mapping_weights_and_node_constants():
    """``gm.params`` is what the step runs on; the diagnostic follows it."""
    gm = _mapped_pair()
    key = "three.x->left.u"
    gm.params["mappings"][key]["H"] = 2.0 * gm.params["mappings"][key]["H"]
    doubled = check_conservation(gm, gm._state, [("left", "q", "right", "q")])
    assert doubled["left.q-right.q"] == pytest.approx(5.0 + 9.0)      # sum(2 H x - H x)
    gm.params["nodes"]["right"]["gain"] = jnp.asarray(2.0, F32)
    assert check_conservation(gm, gm._state, [("left", "q", "right", "q")]) == \
        {"left.q-right.q": 0.0}


def test_check_conservation_works_on_a_graph_that_was_never_compiled():
    gm, _ = _graph("mapping plus an additive edge", compile_it=False)
    state = {"three": {"x": jnp.asarray([1.0, 2.0, 3.0], F32)},
             "two": {"x": jnp.asarray([10.0, 20.0], F32)},
             "tgt": {"seen": jnp.zeros(2, F32)}}
    assert check_conservation(gm, state, [("tgt", "q", "tgt", "q")]) == {"tgt.q-tgt.q": 0.0}


class _FluxOfState(SimulationNode):
    """Reports ``k * x`` as its flux ``qa`` and takes one input ``u``."""

    def __init__(self, name, x0):
        super().__init__(name, 0.1)
        self._x0 = np.asarray(x0, np.float32)

    def initial_state(self):
        return {"x": jnp.asarray(self._x0)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(2,), dtype=F32, default=jnp.zeros(2, F32))}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"]}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        return {"qa": 2.0 * state["x"] + boundary_inputs.get("u", jnp.zeros(2, F32))}


def test_an_edge_that_reads_a_flux_output_is_resolved_not_dropped():
    """Dirichlet-Neumann shape: ``b`` is fed ``a``'s flux.  The diagnostic
    used to skip the edge and compare against a flux computed without it."""
    gm = GraphManager()
    gm.add_node(_FluxOfState("a", [1.0, 4.0]))
    b = _Recorder("b")
    gm.add_node(b)
    add_flux_coupling(gm, "a", "b", "qa", "u", transform=lambda q: 0.5 * q)
    gm.compile()
    result = check_conservation(gm, gm._state, [("a", "qa", "b", "q")])
    np.testing.assert_array_equal(np.asarray(b.fed[-1]["u"]), [1.0, 4.0])     # 0.5 * (2 x)
    assert result == {"a.qa-b.q": pytest.approx((2.0 + 8.0) - (1.0 + 4.0))}
    # The step feeds the same thing (a forward flux edge).
    gm.step()
    np.testing.assert_array_equal(np.asarray(gm._state["b"]["seen"]), [1.0, 4.0])


def test_two_nodes_that_read_each_others_flux_are_refused_not_guessed():
    """Robin-Robin: each input is what the coupling iteration converged
    to.  The diagnostic used to drop both flux edges without a word."""
    gm = GraphManager()
    gm.add_node(_FluxOfState("a", [1.0, 4.0]))
    gm.add_node(_FluxOfState("b", [2.0, 3.0]))
    add_flux_coupling(gm, "a", "b", "qa", "u")
    add_flux_coupling(gm, "b", "a", "qa", "u")
    state = {"a": {"x": jnp.asarray([1.0, 4.0], F32)}, "b": {"x": jnp.asarray([2.0, 3.0], F32)}}
    with pytest.raises(ValueError, match="depend on its own") as caught:
        check_conservation(gm, state, [("a", "qa", "b", "qa")])
    assert "a -> b -> a" in str(caught.value)


def test_a_flux_the_node_does_not_report_is_a_key_error_not_zero():
    """A misspelt flux read as ``0.0``: two misspelt names were "conserved"."""
    gm, _ = _graph("plain")
    with pytest.raises(KeyError, match="reports no flux 'heat'.*\\['q'\\]"):
        check_conservation(gm, gm._state, [("tgt", "heat", "tgt", "heat")])


def test_a_node_missing_from_the_state_or_the_graph_is_a_key_error():
    gm, _ = _graph("plain")
    state = dict(gm._state)
    del state["two"]
    with pytest.raises(KeyError, match="no entry for node 'two', the source of edge"):
        check_conservation(gm, state, [("tgt", "q", "tgt", "q")])
    with pytest.raises(KeyError, match="no node 'nobody'"):
        check_conservation(gm, gm._state, [("nobody", "q", "tgt", "q")])
    with pytest.raises(KeyError, match="neither a state field"):
        gm2, _ = _graph("plain", compile_it=False)
        gm2.add_edge("two", "tgt", "no_such_field", "u", additive=True)
        check_conservation(gm2, {"two": {"x": jnp.zeros(2)}, "three": {"x": jnp.zeros(3)},
                                 "tgt": {"seen": jnp.zeros(2)}}, [("tgt", "q", "tgt", "q")])


# ---------------------------------------------------------------------------
# The surrogate dataset generator
# ---------------------------------------------------------------------------

N_STEPS = 4


def _sweep_initial_states():
    return {"three": {"x": jnp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], F32)},
            "two": {"x": jnp.asarray([[10.0, 20.0], [30.0, 40.0]], F32)},
            "tgt": {"seen": jnp.zeros((2, 2), F32)}}


@pytest.mark.parametrize("variant", sorted(EDGES))
def test_a_dataset_pairs_each_state_with_the_boundary_input_the_step_fed(variant):
    """Constant sources, so the time level cannot matter: ``next_states``
    of the recorder is what the step fed for that sample."""
    gm, _ = _graph(variant)
    ds = DatasetGenerator.from_graph(gm, "tgt", N_STEPS)
    assert ds.boundary_spec == {"u": (2,)}
    assert ds.boundary_inputs["u"].shape == (N_STEPS - 1, 2)
    np.testing.assert_array_equal(np.asarray(ds.boundary_inputs["u"]),
                                  np.asarray(ds.next_states["seen"]))
    np.testing.assert_array_equal(
        np.asarray(ds.boundary_inputs["u"][0]), _expected(variant, [1.0, 2.0, 3.0], [10.0, 20.0]))


@pytest.mark.parametrize("variant", sorted(EDGES))
def test_a_sweep_dataset_pairs_each_state_with_the_boundary_input_the_step_fed(variant):
    gm, _ = _graph(variant)
    ds = DatasetGenerator.from_sweep(gm, "tgt", N_STEPS, _sweep_initial_states())
    assert ds.boundary_spec == {"u": (2,)}
    assert ds.boundary_inputs["u"].shape == (2 * (N_STEPS - 1), 2)
    np.testing.assert_array_equal(np.asarray(ds.boundary_inputs["u"]),
                                  np.asarray(ds.next_states["seen"]))
    np.testing.assert_array_equal(           # the second condition's first sample
        np.asarray(ds.boundary_inputs["u"][N_STEPS - 1]),
        _expected(variant, [4.0, 5.0, 6.0], [30.0, 40.0]))


def test_a_dataset_reads_the_live_mapping_weights():
    gm, _ = _graph("mapping")
    key = "three.x->tgt.u"
    gm.params["mappings"][key]["H"] = 3.0 * gm.params["mappings"][key]["H"]
    ds = DatasetGenerator.from_graph(gm, "tgt", N_STEPS)
    np.testing.assert_array_equal(np.asarray(ds.boundary_inputs["u"][0]), [15.0, 27.0])
    np.testing.assert_array_equal(np.asarray(ds.boundary_inputs["u"]),
                                  np.asarray(ds.next_states["seen"]))


def test_a_dataset_of_a_node_with_only_external_inputs_holds_zeros():
    gm = GraphManager()
    gm.add_node(_Recorder("tgt"))
    gm.add_external_input("tgt", "u", shape=(2,))
    gm.compile()
    ds = DatasetGenerator.from_graph(gm, "tgt", N_STEPS)
    assert ds.boundary_spec == {"u": (2,)}
    np.testing.assert_array_equal(np.asarray(ds.boundary_inputs["u"]), np.zeros((N_STEPS - 1, 2)))
    ds = DatasetGenerator.from_sweep(gm, "tgt", N_STEPS, {"tgt": {"seen": jnp.zeros((3, 2), F32)}})
    np.testing.assert_array_equal(np.asarray(ds.boundary_inputs["u"]),
                                  np.zeros((3 * (N_STEPS - 1), 2)))


def test_a_dataset_of_a_node_fed_by_a_flux_edge_is_a_key_error():
    """Unchanged: a state history holds no flux outputs to rebuild it from."""
    gm = GraphManager()
    gm.add_node(_FluxOfState("a", [1.0, 4.0]))
    gm.add_node(_Recorder("b"))
    add_flux_coupling(gm, "a", "b", "qa", "u")
    gm.compile()
    with pytest.raises(KeyError):
        DatasetGenerator.from_graph(gm, "b", N_STEPS)


def _cycle(order):
    """``two`` ramps and feeds ``tgt``; ``tgt`` feeds ``two`` back (unused by
    it), which closes a cycle so that one of the two edges is a back edge."""
    gm = GraphManager()
    nodes = {"two": _Source("two", [10.0, 20.0], rate=1.0), "tgt": _Recorder("tgt")}
    for name in order:
        gm.add_node(nodes[name])
    gm.add_edge("two", "tgt", "x", "u", mapping=matrix_mapping(np.float32([[0, 1], [1, 0]])))
    gm.add_edge("tgt", "two", "seen", "back")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gm.compile()
    return gm


def test_a_back_edges_input_is_paired_with_the_step_that_read_it():
    """``tgt`` runs first, so it reads ``two`` from the previous step: the
    state the history holds beside ``tgt``'s own."""
    gm = _cycle(["tgt", "two"])
    assert [e.key for e in gm._back_edges] == ["two.x->tgt.u"]
    ds = DatasetGenerator.from_graph(gm, "tgt", N_STEPS)
    np.testing.assert_array_equal(np.asarray(ds.boundary_inputs["u"]),
                                  np.asarray(ds.next_states["seen"]))
    np.testing.assert_array_equal(np.asarray(ds.boundary_inputs["u"][0]), [21.0, 11.0])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "MADD-ANO-194: DatasetGenerator pairs a forward edge's boundary input with the "
    "source's state before the step, where the step read its state after it; "
    "deferred to 0.5.0"))
def test_a_forward_edges_input_is_paired_with_the_step_that_read_it():
    """``two`` runs first, so ``tgt`` reads the value ``two`` has *after*
    this step; the dataset holds the one before it."""
    gm = _cycle(["two", "tgt"])
    assert [e.key for e in gm._back_edges] == ["tgt.seen->two.back"]
    ds = DatasetGenerator.from_graph(gm, "tgt", N_STEPS)
    np.testing.assert_array_equal(np.asarray(ds.boundary_inputs["u"]),
                                  np.asarray(ds.next_states["seen"]))
