"""``coupling_diagnostics()`` describes the step that ran, whatever is written afterwards.

The residual's float floor -- which ``spectral_error_bound`` adds,
``precision_limited`` reads and ``spectral_usable`` rests on -- is measured
on the state the step returned.  The report took that state from the live
graph at report time, so a ``set_node_state`` after the step gave the same
step another report: a float32 pair stalled at ``residual == 0.0`` read
``spectral_error_bound == 0.0`` with ``spectral_usable=True`` once both
members were written to zero (a field at exactly zero leaves the norm),
for a state 1e-5 from its fixed point.

The graph now keeps what the step left from the first later write
(``GraphManager._keep_state_for_reports``).  The invariant, for every
stepping entry point and every door that writes the state without
stepping: **the report read after the write equals the report read before
it**, key for key; and a door that brings its own report (a step, a loaded
checkpoint) is described by the state it brought.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.simulation.profiler import _one_iteration_variant, compile_counts

N = 3
KEY = "a+b"
GAIN = 0.99
ZERO = {"x": jnp.zeros(N, jnp.float32)}
ADAPTIVE = dict(dt_initial=0.01, dt_min=0.01, dt_max=0.01, atol=1e3, rtol=1.0)


class Lin(SimulationNode):
    """``x <- g * u + b``, one evaluation per pass, declared."""

    def __init__(self, name, b):
        super().__init__(name, 0.01, g=jnp.float32(GAIN), b=jnp.full(N, b, jnp.float32))

    def initial_state(self):
        return {"x": jnp.zeros(N, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(N,), dtype=jnp.float32,
                                       default=jnp.zeros(N, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": p["g"] * boundary_inputs["u"] + p["b"]}

    def update_evaluations(self):
        return 1


class Idle(SimulationNode):
    """A node outside the group."""

    def __init__(self, name):
        super().__init__(name, 0.01)

    def initial_state(self):
        return {"y": jnp.ones(N, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return state


def _pair(diagnostics=True):
    """A float32 Gauss-Seidel pair that stalls at ``residual == 0.0`` under
    the default tolerance, 1e-5 from its float64 fixed point: the floor is
    the whole of its bound."""
    gm = GraphManager()
    gm.add_node(Lin("a", 1.0))
    gm.add_node(Lin("b", 2.0))
    gm.add_node(Idle("c"))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=400000, tolerance=1e-9,
                          diagnostics=diagnostics)
    gm.compile()
    return gm


def _report(gm):
    d = gm.coupling_diagnostics().get(KEY)
    if d is None:
        return None
    return {k: ("nan" if isinstance(v, float) and v != v else v) for k, v in d.items()}


def _true_distance(gm):
    """The returned state's distance to the float64 fixed point, in the
    group's norm (each field over its largest entry, root-sum-square)."""
    g = float(np.float32(GAIN))
    a_star = (1.0 + g * 2.0) / (1.0 - g * g)
    b_star = 2.0 + g * a_star
    xa = np.asarray(gm.get_node_state("a")["x"], np.float64)
    xb = np.asarray(gm.get_node_state("b")["x"], np.float64)
    return float(np.sqrt(np.sum(((xa - a_star) / np.max(np.abs(xa))) ** 2)
                         + np.sum(((xb - b_star) / np.max(np.abs(xb))) ** 2)))


STEPPERS = {
    "step": lambda gm: gm.step(),
    "run": lambda gm: gm.run(2),
    "run_scan": lambda gm: gm.run_scan(2),
    "run_scan_with_history": lambda gm: gm.run_scan_with_history(2),
    "run_adaptive": lambda gm: gm.run_adaptive(0.01, **ADAPTIVE),
    "run_adaptive_scan": lambda gm: gm.run_adaptive_scan(0.01, max_steps=1, **ADAPTIVE),
}


def _scaled(gm, name, factor):
    return {f: v * jnp.asarray(factor, v.dtype) for f, v in gm.get_node_state(name).items()}


#: Writes that are not a step, each as ``write(gm)``.
WRITES = {
    "one member to zero": lambda gm: gm.set_node_state("b", ZERO),
    "both members to zero": lambda gm: (gm.set_node_state("a", ZERO),
                                        gm.set_node_state("b", ZERO)),
    "both members a million times larger": lambda gm: (
        gm.set_node_state("a", _scaled(gm, "a", 1e6)),
        gm.set_node_state("b", _scaled(gm, "b", 1e6))),
    "a member to NaN": lambda gm: gm.set_node_state(
        "a", {"x": jnp.full(N, jnp.nan, jnp.float32)}),
    "a node outside the group": lambda gm: gm.set_node_state(
        "c", {"y": jnp.zeros(N, jnp.float32)}),
    "the same member twice": lambda gm: (gm.set_node_state("a", _scaled(gm, "a", 2.0)),
                                         gm.set_node_state("a", ZERO)),
}


@pytest.fixture(scope="module")
def stepped():
    """One stepped pair and its report, for the tests that only write:
    ``(gm, report, returned state)``.  No test steps this graph."""
    gm = _pair()
    gm.step()
    return gm, _report(gm), {n: gm.get_node_state(n) for n in ("a", "b", "c")}


@pytest.fixture(scope="module")
def _second():
    return _pair()


@pytest.fixture
def graph(_second):
    """A compiled pair for the tests that step, reset before each."""
    _second.reset_state()
    return _second


def _restore(gm, returned):
    for name, fields in returned.items():
        gm.set_node_state(name, fields)


def test_the_fixture_rests_its_bound_on_the_floor(stepped):
    """The case can express the defect: the floor is the whole bound, and
    the bound covers the true distance."""
    gm, first, returned = stepped
    _restore(gm, returned)
    assert first["residual"] == 0.0 and first["precision_limited"], first
    assert first["spectral_usable"] and first["spectral_error_bound"] > 0.0, first
    true = _true_distance(gm)
    assert 0.0 < true <= first["spectral_error_bound"], (true, first)


@pytest.mark.parametrize("write", sorted(WRITES))
def test_a_write_after_the_step_does_not_move_its_report(stepped, write):
    gm, first, returned = stepped
    _restore(gm, returned)
    WRITES[write](gm)
    assert _report(gm) == first
    # ...and the state really was written: the report is not a refusal to write.
    _restore(gm, returned)
    assert _report(gm) == first


@pytest.mark.parametrize("stepper", sorted(STEPPERS))
def test_the_report_of_every_stepping_entry_point_survives_a_write(graph, stepper):
    STEPPERS[stepper](graph)
    first = _report(graph)
    assert first is not None and first["spectral_usable"], first
    WRITES["both members to zero"](graph)
    assert _report(graph) == first


def test_the_next_step_is_reported_from_the_state_it_left(graph):
    """A step after the write brings its own report, and nothing is held
    back from the step before it."""
    graph.step()
    first = _report(graph)
    graph.set_node_state("a", _scaled(graph, "a", 0.5))
    assert graph._state_as_reported is not None
    graph.step()
    assert graph._state_as_reported is None
    second = _report(graph)
    assert second["iterations"] != first["iterations"], (first, second)
    true = _true_distance(graph)
    assert second["spectral_usable"] and second["spectral_error_bound"] >= true > 0.0
    # Measured on the state this step left: writing the same values back
    # (which keeps that state) changes nothing.
    for name in ("a", "b"):
        graph.set_node_state(name, graph.get_node_state(name))
    assert _report(graph) == second


def test_a_loaded_checkpoint_is_reported_as_the_step_it_saved(stepped, graph, tmp_path):
    """The decision for ``load_state``: the slots a checkpoint carries
    describe the step that left the state beside them, so the report read
    right after a load is the saved step's -- in a graph that has not
    stepped and over one that has stepped and been written elsewhere -- and
    it survives a later write like any other."""
    gm, first, returned = stepped
    _restore(gm, returned)
    path = gm.save_state(tmp_path / "stepped.npz")

    graph.load_state(path)
    assert _report(graph) == first
    for slot, value in gm._state["_meta"].items():
        got = graph._state["_meta"][slot]
        assert np.asarray(got).tobytes() == np.asarray(value).tobytes(), slot
        assert np.asarray(got).dtype == np.asarray(value).dtype, slot
    WRITES["both members to zero"](graph)
    assert _report(graph) == first

    # Over a graph somewhere else, with a write of its own pending.
    graph.reset_state()
    graph.step()
    graph.step()
    elsewhere = _report(graph)
    assert elsewhere != first
    WRITES["both members to zero"](graph)
    assert _report(graph) == elsewhere
    graph.load_state(path)
    assert _report(graph) == first
    WRITES["one member to zero"](graph)
    assert _report(graph) == first


def test_a_checkpoint_of_a_written_state_carries_no_report(graph, tmp_path):
    """Saved after a write, the archive holds the written state and not
    what the step left: loaded, the group has no report until it steps
    (its counter is saved at 0, "no step taken yet"), and the restart
    steps bit for bit like the graph that saved it."""
    graph.step()
    first = _report(graph)
    WRITES["both members to zero"](graph)
    path = graph.save_state(tmp_path / "written.npz")
    assert _report(graph) == first            # saving moved nothing
    saved_meta = {k: np.asarray(v) for k, v in graph._state["_meta"].items()}
    graph.step()
    want = ({n: np.asarray(graph.get_node_state(n)["x"]) for n in ("a", "b")}, _report(graph))

    graph.reset_state()
    graph.load_state(path)
    assert _report(graph) is None
    for slot, value in saved_meta.items():
        got = np.asarray(graph._state["_meta"][slot])
        if slot == f"coupling_{KEY}_iterations":
            assert int(got) == 0 and got.dtype == value.dtype
        else:
            assert got.tobytes() == value.tobytes(), slot
    np.testing.assert_array_equal(np.asarray(graph.get_node_state("a")["x"]), 0.0)
    graph.step()
    for name in ("a", "b"):
        assert np.asarray(graph.get_node_state(name)["x"]).tobytes() == want[0][name].tobytes()
    assert _report(graph) == want[1]


@pytest.mark.parametrize("write", ["a node outside the group", "the values it already holds"])
def test_a_checkpoint_keeps_the_report_where_no_member_changed(graph, tmp_path, write):
    graph.step()
    first = _report(graph)
    if write == "a node outside the group":
        WRITES[write](graph)
    else:
        for name in ("a", "b"):
            graph.set_node_state(name, {"x": jnp.asarray(np.asarray(
                graph.get_node_state(name)["x"]))})
    path = graph.save_state(tmp_path / "kept.npz")
    graph.reset_state()
    graph.load_state(path)
    assert _report(graph) == first


def test_a_failed_load_leaves_the_report_of_the_step_before_it(stepped, tmp_path):
    gm, first, returned = stepped
    _restore(gm, returned)
    WRITES["both members to zero"](gm)
    bad = tmp_path / "not_a_checkpoint.npz"
    bad.write_bytes(b"not an archive")
    with pytest.raises(Exception):
        gm.load_state(bad)
    assert _report(gm) == first
    _restore(gm, returned)


def test_a_recompile_keeps_the_report_and_what_it_was_measured_on(graph):
    graph.step()
    first = _report(graph)
    WRITES["both members to zero"](graph)
    graph.compile()
    assert _report(graph) == first
    WRITES["both members a million times larger"](graph)
    assert _report(graph) == first


def test_reset_state_and_a_removed_member_still_end_the_report():
    gm = _pair(diagnostics=False)
    gm.step()
    assert _report(gm) is not None
    WRITES["both members to zero"](gm)
    gm.reset_state()
    assert _report(gm) is None
    gm.step()
    first = _report(gm)
    WRITES["one member to zero"](gm)
    assert _report(gm) == first
    with pytest.warns(UserWarning, match="removed the coupling group"):
        gm.remove_node("b")
    assert _report(gm) is None


def test_a_transform_that_stepped_the_graph_leaves_the_report_of_the_step_before_it(graph):
    """A transform of a function that calls ``run_scan`` stores tracers and
    the graph is put back to the state before it: the written one, whose
    report is still the earlier step's."""
    graph.step()
    first = _report(graph)
    WRITES["both members to zero"](graph)

    def loss(p):
        graph.run_scan(1, params=p)
        return jnp.sum(graph.get_node_state("a")["x"])

    jax.make_jaxpr(loss)(graph.params)
    assert graph._state_traced
    with pytest.warns(RuntimeWarning, match="put back"):
        assert _report(graph) == first
    np.testing.assert_array_equal(np.asarray(graph.get_node_state("a")["x"]), 0.0)


def test_a_traced_write_keeps_no_tracer_for_the_report(graph):
    """A differentiable initial condition written inside a transform: the
    report read afterwards is the earlier step's and reads no tracer."""
    graph.step()
    first = _report(graph)

    def loss(scale):
        graph.set_node_state("a", {"x": scale * jnp.ones(N, jnp.float32)})
        graph.set_node_state("b", {"x": scale * jnp.ones(N, jnp.float32)})
        graph.run_scan(1)
        # A write after a traced step: nothing of the trace may be kept.
        graph.set_node_state("b", {"x": scale * jnp.ones(N, jnp.float32)})
        return jnp.sum(graph.get_node_state("a")["x"])

    jax.make_jaxpr(loss)(jnp.float32(0.0))
    assert graph._state_traced
    with pytest.warns(RuntimeWarning, match="put back"):
        assert _report(graph) == first
    WRITES["both members to zero"](graph)
    assert _report(graph) == first


def test_the_profilers_measurements_hand_the_report_back_with_the_state(graph):
    """``compile_counts`` and the one-iteration variant step the graph and
    put the state back: the report comes back with it."""
    graph.step()
    first = _report(graph)
    WRITES["both members to zero"](graph)
    compile_counts(graph, warmup_steps=0, scan_steps=1)
    assert _report(graph) == first
    with _one_iteration_variant(graph):
        graph.step()
    assert _report(graph) == first


def test_the_kept_state_holds_references_not_copies(graph):
    """Keeping what the step left costs no array: the kept fields are the
    step's own objects."""
    graph.step()
    before = {n: graph._state[n] for n in ("a", "b", "c")}
    graph.set_node_state("a", ZERO)
    kept_state, _kept_meta = graph._state_as_reported
    for name, fields in before.items():
        assert kept_state[name] is fields
