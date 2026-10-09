"""The shape ``compile()`` holds an edge's source to is asked of every state a step program is traced for.

``compile()`` refuses an edge whose source field has not the shape its
target's ``BoundaryInputSpec`` declares (``ShapeMismatchError``), on the
state it is given.  ``GraphManager.set_node_state`` takes any layout and is
not a recompile: a field written with one entry for three after
``compile()`` was broadcast into every entry of the node that reads it, by
``step``, ``run_scan`` and every other entry point, with no error or
warning.  A program is traced again whenever a shape changes, so the rule is
asked there, on the host, of the state the program is traced for
(``_graph_specs._refuse_edge_sources``), with ``validate()``'s own line for
each edge that breaks it.

What is asked is exactly what ``compile()`` asks: an edge with a transform,
an input declared with a symbolic dimension and a flux output are held to no
shape by either, and the dtype half of the rule is not asked at a trace at
all (a node's update may widen a field).  The neighbours: the geometry rules
asked where a program is traced (``test_geometry_edge_refusals.py``) and the
checks that are still asked once
(``test_checks_of_state_values_after_compile.py``).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core import _graph_specs
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.coupling.sparse_mapping import sparse_matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.warnings import DtypeMismatchError, ShapeMismatchError


class Cell(SimulationNode):
    """``dx/dt = g u - (1 - k) x`` by one explicit step, on a field of
    shape ``(n,)`` or ``(n, comps)``: ``x <- k x + g u`` at ``dt = 1``.

    ``declared`` is the shape the input ``u`` is declared with when it is
    not the field's own (a symbolic dimension)."""

    def __init__(self, name, timestep, n=3, comps=0, k=0.5, g=0.25, declared=None):
        super().__init__(name, timestep, n=n, comps=comps, k=k, g=g, declared=declared)

    def _shape(self):
        n, comps = int(self.params["n"]), int(self.params["comps"])
        return (n, comps) if comps else (n,)

    def initial_state(self):
        shape = self._shape()
        return {"x": jnp.arange(1, int(np.prod(shape)) + 1, dtype=jnp.float32).reshape(shape)}

    def boundary_input_spec(self):
        shape = self._shape()
        declared = self.params["declared"]
        return {"u": BoundaryInputSpec(shape=shape if declared is None else tuple(declared),
                                       dtype=jnp.float32, default=jnp.zeros(shape, jnp.float32))}

    def update(self, state, boundary_inputs, dt):
        x = state["x"]
        u = boundary_inputs.get("u", jnp.zeros_like(x))
        k, g = jnp.float32(self.params["k"]), jnp.float32(self.params["g"])
        return {**state, "x": x + dt * (g * u - (1 - k) * x)}


_PICK = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], np.float32)


def _plain(timestep_b=1.0):
    gm = GraphManager()
    gm.add_node(Cell("a", 1.0))
    gm.add_node(Cell("b", timestep_b))
    gm.add_edge("a", "b", "x", "u")
    return gm


def _mapped(mapping):
    """Four rows of three components read into two rows of three."""
    gm = GraphManager()
    gm.add_node(Cell("a", 1.0, n=4, comps=3))
    gm.add_node(Cell("b", 1.0, n=2, comps=3))
    gm.add_edge("a", "b", "x", "u", mapping=mapping)
    return gm


# Each graph, and a value for ``a.x`` that broadcasts into what ``b`` reads:
# the silent case.  (A source too LARGE for its target changed the target's
# own shape and was refused already, as a step that changes the layout.)
_GRAPHS = {
    "plain": (_plain, np.asarray([10.0], np.float32), "(1,)", "(3,)"),
    "plain, a scalar": (_plain, np.float32(10.0), "()", "(3,)"),
    "multi-rate": (lambda: _plain(timestep_b=2.0), np.asarray([10.0], np.float32),
                   "(1,)", "(3,)"),
    "dense mapping": (lambda: _mapped(matrix_mapping(_PICK)), np.ones((4, 1), np.float32),
                      "(2, 1)", "(2, 3)"),
    "sparse mapping": (lambda: _mapped(sparse_matrix_mapping(
        np.asarray([[0], [3]]), np.asarray([[1.0], [1.0]], np.float32), n_source=4)),
        np.ones((4, 1), np.float32), "(2, 1)", "(2, 3)"),
}


def _node_states(gm) -> dict:
    return {name: dict(gm.get_node_state(name)) for name in ("a", "b")}


def _whole_state(gm) -> dict:
    """The state as the step function takes it (the step counter of a
    multi-rate graph included)."""
    return {name: dict(fields) for name, fields in gm._state.items()}


def _batched(gm) -> dict:
    """The graph's node states as ``run_sweep`` takes them: two runs."""
    return {name: {f: jnp.stack([v, v]) for f, v in fields.items()}
            for name, fields in _node_states(gm).items()}


_ENTRIES = {
    "step": lambda gm: gm.step(),
    "run": lambda gm: gm.run(2),
    "run_scan": lambda gm: gm.run_scan(2),
    "run_scan_with_history": lambda gm: gm.run_scan_with_history(2),
    "run_sweep": lambda gm: gm.run_sweep(2, _batched(gm)),
    "run_adaptive": lambda gm: gm.run_adaptive(0.02, dt_initial=0.01),
    "run_adaptive_scan": lambda gm: gm.run_adaptive_scan(0.02, max_steps=4, dt_initial=0.01),
    "resolve_boundary_inputs": lambda gm: gm.resolve_boundary_inputs("b"),
    "the step function, handed the state": lambda gm: gm._raw_step_fn(_whole_state(gm), {}),
    "jax.grad of the step function": lambda gm: jax.grad(
        lambda s: jnp.sum(gm._raw_step_fn(s, {})["b"]["x"]), allow_int=True)(_whole_state(gm)),
}
# The adaptive steppers refuse a multi-rate graph before any program is
# traced; every other pair is a case.
_CASES = [(graph, entry) for graph in sorted(_GRAPHS) for entry in sorted(_ENTRIES)
          if not (graph == "multi-rate" and entry.startswith("run_adaptive"))]


@pytest.fixture(scope="module")
def compiled():
    """One compiled, stepped graph per kind, shared by the module: each
    test writes the state it found back."""
    graphs: dict = {}

    def get(kind):
        if kind not in graphs:
            gm = _GRAPHS[kind][0]()
            gm.compile()
            gm.step()
            graphs[kind] = gm
        return graphs[kind]

    return get


def _shape_errors(excinfo) -> list:
    """The messages of an ``ExceptionGroup`` holding only
    ``ShapeMismatchError``, as ``compile()`` raises it."""
    group = excinfo.value
    assert isinstance(group, ExceptionGroup), repr(group)
    assert [type(e) for e in group.exceptions] == [ShapeMismatchError] * len(group.exceptions)
    return [str(e) for e in group.exceptions]


def _shape_lines(gm) -> list:
    return [i for i in gm.validate() if i.startswith("WARNING[shape]")]


@pytest.mark.parametrize("graph, entry", _CASES)
def test_a_source_written_with_another_shape_is_refused_where_the_next_program_is_traced(
        compiled, graph, entry):
    """After ``compile()`` and a step, ``a.x`` written with a shape that
    broadcasts into what ``b`` reads is refused by the next entry point, by
    the edge's name and in ``validate()``'s words, in place of being
    broadcast; nothing is stored, and the graph runs again once the
    compiled shape is written back."""
    _, written, source_shape, declared = _GRAPHS[graph]
    gm = compiled(graph)
    kept = _node_states(gm)
    try:
        gm.set_node_state("a", {"x": jnp.asarray(written)})
        line = (f"WARNING[shape]: edge a.x -> b.u: source shape {source_shape} disagrees "
                f"with target BoundaryInputSpec shape {declared} and no transform is set")
        assert _shape_lines(gm) == [line]
        with pytest.raises(ExceptionGroup, match="edge validation failed") as refused:
            _ENTRIES[entry](gm)
        (message,) = _shape_errors(refused)
        assert message.startswith(line + ".  compile() checks the state it is given")
        # Nothing was stored: ``b`` is what it was, ``a`` what was written.
        after = _node_states(gm)
        np.testing.assert_array_equal(np.asarray(after["b"]["x"]), np.asarray(kept["b"]["x"]))
        assert np.shape(after["a"]["x"]) == np.shape(written)
        # ... and compile() says the same of this state.
        with pytest.raises(ExceptionGroup) as at_compile:
            gm.compile()
        assert _shape_errors(at_compile) == [line]
    finally:
        for name, fields in kept.items():
            gm.set_node_state(name, fields)
    _ENTRIES[entry](gm)
    for name, fields in kept.items():
        gm.set_node_state(name, fields)


def test_the_value_that_was_broadcast_is_no_longer_computed():
    """The reproducer: ``b.x <- b.x / 2 + a.x / 4`` on three entries, with
    ``a.x`` written as one.  The step returned ``b.x`` with that one entry
    in all three; it now raises, and ``b`` keeps its value."""
    gm = _plain()
    gm.compile()
    gm.step()
    before = np.asarray(gm.get_node_state("b")["x"])
    gm.set_node_state("a", {"x": jnp.asarray([10.0], jnp.float32)})
    with pytest.raises(ExceptionGroup) as refused:
        gm.step()
    _shape_errors(refused)
    np.testing.assert_array_equal(np.asarray(gm.get_node_state("b")["x"]), before)
    with pytest.raises(ExceptionGroup) as refused:
        gm.run_scan(3)
    _shape_errors(refused)
    np.testing.assert_array_equal(np.asarray(gm.get_node_state("b")["x"]), before)


# ---------------------------------------------------------------------------
# Inside a coupling group
# ---------------------------------------------------------------------------


def _mean(value):
    return jnp.mean(value, keepdims=True)


def _ring(*, timestep_b=1.0, **group):
    """``a -> b`` entry for entry and ``b -> a`` through its mean, in one
    coupling group.  ``a.x`` written with one entry keeps that shape
    through the group's passes (what it reads back has one entry too), so
    the group solved for ``b`` on a broadcast ``a.x`` and stored it."""
    gm = GraphManager()
    gm.add_node(Cell("a", 1.0, k=0.25))
    gm.add_node(Cell("b", timestep_b, k=0.25))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u", transform=_mean)
    gm.add_coupling_group(["a", "b"], max_iterations=60, tolerance=1e-6, **group)
    return gm


_GROUPS = {
    "gauss-seidel": {},
    "jacobi": dict(iteration_mode="jacobi"),
    "sub-cycled": dict(timestep_b=0.5, subcycling=True, boundary_interpolation="linear"),
    "sub-cycled, jacobi": dict(timestep_b=0.5, subcycling=True, iteration_mode="jacobi",
                               boundary_interpolation="linear"),
    "aitken": dict(acceleration="aitken"),
    "interface norm": dict(convergence_norm="interface"),
    # These two raised already, as a broadcasting error of the history and
    # of the secant matrices that named no edge.
    "a predictor": dict(predictor="linear"),
    "iqn-imvj": dict(acceleration="iqn-imvj"),
}


@pytest.mark.parametrize("entry", ["step", "run_scan"])
@pytest.mark.parametrize("group", sorted(_GROUPS))
def test_a_group_members_source_written_with_another_shape_is_refused(group, entry):
    """A coupled solve is part of the program its step traces: the rule is
    asked before the first pass, in every iteration mode, for a sub-cycled
    member and whatever the group's acceleration."""
    gm = _ring(**_GROUPS[group])
    gm.compile()
    gm.step()
    before = np.asarray(gm.get_node_state("b")["x"])
    gm.set_node_state("a", {"x": jnp.asarray([10.0], jnp.float32)})
    with pytest.raises(ExceptionGroup) as refused:
        _ENTRIES[entry](gm)
    (message,) = _shape_errors(refused)
    assert message.startswith("WARNING[shape]: edge a.x -> b.u: source shape (1,) disagrees")
    np.testing.assert_array_equal(np.asarray(gm.get_node_state("b")["x"]), before)
    gm.set_node_state("a", {"x": jnp.asarray([4.0, 8.0, 12.0], jnp.float32)})
    _ENTRIES[entry](gm)
    assert bool(gm.coupling_diagnostics()["a+b"]["converged"])


def test_a_source_outside_a_group_is_asked_when_the_groups_step_is_traced():
    """An edge INTO a group from a node outside it is asked too, and on a
    graph that is multi-rate as well."""
    gm = _ring()
    gm.add_node(Cell("c", 2.0))
    gm.add_edge("c", "b", "x", "u", additive=True)
    gm.compile()
    gm.step()
    gm.set_node_state("c", {"x": jnp.asarray([10.0], jnp.float32)})
    with pytest.raises(ExceptionGroup) as refused:
        gm.step()
    (message,) = _shape_errors(refused)
    assert message.startswith("WARNING[shape]: edge c.x -> b.u: source shape (1,) disagrees")


# ---------------------------------------------------------------------------
# The rule is compile()'s: every edge at once, and nothing compile() accepts
# ---------------------------------------------------------------------------


def test_every_edge_that_breaks_the_rule_is_named_in_one_group_as_compile_names_them():
    gm = GraphManager()
    gm.add_node(Cell("a", 1.0))
    gm.add_node(Cell("b", 1.0))
    gm.add_node(Cell("c", 1.0))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("a", "c", "x", "u")
    gm.add_edge("b", "c", "x", "u", additive=True)
    gm.compile()
    gm.step()
    gm.set_node_state("a", {"x": jnp.asarray([10.0], jnp.float32)})
    with pytest.raises(ExceptionGroup) as refused:
        gm.step()
    messages = _shape_errors(refused)
    assert [m.split(":")[1] for m in messages] == [" edge a.x -> b.u", " edge a.x -> c.u"]
    assert [m.split(".  compile()")[0] for m in messages] == _shape_lines(gm)


class Fluxed(Cell):
    """A ``Cell`` whose flux hook returns ``x`` and ``flow``, both of one
    entry whatever its state: flux outputs, which no rule holds to a shape."""

    def update(self, state, boundary_inputs, dt):
        if "x" not in state:
            return dict(state)
        return super().update(state, boundary_inputs, dt)

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        return {"flow": jnp.full((1,), 2.0, jnp.float32), "x": jnp.full((1,), 3.0, jnp.float32)}


def _transformed():
    gm = GraphManager()
    gm.add_node(Cell("a", 1.0))
    gm.add_node(Cell("b", 1.0))
    gm.add_edge("a", "b", "x", "u", transform=_mean)
    return gm


def _symbolic():
    gm = GraphManager()
    gm.add_node(Cell("a", 1.0))
    gm.add_node(Cell("b", 1.0, declared=(-1,)))
    gm.add_edge("a", "b", "x", "u")
    return gm


def _flux_sourced():
    gm = GraphManager()
    gm.add_node(Fluxed("a", 1.0))
    gm.add_node(Cell("b", 1.0))
    gm.add_edge("a", "b", "flow", "u")
    return gm


def _undeclared():
    gm = GraphManager()
    gm.add_node(Cell("a", 1.0))
    gm.add_node(Cell("b", 1.0))
    gm.add_edge("a", "b", "x", "not_an_input")
    return gm


# (``resolve_boundary_inputs`` reads no flux output: a ``KeyError``, as documented.)
@pytest.mark.parametrize("graph, entry", [
    (graph, entry) for graph in (_transformed, _symbolic, _flux_sourced, _undeclared)
    for entry in ("step", "run_scan", "resolve_boundary_inputs")
    if not (graph is _flux_sourced and entry == "resolve_boundary_inputs")
], ids=lambda v: v if isinstance(v, str) else v.__name__.strip("_"))
def test_a_shape_compile_accepts_is_not_refused_at_a_trace(graph, entry):
    """No graph that compiles is refused where it is stepped.  ``compile()``
    holds no edge with a transform to a shape (the transform may reshape),
    no input declared with a symbolic dimension, no flux output and no
    input its target does not declare: ``a.x`` written with one entry
    compiles in each of these graphs, and steps."""
    gm = graph()
    gm.compile()
    gm.step()
    gm.set_node_state("a", {"x": jnp.asarray([10.0], jnp.float32)})
    assert not _shape_lines(gm)
    _ENTRIES[entry](gm)
    gm.compile()
    _ENTRIES[entry](gm)


@pytest.mark.parametrize("written, lines", [
    (np.ones(3, np.float32), 0),
    (np.ones(1, np.float32), 1),
    (np.float32(1.0), 1),
    (np.ones((1, 3), np.float32), 1),
    (np.ones((3, 1), np.float32), 1),
    (np.ones(6, np.float32), 1),
], ids=lambda v: str(np.shape(v)) if isinstance(v, (np.ndarray, np.floating)) else str(v))
def test_a_trace_refuses_exactly_the_shapes_validate_names(compiled, written, lines):
    """One rule, two places: a step raises for a written shape if and only
    if ``validate()`` writes a ``WARNING[shape]`` line for that state."""
    gm = compiled("plain")
    kept = _node_states(gm)
    try:
        gm.set_node_state("a", {"x": jnp.asarray(written)})
        named = _shape_lines(gm)
        assert len(named) == lines
        if named:
            with pytest.raises(ExceptionGroup) as refused:
                gm.resolve_boundary_inputs("b")
            assert [m.split(".  compile()")[0] for m in _shape_errors(refused)] == named
            with pytest.raises(ExceptionGroup) as refused:
                gm._raw_step_fn(_whole_state(gm), {})
            assert [m.split(".  compile()")[0] for m in _shape_errors(refused)] == named
        else:
            gm.resolve_boundary_inputs("b")
            gm.step()
    finally:
        for name, fields in kept.items():
            gm.set_node_state(name, fields)


def test_a_mapping_that_declares_its_field_shapes_is_read_through_them():
    """The leading axes a mapping reads are replaced by the ones it
    delivers before the comparison: all of them for a mapping with
    ``field_shapes()``, axis 0 by ``n_target`` for any other."""

    class Shaped:
        n_target = 4

        def field_shapes(self):
            return ((2, 3), (4,))

    class Flat:
        n_target = 4

    edge = _graph_specs.EdgeSpec("a", "b", "x", "u")
    shaped = _graph_specs._edge_mapping_leads(Shaped())
    flat = _graph_specs._edge_mapping_leads(Flat())
    assert shaped == ((2, 3), (4,)) and flat is not None and flat[1] == (4,)
    assert _graph_specs._edge_mapping_leads(None) is None
    issue = _graph_specs._edge_shape_issue
    assert issue(edge, np.ones((2, 3, 5)), (4, 5), shaped) is None
    assert "source shape (4, 1)" in (issue(edge, np.ones((2, 3, 1)), (4, 5), shaped) or "")
    assert issue(edge, np.ones((7, 5)), (4, 5), flat) is None
    assert "source shape (4, 1)" in (issue(edge, np.ones((7, 1)), (4, 5), flat) or "")
    # A scalar has no axis for a mapping to replace; no value, no shape.
    assert "source shape ()" in (issue(edge, np.float32(1.0), (4,), flat) or "")
    assert issue(edge, None, (4,), flat) is None
    assert issue(edge, np.ones(3), None, None) is None


# ---------------------------------------------------------------------------
# A source field that is no longer in the state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", ["step", "run", "run_scan", "run_adaptive"])
def test_a_source_field_dropped_from_the_state_is_named_by_its_edge(entry):
    """``compile()`` refuses an edge whose source is not a field of its
    node.  A state written without the field afterwards was a bare
    ``KeyError: 'x'`` from whatever read it first; it is a ``KeyError``
    that names the edge and the fields the node holds."""
    gm = _plain()
    gm.compile()
    gm.step()
    gm.set_node_state("a", {"y": jnp.zeros(3, jnp.float32)})
    with pytest.raises(KeyError, match=r"edge a\.x -> b\.u: source field 'x' is not in the "
                                       r"state of node 'a' \(available: \['y'\]\)"):
        _ENTRIES[entry](gm)
    gm.set_node_state("a", {"x": jnp.ones(3, jnp.float32)})
    _ENTRIES[entry](gm)


def test_a_field_a_flux_hook_can_supply_is_not_asked_for_in_the_state():
    """A node with a flux hook may deliver a source as a flux output, which
    the step reads when the state has no such field: left to the step."""
    gm = GraphManager()
    gm.add_node(Fluxed("a", 1.0))
    gm.add_node(Cell("b", 1.0, declared=(-1,)))
    gm.add_edge("a", "b", "x", "u")
    gm.compile()
    gm.step()
    held = np.asarray(gm.get_node_state("b")["x"])
    gm.set_node_state("a", {"y": jnp.zeros(3, jnp.float32)})
    out = gm.step()
    # ``b`` read the flux ``x`` (3.0 on one entry): x / 2 + 3 / 4.
    np.testing.assert_allclose(np.asarray(out["b"]["x"]), 0.5 * held + 0.75, rtol=1e-6)


# ---------------------------------------------------------------------------
# What the trace is not asked: the dtype half of compile()'s rule
# ---------------------------------------------------------------------------


def test_an_integer_written_into_a_float_source_is_named_by_validate_and_not_by_the_step():
    """``set_node_state`` does not hold a write to the layout it replaces
    (``load_state`` and ``PUT /graph/state`` do), and the dtype rule is not
    asked where a program is traced: an update may widen float32 to float64
    under x64, which ``compile()`` accepts as the state the graph produced.
    So an int32 written into a float32 source that an edge reads steps:
    here the reader takes the integers as float32, and the node that holds
    them returns float32, which the step refuses as a change of layout.
    ``validate()`` names the edge, and the next ``compile()`` refuses."""
    gm = _plain()
    gm.compile()
    gm.step()
    gm.set_node_state("a", {"x": jnp.asarray([10, 20, 30], jnp.int32)})
    assert any(i.startswith("WARNING[dtype]: edge a.x -> b.u: source dtype int32")
               for i in gm.validate())
    assert not _shape_lines(gm)
    # Not a ShapeMismatchError or a DtypeMismatchError: the layout rule of
    # the stepped state, which names the leaf and stores nothing.
    with pytest.raises(ValueError, match="'a/x' has dtype int32 before the update"):
        gm.step()
    assert gm.get_node_state("a")["x"].dtype == jnp.int32
    with pytest.raises(ExceptionGroup) as refused:
        gm.compile()
    assert [type(e) for e in refused.value.exceptions] == [DtypeMismatchError]


# ---------------------------------------------------------------------------
# Asked once per trace, on the host
# ---------------------------------------------------------------------------


def test_the_rule_is_asked_when_a_program_is_traced_and_not_at_any_other_step(monkeypatch):
    """A step that is not traced again does not come to the check, and a
    write that changes a shape always does (it retraces)."""
    asked = []
    real = _graph_specs._refuse_edge_sources

    def counted(rules, state, **kwargs):
        asked.append(np.shape(state["a"]["x"]))
        return real(rules, state, **kwargs)

    monkeypatch.setattr(_graph_specs, "_refuse_edge_sources", counted)
    gm = _plain()
    gm.compile()
    for _ in range(4):
        gm.step()
    assert asked == [(3,)]
    gm.set_node_state("a", {"x": jnp.asarray([7.0, 8.0, 9.0], jnp.float32)})
    for _ in range(3):
        gm.step()
    assert asked == [(3,)]
    gm.set_node_state("a", {"x": jnp.asarray([10.0], jnp.float32)})
    for _ in range(2):
        with pytest.raises(ExceptionGroup):
            gm.step()
    assert asked == [(3,), (1,), (1,)]


def test_the_check_adds_nothing_to_the_step_program(monkeypatch):
    """The program a step traces is the same with the check and without
    it: it reads shapes on the host and no ``boundary_input_spec()`` is
    called inside a trace."""
    def program():
        gm = _mapped(matrix_mapping(_PICK))
        gm.compile()
        state = _node_states(gm)
        return str(jax.make_jaxpr(lambda s: gm._raw_step_fn(s, {}))(state))

    with_check = program()
    monkeypatch.setattr(_graph_specs, "_refuse_edge_sources", lambda *a, **k: None)
    monkeypatch.setattr(_graph_specs, "_edge_source_rules", lambda *a, **k: ())
    assert program() == with_check


def test_the_inputs_of_a_node_can_still_be_read_under_a_transform(compiled):
    """``resolve_boundary_inputs``'s reader is traceable in the state: under
    ``jit`` and ``vmap`` over a history the rule is asked of each sample's
    shape, and a batch of states of the compiled shape is read."""
    gm = compiled("plain")
    history = _batched(gm)
    read = jax.jit(jax.vmap(lambda s: gm._boundary_inputs_from(s, "b")["u"]))
    np.testing.assert_array_equal(np.asarray(read(history)), np.asarray(history["a"]["x"]))
    history["a"]["x"] = history["a"]["x"][:, :1]
    with pytest.raises(ExceptionGroup) as refused:
        read(history)
    _shape_errors(refused)
