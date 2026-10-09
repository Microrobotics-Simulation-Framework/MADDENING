"""What the graph does when a value a check read at ``compile()`` changes afterwards.

``validate()`` and ``compile()`` ask their questions of the state and the
parameters the graph holds when they are called.  A value can change after
that without a recompile: a node's own ``update``, ``set_node_state`` (which
``load_state`` and ``PUT /graph/state`` write through), a parameter write, a
``params=`` pytree handed to one step.  Each test here is one such check and
one such door, and pins what the library then does: asks again, refuses, or
steps to the right answer anyway.  Three neighbours have files of their
own: the geometry rules asked where a program is traced
(``test_geometry_edge_refusals.py``), the shape an edge's target declares,
asked there too (``test_edge_shape_rule_at_every_trace.py``), and the
underflow-range warning (``test_coupling_underflow_range_warning.py``).

Three doors differ in what they can change.  ``load_state`` and ``PUT
/graph/state`` refuse a field of another shape and cast to the live dtype, so
only values pass; a node's ``update`` is held to the layout it was given
(once per trace of the step); ``GraphManager.set_node_state`` takes any
layout, which is why most of these tests write through it.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.coupling.sparse_mapping import sparse_matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.transforms import extract_first, extract_last
from maddening.nodes.heat import HeatNode
from maddening.warnings import DtypeMismatchError, ShapeMismatchError
from tests.property import geometry_graphs as gg


class Cell(SimulationNode):
    """``x <- k x + u`` on ``n`` entries of ``dtype``."""

    def __init__(self, name, timestep, n=3, k=0.5, dtype="float32"):
        super().__init__(name, timestep, n=n, k=k, dtype=dtype)

    def initial_state(self):
        return {"x": jnp.arange(1, int(self.params["n"]) + 1, dtype=self.params["dtype"])}

    def boundary_input_spec(self):
        n, dtype = int(self.params["n"]), self.params["dtype"]
        return {"u": BoundaryInputSpec(shape=(n,), dtype=jnp.dtype(dtype),
                                       default=jnp.zeros(n, dtype))}

    def update(self, state, boundary_inputs, dt):
        x = state["x"]
        u = boundary_inputs.get("u", jnp.zeros_like(x))
        return {"x": jnp.asarray(self.params["k"], x.dtype) * x + u}


def _quarter(value):
    return 0.25 * value


def _ring(*, dtype="float32", tolerance=1e-6, **group):
    """``a <-> b``, each reading a quarter of the other, in one coupling group."""
    gm = GraphManager()
    gm.add_node(Cell("a", 1.0, k=0.25, dtype=dtype))
    gm.add_node(Cell("b", 1.0, k=0.25, dtype=dtype))
    gm.add_edge("a", "b", "x", "u", transform=_quarter)
    gm.add_edge("b", "a", "x", "u", transform=_quarter)
    gm.add_coupling_group(["a", "b"], max_iterations=80, tolerance=tolerance, **group)
    return gm


def _x(gm) -> dict:
    return {name: np.asarray(gm.get_node_state(name)["x"], np.float64) for name in ("a", "b")}


def _apart(got: dict, want: dict) -> float:
    return max(float(np.max(np.abs(got[k] - want[k]) / np.abs(want[k]))) for k in want)


# ---------------------------------------------------------------------------
# validate()'s edge rules: asked of the state held when validate() is called
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("written, issue, error", [
    (np.asarray([10.0], np.float32), "WARNING[shape]: edge a.x -> b.u: source shape (1,)",
     ShapeMismatchError),
    (np.asarray([10, 20, 30], np.int32), "WARNING[dtype]: edge a.x -> b.u: source dtype int32",
     DtypeMismatchError),
])
def test_validate_and_the_next_compile_name_a_shape_or_dtype_written_after_compile(
        written, issue, error):
    """``compile()`` holds an edge's source to its target's declared shape
    and dtype on the state it is given.  Asked again after a
    ``set_node_state`` that breaks the rule, ``validate()`` names the edge
    and ``compile()`` refuses the graph.  (A step asks the shape rule too,
    where it is traced, and not the dtype rule:
    ``test_edge_shape_rule_at_every_trace.py``.)"""
    gm = GraphManager()
    gm.add_node(Cell("a", 1.0))
    gm.add_node(Cell("b", 1.0))
    gm.add_edge("a", "b", "x", "u")
    gm.compile()
    gm.step()
    assert not [i for i in gm.validate() if i.startswith(("WARNING", "ERROR"))]
    gm.set_node_state("a", {"x": jnp.asarray(written)})
    assert any(i.startswith(issue) for i in gm.validate()), gm.validate()
    with pytest.raises(Exception) as refused:
        gm.compile()
    assert [type(e) for e in refused.value.exceptions] == [error]


# ---------------------------------------------------------------------------
# A coupling group: what compile() seeds from the state
# ---------------------------------------------------------------------------


def test_a_group_field_written_as_another_kind_of_dtype_is_refused_by_the_step():
    """The group's plan calls a field floating on the state ``compile()``
    saw; an integer written into it afterwards does not step: the stepped
    state has not the layout it was given, and nothing is stored."""
    gm = _ring()
    gm.compile()
    gm.step()
    written = jnp.asarray([4, 8, 12], jnp.int32)
    gm.set_node_state("a", {"x": written})
    with pytest.raises(ValueError, match="'a/x' has dtype int32 before the update"):
        gm.step()
    assert gm.get_node_state("a")["x"].dtype == jnp.int32
    with pytest.raises(TypeError, match="carry"):
        gm.run_scan(2)


def test_a_group_compiled_in_float32_solves_a_float64_state_to_a_float64_tolerance():
    """The report slots keep the dtype ``compile()`` seeded them with; the
    residual the solve stops on is computed in the dtype of the state it is
    handed.  A float64 state written into a group compiled on float32 is
    solved to ``1e-10``, as by a group built in float64."""
    with gg.x64(True):
        want_gm = _ring(dtype="float64", tolerance=1e-10)
        want_gm.compile()
        want_gm.step()
        want = _x(want_gm)

        gm = _ring(tolerance=1e-10)
        gm.compile()
        for name in ("a", "b"):
            gm.set_node_state(name, {"x": jnp.arange(1, 4, dtype=jnp.float64)})
        gm.step()
        assert gm.get_node_state("a")["x"].dtype == jnp.float64
        assert _apart(_x(gm), want) < 1e-9
        # A solve stopped at float32's resolution would be about 1e-7 apart.
        assert bool(gm.coupling_diagnostics()["a+b"]["converged"])


_NEW = {"a": [40.0, -8.0, 12.0], "b": [-3.0, 5.0, 70.0]}


def _written(gm):
    for name, x in _NEW.items():
        gm.set_node_state(name, {"x": jnp.asarray(x, jnp.float32)})


@pytest.fixture(scope="module")
def one_step_from_the_written_state():
    """A graph that holds no warm start, stepped once from ``_NEW``."""
    fresh = _ring(convergence_norm="mixed")
    fresh.compile()
    _written(fresh)
    fresh.step()
    return _x(fresh)


@pytest.mark.parametrize("warm", [dict(predictor="quadratic"), dict(acceleration="iqn-imvj")],
                         ids=lambda w: "-".join(w.values()))
def test_a_warm_start_from_an_older_trajectory_does_not_move_the_answer_after_a_write(
        warm, one_step_from_the_written_state):
    """Predictor histories and quasi-Newton secants are state seeded at
    ``compile()`` and grown by the steps.  After a mid-run ``set_node_state``
    they describe another trajectory: the solve starts from a poorer guess
    and converges to the answer of a graph that never held them."""
    gm = _ring(convergence_norm="mixed", **warm)
    gm.compile()
    for _ in range(4):
        gm.step()
    _written(gm)
    gm.step()
    assert bool(gm.coupling_diagnostics()["a+b"]["converged"])
    assert _apart(_x(gm), one_step_from_the_written_state) < 1e-4


@pytest.mark.parametrize("warm", [dict(predictor="linear"), dict(acceleration="iqn-imvj")],
                         ids=lambda w: "-".join(w.values()))
def test_a_warm_start_sized_at_compile_does_not_step_a_state_of_another_size(warm):
    """The history and the secant matrices are sized from the state
    ``compile()`` saw: members written with two entries for three raise at
    the next step instead of being solved on the old size."""
    gm = _ring(convergence_norm="l2", **warm)
    gm.compile()
    gm.step()
    for name in ("a", "b"):
        gm.set_node_state(name, {"x": jnp.asarray([4.0, 8.0], jnp.float32)})
    with pytest.raises((TypeError, ValueError), match="[Ii]ncompatible"):
        gm.step()


# ---------------------------------------------------------------------------
# Mapped edges: the size add_edge asked of initial_state(), and the weights
# ---------------------------------------------------------------------------


def _mapped(mapping):
    gm = GraphManager()
    gm.add_node(Cell("a", 1.0, n=4))
    gm.add_node(Cell("b", 1.0, n=2))
    gm.add_edge("a", "b", "x", "u", mapping=mapping)
    return gm


_PICK = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]], np.float32)
_MAPPINGS = {
    "dense": (lambda: matrix_mapping(_PICK), "dot_general|contracting"),
    "sparse": (lambda: sparse_matrix_mapping(np.asarray([[0], [3]]),
                                             np.asarray([[1.0], [1.0]], np.float32), n_source=4),
               "its first axis must be n_source = 4"),
}


@pytest.mark.parametrize("entries", [2, 6])
@pytest.mark.parametrize("kind", sorted(_MAPPINGS))
def test_a_mapped_source_written_with_another_size_does_not_step(kind, entries):
    """``add_edge`` asks a mapping's ``n_source`` of the node's
    ``initial_state()``, once.  A source written shorter or longer after
    ``compile()`` raises where its program is traced (the sparse mapping by
    name; a dense matrix as the product's own error) and is never gathered
    out of range."""
    make, message = _MAPPINGS[kind]
    gm = _mapped(make())
    gm.compile()
    gm.step()
    gm.set_node_state("a", {"x": jnp.arange(entries, dtype=jnp.float32)})
    with pytest.raises((TypeError, ValueError), match=message):
        gm.step()


def test_mapping_weights_of_another_shape_are_refused_at_every_step():
    """The weights' shape is asked of every step's parameters, the live
    ones and a ``params=`` pytree alike: not once."""
    gm = _mapped(matrix_mapping(_PICK))
    gm.compile()
    gm.step()
    key = "a.x->b.u"
    (leaf,) = gm.params["mappings"][key]
    other = jnp.ones((2, 2), jnp.float32)
    for _ in range(2):
        with pytest.raises(ValueError, match=r"has shape \(2, 2\), expected \(2, 4\)"):
            gm.step(params={"mappings": {key: {leaf: other}}})
    gm.step()
    gm.params["mappings"][key][leaf] = other
    for _ in range(2):
        with pytest.raises(ValueError, match=r"has shape \(2, 2\), expected \(2, 4\)"):
            gm.step()


# ---------------------------------------------------------------------------
# Advisories that read parameters: asked by validate() and compile() only
# ---------------------------------------------------------------------------


def _rods(alpha):
    gm = GraphManager()
    for name in ("a", "b"):
        gm.add_node(HeatNode(name, timestep=1.0, n_cells=8, length=8.0,
                             thermal_diffusivity=alpha))
    gm.add_edge("a", "b", "temperature", "left_temperature", transform=extract_last)
    gm.add_edge("b", "a", "temperature", "right_temperature", transform=extract_first)
    gm.add_coupling_group(["a", "b"], max_iterations=50, tolerance=1e-6)
    return gm


def _pair_advisories(gm) -> list:
    return [i for i in gm.validate() if i.startswith("WARNING: HeatNode rods")]


def test_the_coupled_pair_advisory_is_asked_of_the_parameters_held_when_validate_is_called():
    """Two rods coupled end to end past their pair limit: the advisory
    reads the live parameters, so a diffusivity written after ``compile()``
    is named by the next ``validate()`` and warned of by the next
    ``compile()``; the steps in between are not asked."""
    gm = _rods(0.30)                       # Fourier number 0.30, under 3/8
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.compile()
        gm.step()
        assert _pair_advisories(gm) == []
        for name in ("a", "b"):
            gm.params["nodes"][name]["thermal_diffusivity"] = jnp.asarray(0.45, jnp.float32)
        gm.step()                          # not asked: no warning
    assert len(_pair_advisories(gm)) == 1 and "Fo = 0.45" in _pair_advisories(gm)[0]
    with pytest.warns(UserWarning, match="HeatNode rods 'a' and 'b' are coupled end to end"):
        gm.compile()


def test_the_coarse_grid_advisory_of_a_geometry_is_validates_alone():
    """A ``multilinear_grid`` a float32 geometry locates to between 1/1024
    and 1/16 of a cell: ``compile()`` warns.  Written as float32 after a
    float64 compile, the geometry meets the refusal (1/16) where its program
    is traced and not this advisory, which ``validate()`` gives when asked
    (the registry's residual risk of MADD-ANO-237)."""
    class Markers(Cell):
        def __init__(self, name, timestep, n=2, k=0.5, dtype="float32", g_dtype="float64"):
            SimulationNode.__init__(self, name, timestep, n=n, k=k, dtype=dtype,
                                    g_dtype=g_dtype)

        def initial_state(self):
            return {**super().initial_state(),
                    "g": jnp.asarray([[100.0013], [100.0027]], self.params["g_dtype"])}

        def update(self, state, boundary_inputs, dt):
            return {**super().update(state, boundary_inputs, dt), "g": state["g"]}

    def graph(g_dtype):
        gm = GraphManager()
        gm.add_node(Cell("a", 1.0, n=4))
        gm.add_node(Markers("b", 1.0, g_dtype=g_dtype))
        gm.add_edge("a", "b", "x", "u", geometry=("target", "g"), mapping=gg.multilinear(
            (100.0,), (1.0e-3,), (4,), n_points=2, mode="consistent"))
        return gm

    advisory = "a float32 geometry locates a point on axis 0"
    with gg.x64(True):
        with pytest.warns(UserWarning, match=advisory):
            graph("float32").compile()
        gm = graph("float64")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            gm.compile()
            gm.step()
            state = dict(gm.get_node_state("b"))
            state["g"] = jnp.asarray(np.asarray(state["g"]), jnp.float32)
            gm.set_node_state("b", state)
            gm.step()                      # traced again; the advisory is not raised here
        assert any(advisory in issue and issue.startswith("WARNING") for issue in gm.validate())
