"""A coupling group whose member holds a floating field with no entries (an
empty contact set, a list with none this configuration) steps under every
solver, norm and acceleration, with or without the diagnostics, and returns
and reports what the same group without the field does.

**The rule.**  A field with no entries is not read
(``maddening.core.coupling.acceleration._has_entries``): it has no
magnitude, so it carries no norm, no weight, no floor term and no gain, and
an edge that delivers none is not an edge the interface norm reads.  The
residual norms always skipped such a field; the report's analysis under
``solver="ift"`` with ``diagnostics=True`` took ``max|field|`` of it and
failed to trace, and an accelerator whose vector had no entries failed
under either solver.

**What is asserted.**  For each cell of the battery -- where the field sits
(``empty_field_graphs.SHAPES``), the norm, the solver, diagnostics off and
on, the acceleration, the sweep order, the dtype -- the graph steps; the
state of every other field, every number of ``coupling_diagnostics()`` and
the gradient through a step equal, **to the bit**, those of the *twin*
graph built without the field; and the field itself is handed back with no
entries and its dtype.  To the bit, although the two compiled programs are
not the same text: the graph's differs from the twin's by operands that
have no entries, which add no arithmetic (checked on every jaxlib CI runs).
The exceptions are stated at their tests: a field with no entries whose
dtype is *wider* than every other field's still widens the dtype the group
iterates in, so its report agrees with the twin's to rounding; and a group
that exchanges nothing else has an accelerator with nothing to act on.
"""

from __future__ import annotations

import functools
import math

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.core import empty_field_graphs as eg

F32 = jnp.float32
NORMS = ("l2", "mixed", "interface")


# ---------------------------------------------------------------------------
# The reproducer: two relays and a field with no entries beside the one
# their edges carry
# ---------------------------------------------------------------------------


class _Relay(SimulationNode):
    """``x <- 1 + u0 / 4``, beside a field that has no entries."""

    def initial_state(self):
        return {"x": jnp.full((2,), 0.5, F32), "empty": jnp.zeros((0,), F32)}

    def boundary_input_spec(self):
        return {"u0": BoundaryInputSpec(shape=(2,), dtype=F32, default=jnp.zeros((2,), F32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": (0.25 * boundary_inputs["u0"] + 1.0).astype(F32), "empty": state["empty"]}


def _stepped(norm, solver, diagnostics):
    gm = GraphManager()
    gm.add_node(_Relay("a", 1.0))
    gm.add_node(_Relay("b", 1.0))
    gm.add_edge("b", "a", "x", "u0")
    gm.add_edge("a", "b", "x", "u0")
    gm.add_coupling_group(["a", "b"], convergence_norm=norm, solver=solver,
                          diagnostics=diagnostics, rtol=1e-6, tolerance=1e-6)
    gm.compile()
    gm.step()
    return gm


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("solver, diagnostics", [("fori", True), ("ift", False)])
def test_a_group_with_an_empty_field_steps(norm, solver, diagnostics):
    gm = _stepped(norm, solver, diagnostics)
    assert gm.coupling_diagnostics()["a+b"]["converged"] is True
    assert float(jnp.max(jnp.abs(gm.get_node_state("a")["x"] - 4.0 / 3.0))) < 1e-4


@pytest.mark.parametrize("norm", NORMS)
def test_a_group_with_an_empty_field_steps_under_ift_with_diagnostics(norm):
    gm = _stepped(norm, "ift", True)
    report = gm.coupling_diagnostics()["a+b"]
    assert report["converged"] is True
    assert float(jnp.max(jnp.abs(gm.get_node_state("a")["x"] - 4.0 / 3.0))) < 1e-4
    # The analysis the field used to stop is carried out: a Gauss-Seidel
    # pass of the pair contracts at (1/4)**2.
    assert abs(report["rho_spectral"] - 0.0625) < 1e-4
    assert math.isfinite(report["spectral_error_bound"])


# ---------------------------------------------------------------------------
# The battery: every cell against its twin
# ---------------------------------------------------------------------------

_SHORT = {"gauss-seidel": "gs", "jacobi": "jacobi", "float32": "f32", "float64": "f64",
          "float16": "f16", "bfloat16": "bf16"}


def _cell(shape, norm, solver, diagnostics, acceleration="none", mode="gauss-seidel",
          dtype="float32", none_dtype=None, gradient=False, marks=()):
    """One cell of the battery, as a ``pytest.param`` with a readable id."""
    label = "-".join(
        [shape, norm, solver + ("+diag" if diagnostics else ""), acceleration, _SHORT[mode],
         _SHORT[dtype]] + ([f"none_{_SHORT[none_dtype]}"] if none_dtype else [])
        + (["grad"] if gradient else []))
    return pytest.param((shape, norm, solver, diagnostics, acceleration, mode, dtype,
                         none_dtype, gradient), id=label, marks=marks)


def _knobs(norm, solver, diagnostics, acceleration, mode):
    knobs = dict(convergence_norm=norm, solver=solver, diagnostics=diagnostics,
                 acceleration=acceleration, iteration_mode=mode, max_iterations=40,
                 rtol=1e-6, tolerance=1e-6)
    if acceleration == "fixed":
        knobs["relaxation"] = 0.8
    return knobs


@functools.lru_cache(maxsize=None)
def _run(shape, norm, solver, diagnostics, acceleration, mode, dtype, none_dtype, gradient,
         empty, wide):
    """``(outcome, the fields with no entries were handed back)`` of one
    graph, compiled and stepped once per session: the cells share twins."""
    with eg.x64(wide):
        gm = eg.build(shape, _knobs(norm, solver, diagnostics, acceleration, mode), empty=empty,
                      dtype=dtype, none_dtype=none_dtype)
        out = eg.outcome(gm, with_gradient=gradient)
        kept = eg.none_fields_are_kept(gm, none_dtype or dtype) if empty else None
    return out, kept


def _graph(cell):
    shape, norm, solver, diagnostics, acceleration, mode, dtype, none_dtype, gradient = cell
    return _run(shape, norm, solver, diagnostics, acceleration, mode, dtype, none_dtype, gradient,
                True, "float64" in (dtype, none_dtype))


def _twin(cell):
    """The outcome of *cell*'s twin: the graph without the field (one graph
    for every shape that differs from it only by the field)."""
    shape, norm, solver, diagnostics, acceleration, mode, dtype, none_dtype, gradient = cell
    return _run(eg.TWIN[shape], norm, solver, diagnostics, acceleration, mode, dtype, None,
                gradient, False, "float64" in (dtype, none_dtype))[0]


def _assert_reports_what_its_twin_does(cell):
    got, kept = _graph(cell)
    twin = _twin(cell)
    assert kept, "a field with no entries was not handed back as it was"
    assert eg.differences(got, twin) == []
    if cell[-1]:
        assert np.all(np.isfinite(got["gradient"])) and np.any(got["gradient"] != 0)


#: Every push: each site the rule is applied at, under the analysis that
#: reads it, and each shape once.  A cell compiles at most one graph the
#: twins' test before it has not already compiled.
PER_PUSH = [
    # The interface norm reads the edge's source, twice: the weights, the
    # reading's analysis and the measured evaluation count.
    _cell("read_twice", "interface", "ift", True),
    # A coarser dtype on the field: the coarsest eps of the products.
    _cell("unread", "mixed", "ift", True, "aitken", "jacobi", none_dtype="float16"),
    # A member with no other floating field, and a mapped edge beside one.
    _cell("only_field", "l2", "ift", True),
    _cell("source_mapped", "interface", "ift", True, "iqn-ils", "jacobi"),
    # The solvers and settings that always stepped, with the gradient.
    _cell("source", "l2", "ift", False, "aitken", gradient=True),
    _cell("only_field", "mixed", "fori", True, "fixed", "jacobi", gradient=True),
    _cell("both_ways", "interface", "fori", False, "iqn-imvj", gradient=True),
    _cell("transformed", "interface", "ift", False, "iqn-ils", "jacobi"),
    _cell("unread_mapped", "mixed", "fori", True, "none", "jacobi", dtype="float64"),
]


def _twins_of(cells):
    """One cell per distinct twin of *cells*, in their order."""
    seen, out = set(), []
    for param in cells:
        (shape, norm, solver, diagnostics, acceleration, mode, dtype, none_dtype,
         gradient) = param.values[0]
        key = (eg.TWIN[shape], norm, solver, diagnostics, acceleration, mode, dtype, gradient,
               "float64" in (dtype, none_dtype))
        if key not in seen:
            seen.add(key)
            out.append(pytest.param(param.values[0], id=param.id))
    return out


@pytest.mark.parametrize("cell", _twins_of(PER_PUSH))
def test_the_twin_without_the_field_converges_and_reports_numbers(cell):
    """The comparison below is not between two failures: the twin of every
    per-push cell converges, and where the diagnostics are on under
    ``solver="ift"`` its bounds are finite and usable."""
    twin = _twin(cell)
    _shape, _norm, solver, diagnostics, *_ = cell
    for fields in twin["state"].values():
        for value in fields.values():
            assert np.all(np.isfinite(value.astype(np.float64)))
    if twin["report"]:
        assert twin["report"]["converged"] is True
    if solver == "ift" and diagnostics:
        assert twin["report"]["spectral_usable"] is True
        assert twin["report"]["gradient_bound_usable"] is True
        assert math.isfinite(twin["report"]["spectral_error_bound"])
        assert math.isfinite(twin["report"]["gradient_relative_error_bound"])


@pytest.mark.parametrize("cell", PER_PUSH)
def test_a_group_reports_what_its_twin_without_the_field_does(cell):
    _assert_reports_what_its_twin_does(cell)


# Per push: tests/core/test_coupling_diagnostics_with_an_empty_field.py::test_a_group_reports_what_its_twin_without_the_field_does[read_twice-interface-ift+diag-none-gs-f32]
# Per push: tests/core/test_coupling_diagnostics_with_an_empty_field.py::test_a_group_reports_what_its_twin_without_the_field_does[only_field-l2-ift+diag-none-gs-f32]
@pytest.mark.slow
@pytest.mark.parametrize("cell", [
    _cell(shape, norm, solver, diagnostics)
    for shape in eg.SHAPES if shape not in eg.DEGENERATE
    for norm in NORMS
    for solver, diagnostics in (("ift", True), ("ift", False), ("fori", True), ("fori", False))
])
def test_every_shape_reports_what_its_twin_does_under_every_norm_and_solver(cell):
    _assert_reports_what_its_twin_does(cell)


# Per push: tests/core/test_coupling_diagnostics_with_an_empty_field.py::test_a_group_reports_what_its_twin_without_the_field_does[source_mapped-interface-ift+diag-iqn-ils-jacobi-f32]
# Per push: tests/core/test_coupling_diagnostics_with_an_empty_field.py::test_a_group_reports_what_its_twin_without_the_field_does[unread-mixed-ift+diag-aitken-jacobi-f32-none_f16]
@pytest.mark.slow
@pytest.mark.parametrize("cell", [
    _cell(shape, norm, "ift", True, acceleration, mode)
    for acceleration in eg.ACCELERATIONS[1:]
    for mode, norm in (("gauss-seidel", "interface"), ("jacobi", "mixed"))
    for shape in ("source", "unread")
])
def test_every_acceleration_and_order_reports_what_its_twin_does(cell):
    _assert_reports_what_its_twin_does(cell)


# Per push: tests/core/test_coupling_diagnostics_with_an_empty_field.py::test_a_group_reports_what_its_twin_without_the_field_does[unread_mapped-mixed-fori+diag-none-jacobi-f64]
@pytest.mark.slow
@pytest.mark.parametrize("cell", [
    _cell(shape, norm, "ift", True, dtype="float64")
    for shape in eg.SHAPES if shape not in eg.DEGENERATE
    for norm in NORMS
])
def test_every_shape_reports_what_its_twin_does_in_float64(cell):
    _assert_reports_what_its_twin_does(cell)


# Per push: tests/core/test_coupling_diagnostics_with_an_empty_field.py::test_a_group_reports_what_its_twin_without_the_field_does[unread-mixed-ift+diag-aitken-jacobi-f32-none_f16]
@pytest.mark.slow
@pytest.mark.parametrize("cell", [
    _cell(shape, norm, "ift", True, none_dtype=none_dtype)
    for none_dtype in ("float16", "bfloat16")
    for shape in ("unread", "source", "transformed")
    for norm in NORMS
])
def test_a_field_with_no_entries_of_a_coarser_dtype_coarsens_nothing(cell):
    """The field's dtype has a larger ``eps`` than any field the pass
    evaluates in.  Nothing is evaluated in it."""
    _assert_reports_what_its_twin_does(cell)


# Per push: tests/core/test_coupling_diagnostics_with_an_empty_field.py::test_a_group_reports_what_its_twin_without_the_field_does[source-l2-ift-aitken-gs-f32-grad]
# Per push: tests/core/test_coupling_diagnostics_with_an_empty_field.py::test_a_group_reports_what_its_twin_without_the_field_does[only_field-mixed-fori+diag-fixed-jacobi-f32-grad]
@pytest.mark.slow
@pytest.mark.parametrize("cell", [
    _cell(shape, NORMS[i % 3], solver, diagnostics, gradient=True)
    for i, shape in enumerate(s for s in eg.SHAPES if s not in eg.DEGENERATE)
    for solver, diagnostics in (("ift", True), ("ift", False), ("fori", True))
])
def test_the_gradient_through_every_shape_is_its_twins(cell):
    """``jax.grad`` through a step, under the diagnostics too: finite, not
    zero, and the twin's to the bit."""
    _assert_reports_what_its_twin_does(cell)


# ---------------------------------------------------------------------------
# The exceptions
# ---------------------------------------------------------------------------


def _close(a, b, rtol):
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=rtol, abs_tol=0.0)
    return type(a) is type(b) and a == b


@pytest.mark.parametrize("norm", NORMS)
def test_a_wider_field_with_no_entries_moves_the_report_by_rounding_only(norm):
    """A float64 field with no entries beside float32 members.

    The dtype a group iterates in, and its report's analysis runs in, is
    the promotion of its floating fields' dtypes, and a field with no
    entries still has one: this group's fixed-point vector is float64 and
    its twin's float32.  That is a dtype, not a number the field
    contributes -- each field keeps its own ``eps`` in every floor -- so
    under ``acceleration="none"`` (no arithmetic on the vector) the state
    and the residual are the twin's to the bit, and the report's analysis,
    carried out in the wider dtype, agrees with the twin's to float32
    rounding.
    """
    cell = _cell("unread", norm, "ift", True, none_dtype="float64").values[0]
    got, kept = _graph(cell)
    twin = _twin(cell)
    assert kept
    for name, fields in twin["state"].items():
        for field, value in fields.items():
            np.testing.assert_array_equal(got["state"][name][field], value)
    assert got["report"].keys() == twin["report"].keys()
    for key in ("iterations", "total_iterations", "converged", "residual"):
        assert got["report"][key] == twin["report"][key], key
    for key, value in twin["report"].items():
        assert _close(got["report"][key], value, 1e-5), (key, got["report"][key], value)


#: The settings a group with nothing to iterate on is stepped under on
#: every push: ``(solver, diagnostics, acceleration, gradient taken)``.
#: The first is the reference the others are compared with.
_DEGENERATE_SETTINGS = [
    ("ift", False, "none", True),
    ("ift", False, "iqn-ils", True),
    ("ift", False, "aitken", False),
    ("fori", True, "iqn-imvj", False),
    ("ift", True, "none", False),
]


@pytest.mark.parametrize("shape", eg.DEGENERATE)
@pytest.mark.parametrize("solver, diagnostics, acceleration, gradient", _DEGENERATE_SETTINGS)
def test_a_group_with_nothing_to_iterate_on_steps_and_is_differentiable(
        shape, solver, diagnostics, acceleration, gradient):
    """No entry crosses an internal edge, or no member holds one.

    The accelerator's vector then has no entries (the quasi-Newton
    methods act on the fields the internal edges read), and so, where no
    member holds an entry, do the fixed-point vector, the report's
    weights and the linear system ``jax.grad`` solves.  The group is at
    its fixed point after one pass: it converges with a zero residual,
    the state is the one each member computes alone whatever the
    acceleration, and the gradient is finite and the same.
    """
    cell = _cell(shape, "mixed", solver, diagnostics, acceleration, gradient=gradient).values[0]
    got, kept = _graph(cell)
    report = got["report"]
    assert kept
    assert report["converged"] is True
    assert report["residual"] == 0.0
    plain = _graph(_cell(shape, "mixed", *_DEGENERATE_SETTINGS[0][:3], gradient=True).values[0])[0]
    assert eg.differences({**got, "report": {}, "gradient": None},
                          {**plain, "report": {}, "gradient": None}) == []
    if gradient:
        assert np.all(np.isfinite(got["gradient"])) and np.any(got["gradient"] != 0)
        np.testing.assert_array_equal(got["gradient"], plain["gradient"])
    if solver == "ift" and diagnostics:
        # Reported at its fixed point: a contraction rate of zero and a
        # finite, usable bound; and no gradient bound where the fixed-point
        # vector has no entries (there is no fixed point to be wrong about).
        assert report["rho_spectral"] == 0.0 and report["spectral_usable"] is True
        assert math.isfinite(report["spectral_error_bound"])
        if shape == "no_entries":
            assert report["spectral_error_bound"] == 0.0
            assert report["gradient_bound_usable"] is False


# Per push: tests/core/test_coupling_diagnostics_with_an_empty_field.py::test_a_group_with_nothing_to_iterate_on_steps_and_is_differentiable[ift-True-none-False-no_entries]
# Per push: tests/core/test_coupling_diagnostics_with_an_empty_field.py::test_a_group_with_nothing_to_iterate_on_steps_and_is_differentiable[ift-True-none-False-only_edges]
@pytest.mark.slow
@pytest.mark.parametrize("shape", eg.DEGENERATE)
@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("acceleration", ("none", "iqn-ils"))
def test_a_group_with_nothing_to_iterate_on_is_reported_at_its_fixed_point(
        shape, norm, acceleration):
    """Under ``solver="ift"`` with the diagnostics, every norm and a
    quasi-Newton acceleration, with the gradient taken through them."""
    cell = _cell(shape, norm, "ift", True, acceleration, gradient=True).values[0]
    got, kept = _graph(cell)
    report = got["report"]
    assert kept and report["converged"] is True and report["residual"] == 0.0
    assert report["rho_spectral"] == 0.0 and report["spectral_usable"] is True
    assert math.isfinite(report["spectral_error_bound"])
    assert np.all(np.isfinite(got["gradient"])) and np.any(got["gradient"] != 0)
    if shape == "no_entries":
        assert report["spectral_error_bound"] == 0.0
        assert report["gradient_bound_usable"] is False
