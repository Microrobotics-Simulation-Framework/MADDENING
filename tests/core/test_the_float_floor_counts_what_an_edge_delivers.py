"""A group's coarsest eps counts what its internal edges deliver.

The float floor of a coupling group is counted at the ``eps`` of the
coarsest floating dtype the pass goes through
(``acceleration._group_coarsest_eps``).  That was read from the members'
state fields alone, so a **transform that narrows** between two float64
members (``lambda v: v.astype(float32)``) put a float32 rounding into
every pass that nothing counted: the pair below stalled on it with
``residual`` exactly ``0.0``, reported ``converged=True`` at
``rtol = 1e-12`` 2.8e5 to 3.6e5 tolerances from its fixed point, and read
``spectral_error_bound`` at 5e-8 to 1e-7 of the distance with
``spectral_usable`` and ``gradient_bound_usable`` both set, under all
three norms and both schedules.

The rule now: the coarsest eps is taken over the members' state fields
**and** over what every internal edge delivers, after its mapping and its
transform (``InterfaceEdge.delivered_leaves``: abstract evaluation, the
transform is never run on numbers).  It can only raise a floor; a group
none of whose edges delivers a coarser floating dtype than its fields is
unchanged to the bit.  Not seen: a narrowing inside a member's
``update``, and an edge whose source is not a state field.

**The pair** (every field float64; ``a`` and ``b`` are external inputs)::

    A.x (1 value)    <- bA + 0.9 * u      reads the mean of B.x's 8 values
    B.x (8 values)   <- bB + u            reads A.x, spread onto 8

    A -> B  sparse nearest neighbour, 1 onto 8 (under "interface": read at its source)
    B -> A  sparse matrix, the mean of 8      (under "interface": read as delivered)

Case ``T`` narrows on ``A -> B``, case ``T2`` on ``B -> A``.
"""

from __future__ import annotations

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _coupled_block, _interface_plan
from maddening.core.coupling.acceleration import _group_coarsest_eps, residual_precision_floor
from maddening.core.coupling.sparse_mapping import (
    sparse_matrix_mapping,
    sparse_nearest_neighbor_mapping,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.sparse_mapping_support import x64

KEY = "A+B"
M, N = 1, 8
GAIN = 0.9
RTOL = 1e-12
NORMS = ("interface", "mixed", "l2")
SCHEDULES = ("gauss-seidel", "jacobi")
EPS32 = float(np.finfo(np.float32).eps)
EPS64 = float(np.finfo(np.float64).eps)


def narrow(value):
    """The transform that narrows.  It refuses numbers: the floor's rule
    may only ever ask it for a dtype."""
    assert isinstance(value, jax.core.Tracer), "the transform was run on concrete values"
    return value.astype(jnp.float32)


def keep(value):
    """A transform that delivers the dtype it was handed."""
    return value * 1.0


class Relay(SimulationNode):
    """``x <- b + a * u`` entrywise; ``a`` and ``b`` are external inputs."""

    def __init__(self, name: str, size: int):
        super().__init__(name, 1.0)
        self._n = size

    def initial_state(self):
        return {"x": jnp.zeros(self._n, jnp.float64)}

    def boundary_input_spec(self):
        zero = jnp.zeros(self._n, jnp.float64)
        return {key: BoundaryInputSpec(shape=(self._n,), dtype=jnp.dtype("float64"), default=zero)
                for key in ("u", "a", "b")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        x = boundary_inputs["b"] + boundary_inputs["a"] * boundary_inputs["u"]
        return {"x": x.astype(jnp.float64)}

    def update_evaluations(self):
        return 1


def _mappings():
    spread = sparse_nearest_neighbor_mapping((np.arange(M) + 0.5) / M, (np.arange(N) + 0.5) / N)
    mean = sparse_matrix_mapping(np.arange(N).reshape(M, N // M),
                                 np.full((M, N // M), M / N, np.float64), n_source=N)
    return spread, mean


def graph(norm: str, schedule: str, *, on_ab=None, on_ba=None) -> GraphManager:
    """The pair, not compiled; *on_ab* / *on_ba* are the edges' transforms."""
    spread, mean = _mappings()
    gm = GraphManager()
    gm.add_node(Relay("A", M))
    gm.add_node(Relay("B", N))
    for name, size in (("A", M), ("B", N)):
        for key in ("a", "b"):
            gm.add_external_input(name, key, shape=(size,), dtype=jnp.dtype("float64"))
    gm.add_edge("A", "B", "x", "u", mapping=spread,
                **({} if on_ab is None else {"transform": on_ab}))
    gm.add_edge("B", "A", "x", "u", mapping=mean,
                **({} if on_ba is None else {"transform": on_ba}))
    tolerance = ({"tolerance": RTOL * math.sqrt(M + N)} if norm == "l2" else {"rtol": RTOL})
    gm.add_coupling_group(["A", "B"], convergence_norm=norm, iteration_mode=schedule,
                          max_iterations=6000, solver="ift", diagnostics=True, **tolerance)
    return gm


def _constants():
    rng = np.random.default_rng(3)
    return (np.full(M, GAIN), rng.uniform(1.0, 2.0, M), np.ones(N), rng.uniform(0.1, 0.2, N))


def solve(gm: GraphManager) -> tuple:
    """One step from zero; ``(report, distance of the returned state from
    the float64 fixed point in the group's own units)``.  The distance is
    over the two fields, each over its own magnitude: the mean square in
    tolerances under "mixed" and "interface" (whose two readings are the
    field ``A.x`` and the mean of ``B.x``, each as far off as its field),
    the root of the sum under "l2"."""
    a_a, b_a, a_b, b_b = _constants()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.compile()
    gm.step(external_inputs={"A": {"a": jnp.asarray(a_a), "b": jnp.asarray(b_a)},
                             "B": {"a": jnp.asarray(a_b), "b": jnp.asarray(b_b)}})
    report = dict(gm.coupling_diagnostics()[KEY])
    x_a = np.asarray(gm.get_node_state("A")["x"], np.float64)
    x_b = np.asarray(gm.get_node_state("B")["x"], np.float64)
    # A = bA + 0.9 mean(bB + A)  =>  A = (bA + 0.9 mean(bB)) / (1 - 0.9)
    fixed_a = (b_a + GAIN * np.mean(b_b)) / (1.0 - GAIN)
    fixed_b = b_b + fixed_a[0]
    norm = gm._coupling_groups[0].convergence_norm  # noqa: SLF001
    if norm == "interface":
        parts = [(x_a, fixed_a), (np.array([np.mean(x_b)]), np.array([np.mean(fixed_b)]))]
    else:
        parts = [(x_a, fixed_a), (x_b, fixed_b)]
    squares = [((value - want) / np.max(np.abs(value))) ** 2 for value, want in parts]
    total = float(sum(np.sum(sq) for sq in squares))
    if norm == "l2":
        return report, math.sqrt(total)
    return report, math.sqrt(total / sum(sq.size for sq in squares)) / RTOL


def _check_stalled_on_the_narrow_edge(norm: str, report: dict, distance: float) -> None:
    """The fixture stalls a float32 rounding from its fixed point, and the
    report says so: no flag beside a bound under the distance."""
    tolerances = distance / (RTOL * math.sqrt(M + N)) if norm == "l2" else distance
    assert report["converged"] and tolerances > 1e4, (report, distance)
    # The floor is counted at float32, far above the residual of the stall
    # (its size is pinned by the test of ``residual_precision_floor`` below).
    assert report["precision_limited"] is True, report
    if report["spectral_usable"]:
        assert report["spectral_error_bound"] >= distance, (report, distance)
    else:
        assert report.get("not_usable_reason"), report
    # The pair's rows are within the mapped-row limit, so the flag is set
    # and the assertion above is not vacuous.
    assert report["spectral_usable"] is True, report


CASES = {"T": {"on_ab": narrow}, "T2": {"on_ba": narrow}}


@pytest.fixture
def step_eps(monkeypatch):
    """The eps the STEP hands its own spectral analysis as the group's
    coarsest, one entry per trace: the rule has two readers (the report's
    floor and the step), and the report's numbers alone do not tell
    whether the step was handed the edges."""
    seen = []
    real = _coupled_block._spectral_rate_at

    def spy(*args, **kwargs):
        seen.append(kwargs["map_eps"])
        return real(*args, **kwargs)

    monkeypatch.setattr(_coupled_block, "_spectral_rate_at", spy)
    return seen


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_pair_stalled_on_a_narrowing_transform_has_no_flag_on_a_bound_under_the_distance(
        case, step_eps):
    """Per push: the mixed norm under Gauss-Seidel, both edges."""
    with x64(True):
        report, distance = solve(graph("mixed", "gauss-seidel", **CASES[case]))
    _check_stalled_on_the_narrow_edge("mixed", report, distance)
    assert step_eps, "the patch did not reach the step's analysis"
    assert set(step_eps) == {EPS32}, step_eps


# Per push: tests/core/test_the_float_floor_counts_what_an_edge_delivers.py::test_a_pair_stalled_on_a_narrowing_transform_has_no_flag_on_a_bound_under_the_distance
@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("norm", NORMS)
def test_a_narrowing_transform_is_counted_under_every_norm_and_schedule(
        norm, schedule, case, step_eps):
    with x64(True):
        report, distance = solve(graph(norm, schedule, **CASES[case]))
    _check_stalled_on_the_narrow_edge(norm, report, distance)
    assert step_eps and set(step_eps) == {EPS32}, step_eps


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------

def _state() -> dict:
    return {"A": {"x": jnp.zeros(M, jnp.float64)}, "B": {"x": jnp.zeros(N, jnp.float64)}}


def _edges(**transforms) -> list:
    return list(graph("mixed", "gauss-seidel", **transforms)._edges)  # noqa: SLF001


def test_the_coarsest_eps_counts_a_floating_delivery_and_nothing_else():
    with x64(True):
        state = _state()
        plain = _edges()
        assert _group_coarsest_eps(state, ["A", "B"]) == EPS64
        assert _group_coarsest_eps(state, ["A", "B"], plain) == EPS64
        for transforms in ({"on_ab": narrow}, {"on_ba": narrow}):
            assert _group_coarsest_eps(state, ["A", "B"], _edges(**transforms)) == EPS32
        # A delivery that widens, keeps, is not floating, or has no entries
        # leaves the fields' own.
        for other in (keep, lambda v: v.astype(jnp.int32), lambda v: v[:0].astype(jnp.float32)):
            assert _group_coarsest_eps(state, ["A", "B"], _edges(on_ab=other)) == EPS64
        half = _group_coarsest_eps(state, ["A", "B"],
                                   _edges(on_ab=lambda v: v.astype(jnp.float16)))
        assert half == float(np.finfo(np.float16).eps)


def test_a_coarser_field_still_sets_it_and_a_delivery_never_lowers_it():
    """Monotone: the larger of the fields' and the deliveries'."""
    with x64(True):
        state = {"A": {"x": jnp.zeros(M, jnp.float32)}, "B": {"x": jnp.zeros(N, jnp.float64)}}
        widen = _edges(on_ab=lambda v: v.astype(jnp.float64))
        assert _group_coarsest_eps(state, ["A", "B"], widen) == EPS32
        assert _group_coarsest_eps(state, ["A", "B"]) == EPS32


@pytest.mark.parametrize("norm", NORMS)
def test_the_reports_floor_is_counted_at_what_the_edge_delivers(norm):
    """``residual_precision_floor`` with the narrowing edge is the floor
    of the same state held in float32 behind plain transforms, and
    ``eps32 / eps64`` times the float64 floor without it."""
    with x64(True):
        state = {"A": {"x": jnp.ones(M, jnp.float64)}, "B": {"x": jnp.ones(N, jnp.float64)}}
        fine = float(residual_precision_floor(state, ["A", "B"], norm, 0.0, 1e-6, _edges()))
        coarse = float(residual_precision_floor(state, ["A", "B"], norm, 0.0, 1e-6,
                                                _edges(on_ab=narrow)))
    assert fine > 0 and coarse == pytest.approx(fine * EPS32 / EPS64, rel=1e-12)


def test_an_edge_that_keeps_its_dtype_changes_nothing_to_the_bit(monkeypatch, step_eps):
    """A float64 pair whose transform keeps float64: every number of the
    report is the one it is with the deliveries not looked at, and the
    step's own analysis is handed the float64 eps."""
    def numbers():
        with x64(True):
            report, _distance = solve(graph("mixed", "gauss-seidel", on_ab=keep))
        return {key: value for key, value in report.items()
                if isinstance(value, (bool, int, float))}

    counted = numbers()
    assert step_eps and set(step_eps) == {EPS64}, step_eps
    asked = []

    def nothing(self, state, mappings=None):
        asked.append(self.key)
        return ()

    monkeypatch.setattr(_interface_plan.InterfaceEdge, "delivered_leaves", nothing)
    uncounted = numbers()
    assert asked, "the patch did not reach the rule"
    assert counted == uncounted and counted["precision_limited"] in (True, False)
    assert set(counted) >= {"residual", "precision_limited", "spectral_error_bound",
                            "spectral_usable", "gradient_bound_usable"}


def test_the_patch_that_ignores_deliveries_brings_the_defect_back(monkeypatch):
    """The control of the test above and of the fix: with the deliveries
    not looked at, the narrowing pair reads a flag on a bound far under
    its distance."""
    monkeypatch.setattr(_interface_plan.InterfaceEdge, "delivered_leaves",
                        lambda self, state, mappings=None: ())
    with x64(True):
        report, distance = solve(graph("mixed", "gauss-seidel", on_ab=narrow))
    assert report["spectral_usable"] and report["spectral_error_bound"] < 1e-3 * distance
