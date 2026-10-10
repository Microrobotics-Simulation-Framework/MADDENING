"""A group's float floor is counted at its coarsest member's resolution.

A field a node computes from a coarser neighbour's output carries that
neighbour's rounding: a float64 field of ``n`` values that is an exact
function of one float32 value is as far from its fixed point, relative to
its own size, as the float32 value is.  The floor took each field at its
own dtype's ``eps``, so those ``n`` entries put nothing into it, and the
pooled floor fell as ``1 / sqrt(1 + n)`` while the distance did not: the
pair below, stalled 500 float32 ulps short of its fixed point, read
``spectral_error_bound`` at 0.92 of the distance at ``n = 1e3`` and 0.21
at ``n = 2e4`` with ``spectral_usable`` and ``gradient_bound_usable`` set.

The rule now: every entry of a group's floor is counted at the ``eps`` of
the coarsest floating dtype among the group's fields
(``acceleration._group_coarsest_eps``), in the report's floor
(``residual_precision_floor``) and in the step's own analysis alike.  It
only raises a floor, and in a group of one dtype it is the number it was.

**The pair** (every node an affine relay that declares one evaluation,
no cancellation, every constant exact in the dtype that holds it)::

    p (coarse, one value)      p.x    <- 1 + u_p[0]
    q (fine, n values)         q.x[j] <- c_j + 2 g u_q[0]
    p.x -> q.u   a static mapping 1 -> 3 (weights 0.5, 0.25, 0.25)
    q.x -> p.u   a static sparse mapping n -> n + 1 (the identity, one empty row)

so ``p.x = 1 + c_0 + g p.x``: a loop of gain ``g`` whose fixed point is a
closed form of the constants as each dtype holds them.  Both mappings
deliver more entries than they read, so the interface norm reads both
edges at their source: the same entries ``"mixed"`` and ``"l2"`` read.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import (
    PRECISION_FLOOR_ULPS,
    _group_coarsest_eps,
    residual_precision_floor,
)
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.coupling.sparse_mapping import StaticSparseMapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.sparse_mapping_support import x64

KEY = "p+q"


class P(SimulationNode):
    """One value: ``x <- 1 + u[0]``."""

    def __init__(self, name, timestep, port, dtype):
        super().__init__(name, timestep)
        self._port, self._dt = port, jnp.dtype(dtype)

    def initial_state(self):
        return {"x": jnp.full((1,), 3.0, self._dt)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._port,), dtype=self._dt,
                                       default=jnp.zeros(self._port, self._dt))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": (1.0 + boundary_inputs["u"][:1]).astype(self._dt)}

    def update_evaluations(self):
        return 1


class Q(SimulationNode):
    """``n`` values: ``x[j] <- c_j + two_g * u[0]``."""

    def __init__(self, name, timestep, c, two_g, dtype):
        super().__init__(name, timestep)
        self._c, self._two_g, self._dt = np.asarray(c), two_g, jnp.dtype(dtype)

    def initial_state(self):
        return {"x": jnp.asarray(self._c + 3.0, self._dt)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(3,), dtype=self._dt,
                                       default=jnp.zeros(3, self._dt))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": (jnp.asarray(self._c, self._dt)
                      + jnp.asarray(self._two_g, self._dt) * boundary_inputs["u"][0]
                      ).astype(self._dt)}

    def update_evaluations(self):
        return 1


@dataclass(frozen=True)
class Cell:
    """One run: the two dtypes, the fine field's size, the loop's gain as
    ``two_g / 2``, the tolerance, the schedule and the norm."""

    p_dtype: str
    q_dtype: str
    n: int
    two_g: float
    rtol: float
    schedule: str
    norm: str

    @property
    def id(self) -> str:
        return f"{self.p_dtype}-{self.q_dtype}-n{self.n}-{self.schedule}-{self.norm}"

    @property
    def needs_x64(self) -> bool:
        return "float64" in (self.p_dtype, self.q_dtype)


def _constants(n: int) -> np.ndarray:
    """``c_0 = 0.5`` and the rest in ``[0.25, 0.75]``: exact in every dtype."""
    c = 0.25 + np.arange(n) % 5 * 0.125
    c[0] = 0.5
    return c


def build(cell: Cell, *, diagnostics: bool = True, tie: bool = False) -> GraphManager:
    """The pair of *cell*.  With *tie*, ``q.x`` reaches ``p`` through an
    ``n x n`` identity instead: a tie, which the interface norm reads as
    delivered, through the mapping."""
    c = _constants(cell.n)
    gm = GraphManager()
    gm.add_node(P("p", 1.0, cell.n if tie else cell.n + 1, cell.p_dtype))
    gm.add_node(Q("q", 1.0, c, cell.two_g, cell.q_dtype))
    gm.add_edge("p", "q", "x", "u", mapping=matrix_mapping(
        jnp.asarray([[0.5], [0.25], [0.25]], cell.q_dtype)))
    back = (matrix_mapping(jnp.eye(cell.n, dtype=cell.q_dtype)) if tie else StaticSparseMapping(
        np.arange(cell.n)[:, None], jnp.ones((cell.n, 1), cell.q_dtype), n_source=cell.n,
        n_target=cell.n + 1, layout="scatter"))
    gm.add_edge("q", "p", "x", "u", mapping=back)
    # "l2" is the root of the sum over the entries, not their RMS: the
    # same demand of each entry is ``rtol * sqrt(entries)`` there.
    tolerance = ({"tolerance": cell.rtol * math.sqrt(cell.n + 1)} if cell.norm == "l2"
                 else {"rtol": cell.rtol})
    gm.add_coupling_group(["p", "q"], convergence_norm=cell.norm, max_iterations=40000,
                          iteration_mode=cell.schedule, solver="ift", diagnostics=diagnostics,
                          **tolerance)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


def measure(cell: Cell) -> dict:
    """The report of one step of *cell*'s pair and its true distance, in
    the units the report's numbers are in (run inside ``x64`` as the cell
    needs)."""
    gm = build(cell)
    gm.step()
    report = dict(gm.coupling_diagnostics()[KEY])
    xp = np.asarray(gm.get_node_state("p")["x"], np.float64)
    xq = np.asarray(gm.get_node_state("q")["x"], np.float64)
    c = _constants(cell.n)
    # The loop's gain with the constant as q's dtype holds it; the fixed
    # point of the exact map of those constants.
    g = 0.5 * float(np.asarray(cell.two_g, cell.q_dtype))
    p_star = (1.0 + c[0]) / (1.0 - g)
    q_star = c + g * p_star
    terms = np.concatenate([np.abs(xp - p_star) / np.max(np.abs(xp)),
                            np.abs(xq - q_star) / np.max(np.abs(xq))])
    # Each field over its own magnitude: the RMS in tolerances under
    # "mixed" and "interface", the root of the sum under "l2".
    distance = (float(np.sqrt(np.sum(terms ** 2))) if cell.norm == "l2"
                else float(np.sqrt(np.mean(terms ** 2))) / cell.rtol)
    report["distance"] = distance
    report["p_relative_error"] = float(abs(xp[0] - p_star) / p_star)
    return report


def _cells(pairs, sizes, schedules=("gauss-seidel", "jacobi"),
           norms=("interface", "mixed", "l2")) -> list:
    return [Cell(p, q, n, two_g, rtol, schedule, norm)
            for p, q, two_g, rtol in pairs for n in sizes
            for schedule in schedules for norm in norms]


#: ``(coarse dtype, fine dtype, 2 g, rtol)``.  A float32 value beside
#: float64 ones stalls up to 500 of its ulps short at a gain of 0.999; a
#: 16-bit value beside float32 ones is given a gain of 0.875 (it stalls a
#: few of its much larger ulps short) and a tolerance it can be asked.
FLOAT32_BESIDE_FLOAT64 = ("float32", "float64", 1.998, 1e-7)
SIXTEEN_BIT_BESIDE_FLOAT32 = (("float16", "float32", 1.75, 1e-4),
                              ("bfloat16", "float32", 1.75, 1e-3))

#: One cell per push: the audited worst case (0.21 of the distance).
PER_PUSH = Cell("float32", "float64", 20000, 1.998, 1e-7, "gauss-seidel", "interface")
TABLE = _cells([FLOAT32_BESIDE_FLOAT64, *SIXTEEN_BIT_BESIDE_FLOAT32], (2, 1000, 20000))


def _holds(cell: Cell, report: dict) -> None:
    """A flag that is set stands on a bound at or above the distance; the
    pair is at its float floor (the premise: otherwise the floor is not
    what carries the bound)."""
    assert report["converged"], (cell.id, report)
    assert report["precision_limited"], (cell.id, report)
    assert report["p_relative_error"] > 0.0, (cell.id, "fixture premise: stalled short")
    if report["spectral_usable"]:
        assert report["spectral_error_bound"] >= report["distance"], (cell.id, report)


def test_a_float64_field_downstream_of_a_float32_member_is_floored_at_float32():
    """The audited case: one float32 value, 2e4 float64 values computed
    from it, Gauss-Seidel, the interface norm.  The bound holds with its
    flag set (it read 0.21 of the distance), and the floor is the float32
    one whatever the fine field's size."""
    with x64(True):
        report = measure(PER_PUSH)
    _holds(PER_PUSH, report)
    assert report["spectral_usable"] and report["gradient_bound_usable"], report
    # 500 float32 ulps short: 4e-5 of its size, some 400 tolerances.
    assert report["distance"] > 100.0, report


# Per push: tests/core/test_coupling_floor_at_the_groups_coarsest_dtype.py::test_a_float64_field_downstream_of_a_float32_member_is_floored_at_float32
@pytest.mark.slow
@pytest.mark.parametrize("cell", TABLE, ids=lambda cell: cell.id)
def test_a_set_flag_stands_on_a_bound_at_or_above_the_distance_in_a_group_of_two_dtypes(cell):
    """Every size, both schedules, the three norms, and the 16-bit
    siblings (a float16 or bfloat16 value beside float32 ones)."""
    with x64(cell.needs_x64):
        report = measure(cell)
    _holds(cell, report)


def test_the_tables_gauss_seidel_cells_keep_their_flags():
    """The table is not vacuous: its per-push cell is one of its rows, and
    that row's flags are asserted set by the per-push test."""
    assert PER_PUSH in TABLE
    assert {(c.p_dtype, c.q_dtype) for c in TABLE} == {
        ("float32", "float64"), ("float16", "float32"), ("bfloat16", "float32")}


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------

def _state(**fields) -> dict:
    return {"a": {"x": fields["a"]}, "b": {"y": fields["b"], "count": jnp.int32(3),
                                           "empty": jnp.zeros((0,), jnp.float16)}}


def test_the_groups_eps_is_its_coarsest_floating_field_with_entries():
    """Over the members' floating fields that hold entries: an integer
    field and a float16 field with no entries do not set it; no member
    named, or none floating, gives ``None``."""
    eps32, eps16 = (float(np.finfo(t).eps) for t in (np.float32, np.float16))
    state = _state(a=jnp.ones(3, jnp.float32), b=jnp.ones(5, jnp.float16))
    assert _group_coarsest_eps(state, ["a", "b"]) == eps16
    assert _group_coarsest_eps(state, ["a"]) == eps32
    assert _group_coarsest_eps(state, []) is None
    assert _group_coarsest_eps({"n": {"k": jnp.int32(1)}}, ["n"]) is None
    with x64(True):
        wide = _state(a=jnp.ones(3, jnp.float64), b=jnp.ones(5, jnp.float32))
        assert _group_coarsest_eps(wide, ["a", "b"]) == eps32
        assert _group_coarsest_eps(wide, ["a"]) == float(np.finfo(np.float64).eps)


@pytest.mark.parametrize("norm", ["l2", "mixed", "interface"])
def test_every_entry_of_the_floor_is_counted_at_the_groups_coarsest_eps(norm):
    """Closed form: 3 float64 entries beside 5 float32 ones are 8 entries
    at float32's eps under every norm; under ``"interface"`` what a plain
    edge delivers from the float64 field is counted so too."""
    eps32 = float(np.finfo(np.float32).eps)
    with x64(True):
        state = _state(a=jnp.ones(3, jnp.float64), b=jnp.ones(5, jnp.float32))
        if norm == "interface":
            from maddening.core.edge import EdgeSpec  # noqa: PLC0415
            edges = [EdgeSpec("a", "b", "x", "u"), EdgeSpec("b", "a", "y", "u")]
            floor = residual_precision_floor(state, ["a", "b"], norm, rtol=1e-3,
                                             interface_edges=edges)
            alone = residual_precision_floor(state, [], norm, rtol=1e-3, interface_edges=edges)
        else:
            floor = residual_precision_floor(state, ["a", "b"], norm, rtol=1e-3)
            alone = None
    expected = (PRECISION_FLOOR_ULPS * eps32 * math.sqrt(8) if norm == "l2"
                else PRECISION_FLOOR_ULPS * eps32 / 1e-3)
    assert float(floor) == pytest.approx(expected, rel=1e-12)
    if alone is not None:
        # No member named: each reading at its own dtype's eps and its
        # source's, the rule a reading has outside a group.
        eps64 = float(np.finfo(np.float64).eps)
        own = PRECISION_FLOOR_ULPS * math.sqrt((3 * eps64 ** 2 + 5 * eps32 ** 2) / 8) / 1e-3
        assert float(alone) == pytest.approx(own, rel=1e-12)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_a_group_of_one_dtype_keeps_the_floor_it_had(dtype):
    """One dtype: the coarsest eps is each field's own, and the floor is
    ``4 eps sqrt(n)`` and ``4 eps / rtol`` as it always was."""
    with x64(dtype == "float64"):
        t = jnp.dtype(dtype)
        eps = float(jnp.finfo(t).eps)
        state = {"a": {"x": jnp.ones(3, t)}, "b": {"y": jnp.ones(5, t)}}
        assert _group_coarsest_eps(state, ["a", "b"]) == eps
        l2 = float(residual_precision_floor(state, ["a", "b"], "l2"))
        mixed = float(residual_precision_floor(state, ["a", "b"], "mixed", rtol=1e-2))
    assert l2 == pytest.approx(PRECISION_FLOOR_ULPS * eps * math.sqrt(8), rel=1e-6)
    assert mixed == pytest.approx(PRECISION_FLOOR_ULPS * eps / 1e-2, rel=1e-6)


def test_the_floor_never_falls_below_the_one_each_field_had_alone():
    """The rule only raises: against the floor with every field at its own
    eps (the old rule, restated here), over the dtype pairs."""
    cases = [("float16", "float32"), ("bfloat16", "float32"), ("float32", "float16"),
             ("float32", "float64"), ("float64", "float16")]
    for a, b in cases:
        with x64("float64" in (a, b)):
            state = {"a": {"x": jnp.ones(3, jnp.dtype(a))}, "b": {"y": jnp.ones(7, jnp.dtype(b))}}
            new = float(residual_precision_floor(state, ["a", "b"], "mixed", rtol=1e-2))
            ea, eb = (float(jnp.finfo(jnp.dtype(t)).eps) for t in (a, b))
        old = PRECISION_FLOOR_ULPS * math.sqrt((3 * ea ** 2 + 7 * eb ** 2) / 10) / 1e-2
        assert new >= old * (1 - 1e-6), (a, b)
        assert new == pytest.approx(PRECISION_FLOOR_ULPS * max(ea, eb) / 1e-2, rel=1e-6), (a, b)


def _eager_step(gm: GraphManager):
    """One step of *gm* evaluated eagerly (the raw step function, not its
    compiled form), so what the step hands its analysis is concrete."""
    return gm._raw_step_fn(gm._state, gm._resolve_external_inputs(None), None)  # noqa: SLF001


@pytest.mark.parametrize("norm", ["mixed", "interface"])
def test_the_steps_analysis_counts_every_entry_at_the_groups_coarsest_eps(norm, monkeypatch):
    """The per-entry resolution the step hands the spectral analysis and
    the gradient bound: one float32 entry and six float64 ones, all seven
    at float32's eps (they were 4 eps32 and six times 4 eps64), and the
    products' rounding the same eps.  Read where the step passes it
    (``_coupled_block._spectral_rate_at``, the module that reads it)."""
    from maddening.core.coupling import _coupled_block  # noqa: PLC0415

    seen = []
    real = _coupled_block._spectral_rate_at

    def recorder(*args, **kwargs):
        seen.append((np.asarray(kwargs["resolution"]), kwargs["map_eps"]))
        return real(*args, **kwargs)

    monkeypatch.setattr(_coupled_block, "_spectral_rate_at", recorder)
    eps32 = float(np.finfo(np.float32).eps)
    with x64(True):
        _eager_step(build(Cell("float32", "float64", 6, 1.998, 1e-4, "jacobi", norm)))
    assert len(seen) == 1, "the patch takes effect: the step's one spectral analysis"
    resolution, map_eps = seen[0]
    assert map_eps == eps32
    assert resolution.shape == (7,)
    # One evaluation per pass under Jacobi, every weight's common scale 1.
    assert np.all(resolution == PRECISION_FLOOR_ULPS * eps32), resolution


def test_the_analysis_of_a_reading_counts_what_an_edge_delivers_at_the_groups_coarsest_eps(
        monkeypatch):
    """Under the interface norm with an edge read as delivered (a tie),
    the analysis is taken on the reading, and its resolution is the
    reading's: the float32 value read at its source and the six float64
    values the tie delivers, all at float32's eps."""
    from maddening.core.coupling import _coupled_block  # noqa: PLC0415

    seen = []
    real = _coupled_block._interface_spectral_rate_at

    def recorder(*args, **kwargs):
        seen.append(np.asarray(args[7]))
        return real(*args, **kwargs)

    monkeypatch.setattr(_coupled_block, "_interface_spectral_rate_at", recorder)
    eps32 = float(np.finfo(np.float32).eps)
    with x64(True):
        _eager_step(build(Cell("float32", "float64", 6, 1.998, 1e-4, "jacobi", "interface"),
                          tie=True))
    assert len(seen) == 1, "the patch takes effect: the analysis on the reading"
    assert seen[0].shape == (7,)
    assert np.all(seen[0] == PRECISION_FLOOR_ULPS * eps32), seen[0]
