"""The float floor does not count the rounding of a long scatter row; the flag does.

A sparse mapping in the scatter layout adds the entries of a target's row
up one after another.  An in-order sum of ``k`` terms of one sign rounds
by up to ``(k - 1) / 2`` units of ``eps`` of the sum, and where the terms
are nearly equal the rounding is systematic: it grows like ``k``.  The
float floor of a coupling report counts a fixed number of ulps per
evaluation, so a group stalled behind such a row is further from its
fixed point than ``spectral_error_bound`` says (MADD-ANO-257, open: the
floor is not fixed in 0.4.0).

What 0.4.0 does: the report withdraws ``spectral_usable`` and
``gradient_bound_usable``, with a ``not_usable_reason``, where an
internal edge carries a static sparse mapping in the scatter layout with
a row longer than ``SCATTER_ROW_FLOOR_LIMIT`` entries **and** the
residual does not stand clear of the floor that row could give it
(``floor * row``).  It only withdraws: every number of the report is the
one it was.  The gather layout and the dense kinds are not counted (XLA
chooses the order of their sums); the registry entry has what was
measured on them.

**The pair** (float32 unless a cell says otherwise; every weight
positive, so nothing cancels; both nodes are relays that declare one
evaluation; the constants arrive as external inputs, so one compiled
pair serves every field and gain)::

    coarse (ONE value)     x    <- b + a * u        u = what the fine field sums to
    fine   (n values)      x[j] <- c_j + u[j]       u[j] = the coarse value
    coarse.x -> fine.u     sparse nearest neighbour, 1 -> n
    fine.x   -> coarse.u   the conservative nearest neighbour, n -> 1: one row of n
                           entries, in the scatter layout, the gather layout or dense

with loop gain ``a * n``.  Its fixed point is a closed form of the
constants as the dtype holds them: ``x* = (b + a sum(c)) / (1 - a n)``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _group_layout
from maddening.core.coupling._group_layout import (
    SCATTER_ROW_FLOOR_LIMIT,
    _longest_scatter_row,
    _scatter_row_reason,
    _scatter_rows,
)
from maddening.core.coupling.mapping import matrix_mapping, nearest_neighbor_mapping
from maddening.core.coupling.sparse_mapping import (
    StaticSparseMapping,
    sparse_nearest_neighbor_mapping,
)
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.sparse_mapping_support import x64

KEY = "coarse+fine"
ROW_EDGE = "fine.x->coarse.u"
ANOMALY = "MADD-ANO-257"


class Coarse(SimulationNode):
    """One value: ``x <- b + a * u``, with ``(a, b)`` an external input."""

    def __init__(self, name, timestep, dtype):
        super().__init__(name, timestep)
        self._dt = jnp.dtype(dtype)

    def initial_state(self):
        return {"x": jnp.zeros(1, self._dt)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(1,), dtype=self._dt, default=jnp.zeros(1, self._dt)),
                "ab": BoundaryInputSpec(shape=(2,), dtype=self._dt,
                                        default=jnp.zeros(2, self._dt))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        ab = boundary_inputs["ab"]
        return {"x": ab[1] + ab[0] * boundary_inputs["u"]}

    def update_evaluations(self):
        return 1


class Fine(SimulationNode):
    """``n`` values: ``x[j] <- c_j + u[j]``, with ``c`` an external input."""

    def __init__(self, name, timestep, n, dtype):
        super().__init__(name, timestep)
        self._n, self._dt = n, jnp.dtype(dtype)

    def initial_state(self):
        return {"x": jnp.zeros(self._n, self._dt)}

    def boundary_input_spec(self):
        zeros = jnp.zeros(self._n, self._dt)
        return {"u": BoundaryInputSpec(shape=(self._n,), dtype=self._dt, default=zeros),
                "c": BoundaryInputSpec(shape=(self._n,), dtype=self._dt, default=zeros)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": boundary_inputs["c"] + boundary_inputs["u"]}

    def update_evaluations(self):
        return 1


@dataclass(frozen=True)
class Pair:
    """What a compiled pair is built from: the row's layout and length,
    the schedule, the norm, the dtype and the tolerance (``None``: below
    the dtype's float floor, so the pair runs until it stalls)."""

    layout: str
    n: int
    schedule: str = "jacobi"
    norm: str = "interface"
    dtype: str = "float32"
    rtol: float | None = None

    @property
    def tolerance(self) -> float:
        if self.rtol is not None:
            return self.rtol
        return 1e-16 if self.dtype == "float64" else 1e-7

    @property
    def id(self) -> str:
        rtol = "" if self.rtol is None else f"-rtol{self.rtol:g}"
        return f"{self.layout}-n{self.n}-{self.schedule}-{self.norm}-{self.dtype}{rtol}"


@dataclass(frozen=True)
class Cell:
    """One run of a pair: the field the fine node adds and the loop gain."""

    pair: Pair
    field: str = "uniform"
    gain: float = 0.99

    @property
    def id(self) -> str:
        return f"{self.pair.id}-{self.field}-g{self.gain:g}"


def constants(n: int, field: str, dtype) -> np.ndarray:
    """The fine node's ``c``: ``"uniform"`` (every entry 1.1), ``"ramp"``
    (``1 + j / n``) or ``"random"`` (uniform in ``[1, 2)``, seeded)."""
    if field == "uniform":
        return np.full(n, 1.1, dtype)
    if field == "ramp":
        return (1.0 + np.arange(n) / n).astype(dtype)
    assert field == "random", field
    return np.random.default_rng(1).uniform(1.0, 2.0, n).astype(dtype)


def _row_mapping(layout: str, n: int):
    fine_points, coarse_point = (np.arange(n) + 0.5) / n, np.array([0.5])
    if layout == "dense":
        return nearest_neighbor_mapping(fine_points, coarse_point, mode="conservative")
    return sparse_nearest_neighbor_mapping(
        fine_points, coarse_point, mode="conservative", transpose=layout)


_COMPILED: dict = {}


def compiled(pair: Pair) -> GraphManager:
    """The pair, compiled once per module (run inside ``x64`` as its dtype needs)."""
    if pair not in _COMPILED:
        n, dt = pair.n, jnp.dtype(pair.dtype)
        gm = GraphManager()
        gm.add_node(Coarse("coarse", 1.0, dt))
        gm.add_node(Fine("fine", 1.0, n, dt))
        gm.add_external_input("coarse", "ab", shape=(2,), dtype=dt)
        gm.add_external_input("fine", "c", shape=(n,), dtype=dt)
        gm.add_edge("coarse", "fine", "x", "u", mapping=sparse_nearest_neighbor_mapping(
            np.array([0.5]), (np.arange(n) + 0.5) / n))
        gm.add_edge("fine", "coarse", "x", "u", mapping=_row_mapping(pair.layout, n))
        # "l2" is the root of the sum over the entries, not their RMS: the
        # same demand of each entry is ``rtol * sqrt(entries)`` there.
        tolerance = ({"tolerance": pair.tolerance * math.sqrt(n + 1)} if pair.norm == "l2"
                     else {"rtol": pair.tolerance})
        gm.add_coupling_group(["coarse", "fine"], convergence_norm=pair.norm,
                              max_iterations=40000, iteration_mode=pair.schedule,
                              solver="ift", diagnostics=True, **tolerance)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.compile()
        _COMPILED[pair] = gm
    return _COMPILED[pair]


def measure(cell: Cell) -> dict:
    """The report of one step of *cell* from a reset pair, and its true
    distance in the units the report's numbers are in.  The graph is left
    as the step left it (``report["graph"]``)."""
    pair = cell.pair
    with x64(pair.dtype == "float64"):
        gm = compiled(pair)
        np_dtype = np.dtype(pair.dtype).type
        c = constants(pair.n, cell.field, np_dtype)
        a, b = np_dtype(cell.gain / pair.n), np_dtype(1.0)
        gm.reset_state()
        gm.set_node_state("fine", {"x": jnp.asarray(c)})
        gm.step(external_inputs={"coarse": {"ab": jnp.asarray([a, b])},
                                 "fine": {"c": jnp.asarray(c)}})
        report = dict(gm.coupling_diagnostics()[KEY])
        x = float(np.asarray(gm.get_node_state("coarse")["x"])[0])
        fine = np.asarray(gm.get_node_state("fine")["x"], np.float64)
    # The fixed point of the exact map of the constants as the dtype holds them.
    c64 = np.asarray(c, np.float64)
    sum_c = math.fsum(c64.tolist())
    x_star = (float(b) + float(a) * sum_c) / (1.0 - float(a) * pair.n)
    coarse_error = abs(x - x_star) / abs(x)
    if pair.norm == "interface":
        # Two readings: the coarse value at its source, and what the row
        # delivers (the exact sum of the returned field).
        delivered = math.fsum(fine.tolist())
        delivered_error = abs(delivered - (sum_c + pair.n * x_star)) / abs(delivered)
        distance = math.sqrt((coarse_error ** 2 + delivered_error ** 2) / 2.0) / pair.tolerance
    else:
        terms = np.concatenate([[coarse_error],
                                np.abs(fine - (c64 + x_star)) / np.max(np.abs(fine))])
        distance = (float(np.sqrt(np.sum(terms ** 2))) if pair.norm == "l2"
                    else float(np.sqrt(np.mean(terms ** 2))) / pair.tolerance)
    report["distance"] = distance
    report["graph"] = gm
    return report


def _a_set_flag_stands_on_a_bound_at_or_above_the_distance(cell: Cell, report: dict) -> None:
    assert report["converged"], (cell.id, report)
    assert report["distance"] > 0.0, (cell.id, "fixture premise: stalled short")
    if report["spectral_usable"]:
        assert report["spectral_error_bound"] >= report["distance"], (cell.id, report)
    if report["gradient_bound_usable"]:
        assert report["spectral_usable"], (cell.id, report)


def _withdrawn_for_the_row(cell: Cell, report: dict) -> None:
    assert not report["spectral_usable"] and not report["gradient_bound_usable"], (
        cell.id, report)
    reason = report["not_usable_reason"]
    for said in (ROW_EDGE, f"adds up {cell.pair.n} entries",
                 f"the limit is {SCATTER_ROW_FLOOR_LIMIT}", ANOMALY, "a wider dtype"):
        assert said in reason, (cell.id, said, reason)


# ---------------------------------------------------------------------------
# Per push: the audited mechanism at one row length, and its controls
# ---------------------------------------------------------------------------

#: A uniform field behind a row of 300 entries, stalled at float32: the
#: report read its bound at 0.24 of the distance with both flags set.
STALLED = Cell(Pair("scatter", 300))
#: The same operator in the gather layout, which XLA reduces in another
#: order, behind the same uniform field and behind a ramp.
GATHERED = Cell(Pair("gather", 300))
GATHERED_RAMP = Cell(Pair("gather", 300), "ramp")
#: The same pair in float64 at the float32 tolerance: the verified way
#: out, and a residual that stands clear of the floor.
WIDER = Cell(Pair("scatter", 300, dtype="float64", rtol=1e-7))


@pytest.fixture(scope="module")
def stalled() -> dict:
    return measure(STALLED)


def test_a_stalled_group_behind_a_long_scatter_row_withdraws_its_flags_and_says_why(stalled):
    """At its float floor (``precision_limited``), behind a row of 300
    entries in the scatter layout: both flags withdrawn, and the reason
    names the edge, the row's length, the limit and the two ways out.
    The premise is asserted with it: this report's bound is below the
    distance, so the flags it would have set were wrong."""
    assert stalled["converged"] and stalled["precision_limited"], stalled
    assert math.isfinite(stalled["spectral_error_bound"]), stalled
    assert stalled["spectral_error_bound"] < 0.5 * stalled["distance"], stalled
    _withdrawn_for_the_row(STALLED, stalled)
    assert stalled["graph"]._committed_scatter_rows == {KEY: ((ROW_EDGE, 300),)}  # noqa: SLF001


@pytest.mark.xfail(strict=True, reason=f"{ANOMALY} (open): the float floor does not count "
                   "the rounding of a long scatter row; the floor is fixed in 0.5.0")
def test_the_bound_behind_a_long_scatter_row_is_at_or_above_the_distance(stalled):
    """The number itself: what the fix of the floor will make pass."""
    assert stalled["spectral_error_bound"] >= stalled["distance"], stalled


def test_the_guard_moves_no_number_of_the_report(stalled, monkeypatch):
    """With the limit out of reach the same state's report sets both
    flags and has no reason; every other key is equal bit for bit."""
    gm = stalled["graph"]
    guarded = dict(gm.coupling_diagnostics()[KEY])
    monkeypatch.setattr(_group_layout, "SCATTER_ROW_FLOOR_LIMIT", 10 ** 9)
    bare = dict(gm.coupling_diagnostics()[KEY])
    assert bare["spectral_usable"] and bare["gradient_bound_usable"], bare
    assert "not_usable_reason" not in bare
    moved = {"spectral_usable", "gradient_bound_usable", "not_usable_reason"}
    assert set(guarded) - set(bare) == {"not_usable_reason"}
    for name in set(bare) - moved:
        assert np.asarray(guarded[name]).tobytes() == np.asarray(bare[name]).tobytes(), name


def test_the_guard_gives_its_reason_only_where_it_withdrew_a_flag(stalled, monkeypatch):
    """A flag that is already off for another cause (here a spectrum made
    to read as not settled) is not this guard's: the report carries no
    reason from it, and whatever reason another rule wrote stands."""
    from maddening.core import graph_manager as module  # noqa: PLC0415

    monkeypatch.setattr(module, "spectral_rate_settled", lambda *args, **kwargs: False)
    report = dict(stalled["graph"].coupling_diagnostics()[KEY])
    assert not report["spectral_usable"] and report["precision_limited"], report
    assert "not_usable_reason" not in report, "the patch takes effect, and no reason is added"


def test_the_report_table_gives_the_reason_beside_the_numbers(stalled):
    """``coupling_report()`` keeps the bound and says why its flag is off."""
    table = stalled["graph"].coupling_report()
    assert len(table) == 1
    row = table[0]
    assert math.isfinite(row["spectral_error_bound"]) and row["spectral_usable"] is False
    assert any("spectral_usable=False" in flag and ANOMALY in flag for flag in row["flags"]), row


def test_the_same_operator_in_the_gather_layout_keeps_its_flags_on_a_bound_that_holds():
    """The control: the gather layout is not counted, so at the same
    float floor nothing is withdrawn, behind the same uniform field and
    behind a ramp.  The bound is held to the distance on the ramp, where
    it has room on every backend measured (5x); behind the uniform field
    the gather layout's own margin is the reduction order XLA picks
    (1.8x to 1.9x measured at this length), which is not this test's."""
    for cell in (GATHERED, GATHERED_RAMP):
        report = measure(cell)
        assert report["precision_limited"], (cell.id, report)
        assert report["spectral_usable"] and report["gradient_bound_usable"], (cell.id, report)
        assert "not_usable_reason" not in report, (cell.id, report)
        assert report["graph"]._committed_scatter_rows == {KEY: ()}  # noqa: SLF001
    assert report["spectral_error_bound"] >= report["distance"] > 0.0, report


# Per push: tests/core/test_the_float_floor_of_a_long_scatter_row.py::test_the_flags_are_withdrawn_only_behind_a_long_row_at_the_floor_it_would_give
@pytest.mark.slow
def test_a_residual_that_stands_clear_of_the_rows_floor_keeps_its_flags():
    """The honest report keeps its flags, and the way out the reason
    names is verified: the same pair in float64 at float32's tolerance
    accepts with its residual far above ``floor * row``, and its bound
    holds."""
    report = measure(WIDER)
    assert not report["precision_limited"], report
    assert report["spectral_usable"] and report["gradient_bound_usable"], report
    assert "not_usable_reason" not in report
    assert report["spectral_error_bound"] >= report["distance"] > 0.0, report


# ---------------------------------------------------------------------------
# The rule itself (no step)
# ---------------------------------------------------------------------------

def _scatter(n_source: int, targets, n_target: int, counts=None) -> StaticSparseMapping:
    targets = np.asarray(targets)
    weights = np.ones(targets.shape, np.float32)
    if counts is not None:
        weights = np.where(np.arange(targets.shape[1])[None, :] < np.asarray(counts)[:, None],
                           weights, 0.0).astype(np.float32)
    return StaticSparseMapping(targets, jnp.asarray(weights), n_source=n_source, counts=counts,
                               n_target=n_target, layout="scatter")


def test_the_longest_row_is_the_most_entries_one_target_is_handed():
    """Counted per target over the valid slots: not the storage's row
    (a source's), and not a padded slot (which holds index 0)."""
    assert _longest_scatter_row(_scatter(5, [[0], [0], [0], [1], [1]], 3)) == 3
    assert _longest_scatter_row(_scatter(3, [[0, 1], [1, 2], [1, 0]], 3)) == 3
    # Source 0 has one valid slot and one padded slot (index 0): target 0
    # is handed two entries, not three.
    padded = _scatter(3, [[1, 0], [0, 2], [0, 0]], 3, counts=np.array([1, 2, 1]))
    assert _longest_scatter_row(padded) == 2
    assert _longest_scatter_row(_scatter(2, [[0], [0]], 1, counts=np.array([0, 0]))) == 0


def test_only_a_static_sparse_mapping_in_the_scatter_layout_is_counted():
    """Per internal edge: the scatter layout by its key and longest row;
    the gather layout, a dense mapping and a plain edge not at all."""
    n = 40
    scatter, gather, dense = (_row_mapping(layout, n) for layout in ("scatter", "gather", "dense"))
    edges = [EdgeSpec("a", "b", "x", "u", mapping=scatter),
             EdgeSpec("a", "b", "y", "v", mapping=gather),
             EdgeSpec("a", "b", "z", "w", mapping=dense),
             EdgeSpec("b", "a", "x", "u"),
             EdgeSpec("b", "a", "y", "v", mapping=matrix_mapping(jnp.ones((2, 3))))]
    assert _scatter_rows(edges) == (("a.x->b.u", n),)
    assert _scatter_rows(edges[1:]) == ()


def test_the_flags_are_withdrawn_only_behind_a_long_row_at_the_floor_it_would_give():
    """The rule on the report's two numbers: a row within the limit never
    withdraws; a longer one does where the residual is at or below
    ``floor * row`` and not above it; nothing is withdrawn without a
    floor; the reason names the longest row and lists the rest."""
    limit = SCATTER_ROW_FLOOR_LIMIT
    assert _scatter_row_reason((), 0.0, 1.0) is None
    assert _scatter_row_reason((("e", limit),), 0.0, 1.0) is None
    assert _scatter_row_reason((("e", limit + 1),), 0.0, 1.0) is not None
    long = (("e", 40 * limit),)
    assert _scatter_row_reason(long, 0.5, 1.0) is not None      # precision_limited
    assert _scatter_row_reason(long, 40.0 * limit, 1.0) is not None     # inside the row's reach
    assert _scatter_row_reason(long, 40.0 * limit + 0.5, 1.0) is None   # stands clear
    assert _scatter_row_reason(long, 0.0, 0.0) is None          # no floor: nothing rests on it
    assert _scatter_row_reason(long, float("nan"), 1.0) is None
    reason = _scatter_row_reason((("short", limit), ("e", 20 * limit), ("f", 30 * limit)), 0.0, 1.0)
    assert reason is not None
    assert "edge f " in reason and f"adds up {30 * limit} entries" in reason
    assert "['e']" in reason and "short" not in reason


def test_the_limit_is_a_power_of_ten():
    """It is a measured constant (see its comment): the largest power of
    ten at which every measured run held by a factor of two."""
    assert SCATTER_ROW_FLOOR_LIMIT >= 1
    assert 10 ** round(math.log10(SCATTER_ROW_FLOOR_LIMIT)) == SCATTER_ROW_FLOOR_LIMIT


# ---------------------------------------------------------------------------
# The table (slow): layouts, row lengths, fields, gains, schedules, norms, dtypes
# ---------------------------------------------------------------------------

def _cells(pairs, fields=("uniform", "ramp", "random"), gains=(0.9, 0.99)) -> list:
    return [Cell(pair, field, gain) for pair in pairs for field in fields for gain in gains]


_SCHEDULES = ("jacobi", "gauss-seidel")
#: Rows within the limit, in the scatter layout: the flag is kept.
WITHIN = _cells([Pair("scatter", n, schedule, norm) for n in (3, 10)
                 for schedule in _SCHEDULES for norm in ("interface", "mixed")])
#: Longer rows at their float floor: withdrawn.  Both dtypes (float64 at
#: its own floor), the three norms.
BEYOND = _cells(
    [Pair("scatter", n, schedule, norm) for n in (100, 3000, 30000)
     for schedule in _SCHEDULES for norm in ("interface", "mixed")]
    + [Pair("scatter", 3000, "jacobi", "l2"),
       Pair("scatter", 300, "jacobi", "interface", "float64"),
       Pair("scatter", 30000, "gauss-seidel", "mixed", "float64")])
#: The controls, at the floor, one row: the gather layout and the dense
#: matrix at 300 entries, and the dense matrix at 3e4.  They keep their
#: flags (the guard does not count them).  Their bound is held to the
#: distance behind the fields that are not uniform, where it has room on
#: every backend measured (5x); behind a uniform field, and for other
#: shapes, the margin is the order XLA picks for the sum, which is
#: another on another jax or CPU and is not this guard's to promise (the
#: registry entry has the measurements, the failing ones included).
REDUCED = (
    _cells([Pair(layout, 300, schedule, norm) for layout in ("gather", "dense")
            for schedule in _SCHEDULES for norm in ("interface", "mixed")])
    + _cells([Pair("dense", 30000, schedule, "interface") for schedule in _SCHEDULES]
             + [Pair("dense", 300, "jacobi", "interface", "float64")]))


# Per push: tests/core/test_the_float_floor_of_a_long_scatter_row.py::test_the_flags_are_withdrawn_only_behind_a_long_row_at_the_floor_it_would_give
@pytest.mark.slow
@pytest.mark.parametrize("cell", WITHIN, ids=lambda cell: cell.id)
def test_a_scatter_row_within_the_limit_keeps_its_flags_on_a_bound_that_holds(cell):
    report = measure(cell)
    assert report["precision_limited"], (cell.id, report)
    assert report["spectral_usable"] and "not_usable_reason" not in report, (cell.id, report)
    assert report["spectral_error_bound"] >= 2.0 * report["distance"], (cell.id, report)
    _a_set_flag_stands_on_a_bound_at_or_above_the_distance(cell, report)


# Per push: tests/core/test_the_float_floor_of_a_long_scatter_row.py::test_a_stalled_group_behind_a_long_scatter_row_withdraws_its_flags_and_says_why
@pytest.mark.slow
@pytest.mark.parametrize("cell", BEYOND, ids=lambda cell: cell.id)
def test_a_set_flag_stands_on_a_bound_at_or_above_the_distance_behind_a_long_scatter_row(cell):
    """Every such report at its float floor has its flags withdrawn for
    the row, whether its bound happened to hold or not."""
    report = measure(cell)
    assert report["precision_limited"], (cell.id, report)
    _withdrawn_for_the_row(cell, report)
    _a_set_flag_stands_on_a_bound_at_or_above_the_distance(cell, report)


# Per push: tests/core/test_the_float_floor_of_a_long_scatter_row.py::test_the_same_operator_in_the_gather_layout_keeps_its_flags_on_a_bound_that_holds
@pytest.mark.slow
@pytest.mark.parametrize("cell", REDUCED, ids=lambda cell: cell.id)
def test_the_gather_layout_and_the_dense_matrix_keep_their_flags_on_a_bound_that_holds(cell):
    report = measure(cell)
    assert report["precision_limited"], (cell.id, report)
    assert report["spectral_usable"] and "not_usable_reason" not in report, (cell.id, report)
    if cell.field != "uniform":
        assert report["spectral_error_bound"] >= report["distance"] > 0.0, (cell.id, report)


#: Tolerances above the float floor (float32): ``precision_limited`` is
#: False in every one.  ``True``: the residual is above the floor as
#: counted and inside the row's reach (2 to 5 floors behind 3e4 entries;
#: the first two read 0.012 and 0.0086 of their distance), so the flags
#: are withdrawn.  ``False``: it stands clear (over 1000 floors behind
#: 100 entries) and they are kept, on a bound that holds.
LOOSER = [
    (Cell(Pair("scatter", 30000, "jacobi", "interface", rtol=1e-5)), True),
    (Cell(Pair("scatter", 30000, "gauss-seidel", "mixed", rtol=1e-5)), True),
    (Cell(Pair("scatter", 30000, "jacobi", "mixed", rtol=1e-5), "random"), True),
    (Cell(Pair("scatter", 100, "jacobi", "interface", rtol=1e-1)), False),
    (Cell(Pair("scatter", 100, "gauss-seidel", "interface", rtol=1e-1), "random"), False),
    (Cell(Pair("scatter", 100, "jacobi", "mixed", rtol=1e-1)), False),
]


# Per push: tests/core/test_the_float_floor_of_a_long_scatter_row.py::test_the_flags_are_withdrawn_only_behind_a_long_row_at_the_floor_it_would_give
@pytest.mark.slow
@pytest.mark.parametrize("cell, withdrawn", LOOSER, ids=lambda v: v.id if isinstance(v, Cell) else "")
def test_a_residual_above_the_counted_floor_is_judged_against_the_rows_own(cell, withdrawn):
    """``precision_limited`` is False in every one of these: the residual
    is above the floor as counted.  Inside ``floor * row`` the flags are
    withdrawn; clear of it they are kept and the bound holds."""
    report = measure(cell)
    assert not report["precision_limited"], (cell.id, report)
    if withdrawn:
        _withdrawn_for_the_row(cell, report)
    else:
        assert report["spectral_usable"] and "not_usable_reason" not in report, (cell.id, report)
    _a_set_flag_stands_on_a_bound_at_or_above_the_distance(cell, report)
