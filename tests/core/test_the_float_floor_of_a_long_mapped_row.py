"""The float floor does not count the rounding of a mapped edge's row sum; the flag does.

A static mapping delivers sums over its rows.  A float sum of ``k`` terms
of one sign rounds by up to ``(k - 1) / 2`` units of ``eps`` of the sum,
and where the terms are nearly equal an in-order sum rounds
systematically: it grows like ``k``.  The float floor of a coupling
report counts a fixed number of ulps per evaluation, so a group stalled
behind such a row is further from its fixed point than
``spectral_error_bound`` says (MADD-ANO-257, open: the floor is not fixed
in 0.4.0).

What 0.4.0 does: the report withdraws ``spectral_usable`` and
``gradient_bound_usable``, with a ``not_usable_reason``, where an
internal edge carries a static mapping of **any kind** (a dense matrix,
a sparse mapping in the gather layout or in the scatter layout, a
registered kind's own static class) with a row longer than
``MAPPED_ROW_FLOOR_LIMIT`` entries **and** the residual does not stand
clear of the floor that row could give it (``floor * row``).  It only
withdraws: every number of the report is the one it was.

Which rows fail, measured (the registry entry has the tables): the
scatter layout from 100 entries on, on every jax version (a scatter-add
is the in-order sum on the CPU); a dense matrix of several long rows on
every version; the gather layout's rows of 1e4 entries and more on jax
0.10.2 in float32, where XLA sums them in order.  A single gather row,
and a dense matrix of one row, held their bound at every length on the
other versions.  The guard does not tell these apart: the order of a
reduced sum is XLA's to choose, per version, dtype and shape, so every
static kind is counted by its row's length.

**The pair** (float32 unless a cell says otherwise; every weight
positive, so nothing cancels; both nodes are relays that declare one
evaluation; the constants arrive as external inputs, so one compiled
pair serves every field and gain)::

    coarse (m values)      x[i] <- b + a * u[i]     u[i] = what the fine block i sums to
    fine   (m * n values)  x[j] <- c_j + u[j]       u[j] = its block's coarse value
    coarse.x -> fine.u     sparse nearest neighbour, m -> m * n
    fine.x   -> coarse.u   the conservative nearest neighbour, m * n -> m: m rows of n
                           entries, in the scatter layout, the gather layout or dense

with loop gain ``a * n``; ``m`` is one unless a pair says otherwise.
Its fixed point is a closed form of the constants as the dtype holds
them: ``x*[i] = (b + a sum(c over block i)) / (1 - a n)``.
"""

from __future__ import annotations

import gc
import math
import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _group_layout, reason_codes
from maddening.core.coupling._group_layout import (
    MAPPED_ROW_FLOOR_LIMIT,
    _joined_reasons,
    _longest_row,
    _mapped_row_reason,
    _mapped_rows,
)
from maddening.core.coupling.acceleration import PRECISION_FLOOR_ULPS, SPECTRAL_SETTLED_FRACTION
from maddening.core.coupling.grid_mapping import multilinear_grid_mapping
from maddening.core.coupling.mapping import matrix_mapping, nearest_neighbor_mapping
from maddening.core.coupling.sparse_mapping import (
    StaticSparseMapping,
    sparse_nearest_neighbor_mapping,
)
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.registered_mapping_kinds import inverse_distance_mapping, selection_mapping
from tests.sparse_mapping_support import x64

KEY = "coarse+fine"
ROW_EDGE = "fine.x->coarse.u"
BACK_EDGE = "coarse.x->fine.u"
ANOMALY = "MADD-ANO-257"
#: The guard's reason code (experimental).
LONG_ROW = reason_codes.LONG_MAPPED_ROW
LAYOUTS = ("scatter", "gather", "dense")
#: How the reason names each form, and what it must not name beside it:
#: the way out is a wider dtype, never another kind or layout.
SAID = {
    "scatter": "a static sparse mapping (sparse_nearest_neighbor) in the scatter layout",
    "gather": "a static sparse mapping (sparse_nearest_neighbor) in the gather layout",
    "dense": "a dense matrix mapping (nearest_neighbor)",
}
NOT_SAID = {
    "scatter": ("gather layout", "dense", "transpose"),
    "gather": ("scatter layout", "dense", "transpose"),
    "dense": ("gather layout", "scatter layout", "sparse", "transpose"),
}


class Coarse(SimulationNode):
    """``m`` values: ``x <- b + a * u``, with ``(a, b)`` an external input."""

    def __init__(self, name, timestep, m, dtype):
        super().__init__(name, timestep)
        self._m, self._dt = m, jnp.dtype(dtype)

    def initial_state(self):
        return {"x": jnp.zeros(self._m, self._dt)}

    def boundary_input_spec(self):
        zeros = jnp.zeros(self._m, self._dt)
        return {"u": BoundaryInputSpec(shape=(self._m,), dtype=self._dt, default=zeros),
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
    """What a compiled pair is built from: the layout of the rows, the
    entries each row adds up, the schedule, the norm, the dtype, the
    tolerance (``None``: below the dtype's float floor, so the pair runs
    until it stalls) and how many rows there are."""

    layout: str
    n: int
    schedule: str = "jacobi"
    norm: str = "interface"
    dtype: str = "float32"
    rtol: float | None = None
    rows: int = 1

    @property
    def tolerance(self) -> float:
        if self.rtol is not None:
            return self.rtol
        return 1e-16 if self.dtype == "float64" else 1e-7

    @property
    def fine(self) -> int:
        """The fine field's entries: every row's ``n``."""
        return self.n * self.rows

    @property
    def counted(self) -> int:
        """The row the guard counts: a sparse layout's own entries; a
        dense matrix's width, which is the whole fine field."""
        return self.fine if self.layout == "dense" else self.n

    @property
    def id(self) -> str:
        rtol = "" if self.rtol is None else f"-rtol{self.rtol:g}"
        rows = "" if self.rows == 1 else f"-m{self.rows}"
        return f"{self.layout}-n{self.n}{rows}-{self.schedule}-{self.norm}-{self.dtype}{rtol}"


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
    (``1 + j / n``), ``"random"`` (uniform in ``[1, 2)``, seeded) or
    ``"cancelling"`` (alternating ``+1, -1`` beside 0.004: the terms of
    a row cancel, and the coarse relay then adds no offset, so that the
    returned field keeps its sign changes)."""
    if field == "uniform":
        return np.full(n, 1.1, dtype)
    if field == "cancelling":
        return (np.where(np.arange(n) % 2 == 0, 1.0, -1.0) + 0.004).astype(dtype)
    if field == "ramp":
        return (1.0 + np.arange(n) / n).astype(dtype)
    assert field == "random", field
    return np.random.default_rng(1).uniform(1.0, 2.0, n).astype(dtype)


def _points(pair: Pair) -> tuple:
    """``(fine points, coarse points)``: block ``i`` of ``n`` fine points
    is nearest coarse point ``i``."""
    return ((np.arange(pair.fine) + 0.5) / pair.fine, (np.arange(pair.rows) + 0.5) / pair.rows)


def _row_mapping(layout: str, n: int, rows: int = 1):
    fine_points, coarse_points = _points(Pair(layout, n, rows=rows))
    if layout == "dense":
        return nearest_neighbor_mapping(fine_points, coarse_points, mode="conservative")
    return sparse_nearest_neighbor_mapping(
        fine_points, coarse_points, mode="conservative", transpose=layout)


_COMPILED: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _the_compiled_pairs_are_dropped_after_the_module():
    """The module keeps every compiled pair; a shard runs many modules in
    one process, so the module's own table is emptied after the last test
    here.  Nothing else: a collection or ``jax.clear_caches()`` here costs
    what the whole process has compiled, is charged to this module's last
    test, and throws away the programs of the modules that follow."""
    yield
    _COMPILED.clear()


#: The compiled pairs the slow tests may hold before they are released.
#: Their cells come pair by pair, so a release after every one of them made
#: each cell compile its pair again: the slow tests took 1461 s; with every
#: pair kept, 262 s and 5.9 GB at the peak; with this limit 276 s and 2.4 GB
#: (jax 0.11.0, 8 cores).
PAIRS_HELD = 16


@pytest.fixture(autouse=True)
def _a_slow_test_releases_what_it_compiled(request):
    """The slow tests compile a pair for every cell they sweep, and the
    slow lane runs a whole shard in one process: a slow test that ends
    with more than ``PAIRS_HELD`` pairs held releases them and what JAX
    compiled for them.  The per-push tests keep their pairs until the
    module ends."""
    yield
    if request.node.get_closest_marker("slow") is not None and len(_COMPILED) > PAIRS_HELD:
        _COMPILED.clear()
        gc.collect()
        jax.clear_caches()
        gc.collect()


def compiled(pair: Pair) -> GraphManager:
    """The pair, compiled once per module (run inside ``x64`` as its dtype needs)."""
    if pair not in _COMPILED:
        dt = jnp.dtype(pair.dtype)
        fine_points, coarse_points = _points(pair)
        gm = GraphManager()
        gm.add_node(Coarse("coarse", 1.0, pair.rows, dt))
        gm.add_node(Fine("fine", 1.0, pair.fine, dt))
        gm.add_external_input("coarse", "ab", shape=(2,), dtype=dt)
        gm.add_external_input("fine", "c", shape=(pair.fine,), dtype=dt)
        gm.add_edge("coarse", "fine", "x", "u", mapping=sparse_nearest_neighbor_mapping(
            coarse_points, fine_points))
        gm.add_edge("fine", "coarse", "x", "u",
                    mapping=_row_mapping(pair.layout, pair.n, pair.rows))
        # "l2" is the root of the sum over the entries, not their RMS: the
        # same demand of each entry is ``rtol * sqrt(entries)`` there.
        tolerance = ({"tolerance": pair.tolerance * math.sqrt(pair.fine + pair.rows)}
                     if pair.norm == "l2" else {"rtol": pair.tolerance})
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
    n, m = pair.n, pair.rows
    with x64(pair.dtype == "float64"):
        gm = compiled(pair)
        np_dtype = np.dtype(pair.dtype).type
        c = constants(pair.fine, cell.field, np_dtype)
        a, b = np_dtype(cell.gain / n), np_dtype(0.0 if cell.field == "cancelling" else 1.0)
        gm.reset_state()
        gm.set_node_state("fine", {"x": jnp.asarray(c)})
        gm.step(external_inputs={"coarse": {"ab": jnp.asarray([a, b])},
                                 "fine": {"c": jnp.asarray(c)}})
        report = dict(gm.coupling_diagnostics()[KEY])
        x = np.asarray(gm.get_node_state("coarse")["x"], np.float64)
        fine = np.asarray(gm.get_node_state("fine")["x"], np.float64)
    # The fixed point of the exact map of the constants as the dtype holds them.
    c64 = np.asarray(c, np.float64)
    sums_c = np.array([math.fsum(c64[i * n:(i + 1) * n].tolist()) for i in range(m)])
    x_star = (float(b) + float(a) * sums_c) / (1.0 - float(a) * n)
    coarse_errors = np.abs(x - x_star) / np.max(np.abs(x))
    if pair.norm == "interface":
        # Two readings: the coarse values at their source, and what the
        # rows deliver (the exact sums of the returned field).
        delivered = np.array([math.fsum(fine[i * n:(i + 1) * n].tolist()) for i in range(m)])
        delivered_errors = (np.abs(delivered - (sums_c + n * x_star))
                            / np.max(np.abs(delivered)))
        distance = math.sqrt(float(np.sum(coarse_errors ** 2) + np.sum(delivered_errors ** 2))
                             / (2.0 * m)) / pair.tolerance
    else:
        terms = np.concatenate([coarse_errors, np.abs(fine - (c64 + np.repeat(x_star, n)))
                                / np.max(np.abs(fine))])
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
    """Both flags off, and the reason names the edge, how its mapping is
    applied, the row the guard counted, the limit, the anomaly and the
    one way out; it names no other kind or layout."""
    assert not report["spectral_usable"] and not report["gradient_bound_usable"], (
        cell.id, report)
    reason = report["not_usable_reason"]
    for said in (ROW_EDGE, SAID[cell.pair.layout], f"adds up {cell.pair.counted} entries",
                 f"the limit is {MAPPED_ROW_FLOOR_LIMIT}", ANOMALY, "a wider dtype"):
        assert said in reason, (cell.id, said, reason)
    for unsaid in NOT_SAID[cell.pair.layout]:
        assert unsaid not in reason, (cell.id, unsaid, reason)


def _withdrawn_for_the_row_where_a_flag_was_set(cell: Cell, report: dict, monkeypatch) -> None:
    """:func:`_withdrawn_for_the_row`, for a report the guard had a flag
    to withdraw from.  A pair of one row always has one.  A pair of
    several rows repeats one loop, and where its spectrum does not
    resolve (a bound of ``inf``) another cause held the flag down before
    this one was asked: the guard then withdraws nothing and names its
    caveat all the same (the floor the entry reports does not count the
    row), before that cause's own words, which are the whole reason of
    the same state with the limit out of reach."""
    with x64(cell.pair.dtype == "float64"):
        monkeypatch.setattr(_group_layout, "MAPPED_ROW_FLOOR_LIMIT", 10 ** 9)
        bare = dict(report["graph"].coupling_diagnostics()[KEY])
        monkeypatch.undo()
    if bare["spectral_usable"]:
        _withdrawn_for_the_row(cell, report)
        return
    assert cell.pair.rows > 1, (cell.id, "a pair of one row has a flag to withdraw", bare)
    assert not report["spectral_usable"] and not report["gradient_bound_usable"], (cell.id, report)
    rows_said, also, own = report["not_usable_reason"].partition(" Also: ")
    assert also and ANOMALY in rows_said and ANOMALY not in own, (cell.id, report)
    assert own == bare["not_usable_reason"], (cell.id, report, bare)
    kept = bare["reason_codes"]["spectral_usable"]
    assert kept and LONG_ROW not in kept, (cell.id, bare["reason_codes"])
    assert set(report["reason_codes"]["spectral_usable"]) == {*kept, LONG_ROW}, (cell.id, report)


def _committed(pair: Pair) -> dict:
    """What ``compile()`` commits for the pair: every static mapped edge
    of the group, the row edge at the length the guard counts and the
    edge back at its one entry a row."""
    return {KEY: ((BACK_EDGE, "a static sparse mapping (sparse_nearest_neighbor) in the "
                              "gather layout", 1),
                  (ROW_EDGE, _longest_row(_row_mapping(pair.layout, pair.n, pair.rows))[0],
                   pair.counted))}


# ---------------------------------------------------------------------------
# Per push: one stalled pair per kind, and a short row that keeps its flags
# ---------------------------------------------------------------------------

#: A uniform field, stalled at float32, behind a long row of each kind.
#: Before the guard every one of these reports set both flags, and read
#: its bound at:
#:
#: * the scatter layout, one row of 300 entries: 0.24 of the distance;
#: * a dense matrix of three rows of 3000 entries: 0.18 of the distance
#:   (jax 0.10.2, 0.11.0 and 0.11.2 alike);
#: * the gather layout, one row of 300 entries: 1.8 times the distance.
#:   It held here.  The gather layout's sum is in the order XLA chooses,
#:   and the rows it sums in order read as the scatter layout's do (1e4
#:   entries on jax 0.10.2 in float32: 0.0068, in the slow table), so
#:   the guard counts it by its length like the others; this report is
#:   part of the price.
STALLED = {
    "scatter": Cell(Pair("scatter", 300)),
    "gather": Cell(Pair("gather", 300)),
    "dense": Cell(Pair("dense", 3000, rows=3)),
}
#: The kinds whose stalled report above reads its bound below the distance.
UNDER = ("scatter", "dense")
#: A dense row of exactly the limit: counted, and within it.
SHORT = Cell(Pair("dense", MAPPED_ROW_FLOOR_LIMIT))
#: One row of 300 entries of each kind in float64 at the float32
#: tolerance: the way out the reason names, and a residual that stands
#: clear of the row's floor.  (One row: the three-row dense pair does
#: not resolve its spectrum there, and has no flag for another cause.)
WIDER = [Cell(Pair(layout, 300, dtype="float64", rtol=1e-7)) for layout in LAYOUTS]


@pytest.fixture(scope="module", params=LAYOUTS)
def stalled(request) -> dict:
    report = measure(STALLED[request.param])
    report["cell"] = STALLED[request.param]
    return report


@pytest.fixture(scope="module", params=UNDER)
def stalled_under(request) -> dict:
    return measure(STALLED[request.param])


def test_a_stalled_group_behind_a_long_row_withdraws_its_flags_and_says_why(stalled):
    """At its float floor (``precision_limited``), behind a long row of
    each static kind: both flags withdrawn, and the reason names the
    edge, the kind and layout, the row's length, the limit and the way
    out.  The premise is asserted with it for the kinds whose bound is
    below the distance here: the flags they would have set were wrong."""
    cell = stalled["cell"]
    assert stalled["converged"] and stalled["precision_limited"], stalled
    assert math.isfinite(stalled["spectral_error_bound"]), stalled
    if cell.pair.layout in UNDER:
        assert stalled["spectral_error_bound"] < 0.5 * stalled["distance"], stalled
    _withdrawn_for_the_row(cell, stalled)
    assert stalled["graph"]._committed_mapped_rows == _committed(cell.pair)  # noqa: SLF001


@pytest.mark.xfail(strict=True, reason=f"{ANOMALY} (open): the float floor does not count "
                   "the rounding of a mapped edge's row sum; the floor is fixed in 0.5.0")
def test_the_bound_behind_a_long_row_is_at_or_above_the_distance(stalled_under):
    """The number itself, behind the scatter layout and behind a dense
    matrix: what the fix of the floor will make pass."""
    assert stalled_under["spectral_error_bound"] >= stalled_under["distance"], stalled_under


def test_the_guard_moves_no_number_of_the_report(stalled, monkeypatch):
    """With the limit out of reach the same state's report sets its
    flags and has no reason of the guard's, nor its code; every other key
    is equal bit for bit.  (The three-row dense pair's gradient flag is
    off without the guard too: a cause of its own, which that report
    names, as the guarded one does after the guard's.)"""
    gm = stalled["graph"]
    guarded = dict(gm.coupling_diagnostics()[KEY])
    monkeypatch.setattr(_group_layout, "MAPPED_ROW_FLOOR_LIMIT", 10 ** 9)
    bare = dict(gm.coupling_diagnostics()[KEY])
    assert bare["spectral_usable"], bare
    assert bare["gradient_bound_usable"] == (stalled["cell"].pair.rows == 1), bare
    assert ("not_usable_reason" in bare) == (not bare["gradient_bound_usable"]), bare
    assert ANOMALY not in bare.get("not_usable_reason", "")
    own = bare["reason_codes"]["gradient_bound_usable"]
    assert bare["reason_codes"]["spectral_usable"] == [] and LONG_ROW not in own, bare
    assert guarded["reason_codes"]["spectral_usable"] == [LONG_ROW], guarded["reason_codes"]
    assert guarded["reason_codes"]["gradient_bound_usable"] == [LONG_ROW, *own]
    assert guarded["not_usable_reason"].startswith("the group's internal edge")
    moved = {"spectral_usable", "gradient_bound_usable", "not_usable_reason", "reason_codes"}
    assert set(guarded) - set(bare) <= {"not_usable_reason"}
    for name in set(bare) - moved:
        assert np.asarray(guarded[name]).tobytes() == np.asarray(bare[name]).tobytes(), name


def test_the_guard_names_its_cause_beside_a_flag_another_cause_holds_down(stalled, monkeypatch):
    """A flag that is already off for another cause (here a spectrum made
    to read as not settled) still rests on a floor that does not count
    the row: the report names both causes, in codes and in words, the
    guard's first, and moves no number.  (Before every report gave the
    causes of a ``False`` flag, the guard spoke only where it withdrew
    one, and this report had no reason at all.)"""
    from maddening.core import graph_manager as module  # noqa: PLC0415

    honest = dict(stalled["graph"].coupling_diagnostics()[KEY])
    monkeypatch.setattr(module, "spectral_rate_settled", lambda *args, **kwargs: False)
    report = dict(stalled["graph"].coupling_diagnostics()[KEY])
    assert not report["spectral_usable"] and report["precision_limited"], report
    listed = report["reason_codes"]["spectral_usable"]
    assert LONG_ROW in listed and set(listed) & {
        reason_codes.INTERFACE_TOO_WIDE, reason_codes.SPECTRAL_SELF_CHECK_FAILED}, (
        "the patch takes effect, and both causes are listed", listed)
    reason = report["not_usable_reason"]
    assert reason.startswith("the group's internal edge") and ANOMALY in reason
    assert "the spectral estimate did not settle" in reason, reason
    for name in set(report) - {"not_usable_reason", "reason_codes"}:
        assert np.asarray(report[name]).tobytes() == np.asarray(honest[name]).tobytes(), name


def test_the_report_table_gives_the_reason_beside_the_numbers(stalled):
    """``coupling_report()`` keeps the bound and says why its flag is off."""
    table = stalled["graph"].coupling_report()
    assert len(table) == 1
    row = table[0]
    assert math.isfinite(row["spectral_error_bound"]) and row["spectral_usable"] is False
    assert any("spectral_usable=False" in flag and ANOMALY in flag for flag in row["flags"]), row


def test_a_row_within_the_limit_keeps_its_flags_on_a_bound_that_holds():
    """The control: a dense row of exactly the limit is counted and is
    within it, so at the same float floor nothing is withdrawn, and the
    bound holds (4.9 times the distance measured)."""
    report = measure(SHORT)
    assert report["converged"] and report["precision_limited"], report
    assert report["spectral_usable"] and report["gradient_bound_usable"], report
    assert "not_usable_reason" not in report, report
    assert report["graph"]._committed_mapped_rows == _committed(SHORT.pair)  # noqa: SLF001
    assert report["spectral_error_bound"] >= report["distance"] > 0.0, report


# Per push: tests/core/test_the_float_floor_of_a_long_mapped_row.py::test_the_flags_are_withdrawn_only_behind_a_long_row_at_the_floor_it_would_give
@pytest.mark.slow
@pytest.mark.parametrize("cell", WIDER, ids=lambda cell: cell.id)
def test_a_residual_that_stands_clear_of_the_rows_floor_keeps_its_flags(cell):
    """The honest report keeps its flags, and the way out the reason
    names is verified for each kind: the same pair in float64 at
    float32's tolerance accepts with its residual far above
    ``floor * row``, and its bound holds."""
    report = measure(cell)
    assert not report["precision_limited"], report
    assert report["spectral_usable"] and report["gradient_bound_usable"], report
    assert "not_usable_reason" not in report
    assert report["spectral_error_bound"] >= report["distance"] > 0.0, report


# ---------------------------------------------------------------------------
# The rule itself (no step)
# ---------------------------------------------------------------------------

def _sparse(layout: str, rows, n_source: int, n_target: int, counts=None) -> StaticSparseMapping:
    rows = np.asarray(rows)
    weights = np.ones(rows.shape, np.float32)
    if counts is not None:
        weights = np.where(np.arange(rows.shape[1])[None, :] < np.asarray(counts)[:, None],
                           weights, 0.0).astype(np.float32)
    return StaticSparseMapping(rows, jnp.asarray(weights), n_source=n_source, counts=counts,
                               n_target=n_target, layout=layout)


def _scatter(n_source: int, targets, n_target: int, counts=None) -> StaticSparseMapping:
    return _sparse("scatter", targets, n_source, n_target, counts)


def test_a_scatter_row_is_the_most_entries_one_target_is_handed():
    """Counted per target over the valid slots: not the storage's row
    (a source's), and not a padded slot (which holds index 0)."""
    assert _longest_row(_scatter(5, [[0], [0], [0], [1], [1]], 3))[1] == 3
    assert _longest_row(_scatter(3, [[0, 1], [1, 2], [1, 0]], 3))[1] == 3
    # Source 0 has one valid slot and one padded slot (index 0): target 0
    # is handed two entries, not three.
    padded = _scatter(3, [[1, 0], [0, 2], [0, 0]], 3, counts=np.array([1, 2, 1]))
    assert _longest_row(padded)[1] == 2
    assert _longest_row(_scatter(2, [[0], [0]], 1, counts=np.array([0, 0])))[1] == 0
    assert "scatter layout" in _longest_row(padded)[0]


def test_a_gather_row_is_its_valid_slots():
    """The storage's row is the operator's there: every slot of a full
    row, the most valid slots where rows are padded (a padded slot reads
    a zero whatever its weight is), whatever the weights are."""
    full = _sparse("gather", [[0, 1, 2], [2, 3, 4]], 5, 2)
    assert _longest_row(full)[1] == 3 and "gather layout" in _longest_row(full)[0]
    padded = _sparse("gather", [[0, 1, 2, 3], [4, 0, 0, 0], [1, 2, 0, 0]], 5, 3,
                     counts=np.array([4, 1, 2]))
    assert _longest_row(padded)[1] == 4
    assert _longest_row(_sparse("gather", [[0, 0], [1, 0]], 2, 2, counts=np.array([0, 1])))[1] == 1
    assert _longest_row(_sparse("gather", [[0, 0]], 2, 1, counts=np.array([0])))[1] == 0
    zero_weights = StaticSparseMapping(np.array([[0, 1, 2]]), jnp.zeros((1, 3), jnp.float32),
                                       n_source=3)
    assert _longest_row(zero_weights)[1] == 3


def test_a_dense_row_is_the_matrix_width_whatever_its_weights():
    """``H @ field`` adds up one product per source entry.  Which are
    zero is for the weights to say, and a step may be handed other
    weights of the same shape, so the row is the width: a selection
    matrix (one non-zero a row) and a full one count alike, and the
    number of rows does not enter."""
    selection = matrix_mapping(jnp.eye(4, 40))
    full = matrix_mapping(jnp.ones((2, 40)))
    assert _longest_row(selection) == (
        "a dense matrix mapping (matrix), counted at the matrix's width,", 40)
    assert _longest_row(full)[1] == 40
    assert _longest_row(matrix_mapping(jnp.ones((40, 3))))[1] == 3
    assert _longest_row(_row_mapping("dense", 7, rows=3))[1] == 21
    assert SAID["dense"] in _longest_row(_row_mapping("dense", 7, rows=3))[0]


def test_a_static_mapping_of_another_class_is_counted_at_its_source_side():
    """A registered kind's own static class: the most one delivered value
    can add up is the source side's entries, and the reason names the
    class, since its ``apply`` is its author's."""
    points = np.linspace(0.0, 1.0, 23)
    shepard = inverse_distance_mapping(points, points[:4])
    what, row = _longest_row(shepard)
    assert row == 23 and "class InverseDistanceMapping" in what and "source side" in what
    chosen = selection_mapping(points, points[:4])
    assert _longest_row(chosen)[1] == 23 and "class SelectionMapping" in _longest_row(chosen)[0]


def test_every_static_mapping_on_an_internal_edge_is_counted():
    """Per internal edge, in the order given: the scatter layout, the
    gather layout, a dense matrix and a registered kind's own class by
    their key, their form and their longest row; a plain edge and a
    geometry-dependent mapping not at all."""
    n = 40
    scatter, gather, dense = (_row_mapping(layout, n) for layout in LAYOUTS)
    other = inverse_distance_mapping(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 2))
    grid = multilinear_grid_mapping([0.0], [1.0], [2], n_points=n, mode="conservative")
    edges = [EdgeSpec("a", "b", "x", "u", mapping=scatter),
             EdgeSpec("a", "b", "y", "v", mapping=gather),
             EdgeSpec("a", "b", "z", "w", mapping=dense),
             EdgeSpec("b", "a", "x", "u"),
             EdgeSpec("b", "a", "y", "v", mapping=matrix_mapping(jnp.ones((2, 3)))),
             EdgeSpec("b", "a", "z", "w", mapping=other),
             EdgeSpec("a", "b", "p", "q", mapping=grid, geometry=("source", "pos"))]
    rows = _mapped_rows(edges)
    assert [(key, row) for key, _what, row in rows] == [
        ("a.x->b.u", n), ("a.y->b.v", n), ("a.z->b.w", n), ("b.y->a.v", 3), ("b.z->a.w", 5)]
    for (_key, what, _row), layout in zip(rows, LAYOUTS):
        assert SAID[layout] in what, (layout, what)
    assert _mapped_rows(edges[3:4]) == () and _mapped_rows(edges[6:]) == ()


def test_the_flags_are_withdrawn_only_behind_a_long_row_at_the_floor_it_would_give():
    """The rule on the report's two numbers: a row within the limit never
    withdraws; a longer one does where the residual is at or below
    ``floor * row`` and not above it; nothing is withdrawn without a
    floor; the reason names the longest row, how it is applied, and
    lists the rest.  The same for every form a row comes in."""
    limit = MAPPED_ROW_FLOOR_LIMIT
    assert _mapped_row_reason((), 0.0, 1.0) is None
    for what in (*SAID.values(), "a static mapping of class Mine"):
        assert _mapped_row_reason((("e", what, limit),), 0.0, 1.0) is None
        reason = _mapped_row_reason((("e", what, limit + 1),), 0.0, 1.0)
        assert reason is not None and f"edge e carries {what} whose longest row" in reason
        long = (("e", what, 40 * limit),)
        assert _mapped_row_reason(long, 0.5, 1.0) is not None      # precision_limited
        assert _mapped_row_reason(long, 40.0 * limit, 1.0) is not None     # inside the row's reach
        assert _mapped_row_reason(long, 40.0 * limit + 0.5, 1.0) is None   # stands clear
        assert _mapped_row_reason(long, 0.0, 0.0) is None          # no floor: nothing rests on it
        assert _mapped_row_reason(long, float("nan"), 1.0) is None
    reason = _mapped_row_reason((("short", SAID["dense"], limit), ("e", SAID["gather"], 20 * limit),
                                 ("f", SAID["scatter"], 30 * limit)), 0.0, 1.0)
    assert reason is not None
    assert f"edge f carries {SAID['scatter']}" in reason
    assert f"adds up {30 * limit} entries" in reason
    assert "['e']" in reason and "short" not in reason


def test_the_reason_names_one_way_out_and_no_other_kind_or_layout():
    """Every static kind is counted alike, so the reason must not send
    its reader to another one: it names the edge's own form and a wider
    dtype, nothing else."""
    for layout in LAYOUTS:
        reason = _mapped_row_reason((("e", SAID[layout], 1000),), 0.0, 1.0)
        assert reason is not None and "a wider dtype" in reason
        for unsaid in NOT_SAID[layout]:
            assert unsaid not in reason, (layout, unsaid, reason)


def test_a_reason_another_rule_left_is_kept_beside_the_rows():
    """Another rule may leave a flag of its own off, with its reason,
    while ``spectral_usable`` stands (a geometry group with constant
    positions whose gradient bound is not finite).  Where the row guard
    then withdraws ``spectral_usable``, the report keeps that reason and
    adds this one."""
    assert _joined_reasons(None, "the row.") == "the row."
    assert _joined_reasons("", "the row.") == "the row."
    assert _joined_reasons("the gradient bound.", "the row.") == "the gradient bound. Also: the row."


def test_the_limit_is_a_power_of_ten():
    """It is a measured constant (see its comment): the largest power of
    ten at which every measured run held by a factor of two, on every
    static kind."""
    assert MAPPED_ROW_FLOOR_LIMIT >= 1
    assert 10 ** round(math.log10(MAPPED_ROW_FLOOR_LIMIT)) == MAPPED_ROW_FLOOR_LIMIT


# ---------------------------------------------------------------------------
# The table (slow): kinds, row lengths, fields, gains, schedules, norms, dtypes
# ---------------------------------------------------------------------------

def _cells(pairs, fields=("uniform", "ramp", "random"), gains=(0.9, 0.99)) -> list:
    return [Cell(pair, field, gain) for pair in pairs for field in fields for gain in gains]


_SCHEDULES = ("jacobi", "gauss-seidel")
#: Rows within the limit, of each kind: the flag is kept, on a bound
#: that holds by a factor of two.  A dense matrix of three rows of three
#: entries is nine wide, which is within the limit too.
WITHIN = _cells(
    [Pair("scatter", n, schedule, norm) for n in (3, 10)
     for schedule in _SCHEDULES for norm in ("interface", "mixed")]
    + [Pair(layout, n, schedule) for layout in ("gather", "dense") for n in (3, 10)
       for schedule in _SCHEDULES]
    + [Pair(layout, 10, "jacobi", "mixed") for layout in ("gather", "dense")]
    + [Pair("gather", 10, "jacobi", rows=3), Pair("dense", 3, "jacobi", rows=3),
       Pair("dense", 10, "jacobi", "interface", "float64")])
#: Longer rows at their float floor: withdrawn.  Both dtypes (float64 at
#: its own floor), the three norms, one row and three.  A dense matrix
#: of three rows of ten entries is thirty wide: over the limit by its
#: width, though each row holds ten non-zeros (see ``_longest_row``).
BEYOND = _cells(
    [Pair("scatter", n, schedule, norm) for n in (100, 3000, 30000)
     for schedule in _SCHEDULES for norm in ("interface", "mixed")]
    + [Pair("scatter", 3000, "jacobi", "l2"),
       Pair("scatter", 300, "jacobi", "interface", "float64"),
       Pair("scatter", 30000, "gauss-seidel", "mixed", "float64")]
    + [Pair(layout, n, schedule) for layout in ("gather", "dense") for n in (100, 3000)
       for schedule in _SCHEDULES]
    + [Pair(layout, 300, schedule, norm) for layout in ("gather", "dense")
       for schedule in _SCHEDULES for norm in ("interface", "mixed")]
    + [Pair("gather", 10000), Pair("gather", 30000, "gauss-seidel"),
       Pair("gather", 3000, "jacobi", "l2"), Pair("dense", 3000, "jacobi", "l2"),
       Pair("gather", 300, "jacobi", "interface", "float64"),
       Pair("dense", 300, "jacobi", "interface", "float64"),
       Pair("dense", 30000), Pair("dense", 30000, "gauss-seidel"),
       Pair("dense", 3000, "gauss-seidel", rows=3),
       Pair("dense", 3000, "jacobi", "interface", "float64", rows=3),
       Pair("dense", 300, "jacobi", "interface", "float64", rows=3),
       Pair("gather", 3000, "jacobi", rows=3),
       Pair("dense", 10, "jacobi", rows=3)])


# Per push: tests/core/test_the_float_floor_of_a_long_mapped_row.py::test_a_row_within_the_limit_keeps_its_flags_on_a_bound_that_holds
@pytest.mark.slow
@pytest.mark.parametrize("cell", WITHIN, ids=lambda cell: cell.id)
def test_a_row_within_the_limit_keeps_its_flags_on_a_bound_that_holds_by_two(cell):
    report = measure(cell)
    assert report["precision_limited"], (cell.id, report)
    assert report["spectral_usable"] and "not_usable_reason" not in report, (cell.id, report)
    assert report["spectral_error_bound"] >= 2.0 * report["distance"], (cell.id, report)
    _a_set_flag_stands_on_a_bound_at_or_above_the_distance(cell, report)


# Per push: tests/core/test_the_float_floor_of_a_long_mapped_row.py::test_a_stalled_group_behind_a_long_row_withdraws_its_flags_and_says_why
@pytest.mark.slow
@pytest.mark.parametrize("cell", BEYOND, ids=lambda cell: cell.id)
def test_a_set_flag_stands_on_a_bound_at_or_above_the_distance_behind_a_long_row(
        cell, monkeypatch):
    """Every such report at its float floor has its flags withdrawn for
    the row, whether its bound happened to hold or not."""
    report = measure(cell)
    assert report["precision_limited"], (cell.id, report)
    _withdrawn_for_the_row_where_a_flag_was_set(cell, report, monkeypatch)
    _a_set_flag_stands_on_a_bound_at_or_above_the_distance(cell, report)


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
    (Cell(Pair("gather", 30000, "jacobi", "interface", rtol=1e-5)), True),
    (Cell(Pair("dense", 30000, "jacobi", "interface", rtol=1e-5)), True),
    (Cell(Pair("dense", 30000, "gauss-seidel", "mixed", rtol=1e-5), "random"), True),
    (Cell(Pair("scatter", 100, "jacobi", "interface", rtol=1e-1)), False),
    (Cell(Pair("scatter", 100, "gauss-seidel", "interface", rtol=1e-1), "random"), False),
    (Cell(Pair("scatter", 100, "jacobi", "mixed", rtol=1e-1)), False),
    (Cell(Pair("gather", 100, "jacobi", "interface", rtol=1e-1)), False),
    (Cell(Pair("dense", 100, "jacobi", "interface", rtol=1e-1)), False),
    (Cell(Pair("dense", 100, "gauss-seidel", "mixed", rtol=1e-1), "random"), False),
]


# Per push: tests/core/test_the_float_floor_of_a_long_mapped_row.py::test_the_flags_are_withdrawn_only_behind_a_long_row_at_the_floor_it_would_give
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


# ---------------------------------------------------------------------------
# The price of counting a dense matrix at its width: a selection matrix
# ---------------------------------------------------------------------------

def _selection_report(sparse: bool, n: int = 40, gain: float = 0.9) -> dict:
    """Two fields of ``n`` values joined entry to entry, each way, by the
    nearest neighbour between the same points: a selection, one non-zero
    a row, held as a dense ``n x n`` matrix or as a sparse mapping.
    ``coarse.x[i] <- b + a * fine.x[i]`` and ``fine.x[i] <- c_i +
    coarse.x[i]``, stalled at float32 under ``"mixed"``; the report and
    its true distance."""
    dt, points, rtol = jnp.dtype("float32"), (np.arange(n) + 0.5) / n, 1e-7
    build = sparse_nearest_neighbor_mapping if sparse else nearest_neighbor_mapping
    gm = GraphManager()
    gm.add_node(Coarse("coarse", 1.0, n, dt))
    gm.add_node(Fine("fine", 1.0, n, dt))
    gm.add_external_input("coarse", "ab", shape=(2,), dtype=dt)
    gm.add_external_input("fine", "c", shape=(n,), dtype=dt)
    gm.add_edge("coarse", "fine", "x", "u", mapping=build(points, points))
    gm.add_edge("fine", "coarse", "x", "u", mapping=build(points, points))
    gm.add_coupling_group(["coarse", "fine"], convergence_norm="mixed", max_iterations=40000,
                          iteration_mode="jacobi", solver="ift", diagnostics=True, rtol=rtol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    c = constants(n, "ramp", np.float32)
    a, b = np.float32(gain), np.float32(1.0)
    gm.set_node_state("fine", {"x": jnp.asarray(c)})
    gm.step(external_inputs={"coarse": {"ab": jnp.asarray([a, b])}, "fine": {"c": jnp.asarray(c)}})
    report = dict(gm.coupling_diagnostics()[KEY])
    x = np.asarray(gm.get_node_state("coarse")["x"], np.float64)
    fine = np.asarray(gm.get_node_state("fine")["x"], np.float64)
    c64 = np.asarray(c, np.float64)
    x_star = (float(b) + float(a) * c64) / (1.0 - float(a))
    terms = np.concatenate([np.abs(x - x_star) / np.max(np.abs(x)),
                            np.abs(fine - (c64 + x_star)) / np.max(np.abs(fine))])
    report["distance"] = float(np.sqrt(np.mean(terms ** 2))) / rtol
    report["rows"] = [row for _key, _what, row in gm._committed_mapped_rows[KEY]]  # noqa: SLF001
    return report


# Per push: tests/core/test_the_float_floor_of_a_long_mapped_row.py::test_a_dense_row_is_the_matrix_width_whatever_its_weights
@pytest.mark.slow
def test_a_dense_selection_matrix_loses_its_flags_and_the_same_operator_held_sparse_keeps_them():
    """Part of the guard's price, stated: a dense matrix is counted at
    its width whatever its weights are, so a selection matrix 40 wide
    loses its flags at the float floor although each of its rows adds up
    one non-zero and its bound holds.  The sparse mapping of the same
    operator has rows of one entry, is counted at one, and keeps its
    flags on a bound that holds."""
    dense, sparse = _selection_report(sparse=False), _selection_report(sparse=True)
    for report in (dense, sparse):
        assert report["converged"] and report["precision_limited"], report
        assert report["spectral_error_bound"] >= report["distance"] > 0.0, report
    assert dense["rows"] == [40, 40] and sparse["rows"] == [1, 1]
    assert not dense["spectral_usable"] and not dense["gradient_bound_usable"], dense
    for said in ("a dense matrix mapping (nearest_neighbor), counted at the matrix's width, "
                 "whose longest row adds up 40 entries", ANOMALY, "a wider dtype"):
        assert said in dense["not_usable_reason"], (said, dense["not_usable_reason"])
    # (Forty loops that are one loop forty times: the gradient flag is
    # off in both forms, for a cause of its own, which the sparse form's
    # reason is the whole of.)
    assert sparse["spectral_usable"] and not sparse["gradient_bound_usable"], sparse
    assert sparse["reason_codes"]["spectral_usable"] == [], sparse["reason_codes"]
    assert LONG_ROW not in sparse["reason_codes"]["gradient_bound_usable"]
    assert ANOMALY not in sparse["not_usable_reason"], sparse["not_usable_reason"]


# ---------------------------------------------------------------------------
# Not counted: a geometry-dependent scatter (a characterisation)
# ---------------------------------------------------------------------------

class GridCoarse(SimulationNode):
    """Two grid nodes: ``x <- b + a * u``."""

    def initial_state(self):
        return {"x": jnp.zeros(2, jnp.float32)}

    def boundary_input_spec(self):
        zeros = jnp.zeros(2, jnp.float32)
        return {"u": BoundaryInputSpec(shape=(2,), dtype=jnp.float32, default=zeros),
                "ab": BoundaryInputSpec(shape=(2,), dtype=jnp.float32, default=zeros)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        ab = boundary_inputs["ab"]
        return {"x": ab[1] + ab[0] * boundary_inputs["u"]}

    def update_evaluations(self):
        return 1


class Markers(SimulationNode):
    """``k`` markers at one place, held still: ``x[j] <- c_j + u[j]``."""

    def __init__(self, name, timestep, k):
        super().__init__(name, timestep)
        self._k = k

    def initial_state(self):
        return {"x": jnp.zeros(self._k, jnp.float32),
                "pos": jnp.full((self._k, 1), 0.5, jnp.float32)}

    def boundary_input_spec(self):
        zeros = jnp.zeros(self._k, jnp.float32)
        return {"u": BoundaryInputSpec(shape=(self._k,), dtype=jnp.float32, default=zeros),
                "c": BoundaryInputSpec(shape=(self._k,), dtype=jnp.float32, default=zeros)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": boundary_inputs["c"] + boundary_inputs["u"], "pos": state["pos"]}

    def update_evaluations(self):
        return 1


# Per push: tests/core/test_the_float_floor_of_a_long_mapped_row.py::test_every_static_mapping_on_an_internal_edge_is_counted
@pytest.mark.slow
def test_a_geometry_dependent_scatter_is_not_counted_and_its_bound_holds_by_less_than_two():
    """A characterisation, not a promise.  A conservative
    ``multilinear_grid`` mapping from points to a grid is a scatter-add
    whose fan-in is the number of markers in a cell's support, which is
    decided in the step: no static number, so the guard does not count
    it.  With 3000 markers in one cell behind a uniform field (float32,
    ``"mixed"``, loop gain 0.99) each of the two grid nodes adds up 3000
    entries, and the bound read 1.27 times the distance (3.8 with 300
    markers, 13.8 with 8; jax 0.10.2 and 0.11.0 alike): it holds, and
    not by the factor of two the limit is taken at.  The distance pools
    the markers' positions, which do not move, as ``"mixed"`` reads
    every floating field of the members.

    **Its flags.**  The markers' positions are a member's own state, so
    this group solves positions, and in 0.4.0 such a group has no usable
    flag on any step, with that rule's reason (MADD-ANO-252): the row
    guard is not what takes them, and it is not asked.  A group whose
    positions are constants of the pass keeps its flags behind the same
    uncounted rows; that form was not measured.

    When this fails, the registry entry and the guide state other
    numbers than the tree gives."""
    k, gain, rtol = 3000, 0.99, 1e-7
    gm = GraphManager()
    gm.add_node(GridCoarse("coarse", 1.0))
    gm.add_node(Markers("fine", 1.0, k))
    gm.add_external_input("coarse", "ab", shape=(2,), dtype=jnp.float32)
    gm.add_external_input("fine", "c", shape=(k,), dtype=jnp.float32)
    gm.add_edge("coarse", "fine", "x", "u",
                mapping=matrix_mapping(jnp.full((k, 2), 0.5, jnp.float32)))
    gm.add_edge("fine", "coarse", "x", "u",
                mapping=multilinear_grid_mapping([0.0], [1.0], [2], n_points=k,
                                                 mode="conservative"),
                geometry=("source", "pos"))
    gm.add_coupling_group(["coarse", "fine"], convergence_norm="mixed", max_iterations=8000,
                          iteration_mode="jacobi", solver="ift", diagnostics=True, rtol=rtol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    c = np.full(k, 1.1, np.float32)
    a, b = np.float32(gain / (k / 2)), np.float32(1.0)
    gm.set_node_state("fine", {"x": jnp.asarray(c), "pos": jnp.full((k, 1), 0.5, jnp.float32)})
    gm.step(external_inputs={"coarse": {"ab": jnp.asarray([a, b])}, "fine": {"c": jnp.asarray(c)}})
    report = dict(gm.coupling_diagnostics()[KEY])
    # Only the dense edge back (two entries a row) is a static mapping.
    rows = gm._committed_mapped_rows[KEY]  # noqa: SLF001
    assert [(key, row) for key, _what, row in rows] == [(BACK_EDGE, 2)]
    assert report["converged"] and report["precision_limited"], report
    assert not report["spectral_usable"] and not report["gradient_bound_usable"], report
    assert "the group solves position(s)" in report["not_usable_reason"], report
    assert ANOMALY not in report["not_usable_reason"], report
    x = np.asarray(gm.get_node_state("coarse")["x"], np.float64)
    fine = np.asarray(gm.get_node_state("fine")["x"], np.float64)
    c64 = np.asarray(c, np.float64)
    x_star = (float(b) + float(a) * 0.5 * math.fsum(c64.tolist())) / (1.0 - float(a) * k / 2)
    terms = np.concatenate([np.abs(x - x_star) / np.max(np.abs(x)),
                            np.abs(fine - (c64 + x_star)) / np.max(np.abs(fine)),
                            np.zeros(k)])                # the positions: they do not move
    distance = float(np.sqrt(np.mean(terms ** 2))) / rtol
    assert distance > 0.0
    assert distance <= report["spectral_error_bound"] < 2.0 * distance, (report, distance)


# ---------------------------------------------------------------------------
# What the row rule does not cover, and how close a kept flag comes
# (characterisations: each pins a measured number, none is a claim)
# ---------------------------------------------------------------------------

#: A row of ten entries, within the limit, behind a field whose terms
#: cancel (``sum |t| / |sum t|`` of 25 at the fixed point), stalled at the
#: float64 floor: three rows in the gather layout under Jacobi.
CANCELLING = Cell(Pair("gather", MAPPED_ROW_FLOOR_LIMIT, dtype="float64", rows=3),
                  field="cancelling", gain=0.9)


# Per push: tests/core/test_the_float_floor_of_a_long_mapped_row.py::test_a_row_within_the_limit_keeps_its_flags_on_a_bound_that_holds
@pytest.mark.slow
def test_a_short_row_behind_a_cancelling_field_keeps_its_flag_on_a_bound_under_the_distance():
    """The premise of the strict xfail below, as what happens today
    (MADD-ANO-247's mechanism, reached through the field's signs): the
    row rule counts a row's LENGTH, the row is within the limit, and the
    flag stands on a bound of 0.85 of the distance (jax 0.10.2, 0.11.0
    and 0.11.2)."""
    report = measure(CANCELLING)
    assert report["converged"] and report["precision_limited"], report
    assert report["spectral_usable"] and "not_usable_reason" not in report, report
    assert 0.7 < report["spectral_error_bound"] / report["distance"] < 1.0, report


# Per push: tests/core/test_the_float_floor_of_a_long_mapped_row.py::test_a_row_within_the_limit_keeps_its_flags_on_a_bound_that_holds
@pytest.mark.slow
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "MADD-ANO-247 (open): the float floor takes a delivered value at its own magnitude, so "
    "behind a field whose terms cancel in a row the bound reads under the distance with the "
    "flag set whatever the row's length; the floor is fixed in 0.5.0"))
def test_the_bound_behind_a_short_row_of_a_cancelling_field_is_at_or_above_the_distance():
    """The number itself: what a floor that counts the cancellation
    within a row will make pass."""
    report = measure(CANCELLING)
    assert report["spectral_error_bound"] >= report["distance"], report


# Per push: tests/core/test_the_float_floor_of_a_long_mapped_row.py::test_the_flags_are_withdrawn_only_behind_a_long_row_at_the_floor_it_would_give
@pytest.mark.slow
def test_a_kept_flag_on_a_group_of_one_member_reads_just_under_its_distance():
    """How far under the distance a flag the rule KEEPS can read.  The
    measured pairs bottom at 0.9925 (two evaluations a pass); a group of
    ONE member with an edge to itself evaluates once a pass, so its
    threshold is half the pair's, and by construction (a report just
    above the threshold ``residual > floor * row``, a uniform field,
    gain 0.999, a scatter row of 1000 entries, ``"interface"``) its kept
    flag stands on a bound of 0.982 of the distance.  Pinned between 0.97
    and 0.9925: the rule is as it was, and this is its measured margin."""
    k, gain, value = 1000, 0.999, 1.1
    rtol = 1.1 * PRECISION_FLOOR_ULPS * float(np.finfo(np.float32).eps) * k
    weights = np.full((k, k), np.float32(gain / k), np.float32)
    gm = GraphManager()
    gm.add_node(Fine("s", 1.0, k, jnp.float32))
    gm.add_external_input("s", "c", shape=(k,), dtype=jnp.dtype("float32"))
    # The scatter layout of "gain / k times the sum, to every entry".
    gm.add_edge("s", "s", "x", "u", mapping=StaticSparseMapping(
        indices=np.tile(np.arange(k, dtype=np.int32), (k, 1)), weights=jnp.asarray(weights),
        n_source=k, layout="scatter", n_target=k))
    gm.add_coupling_group(["s"], convergence_norm="interface", max_iterations=60000,
                          solver="ift", diagnostics=True, rtol=rtol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
        c = np.full(k, value, np.float32)
        gm.set_node_state("s", {"x": jnp.asarray(c)})
        gm.step(external_inputs={"s": {"c": jnp.asarray(c)}})
    (report,) = gm.coupling_diagnostics().values()
    x = np.asarray(gm.get_node_state("s")["x"], np.float64)
    weight = float(np.float32(gain / k))
    total = math.fsum(np.asarray(c, np.float64).tolist()) / (1.0 - weight * k)
    # The one reading: what the edge delivers, ``weight * sum(x)`` to every entry.
    delivered = weight * math.fsum(x.tolist())
    distance = abs(delivered - weight * total) / abs(delivered) / rtol
    assert report["converged"] and not report["precision_limited"], report
    assert report["spectral_usable"] and report["gradient_bound_usable"], report
    assert "not_usable_reason" not in report, report
    assert 0.97 < report["spectral_error_bound"] / distance < 0.9925, (report, distance)


# Per push: tests/core/test_the_float_floor_of_a_long_mapped_row.py::test_the_flags_are_withdrawn_only_behind_a_long_row_at_the_floor_it_would_give
@pytest.mark.slow
def test_rho_spectral_behind_a_long_row_is_an_estimate_while_the_bound_holds():
    """Behind long mapped rows at a loose tolerance ``rho_spectral`` can
    be off by more than the flag's margin with both flags set (CPL-087's
    statement (1) is not claimed there): three scatter rows of 3333
    entries, float32, ``"interface"``, Gauss-Seidel, ``rtol = 0.1``, an
    exact radius of 0.9.  ``spectral_error_bound`` stays conservative."""
    pair = Pair("scatter", 3333, schedule="gauss-seidel", rtol=0.1, rows=3)
    gain, n = 0.9, pair.n
    with x64(False):
        gm = compiled(pair)
        c = np.random.default_rng(0).uniform(1.0, 2.0, pair.fine).astype(np.float32)
        a, b = np.float32(gain / n), np.float32(1.0)
        gm.reset_state()
        gm.step(external_inputs={"coarse": {"ab": jnp.asarray([a, b])},
                                 "fine": {"c": jnp.asarray(c)}})
        report = dict(gm.coupling_diagnostics()[KEY])
        x = np.asarray(gm.get_node_state("coarse")["x"], np.float64)
        fine = np.asarray(gm.get_node_state("fine")["x"], np.float64)
    c64 = np.asarray(c, np.float64)
    sums = np.array([math.fsum(c64[i * n:(i + 1) * n].tolist()) for i in range(pair.rows)])
    x_star = (1.0 + float(a) * sums) / (1.0 - float(a) * n)
    delivered = np.array([math.fsum(fine[i * n:(i + 1) * n].tolist()) for i in range(pair.rows)])
    errors = np.concatenate([np.abs(x - x_star) / np.max(np.abs(x)),
                             np.abs(delivered - (sums + n * x_star)) / np.max(np.abs(delivered))])
    distance = float(np.sqrt(np.mean(errors ** 2))) / pair.tolerance
    assert report["converged"], report
    assert report["spectral_usable"] and report["gradient_bound_usable"], report
    margin = SPECTRAL_SETTLED_FRACTION * (1.0 - report["rho_spectral"])
    assert abs(report["rho_spectral"] - gain) > margin, (report, margin)
    assert abs(report["rho_spectral"] - gain) < 0.05, report
    assert report["spectral_error_bound"] >= distance > 0.0, (report, distance)
