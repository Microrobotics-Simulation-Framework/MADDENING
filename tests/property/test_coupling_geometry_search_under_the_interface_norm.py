"""The geometry search's cells under ``convergence_norm="interface"``.

``test_coupling_geometry_search.py`` scores the bounds of a group with a
``multilinear_grid`` edge under ``"l2"`` and ``"mixed"`` against the
numerical float64 reference (``coupling_reference.PassReference``).  Under
``"interface"`` such a group now solves, and its report withholds every
bound (the next stage's).  What it does report is scored here, on cells of
the same pair (``geometry_cells``: a grid-side ``F``, a point-side ``P``,
positions that move with the iterate) whose geometry-dependent edges are
the gather ``F.x -> P.u``, the scatter ``P.x -> F.u``, or both (the other
direction is then a static matrix):

* **the reading** (``reading``): the reported residual is the exact
  residual ``P(x) - x`` of the iterate the loop accepted, in the rule's
  reading, written here as a function of the reference's flat iterate
  (:func:`reading_of`) -- a gather as delivered, at the positions the step
  uses; a scatter at its source value and, anchored there, its source's
  positions over the grid spacing of each axis, measured against one
  spacing.  Within the rounding of the graph's own dtype, stated
  independently (:func:`floor_of`);
* **the state returned** (``returned``): beside the graph observed at its
  accepted iterate (``coupled_graphs.accepted_iterate``), the same cell
  with the return rule on reports the same solve, returns every field the
  reading holds whole as that iterate holds it, to the bit, and every
  other as the reference's pass at it computes it.

The cells are this module's own table (:data:`CELLS`), appended to and
never rotated, with their configurations in :data:`INTERFACE_KNOBS`;
``test_coupling_search_cells_are_pinned.py`` pins each by digest.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import warnings

import numpy as np
import pytest

from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from tests.property import coupled_graphs as cg
from tests.property import coupling_reference as cr
from tests.property import geometry_cells as gc
from tests.property import test_coupling_targeted_search as linear
from tests.property.geometry_cells import Case
from tests.property.sysid_transform_grid import precision
from tests.property.targeted_search import PER_PUSH, SLOW, targeted_search

#: One ``solver="ift"`` configuration per row, FROZEN: a cell names its row
#: by index, and a new configuration is a new row at the end.
INTERFACE_KNOBS = (
    dict(acceleration="none", iteration_mode="gauss-seidel"),
    dict(acceleration="none", iteration_mode="jacobi"),
    dict(acceleration="aitken", iteration_mode="jacobi"),
    dict(acceleration="iqn-ils", iteration_mode="gauss-seidel"),
    dict(acceleration="fixed", relaxation=0.7, iteration_mode="gauss-seidel"),
    dict(acceleration="iqn-imvj", jacobian_reuse=2, iteration_mode="jacobi"),
)
EDGES = ("both", "gather", "scatter")


@dataclasses.dataclass(frozen=True)
class InterfaceCell(gc.Cell):
    """A cell of the geometry search under the interface norm."""

    #: Which direction carries a ``multilinear_grid`` edge: ``"both"``, the
    #: ``"gather"`` alone or the ``"scatter"`` alone (the other is a static
    #: matrix, read by the static rule).
    edges: str = "both"

    @property
    def knobs(self) -> dict:
        return cg.live_knobs(dict(
            INTERFACE_KNOBS[self.knob], solver="ift", max_iterations=self.cap,
            convergence_norm="interface", rtol=linear.RTOL))

    def __repr__(self) -> str:
        k = INTERFACE_KNOBS[self.knob]
        return (f"{self.edges} {'-'.join(self.anchors)} {self.dtype} {k['acceleration']} "
                f"{k['iteration_mode']} cap{self.cap} {self.d}d m{self.m} o{self.origin:g} "
                f"{''.join(self.order)}")


#: Appended to, never reordered (the digest guard holds each index).
CELLS = (
    InterfaceCell(("target", "source"), True, "float64", 0, 60),
    InterfaceCell(("source", "source"), True, "float32", 1, 60, d=2),
    InterfaceCell(("source", "target"), True, "float64", 2, 60, d=2, m=3, order=("P", "F")),
    InterfaceCell(("target", "target"), True, "float32", 3, 5),
    InterfaceCell(("source", "source"), True, "float64", 4, 60, edges="gather"),
    InterfaceCell(("target", "source"), True, "float32", 5, 5, d=2, edges="gather",
                  order=("P", "F")),
    InterfaceCell(("source", "source"), True, "float64", 1, 5, d=2, edges="scatter"),
    InterfaceCell(("source", "target"), True, "float32", 0, 60, edges="scatter"),
    InterfaceCell(("target", "source"), True, "float64", 3, 60, d=2, origin=40.0,
                  edges="scatter", order=("P", "F")),
)
PER_PUSH_CELLS = (0,)
#: The slow hunt's blocks: each cell compiles two graphs and a twin.
BLOCKS = tuple(tuple(range(len(CELLS)))[k::3] for k in range(3))
SCORES = ("reading", "returned")


def static_matrix(n_target: int, n_source: int) -> np.ndarray:
    """The fixed matrix of the direction that carries no geometry."""
    rng = np.random.default_rng(100 * n_target + n_source)
    return rng.uniform(0.2, 1.0, (n_target, n_source)) / math.sqrt(n_source)


def build(cell: InterfaceCell, knobs: dict, dtype: str) -> GraphManager:
    """A compiled graph of *cell* under *knobs* in *dtype*."""
    origin, spacing, shape = cell.grid
    sizes = {"F": cell.n_grid, "P": cell.m}
    gm = GraphManager()
    for name in cell.order:
        gm.add_node(gc.GeoRelay(name, gc.DT, n=sizes[name], dtype=dtype,
                                points=(cell.m, cell.d) if name in cell.holders else None))
    down, up = cell.anchors
    if cell.edges in ("both", "gather"):
        gm.add_edge("F", "P", "x", "u", geometry=(down, "pos"),
                    mapping=gc.multilinear_grid_mapping(origin, spacing, shape, n_points=cell.m,
                                                        mode="consistent"))
    else:
        gm.add_edge("F", "P", "x", "u", mapping=matrix_mapping(
            np.asarray(static_matrix(cell.m, cell.n_grid), dtype)))
    if cell.edges in ("both", "scatter"):
        gm.add_edge("P", "F", "x", "u", geometry=(up, "pos"),
                    mapping=gc.multilinear_grid_mapping(origin, spacing, shape, n_points=cell.m,
                                                        mode="conservative"))
    else:
        gm.add_edge("P", "F", "x", "u", mapping=matrix_mapping(
            np.asarray(static_matrix(cell.n_grid, cell.m), dtype)))
    gm.add_coupling_group(["F", "P"], **cg.live_knobs(knobs))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        gm.compile()
    return gm


def built(cell: InterfaceCell) -> tuple:
    """``(the graph observed at its accepted iterate, the same graph with
    the return rule on, its x64 twin, the twin's reference)``."""
    with precision(cell.dtype == "float64"):
        with cg.accepted_iterate() as asked:
            accepted = build(cell, cell.knobs, cell.dtype)
            values = gc.values_of(gc._PROBE, cell)                 # noqa: SLF001
            gc.set_initial(accepted, values)
            accepted.step(params=gc.params_for(accepted, values))
        assert asked, "the graph was traced with the return rule off"
        returned = build(cell, cell.knobs, cell.dtype)
    with cr.x64():
        twin = build(cell, cr.twin_knobs(cell.knobs), "float64")
        values = gc.values_of(gc._PROBE, cell)                     # noqa: SLF001
        gc.set_initial(twin, values)
        return accepted, returned, twin, cr.PassReference.of(
            twin, params=gc.params_for(twin, values))


# ---------------------------------------------------------------------------
# The rule, stated on the reference's flat iterate
# ---------------------------------------------------------------------------


def gathered(field, pos, grid: tuple):
    """The multilinear gather of the flat grid *field* at *pos* (NumPy or
    JAX; clamped to the lattice's hull), written from the kind's
    specification."""
    xp = np if isinstance(field, np.ndarray) else __import__("jax.numpy").numpy
    origin, spacing, shape = grid
    lower, frac = [], []
    for a, n in enumerate(shape):
        u = xp.clip((pos[:, a] - origin[a]) / spacing[a], 0.0, n - 1.0)
        base = xp.minimum(xp.floor(u), max(n - 2, 0))
        frac.append(u - base)
        lower.append(base.astype(int))
    strides = [int(np.prod(shape[a + 1:])) for a in range(len(shape))]
    out = 0.0
    for corner in itertools.product((0, 1), repeat=len(shape)):
        flat, w = 0, 1.0
        for a, n in enumerate(shape):
            flat = flat + strides[a] * (xp.minimum(lower[a] + 1, n - 1) if corner[a] else lower[a])
            w = w * (frac[a] if corner[a] else 1.0 - frac[a])
        out = out + w * field[flat]
    return out


def reading_of(cell: InterfaceCell, ref: cr.PassReference, values: dict) -> tuple:
    """``(fields, fixed, held, whole)``: the interface norm's reading of
    *cell* as a function of the flat iterate, what each field is measured
    against (``None``: its own magnitude; 1.0: one grid spacing), the
    dtype-resolution inputs of each (:func:`floor_of`), and the fields the
    reading holds whole."""
    origin, spacing, _shape = cell.grid
    down, up = cell.anchors
    pre_pos = {name: np.asarray(np.asarray(values[name]["pos"], cell.dtype), np.float64)
               for name in set(cell.holders)}
    spacing_a = np.asarray(spacing)
    geometry_gather = cell.edges in ("both", "gather")
    geometry_scatter = cell.edges in ("both", "scatter")
    whole = [("P", "x")] + ([("P", "pos")] if geometry_scatter and up == "source" else [])
    fixed = (None,) + ((None, 1.0) if len(whole) == 2 else (None,))
    #: Per field: the positions (in spacings) its rounding carries, or None.
    held = []

    def fields(x):
        out, held[:] = [], []
        f_x, p_x = ref.field(x, "F", "x"), ref.field(x, "P", "x")
        if geometry_gather:
            pos = ref.field(x, "F", "pos") if down == "source" else pre_pos["P"]
            out.append(gathered(f_x, pos, cell.grid))
            held.append(np.asarray(pos) / spacing_a)
        else:
            out.append(static_matrix(cell.m, cell.n_grid) @ f_x)
            held.append(None)
        out.append(p_x)
        held.append(None)
        if len(whole) == 2:
            out.append(ref.field(x, "P", "pos") / spacing_a)
            held.append(None)
        return out

    return fields, fixed, held, tuple(whole)


def floor_of(cell: InterfaceCell, parts: list, fixed: tuple, held: list) -> float:
    """Four units of each entry's resolution over what it is measured
    against, over ``rtol``, as a root mean square: ``eps`` for a value
    (and ``eps`` times the size in spacings of the positions it was
    gathered at, where that is larger), ``eps |u|`` for a position ``u``
    spacings from zero."""
    eps = float(np.finfo(cell.dtype).eps)
    total, count = 0.0, 0
    for value, unit, positions in zip(parts, fixed, held):
        value = np.asarray(value)
        if unit is not None:
            e = eps * float(np.max(np.abs(value)))
        elif positions is not None:
            e = eps * max(1.0, float(np.max(np.abs(positions))))
        else:
            e = eps
        total += value.size * (e / linear.RTOL) ** 2
        count += value.size
    return 4.0 * math.sqrt(total / count)


def observe(cell: InterfaceCell, case: Case, quad: tuple) -> dict:
    """One step of *case* on *cell* (``quad``: :func:`built`) and its scores."""
    accepted, returned, twin, ref = quad
    values = gc.values_of(case, cell)
    ref = gc.bound_reference(ref, twin, values)
    eps = float(np.finfo(cell.dtype).eps)
    with precision(cell.dtype == "float64"):
        pre, state, d, _meta = gc.run_once(accepted, values)
        _pre, after, d_returned, _meta = gc.run_once(returned, values)
    out = dict(reading=0.0, returned=0.0, scored=False, converged=bool(d["converged"]),
               report={k: d[k] for k in ("iterations", "converged", "residual")})
    assert "under convergence_norm='interface'" in d["not_usable_reason"], d
    assert not d["spectral_usable"] and math.isnan(d["spectral_error_bound"]), d
    for key in ("iterations", "converged"):
        assert d[key] == d_returned[key], (key, d, d_returned)
    assert (d["residual"] == d_returned["residual"]
            or (math.isnan(d["residual"]) and math.isnan(d_returned["residual"]))), (d, d_returned)
    finite = all(np.all(np.isfinite(f)) for s in (state, after) for n in s.values()
                 for f in n.values())
    if not finite or not math.isfinite(d["residual"]):
        return out
    x = ref.flat(state)
    passed = ref.apply(x)
    if not np.all(np.isfinite(passed)) or not gc.same_cells(cell, ref, x, passed):
        return out          # a point on a lattice plane between the two: the float32
        #                     pass and its float64 twin may stand on either side of it
    out["scored"] = True
    fields, fixed, held, whole = reading_of(cell, ref, values)
    norm = ref.norm("interface", linear.RTOL, fields=fields, fixed=fixed)
    true = ref.residual(x, norm)
    floor = floor_of(cell, fields(x), fixed, held)
    cancels = gc._cancellation(cell, values, pre, state)           # noqa: SLF001
    reported = float(d["residual"])
    slack = 2.0 ** 8 * eps * max(true, reported) + 4.0 * cancels * floor
    # A solve that stopped on its criterion reports the residual of the
    # iterate it accepted; one stopped at its cap reports the last residual
    # it measured, which the exact one of the iterate it holds may not
    # exceed by more than rounding.
    gap = abs(reported - true) if out["converged"] else max(0.0, true - reported)
    out["reading"] = gap / slack
    out["report"].update(residual_true=true, floor=floor, cancellation=cancels)

    worst = 0.0
    for name, field, _shape, a, b in ref.layout:
        if (name, field) in whole:
            assert np.array_equal(after[name][field], state[name][field]), (
                f"{name}.{field} is measured whole and was not returned as accepted")
            continue
        scale = max(float(np.max(np.abs(passed[a:b]))), float(np.max(np.abs(x[a:b]))))
        error = float(np.max(np.abs(np.ravel(np.asarray(after[name][field], np.float64))
                                    - passed[a:b])))
        worst = max(worst, error / (2.0 ** 6 * eps * cancels * max(scale, 1e-300)))
    out["returned"] = worst
    return out


class Search:
    """The two scores over :data:`CELLS`, each cell compiled once."""

    def __init__(self):
        self._built = functools.lru_cache(maxsize=4)(lambda index: built(CELLS[index]))
        self.seen: dict = {}

    def observe(self, case: Case) -> dict:
        if case not in self.seen:
            self.seen[case] = observe(CELLS[case.cell], case, self._built(case.cell))
        return self.seen[case]

    def run(self, name: str, indices, *, profile=None) -> tuple:
        drawn = []

        def score(case: Case):
            drawn.append(case)
            seen = self.observe(case)
            return seen[name], seen["report"]

        report = targeted_search(gc.cases(CELLS, indices), score, 1.0,
                                 profile=PER_PUSH if profile is None else profile, label=name)
        seen = [self.observe(c) for c in drawn]
        count = max(len(seen), 1)
        return report, {k: sum(bool(s[k]) for s in seen) / count for k in ("scored", "converged")}


SEARCH = Search()


def test_the_cells_hold_a_gather_a_scatter_and_both_at_each_anchor():
    """The premise, with no graph: every direction alone and both, each
    anchor of a gather and of a scatter, both dtypes, both schedules,
    every stock acceleration, one and two axes; and a cell whose scatter
    reads positions and one whose scatter reads none."""
    assert {c.edges for c in CELLS} == set(EDGES)
    gathers = {c.anchors[0] for c in CELLS if c.edges != "scatter"}
    scatters = {c.anchors[1] for c in CELLS if c.edges != "gather"}
    assert gathers == scatters == {"source", "target"}
    assert {c.dtype for c in CELLS} == {"float32", "float64"} and {c.d for c in CELLS} == {1, 2}
    knobs = [INTERFACE_KNOBS[c.knob] for c in CELLS]
    assert {k["acceleration"] for k in knobs} == {"none", "aitken", "fixed", "iqn-ils", "iqn-imvj"}
    assert {k["iteration_mode"] for k in knobs} == {"gauss-seidel", "jacobi"}
    assert all(c.knobs["convergence_norm"] == "interface" for c in CELLS)
    assert all(c.n_grid > c.m for c in CELLS), "the scatter delivers more entries than it reads"


@pytest.mark.parametrize("name", SCORES)
def test_what_a_group_with_a_geometry_edge_reports_under_the_interface_norm_per_push(name):
    """Slow sibling: :func:`test_the_hunt_under_the_interface_norm`."""
    report, fractions = SEARCH.run(name, PER_PUSH_CELLS)
    assert fractions["scored"] >= 0.5 and fractions["converged"] >= 0.25, fractions
    assert report.score <= 1.0, report


# Slow: 60 seeded examples a score on each of three blocks of three cells,
# each cell two compiled graphs and a twin.
# Per push: tests/property/test_coupling_geometry_search_under_the_interface_norm.py::test_what_a_group_with_a_geometry_edge_reports_under_the_interface_norm_per_push
@pytest.mark.slow
@pytest.mark.parametrize("block", range(len(BLOCKS)))
def test_the_hunt_under_the_interface_norm(block):
    search = Search()
    for name in SCORES:
        profile = dataclasses.replace(SLOW, max_examples=60).seeded(7 + 10 * block, shrink=False)
        report, fractions = search.run(name, BLOCKS[block], profile=profile)
        print(f"{name}, block {block}: worst {report.score:.4g}; {fractions}")
        assert fractions["scored"] >= 0.5, fractions
