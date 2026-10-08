"""What ``convergence_norm="interface"`` reads on an edge with a moving geometry.

The rule (``tests/property/geometry_interface_graphs.py`` states it, with
its reference): a gather is read as delivered, at the geometry the step
uses; a scatter at its inputs, the source value and -- for a source
anchor -- the positions **in grid spacings**; a target-anchored geometry
is the pre-step state and is no reading.  This module holds the library to
it on the ``multilinear_grid`` kind:

* **the reading, edge by edge, with no graph compiled**: the library's
  residual and float floor of two states, against the reference's, on
  every kind, both anchors of each edge and grids of one and two axes --
  and *not* the residual of any reading the rule is not (as delivered,
  without the positions, positions over their own magnitude, a
  target-anchored geometry counted);
* **the solve**: a plain iteration stops on the pass, with the residual
  and the state the reference's own loop gives; under every stock
  acceleration a group that reports ``converged`` is within ``K``
  tolerances of its fixed point in the compact readings;
* **two properties a wrong scale of the positions breaks**: the same
  problem 1000 spacings from the origin takes the same passes and reports
  the same residual, and ``K`` does not grow when the grid is refined
  under fixed markers;
* **the marker-side twin**: the same problem with the scatter inside the
  grid node and plain edges carrying the marker values and the positions
  in spacings reports the same numbers and returns the same state;
* what follows from the reading: the fields a solve keeps as it accepted
  them, who owns the ``reading_floor`` slot, checkpoints, a scan, a batch
  and a derivative through the solve; and what stays refused or withheld.

Seeded faults (``plans/tools/mutants.py`` on a scratch copy of ``src/``,
jaxlib 0.11.0; each run against five per-push instruments separately:
**pins** ``tests/core/test_interface_plan.py``, **edges** the edge-by-edge
tests here, **solve** the compiled tests here, **search**
``test_coupling_geometry_search_under_the_interface_norm.py`` (its one
per-push cell: both edges, the gather anchored at its target, one axis),
**frozen** the frozen identity of ``test_differential_geometry_edges.py``).
Fourteen faults, each caught by two instruments or more:

===  ================================================  ====  =====  =====  ======  ======
#    fault                                             pins  edges  solve  search  frozen
===  ================================================  ====  =====  =====  ======  ======
G1   a scatter read as delivered                       yes   yes    yes    yes     yes
G2   the positions left out of a scatter's reading     yes   yes    yes    yes     yes
G3   the positions over their own magnitude            yes   yes    yes    yes     -
G4   the spacing of the wrong axis                     yes   yes    yes    -       -
G5   a target-anchored geometry counted as a reading   yes   yes    yes    -       yes
G6   a target anchor read from the iterate             yes   yes    yes    yes     -
G7   a source anchor read from the pre-step state      -     yes    yes    -       -
G8   the positions' dtype and rounding left out of     -     yes    yes    -       -
     a delivered value's floor
G9   the positions' entries left out of the pool       -     yes    yes    yes     yes
G10  the kept fields not following the reading         yes   -      yes    yes     -
G11  the slot not recorded for a target anchor         yes   -      yes    -       -
G12  the report's fallback floor asked outside the     yes   -      yes    -       -
     step (it raises)
G13  a sub-cycled group not refused                    yes   -      yes    -       -
G14  a position's floor without its size in spacings   -     yes    yes    -       -
===  ================================================  ====  =====  =====  ======  ======

First signals: ``test_the_residual_of_two_states_is_the_rules`` (G1 to G7,
G9), ``test_the_float_floor_counts_each_part_at_its_own_resolution`` (G8,
G14), ``test_a_plain_iteration_stops_where_the_reference_s_loop_does``
(G1 to G7, G9, G10: the pass, the residual or the returned state),
``test_the_step_records_the_floor_of_a_reading_at_a_pre_step_geometry``
(G8, G11, G14), ``test_the_report_of_such_a_group_withholds_every_bound_and_says_why``
(G12), ``test_a_sub_cycled_group_with_a_geometry_edge_is_refused_under_the_interface_norm``
(G13).  The search's misses are its per-push cell's (one axis, a
source-anchored scatter and a target-anchored gather: G4, G5 and G7 are
not expressible on it; its slow hunt has the other cells).
"""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import (
    coupling_residual_interface,
    residual_precision_floor,
)
from maddening.core.graph_manager import GraphManager
from tests.property import geometry_interface_graphs as gi
from tests.property.geometry_interface_graphs import Draw, Reference, Shape
from tests.property.interface_side_graphs import ACCELERATIONS
from tests.property.sysid_transform_grid import precision

KEY = "p+q"
ANCHORS = (("source", "target"), ("target", "source"), ("source", "source"),
           ("target", "target"))

#: The compiled cells of every push: each kind, both schedules, both anchors
#: of a gather and of a scatter, one and two axes, float32 and float64.
PER_PUSH = (
    Shape("two-way", 120, 5, ("source", "target"), d=2),
    Shape("two-way", 120, 5, ("target", "source"), schedule="jacobi", dtype="float32"),
    Shape("gather-only", 120, 5, ("source", "target"), d=2),
    Shape("scatter-only", 120, 5, ("source", "target"), schedule="jacobi", d=2),
)
DRAWS = (Draw(3, 0.5), Draw(4, 0.3, sign=-1.0))

#: How closely a float64 graph reproduces the float64 reference.
TIGHT = 1e-8
#: The claim's allowance: ``K`` is the linearisation at the fixed point and
#: a converged iterate is ``1e-4`` of a reading from it, so the identity
#: holds to that order; a float32 state adds its own rounding of a position
#: (``eps |u| / rtol`` of a tolerance, a few hundredths here).
ALLOWED = {"float64": 1.02, "float32": 1.10}


def _id(shape) -> str:
    return (f"{shape.kind}-{shape.n_small}-{shape.n_large}-{'-'.join(shape.anchors)}-"
            f"{shape.schedule}-{shape.dtype}"
            + (f"-geom-{shape.geom_dtype}" if shape.geom_dtype else "")
            + f"-{shape.acceleration}-{shape.d}d"
            + (f"-o{shape.origin:g}" if shape.origin else ""))


# ---------------------------------------------------------------------------
# Premises: the reference's kernel, and cells that can tell the rules apart
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("d", [1, 2])
def test_the_references_kernel_is_the_independent_one(d):
    """The stencil restated here against the exact one of
    ``tests/core/multilinear_reference.py`` (rational arithmetic)."""
    from tests.core import multilinear_reference as mref  # noqa: PLC0415

    shape = Shape("two-way", 24, 4, d=d, origin=2.0)
    ref = Reference(shape, DRAWS[0])
    grid = mref.Grid(shape.grid_origin, shape.spacing, shape.grid_shape)
    pos = ref.pre["p"]["pos"]
    field = np.random.default_rng(0).uniform(-1.0, 1.0, shape.n_large)
    values = np.random.default_rng(1).uniform(-1.0, 1.0, shape.n_small)
    np.testing.assert_allclose(ref.gather(field, pos), mref.gather(grid, field, pos),
                               rtol=1e-11, atol=1e-14)
    np.testing.assert_allclose(ref.scatter(values, pos), mref.scatter(grid, values, pos),
                               rtol=1e-11, atol=1e-14)


def _cells(dtype="float64", geom_dtype=None):
    return [Shape(kind, 24 * (3 if d == 2 else 1), 4, anchors, d=d, dtype=dtype,
                  geom_dtype=geom_dtype, origin=3.0)
            for kind in gi.KINDS for anchors in ANCHORS for d in (1, 2)]


#: The readings the rule is not, and the cells on which each is another
#: number than the rule's (:meth:`Reference.parts`).
OTHER_RULES = {
    "delivered": lambda s: "scatter" in gi.WAYS[s.kind],
    "no-positions": lambda s: _source_anchored_scatter(s),
    "own-magnitude": lambda s: _source_anchored_scatter(s),
    "anchored-anywhere": lambda s: any(
        way == "scatter" and anchor == "target"
        for way, anchor in zip(gi.WAYS[s.kind], s.anchors)),
}


def _source_anchored_scatter(shape) -> bool:
    return any(way == "scatter" and anchor == "source"
               for way, anchor in zip(gi.WAYS[shape.kind], shape.anchors))


def _two_states(ref: Reference):
    """Two successive iterates in which every field moved, positions too."""
    old = ref.one_pass(ref.pre)
    return ref.one_pass(old), old


def _as_jax(shape, x: dict) -> dict:
    return {n: {"x": jnp.asarray(x[n]["x"], shape.dtype),
                "pos": jnp.asarray(x[n]["pos"], shape.geometry_dtype)} for n in x}


def _edges(shape):
    gm = GraphManager()
    for node in gi._nodes(shape).values():                # noqa: SLF001
        gm.add_node(node)
    for (src, dst), way, anchor in zip(gi.EDGES, gi.WAYS[shape.kind], shape.anchors):
        gm.add_edge(src, dst, "x", "u", mapping=gi._mapping(shape, way),   # noqa: SLF001
                    geometry=(anchor, "pos"))
    return list(gm._edges)                                # noqa: SLF001


# ---------------------------------------------------------------------------
# The reading, edge by edge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", _cells(), ids=_id)
def test_the_residual_of_two_states_is_the_rules(shape):
    """``coupling_residual_interface`` over the bare edges, against the
    reference's residual of the same two states: the parts, what each is
    measured against, the time level of each geometry and the pooled
    count.  And not the residual of any other reading."""
    ref = Reference(shape, DRAWS[0])
    new, old = _two_states(ref)
    assert np.all(new["p"]["pos"] != ref.pre["p"]["pos"]), "premise: the positions moved"
    with precision(True):
        got = float(coupling_residual_interface(
            _as_jax(shape, new), _as_jax(shape, old), _edges(shape), 0.0, gi.RTOL,
            pre_step=_as_jax(shape, ref.pre)))
    want = ref.residual(new, old)
    assert want > 1.0 and abs(got - want) <= 1e-9 * want, (got, want)
    for rule, applies in OTHER_RULES.items():
        other = ref.residual(new, old, rule)
        if applies(shape):
            assert abs(got - other) > 1e-4 * got, (
                f"{rule}: the cell cannot tell it from the rule ({got!r}, {other!r})")
        else:
            assert abs(other - want) <= 1e-12 * want, (rule, other, want)


def test_the_cells_tell_every_other_reading_apart_and_each_time_level():
    """The premise of the comparison above: each reading the rule is not is
    another number on some cell, and a geometry read at the other time
    level is too (the reference with its anchors' states swapped)."""
    cells = _cells()
    for rule, applies in OTHER_RULES.items():
        assert sum(applies(s) for s in cells) >= 6, rule
    moved = 0
    for shape in cells:
        ref = Reference(shape, DRAWS[0])
        new, old = _two_states(ref)
        want = ref.residual(new, old)
        # The other time level: the pre-step positions a target anchor reads
        # replaced by the iterate's, and the reverse.
        other = Reference(shape, DRAWS[0])
        other.geometry = lambda i, source, _r=ref: (          # type: ignore[method-assign]
            _r.pre[gi.EDGES[i][0]]["pos"] if shape.anchors[i] == "source"
            else new[gi.EDGES[i][1]]["pos"])
        has_gather = "gather" in gi.WAYS[shape.kind]
        if has_gather:
            assert abs(other.residual(new, old) - want) > 1e-4 * want, shape
            moved += 1
    assert moved >= 12


@pytest.mark.parametrize(
    "shape", [s for s in _cells() if s.d == 2 and _source_anchored_scatter(s)], ids=_id)
def test_a_position_is_measured_in_the_spacing_of_its_own_axis(shape):
    """The premise of the two-axis cells above: with spacings 4 to 1, the
    residual with each coordinate over the *other* axis' spacing is
    another number, so the comparison there holds the axis."""
    ref = Reference(shape, DRAWS[0])
    new, old = _two_states(ref)
    wrong = Reference.residual(_Readings(ref, ref.h[::-1].copy()), new, old)
    assert abs(wrong - ref.residual(new, old)) > 1e-2 * wrong


class _Readings:
    """*ref* reading its positions in the spacings *h* (a wrong axis' for the premise)."""

    def __init__(self, ref, h):
        self._ref, self._h = ref, h

    def parts(self, x, rule="decision"):
        true_h = self._ref.h
        return [(i, v * true_h / self._h if unit == gi.SPACINGS else v, unit)
                for i, v, unit in self._ref.parts(x, rule)]


@pytest.mark.parametrize("dtype, geom_dtype", [("float64", "float64"), ("float32", "float32"),
                                               ("float64", "float32"), ("float32", "float64")])
@pytest.mark.parametrize("kind", gi.KINDS)
@pytest.mark.parametrize("anchors", ANCHORS[:2])
def test_the_float_floor_counts_each_part_at_its_own_resolution(dtype, geom_dtype, kind, anchors):
    """``residual_precision_floor`` over the bare edges: a value at the eps
    of its dtype, a delivered one no finer than the positions it was
    gathered at (their dtype, and their rounding ``eps |u|`` in spacings),
    and a position at ``eps |u|`` of one spacing."""
    shape = Shape(kind, 72, 4, anchors, d=2, dtype=dtype, geom_dtype=geom_dtype, origin=40.0)
    ref = Reference(shape, DRAWS[0])
    state = ref.one_pass(ref.pre)
    with precision(True):
        got = float(residual_precision_floor(
            _as_jax(shape, state), ["p", "q"], "interface", 0.0, gi.RTOL, _edges(shape),
            pre_step=_as_jax(shape, ref.pre)))
    want = ref.floor(state)
    assert abs(got - want) <= 1e-5 * want, (got, want)
    if dtype != geom_dtype and ("gather" in gi.WAYS[kind] or _source_anchored_scatter(shape)):
        # Premise: with the positions' dtype left out the floor is another number.
        same = Reference(dataclasses.replace(shape, geom_dtype=None), DRAWS[0]).floor(state)
        assert abs(same - want) > 0.2 * min(same, want), (same, want)


def test_a_non_finite_position_fails_the_criterion_and_a_dead_band_does_not_hide_one():
    """A position is always read: an ``atol`` above every magnitude removes
    the values and leaves the positions' change; a NaN position reads
    ``inf``."""
    shape = Shape("scatter-only", 24, 4, ("source", "source"))
    ref = Reference(shape, DRAWS[0])
    new, old = _two_states(ref)
    with precision(True):
        edges = _edges(shape)
        banded = float(coupling_residual_interface(
            _as_jax(shape, new), _as_jax(shape, old), edges, 1e30, gi.RTOL))
        positions = [np.asarray(v) for _i, v, unit in ref.parts(new) if unit == gi.SPACINGS]
        before = [np.asarray(v) for _i, v, unit in ref.parts(old) if unit == gi.SPACINGS]
        want = math.sqrt(sum(float(np.sum(((a - b) / gi.RTOL) ** 2))
                             for a, b in zip(positions, before))
                         / sum(a.size for a in positions))
        assert abs(banded - want) <= 1e-9 * want and want > 0, (banded, want)
        broken = _as_jax(shape, new)
        broken["p"]["pos"] = broken["p"]["pos"].at[0, 0].set(jnp.nan)
        assert not np.isfinite(float(coupling_residual_interface(
            broken, _as_jax(shape, old), edges, 0.0, gi.RTOL)))


# ---------------------------------------------------------------------------
# The solve
# ---------------------------------------------------------------------------


def _check_plain_loop(shape, draw):
    out = gi.run(shape, draw)
    ref = out["reference"]
    plain = ref.plain_exit()
    exact = shape.dtype == "float64" and shape.geometry_dtype == "float64"
    assert plain["converged"] and out["converged"], (plain["iterations"], out["report"])
    if plain["margin"] > (1.0 + 1e-6 if exact else 1.25):
        assert out["iterations"] == plain["iterations"], (out["iterations"], plain)
    else:
        assert abs(out["iterations"] - plain["iterations"]) <= 1, (out["iterations"], plain)
    if out["iterations"] == plain["iterations"]:
        allowed = TIGHT * plain["residual"] if exact else (
            0.05 * plain["residual"] + 0.5 * ref.floor(plain["state"]))
        assert abs(out["residual"] - plain["residual"]) <= allowed, (out["residual"], plain)
        want = ref.returned(plain["state"], out["whole"])
        after = ref.one_pass(plain["state"])
        for name in ("p", "q"):
            for field in ("x", "pos"):
                # A value over its own size; a position in spacings.
                scale = (min(shape.spacing) if field == "pos"
                         else float(np.max(np.abs(want[name][field]))))
                allowed = (TIGHT if exact else 2e-5) * scale
                np.testing.assert_allclose(
                    out["state"][name][field], want[name][field], rtol=0, atol=allowed,
                    err_msg=f"{name}.{field}")
                if exact and ref.pull:
                    # The accepted iterate and one pass on differ by far more
                    # than the comparison allows, so a field on the wrong
                    # side of the return rule is seen.
                    gap = float(np.max(np.abs(after[name][field] - plain["state"][name][field])))
                    assert gap > 100 * allowed, (name, field, gap, allowed)
    assert out["excess"] <= ALLOWED[shape.dtype], (out["distance"], out["K"])
    return out


@pytest.mark.parametrize("shape", PER_PUSH, ids=_id)
def test_a_plain_iteration_stops_where_the_reference_s_loop_does(shape):
    """Per push; slow sibling: :func:`test_the_plain_loop_on_every_kind_anchor_and_schedule`."""
    for draw in DRAWS:
        _check_plain_loop(shape, draw)


@pytest.mark.parametrize("shape", [
    dataclasses.replace(PER_PUSH[0], acceleration="aitken"),
    dataclasses.replace(PER_PUSH[3], acceleration="iqn-ils"),
], ids=_id)
def test_a_converged_group_is_within_K_tolerances_in_the_compact_readings(shape):
    """Per push; slow sibling: :func:`test_the_claim_at_every_size_and_acceleration`."""
    for draw in DRAWS:
        out = gi.run(shape, draw)
        assert out["converged"], out["report"]
        assert out["excess"] <= ALLOWED[shape.dtype], (out["distance"], out["K"], out["report"])


def test_the_same_problem_a_thousand_spacings_from_the_origin_reports_the_same():
    """Translate the grid and the markers together: the passes, the verdict,
    the residual and the returned values do not change, to the rounding of
    a position a thousand spacings out, and neither does ``K``."""
    here, there = PER_PUSH[0], dataclasses.replace(PER_PUSH[0], origin=1000.0)
    for draw in DRAWS:
        a, b = gi.run(here, draw), gi.run(there, draw)
        assert a["converged"] and b["converged"]
        assert a["iterations"] == b["iterations"], (a["iterations"], b["iterations"])
        assert abs(a["residual"] - b["residual"]) <= 1e-6 * a["residual"], (
            a["residual"], b["residual"])
        assert abs(a["K"] - b["K"]) <= 1e-4 * a["K"], (a["K"], b["K"])
        assert abs(a["distance"] - b["distance"]) <= 1e-4 * a["distance"] + 1e-6
        shift = 1000.0 * np.asarray(here.spacing)
        for name in ("p", "q"):
            np.testing.assert_allclose(a["state"][name]["x"], b["state"][name]["x"],
                                       rtol=1e-9, atol=0)
            np.testing.assert_allclose(a["state"][name]["pos"] + shift,
                                       b["state"][name]["pos"], rtol=0, atol=1e-9 * shift[0])
        # Premise: measured over their own magnitude, the positions of the
        # two problems would not be read alike (26 times less, a thousand
        # spacings out), and in spacings they are.
        shares = {rule: [_positions_share(out["reference"], rule) for out in (a, b)]
                  for rule in ("decision", "own-magnitude")}
        assert shares["own-magnitude"][0] > 100 * shares["own-magnitude"][1]
        assert abs(shares["decision"][0] - shares["decision"][1]) <= 1e-6 * shares["decision"][0]


def _positions_share(ref: Reference, rule: str) -> float:
    """The positions' sum of squares in the residual of two iterates under *rule*."""
    new, old = _two_states(ref)
    total = 0.0
    for (_i, a, unit), (_j, b, _unit) in zip(ref.parts(new, rule), ref.parts(old, rule)):
        if a.ndim == 2:
            scale = 1.0 if unit == gi.SPACINGS else max(float(np.max(np.abs(a))),
                                                        float(np.max(np.abs(b))))
            total += float(np.sum(((a - b) / (gi.RTOL * scale)) ** 2))
    assert total > 0.0
    return total


def _refined(shape, sizes, draw):
    rows = []
    for n in sizes:
        out = gi.run(dataclasses.replace(shape, n_large=n), draw)
        assert out["converged"], (n, out["report"])
        assert out["excess"] <= ALLOWED[shape.dtype], (n, out["distance"], out["K"])
        rows.append((n, out["K"], out["distance"], out["iterations"]))
    smallest = rows[0][1]
    for n, K, distance, _iterations in rows:
        assert K <= 3.0 * smallest, f"K grew with the grid: {rows}"
        assert distance <= 3.0 * smallest * ALLOWED[shape.dtype], rows
    return rows


def test_refining_the_grid_under_fixed_markers_does_not_grow_K():
    """Per push: two sizes a hundredfold apart; slow sibling:
    :func:`test_K_does_not_grow_over_four_decades_of_grid`."""
    _refined(PER_PUSH[0], (120, 12_000), DRAWS[0])


# ---------------------------------------------------------------------------
# The marker-side twin
# ---------------------------------------------------------------------------


def _check_twin(shape, draw):
    ref = Reference(shape, draw, pin_first=True)
    with precision(shape.needs_x64):
        edge_mapped = gi.build_pinned(shape)
        twin = gi.marker_side_twin(shape, ref.pre["p"]["pos"])
    a = gi.run(shape, draw, graph=edge_mapped, reference=ref)
    with precision(shape.needs_x64):
        ref.start(twin.gm)
        twin.gm.step(params=ref.params(twin.gm))
        report = dict(twin.gm.coupling_diagnostics()[KEY])
        state = gi.state_of(twin.gm)
    assert a["iterations"] > 2 and a["converged"] and bool(report["converged"])
    assert int(report["iterations"]) == a["iterations"], (report, a["report"])
    tight = 1e-9 if shape.dtype == "float64" else 1e-3
    assert abs(float(report["residual"]) - a["residual"]) <= tight * a["residual"], (
        report["residual"], a["residual"])
    for name in ("p", "q"):
        for field in ("x", "pos"):
            scale = (min(shape.spacing) if field == "pos"
                     else float(np.max(np.abs(state[name][field]))))
            if (name, field) == ("p", "pos") and shape.anchors[0] == "source":
                # The one field the two graphs return differently, and
                # rightly: the edge-mapped graph's norm measures the
                # positions whole and keeps the accepted iterate's; the twin
                # carries them through a transform, which the return rule
                # cannot know to be one-to-one, and returns them one pass on.
                # The two are within the last pass's step, a tolerance.
                assert np.max(np.abs(a["state"][name][field] - state[name][field])) <= (
                    10 * gi.RTOL * scale)
                continue
            np.testing.assert_allclose(a["state"][name][field], state[name][field], rtol=0,
                                       atol=tight * scale, err_msg=f"{name}.{field}")
    # Premise: the twin's position edge is measured against one spacing.
    carried = (ref.pre["p"]["pos"] - gi.twin_reference(shape, ref.pre["p"]["pos"])) / ref.h
    if shape.anchors[0] == "source":
        assert np.max(np.abs(carried)) == pytest.approx(1.0, abs=1e-12)
        assert np.all(carried[0] == pytest.approx(1.0, abs=1e-12)) and np.all(carried[1:] == 0.0)
        assert np.all(a["state"]["p"]["pos"][0] == ref.pre["p"]["pos"][0]), "the pinned marker"


def test_an_edge_mapped_scatter_reports_what_its_marker_side_twin_reports():
    """Per push: the scatter anchored at its source (the twin carries the
    positions in spacings on a plain edge).  Slow sibling:
    :func:`test_the_marker_side_twin_on_both_anchors_and_dtypes`."""
    _check_twin(PER_PUSH[0], DRAWS[0])


# ---------------------------------------------------------------------------
# What follows from the reading
# ---------------------------------------------------------------------------

MIXED = Shape("two-way", 120, 5, ("source", "target"), d=2, dtype="float64",
              geom_dtype="float32")


def test_the_kept_fields_are_the_ones_a_part_holds_whole():
    """The independent statement of the set against the library's plan, on
    every compiled cell."""
    for shape in (*PER_PUSH, MIXED):
        with precision(shape.needs_x64):
            gm = gi.built(shape).gm
            from maddening.core.coupling import _interface_plan  # noqa: PLC0415
            plan = _interface_plan.interface_plan(
                frozenset("pq"), gm._edges, ["p", "q"], gm._state, None)    # noqa: SLF001
        assert set(plan.measured_whole()) == set(gi.measured_whole(shape)), shape


def test_the_step_records_the_floor_of_a_reading_at_a_pre_step_geometry():
    """A gather anchored at its target is read at the target's pre-step
    positions, which the returned state does not hold: the step records
    the floor (per evaluation), with the positions' dtype in it.  A group
    with no such edge owns no slot."""
    slot = f"coupling_{KEY}_reading_floor"
    out = gi.run(MIXED, DRAWS[0])
    assert slot in out["meta"], sorted(out["meta"])
    ref = out["reference"]
    want = ref.floor(out["state"])
    assert abs(float(out["meta"][slot]) - want) <= 1e-4 * want, (out["meta"][slot], want)
    finer = Reference(dataclasses.replace(MIXED, geom_dtype=None), DRAWS[0]).floor(out["state"])
    assert want > 1e6 * finer, "premise: the float32 positions set the floor"
    assert out["excess"] <= ALLOWED["float32"], (out["distance"], out["K"])
    without = gi.run(PER_PUSH[1], DRAWS[0])      # scatter at its target, gather at its source
    assert slot not in without["meta"], sorted(without["meta"])
    with_one = gi.run(PER_PUSH[0], DRAWS[0])
    assert np.isfinite(with_one["meta"][slot])


def _check_report_is_withheld(shape, tmp_path):
    from tests.property import geometry_graphs as gg  # noqa: PLC0415

    ref = Reference(shape, DRAWS[0])
    keys = ["p.x->q.u", "q.x->p.u"]
    with precision(True):
        gm = gi.built(shape).gm
        ref.start(gm)
        before = dict(gm.coupling_diagnostics().get(KEY, {}))
        gm.step(params=ref.params(gm))
        report = dict(gm.coupling_diagnostics()[KEY])
        gg.assert_not_diagnosed(report, keys, "norm")
        stepped = gi.state_of(gm)
        archive = str(tmp_path / "stepped.npz")
        gm.save_state(archive)
        name = f"coupling_{KEY}_reading_floor"
        owns = name in gm._state["_meta"]                                         # noqa: SLF001
        assert owns == any(way == "gather" and anchor == "target"
                           for way, anchor in zip(gi.WAYS[shape.kind], shape.anchors))
        slot = np.asarray(gm._state["_meta"][name]) if owns else None             # noqa: SLF001
        ref.start(gm)
        gm.load_state(archive)
        if owns:
            assert np.array_equal(np.asarray(gm._state["_meta"][name]), slot)     # noqa: SLF001
        gg.assert_not_diagnosed(dict(gm.coupling_diagnostics()[KEY]), keys, "norm")
        for name in ("p", "q"):
            for field in ("x", "pos"):
                assert np.array_equal(gi.state_of(gm)[name][field], stepped[name][field])
        if owns:
            # A state whose slot no step wrote: nothing to fall back to, and no raise.
            meta = dict(gm._state["_meta"])                                       # noqa: SLF001
            meta[name] = jnp.asarray(jnp.nan, slot.dtype)
            gm._state = {**gm._state, "_meta": meta}                               # noqa: SLF001
            gg.assert_not_diagnosed(dict(gm.coupling_diagnostics()[KEY]), keys, "norm")
    assert before == {} or "not_usable_reason" in before
    return stepped


def test_the_report_of_such_a_group_withholds_every_bound_and_says_why(tmp_path):
    """The bounds under this norm are a later stage's: the report carries
    the solve's own outcome, no bound, no usable flag and the reason,
    before a step, after one, and from a checkpoint (which brings the
    recorded floor and the state back to the bit) -- and it never raises,
    though the floor cannot be measured outside the step.  Slow sibling,
    with the step's own analysis traced:
    :func:`test_the_steps_analysis_traces_over_such_a_group_and_moves_no_state`."""
    _check_report_is_withheld(MIXED, tmp_path)


# Per push: tests/property/test_coupling_geometry_interface_norm.py::test_the_report_of_such_a_group_withholds_every_bound_and_says_why
@pytest.mark.slow
@pytest.mark.parametrize("shape", [MIXED, PER_PUSH[3], PER_PUSH[2]], ids=_id)
def test_the_steps_analysis_traces_over_such_a_group_and_moves_no_state(shape, tmp_path):
    """``diagnostics=True``: the step traces its spectral analysis over a
    reading of several parts (what it computes is not reported: the bounds
    under this norm are the next stage's), and the state it returns is the
    one the same group returns without it."""
    with precision(True):
        diagnosed = _check_report_is_withheld(
            dataclasses.replace(shape, diagnostics=True), tmp_path)
    plain = gi.run(shape, DRAWS[0])["state"]
    for name in ("p", "q"):
        for field in ("x", "pos"):
            np.testing.assert_allclose(diagnosed[name][field], plain[name][field],
                                       rtol=1e-12, atol=0, err_msg=f"{name}.{field}")


def test_the_fallback_floor_says_it_cannot_be_served_where_only_the_step_could_measure_it():
    """The same question without the geometry mask over it: the layout's
    own answer for the bare edges a report keeps."""
    from maddening.core.coupling import _group_layout  # noqa: PLC0415

    with precision(True):
        gm = gi.built(MIXED).gm
        (group,) = gm._coupling_groups                                           # noqa: SLF001
        _ev, _declared, edges = gm._committed_floor_inputs[KEY]                  # noqa: SLF001
        assert _group_layout._floor_needs_the_step(group, edges)
        ref = Reference(MIXED, DRAWS[0])
        with pytest.raises(ValueError, match="only the step that solved the group holds"):
            residual_precision_floor(_as_jax(MIXED, ref.pre), ["p", "q"], "interface", 0.0,
                                     gi.RTOL, list(edges))
    assert "only the step that solved the group holds" in (
        _group_layout._FLOOR_NEEDS_THE_STEP_REASON)


def _functional(shape, ref):
    gm = gi.built(shape).gm
    ref.start(gm)
    return gm._raw_step_fn, gm._state, gm._default_external_inputs(), ref.params(gm)   # noqa: SLF001


def _with_gain(params, a):
    nodes = {k: dict(v) for k, v in params["nodes"].items()}
    nodes["p"]["a"] = a
    return {**params, "nodes": nodes}


def _fixed_point_derivative(shape, draw, ref):
    """``d/da`` of the fixed point's marker values and positions (in
    spacings), by a central difference of the reference."""
    def fixed(a):
        r = Reference(shape, draw)
        r.a = {**r.a, "p": a}
        x = r.fixed_point
        return float(np.sum(x["p"]["x"]) + np.sum(x["p"]["pos"] / r.h))

    step = 1e-5 * abs(ref.a["p"])
    return (fixed(ref.a["p"] + step) - fixed(ref.a["p"] - step)) / (2.0 * step)


def _loss(shape, step, state0, ext, params):
    def loss(a):
        out = step(state0, ext, _with_gain(params, a))
        return jnp.sum(out["p"]["x"]) + jnp.sum(out["p"]["pos"] / jnp.asarray(shape.spacing))
    return loss


def test_a_derivative_goes_through_the_solve():
    """The derivative of the returned marker values and positions with
    respect to a gain is the fixed point's (the reference's, by a central
    difference), to the tolerance the solve stopped at; forward mode here
    (the cheaper program).  Slow sibling, with the reverse one:
    :func:`test_a_scan_a_batch_and_a_forward_derivative_go_through_the_solve`."""
    shape, draw = PER_PUSH[0], DRAWS[0]
    ref = Reference(shape, draw)
    with precision(True):
        step, state0, ext, params = _functional(shape, ref)
        a0 = jnp.asarray(ref.a["p"])
        forward = float(jax.jvp(_loss(shape, step, state0, ext, params), (a0,),
                                (jnp.ones_like(a0),))[1])
    want = _fixed_point_derivative(shape, draw, ref)
    assert abs(want) > 1.0 and abs(forward - want) <= 5e-3 * abs(want), (forward, want)


# Per push: tests/property/test_coupling_geometry_interface_norm.py::test_a_derivative_goes_through_the_solve
@pytest.mark.slow
@pytest.mark.parametrize("shape", [PER_PUSH[0], PER_PUSH[3], PER_PUSH[2]], ids=_id)
def test_a_scan_a_batch_and_a_forward_derivative_go_through_the_solve(shape):
    """``run_scan`` is the steps one by one; a ``vmap`` over a parameter is
    each member's own step; and the forward derivative is the reverse
    one."""
    draw = DRAWS[0]
    ref = Reference(shape, draw)
    with precision(True):
        step, state0, ext, params = _functional(shape, ref)
        gm = gi.built(shape).gm

        ref.start(gm)
        one = [gm.step(params=params) for _ in range(3)][-1]
        one = {n: {f: np.asarray(v) for f, v in one[n].items()} for n in ("p", "q")}
        scanned = jax.lax.scan(lambda carry, _: (step(carry, ext, params), None),
                               state0, None, length=3)[0]
        ref.start(gm)
        gm.run_scan(3, params=params)
        run = gi.state_of(gm)
        for name in ("p", "q"):
            for field in ("x", "pos"):
                np.testing.assert_allclose(np.asarray(scanned[name][field]), one[name][field],
                                           rtol=1e-12, atol=0)
                np.testing.assert_allclose(run[name][field], one[name][field],
                                           rtol=1e-12, atol=0)

        gains = jnp.asarray([ref.a["p"], 0.8 * ref.a["p"]])
        batch = jax.vmap(lambda a: step(state0, ext, _with_gain(params, a)))(gains)
        for k in range(2):
            alone = step(state0, ext, _with_gain(params, gains[k]))
            assert int(batch["_meta"][f"coupling_{KEY}_iterations"][k]) == int(
                alone["_meta"][f"coupling_{KEY}_iterations"])
            for name in ("p", "q"):
                for field in ("x", "pos"):
                    np.testing.assert_allclose(np.asarray(batch[name][field][k]),
                                               np.asarray(alone[name][field]), rtol=1e-12, atol=0)

        loss = _loss(shape, step, state0, ext, params)
        a0 = jnp.asarray(ref.a["p"])
        reverse = float(jax.grad(loss)(a0))
        forward = float(jax.jvp(loss, (a0,), (jnp.ones_like(a0),))[1])
    want = _fixed_point_derivative(shape, draw, ref)
    assert abs(reverse - want) <= 5e-3 * abs(want), (reverse, want)
    assert abs(forward - reverse) <= 1e-8 * abs(reverse), (forward, reverse)


def test_a_sub_cycled_group_with_a_geometry_edge_is_refused_under_the_interface_norm():
    """The reading is one value per pass, which is not what a member that
    takes several sub-steps per pass was handed: refused at compile with
    the reason, and accepted at one rate and under the other norms."""
    from tests.property import geometry_graphs as gg  # noqa: PLC0415

    shape = Shape("two-way", 24, 4)

    def make(norm, dt_q, **knobs):
        gm = GraphManager()
        nodes = gi._nodes(shape, {"q": dt_q})                      # noqa: SLF001
        for node in nodes.values():
            gm.add_node(node)
        for (src, dst), way, anchor in zip(gi.EDGES, gi.WAYS[shape.kind], shape.anchors):
            gm.add_edge(src, dst, "x", "u", mapping=gi._mapping(shape, way),   # noqa: SLF001
                        geometry=(anchor, "pos"))
        tolerance = {"tolerance": 1e-6} if norm == "l2" else {"rtol": 1e-4}
        gm.add_coupling_group(["p", "q"], convergence_norm=norm, max_iterations=20,
                              **tolerance, **knobs)
        return gm

    with precision(True):
        refused = make("interface", gi.DT / 2, subcycling=True)
        gg.assert_interface_norm_refused(refused.compile, ["p.x->q.u", "q.x->p.u"], "sub-cycled")
        assert "['q']" in "".join(i for i in refused.validate() if i.startswith("ERROR"))
        assert not [i for i in make("interface", gi.DT).validate() if i.startswith("ERROR")]
        for norm in ("mixed", "l2"):
            assert not [i for i in make(norm, gi.DT / 2, subcycling=True).validate()
                        if i.startswith("ERROR")]


# ---------------------------------------------------------------------------
# Slow: the matrix
# ---------------------------------------------------------------------------

#: ``(markers, grid points)`` of the slow sweep (the grid a multiple of 3).
SIZES = ((5, 120), (20, 1_200), (60, 12_000), (60, 120_000))
#: Each ``(kind, schedule, acceleration)`` at four sizes; the anchors and
#: the dtype are written per kind and size, so that every acceleration
#: meets both anchors of a gather and of a scatter in both dtypes.
_SWEEP = [(kind, schedule, acceleration)
          for kind in gi.KINDS for schedule in ("gauss-seidel", "jacobi")
          for acceleration in ACCELERATIONS]


def _sweep_shape(kind, schedule, acceleration, k):
    m, n = SIZES[k]
    position = (gi.KINDS.index(kind) + (schedule == "jacobi")
                + list(ACCELERATIONS).index(acceleration) + k)
    dtype = ("float64", "float32")[position % 2]
    return Shape(kind, n, m, ANCHORS[position % 4], schedule=schedule, dtype=dtype,
                 # A float32 position hundreds of spacings from zero is stored
                 # to more than the tolerance of ``rtol`` spacings (the float
                 # floor says so): on the larger grids the positions are
                 # float64 under float32 values (a mixed-dtype cell).
                 geom_dtype="float64" if dtype == "float32" and n > 200 else None,
                 acceleration=acceleration, d=1 + position % 2, origin=float(7 * (position % 3)))


# Per push: tests/property/test_coupling_geometry_interface_norm.py::test_a_converged_group_is_within_K_tolerances_in_the_compact_readings
@pytest.mark.slow
@pytest.mark.parametrize("kind, schedule, acceleration", _SWEEP,
                         ids=["-".join(row) for row in _SWEEP])
def test_the_claim_at_every_size_and_acceleration(kind, schedule, acceleration):
    worst = []
    for k in range(len(SIZES)):
        if acceleration.startswith("iqn") and SIZES[k][1] > 12_000:
            continue        # a quasi-Newton history over a field of 1e5 entries
        shape = _sweep_shape(kind, schedule, acceleration, k)
        out = gi.run(shape, DRAWS[k % 2])
        assert out["converged"], (shape, out["report"])
        assert out["excess"] <= ALLOWED[shape.dtype], (shape, out["distance"], out["K"])
        worst.append(round(out["excess"], 3))
        gi.built.cache_clear()
    print(f"\n{kind} {schedule} {acceleration}: excess {worst}")


def test_the_sweep_meets_every_anchor_dtype_and_axis_under_every_acceleration():
    """The premise of the slow sweep, per push and with no graph."""
    for acceleration in ACCELERATIONS:
        shapes = [_sweep_shape(kind, schedule, acc, k)
                  for kind, schedule, acc in _SWEEP if acc == acceleration
                  for k in range(len(SIZES))]
        assert {s.anchors for s in shapes} == set(ANCHORS)
        assert {s.dtype for s in shapes} == {"float32", "float64"}
        assert {s.d for s in shapes} == {1, 2} and {s.origin for s in shapes} == {0.0, 7.0, 14.0}
        assert any(s.geom_dtype == "float64" and s.dtype == "float32" for s in shapes)


# Per push: tests/property/test_coupling_geometry_interface_norm.py::test_a_plain_iteration_stops_where_the_reference_s_loop_does
@pytest.mark.slow
@pytest.mark.parametrize("kind", gi.KINDS)
@pytest.mark.parametrize("anchors", ANCHORS, ids="-".join)
@pytest.mark.parametrize("schedule", ["gauss-seidel", "jacobi"])
def test_the_plain_loop_on_every_kind_anchor_and_schedule(kind, anchors, schedule):
    for dtype, geom_dtype, d in (("float64", None, 2), ("float32", "float64", 1),
                                 ("float32", None, 2)):
        # (float32 positions on the grid of 40 cells along an axis only.)
        shape = Shape(kind, 600 if geom_dtype or dtype == "float64" else 120, 20 if d == 1 else 5,
                      anchors, schedule=schedule, dtype=dtype, geom_dtype=geom_dtype, d=d)
        _check_plain_loop(shape, DRAWS[0])
    gi.built.cache_clear()


# Per push: tests/property/test_coupling_geometry_interface_norm.py::test_refining_the_grid_under_fixed_markers_does_not_grow_K
@pytest.mark.slow
@pytest.mark.parametrize("shape", [PER_PUSH[0], PER_PUSH[3],
                                   dataclasses.replace(PER_PUSH[0], acceleration="aitken")],
                         ids=_id)
def test_K_does_not_grow_over_four_decades_of_grid(shape):
    shape = dataclasses.replace(shape, n_small=30)
    rows = _refined(shape, (120, 1_200, 12_000, 120_000, 1_200_000), DRAWS[0])
    print(f"\n{_id(shape)}: (N, K, distance in tolerances, passes) {rows}")
    gi.built.cache_clear()


# Per push: tests/property/test_coupling_geometry_interface_norm.py::test_an_edge_mapped_scatter_reports_what_its_marker_side_twin_reports
@pytest.mark.slow
@pytest.mark.parametrize("shape", [
    PER_PUSH[0],
    dataclasses.replace(PER_PUSH[0], anchors=("target", "source")),
    dataclasses.replace(PER_PUSH[0], schedule="jacobi", anchors=("source", "source")),
    dataclasses.replace(PER_PUSH[0], dtype="float32", d=1, acceleration="aitken"),
    dataclasses.replace(PER_PUSH[0], n_large=12_000, n_small=20, acceleration="iqn-ils"),
], ids=_id)
def test_the_marker_side_twin_on_both_anchors_and_dtypes(shape):
    for draw in DRAWS:
        _check_twin(shape, draw)


# Per push: tests/property/test_coupling_geometry_interface_norm.py::test_the_same_problem_a_thousand_spacings_from_the_origin_reports_the_same
@pytest.mark.slow
@pytest.mark.parametrize("shape", [
    PER_PUSH[3], dataclasses.replace(PER_PUSH[0], acceleration="iqn-imvj"),
    dataclasses.replace(PER_PUSH[2], schedule="jacobi", acceleration="aitken"),
], ids=_id)
def test_a_translated_problem_takes_the_same_passes_under_other_kinds_and_accelerations(shape):
    for draw in DRAWS:
        a = gi.run(shape, draw)
        b = gi.run(dataclasses.replace(shape, origin=1000.0), draw)
        assert a["converged"] and b["converged"]
        assert a["iterations"] == b["iterations"], (a["iterations"], b["iterations"])
        assert abs(a["residual"] - b["residual"]) <= 1e-5 * a["residual"] + 1e-9, (
            a["residual"], b["residual"])
        assert abs(a["K"] - b["K"]) <= 1e-4 * a["K"]
