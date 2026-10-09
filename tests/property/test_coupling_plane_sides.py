"""The lattice-plane table of a group with a geometry edge: every reported
number whose flag is set, against an independent reference, with the
returned iterate, the Newton point and the fixed point on every combination
of sides of a lattice plane (MADD-ANO-248, MAP-049, CPL-093).

``gradient_relative_error_bound`` is the smooth theory's bound: it takes the
pass's Jacobian at the returned iterate and at the Newton point.  A
``multilinear_grid`` mapping is another polynomial of the positions across a
lattice plane, so the bound describes the fixed point only where all three
are in one polynomial piece -- and the flag stood wherever the two sampled
Jacobians agreed, which they do when the fixed point is just past a plane
the Newton point stops short of: a bound 15 to 70,000 times under the
error, flagged usable.

``spectral_error_bound`` **has the same hole.**  It is a
bound to the fixed point of the polynomial the pass is in the iterate's
lattice cells, and with that fixed point past a plane -- the iterate and
the Newton point both short of it -- the pass has no fixed point there: the
flag stood on a bound 17 to 1,294 times under the distance where the next
cell expands.  And a rule that asked for no plane inside the
Newton-Kantorovich ball around the iterate
(``_bounds._kantorovich_ball_plane_margin``) kept the flag on a cell whose
polynomial has no fixed point at all: 34 to 1,874 times under.

**The rule of 0.4.0** (``_group_layout._geometry_flags``): a group whose
pass reads a position from the iterate, or builds one and reads it in the
same pass, has **neither flag, on any step**, with one reason; the numbers
stay as computed, and the stored limit and margin are still the numbers
their definitions give.  A group whose positions are constants of the pass
keeps a smooth group's flags, and no reason of its report names a lattice
plane.  So every test here holds a report to that, row by row, and keeps
scoring the *numbers* on the rows where they were scored when a flag stood
on them (the table's ``*_number_holds``), with a fourth point
beside the three: the fixed point of the iterate's polynomial piece
(``plane_sides.cell_fixed_point``), which the margin's argument puts in the
iterate's cell wherever the margin is over one.  The cases of that audit are
constructed, not drawn (``tests/property/plane_placed.py``): the curvature
from the kernel alone on a two-dimensional lattice and from a member's
quadratic response on a one-dimensional one, with the next cell expanding
or contracting at 0.99 and 0.999.

**The instrument** (``tests/property/plane_sides.py``): a NumPy reference
of the pass that does not import the library's kernel, a *constructive*
placement of the fixed point at a chosen signed distance from a plane, and
the row of the table a case is in -- which of the iterate and the Newton
point are in the fixed point's lattice cells.

* **Per push**: the pinned table (``plane_sides_table.json``, drawn once by
  the sweep's generator, seed 465): three structures (float32 Gauss-Seidel
  ``l2``; float64 Jacobi ``l2``; float64 Jacobi ``mixed`` with Aitken), each
  with every row it reaches, a row whose flag stands and a row within a few
  float resolutions of a plane; and the five audited cases, pinned as the
  audit recorded them.
* **Slow**: the sweep -- every anchoring, both schedules, both norms,
  float32 and float64, a loose and a tight tolerance and a plain and an
  Aitken iteration spread evenly over them -- with the signed distances
  drawn log-uniformly from four float resolutions to a tenth of a spacing.
"""

from __future__ import annotations

import collections
import ctypes
import functools
import gc
import json
import os
import re
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

from tests.property import plane_placed as pp
from tests.property import plane_sides as ps
from tests.property.sysid_transform_grid import precision

KEY = "F+P"
TABLE = json.loads(Path(__file__).with_name("plane_sides_table.json").read_text())
#: The fields of a case that are its structure: one compiled graph each.
STRUCTURE = ("N", "m", "origin", "spacing", "dtype", "anchors", "order", "mode", "norm",
             "tolerance", "rtol", "max_iterations", "acceleration", "relaxation")
SEED = 465


def _structure_key(cfg: ps.Cfg) -> tuple:
    return tuple((name, getattr(cfg, name)) for name in STRUCTURE)


@functools.lru_cache(maxsize=None)
def _graph(key: tuple):
    """The compiled graph of a structure (inside its precision)."""
    structure = dict(key)
    cfg = ps.rounded(ps.drawn(ps.Cfg(**structure), np.random.default_rng(0)))
    gm = ps.build(cfg)
    gm.compile()
    return gm


def _release_what_was_compiled() -> None:
    """Drop the structures' graphs and every program JAX compiled, and
    hand the freed pages back to the system."""
    _graph.cache_clear()
    gc.collect()
    jax.clear_caches()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):          # not glibc: the pages stay with the process
        pass


@pytest.fixture(autouse=True)
def _a_slow_test_releases_what_it_compiled(request):
    """The slow tests of this module compile a graph for every cell they
    sweep.  Kept, those programs took the module to 6.9 GB on its own and
    the slow-lane shard that ran it aborted in a later file's compile, on
    both jax lanes; released after each slow test, the module peaks at
    0.9 GB for a tenth more time.  The per-push tests keep their graphs
    until the module ends (they share one per structure)."""
    yield
    if request.node.get_closest_marker("slow") is not None:
        _release_what_was_compiled()


@pytest.fixture(scope="module", autouse=True)
def _the_compiled_graphs_live_for_this_module_only():
    yield
    _release_what_was_compiled()


def _stepped(cfg: ps.Cfg):
    """``(x, c, report, slots)`` of one step of *cfg* on its structure's graph."""
    gm = _graph(_structure_key(cfg))
    ps.load(cfg, gm)
    x, c, report = ps.step_report(cfg, gm)
    meta = gm._state["_meta"]                                          # noqa: SLF001
    slots = {name: float(meta[f"coupling_{KEY}_geometry_{name}"])
             for name in ("plane_limit", "plane_margin")}
    return x, c, report, slots


def _rows_of(structure: str) -> list:
    return [row for row in TABLE["rows"] if row["structure"] == structure]


def _not_the_rule_of_a_group_that_solves_positions(report: dict) -> list:
    """What the report of a group whose pass reads a position from the
    iterate has that the rule of 0.4.0 does not allow, as text: a flag, or
    (where the step computed its estimate) another reason than the rule's."""
    bad = []
    if report["spectral_usable"] or report["gradient_bound_usable"]:
        bad.append("a flag is set in a group that solves positions")
    if np.isfinite(report["rho_spectral"]):
        reason = report.get("not_usable_reason", "")
        if not ("solves position(s)" in reason and "fixed during the pass" in reason
                and "0.4.0 does not certify a bound for such a group" in reason):
            bad.append(f"the reason is not the rule's: {reason[:160]!r}")
    return bad


def _names_a_lattice_plane(report: dict) -> bool:
    return "lattice plane" in report.get("not_usable_reason", "")


def _scored_as(report: dict, spectral: bool, gradient: bool) -> dict:
    """*report* with its flags replaced, for the instruments that score
    every number whose flag is set: the numbers of a group that solves
    positions carry no flag, and are still held to the reference on the
    rows where they were when one stood on them."""
    return {**report, "spectral_usable": bool(spectral), "gradient_bound_usable": bool(gradient)}


#: The two audited cases of a position within a rounding of a plane are on
#: lattices of their own (two axes; an origin forty spacings out with a
#: quasi-Newton acceleration): two more compiles, run in the slow lane.
SLOW_STRUCTURE = "audited: a position within a rounding of a plane"
STRUCTURES = sorted({row["structure"] for row in TABLE["rows"]} - {SLOW_STRUCTURE})
PER_PUSH = {row["name"] for row in TABLE["rows"] if row["structure"] != SLOW_STRUCTURE}


# ---------------------------------------------------------------------------
# Per push: the pinned table
# ---------------------------------------------------------------------------


def test_the_pinned_table_has_every_row_of_sides_a_standing_flag_and_the_audited_cases():
    """What the per-push table can express, so that it cannot lose a row
    without a test failing: the four combinations of sides, each from more
    than one structure where it is reachable; rows whose flags stand (the
    positions that are constants of the pass) and none on any other; rows
    whose numbers are still scored; a row within a few float resolutions of
    a plane; float32 and float64, both schedules, both norms."""
    rows = TABLE["rows"]
    control = [r for r in rows if "constant of the pass" in r["structure"]]
    # Positions that are constants of the pass: a plane between the
    # iterate's and the fixed point's, and both flags.
    assert {r["row"] for r in control} == {"k=* N=*", "k!=* N=*"}
    assert all(r["gradient_bound_usable"] and r["spectral_usable"] for r in control)
    rows = [r for r in rows if r not in control]
    # A position read from the iterate: no row asks for a flag.
    assert not any(r["gradient_bound_usable"] or r["spectral_usable"] for r in rows)
    drawn = [r for r in rows if not r["structure"].startswith("audited")]
    by_row = collections.Counter(r["row"] for r in drawn)
    assert set(by_row) == set(ps.ROWS), by_row
    # The row of this audit from three structures (the audited cases are
    # those structures' own), the row of the one before from three.
    for name in ("k!=* N!=*", "k!=* N=*"):
        told = {(c.dtype, c.mode, c.norm) for c in (
            ps.as_cfg(r["cfg"]) for r in rows if r["row"] == name and r["name"] in PER_PUSH)}
        assert len(told) >= 3, (name, told)
    # The numbers are still held to the reference on a row of each drawn
    # structure, and both are on each of them.
    held = [r for r in drawn if r["gradient_number_holds"]]
    assert {r["structure"] for r in held} == {r["structure"] for r in drawn}
    assert all(r["spectral_number_holds"] and r["row"] == "k=* N=*" for r in held)
    assert any(not r["gradient_number_holds"] and r["row"] == "k=* N=*" for r in drawn)
    assert any(r["on_a_plane"] for r in drawn)
    cfgs = [ps.as_cfg(r["cfg"]) for r in rows]
    assert all(ps.moving(c) for c in cfgs) and not any(
        ps.moving(ps.as_cfg(r["cfg"])) for r in control)
    assert {c.dtype for c in cfgs} == {"float32", "float64"}
    assert {c.mode for c in cfgs} == {"gauss-seidel", "jacobi"}
    assert {c.norm for c in cfgs} == {"l2", "mixed"}
    assert sum(r["structure"].startswith("audited") for r in rows) == 5
    # No row scores a number on a fixed point across a plane.
    assert not any((r["gradient_number_holds"] or r["spectral_number_holds"])
                   and r["row"] != "k=* N=*" for r in rows)
    # A report with every point in one cell and a plane in the
    # Newton-Kantorovich ball (a margin at or under one), from more than
    # one structure.
    assert len({r["structure"] for r in drawn if r["row"] == "k=* N=*"
                and not r["on_a_plane"] and not r["spectral_number_holds"]}) >= 2


def _check_rows(structure: str) -> None:
    """Row by row: the sides are the pinned ones (the instrument still
    expresses the row); the flags are the pinned ones -- none on a row whose
    pass reads a position from the iterate, with the rule's reason, both on
    the rows of constant positions, with no lattice plane named; and every
    number the row scores holds against the reference, as it did when a
    flag stood on it."""
    rows = _rows_of(structure)
    assert rows
    for row in rows:
        cfg = ps.as_cfg(row["cfg"])
        with precision(cfg.dtype == "float64"):
            x, c, report, slots = _stepped(cfg)
            where = ps.sides(cfg, x, c)
            told = (row["name"], where, {k: report.get(k) for k in (
                "spectral_error_bound", "spectral_usable", "gradient_relative_error_bound",
                "gradient_bound_usable")}, slots)
            assert where["fp_ok"], told
            if not row["on_a_plane"]:
                # Away from the float resolution the row does not depend on
                # a rounding (the drawn rows are pinned at least two
                # hundred resolutions off; the audited ones are the audit's,
                # and the rows of constant positions thirty).
                assert where["row"] == row["row"], told
                assert where["nearest_resolutions"] > (
                    100 if ps.moving(cfg) and not structure.startswith("audited") else 10), told
            assert ps.wrong_numbers(cfg, x, c, report, where) == [], told
            assert bool(report["gradient_bound_usable"]) == row["gradient_bound_usable"], told
            assert bool(report["spectral_usable"]) == row["spectral_usable"], told
            margin = slots["plane_margin"]
            if not ps.moving(cfg):
                # Every position a constant of the pass: nothing to cross,
                # a smooth group's flags, and no lattice plane in a reason.
                assert margin == np.inf == slots["plane_limit"], told
                assert not _names_a_lattice_plane(report), told
                if row["gradient_bound_usable"]:
                    assert "not_usable_reason" not in report, told
                continue
            # A position read from the iterate: no flag, the rule's reason,
            # and the numbers as computed.
            assert _not_the_rule_of_a_group_that_solves_positions(report) == [], told
            assert np.isfinite(report["spectral_error_bound"]), told
            # The numbers the row scores, held as when a flag stood on them.
            assert ps.wrong_numbers(cfg, x, c, _scored_as(
                report, row["spectral_number_holds"], row["gradient_number_holds"]),
                where) == [], told
            if row["gradient_number_holds"]:
                assert margin > 1.0, told
            if margin > 1.0 and where["p_ok"] and not where["on_a_plane"]:
                # The margin's argument, on the reference: with no plane in
                # the ball the fixed point of the iterate's polynomial is in
                # the iterate's cell, and is the pass's.
                assert where["p_in_cell"] and where["p_is_fixed_point"], told


def test_the_report_of_a_marker_at_rest_on_a_hull_face_is_the_same_on_every_step():
    """A marker at rest at coordinate 0.0, the lower face of a lattice whose
    origin is 0.0, its position read from the iterate: both flags are
    ``False`` on every step with one reason, the rule's, and the stored
    margin is zero on every step.  (The Newton step's entry for
    that position is rounding of either sign, which nothing absorbs at
    exactly zero: on the steps where it is negative the Newton point is
    outside the hull and ``gradient_relative_error_bound`` reads ``inf``.
    ``spectral_usable`` used to follow it, on a third to a half of the
    steps of one resting state.)"""
    row = next(r for r in _rows_of("f32 gauss-seidel l2") if r["gradient_number_holds"])
    cfg = ps.as_cfg({**row["cfg"], "aF": 0.5, "bF": 0.1, "aP": 0.5, "bP": 1.0, "cP": 0.0,
                     "quad": 0.0, "kP": 0.0, "vP": [0.0], "posP0": [0.0]})
    assert cfg.anchors == ("target", "source") and cfg.origin == (0.0,)
    reasons = set()
    with precision(False):
        gm = _graph(_structure_key(cfg))
        ps.load(cfg, gm)
        for _step in range(12):
            x, _c, report = ps.step_report(cfg, gm)
            assert float(x[ps.pos_slices(cfg)["P"]][0]) == 0.0           # at rest, on the face
            assert report["spectral_usable"] is False, report
            assert report["gradient_bound_usable"] is False, report
            assert not report["precision_limited"] and np.isfinite(report["rho_spectral"])
            margin = float(gm._state["_meta"][f"coupling_{KEY}_geometry_plane_margin"])  # noqa: SLF001
            assert margin == 0.0
            reasons.add(report["not_usable_reason"])
    (reason,) = reasons
    assert "solves position(s) ['P.pos']" in reason, reason
    assert "fixed during the pass" in reason and "tighter tolerance" not in reason, reason


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@pytest.mark.parametrize("structure", STRUCTURES, ids=[_slug(s) for s in STRUCTURES])
def test_every_flagged_number_of_a_table_row_holds_against_the_reference(structure):
    _check_rows(structure)


# Per push: tests/core/test_grid_probe_step_and_plane_distance.py::test_a_position_within_eight_float_resolutions_of_a_plane_is_on_it
@pytest.mark.slow
def test_the_audited_positions_within_a_rounding_of_a_plane_lose_the_gradient_flag():
    """The audit's two float32 cases (a lattice origin forty spacings from
    zero): the iterate 4.7 float resolutions before a plane with the fixed
    point 0.7 past it, where the gradient bound was 25x under the error
    with its flag set; and a Gauss-Seidel sweep that reads a position a
    third of a resolution from a plane (the float32 pass evaluates one
    cell, exact arithmetic the other: MADD-ANO-239, whose ``rho_spectral``
    is no longer flagged either: a position within the window of a plane
    is on it, the margin is zero and both flags go)."""
    _check_rows(SLOW_STRUCTURE)


def test_the_audited_bounds_were_flagged_and_far_under_the_error():
    """The three audited rows of MADD-ANO-248 as the audit recorded them:
    the reference's gradient error is 15 to 70,000 times the bound the
    report still carries, the fixed point is across a plane the Newton point
    is short of, and the flags are what changed (the spectral one too: on
    these three its bound happened to hold, within twice, in a next cell
    that contracts as the first does; the rule cannot know the next cell)."""
    worst = {}
    for row in _rows_of("audited: the gradient bound short of a plane"):
        cfg = ps.as_cfg(row["cfg"])
        with precision(cfg.dtype == "float64"):
            x, c, report, _slots = _stepped(cfg)
            where = ps.sides(cfg, x, c)
            score = ps.score(cfg, x, c, report)
        assert where["row"] == "k!=* N!=*", (row["name"], where)
        assert not report["gradient_bound_usable"], row["name"]
        assert not report["spectral_usable"], row["name"]
        worst[row["name"]] = score["grad_err"] / float(report["gradient_relative_error_bound"])
    assert len(worst) == 3
    assert min(worst.values()) > 10 and max(worst.values()) > 5e4, worst


# ---------------------------------------------------------------------------
# Per push: the fixed point of the iterate's polynomial placed past a plane
# ---------------------------------------------------------------------------

_PLACED_GRAPHS: dict = {}


def _placed(mo: pp.Model):
    """``(report, slots, where)`` of one step of *mo* on its structure's graph."""
    key = mo.structure()
    if key not in _PLACED_GRAPHS:
        gm = pp.build(mo)
        gm.compile()
        _PLACED_GRAPHS[key] = gm
    gm = _PLACED_GRAPHS[key]
    pp.load(mo, gm)
    x, c, report, slots = pp.step_report(mo, gm)
    return report, slots, pp.located(mo, x, c)


#: ``kind -> (name, arguments of the construction, what the row is)``.
#: ``short``: the returned iterate and the Newton point both before the
#: plane, the fixed point of their cell's polynomial past it (this audit's
#: case), with the next cell expanding (the iterate ends two cells on) or
#: contracting at 0.999 or 0.99 (a fixed point of its own in the next
#: cell); ``crossed``: the Newton point past the plane too; ``near``: the
#: fixed point in the iterate's cell with the plane in the ball (an honest
#: report that loses its flags); ``clear``: no plane in the ball.
CONSTRUCTED = {
    "kernel": (pp.kernel_curvature, [
        ("the next cell expands", dict(s=2e-6), "short"),
        ("the next cell expands, twice the curvature", dict(s=2e-6, Q=1.0), "short"),
        ("the next cell contracts at 0.999", dict(s=2e-6, after=(0.999,)), "short"),
        ("the next cell contracts at 0.99", dict(s=2e-6, after=(0.99,)), "short"),
        ("the Newton point crosses", dict(s=5e-6), "crossed"),
        ("the fixed point just inside the cell", dict(s=-2e-3), "near"),
        ("the fixed point deep inside the cell", dict(s=-5e-2), "clear"),
    ]),
    "member": (pp.member_curvature, [
        ("the next cell expands", dict(s=1e-6), "short"),
        ("the next cell contracts at 0.999", dict(s=1e-6, after=(0.999,)), "short"),
        ("the next cell contracts at 0.99", dict(s=1e-6, after=(0.99,)), "short"),
        ("the Newton point crosses", dict(s=2e-5), "crossed"),
        ("the fixed point just inside the cell", dict(s=-4e-4), "near"),
        ("the fixed point deep inside the cell", dict(s=-5e-2), "clear"),
    ]),
}


def _check_constructed(curvature: str, dtype: str) -> dict:
    """Row by row: the construction is the row it says it is (by the
    reference), the stored margin is over one only with the fixed point
    deep in the cell, the flags are the rule's -- none on any row, the
    marker's position being read from the iterate, with its reason -- and
    the numbers of the deep row hold as when its flags stood.  Returns the
    distance over the bound per row."""
    make, rows = CONSTRUCTED[curvature]
    ratios = {}
    with precision(dtype == "float64"):
        for name, arguments, kind in rows:
            mo = make(dtype=dtype, **arguments)
            report, slots, where = _placed(mo)
            told = (curvature, dtype, name, where, slots, {k: report.get(k) for k in (
                "spectral_error_bound", "spectral_usable", "gradient_relative_error_bound",
                "gradient_bound_usable", "converged")})
            assert where["fp_ok"] and where["p_ok"] and report["converged"], told
            cells, past = where["cells"], where["past"]
            if kind in ("short", "crossed"):
                assert past["k"] < 0 < past["p"], told
                assert not where["p_in_cell"] and cells["*"] != cells["k"], told
                assert (cells["N"] == cells["k"]) is (kind == "short"), told
            else:
                assert where["p_in_cell"] and where["one_cell"] and past["p"] < 0, told
            clear = kind == "clear"
            assert (slots["plane_margin"] > 1.0) is clear, told
            assert _not_the_rule_of_a_group_that_solves_positions(report) == [], told
            assert np.isfinite(report["spectral_error_bound"]), told
            assert pp.wrong_numbers(_scored_as(report, clear, clear), where) == [], told
            ratios[name] = where["distance"] / float(report["spectral_error_bound"])
    # What the withdrawn flag stood on: where the next cell expands the
    # pass's fixed point is two cells on, tens to hundreds of bounds away.
    assert ratios["the next cell expands"] > 10.0, ratios
    assert ratios["the fixed point deep inside the cell"] <= 1.001, ratios
    return ratios


@pytest.mark.parametrize("curvature, dtype", [("kernel", "float64"), ("member", "float32")])
def test_no_flag_stands_with_the_cell_polynomials_fixed_point_past_a_plane(curvature, dtype):
    """The audited construction (MADD-ANO-248, the spectral flag): the
    iterate and the Newton point in one lattice cell, the Newton-Kantorovich
    check reading that cell's small ``h``, and that cell's polynomial with
    its fixed point past the plane.  Neither flag is set whatever the
    next cell does -- expanding (the bound tens of times under the
    distance), contracting at 0.999 or at 0.99 -- nor with the fixed
    point deep in the cell, where the bound holds: the group solves the
    marker's position."""
    _check_constructed(curvature, dtype)


# Per push: tests/property/test_coupling_plane_sides.py::test_no_flag_stands_with_the_cell_polynomials_fixed_point_past_a_plane
@pytest.mark.slow
@pytest.mark.parametrize("curvature, dtype", [("kernel", "float32"), ("member", "float64")])
def test_no_flag_stands_with_the_cell_polynomials_fixed_point_past_a_plane_in_the_other_dtype(
        curvature, dtype):
    _check_constructed(curvature, dtype)


#: The placement sweep's structures: the member-curvature pair with the
#: fixed point of the first cell's polynomial at a chosen signed distance
#: from the plane.  The three in which the next cell expands are where an
#: audit found 66 of 2,700 float32 reports flagged on a bound 17 to 1,294
#: times under the distance.
PLACED_SWEEP = {
    "gauss-seidel marker first": dict(),
    "gauss-seidel grid first": dict(order=("G", "M")),
    "jacobi": dict(mode="jacobi", max_iterations=200),
    "mixed norm": dict(norm="mixed"),
    "aitken": dict(acceleration="aitken"),
    "origin 20": dict(origin=20.0),
    "strong curvature": dict(Q=8.0, e1=0.01),
    "the next cell expands": dict(after=(1.2, 0.5)),
    "jacobi, mixed, the next cell expands": dict(
        mode="jacobi", norm="mixed", max_iterations=200, after=(1.2, 0.5)),
    "gauss-seidel grid first, the next cell expands": dict(order=("G", "M"), after=(1.2, 0.5)),
}
PLACED_TOLERANCES = (3e-2, 1e-2, 1e-3, 1e-4)


def _placed_distances(mo: pp.Model) -> list:
    """Signed distances of the cell polynomial's fixed point from the
    plane: float resolutions of the position at the lattice's scale, and
    fractions of a spacing."""
    eps = float(np.finfo(np.dtype(mo.dtype)).eps)
    top = mo.origin[0] + (mo.shape[0] - 1) * mo.spacing[0]
    resolution = eps * max(abs(mo.origin[0]), abs(top), abs(mo.plane))
    sizes = [k * resolution for k in (0.5, 3.0, 12.0, 100.0, 1000.0)]
    sizes += [f * mo.spacing[0] for f in (1e-5, 1e-4, 1e-3, 1e-2, 5e-2)]
    return [sign * size for sign in (1.0, -1.0) for size in sizes] + [0.0]


# Per push: tests/property/test_coupling_plane_sides.py::test_no_flag_stands_with_the_cell_polynomials_fixed_point_past_a_plane
@pytest.mark.slow
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("structure", sorted(PLACED_SWEEP), ids=_slug)
def test_the_placement_sweep_finds_no_flagged_bound_under_the_distance(structure, dtype):
    """The fixed point of the first cell's polynomial at 21 signed distances
    from the plane, four tolerances, ten structures, both dtypes: the
    marker's position is read from the iterate in every one, so no report
    has a flag and each carries the rule's reason, on either side of the
    plane and at any distance.  The realised counts are printed."""
    knobs = dict(after=(0.9,), Q=2.0, e1=0.02, c0=0.1)
    knobs.update(PLACED_SWEEP[structure])
    counts = collections.Counter()
    failures = []
    with precision(dtype == "float64"):
        for tolerance in PLACED_TOLERANCES:
            base = pp.member_curvature(dtype=dtype, tolerance=tolerance, **knobs)
            for s in _placed_distances(base):
                mo = pp.member_curvature(dtype=dtype, tolerance=tolerance, s=s, **knobs)
                report, slots, where = _placed(mo)
                if not where["fp_ok"]:
                    counts["no reference"] += 1
                    continue
                flags = ("both" if report["gradient_bound_usable"] else
                         "spectral only" if report["spectral_usable"] else "none")
                counts[("one cell" if where["one_cell"] else "across", flags)] += 1
                bad = pp.wrong_numbers(report, where)
                bad += _not_the_rule_of_a_group_that_solves_positions(report)
                if slots["plane_margin"] == np.inf or slots["plane_limit"] == np.inf:
                    bad.append(f"a slot of a pass with no reader: {slots}")
                if bad:
                    failures.append((tolerance, s, bad, where))
    print(f"\n[placed] {structure} {dtype}: " + ", ".join(
        f"{key}: {count}" for key, count in sorted(counts.items(), key=str)))
    assert not failures, failures[:3]
    scored = sum(count for key, count in counts.items() if key != "no reference")
    assert scored >= len(PLACED_TOLERANCES) * 10, counts


# ---------------------------------------------------------------------------
# The cell whose polynomial has no fixed point: the case that ended the rules
# ---------------------------------------------------------------------------

#: The pair of the third audit (``benchmarks/results/audit_040_p4_21/
#: plane_margin/repro_r3e_bottleneck_kernel_only.py``): one marker ``M`` on a
#: one-dimensional lattice of six points (origin 0, spacing 1/2), affine
#: members, the only nonlinearity the scatter's own (a value times a weight):
#:
#:     ``M:  pos' = pos_pre + V + w . G.x      y' = cy + wy . G.x``
#:     ``G:  x'   = scatter(y' at pos')``
#:
#: With ``t = (pos - 1) / (1/2)`` the marker's coordinate in the cell between
#: the lattice points 1 and 3/2, the pass is the quadratic map
#: ``t' = e1 + y (1/4 + t/2)``, ``y' = 3/4 + y (-1/8 + 3 t/4)``, whose
#: Jacobian at ``(1/2, 1)`` has the eigenvalues 1 and -1/4: a saddle-node.
#: With ``e1 = gap / 0.6 > 0`` that cell's polynomial has **no fixed point**;
#: the iterate creeps towards the bottleneck in the middle of the cell, the
#: solve's estimate falls under the tolerance before it gets there, and the
#: pass's only fixed point is one cell on, at ``t = 3/2 + e1``, ``y = 2``.
_SADDLE_W = (0.125, 0.125, 0.125, 0.375, 0.375, 0.375)
_SADDLE_WY = (-0.125, -0.125, -0.125, 0.625, 0.625, 0.625)
_SADDLE_POINTS, _SADDLE_SPACING, _SADDLE_START = 6, 0.5, -0.25

#: ``id -> (dtype, add_node order, norm, acceleration, gap, tolerance)``:
#: float32 and float64, both sweep orders, both norms and Aitken.  Each is
#: a report the audit recorded (float32; float64 for the first sweep order)
#: or the same construction in the other dtype, with ``converged=True``.
SADDLE_NODE = {
    "float32-marker-first-l2": ("float32", ("M", "G"), "l2", "none", 1e-6, 1e-2),
    "float64-marker-first-l2": ("float64", ("M", "G"), "l2", "none", 1e-6, 1e-2),
    "float32-marker-first-mixed": ("float32", ("M", "G"), "mixed", "none", 1e-6, 1e-2),
    "float64-marker-first-mixed": ("float64", ("M", "G"), "mixed", "none", 1e-6, 1e-2),
    "float32-grid-first-l2": ("float32", ("G", "M"), "l2", "none", 1e-6, 1e-2),
    "float64-grid-first-l2": ("float64", ("G", "M"), "l2", "none", 1e-6, 1e-2),
    "float32-marker-first-l2-aitken": ("float32", ("M", "G"), "l2", "aitken", 1e-6, 3e-3),
    "float64-marker-first-l2-aitken": ("float64", ("M", "G"), "l2", "aitken", 1e-6, 3e-3),
}
SADDLE_NODE_PER_PUSH = ("float32-marker-first-l2", "float64-marker-first-l2")


def _saddle_node_graph(dtype, order, norm, acceleration, gap, tolerance):
    """The pair as a graph, and the offset ``e1`` of its bottleneck."""
    import jax.numpy as jnp
    from maddening.core.coupling.grid_mapping import multilinear_grid_mapping
    from maddening.core.graph_manager import GraphManager
    from maddening.core.node import BoundaryInputSpec, SimulationNode

    kind = jnp.dtype(dtype)
    n, h = _SADDLE_POINTS, _SADDLE_SPACING
    e1 = gap / 0.6
    t0, y0 = 0.5 + _SADDLE_START, 1.0 + _SADDLE_START
    x0 = np.zeros(n)
    x0[2], x0[3] = y0 * (1.0 - t0), y0 * t0
    pos0 = 1.0 + h * t0

    class Grid(SimulationNode):
        def initial_state(self):
            return {"x": jnp.asarray(x0, kind)}

        def boundary_input_spec(self):
            return {"deposit": BoundaryInputSpec(shape=(n,), dtype=kind)}

        def update(self, state, boundary_inputs, dt, *, params=None):
            return {"x": boundary_inputs.get("deposit", state["x"]) + 0 * state["x"]}

    class Marker(SimulationNode):
        def initial_state(self):
            return {"pos": jnp.full((1, 1), pos0, kind), "y": jnp.full((1,), y0, kind)}

        def boundary_input_spec(self):
            return {"field": BoundaryInputSpec(shape=(n,), dtype=kind)}

        def update(self, state, boundary_inputs, dt, *, params=None):
            p = {**self.params, **(params or {})}
            field = boundary_inputs.get("field", jnp.zeros((n,), kind))
            return {"pos": state["pos"] + p["v"] + p["w"] @ field,
                    "y": p["cy"] + (p["wy"] @ field) * jnp.ones((1,), kind)}

    made = {"G": Grid("G", 0.01),
            "M": Marker("M", 0.01, v=np.asarray(1.0 + h * e1 - pos0, kind),
                        w=np.asarray(_SADDLE_W, kind), wy=np.asarray(_SADDLE_WY, kind),
                        cy=np.asarray(0.75, kind))}
    gm = GraphManager()
    for name in order:
        gm.add_node(made[name])
    gm.add_edge("G", "M", "x", "field")
    gm.add_edge("M", "G", "y", "deposit", geometry=("source", "pos"),
                mapping=multilinear_grid_mapping(
                    mode="conservative", origin=[0.0], spacing=[h], shape=[n], n_points=1))
    knobs = {"tolerance": tolerance} if norm == "l2" else {"rtol": tolerance}
    gm.add_coupling_group(["G", "M"], solver="ift", diagnostics=True, convergence_norm=norm,
                          iteration_mode="gauss-seidel", acceleration=acceleration,
                          max_iterations=3000, **knobs)
    return gm, e1


def _saddle_node_pass(x, e1):
    """The pass at ``x = (G.x, M.pos, M.y)`` in NumPy, marker first: a hat
    function for the scatter, nothing of the library's."""
    n, h = _SADDLE_POINTS, _SADDLE_SPACING
    pos = 1.0 + h * e1 + np.dot(_SADDLE_W, x[:n])
    y = 0.75 + np.dot(_SADDLE_WY, x[:n])
    cell = int(np.clip(np.floor(pos / h), 0, n - 2))
    t = pos / h - cell
    field = np.zeros(n)
    field[cell], field[cell + 1] = y * (1.0 - t), y * t
    return np.concatenate([field, [pos, y]])


def _check_the_saddle_node(name: str) -> None:
    dtype, order, norm, acceleration, gap, tolerance = SADDLE_NODE[name]
    n, h = _SADDLE_POINTS, _SADDLE_SPACING
    with precision(dtype == "float64"):
        gm, e1 = _saddle_node_graph(dtype, order, norm, acceleration, gap, tolerance)
        gm.compile()
        gm.step()
        report = dict(gm.coupling_diagnostics()["G+M"])
        meta = gm._state["_meta"]                                      # noqa: SLF001
        margin, limit = (float(meta[f"coupling_G+M_geometry_plane_{slot}"])
                         for slot in ("margin", "limit"))
        grid, marker = gm.get_node_state("G"), gm.get_node_state("M")
        x = np.concatenate([np.asarray(part, np.float64).ravel()
                            for part in (grid["x"], marker["pos"], marker["y"])])
    # The pass's only fixed point, one lattice cell on from the iterate's.
    star = np.zeros(n + 2)
    star[3], star[4], star[n], star[n + 1] = 2.0 * (0.5 - e1), 2.0 * (0.5 + e1), 1.75 + h * e1, 2.0
    assert np.max(np.abs(_saddle_node_pass(star, e1) - star)) < 1e-12
    weight = np.concatenate([np.full(n, 1.0 / np.max(np.abs(x[:n]))), 1.0 / np.abs(x[n:])])
    unit = 1.0 if norm == "l2" else 1.0 / (tolerance * np.sqrt(n + 2))
    distance = float(unit * np.linalg.norm(weight * (x - star)))
    bound, gradient = (float(report[k]) for k in (
        "spectral_error_bound", "gradient_relative_error_bound"))
    told = (name, report, margin, limit, distance)
    assert report["converged"] is True, told
    assert 1.0 < x[n] < 1.5 < star[n] < 2.0, told
    # The rule: no flag, its reason; the numbers as computed.
    assert report["spectral_usable"] is False and report["gradient_bound_usable"] is False, told
    assert _not_the_rule_of_a_group_that_solves_positions(report) == [], told
    assert "solves position(s) ['M.pos']" in report["not_usable_reason"], told
    assert "['M.y->G.deposit']" in report["not_usable_reason"], told
    # What the number is worth, and what every withdrawn rule read here: a
    # bound far under the distance, a margin over one, the step's own
    # Newton-Kantorovich check failed, and a bound under the limit (the
    # Aitken cells end beside the limit, 0.079 under 0.094 in float32 and
    # 0.098 over it in float64: not pinned there).
    assert np.isfinite(bound) and distance > 10.0 * bound, told
    assert margin > 1.0 and gradient == np.inf, told
    assert bound <= limit or acceleration == "aitken", told


@pytest.mark.parametrize("name", SADDLE_NODE_PER_PUSH)
def test_no_flag_stands_in_a_cell_whose_polynomial_has_no_fixed_point(name):
    """The case that ended the plane rules (MADD-ANO-248): the returned
    iterate ``converged`` in a lattice cell whose polynomial has no fixed
    point, the pass's only fixed point one cell on and tens to hundreds of
    bounds away.  The step's Newton-Kantorovich check had failed, the
    stored margin is over one and the bound under the stored limit, so
    every earlier rule kept ``spectral_usable``; the group solves the
    marker's position, so in 0.4.0 it has neither flag, and says why."""
    _check_the_saddle_node(name)


# Per push: tests/property/test_coupling_plane_sides.py::test_no_flag_stands_in_a_cell_whose_polynomial_has_no_fixed_point
@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(set(SADDLE_NODE) - set(SADDLE_NODE_PER_PUSH)))
def test_no_flag_stands_in_a_cell_whose_polynomial_has_no_fixed_point_in_the_other_cells(name):
    """Both sweep orders, the mixed norm and Aitken, in float32 and float64."""
    _check_the_saddle_node(name)


# ---------------------------------------------------------------------------
# Slow: the sweep
# ---------------------------------------------------------------------------

#: Every anchoring (gather, scatter), both schedules, both norms, both
#: dtypes: 32 cells.  A loose and a tight tolerance, the order the members
#: are added in and a plain or an Aitken iteration are spread over them so
#: that each is half of every anchoring, schedule, norm and dtype (a parity
#: of the cell's four indices: nothing here is read off another table).
#: The mixed cells are on a lattice whose origin is forty spacings from
#: zero, where a float32 resolution is 4e-6 of a spacing.
def _sweep_cells() -> list:
    cells = []
    anchorings = (("target", "source"), ("source", "source"), ("source", "target"),
                  ("target", "target"))
    for a, anchors in enumerate(anchorings):
        for m, mode in enumerate(("gauss-seidel", "jacobi")):
            for n, norm in enumerate(("l2", "mixed")):
                for d, dtype in enumerate(("float32", "float64")):
                    loose = (a + m + n + d) % 2 == 0
                    tight = 1e-5 if dtype == "float64" else 1e-3
                    lattice = (dict(N=(6,), origin=(20.0,), spacing=(0.5,)) if norm == "mixed"
                               else dict(N=(6,), origin=(0.0,), spacing=(0.25,)))
                    knobs = {"tolerance" if norm == "l2" else "rtol": 0.03 if loose else tight}
                    cells.append(dict(
                        anchors=anchors, mode=mode, norm=norm, dtype=dtype, m=1,
                        order=("F", "P") if (a + n) % 2 == 0 else ("P", "F"),
                        max_iterations=60,
                        acceleration="aitken" if (m + d) % 2 == 1 else "none",
                        **lattice, **knobs))
    return cells


SWEEP = _sweep_cells()
BASES, DISTANCES = 6, 12
#: The flagged reports a cell of constant positions (two target anchors:
#: eight of the 32 cells) must see: the only cells whose flagged numbers
#: the sweep scores under the rule of 0.4.0.
MIN_FLAGGED_WITH_CONSTANT_POSITIONS = 1


def _cell_id(cell: dict) -> str:
    return "-".join([cell["anchors"][0][0] + cell["anchors"][1][0], cell["mode"], cell["norm"],
                     cell["dtype"], "loose" if 0.03 in (cell.get("tolerance"), cell.get("rtol"))
                     else "tight", cell["acceleration"]])


# Per push: tests/property/test_coupling_plane_sides.py::test_every_flagged_number_of_a_table_row_holds_against_the_reference
@pytest.mark.slow
@pytest.mark.parametrize("index", range(len(SWEEP)), ids=[_cell_id(c) for c in SWEEP])
def test_the_sweep_finds_no_flagged_number_wrong_on_any_side_of_a_plane(index):
    """Six drawn pairs a cell, the fixed point of each placed on a lattice
    plane and then at twelve signed distances from it: every flagged number
    of every example against the reference, the realised table printed.
    With a source anchor the pass reads a position from the iterate: no
    flag on any report, and the rule's reason.  With two target anchors
    every position is a constant of the pass: the flags are a smooth
    group's (the gradient's is the spectral one's wherever the bound is
    finite), no reason names a lattice plane, and the cell must see flagged
    reports, so that these eight cells are what the sweep scores."""
    cell = SWEEP[index]
    structure = ps.Cfg(**cell)
    rng = np.random.default_rng([SEED, index])
    table = collections.Counter()
    flagged = 0
    failures = []
    with precision(cell["dtype"] == "float64"):
        bases = 0
        for _attempt in range(40):
            if bases == BASES:
                break
            placed = ps.placed_on_a_plane(structure, rng)
            if placed is None:
                continue
            cfg, key, j, a, on_plane = placed
            bases += 1
            for offset in ps.signed_distances(cfg, on_plane, a, rng, DISTANCES):
                case = ps.rounded(ps.shifted(cfg, key, j, a, on_plane + offset))
                x, c, report, slots = _stepped(case)
                if not np.all(np.isfinite(x)):
                    continue
                where = ps.sides(case, x, c)
                if not where["fp_ok"]:
                    table["no reference"] += 1
                    continue
                flags = ("gradient" if report["gradient_bound_usable"] else
                         "spectral only" if report["spectral_usable"] else "none")
                table[(where["row"], "on a plane" if where["on_a_plane"] else "off", flags)] += 1
                bad = ps.wrong_numbers(case, x, c, report, where)
                if ps.moving(case):
                    bad += _not_the_rule_of_a_group_that_solves_positions(report)
                else:
                    flagged += bool(report["spectral_usable"])
                    if _names_a_lattice_plane(report):
                        bad.append("a lattice plane in the reason of constant positions")
                    finite = np.isfinite(float(report["gradient_relative_error_bound"]))
                    if bool(report["gradient_bound_usable"]) != bool(
                            report["spectral_usable"] and finite):
                        bad.append("a flag withdrawn with every position a constant of the pass")
                    if not slots["plane_margin"] == np.inf:
                        bad.append(f"plane_margin {slots['plane_margin']} with no reader")
                if bad:
                    failures.append((offset, where, bad, case))
    print(f"\n[plane sides] {_cell_id(cell)}: {bases} bases")
    for entry in sorted(table, key=str):
        print(f"    {entry!s:60s} {table[entry]}")
    assert bases >= BASES // 2, "the placement found too few pairs with a fixed point on a plane"
    assert not failures, failures[:3]
    scored = sum(v for k, v in table.items() if k != "no reference")
    assert scored >= bases * DISTANCES // 2, table
    if not ps.moving(structure):
        assert flagged >= MIN_FLAGGED_WITH_CONSTANT_POSITIONS, (flagged, table)
