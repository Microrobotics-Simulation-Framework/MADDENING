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
error, flagged usable.  The flag now stands only where no lattice plane is
inside the Newton-Kantorovich ball around the iterate
(``_bounds._kantorovich_ball_plane_margin`` has the argument and its five
assumptions).

``spectral_error_bound`` **has the same hole and the same rule.**  It is a
bound to the fixed point of the polynomial the pass is in the iterate's
lattice cells, and with that fixed point past a plane -- the iterate and
the Newton point both short of it -- the pass has no fixed point there: the
flag stood on a bound 17 to 1,294 times under the distance where the next
cell expands.  So every row scores **both** flags, with a fourth point
beside the three: the fixed point of the iterate's polynomial piece
(``plane_sides.cell_fixed_point``), which the rule's argument puts in the
iterate's cell wherever a flag stands.  The cases of that audit are
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
import functools
import json
import os
import re
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

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
    than one structure where it is reachable; a row whose gradient flag
    stands (the rule does not withdraw everything); a row within a few float
    resolutions of a plane; float32 and float64, both schedules, both norms."""
    rows = TABLE["rows"]
    control = [r for r in rows if "constant of the pass" in r["structure"]]
    # Positions that are constants of the pass: a plane between the
    # iterate's and the fixed point's, and both flags.
    assert {r["row"] for r in control} == {"k=* N=*", "k!=* N=*"}
    assert all(r["gradient_bound_usable"] and r["spectral_usable"] for r in control)
    rows = [r for r in rows if r not in control]
    drawn = [r for r in rows if not r["structure"].startswith("audited")]
    by_row = collections.Counter(r["row"] for r in drawn)
    assert set(by_row) == set(ps.ROWS), by_row
    # The row of this audit from three structures (the audited cases are
    # those structures' own), the row of the one before from three.
    for name in ("k!=* N!=*", "k!=* N=*"):
        told = {(c.dtype, c.mode, c.norm) for c in (
            ps.as_cfg(r["cfg"]) for r in rows if r["row"] == name and r["name"] in PER_PUSH)}
        assert len(told) >= 3, (name, told)
    assert any(r["gradient_bound_usable"] and r["row"] == "k=* N=*" for r in drawn)
    assert any(not r["gradient_bound_usable"] and r["row"] == "k=* N=*" for r in drawn)
    assert any(r["on_a_plane"] for r in drawn)
    cfgs = [ps.as_cfg(r["cfg"]) for r in rows]
    assert all(ps.moving(c) for c in cfgs) and not any(
        ps.moving(ps.as_cfg(r["cfg"])) for r in control)
    assert {c.dtype for c in cfgs} == {"float32", "float64"}
    assert {c.mode for c in cfgs} == {"gauss-seidel", "jacobi"}
    assert {c.norm for c in cfgs} == {"l2", "mixed"}
    assert sum(r["structure"].startswith("audited") for r in rows) == 5
    # No row asks for a flag on a fixed point across a plane: neither flag.
    assert not any((r["gradient_bound_usable"] or r["spectral_usable"])
                   and r["row"] != "k=* N=*" for r in rows)
    # An honest report that loses both flags to a plane in the ball (the
    # rule's price), from more than one structure.
    assert len({r["structure"] for r in drawn if r["row"] == "k=* N=*"
                and not r["on_a_plane"] and not r["spectral_usable"]}) >= 2


def _check_rows(structure: str) -> None:
    """Row by row: the sides are the pinned ones (the instrument still
    expresses the row), every number whose flag is set holds against the
    reference, and the gradient flag is the pinned one -- withdrawn on every
    row that has a plane between two of the three points, standing on the
    row that has none in the ball."""
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
                # Every position a constant of the pass: nothing to cross.
                assert margin == np.inf == slots["plane_limit"], told
            if row["gradient_bound_usable"]:
                assert margin > 1.0 and "not_usable_reason" not in report, told
            if not margin > 1.0:
                # A plane in the ball: both flags, the reason names the
                # plane, the numbers stay, and the margin is the step's.
                assert not report["spectral_usable"], told
                assert not report["gradient_bound_usable"], told
                assert "lattice plane" in report["not_usable_reason"], told
                assert np.isfinite(report["spectral_error_bound"]), told
            elif ps.moving(cfg) and where["p_ok"] and not where["on_a_plane"]:
                # The rule's argument, on the reference: with no plane in
                # the ball the fixed point of the iterate's polynomial is in
                # the iterate's cell, and is the pass's.
                assert where["p_in_cell"] and where["p_is_fixed_point"], told


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
    reference), no flagged number is wrong, and the flags are the rule's --
    both withdrawn with the reason wherever the plane is in the ball, both
    standing where it is not.  Returns the distance over the bound per row."""
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
            assert pp.wrong_numbers(report, where) == [], told
            flagged = kind == "clear"
            assert (slots["plane_margin"] > 1.0) is flagged, told
            assert bool(report["spectral_usable"]) is flagged, told
            assert bool(report["gradient_bound_usable"]) is flagged, told
            assert np.isfinite(report["spectral_error_bound"]), told
            if flagged:
                assert "not_usable_reason" not in report, told
            else:
                assert "Newton-Kantorovich ball" in report["not_usable_reason"], told
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
    its fixed point past the plane.  Both flags are withdrawn whatever the
    next cell does -- expanding (the bound tens of times under the
    distance), contracting at 0.999 or at 0.99 -- and stand with the fixed
    point deep in the cell."""
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
    from the plane, four tolerances, ten structures, both dtypes: no report
    with a flag set has its fixed point further than twice the bound, a
    lattice plane between the points it rests on, or the polynomial's fixed
    point past a plane.  The realised counts are printed."""
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
                if (report["spectral_usable"] or report["gradient_bound_usable"]) and not (
                        slots["plane_margin"] > 1.0):
                    bad.append(f"a flag with plane_margin {slots['plane_margin']}")
                if bad:
                    failures.append((tolerance, s, bad, where))
    print(f"\n[placed] {structure} {dtype}: " + ", ".join(
        f"{key}: {count}" for key, count in sorted(counts.items(), key=str)))
    assert not failures, failures[:3]
    scored = sum(count for key, count in counts.items() if key != "no reference")
    assert scored >= len(PLACED_TOLERANCES) * 10, counts


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
    With two target anchors every position is a constant of the pass: the
    rule withdraws nothing there, and the gradient's flag is the spectral
    one's wherever the bound is finite."""
    cell = SWEEP[index]
    structure = ps.Cfg(**cell)
    rng = np.random.default_rng([SEED, index])
    table = collections.Counter()
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
                if not ps.moving(case):
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
