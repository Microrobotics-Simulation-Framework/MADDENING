"""What the coupling diagnostics ask of a ``multilinear_grid`` mapping's lattice.

Experimental.  The diagnostics of a group with a geometry edge read two
things off the mapping besides its value (MAP-045, MAP-049):

* ``_probe_step``: a small step of every position along which the stencil
  is one polynomial, so that a finite difference over it is the kernel's
  derivative.  The self-check of the pass's Jacobian-vector product
  compares the two.
* ``_plane_distance``: how far every coordinate is from the nearest place
  the stencil changes polynomial.  The report's lattice-plane limit is
  built on it (``_bounds._geometry_plane_limit``, tested here on a toy
  pass; on compiled groups in
  ``tests/property/test_coupling_geometry_search.py``).

The statements, each on float32 and float64 where a dtype enters:

0. a step taken *beside* a second set of positions that move with the
   first (the positions a pass derives from a member's pre-step ones)
   carries neither across a lattice plane: each coordinate steps towards
   the middle of the cell of whichever is nearer to one (MADD-ANO-243);
1. over the probe step the gather's finite difference is its
   Jacobian-vector product, for a point anywhere: mid-cell, on an interior
   lattice plane, on either face of the hull, a rounding inside or outside
   either face, far outside (MADD-ANO-243: a point on the top face stepped
   out of the hull, where the kernel clamps, and the difference was zero);
2. the step is ``sqrt`` of the coarser of the geometry's rounding and the
   pass's, and a float32 gather read at float64 positions is resolved by
   it and not by the geometry's own (MADD-ANO-244);
3. the plane distance is the distance to the nearest lattice plane inside
   the hull, to the clamping face outside, zero on a plane, ``inf`` on an
   axis of one point and NaN for a coordinate that is not finite; a move
   shorter than it changes no stencil index and one longer towards the
   plane does;
4. the lattice-plane limit of a pass is ``unit * min_j(D_j d_j) / 2`` over
   the entries of the readers' fields, zero where the pass itself moves a
   position across, NaN where it cannot be evaluated, ``inf`` with no
   reader.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _bounds
from maddening.core.coupling.grid_mapping import multilinear_grid_mapping
from tests.property.sysid_transform_grid import precision

ORIGIN, SPACING, POINTS = 0.25, 0.5, 4          # lattice points at 0.25, 0.75, 1.25, 1.75
TOP = ORIGIN + (POINTS - 1) * SPACING


def _line(n_points: int, mode: str = "consistent"):
    return multilinear_grid_mapping([ORIGIN], [SPACING], [POINTS], n_points=n_points, mode=mode)


def _places(dtype) -> dict:
    """Coordinates on the one-dimensional lattice, by name."""
    tick = float(np.spacing(np.asarray(TOP, dtype)))
    return {
        "mid-cell": ORIGIN + 1.3 * SPACING,
        "above the middle": ORIGIN + 0.8 * SPACING,
        "on an interior plane": ORIGIN + 2 * SPACING,
        "on the bottom face": ORIGIN,
        "on the top face": TOP,
        "a rounding inside the top face": TOP - tick,
        "a rounding outside the top face": TOP + tick,
        "a rounding inside the bottom face": ORIGIN + float(np.spacing(np.asarray(ORIGIN, dtype))),
        "a rounding outside the bottom face": ORIGIN - float(np.spacing(np.asarray(ORIGIN,
                                                                                 dtype))),
        "far above": TOP + 3.3 * SPACING,
        "far below": ORIGIN - 2.6 * SPACING,
    }


FIELD = np.asarray([1.0, -0.5, 2.0, 0.75])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_finite_difference_over_the_probe_step_is_the_kernel_s_derivative(dtype):
    """The premise of the self-check, point by point.  On the top face the
    kernel's derivative is the interior one (the last cell's slope); a
    step out of the hull reads a difference of zero there."""
    places = _places(dtype)
    with precision(dtype == "float64"):
        mapping = _line(len(places))
        pos = jnp.asarray(list(places.values()), dtype)[:, None]
        field = jnp.asarray(FIELD, dtype)
        step = mapping._probe_step(pos)                                # noqa: SLF001
        assert step.shape == pos.shape and step.dtype == pos.dtype
        assert np.all(np.asarray(step) != 0)
        moved = pos + step
        realised = moved - pos
        base, product = jax.jvp(lambda g: mapping.apply(field, geom=g), (pos,), (realised,))
        difference = np.asarray(mapping.apply(field, geom=moved) - base, np.float64)
    product = np.asarray(product, np.float64)
    slope = np.max(np.abs(np.diff(FIELD))) / SPACING
    # The difference of two values of size one, over a step of sqrt(eps).
    allowed = 8 * float(np.finfo(dtype).eps) * np.max(np.abs(FIELD))
    for k, name in enumerate(places):
        assert abs(difference[k] - product[k]) <= allowed, (name, difference[k], product[k])
    by_name = dict(zip(places, product))
    last_slope = (FIELD[-1] - FIELD[-2]) / SPACING
    on_top = by_name["on the top face"] / float(np.asarray(realised)[list(places).index(
        "on the top face"), 0])
    assert abs(on_top - last_slope) <= 1e-3 * slope, (on_top, last_slope)
    assert by_name["far above"] == 0.0 and by_name["far below"] == 0.0
    assert float(np.asarray(step)[list(places).index("on the top face"), 0]) < 0
    assert float(np.asarray(step)[list(places).index("on the bottom face"), 0]) > 0


def test_the_probe_step_moves_no_point_of_a_grid_of_several_axes_across_a_plane():
    """Two axes of different sizes: every corner of the hull, every edge
    and the middle; the stencil's cell (its lowest corner) is the same
    after the step."""
    mapping = multilinear_grid_mapping([0.0, 1.0], [0.5, 0.25], [3, 4], n_points=9)
    xs = [0.0, 0.6, 1.0]                 # the two faces of axis 0 and inside
    ys = [1.0, 1.4, 1.75]                # the two faces of axis 1 and inside
    pos = jnp.asarray([[x, y] for x in xs for y in ys], jnp.float32)
    step = mapping._probe_step(pos)                                    # noqa: SLF001
    before, _w = mapping._stencil(pos)                                 # noqa: SLF001
    after, _w = mapping._stencil(pos + step)                           # noqa: SLF001
    assert np.array_equal(np.asarray(before), np.asarray(after))
    assert np.all(np.asarray(step)[np.asarray(pos)[:, 0] == 1.0, 0] < 0)
    assert np.all(np.asarray(step)[np.asarray(pos)[:, 1] == 1.75, 1] < 0)
    assert np.all(np.asarray(step)[np.asarray(pos)[:, 0] == 0.0, 0] > 0)
    assert not np.any(np.asarray(mapping._probe_step(                  # noqa: SLF001
        jnp.asarray([[np.nan, 1.2]] * 9, jnp.float32)))[:, 0])


def test_a_step_beside_the_positions_derived_from_it_carries_neither_across_a_plane():
    """A pre-step position 0.32 of a cell above a plane, and the position
    the pass builds from it and reads, a rounding under the next plane:
    stepped upwards (towards the middle of the first one's cell) the
    second crosses; beside it, the step is downwards, and the gather of
    the derived position is one polynomial along it."""
    mapping = _line(4)
    held = jnp.asarray([[ORIGIN + 1.32 * SPACING], [ORIGIN + 0.7 * SPACING],
                        [ORIGIN + 1.32 * SPACING], [ORIGIN + 2.0 * SPACING]], jnp.float32)
    shift = jnp.asarray([[0.6799995 * SPACING], [0.0], [0.1 * SPACING], [0.31 * SPACING]],
                        jnp.float32)
    derived = held + shift           # 1.9999995, 0.7, 1.42, 2.31 spacings from the origin
    alone = np.asarray(mapping._probe_step(held))[:, 0]                 # noqa: SLF001
    beside = np.asarray(mapping._probe_step(held, beside=derived))[:, 0]  # noqa: SLF001
    assert alone[0] > 0 and beside[0] < 0, "the derived position is the one beside a plane"
    assert beside[1] == alone[1] < 0                      # the same place: the same step
    assert beside[2] == alone[2] > 0                      # the held one is the nearer
    assert beside[3] == alone[3] > 0                      # on a plane: into the cell above
    assert np.all(np.abs(beside) == np.abs(alone))
    field = jnp.asarray(FIELD, jnp.float32)

    def gap(step):
        realised = (held + step) - held
        base, product = jax.jvp(lambda c: mapping.apply(field, geom=c + shift), (held,),
                                (realised,))
        difference = mapping.apply(field, geom=held + step + shift) - base
        return float(jnp.max(jnp.abs(difference - product)) / jnp.max(jnp.abs(product)))

    assert gap(mapping._probe_step(held, beside=derived)) <= 0.01      # noqa: SLF001
    assert gap(mapping._probe_step(held)) > 0.25, (                     # noqa: SLF001
        "premise: stepped for the held position alone, the derived one crosses")
    # One-dimensional geometries and other dtypes of the second set.
    flat = mapping._probe_step(held[:, 0], beside=np.asarray(derived, np.float64)[:, 0])  # noqa: SLF001
    assert flat.shape == (4,) and flat.dtype == held.dtype
    assert np.array_equal(np.asarray(flat), beside)


def test_the_probe_step_is_sized_for_the_coarser_of_the_geometry_and_the_pass():
    eps32, eps64 = float(np.finfo(np.float32).eps), float(np.finfo(np.float64).eps)
    with precision(True):
        mapping = _line(1)
        wide = jnp.asarray([[ORIGIN + 1.3 * SPACING]], jnp.float64)
        narrow = wide.astype(jnp.float32)
        size = lambda geom, **kw: abs(float(mapping._probe_step(geom, **kw)[0, 0]))   # noqa: E731,SLF001
        assert size(wide) == pytest.approx(math.sqrt(eps64) * SPACING, rel=1e-12)
        assert size(wide, eps=eps32) == pytest.approx(math.sqrt(eps32) * SPACING, rel=1e-12)
        assert size(wide, eps=eps64) == size(wide)
        assert size(narrow, eps=eps64) == size(narrow) == pytest.approx(
            math.sqrt(eps32) * SPACING, rel=1e-6)
        assert mapping._probe_step(wide, eps=eps32).dtype == jnp.float64   # noqa: SLF001


def _mixed_gap(eps):
    """The self-check's gap of a toy pass whose positions are float64 and
    whose gather is computed in float32: ``pos' = pos + 0.4 gather(field,
    pos)``, the positions a constant of the pass."""
    mapping = _line(3)
    field = jnp.asarray(FIELD, jnp.float32)

    def step_pure(x, pos):
        sampled = mapping.apply(field, geom=pos)                    # float32 weights and values
        return x * 0.0 + (pos[:, 0] + 0.4 * sampled.astype(jnp.float64)), jnp.zeros(())

    pos = jnp.asarray([[0.41], [0.97], [1.52]], jnp.float64)
    x = step_pure(jnp.zeros(3, jnp.float64), pos)[0]
    step = mapping._probe_step(pos, eps=eps)                           # noqa: SLF001
    resolution = jnp.full(3, 8 * np.finfo(np.float64).eps)
    return float(_bounds._geometry_product_gap(                        # noqa: SLF001
        step_pure, x, (pos,), [(jnp.zeros_like(x), [step])], 1.0 / jnp.abs(x), resolution,
        np.zeros(3, np.int32), 1))


def test_float32_fields_at_float64_positions_pass_the_self_check_at_the_pass_s_step():
    """MADD-ANO-244: a step sized for the float64 positions alone (1.5e-8
    of a spacing) is below what the float32 gather resolves, and the
    difference is rounding against a product that is not: the gap of an
    honest pass read 0.27 to 1.0.  At the pass's coarsest rounding it is
    the finite difference's own error."""
    with precision(True):
        honest = _mixed_gap(float(np.finfo(np.float32).eps))
        premise = _mixed_gap(None)
    assert honest <= 0.02, honest
    assert premise > _bounds.GEOMETRY_GAP_TOLERANCE, (
        f"premise: the geometry's own step is resolved here (gap {premise:.3g})")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_plane_distance_is_to_the_nearest_plane_inside_and_to_the_face_outside(dtype):
    want = {
        "mid-cell": 0.3 * SPACING, "above the middle": 0.2 * SPACING,
        "on an interior plane": 0.0, "on the bottom face": 0.0, "on the top face": 0.0,
        "far above": 3.3 * SPACING, "far below": 2.6 * SPACING,
    }
    places = _places(dtype)
    with precision(dtype == "float64"):
        mapping = _line(len(places))
        pos = jnp.asarray(list(places.values()), dtype)[:, None]
        got = mapping._plane_distance(pos)                             # noqa: SLF001
        assert got.shape == pos.shape and got.dtype == pos.dtype
        flat = mapping._plane_distance(pos[:, 0])                      # noqa: SLF001
        assert flat.shape == (len(places),) and np.array_equal(np.asarray(flat),
                                                               np.asarray(got)[:, 0])
    got = dict(zip(places, np.asarray(got, np.float64)[:, 0]))
    slack = 8 * float(np.finfo(dtype).eps) * 4 * SPACING
    for name, value in want.items():
        assert abs(got[name] - value) <= slack, (name, got[name], value)
    for name in places:
        if "a rounding" in name:
            assert 0 < got[name] <= slack, (name, got[name])


def test_the_plane_distance_of_an_axis_of_one_point_and_of_a_coordinate_that_is_not_finite():
    mapping = multilinear_grid_mapping([0.0, 1.0], [0.5, 0.25], [1, 3], n_points=3)
    pos = jnp.asarray([[0.3, 1.1], [np.nan, 1.3], [-4.0, np.inf]], jnp.float32)
    got = np.asarray(mapping._plane_distance(pos))                     # noqa: SLF001
    assert got[0, 0] == np.inf and got[2, 0] == np.inf       # one point: no plane on axis 0
    assert got[0, 1] == pytest.approx(0.1, rel=1e-5)
    assert np.isnan(got[1, 0]) and np.isnan(got[2, 1])
    assert got[1, 1] == pytest.approx(0.05, rel=1e-5)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_move_shorter_than_the_plane_distance_keeps_the_stencil_s_cell(dtype):
    """Random points of a two-axis grid, inside the hull and outside it:
    0.9 of the distance, in either direction on either axis, changes no
    stencil index; inside the hull, 1.1 of it towards the nearest plane of
    an axis changes one (or reaches a face)."""
    rng = np.random.default_rng(11)
    origin, spacing, shape = (40.0, -3.25), (0.5, 0.25), (4, 3)
    with precision(dtype == "float64"):
        mapping = multilinear_grid_mapping(origin, spacing, shape, n_points=64)
        u = np.stack([rng.uniform(-0.6, shape[a] - 0.4, 64) for a in range(2)], axis=1)
        pos = jnp.asarray(np.asarray(origin) + u * np.asarray(spacing), dtype)
        dist = np.asarray(mapping._plane_distance(pos), np.float64)    # noqa: SLF001
        base = np.asarray(mapping._stencil(pos)[0])                    # noqa: SLF001
        for axis in range(2):
            for sign in (-1.0, 1.0):
                step = np.zeros((64, 2))
                step[:, axis] = sign * 0.9 * dist[:, axis]
                near, _w = mapping._stencil(pos + jnp.asarray(step, dtype))   # noqa: SLF001
                same = np.all(np.asarray(near) == base, axis=1)
                assert np.all(same), (axis, sign, np.flatnonzero(~same))
        inside = np.all((u > 0) & (u < np.asarray(shape) - 1), axis=1)
        lattice = (np.asarray(pos, np.float64) - np.asarray(origin)) / np.asarray(spacing)
        towards = np.where(lattice - np.floor(lattice) < 0.5, -1.0, 1.0)
        for axis in range(2):
            step = np.zeros((64, 2))
            step[:, axis] = towards[:, axis] * 1.1 * dist[:, axis]
            after = (np.asarray(pos + jnp.asarray(step, dtype), np.float64)
                     - np.asarray(origin)) / np.asarray(spacing)
            crossed = np.floor(after[:, axis]) != np.floor(lattice[:, axis])
            assert np.all(crossed[inside]), axis
    assert inside.sum() >= 10 and (~inside).sum() >= 10


# ---------------------------------------------------------------------------
# The lattice-plane limit of a pass (``_bounds._geometry_plane_limit``)
# ---------------------------------------------------------------------------


def _limit(positions, *, pull=0.0, weights=None, unit=1.0, readers="default",
           mapping=None, dtype=jnp.float32, gain=0.5):
    """The limit of the toy pass ``F([a, p]) = [gain a, p0 + pull a]``: two
    values and two positions on the one-dimensional lattice, the positions
    read by *mapping*."""
    mapping = _line(2) if mapping is None else mapping
    start = jnp.asarray(positions, dtype)

    def step_pure(x, p0):
        return jnp.concatenate([gain * x[:2], p0 + pull * x[:2]]), jnp.zeros(())

    x = jnp.concatenate([jnp.asarray([1.0, -2.0], dtype), start])
    if readers == "default":
        readers = [(np.asarray([2, 3]), (2, 1), np.dtype(dtype), mapping)]
    weights = jnp.asarray([1.0, 1.0, 0.5, 0.5] if weights is None else weights, dtype)
    return float(_bounds._geometry_plane_limit(                        # noqa: SLF001
        step_pure, x, (start,), readers, weights, unit))


def test_the_plane_limit_is_the_nearest_weighted_plane_distance_over_the_reach():
    mid, near = ORIGIN + 1.3 * SPACING, ORIGIN + 2.02 * SPACING
    reach = _bounds.GEOMETRY_PLANE_REACH
    assert reach == 2.0
    # Distances 0.15 and 0.01, each times the weight 0.5.
    assert _limit([mid, near]) == pytest.approx(0.5 * 0.01 / reach, rel=1e-4)
    assert _limit([mid, mid]) == pytest.approx(0.5 * 0.15 / reach, rel=1e-5)
    # The entry's own weight, and the norm's constant.
    assert _limit([mid, near], weights=[1.0, 1.0, 0.01, 4.0]) == pytest.approx(
        0.01 * 0.15 / reach, rel=1e-5)
    assert _limit([mid, mid], unit=0.125) == pytest.approx(0.125 * 0.5 * 0.15 / reach, rel=1e-5)
    # The hull's faces are planes; outside, the face clamped to.
    assert _limit([mid, TOP - 0.004]) == pytest.approx(0.5 * 0.004 / reach, rel=1e-3)
    assert _limit([mid, TOP + 0.03]) == pytest.approx(0.5 * 0.03 / reach, rel=1e-4)
    assert _limit([ORIGIN - 0.02, mid]) == pytest.approx(0.5 * 0.02 / reach, rel=1e-4)
    assert _limit([mid, TOP]) == 0.0


def test_the_plane_limit_reads_the_readers_fields_and_every_mapping_that_reads_them():
    mid = ORIGIN + 1.3 * SPACING
    assert _limit([mid, mid], readers=[]) == math.inf
    # The value entries are not positions: a reader of entries 0 and 1
    # (values 1.0 and -2.0, which this pass leaves where they are: 0.25
    # from a plane, and far below the hull).
    other = [(np.asarray([0, 1]), (2, 1), np.dtype(np.float32), _line(2))]
    assert _limit([mid, mid], readers=other, gain=1.0) == pytest.approx(0.25 / 2.0, rel=1e-5)
    # A second mapping on a finer lattice reads the same positions.
    fine = multilinear_grid_mapping([ORIGIN], [SPACING / 8], [25], n_points=2)
    both = [(np.asarray([2, 3]), (2, 1), np.dtype(np.float32), _line(2)),
            (np.asarray([2, 3]), (2, 1), np.dtype(np.float32), fine)]
    # 1.3 spacings is 10.4 fine ones: 0.4 of a sixteenth.
    assert _limit([mid, mid], readers=both) == pytest.approx(
        0.5 * 0.4 * SPACING / 8 / 2.0, rel=1e-4)


def test_the_plane_limit_is_zero_where_the_pass_moves_a_position_across():
    """The position a Gauss-Seidel sweep reads after its holder's update
    is the pass's own output: where that is further from the iterate's
    than the iterate is from a plane, the limit is zero."""
    mid, near = ORIGIN + 1.3 * SPACING, ORIGIN + 2.02 * SPACING
    # ``p' = p0 + pull a``: the iterate holds p0, the pass returns p0 + pull a.
    assert _limit([mid, near], pull=0.001) == pytest.approx(0.5 * 0.01 / 2.0, rel=1e-3)
    assert _limit([mid, near], pull=0.02) == 0.0          # entry 3 moves 0.04 > 0.01
    assert _limit([mid, mid], pull=0.02) == pytest.approx(0.5 * 0.15 / 2.0, rel=1e-5)


def test_the_plane_limit_is_not_a_number_where_it_cannot_be_evaluated():
    mid = ORIGIN + 1.3 * SPACING
    assert math.isnan(_limit([mid, np.nan]))
    assert math.isnan(_limit([np.inf, mid]))


# ---------------------------------------------------------------------------
# A position within a few float resolutions of a plane is on it
# (``_bounds.GEOMETRY_PLANE_ULPS``, ``_plane_resolution``; MADD-ANO-252)
# ---------------------------------------------------------------------------

#: Lattices whose plane the position sits beside: at a position of order
#: one, at an origin forty spacings from zero (where one float32
#: resolution is 7.6e-6 of a spacing), and at a plane on zero of a lattice
#: whose origin is far from it (the kernel rounds the offset from the
#: origin, and a position near zero there is the rounding of terms of the
#: lattice's size: the resolution is the lattice's, not the position's).
_ULP_LATTICES = {
    "order one": (0.25, 0.5, 4, 1.25),
    "origin forty spacings out": (20.0, 0.5, 6, 21.0),
    "a plane on zero, origin far": (-20.0, 0.5, 60, 0.0),
}


def _resolution(origin: float, spacing: float, n: int, dtype) -> float:
    """``eps`` times the largest coordinate magnitude of the lattice."""
    return float(np.finfo(dtype).eps) * max(abs(origin), abs(origin + (n - 1) * spacing))


def _ulps_from(plane: float, res: float, k: float, dtype) -> float:
    """The float of *dtype* about *k* resolutions *res* from *plane*."""
    return float(np.asarray(plane + k * res, dtype))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("lattice", sorted(_ULP_LATTICES))
def test_a_position_within_eight_float_resolutions_of_a_plane_is_on_it(dtype, lattice):
    origin, spacing, n, plane = _ULP_LATTICES[lattice]
    assert _bounds.GEOMETRY_PLANE_ULPS == 8.0
    with precision(dtype == "float64"):
        T = jnp.dtype(dtype)
        mapping = multilinear_grid_mapping([origin], [spacing], [n], n_points=2)
        mid = origin + 1.3 * spacing
        res = _resolution(origin, spacing, n, dtype)
        far = origin + 1000.0 * n * spacing                 # outside: its own magnitude
        got = np.asarray(mapping._plane_resolution(                    # noqa: SLF001
            jnp.asarray([[plane], [far]], T)))
        assert got.dtype == np.dtype(dtype) and got.shape == (2, 1)
        assert got[0, 0] == pytest.approx(res, rel=1e-6)
        assert got[1, 0] == pytest.approx(
            float(np.finfo(dtype).eps) * max(abs(far), abs(far - origin)), rel=1e-6)

        def step_pure(x, p0):
            return jnp.concatenate([0.5 * x[:2], p0]), jnp.zeros(())

        def limit(position):
            start = jnp.asarray([mid, position], T)
            x = jnp.concatenate([jnp.asarray([1.0, -2.0], T), start])
            readers = [(np.asarray([2, 3]), (2, 1), np.dtype(dtype), mapping)]
            return float(_bounds._geometry_plane_limit(                # noqa: SLF001
                step_pure, x, (start,), readers, jnp.asarray([1.0, 1.0, 0.5, 0.5], T), 1.0))

        for k in (0.3, 1.0, 5.0):
            for sign in (-1.0, 1.0):
                assert limit(_ulps_from(plane, res, sign * k, dtype)) == 0.0, (k, sign)
        for sign in (-1.0, 1.0):
            away = _ulps_from(plane, res, sign * 50.0, dtype)
            # Fifty resolutions, to the rounding of the position itself.
            assert limit(away) == pytest.approx(0.5 * 50.0 * res / 2.0, rel=0.05), sign


def test_the_window_is_taken_at_the_position_the_pass_builds_too():
    """The iterate fifty resolutions before a plane and the position the
    pass returns a third of one before it, on the same side: the move is
    shorter than the iterate's distance, and the limit is zero because
    what the sweep reads is on the plane."""
    plane = ORIGIN + 2 * SPACING
    res = _resolution(ORIGIN, SPACING, POINTS, np.float32)
    at_50, at_third = np.float32(plane - 50 * res), np.float32(plane - res / 3)
    mapping, mid = _line(2), ORIGIN + 1.3 * SPACING

    def limit(built):
        def step_pure(x, target):
            return jnp.concatenate([0.5 * x[:2], target]), jnp.zeros(())

        x = jnp.asarray([1.0, -2.0, mid, at_50], jnp.float32)
        readers = [(np.asarray([2, 3]), (2, 1), np.dtype(np.float32), mapping)]
        return float(_bounds._geometry_plane_limit(                    # noqa: SLF001
            step_pure, x, (jnp.asarray([mid, built], jnp.float32),), readers,
            jnp.asarray([1.0, 1.0, 0.5, 0.5], jnp.float32), 1.0))

    assert limit(at_third) == 0.0
    assert limit(np.float32(plane - 40 * res)) == pytest.approx(0.5 * 50 * res / 2.0, rel=0.05)


# ---------------------------------------------------------------------------
# The lattice-plane margin of the Kantorovich ball around the iterate
# (``_bounds._kantorovich_ball_plane_margin``; MADD-ANO-252)
# ---------------------------------------------------------------------------


def _margin(positions, *, radius=0.01, built_k=None, built_n=None, weights=None,
            readers="default", moved=(0.0, 0.0, 0.0, 0.0)):
    """The margin of two values and two positions on the one-dimensional
    lattice: the iterate's positions, the ones the pass builds at the
    iterate and at the Newton point (the iterate's own by default), and
    the Newton step's weighted move of each entry (none by default)."""
    mapping = _line(2)
    start = np.asarray(positions, np.float32)
    values = np.asarray([1.0, -2.0], np.float32)

    def flat(pos):
        return jnp.asarray(np.concatenate([values, np.asarray(pos, np.float32)]))

    if readers == "default":
        readers = [(np.asarray([2, 3]), (2, 1), np.dtype(np.float32), mapping)]
    weights = jnp.asarray([1.0, 1.0, 0.5, 0.5] if weights is None else weights, jnp.float32)
    return float(_bounds._kantorovich_ball_plane_margin(               # noqa: SLF001
        readers, flat(start), flat(start if built_k is None else built_k),
        flat(start if built_n is None else built_n), weights,
        jnp.asarray(moved, jnp.float32), jnp.asarray(radius, jnp.float32)))


def test_the_ball_margin_is_the_nearest_weighted_plane_distance_over_the_radius():
    mid, near = ORIGIN + 1.3 * SPACING, ORIGIN + 2.02 * SPACING
    # Eight float resolutions of the lattice: the least a plane is away by.
    window = 8.0 * _resolution(ORIGIN, SPACING, POINTS, np.float32)
    assert 1e-6 < window < 2e-6
    # Distances 0.15 and 0.01 less the window, each times the weight 0.5,
    # over the radius.
    assert _margin([mid, near]) == pytest.approx(0.5 * (0.01 - window) / 0.01, rel=2e-5)
    assert _margin([mid, mid]) == pytest.approx(0.5 * (0.15 - window) / 0.01, rel=2e-6)
    assert _margin([mid, mid], radius=0.3) == pytest.approx(0.5 * (0.15 - window) / 0.3, rel=2e-6)
    # In the units of the norm, not of the position: the entry's weight.
    assert _margin([mid, near], weights=[1.0, 1.0, 0.01, 4.0]) == pytest.approx(
        0.01 * (0.15 - window) / 0.01, rel=2e-6)
    # A face of the hull is a plane.
    assert _margin([mid, TOP - 0.004]) == pytest.approx(0.5 * (0.004 - window) / 0.01, rel=1e-4)
    # The Newton step's own move of the entry is added to its reach, and
    # only to its own.
    assert _margin([mid, near], moved=[9.0, 9.0, 0.0, 0.015]) == pytest.approx(
        0.5 * (0.01 - window) / (0.015 + 0.01), rel=2e-5)
    assert _margin([mid, near], moved=[9.0, 9.0, 0.5, 0.0]) == pytest.approx(
        min(0.5 * (0.15 - window) / (0.5 + 0.01), 0.5 * (0.01 - window) / 0.01), rel=2e-5)
    # An entry the norm does not read is bounded by nothing.
    assert _margin([mid, mid], weights=[1.0, 1.0, 0.5, 0.0]) == 0.0
    # Within eight float resolutions of a plane: on it.  And the window is
    # off the distance wherever the plane is in the ball: twelve
    # resolutions away, the margin is of four.
    plane = ORIGIN + 2 * SPACING
    assert _margin([mid, plane + 5.0 / 8.0 * window]) == 0.0
    assert _margin([mid, plane + 1.5 * window], radius=1e-6) == pytest.approx(
        0.5 * 0.5 * window / 1e-6, rel=0.1)


def test_the_ball_margin_reads_only_the_positions_that_move_with_the_iterate():
    mid, near = ORIGIN + 1.3 * SPACING, ORIGIN + 2.02 * SPACING
    # No reader (every position a constant of the pass): no plane can come
    # between the iterate and the fixed point.
    assert _margin([mid, near], readers=[]) == math.inf
    only_first = [(np.asarray([2]), (1, 1), np.dtype(np.float32), _line(1))]
    assert _margin([mid, near], readers=only_first) == pytest.approx(0.5 * 0.15 / 0.01, rel=1e-4)


def test_the_ball_margin_is_zero_where_a_position_the_pass_builds_leaves_the_cell():
    """At the iterate and at the Newton point: the position a Gauss-Seidel
    sweep reads after its holder's update is the pass's own output."""
    mid, near = ORIGIN + 1.3 * SPACING, ORIGIN + 2.02 * SPACING
    stays, leaves = [mid, near + 0.004], [mid, near - 0.03]
    assert _margin([mid, near], built_k=stays, built_n=stays) == pytest.approx(0.5, rel=1e-3)
    assert _margin([mid, near], built_k=leaves) == 0.0
    assert _margin([mid, near], built_n=leaves) == 0.0
    # Away from the plane by more than the iterate's distance: another
    # cell's plane may be between, and nothing here says it is not.
    assert _margin([mid, near], built_n=[mid, near + 0.02]) == 0.0
    # On the plane itself, from the same side.
    assert _margin([mid, near], built_k=[mid, ORIGIN + 2 * SPACING]) == 0.0


def test_the_ball_margin_is_not_a_number_where_it_cannot_be_evaluated():
    mid = ORIGIN + 1.3 * SPACING
    assert math.isnan(_margin([mid, np.nan]))
    assert math.isnan(_margin([mid, mid], built_n=[mid, np.inf]))
    assert math.isnan(_margin([mid, mid], radius=np.nan))
    assert math.isnan(_margin([mid, mid], moved=[0.0, 0.0, np.nan, 0.0]))
    assert _margin([mid, mid], radius=0.0) == math.inf
    assert _margin([mid, ORIGIN + 2 * SPACING], radius=0.0) == 0.0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_radius_the_step_takes_is_two_newton_steps_and_the_float_floor(dtype):
    """``_gradient_error_bound_and_plane_margin_at``: on an affine pass the
    Newton step is the distance to the fixed point, and the stored margin
    is the weighted plane distance over the entry's own move plus ``eta +
    floor``, the floor between nothing and two more ``eta`` -- and without
    the readers the return is the bound alone."""
    with precision(dtype == "float64"):
        T = jnp.dtype(dtype)
        mapping = multilinear_grid_mapping([ORIGIN], [SPACING], [POINTS], n_points=1)
        gain, pull = 0.5, 0.25
        fixed = np.asarray([0.8, ORIGIN + 1.3 * SPACING])           # value, position

        def step_pure(x, bias):
            # ``F(x) = fixed + A (x - fixed)``, the position pulled by the value.
            a = x[0] - fixed[0]
            return jnp.stack([fixed[0] + gain * a + bias[0], fixed[1] + pull * a]).astype(T), \
                jnp.zeros(())

        offset = 0.02
        x_k = jnp.asarray([fixed[0] + offset, fixed[1] + 2 * pull * offset], T)
        weights = jnp.asarray([1.0 / 0.82, 1.0 / float(x_k[1])], T)
        readers = [(np.asarray([1]), (1, 1), np.dtype(dtype), mapping)]
        consts = (jnp.zeros((1,), T) + jnp.asarray(0.0, T),)
        args = (step_pure, x_k, consts, weights, jnp.asarray(gain, T), jnp.asarray(0.0, T),
                jnp.asarray(1.0 / (1.0 - gain), T))
        # A resolution the float64 case can see: a twentieth of the residual.
        resolution = jnp.full((2,), 0.05 * (1.0 - gain) * offset / 0.82, T)
        alone = _bounds._gradient_error_bound_at(*args, resolution=resolution)   # noqa: SLF001
        bound, margin = _bounds._gradient_error_bound_and_plane_margin_at(       # noqa: SLF001
            *args, resolution, readers)
        assert np.ndim(alone) == 0 and float(bound) == pytest.approx(float(alone), rel=1e-5)
        w = np.asarray(weights, np.float64)
        delta = np.asarray([fixed[0], fixed[1]]) - np.asarray(x_k, np.float64)
        eta = float(np.linalg.norm(w * delta))
        nearest = w[1] * (float(x_k[1]) - (ORIGIN + SPACING))        # the plane below
        radius = nearest / float(margin) - abs(w[1] * delta[1])
        assert abs(w[1] * delta[1]) > 0.1 * eta                 # the entry's move counts
        assert eta * 1.02 < radius <= 3.0 * eta * 1.01, (radius, eta)
        assert _bounds.GEOMETRY_PLANE_REACH == 2.0


# ---------------------------------------------------------------------------
# The report's flags: none for a group that solves positions; a smooth
# group's where every position is a constant of the pass
# ---------------------------------------------------------------------------

_SOLVED = ("M.pos",)


def _flags(**changed):
    """``_geometry_flags`` of an honest report of a group whose positions
    are constants of the pass (both flags stand), with *changed* inputs."""
    from maddening.core.coupling import _group_layout       # noqa: PLC0415

    inputs = dict(solved=(), bound=1e-3, gradient_bound=2e-2, rho=0.5, arnoldi_residual=1e-9,
                  settled=True, precision_limited=False, declared=False, limit=np.inf,
                  margin=np.inf, fraction=0.05, steps=8)
    inputs.update(changed)
    return _group_layout._geometry_flags(["M.y->G.deposit"], **inputs)   # noqa: SLF001


def test_a_group_that_solves_positions_has_no_flag_whatever_its_numbers_read():
    """The rule of 0.4.0 (MADD-ANO-252): with a position the pass reads
    from the iterate, both flags are ``False`` at every margin, limit,
    bound and gradient bound -- the readings that set them under the three
    earlier rules included -- and the reason is one sentence, the same at
    every reading: what the group does, that the numbers are uncertified,
    and the two ways to have the flags."""
    above = float(np.nextafter(1.0, 2.0))
    reasons = set()
    for margin in (np.inf, 1e30, 186.0, 3.0, above, 1.0, 0.25, 0.0, -1.0, np.nan, None):
        for limit in (np.inf, 1.0, 0.0, np.nan, None):
            for gradient in (2e-2, 0.0, np.inf, np.nan):
                for more in ({}, {"bound": 0.0}, {"bound": np.inf, "settled": False},
                             {"precision_limited": True, "declared": True},
                             {"precision_limited": True}):
                    spectral, gradient_flag, reason = _flags(
                        solved=_SOLVED, margin=margin, limit=limit, gradient_bound=gradient,
                        **more)
                    assert (spectral, gradient_flag) == (False, False), (
                        margin, limit, gradient, more)
                    reasons.add(reason)
    assert len(reasons) == 1, reasons
    reason = next(iter(reasons))
    for told in ("solves position(s) ['M.pos']", "edge(s) ['M.y->G.deposit']",
                 "0.4.0 does not certify a bound for such a group",
                 "spectral_usable and gradient_bound_usable are False on every step",
                 "makes the pass another polynomial", "MADD-ANO-252",
                 "reported as computed, uncertified",
                 "a target-anchored geometry read by update",
                 "positions held by a node outside the group",
                 "no convergence_norm restores them for this group in 0.4.0"):
        assert told in reason, (told, reason)
    # The reasons of the rules that no longer decide anything are gone.
    for gone in ("Newton-Kantorovich ball", "times spectral_error_bound of a",
                 "tighter tolerance", "is on a lattice plane", "did not record"):
        assert gone not in reason, (gone, reason)
    # More than one solved field: each is named, in the order committed.
    assert "['G.x', 'M.pos']" in _flags(solved=("G.x", "M.pos"))[2]
    # No estimate: the flags are False and the NaN numbers are the reason.
    assert _flags(solved=_SOLVED, rho=np.nan, bound=np.nan, gradient_bound=np.nan,
                  settled=False, margin=np.nan, limit=np.nan) == (False, False, None)


def test_constant_positions_keep_a_smooth_group_s_flags_only_where_the_step_recorded_them():
    """Fail closed: with no position solved, the flags stand only where
    both slots read ``inf``, which is what this build's step writes for
    such a pass.  A slot that is absent, not a number, not numeric, or any
    finite number (the reading of a step that had a reader), and a missing
    record of what the pass solves, set no flag; the reason says the
    record is missing and names no lattice plane."""
    assert _flags() == (True, True, None)
    above = float(np.nextafter(1.0, 2.0))
    unrecorded = (None, np.nan, 3.0, above, 1.0, 0.0, -1.0, -np.inf, 1e30, "inf?", [1.0, 2.0],
                  np.array([np.inf, np.inf]))
    for slot in ("margin", "limit"):
        for value in unrecorded:
            for gradient in (2e-2, np.inf, np.nan):
                spectral, gradient_flag, reason = _flags(
                    gradient_bound=gradient, **{slot: value})
                assert (spectral, gradient_flag) == (False, False), (slot, value, gradient)
                assert "did not record" in reason and "M.y->G.deposit" in reason, reason
                assert "lattice plane" not in reason and "solves position" not in reason
    spectral, gradient_flag, reason = _flags(solved=None)
    assert (spectral, gradient_flag) == (False, False) and "did not record" in reason
    # A zero-dimensional array is the slot's own form.
    assert _flags(margin=np.asarray(np.inf, np.float32),
                  limit=np.asarray(np.inf, np.float64)) == (True, True, None)
    assert _flags(margin=np.asarray(2.5, np.float32))[:2] == (False, False)


def test_a_false_flag_of_constant_positions_names_every_cause_and_no_lattice_plane():
    """Every cause of a ``False`` flag of a step that computed the estimate,
    and nothing else: the float floor, an estimate that did not settle, a
    bound that is not finite, a gradient bound that was not computed (NaN)
    or did not certify (``inf``).  No lattice plane is named with any of
    them.  With no estimate (a NaN radius) the numbers say so and there is
    no reason."""
    spectral, gradient, reason = _flags(precision_limited=True)
    assert (spectral, gradient) == (False, False)
    assert "float floor" in reason and "update_evaluations" in reason
    assert "lattice plane" not in reason and "M.y->G.deposit" not in reason
    assert _flags(precision_limited=True, declared=True) == (True, True, None)

    spectral, gradient, reason = _flags(settled=False, arnoldi_residual=0.3)
    assert (spectral, gradient) == (False, False)
    assert "did not settle" in reason and "0.3" in reason and "8 Krylov steps" in reason
    assert "lattice plane" not in reason and "no tolerance changes that" in reason

    spectral, gradient, reason = _flags(bound=np.inf, settled=False)
    assert (spectral, gradient) == (False, False) and "spectral_error_bound is inf" in reason
    assert "did not settle" not in reason and "lattice plane" not in reason

    for value, told, other in ((np.nan, "was not computed (NaN)", "is inf"),
                               (np.inf, "is inf", "was not computed")):
        spectral, gradient, reason = _flags(gradient_bound=value)
        assert (spectral, gradient) == (True, False), value
        assert reason.startswith("gradient_bound_usable is False (spectral_usable stands)")
        assert told in reason and other not in reason and "lattice plane" not in reason

    # Both kinds of cause: each named.
    spectral, gradient, reason = _flags(precision_limited=True, margin=0.25,
                                        gradient_bound=np.inf)
    assert (spectral, gradient) == (False, False)
    for told in ("float floor", "did not record", "read inf and 0.25",
                 "has a cause of its own", "gradient_relative_error_bound is inf"):
        assert told in reason, (told, reason)
    assert "lattice plane" not in reason and reason.endswith(
        "The numbers are reported as computed.")

    # No estimate: the flags are False and the NaN numbers are the reason.
    assert _flags(rho=np.nan, bound=np.nan, gradient_bound=np.nan, settled=False,
                  margin=np.nan, limit=np.nan) == (False, False, None)
