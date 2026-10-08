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
   the middle of the cell of whichever is nearer to one (MADD-ANO-240);
1. over the probe step the gather's finite difference is its
   Jacobian-vector product, for a point anywhere: mid-cell, on an interior
   lattice plane, on either face of the hull, a rounding inside or outside
   either face, far outside (MADD-ANO-240: a point on the top face stepped
   out of the hull, where the kernel clamps, and the difference was zero);
2. the step is ``sqrt`` of the coarser of the geometry's rounding and the
   pass's, and a float32 gather read at float64 positions is resolved by
   it and not by the geometry's own (MADD-ANO-241);
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
    """MADD-ANO-241: a step sized for the float64 positions alone (1.5e-8
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
