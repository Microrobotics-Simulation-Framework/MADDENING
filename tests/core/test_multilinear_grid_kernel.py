"""The ``multilinear_grid`` mapping kind against an independent reference.

The kind gathers a field from a uniform grid to moving points
(``mode="consistent"``) and scatters point amounts onto the grid
(``mode="conservative"``); the points are the geometry an edge hands it.
Every test here calls the kind's ``apply`` / ``apply_T`` directly and
compares it with :mod:`tests.core.multilinear_reference`, which computes
the same stencil in exact rational arithmetic with explicit loops and
shares no code with the library.

**The identities** (``K = 2**d`` corners; ``eps_g`` the geometry dtype's
eps; ``E`` the larger of the geometry's and the field's; ``S1[p] = sum_s
|W[p, s]| |u[I[p, s]]|``; ``m_c`` the number of contributions into cell
``c``):

1. the weights of a finite point sum to one, within ``2 d eps_g``;
2. the stencil is the reference's: no weight outside it, and the weights
   within ``4 eps_g max(1, max|index coordinate|)``;
3. gather matches the reference -- on a linear field inside the hull and
   on any field anywhere -- within ``(K + 4) E S1`` plus, per axis, ``4
   eps_g max(1, |u_a|)`` times the field's largest change across one cell;
4. scatter is the transpose of gather: ``<G u, f> == <u, S f>`` within
   ``(2K + 4) E sum|W f u[I]|``;
5. scatter preserves the plain sum, within ``(K + 2d + 2 + max m_c) E
   sum|f|`` -- points outside the hull included, which are clamped, never
   dropped;
6. gather commutes with a shift by one cell;
7. scaling every length by a power of two changes no bit;
8. a permutation of the points permutes gather's result bit for bit and
   leaves scatter's unchanged to rounding;
9. the float32 kernel matches the reference run on the float32 inputs,
   the float64 kernel matches it within ``64 eps64 S1``;
10. a ``vmap`` over the geometry, the field or both matches the single
    calls to rounding (whether bit for bit is recorded, never asserted);
11. a point outside the hull gives the result of its projection onto it,
    at any finite distance;
12. a non-finite coordinate poisons its own point (gather) or the corner
    cells at index 0 (scatter), and nothing else;
13. a spacing of ``2**-125`` in float32 is resolved;
14. ``layout="shaped"`` is ``"flat"`` reshaped, bit for bit;
15. the derivative with respect to a coordinate is the interior one-sided
    one on the hull and zero outside, exactly;
16. ``apply_T`` of either mode is ``apply`` of the other, bit for bit.

Grids have a different size on every axis, so a transposed stride is not a
symmetry of the test, and every dimension has points outside the hull.

The reference itself is checked at the foot of the module (exact partition
of unity, exact reproduction of multilinear fields, exact adjointness), so
a fault in it is not read as one in the library.
"""

from __future__ import annotations

import contextlib
import itertools
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.core import multilinear_reference as ref

KIND = "multilinear_grid"


def factory():
    """The library's ``multilinear_grid_mapping``.

    Imported when a test runs, not when the module is collected: until the
    kind exists every test that needs it fails by itself, with the import
    error, and the reference's own tests below still run.
    """
    from maddening.core.coupling.grid_mapping import (  # noqa: PLC0415
        multilinear_grid_mapping,
    )
    return multilinear_grid_mapping


@contextlib.contextmanager
def _x64(on: bool):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", bool(on))
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _eps(dtype) -> float:
    return float(np.finfo(np.dtype(dtype)).eps)


#: Every grid has a different size on each axis.  ``2d-thin`` has an axis
#: of one point (``i0 = i1 = 0``, ``t = 0``: its two corners coincide).
GRIDS = {
    "1d": ref.Grid((0.3,), (0.5,), (6,)),
    "2d": ref.Grid((0.3, -1.0), (0.5, 0.25), (5, 3)),
    "3d": ref.Grid((-0.2, 1.5, 0.125), (0.25, 0.5, 2.0), (4, 3, 2)),
    "2d-thin": ref.Grid((0.0, 2.0), (1.0, 0.5), (4, 1)),
}

#: ``(field dtype, geometry dtype)``.
DTYPES = [("float32", "float32"), ("float64", "float64"),
          ("float32", "float64"), ("float64", "float32")]
_DTYPE_IDS = [f"field-{f}-geom-{g}" for f, g in DTYPES]


def _needs_x64(*dtypes) -> bool:
    return any(np.dtype(d) == np.float64 for d in dtypes)


def _index_points(grid: ref.Grid, seed: int = 0, n_inside: int = 7) -> np.ndarray:
    """Index coordinates of a fixed point set: inside the hull at generic
    fractions (0.1 to 0.9 of a cell on every axis), then one point outside
    each face, then one far outside a corner."""
    rng = np.random.default_rng(1000 + seed)
    rows = []
    for _ in range(n_inside):
        rows.append([float(rng.integers(0, max(n - 1, 1))) + float(rng.uniform(0.1, 0.9))
                     if n > 1 else float(rng.uniform(-0.4, 0.4)) for n in grid.shape])
    centre = [0.5 * (n - 1) + 0.137 for n in grid.shape]
    for a in range(grid.d):
        low, high = list(centre), list(centre)
        low[a] = -0.7 - a
        high[a] = grid.shape[a] - 1 + 0.6 + a
        rows += [low, high]
    rows.append([-3.25 - a for a in range(grid.d)])
    return np.asarray(rows, np.float64)


def _points(grid: ref.Grid, gdtype, seed: int = 0) -> np.ndarray:
    return grid.points_at(_index_points(grid, seed), np.dtype(gdtype))


def _field(grid: ref.Grid, fdtype, seed: int = 0, channels: int = 0) -> np.ndarray:
    rng = np.random.default_rng(2000 + seed)
    shape = (grid.size,) + ((channels,) if channels else ())
    return np.asarray(rng.normal(size=shape) + 0.5, np.dtype(fdtype))


def _amounts(n_points: int, fdtype, seed: int = 0, channels: int = 0) -> np.ndarray:
    rng = np.random.default_rng(3000 + seed)
    shape = (n_points,) + ((channels,) if channels else ())
    return np.asarray(rng.normal(size=shape) - 0.25, np.dtype(fdtype))


def _mapping(grid: ref.Grid, n_points: int, mode: str, **kw):
    return factory()(grid.origin, grid.spacing, grid.shape, n_points=n_points, mode=mode, **kw)


def _gather(grid, field, x, **kw) -> np.ndarray:
    m = _mapping(grid, np.shape(x)[0], "consistent", **kw)
    return np.asarray(m.apply(jnp.asarray(field), None, jnp.asarray(x)))


def _scatter(grid, amounts, x, **kw) -> np.ndarray:
    m = _mapping(grid, np.shape(x)[0], "conservative", **kw)
    return np.asarray(m.apply(jnp.asarray(amounts), None, jnp.asarray(x)))


def _gather_tolerance(grid, field, x, fdtype, gdtype) -> np.ndarray:
    """Identity 3's tolerance, per point (and channel)."""
    E = max(_eps(fdtype), _eps(gdtype))
    s1 = ref.magnitudes(grid, field, x)
    extent = ref.index_extent(grid, x)                        # (n_points, d)
    variation = ref.cell_variation(grid, field)               # (d,)
    weights = 4.0 * _eps(gdtype) * (extent @ variation)       # (n_points,)
    return (grid.corners + 4) * E * s1 + weights.reshape((-1,) + (1,) * (s1.ndim - 1))


def _scatter_tolerance(grid, amounts, x, fdtype, gdtype) -> np.ndarray:
    """The transposed form of identity 3's tolerance, per cell (and channel):
    ``(m_c + K + 4) E sum|W f|`` into the cell, plus ``4 eps_g max(1,
    max|u|) |f|`` for every contribution's weight."""
    E = max(_eps(fdtype), _eps(gdtype))
    index, weight = ref.stencil(grid, x)
    w = np.abs(np.nan_to_num(ref.weights_float64(weight)))
    f = np.abs(np.asarray(amounts, np.float64)).reshape(index.shape[0], -1)
    extent = np.max(ref.index_extent(grid, x), axis=1)
    mag = np.zeros((grid.size, f.shape[1]))
    wtol = np.zeros((grid.size, f.shape[1]))
    for p in range(index.shape[0]):
        for s in range(grid.corners):
            mag[index[p, s]] += w[p, s] * f[p]
            wtol[index[p, s]] += 4.0 * _eps(gdtype) * extent[p] * f[p]
    m = ref.contributions(grid, x).reshape(-1, 1)
    out = (m + grid.corners + 4) * E * mag + wtol
    return out.reshape((grid.size,) + np.shape(amounts)[1:])


def _assert_within(got, want, tol, what: str) -> None:
    got, want, tol = np.asarray(got, np.float64), np.asarray(want), np.asarray(tol)
    gap = np.abs(got - want)
    bad = ~(gap <= tol)
    assert not np.any(bad), (
        f"{what}: off the reference by up to {np.nanmax(gap):.3e} where "
        f"{np.max(np.where(bad, tol, 0.0)):.3e} is allowed "
        f"(worst entry {np.unravel_index(np.argmax(np.where(bad, gap, 0.0)), gap.shape)})")


# ---------------------------------------------------------------------------
# What the object declares
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_the_kind_declares_its_geometry_and_has_no_weights(name):
    """``needs_geometry``, the geometry's static shape ``(n_points, d)``, an
    empty weights table, and the two sizes by mode."""
    grid = GRIDS[name]
    for mode, sizes in (("consistent", (grid.size, 11)), ("conservative", (11, grid.size))):
        m = _mapping(grid, 11, mode)
        assert m.kind == KIND and m.mode == mode
        assert m.needs_geometry is True
        assert tuple(m.geometry_shape) == (11, grid.d)
        assert m.params_pytree() == {}
        assert (m.n_source, m.n_target) == sizes
        assert type(m.n_source) is int and type(m.n_target) is int


def test_the_factory_refuses_a_grid_or_an_option_it_does_not_support():
    """Each a ``ValueError`` when the mapping is built: no axis or more than
    three, a spacing that is not positive and finite, a non-finite origin,
    an axis without a point, more than ``2**31 - 1`` grid points (the flat
    index is int32), and any ``outside=`` but ``"clamp"``."""
    make = factory()
    bad = [
        dict(origin=(), spacing=(), shape=()),
        dict(origin=(0.0,) * 4, spacing=(1.0,) * 4, shape=(2,) * 4),
        dict(origin=(0.0,), spacing=(0.0,), shape=(4,)),
        dict(origin=(0.0,), spacing=(-1.0,), shape=(4,)),
        dict(origin=(0.0,), spacing=(float("inf"),), shape=(4,)),
        dict(origin=(float("nan"),), spacing=(1.0,), shape=(4,)),
        dict(origin=(0.0,), spacing=(1.0,), shape=(0,)),
        dict(origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0), shape=(2048, 2048, 512)),
        dict(origin=(0.0,), spacing=(1.0,), shape=(4,), outside="zero"),
        dict(origin=(0.0,), spacing=(1.0,), shape=(4,), outside="periodic"),
        dict(origin=(0.0,), spacing=(1.0,), shape=(4,), mode="nearest"),
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            make(kw.pop("origin"), kw.pop("spacing"), kw.pop("shape"), n_points=3, **kw)
    # The control: the same call with nothing wrong builds.
    assert make((0.0,), (1.0,), (4,), n_points=3, outside="clamp").n_source == 4


@pytest.mark.parametrize("mode", ["consistent", "conservative"])
def test_a_non_floating_field_is_refused_at_trace(mode):
    """An integer field through the kind is a ``TypeError`` naming the kind
    and the dtype, not a field silently truncated or promoted."""
    grid = GRIDS["2d"]
    x = _points(grid, "float32")
    m = _mapping(grid, x.shape[0], mode)
    field = jnp.arange(m.n_source, dtype=jnp.int32)
    with pytest.raises(TypeError) as err:
        m.apply(field, None, jnp.asarray(x))
    assert KIND in str(err.value) and "int32" in str(err.value), str(err.value)
    with pytest.raises(TypeError):
        jax.jit(lambda f, g: m.apply(f, None, g))(field, jnp.asarray(x))


@pytest.mark.parametrize("dtypes", DTYPES, ids=_DTYPE_IDS)
def test_the_result_has_the_field_s_dtype_whatever_the_geometry_s(dtypes):
    """Weights are computed in the geometry's dtype and cast to the field's:
    a float32 field is never promoted by a float64 geometry."""
    fdtype, gdtype = dtypes
    grid = GRIDS["2d"]
    with _x64(_needs_x64(fdtype, gdtype)):
        x = _points(grid, gdtype)
        assert _gather(grid, _field(grid, fdtype), x).dtype == np.dtype(fdtype)
        assert _scatter(grid, _amounts(x.shape[0], fdtype), x).dtype == np.dtype(fdtype)


# ---------------------------------------------------------------------------
# 1, 2: the stencil
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gdtype", ["float32", "float64"])
@pytest.mark.parametrize("name", sorted(GRIDS))
def test_the_weights_of_a_finite_point_sum_to_one(name, gdtype):
    """Identity 1, by gathering a field of ones, at points inside and outside."""
    grid = GRIDS[name]
    with _x64(_needs_x64(gdtype)):
        x = _points(grid, gdtype)
        total = _gather(grid, np.ones(grid.size, np.dtype(gdtype)), x)
    gap = np.abs(total.astype(np.float64) - 1.0)
    assert np.all(gap <= 2 * grid.d * _eps(gdtype)), (name, gdtype, gap.max() / _eps(gdtype))


@pytest.mark.parametrize("gdtype", ["float32", "float64"])
@pytest.mark.parametrize("name", sorted(GRIDS))
def test_the_stencil_and_its_weights_are_the_reference_s(name, gdtype):
    """Identity 2.  Gathering the unit fields gives the kind's matrix, one
    column per cell and exactly (each product is a weight times one or
    zero): it has no entry outside the reference's stencil -- the indices
    are the reference's, on a grid whose axes all differ -- and every
    entry is the reference's weight to rounding."""
    grid = GRIDS[name]
    with _x64(_needs_x64(gdtype)):
        x = _points(grid, gdtype)
        H = _gather(grid, np.eye(grid.size, dtype=np.dtype(gdtype)), x)
    assert H.shape == (x.shape[0], grid.size)
    outside_stencil = ~ref.support(grid, x)
    assert not np.any(H[outside_stencil] != 0), (
        f"{name}: weight in a cell outside the reference's stencil: rows "
        f"{sorted(set(np.nonzero((H != 0) & outside_stencil)[0].tolist()))}")
    tol = 4.0 * _eps(gdtype) * np.max(ref.index_extent(grid, x), axis=1, keepdims=True)
    _assert_within(H, ref.dense_matrix(grid, x), np.broadcast_to(tol, H.shape),
                   f"{name} weights ({gdtype})")


# ---------------------------------------------------------------------------
# 3, 9, 11: gather against the reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtypes", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("name", sorted(GRIDS))
def test_gather_reproduces_a_linear_field_inside_the_hull(name, dtypes):
    """Identity 3: ``a . x + b`` sampled on the lattice, gathered at points
    inside the hull, against the reference evaluated at the same float
    points (which reproduces the field exactly)."""
    fdtype, gdtype = dtypes
    grid = GRIDS[name]
    slope = np.asarray([1.5, -0.75, 0.375])[:grid.d]
    with _x64(_needs_x64(fdtype, gdtype)):
        x = _points(grid, gdtype)
        x = x[ref.interior(grid, x)]
        assert x.shape[0] >= 5, "premise: points inside the hull"
        field = np.asarray(ref.sample(grid, lambda c: c @ slope + 0.25), np.dtype(fdtype))
        got = _gather(grid, field, x)
    _assert_within(got, ref.gather(grid, field, x),
                   _gather_tolerance(grid, field, x, fdtype, gdtype), f"{name} linear field")


@pytest.mark.parametrize("channels", [0, 3])
@pytest.mark.parametrize("dtypes", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("name", sorted(GRIDS))
def test_gather_matches_the_reference_at_every_point(name, dtypes, channels):
    """Identities 3, 9 and 11 on a field with no structure, at points inside
    the hull, outside each face and far outside a corner, for a plain and a
    multi-component field: the float32 kernel against the reference run on
    the float32 inputs."""
    fdtype, gdtype = dtypes
    grid = GRIDS[name]
    with _x64(_needs_x64(fdtype, gdtype)):
        x = _points(grid, gdtype, seed=1)
        field = _field(grid, fdtype, seed=1, channels=channels)
        got = _gather(grid, field, x)
    assert got.shape == (x.shape[0],) + field.shape[1:]
    _assert_within(got, ref.gather(grid, field, x),
                   _gather_tolerance(grid, field, x, fdtype, gdtype), f"{name} gather")


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_the_float64_kernel_matches_the_reference_to_64_eps(name):
    """Identity 9, the float64 half: within ``64 eps64 S1`` times the point's
    index extent ``max(1, max|u|)`` (the weights' own rounding grows with
    it) -- a kernel that did its index arithmetic in float32 for a float64
    geometry misses this by seven orders of magnitude."""
    grid = GRIDS[name]
    with _x64(True):
        x = _points(grid, "float64", seed=2)
        field = _field(grid, "float64", seed=2)
        got = _gather(grid, field, x)
        amounts = _amounts(x.shape[0], "float64", seed=2)
        spread = _scatter(grid, amounts, x)
    extent = np.max(ref.index_extent(grid, x), axis=1)
    _assert_within(got, ref.gather(grid, field, x),
                   64 * _eps("float64") * extent * ref.magnitudes(grid, field, x),
                   f"{name} float64 gather")
    w = np.abs(ref.dense_matrix(grid, x))
    _assert_within(spread, ref.scatter(grid, amounts, x),
                   64 * _eps("float64") * ((w * extent[:, None]).T @ np.abs(amounts))
                   + 64 * np.finfo(np.float64).tiny,
                   f"{name} float64 scatter")


@pytest.mark.parametrize("gdtype", ["float32", "float64"])
@pytest.mark.parametrize("name", sorted(GRIDS))
def test_a_point_outside_the_hull_is_its_projection_onto_the_hull(name, gdtype):
    """Identity 11: constant extrapolation.  Every finite point, at any
    distance (``+-1e30`` included), reads what its nearest point of the hull
    reads, and puts what it carries where that point would."""
    grid = GRIDS[name]
    with _x64(_needs_x64(gdtype)):
        x = _points(grid, gdtype, seed=3)
        far = np.asarray(list(itertools.islice(itertools.cycle([1e30, -1e30, 3e9, -7e12]),
                                               2 * grid.d))).reshape(2, grid.d)
        x = np.concatenate([x, far.astype(x.dtype)])
        projected = ref.project_onto_hull(grid, x)
        assert np.sum(np.any(projected != x, axis=1)) >= 4, "premise: points outside"
        field = _field(grid, gdtype, seed=3)
        got, on_hull = _gather(grid, field, x), _gather(grid, field, projected)
        amounts = _amounts(x.shape[0], gdtype, seed=3)
        spread, spread_on_hull = _scatter(grid, amounts, x), _scatter(grid, amounts, projected)
    tol = _gather_tolerance(grid, field, projected, gdtype, gdtype)
    _assert_within(got, ref.gather(grid, field, projected), tol, f"{name} clamped gather")
    _assert_within(got, on_hull.astype(np.float64), 2 * tol, f"{name} gather at the projection")
    stol = _scatter_tolerance(grid, amounts, projected, gdtype, gdtype)
    _assert_within(spread, ref.scatter(grid, amounts, projected), stol,
                   f"{name} clamped scatter")
    _assert_within(spread, spread_on_hull.astype(np.float64), 2 * stol,
                   f"{name} scatter at the projection")


# ---------------------------------------------------------------------------
# 4, 5, 16: scatter is the transpose
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtypes", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("name", sorted(GRIDS))
def test_scatter_matches_the_reference_in_every_cell(name, dtypes):
    """Scatter against the reference cell by cell (the transposed form of
    identity 3's tolerance), plain and multi-component."""
    fdtype, gdtype = dtypes
    grid = GRIDS[name]
    with _x64(_needs_x64(fdtype, gdtype)):
        x = _points(grid, gdtype, seed=4)
        for channels in (0, 2):
            amounts = _amounts(x.shape[0], fdtype, seed=4, channels=channels)
            got = _scatter(grid, amounts, x)
            assert got.shape == (grid.size,) + amounts.shape[1:]
            _assert_within(got, ref.scatter(grid, amounts, x),
                           _scatter_tolerance(grid, amounts, x, fdtype, gdtype),
                           f"{name} scatter (channels={channels})")


@pytest.mark.parametrize("dtypes", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("name", sorted(GRIDS))
def test_scatter_is_the_transpose_of_gather(name, dtypes):
    """Identity 4: ``<G u, f> == <u, S f>``, the inner products taken in float64."""
    fdtype, gdtype = dtypes
    grid = GRIDS[name]
    E = max(_eps(fdtype), _eps(gdtype))
    with _x64(_needs_x64(fdtype, gdtype)):
        x = _points(grid, gdtype, seed=5)
        u = _field(grid, fdtype, seed=5)
        f = _amounts(x.shape[0], fdtype, seed=5)
        Gu = _gather(grid, u, x).astype(np.float64)
        Sf = _scatter(grid, f, x).astype(np.float64)
    lhs = float(np.dot(Gu, f.astype(np.float64)))
    rhs = float(np.dot(u.astype(np.float64), Sf))
    scale = float(np.sum(ref.magnitudes(grid, u, x) * np.abs(f.astype(np.float64))))
    assert abs(lhs - rhs) <= (2 * grid.corners + 4) * E * scale, (
        f"{name}: <G u, f> = {lhs!r}, <u, S f> = {rhs!r}, "
        f"gap {abs(lhs - rhs) / (E * scale):.2f} E sum|W f u|")


@pytest.mark.parametrize("dtypes", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("name", sorted(GRIDS))
def test_scatter_preserves_the_sum(name, dtypes):
    """Identity 5: what the points carry is what the grid receives -- the
    plain sum, no cell volume -- with points outside the hull, which are
    clamped onto it and never dropped."""
    fdtype, gdtype = dtypes
    grid = GRIDS[name]
    E = max(_eps(fdtype), _eps(gdtype))
    with _x64(_needs_x64(fdtype, gdtype)):
        x = _points(grid, gdtype, seed=6)
        far = np.full((1, grid.d), -1e30, x.dtype)
        x = np.concatenate([x, far])
        assert np.sum(np.any(ref.project_onto_hull(grid, x) != x, axis=1)) >= 4, (
            "premise: points outside")
        f = np.abs(_amounts(x.shape[0], fdtype, seed=6)) + 0.5   # one sign: nothing cancels
        got = _scatter(grid, f, x).astype(np.float64)
    total, carried = float(np.sum(got)), float(np.sum(f.astype(np.float64)))
    m_max = int(np.max(ref.contributions(grid, x)))
    allowed = (grid.corners + 2 * grid.d + 2 + m_max) * E * float(np.sum(np.abs(f)))
    assert abs(total - carried) <= allowed, (
        f"{name}: the grid received {total!r} of {carried!r}: "
        f"{abs(total - carried) / carried:.3e} lost or gained")


@pytest.mark.parametrize("dtypes", DTYPES[:2], ids=_DTYPE_IDS[:2])
@pytest.mark.parametrize("name", sorted(GRIDS))
def test_apply_T_is_apply_of_the_other_mode(name, dtypes):
    """Identity 16: one object, two directions.  ``apply_T`` of the
    consistent mapping is the conservative mapping's ``apply`` bit for bit,
    and the reverse."""
    fdtype, gdtype = dtypes
    grid = GRIDS[name]
    with _x64(_needs_x64(fdtype, gdtype)):
        x = jnp.asarray(_points(grid, gdtype, seed=7))
        u = jnp.asarray(_field(grid, fdtype, seed=7))
        f = jnp.asarray(_amounts(x.shape[0], fdtype, seed=7))
        consistent = _mapping(grid, x.shape[0], "consistent")
        conservative = _mapping(grid, x.shape[0], "conservative")
        pairs = [(consistent.apply_T(f, None, x), conservative.apply(f, None, x)),
                 (conservative.apply_T(u, None, x), consistent.apply(u, None, x))]
        for a, b in pairs:
            a, b = np.asarray(a), np.asarray(b)
            assert a.dtype == b.dtype and a.tobytes() == b.tobytes()


# ---------------------------------------------------------------------------
# 6, 7, 8, 10, 14: symmetries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gdtype", ["float32", "float64"])
@pytest.mark.parametrize("name", ["1d", "2d", "3d"])
def test_gather_commutes_with_a_shift_by_one_cell(name, gdtype):
    """Identity 6: ``G(u; x + spacing[a] e_a) == G(roll(u, -1, a); x)`` for
    points whose stencil is interior before and after the shift."""
    grid = GRIDS[name]
    with _x64(_needs_x64(gdtype)):
        field = _field(grid, gdtype, seed=8)
        for a in range(grid.d):
            if grid.shape[a] < 3:
                continue
            rng = np.random.default_rng(80 + a)
            u = np.stack([rng.uniform(0.1, n - 1.1, size=6) if b != a
                          else rng.uniform(0.1, n - 2.1, size=6)
                          for b, n in enumerate(grid.shape)], axis=1)
            x = grid.points_at(u, np.dtype(gdtype))
            step = np.zeros(grid.d)
            step[a] = grid.spacing[a]
            shifted = np.asarray(x.astype(np.float64) + step, np.dtype(gdtype))
            assert np.all(ref.interior(grid, shifted, margin=0.01)), "premise: still interior"
            rolled = np.asarray(ref.roll(grid, field, -1, a), np.dtype(gdtype))
            got = _gather(grid, field, shifted)
            other = _gather(grid, rolled, x)
            tol = (_gather_tolerance(grid, field, shifted, gdtype, gdtype)
                   + _gather_tolerance(grid, rolled, x, gdtype, gdtype))
            _assert_within(got, other.astype(np.float64), tol, f"{name} shift along axis {a}")


_SCALINGS = {"float32": (-100, -40, -3, 5, 60, 100),
             "float64": (-900, -300, -100, -40, -3, 5, 60, 100, 300, 900)}


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", sorted(GRIDS))
def test_scaling_every_length_by_a_power_of_two_changes_no_bit(name, dtype):
    """Identity 7: origin, spacing and points times ``2**m`` gather and
    scatter the same bits, across the dtype's range."""
    grid = GRIDS[name]
    with _x64(_needs_x64(dtype)):
        x = _points(grid, dtype, seed=9)
        field = _field(grid, dtype, seed=9)
        amounts = _amounts(x.shape[0], dtype, seed=9)
        base = (_gather(grid, field, x), _scatter(grid, amounts, x))
        for m in _SCALINGS[dtype]:
            factor = float(2.0 ** m)
            scaled_x = np.asarray(x * np.dtype(dtype).type(factor))
            assert np.all(np.isfinite(scaled_x)), "premise: the scaled points are finite"
            got = (_gather(grid.scaled(factor), field, scaled_x),
                   _scatter(grid.scaled(factor), amounts, scaled_x))
            for what, a, b in zip(("gather", "scatter"), base, got):
                assert a.tobytes() == b.tobytes(), (
                    f"{name} {dtype}: {what} at lengths times 2**{m} differs by up to "
                    f"{np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))):.3e}")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", sorted(GRIDS))
def test_a_permutation_of_the_points_permutes_gather_and_leaves_scatter(name, dtype):
    """Identity 8.  Gather is a function of each point alone: bit for bit.
    Scatter accumulates in point order, so a cell's contributions are added
    in another order: equal to rounding, ``m_c E sum|contributions|``."""
    grid = GRIDS[name]
    E = _eps(dtype)
    with _x64(_needs_x64(dtype)):
        x = _points(grid, dtype, seed=10)
        perm = np.random.default_rng(10).permutation(x.shape[0])
        field = _field(grid, dtype, seed=10)
        amounts = _amounts(x.shape[0], dtype, seed=10)
        g, g_perm = _gather(grid, field, x), _gather(grid, field, x[perm])
        s, s_perm = _scatter(grid, amounts, x), _scatter(grid, amounts[perm], x[perm])
    assert g[perm].tobytes() == g_perm.tobytes(), f"{name}: gather is not equivariant"
    index, weight = ref.stencil(grid, x)
    mag = np.zeros(grid.size)
    w = np.abs(ref.weights_float64(weight))
    for p in range(index.shape[0]):
        for k in range(grid.corners):
            mag[index[p, k]] += w[p, k] * abs(float(amounts[p]))
    _assert_within(s_perm, s.astype(np.float64), ref.contributions(grid, x) * E * mag,
                   f"{name} scatter under a permutation")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", ["1d", "3d"])
def test_a_batched_call_matches_the_single_calls(name, dtype, record_property):
    """Identity 10: ``vmap`` over the geometry, over the field and over
    both, for gather and scatter, each batch member within ``2 E S1`` of
    its own call.  Whether it is bit for bit is recorded as a property of
    the test, and not asserted: it is not claimed."""
    grid = GRIDS[name]
    E = _eps(dtype)
    with _x64(_needs_x64(dtype)):
        xs = np.stack([_points(grid, dtype, seed=20 + b) for b in range(3)])
        fields = np.stack([_field(grid, dtype, seed=20 + b) for b in range(3)])
        amounts = np.stack([_amounts(xs.shape[1], dtype, seed=20 + b) for b in range(3)])
        G = _mapping(grid, xs.shape[1], "consistent")
        S = _mapping(grid, xs.shape[1], "conservative")
        bitwise = True
        for what, m, values in (("gather", G, fields), ("scatter", S, amounts)):
            batched = {
                "geometry": jax.vmap(lambda g, m=m, v=values[0]: m.apply(jnp.asarray(v), None, g))(
                    jnp.asarray(xs)),
                "field": jax.vmap(lambda v, m=m, g=xs[0]: m.apply(v, None, jnp.asarray(g)))(
                    jnp.asarray(values)),
                "both": jax.vmap(lambda v, g, m=m: m.apply(v, None, g))(
                    jnp.asarray(values), jnp.asarray(xs)),
            }
            for over, out in batched.items():
                out = np.asarray(out)
                for b in range(3):
                    v = values[b] if over != "geometry" else values[0]
                    g = xs[b] if over != "field" else xs[0]
                    single = np.asarray(m.apply(jnp.asarray(v), None, jnp.asarray(g)))
                    if what == "gather":
                        tol = 2 * E * ref.magnitudes(grid, v, g)
                    else:
                        tol = 2 * E * (np.abs(ref.dense_matrix(grid, g)).T
                                       @ np.abs(v.astype(np.float64)))
                    _assert_within(out[b], single.astype(np.float64), tol,
                                   f"{name} {what} batched over the {over}, member {b}")
                    bitwise = bitwise and out[b].tobytes() == single.tobytes()
    record_property("vmap_bit_identical", bitwise)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", ["2d", "3d", "2d-thin"])
def test_the_shaped_layout_is_the_flat_layout_reshaped(name, dtype):
    """Identity 14: ``layout="shaped"`` takes and returns the grid field as
    ``shape`` (or ``shape + (C,)``) and changes no arithmetic."""
    grid = GRIDS[name]
    with _x64(_needs_x64(dtype)):
        x = _points(grid, dtype, seed=11)
        for channels in (0, 2):
            field = _field(grid, dtype, seed=11, channels=channels)
            amounts = _amounts(x.shape[0], dtype, seed=11, channels=channels)
            tail = field.shape[1:]
            flat_g = _gather(grid, field, x)
            shaped_g = _gather(grid, field.reshape(grid.shape + tail), x, layout="shaped")
            assert flat_g.tobytes() == shaped_g.tobytes() and flat_g.shape == shaped_g.shape
            flat_s = _scatter(grid, amounts, x)
            shaped_s = _scatter(grid, amounts, x, layout="shaped")
            assert shaped_s.shape == grid.shape + tail
            assert flat_s.reshape(grid.shape + tail).tobytes() == shaped_s.tobytes()
    m = _mapping(grid, x.shape[0], "consistent", layout="shaped")
    assert tuple(map(tuple, m.field_shapes())) == (grid.shape, (x.shape[0],))
    m = _mapping(grid, x.shape[0], "conservative", layout="shaped")
    assert tuple(map(tuple, m.field_shapes())) == ((x.shape[0],), grid.shape)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_one_dimensional_geometry_may_be_a_flat_array(dtype):
    """For ``d = 1`` the points may be given as ``(n_points,)``: the same
    bits as ``(n_points, 1)``."""
    grid = GRIDS["1d"]
    with _x64(_needs_x64(dtype)):
        x = _points(grid, dtype, seed=12)
        field = _field(grid, dtype, seed=12)
        amounts = _amounts(x.shape[0], dtype, seed=12)
        assert _gather(grid, field, x).tobytes() == _gather(grid, field, x[:, 0]).tobytes()
        assert _scatter(grid, amounts, x).tobytes() == _scatter(grid, amounts, x[:, 0]).tobytes()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", ["1d", "3d"])
def test_the_kernel_under_jit_matches_the_reference(name, dtype):
    """The same identities hold for the compiled kernel (static shapes, no
    Python control flow on values)."""
    grid = GRIDS[name]
    with _x64(_needs_x64(dtype)):
        x = _points(grid, dtype, seed=13)
        field = _field(grid, dtype, seed=13)
        amounts = _amounts(x.shape[0], dtype, seed=13)
        G = _mapping(grid, x.shape[0], "consistent")
        S = _mapping(grid, x.shape[0], "conservative")
        got = np.asarray(jax.jit(lambda f, g: G.apply(f, None, g))(jnp.asarray(field),
                                                                   jnp.asarray(x)))
        spread = np.asarray(jax.jit(lambda f, g: S.apply(f, None, g))(jnp.asarray(amounts),
                                                                      jnp.asarray(x)))
    _assert_within(got, ref.gather(grid, field, x),
                   _gather_tolerance(grid, field, x, dtype, dtype), f"{name} jitted gather")
    _assert_within(spread, ref.scatter(grid, amounts, x),
                   _scatter_tolerance(grid, amounts, x, dtype, dtype), f"{name} jitted scatter")


# ---------------------------------------------------------------------------
# 12, 13: non-finite coordinates and extreme units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_a_diverged_position_reads_nan_and_a_far_one_clamps(dtype):
    """The worked example of the specification: positions ``[0.5, nan, inf,
    -inf, 1e30, -1e30, 5, 2]`` on a six-point grid holding ``10 .. 15``
    gather to ``[10.5, nan, nan, nan, 15, 10, 15, 12]``."""
    with _x64(_needs_x64(dtype)):
        grid = ref.Grid((0.0,), (1.0,), (6,))
        x = np.asarray([0.5, np.nan, np.inf, -np.inf, 1e30, -1e30, 5.0, 2.0],
                       np.dtype(dtype)).reshape(-1, 1)
        got = _gather(grid, np.arange(10, 16, dtype=np.dtype(dtype)), x)
    want = np.asarray([10.5, np.nan, np.nan, np.nan, 15.0, 10.0, 15.0, 12.0])
    assert np.array_equal(np.isnan(got), np.isnan(want)), got
    assert np.array_equal(got[~np.isnan(want)], want[~np.isnan(want)]), got


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", sorted(GRIDS))
def test_a_non_finite_coordinate_poisons_its_own_point_only(name, dtype, bad):
    """Identity 12.  Gather: NaN for that point and no other.  Scatter: NaN
    in exactly the corner cells at index 0, and every other cell what it
    is without that point's contribution."""
    grid = GRIDS[name]
    with _x64(_needs_x64(dtype)):
        x = _points(grid, dtype, seed=14)
        victim = 2
        poisoned = x.copy()
        poisoned[victim, grid.d - 1] = bad
        field = _field(grid, dtype, seed=14)
        got, clean = _gather(grid, field, poisoned), _gather(grid, field, x)
        amounts = _amounts(x.shape[0], dtype, seed=14)
        spread = _scatter(grid, amounts, poisoned)
        silenced_x, silenced = x.copy(), amounts.copy()
        silenced_x[victim] = grid.origin
        silenced[victim] = 0.0
        without = _scatter(grid, silenced, silenced_x)
    others = np.arange(x.shape[0]) != victim
    assert np.isnan(got[victim]) and not np.any(np.isnan(got[others])), got
    assert got[others].tobytes() == clean[others].tobytes()
    corner_cells = np.isnan(ref.scatter(grid, amounts, poisoned))
    assert int(corner_cells.sum()) == int(np.prod([min(n, 2) for n in grid.shape]))
    assert np.array_equal(np.isnan(spread), corner_cells), (
        f"{name}: NaN in cells {np.nonzero(np.isnan(spread))[0].tolist()}, expected "
        f"exactly {np.nonzero(corner_cells)[0].tolist()}")
    assert np.array_equal(spread[~corner_cells], without[~corner_cells])


@pytest.mark.parametrize("name", ["1d", "2d"])
def test_a_spacing_of_2_to_the_minus_125_is_resolved_in_float32(name):
    """Identity 13.  ``spacing = 2**-125``, ``origin = 5 * spacing``, index
    coordinates 0.3 to 7.9: the difference ``x - origin`` is subnormal for
    the first points, and a kernel that formed it unframed reads index
    coordinate 0.0 where the reference reads 0.3."""
    h = float(2.0 ** -125)
    shape = (9,) if name == "1d" else (9, 4)
    grid = ref.Grid((5 * h,) * len(shape), (h,) * len(shape), shape)
    u = np.stack([np.asarray([0.3, 0.45, 1.3, 2.75, 4.5, 6.125, 7.9]),
                  np.asarray([0.3, 2.6, 0.45, 1.5, 2.9, 0.7, 1.1])], axis=1)[:, :grid.d]
    x = grid.points_at(u, np.float32)
    assert np.all(np.abs(x) >= np.finfo(np.float32).tiny), "premise: normal coordinates"
    assert np.any(np.abs(x.astype(np.float64) - 5 * h) < float(np.finfo(np.float32).tiny)), (
        "premise: a subnormal difference")
    field = np.asarray(ref.sample(grid, lambda c: np.sum(c, axis=1) / h) ** 2
                       + 1.0, np.float32)
    got = _gather(grid, field, x)
    _assert_within(got, ref.gather(grid, field, x),
                   _gather_tolerance(grid, field, x, "float32", "float32"),
                   f"{name} gather at spacing 2**-125")
    amounts = _amounts(x.shape[0], "float32", seed=15)
    _assert_within(_scatter(grid, amounts, x), ref.scatter(grid, amounts, x),
                   _scatter_tolerance(grid, amounts, x, "float32", "float32"),
                   f"{name} scatter at spacing 2**-125")


# ---------------------------------------------------------------------------
# 15: the derivative with respect to a coordinate
# ---------------------------------------------------------------------------

#: ``u_i = i**2`` on five points at spacing 0.5: position -> d(gather)/dx.
_KINKS = [(0.0, 2.0), (0.5, 6.0), (1.0, 10.0), (2.0, 14.0), (-0.1, 0.0), (2.1, 0.0),
          (0.25, 2.0)]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_the_derivative_at_a_lattice_point_is_the_interior_one_sided_one(dtype):
    """Identity 15, exactly: the right-hand derivative at a lattice point,
    the left-hand one at the last point of the axis, zero strictly outside
    the hull (the clamp is written so that it does not halve the derivative
    at the bound)."""
    grid = ref.Grid((0.0,), (0.5,), (5,))
    with _x64(_needs_x64(dtype)):
        u = jnp.asarray(np.arange(5.0) ** 2, np.dtype(dtype))
        G = _mapping(grid, 1, "consistent")
        S = _mapping(grid, 1, "conservative")
        for position, want in _KINKS:
            x = jnp.asarray([[position]], np.dtype(dtype))
            dG = jax.grad(lambda g: G.apply(u, None, g)[0])(x)
            assert float(dG[0, 0]) == want, (dtype, position, float(dG[0, 0]), want)
            # Scatter is the transpose: d/dx <u, S(f; x)> = f dG/dx.
            f = jnp.asarray([3.0], np.dtype(dtype))
            dS = jax.grad(lambda g: jnp.vdot(u, S.apply(f, None, g)))(x)
            assert float(dS[0, 0]) == 3.0 * want, (dtype, position, float(dS[0, 0]))


@pytest.mark.parametrize("name", ["1d", "2d", "3d"])
def test_the_derivative_matches_central_differences_away_from_the_kinks(name):
    """Gather and scatter are piecewise linear in each coordinate: ``jax.grad``
    against float64 central differences at points with index fractions in
    0.2 to 0.8, relative ``1e-7``."""
    grid = GRIDS[name]
    with _x64(True):
        rng = np.random.default_rng(31)
        u_idx = np.stack([rng.integers(0, n - 1, size=5) + rng.uniform(0.2, 0.8, size=5)
                          for n in grid.shape], axis=1)
        x = grid.points_at(u_idx)
        field = jnp.asarray(_field(grid, "float64", seed=16))
        amounts = jnp.asarray(_amounts(5, "float64", seed=16))
        probe = jnp.asarray(_field(grid, "float64", seed=17))
        G = _mapping(grid, 5, "consistent")
        S = _mapping(grid, 5, "conservative")
        losses = {"gather": jax.jit(lambda g: jnp.sum(G.apply(field, None, g) ** 2)),
                  "scatter": jax.jit(
                      lambda g: jnp.vdot(probe, S.apply(amounts, None, g) ** 2))}
        for what, loss in losses.items():
            grad = np.asarray(jax.grad(loss)(jnp.asarray(x)))
            fd = np.zeros_like(x)
            for p, a in itertools.product(range(5), range(grid.d)):
                h = 1e-6 * grid.spacing[a]
                up, down = x.copy(), x.copy()
                up[p, a] += h
                down[p, a] -= h
                fd[p, a] = (float(loss(jnp.asarray(up))) - float(loss(jnp.asarray(down)))) / (2 * h)
            scale = float(np.max(np.abs(fd)))
            assert scale > 0, "premise: a derivative to compare"
            assert np.max(np.abs(grad - fd)) <= 1e-7 * scale, (
                name, what, float(np.max(np.abs(grad - fd)) / scale))


# ---------------------------------------------------------------------------
# The reference against the kernel's identities, in exact arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_the_reference_s_weights_sum_to_exactly_one(name):
    grid = GRIDS[name]
    index, weight = ref.stencil(grid, _points(grid, "float32"))
    assert index.shape == weight.shape == (index.shape[0], grid.corners)
    assert index.min() >= 0 and index.max() < grid.size
    for p in range(index.shape[0]):
        assert sum(weight[p], Fraction(0)) == 1
        assert all(0 <= w <= 1 for w in weight[p])


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_the_reference_reproduces_a_multilinear_field_inside_the_hull(name):
    """With every lattice value an integer and the stencil exact, the
    gathered value of ``prod_a (1 + i_a)`` is the same product at the
    point's exact index coordinates."""
    grid = GRIDS[name]
    x = _points(grid, "float64")
    x = x[ref.interior(grid, x)]
    field = np.asarray([np.prod([1 + i for i in idx])
                        for idx in itertools.product(*[range(n) for n in grid.shape])],
                       np.float64)
    got = ref.gather(grid, field, x)
    for p, u in enumerate(ref.index_coordinates(grid, x)):
        want = Fraction(1)
        for a in range(grid.d):
            want *= 1 + (u[a] if grid.shape[a] > 1 else 0)   # one point: its value
        assert got[p] == float(want)


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_the_reference_s_scatter_is_the_transpose_of_its_gather(name):
    """``gather == H @ u`` and ``scatter == H.T @ f`` for the reference's own
    dense matrix, the total is preserved, and a clamped point is its
    projection onto the hull."""
    grid = GRIDS[name]
    x = _points(grid, "float64", seed=1)
    u, f = _field(grid, "float64", seed=1), _amounts(x.shape[0], "float64", seed=1)
    H = ref.dense_matrix(grid, x)
    assert np.allclose(ref.gather(grid, u, x), H @ u, rtol=0, atol=64 * _eps("float64"))
    assert np.allclose(ref.scatter(grid, f, x), H.T @ f, rtol=0, atol=64 * _eps("float64"))
    assert abs(float(np.sum(ref.scatter(grid, f, x))) - float(np.sum(f))) <= (
        64 * _eps("float64") * float(np.sum(np.abs(f))))
    # (The projection is formed in float64, so the hull's far face is a
    # rounding away from the exact one the stencil clamps to.)
    assert np.allclose(H, ref.dense_matrix(grid, ref.project_onto_hull(grid, x)),
                       rtol=0, atol=64 * _eps("float64"))
    assert np.array_equal((H != 0) & ~ref.support(grid, x), np.zeros_like(H, bool))


def test_the_reference_orders_the_corners_with_axis_0_most_significant():
    """A point in the cell at index ``(1, 0, 0)`` of a ``4 x 3 x 2`` lattice:
    its eight corners, in order, are the flat indices of ``(1 or 2, 0 or 1,
    0 or 1)`` with the last axis fastest, and the weights are the products
    of the per-axis fractions."""
    grid = ref.Grid((0.0, 1.5, 0.125), (0.25, 0.5, 2.0), (4, 3, 2))   # exact in binary
    x = grid.points_at([[1.25, 0.5, 0.75]])
    index, weight = ref.stencil(grid, x)
    assert index[0].tolist() == [6, 7, 8, 9, 12, 13, 14, 15]
    t = (Fraction(1, 4), Fraction(1, 2), Fraction(3, 4))
    want = [(t[0] if c[0] else 1 - t[0]) * (t[1] if c[1] else 1 - t[1])
            * (t[2] if c[2] else 1 - t[2]) for c in itertools.product((0, 1), repeat=3)]
    assert list(weight[0]) == want
