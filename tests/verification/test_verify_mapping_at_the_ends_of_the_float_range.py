"""`verify_mapping` at the ends of the float range.

An honest edge passes every check it claims whatever the magnitude of the
field, the factor of its transform and the size of its weights; and the
allowance that makes it so still fails a seeded fault at small magnitudes.

The delivery flushes: a result below the smallest normal number of its
dtype (``tiny``) is zero, and an operand below it is read as zero.  One
flush costs up to ``tiny`` in the units of the value flushed, whatever the
operator's gain, so a tolerance whose floor is the gain times a few
``tiny`` fails an honest edge that scales down (the draw pinned in
:func:`test_the_draw_that_failed_an_honest_edge_passes`), one whose
weights are small, and any edge on a scalar below ``tiny``.  Three halves:

* honest edges over a table of transform factors, weight sums and field
  magnitudes, each check fed pinned fields (:class:`_Pinned`), and the
  battery as it is called, on drawn fields;
* seeded faults at small magnitudes, which must still fail;
* every tolerance at ordinary magnitudes, which must be the rounding
  tolerance it was, to the last bit.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import contextlib  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from maddening.core.coupling.grid_mapping import multilinear_grid_mapping  # noqa: E402
from maddening.core.coupling.mapping import (  # noqa: E402
    matrix_mapping,
    nearest_neighbor_mapping,
)
from maddening.core.edge import EdgeSpec, _delivered  # noqa: E402
from maddening.testing import mapping as battery  # noqa: E402
from maddening.testing.mapping import DEFAULT_ROUNDING_UNITS, verify_mapping  # noqa: E402
from maddening.testing.verification import VerificationResult  # noqa: E402

N_SOURCE, N_TARGET = 9, 7
_RNG = np.random.default_rng(5)
_H = np.abs(_RNG.normal(size=(N_TARGET, N_SOURCE)))
#: Rows that sum to one (a consistent transfer) and columns that do (a
#: conservative one); times a weight sum, neither.
ROWS = _H / _H.sum(axis=1, keepdims=True)
_C = np.abs(_RNG.normal(size=(N_TARGET, N_SOURCE)))
COLS = _C / _C.sum(axis=0, keepdims=True)
FLUID, SOLID = np.linspace(0.0, 1.0, N_SOURCE), np.linspace(0.05, 0.95, N_TARGET)

#: What the edge's transform multiplies by, and what the weights sum to.
SCALES = (1e-9, 1e-3, -1e-3, 1.0, 1e3, 1e9)
WEIGHT_SUMS = (1e-6, 1.0, 1e6)

#: Field magnitudes, from below the smallest subnormal's neighbourhood to
#: near the top of the range.  The float64 list holds float64's own
#: underflow end, the magnitudes at which the product of two fields
#: leaves float64 (1e-162 and 1e160), and float32's (1e-44).
MAGNITUDES = {
    np.dtype(np.float32): (1e-44, 1e-40, 1.2e-38, 1e-37, 1e-36, 1e-35, 1e-33, 1e-31, 1e-28,
                           1e-24, 1e-20, 1e-10, 1.0, 1e10, 1e20, 1e30),
    np.dtype(np.float64): (1e-320, 1e-310, 2.3e-308, 1e-307, 1e-305, 1e-300, 1e-296, 1e-292,
                           1e-280, 1e-200, 1e-162, 1e-44, 1.0, 1e100, 1e150, 1e160, 1e200,
                           1e300),
}
#: Magnitudes at which nothing is near either end: every tolerance there
#: is the rounding tolerance.
ORDINARY = (1e-20, 1e-10, 1.0, 1e10, 1e20)


def _times(scale):
    def scaled(value):
        return scale * value
    return scaled


def _edge(kind, scale, weight_sum=1.0, dtype=np.float32):
    """An honest edge and the claims it makes: ``(edge, verify_mapping keywords)``."""
    if kind == "nearest":
        mapping = nearest_neighbor_mapping(FLUID, SOLID, mode="conservative")
        claims = dict(conservative=True, consistent=False)
    elif kind == "rows":
        mapping = matrix_mapping((weight_sum * ROWS).astype(dtype))
        claims = dict(consistent=weight_sum == 1.0, conservative=False)
    else:
        mapping = matrix_mapping((weight_sum * COLS).astype(dtype), mode="conservative")
        claims = dict(conservative=weight_sum == 1.0, consistent=False)
    edge = EdgeSpec("fluid", "solid", "traction", "force", mapping=mapping,
                    transform=None if scale == 1.0 else _times(scale))
    return edge, dict(scale=None if scale == 1.0 else scale, **claims)


def _in_range(scale, weight_sum, magnitude, dtype):
    """Whether a field of *magnitude* is delivered inside *dtype*'s range,
    with room for the sums the checks form."""
    top = float(np.finfo(dtype).max) / 64.0
    return weight_sum * magnitude <= top and abs(scale) * weight_sum * magnitude <= top


def _fields(n, magnitude, dtype, zero=True):
    """Fields of one magnitude: uniform, alternating, a ramp, a spike, a
    decay over nine binades, a sparse one, an uneven alternation, and
    (with ``zero``) the zero field."""
    j = np.arange(n)
    shapes = [np.ones(n), np.where(j % 2, -1.0, 1.0), (j + 1) / n, 1.0 * (j == 0), 0.5 ** j,
              np.where(j % 3, 0.0, 1.0), np.where(j % 2, -1.0, 1.0) * (1 + 0.37 * j / n)]
    shapes += [np.zeros(n)] if zero else []
    with np.errstate(all="ignore"):
        return [(magnitude * shape).astype(dtype) for shape in shapes]


def _scalars(dtype, ordinary=False):
    """Pairs ``(a, b)`` for the linearity check: ordinary ones, the smallest
    normal number, numbers below it, and the pair of the draw that failed."""
    usual = [(1.0, 1.0), (4.0, -4.0), (-3.5, 0.25), (4.0, 3.9999)]
    if ordinary:
        return usual
    tiny = float(np.finfo(dtype).tiny)
    least = float(np.nextafter(dtype.type(0), dtype.type(1)))
    failed = 4.852989173076193e-38 if dtype == np.float32 else 4.0 * tiny
    return usual + [(0.0, failed), (tiny, tiny), (least, -4.0), (0.6 * tiny, 1.0),
                    (tiny, -1.0), (1.0, 0.99 * tiny)]


def _examples(name, magnitudes, dtype, *, n_source=N_SOURCE, n_target=N_TARGET,
              ordinary=False):
    """Pinned examples for the check *name* of a static mapping.  With
    ``ordinary``, nothing in them is zero or below ``tiny``: neither a
    field nor a scalar."""
    out = []
    for magnitude in magnitudes:
        xs = _fields(n_source, magnitude, dtype, zero=not ordinary)
        ys = _fields(n_target, magnitude, dtype, zero=not ordinary)
        if name == "linearity":
            out += [dict(x=x, y=y, a=a, b=b, geom=None)
                    for x, y in zip(xs, xs[1:] + xs[:1]) for a, b in _scalars(dtype, ordinary)]
        elif name in ("structure", "adjoint"):
            out += [dict(x=x, y=y, geom=None) for x in xs for y in ys[::2]]
        elif name == "consistent":
            with np.errstate(all="ignore"):
                out += [dict(c=float(dtype.type(c * magnitude)), geom=None)
                        for c in (1.0, -1.0, 0.5, 0.3) + (() if ordinary else (0.0,))]
        else:
            out += [dict(x=x, geom=None) for x in xs]
    return out


class _Pinned:
    """Stands in for the battery's ``_drive``: the body of every check runs
    on the examples given for it, in order, and on no others.  What ran is
    kept, so a test can show the stand-in was the one called."""

    def __init__(self, examples):
        self.examples, self.ran = examples, {}

    def __call__(self, name, strategy, body, *, max_examples, derandomize, notes=None):
        pinned = self.examples(name)
        self.ran[name] = len(pinned)
        for example in pinned:
            try:
                body(**example)
            except AssertionError as failure:
                return VerificationResult(name, "FAIL", detail=str(failure),
                                          counterexample=dict(example), n_examples=len(pinned))
        return VerificationResult(name, "PASS", n_examples=len(pinned),
                                  detail=notes() if notes is not None else "")


@pytest.fixture
def pin(monkeypatch):
    """``pin(examples)`` replaces the battery's driver, in the module that
    reads it, by a :class:`_Pinned`."""
    def install(examples):
        driver = _Pinned(examples)
        monkeypatch.setattr(battery, "_drive", driver)
        return driver
    return install


@contextlib.contextmanager
def _float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _report(results):
    return "\n".join(f"{r}\n  counterexample: {r.counterexample}" for r in results.values()
                     if not r.passed)


def _failed(results):
    return sorted(name for name, r in results.items() if not r.passed)


# ---------------------------------------------------------------------------
# The draw that failed
# ---------------------------------------------------------------------------

#: What the linearity check drew on the edge of the developer guide (a
#: conservative nearest-neighbour transfer, N to kN and a sign) when it
#: failed that edge in CI: 66.1 rounding units where 64 were allowed.
FAILED_DRAW = dict(x=np.zeros(N_SOURCE, np.float32), y=np.full(N_SOURCE, 16.0, np.float32),
                   a=0.0, b=4.852989173076193e-38, geom=None)


def test_the_draw_that_failed_an_honest_edge_passes(pin):
    edge, claims = _edge("nearest", -1e-3)
    driver = pin(lambda name: [FAILED_DRAW])
    results = verify_mapping(edge, checks=["linearity"], **claims)
    assert driver.ran == {"linearity": 1} and results["linearity"].n_examples == 1
    assert results["linearity"].status == "PASS", _report(results)


def test_the_draw_that_failed_is_a_flush_of_the_delivered_value_the_old_floor_did_not_cover():
    """What the draw is, stated without relying on whether this build
    flushes: ``b * y`` is a normal number, what the edge delivers of it is
    below ``tiny``, the delivery returns it to within one flush, and the
    floor the check had (the gain times 64 ``tiny``) is below it."""
    edge, _ = _edge("nearest", -1e-3)
    tiny = float(np.finfo(np.float32).tiny)
    b, y = np.float32(FAILED_DRAW["b"]), FAILED_DRAW["y"]
    assert float(b) * 16.0 > tiny
    exact = -1e-3 * float(b) * np.asarray(edge.mapping.apply(jnp.asarray(y)), np.float64)
    delivered = np.asarray(_delivered(edge, jnp.asarray(b * y)), np.float64)
    assert 0.0 < float(np.max(np.abs(exact))) < tiny
    assert float(np.max(np.abs(delivered - exact))) < tiny
    gain = battery._subject(edge, None, -1e-3, None).gain(np.float32)(None)
    assert gain == pytest.approx(2e-3, rel=1e-6)
    assert DEFAULT_ROUNDING_UNITS * gain * tiny < float(np.max(np.abs(exact)))


# ---------------------------------------------------------------------------
# Honest edges over the table: pinned fields
# ---------------------------------------------------------------------------

#: ``id -> (kind, scale, weight sum)``: every factor against every weight
#: sum for a matrix with unit row sums and one with unit column sums, and
#: every factor for the nearest-neighbour transfer.
EDGES = {
    **{f"nearest-s{scale:g}": ("nearest", scale, 1.0) for scale in SCALES},
    **{f"{kind}-w{weight_sum:g}-s{scale:g}": (kind, scale, weight_sum)
       for kind in ("rows", "cols") for weight_sum in WEIGHT_SUMS for scale in SCALES},
}

#: The edges that run on every push: the one of the developer guide, the
#: two extremes of the factor on unit sums, and small and large weights
#: against the opposite factor (the row's own products flush; the gain is
#: far above one).
EDGES_PER_PUSH = ("nearest-s-0.001", "rows-w1-s1e-09", "cols-w1-s1e+09", "rows-w1e-06-s1e+09",
                  "cols-w1e+06-s1e-09", "rows-w1e-06-s1")


def _run_table_cell(case, dtype, pin):
    kind, scale, weight_sum = EDGES[case]
    dtype = np.dtype(dtype)
    edge, claims = _edge(kind, scale, weight_sum, dtype)

    def examples(name):
        # A dtype check draws its own dtype: a float32 field under x64.
        drawn = np.dtype(name[len("dtype_"):]) if name.startswith("dtype_") else dtype
        magnitudes = [m for m in MAGNITUDES[drawn] if _in_range(scale, weight_sum, m, drawn)]
        return _examples(name, magnitudes, drawn)

    driver = pin(examples)
    dtypes = (np.float32, np.float64) if dtype == np.float64 else (np.float32,)
    results = verify_mapping(edge, dtype=dtype, dtypes=dtypes, **claims)
    assert not _failed(results), _report(results)
    # Nothing passed by not being looked at.
    ran = ["structure", "linearity", "adjoint", "jit_consistent"]
    ran += [name for name in ("consistent", "conservative") if claims[name]]
    ran += ["dtype_float64"] if dtype == np.float64 else []
    for name in ran:
        assert results[name].status == "PASS" and driver.ran[name] >= len(MAGNITUDES[dtype]) - 1


def test_the_per_push_edges_are_edges_of_the_table():
    assert set(EDGES_PER_PUSH) <= set(EDGES) and len(EDGES) == 42


# Per push: tests/verification/test_verify_mapping_at_the_ends_of_the_float_range.py::test_an_honest_edge_passes_at_every_magnitude_of_a_float32_field[nearest-s-0.001]
# (EDGES_PER_PUSH: the guide's edge, both extremes of the factor, small and
# large weights; the slow cases are the other cells of the same table)
@pytest.mark.parametrize("case", [
    pytest.param(case, marks=() if case in EDGES_PER_PUSH else pytest.mark.slow)
    for case in sorted(EDGES)])
def test_an_honest_edge_passes_at_every_magnitude_of_a_float32_field(case, pin):
    _run_table_cell(case, np.float32, pin)


#: Float64 fields on every push: a factor below one, and small weights.
EDGES_FLOAT64_PER_PUSH = ("nearest-s-0.001", "rows-w1e-06-s1e+09")


# Per push: tests/verification/test_verify_mapping_at_the_ends_of_the_float_range.py::test_an_honest_edge_passes_at_every_magnitude_of_a_float64_field[nearest-s-0.001]
# (EDGES_FLOAT64_PER_PUSH; the slow cases are the other cells of the same table)
@pytest.mark.parametrize("case", [
    pytest.param(case, marks=() if case in EDGES_FLOAT64_PER_PUSH else pytest.mark.slow)
    for case in sorted(EDGES)])
def test_an_honest_edge_passes_at_every_magnitude_of_a_float64_field(case, pin):
    with _float64():
        _run_table_cell(case, np.float64, pin)


def test_float32_weights_under_a_float64_field_pass_at_both_ends(pin):
    """Mixed precision: the field is read in its own dtype and the row's
    products are float64."""
    with _float64():
        edge, claims = _edge("rows", 1e-3, 1.0, np.float32)
        magnitudes = [m for m in MAGNITUDES[np.dtype(np.float64)] if 1e-300 <= m <= 1e30]
        driver = pin(lambda name: _examples(
            name, MAGNITUDES[np.dtype(np.float32)] if name == "dtype_float32" else magnitudes,
            np.dtype(np.float32 if name == "dtype_float32" else np.float64)))
        results = verify_mapping(edge, dtype=np.float64, **claims)
        assert not _failed(results), _report(results)
        assert results["dtype_float32"].status == "PASS" and driver.ran["dtype_float32"] > 0


# ---------------------------------------------------------------------------
# A geometry-dependent kind on a scaled edge: the hull and the derivative
# ---------------------------------------------------------------------------

ORIGIN, SPACING, SHAPE, N_POINTS = (0.0, -1.0), (0.25, 0.5), (5, 4), 6
LOWER = np.asarray(ORIGIN)
UPPER = LOWER + np.asarray(SPACING) * (np.asarray(SHAPE) - 1)
LATTICE = np.stack(np.meshgrid(*[ORIGIN[a] + SPACING[a] * np.arange(SHAPE[a])
                                 for a in range(2)], indexing="ij"), axis=-1).reshape(-1, 2)
POSITIONS = (LOWER + np.random.default_rng(11).uniform(0.02, 0.98, (N_POINTS, 2))
             * (UPPER - LOWER)).astype(np.float32)

#: ``id -> (mode, scale)`` for the grid kind.  The derivative check weighs
#: the delivered values with a drawn field of the same magnitude, so its
#: functional is a product of two fields: the magnitudes stop where that
#: product leaves float32 (see ``_grid_magnitudes``).
GRID_EDGES = {f"{mode}-s{scale:g}": (mode, scale)
              for mode in ("consistent", "conservative") for scale in SCALES}
GRID_EDGES_PER_PUSH = ("consistent-s-0.001",)


def _grid_magnitudes(scale):
    top = float(np.finfo(np.float32).max) / 64.0
    return [m for m in MAGNITUDES[np.dtype(np.float32)]
            if abs(scale) * m <= top and max(abs(scale), 1.0) * m * m * N_POINTS * 64.0 <= top]


def _grid_examples(name, magnitudes, n_source, n_target):
    geom = jnp.asarray(POSITIONS)
    out = []
    for magnitude in magnitudes:
        xs = _fields(n_source, magnitude, np.float32)[:3]
        rs = _fields(n_target, magnitude, np.float32)
        if name == "linearity":
            out += [dict(x=x, y=y, a=a, b=b, geom=geom) for x, y in zip(xs, xs[1:] + xs[:1])
                    for a, b in _scalars(np.dtype(np.float32))[::2]]
        elif name in ("structure", "adjoint"):
            out += [dict(x=x, y=y, geom=geom) for x in xs for y in rs[:2]]
        elif name == "consistent":
            with np.errstate(all="ignore"):
                out += [dict(c=float(np.float32(c * magnitude)), geom=geom)
                        for c in (1.0, -0.5, 0.0)]
        elif name == "geometry_derivative":
            out += [dict(x=x, r=r, geom=geom, where=where) for x in xs[:2]
                    for r in (rs[1], np.ones(n_target, np.float32))
                    for where in (0.0, 0.31, 0.77)]
        elif name == "outside_hull":
            out += [dict(x=x, geom=geom, push=push, far=3.0) for x in xs
                    for push in ([1], [4, 1, 0, 0, 2])]
        else:
            out += [dict(x=x, geom=geom) for x in xs]
    return out


# Per push: tests/verification/test_verify_mapping_at_the_ends_of_the_float_range.py::test_a_geometry_dependent_kind_on_a_scaled_edge_passes_at_every_magnitude[consistent-s-0.001]
# (GRID_EDGES_PER_PUSH; the slow cases are the other mode and the other factors)
@pytest.mark.parametrize("case", [
    pytest.param(case, marks=() if case in GRID_EDGES_PER_PUSH else pytest.mark.slow)
    for case in sorted(GRID_EDGES)])
def test_a_geometry_dependent_kind_on_a_scaled_edge_passes_at_every_magnitude(case, pin):
    mode, scale = GRID_EDGES[case]
    mapping = multilinear_grid_mapping(ORIGIN, SPACING, SHAPE, n_points=N_POINTS, mode=mode)
    n_source, n_target = (len(LATTICE), N_POINTS) if mode == "consistent" else (
        N_POINTS, len(LATTICE))
    points = dict(source_coordinates=LATTICE, target_coordinates=lambda g: g)
    if mode == "conservative":
        points = dict(source_coordinates=lambda g: g, target_coordinates=LATTICE)
    magnitudes = _grid_magnitudes(scale)
    assert magnitudes[0] == 1e-44 and magnitudes[-1] >= 1.0
    driver = pin(lambda name: _grid_examples(name, magnitudes, n_source, n_target))
    transform = {} if scale == 1.0 else dict(transform=_times(scale), scale=scale)
    results = verify_mapping(mapping, geometry=POSITIONS, polynomial_order=1,
                             hull=(LOWER, UPPER), outside="clamp", dtypes=(np.float32,),
                             **points, **transform)
    assert not _failed(results), _report(results)
    for name in ("linearity", mode, "adjoint", "outside_hull", "jit_consistent"):
        assert results[name].status == "PASS" and driver.ran[name] > 0
    # The derivative was compared with a difference, not only counted as kinks.
    assert results["geometry_derivative"].status == "PASS"
    assert "draws compared" in results["geometry_derivative"].detail
    assert not results["geometry_derivative"].detail.startswith("0 draws compared")


# ---------------------------------------------------------------------------
# Honest edges: the battery as it is called, on drawn fields
# ---------------------------------------------------------------------------

#: ``id -> (kind, scale, weight sum, bound of the drawn fields)``.
DRAWN = {
    "the-guides-edge": ("nearest", -1e-3, 1.0, 100.0),
    "the-guides-edge-near-tiny": ("nearest", -1e-3, 1.0, 1e-34),
    "small-weights-near-tiny": ("rows", 1e9, 1e-6, 1e-30),
    "a-factor-of-1e-9": ("rows", 1e-9, 1.0, 1e-28),
    "large-fields": ("cols", 1e3, 1.0, 1e30),
    "subnormal-bounds": ("cols", 1.0, 1.0, 1e-40),
}
DRAWN_PER_PUSH = ("the-guides-edge-near-tiny", "small-weights-near-tiny")


# Per push: tests/verification/test_verify_mapping_at_the_ends_of_the_float_range.py::test_the_battery_as_it_is_called_passes_an_honest_edge_on_drawn_fields[the-guides-edge-near-tiny]
# (DRAWN_PER_PUSH; the slow cases are other factors and bounds, at the same depth)
@pytest.mark.parametrize("case", [
    pytest.param(case, marks=() if case in DRAWN_PER_PUSH else pytest.mark.slow)
    for case in sorted(DRAWN)])
def test_the_battery_as_it_is_called_passes_an_honest_edge_on_drawn_fields(case):
    kind, scale, weight_sum, bound = DRAWN[case]
    edge, claims = _edge(kind, scale, weight_sum)
    results = verify_mapping(edge, bounds=(-bound, bound), derandomize=True, **claims)
    assert not _failed(results), _report(results)
    assert results["linearity"].status == "PASS" and results["linearity"].n_examples >= 50


# ---------------------------------------------------------------------------
# Seeded faults at small magnitudes still fail
# ---------------------------------------------------------------------------

TINY = float(np.finfo(np.float32).tiny)
KW = dict(max_examples=30, derandomize=True)


class _Dense:
    """A dense kind written by hand."""

    kind = "dense_by_hand"

    def __init__(self, matrix, mode="consistent"):
        self.matrix = jnp.asarray(matrix, jnp.float32)
        self.mode = mode
        self.n_target, self.n_source = (int(n) for n in self.matrix.shape)

    def params_pytree(self):
        return {}

    def apply(self, field, weights=None, geom=None):
        return self.matrix @ field

    def apply_T(self, field, weights=None, geom=None):
        return self.matrix.T @ field


class _DropsItsLastRow(_Dense):
    """The last target entry is never written."""

    def apply(self, field, weights=None, geom=None):
        return (self.matrix @ field).at[-1].set(0.0)


def _offset(scale, units):
    """The declared scaling, and ``units * tiny`` added to every value."""
    def shifted(value):
        return scale * value + np.float32(units * TINY)
    return shifted


def _clamped(scale, bound):
    def clamp(value):
        return jnp.clip(scale * value, -bound, bound)
    return clamp


#: ``(factor, bound of the drawn fields)``: what is delivered is some
#: three decades above ``tiny`` in both.
DROPPED_ROW = ((1.0, 1e-33), (1e-3, 1e-30))


def _guides_edge(transform):
    return EdgeSpec("fluid", "solid", "traction", "force", transform=transform,
                    mapping=nearest_neighbor_mapping(FLUID, SOLID, mode="conservative"))


def test_the_guides_edge_passes_with_these_keywords_when_nothing_is_seeded():
    """The fault tests below fail for the fault, not for the bounds."""
    for bound in (100.0, 1e-25):
        results = verify_mapping(_guides_edge(_times(-1e-3)), scale=-1e-3,
                                 bounds=(-bound, bound), **KW)
        assert not _failed(results), _report(results)
    for scale, bound in DROPPED_ROW:
        transform = {} if scale == 1.0 else dict(transform=_times(scale), scale=scale)
        results = verify_mapping(_Dense(ROWS), bounds=(-bound, bound), **transform, **KW)
        assert not _failed(results), _report(results)


@pytest.mark.parametrize("scale, units", [(-1e-3, 4.0), (1.0, 1024.0)])
def test_a_transform_with_an_offset_of_a_few_tiny_fails_conservation(scale, units):
    """``units`` is above what the flushes of one delivered value can cost
    on that edge: 2.04 ``tiny`` at a factor of 1e-3, 128 at a factor of
    one (where the rounding floor, 64 times the gain of two, is the
    larger and always was)."""
    results = verify_mapping(_guides_edge(_offset(scale, units)), scale=scale, **KW)
    assert results["conservative"].failed, _report(results)
    assert "not preserved" in results["conservative"].detail


def test_an_offset_of_one_tiny_is_one_flush_and_is_not_told_from_an_honest_edge():
    """The resolution of the battery at the underflow end, stated: an
    honest delivery may itself return zero for a value of all but one ulp
    of ``tiny``, so an offset of one ``tiny`` is within what a flush
    costs.  (The floor this replaces refused it on an edge with a gain
    below 1/64 -- and refused the honest edge with it.)"""
    results = verify_mapping(_guides_edge(_offset(-1e-3, 1.0)), scale=-1e-3, **KW)
    assert not _failed(results), _report(results)


@pytest.mark.parametrize("scale, bound", DROPPED_ROW)
def test_a_mapping_that_drops_its_last_row_fails_a_few_decades_above_tiny(scale, bound):
    transform = {} if scale == 1.0 else dict(transform=_times(scale), scale=scale)
    results = verify_mapping(_DropsItsLastRow(ROWS), bounds=(-bound, bound), **transform, **KW)
    assert results["consistent"].failed and results["adjoint"].failed, _report(results)
    assert results["linearity"].passed and results["jit_consistent"].passed


@pytest.mark.parametrize("scale, bound", [(1.0, 1e-27), (-1e-3, 1e-25)])
def test_a_clamp_at_1e_30_fails_on_fields_a_few_decades_above_it(scale, bound):
    results = verify_mapping(_guides_edge(_clamped(scale, 1e-30)), scale=scale,
                             bounds=(-bound, bound), **KW)
    assert results["linearity"].failed and results["conservative"].failed, _report(results)


def test_a_transpose_that_is_not_the_adjoint_fails_on_float64_fields_of_1e200(pin):
    """The adjoint identity's inner products are framed by powers of two;
    the frame hides nothing: a transpose one part in a thousand off fails
    where the product of the two fields is not a float64."""
    class _TransposeOff(_Dense):
        def apply_T(self, field, weights=None, geom=None):
            return 1.001 * (self.matrix.T @ field)

    def examples(name):
        return _examples(name, (1e-200, 1e200), np.dtype(np.float64))

    with _float64():
        pin(examples)
        honest = verify_mapping(_Dense(ROWS), checks=["adjoint"], dtype=np.float64)
        assert honest["adjoint"].status == "PASS", _report(honest)
        pin(examples)
        results = verify_mapping(_TransposeOff(ROWS), checks=["adjoint"], dtype=np.float64)
        assert results["adjoint"].failed
        assert "rounding units" in results["adjoint"].detail


# ---------------------------------------------------------------------------
# The allowance: unchanged at ordinary magnitudes, and never below a flush
# ---------------------------------------------------------------------------


@pytest.fixture
def allowed(monkeypatch):
    """Every tolerance the battery takes: ``(rounding, flushes, tolerance)``."""
    taken = []
    real = battery._allowed

    def recording(rounding, flushes):
        tolerance = real(rounding, flushes)
        taken.append((rounding, flushes, tolerance))
        return tolerance

    monkeypatch.setattr(battery, "_allowed", recording)
    return taken


#: Gains (the factor times the weight sum) from 1e-6 to 1e6, and the
#: guide's edge.
ORDINARY_EDGES = ("nearest-s-0.001", "rows-w1e-06-s1", "rows-w1e-06-s1e+09", "rows-w1-s0.001",
                  "cols-w1-s1000", "cols-w1e+06-s1e-09", "cols-w1e+06-s1")


@pytest.mark.parametrize("case", ORDINARY_EDGES)
def test_at_ordinary_magnitudes_every_tolerance_is_the_rounding_tolerance(case, pin, allowed):
    """Fields from 1e-20 to 1e20 and scalars of ordinary size: the
    allowance for flushes is below the rounding tolerance in every
    comparison of every check, so the tolerance is the value it was."""
    kind, scale, weight_sum = EDGES[case]
    assert 1e-6 <= abs(scale) * weight_sum <= 1e6
    edge, claims = _edge(kind, scale, weight_sum)
    driver = pin(lambda name: _examples(name, ORDINARY, np.dtype(np.float32), ordinary=True))
    results = verify_mapping(edge, **claims)
    assert not _failed(results), _report(results)
    # Every comparison of the battery took its tolerance there: one per
    # example (the round trip compares bit for bit, and the float32 dtype
    # check has no wider evaluation to compare with while x64 is off).
    assert len(allowed) == sum(n for name, n in driver.ran.items()
                               if name not in ("round_trip", "dtype_float32"))
    for rounding, flushes, tolerance in allowed:
        assert 0.0 < flushes < rounding and tolerance == rounding
        assert tolerance.hex() == rounding.hex()


def test_the_allowance_only_grows_and_grows_on_the_draw_that_failed(pin, allowed):
    edge, claims = _edge("nearest", -1e-3)
    pin(lambda name: [FAILED_DRAW] if name == "linearity" else _examples(
        name, MAGNITUDES[np.dtype(np.float32)][:12], np.dtype(np.float32)))
    # The checks whose tolerance is in delivered units (the adjoint's is
    # in framed units of a product of two fields).
    verify_mapping(edge, checks=["structure", "linearity", "conservative", "jit_consistent"],
                   **claims)
    assert all(tolerance >= rounding for rounding, _, tolerance in allowed)
    tiny = float(np.finfo(np.float32).tiny)
    grown = [(rounding, tolerance) for rounding, _, tolerance in allowed if tolerance > rounding]
    # Where it grew it is a few tiny: the cost of flushes, nothing larger.
    assert grown and all(tolerance < 1e4 * tiny for _, tolerance in grown)


def test_a_flush_costs_what_it_costs_whatever_the_gain():
    """The three terms of ``_flush_cost``, each in the units it is flushed
    in: the delivered value does not shrink with the gain, the row's
    products are carried by the factor alone, the field as read by the
    gain."""
    tiny32, tiny64 = float(np.finfo(np.float32).tiny), float(np.finfo(np.float64).tiny)
    edge, _ = _edge("nearest", -1e-3)
    subject = battery._subject(edge, None, -1e-3, None)
    cost = battery._flush_cost
    rows = 2 * N_SOURCE - 1
    assert cost(subject, 0.0, np.float32, np.float32) == pytest.approx(
        2.0 * tiny32 * (1.0 + 1e-3 * rows))
    assert cost(subject, 1e-9, np.float32, np.float32) > 2.0 * tiny32
    assert cost(subject, 5.0, np.float32, np.float32) - cost(
        subject, 0.0, np.float32, np.float32) == pytest.approx(2.0 * 5.0 * tiny32)
    # The field is read in its own dtype, the products are the delivery's.
    assert cost(subject, 5.0, np.float64, np.float32) == pytest.approx(
        2.0 * (tiny64 * (1.0 + 1e-3 * rows) + 5.0 * tiny32))
    columns = 2 * N_TARGET - 1
    assert cost(subject, 5.0, np.float32, np.float32, transposed=True) == pytest.approx(
        2.0 * tiny32 * (1.0 + 1e-3 * columns + N_TARGET * 5.0))


def test_a_field_below_tiny_is_read_as_zero_and_allowed_for_at_a_stated_narrower_width(pin):
    """The allowance for flushes is not in rounding units: at a quarter of
    a unit, a field below ``tiny`` (read as zero by the delivery, kept by
    the float64 reference) still passes through weights a million strong,
    where what was read as zero is carried by the gain."""
    edge, claims = _edge("rows", 1.0, 1e6)
    tiny = float(np.finfo(np.float32).tiny)

    def examples(name):
        xs = _fields(N_SOURCE, 0.9 * tiny, np.float32)
        ys = _fields(N_TARGET, 1.0, np.float32)
        return [dict(x=x, y=y, geom=None) for x in xs for y in ys]

    driver = pin(examples)
    results = verify_mapping(edge, checks=["adjoint"], rounding_units=0.25, **claims)
    assert driver.ran == {"adjoint": 64}
    assert results["adjoint"].status == "PASS", _report(results)
