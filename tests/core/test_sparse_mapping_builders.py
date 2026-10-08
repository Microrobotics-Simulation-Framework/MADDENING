"""The three sparse builders: nearest neighbour, 1-D projection, your own rows.

* **The matrix is the dense kind's, bit for bit.**  Each builder's rows,
  scattered into a zero matrix, are compared with the ``H`` of the dense
  factory of the same kind: nearest neighbour on point sets built to tie
  (lattices, duplicates, co-circular sets, signed zeros, distances that
  underflow), the projection on grids that share boundaries, nest, overlap
  partly or not at all.
* **The tie rule**: the lowest index among the points at the minimal dense
  squared distance, which is what ``np.argmin`` returns.  A k-d tree only
  proposes candidates.
* **Every limit and refusal**, each raised before the allocation it
  bounds: an oversized row structure, a degenerate tie set, boundaries that
  do not rise, an index out of range, a value in an unused slot.  The two
  allocation bounds are also run in a process whose address space is
  capped, where a missing bound is a ``MemoryError``.
* **Scale**: a hundred thousand points per push and a million in the slow
  lane, built and applied in a process whose address space is capped.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json
import resource
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import sparse_mapping
from maddening.core.coupling.mapping import (
    matrix_mapping,
    nearest_neighbor_mapping,
    projection_1d_mapping,
)
from maddening.core.coupling.mapping_spec import (
    MappingRebuildError,
    PointReferenceError,
)
from maddening.core.coupling.sparse_mapping import (
    SparseMappingLimitError,
    sparse_matrix_mapping,
    sparse_nearest_neighbor_mapping,
    sparse_projection_1d_mapping,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.sparse_mapping_support import assert_within_rows, densify, valid_slots, x64

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = Path(sparse_mapping.__file__).resolve().parents[3]

FORMS = [("consistent", "gather"), ("conservative", "gather"), ("conservative", "scatter")]


def _same_matrix(sparse, dense, what=""):
    """The rows are the dense matrix: dtype, shape and every bit."""
    got, want = densify(sparse), np.asarray(dense.H)
    assert got.dtype == want.dtype and got.shape == want.shape, what
    assert got.tobytes() == want.tobytes(), (
        f"{what}: {int((got != want).sum())} entries differ, first at "
        f"{tuple(int(i) for i in np.argwhere(got != want)[0])}")
    assert (sparse.n_target, sparse.n_source) == (dense.n_target, dense.n_source)
    assert sparse.mode == dense.mode


# ---------------------------------------------------------------------------
# Nearest neighbour: point sets built to tie
# ---------------------------------------------------------------------------

def _lattice(dim: int, n: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """An integer lattice and the centres of its cells: every centre is at
    one distance from ``2 ** dim`` lattice points."""
    axis = np.arange(float(n))
    grid = np.stack(np.meshgrid(*([axis] * dim), indexing="ij"), axis=-1).reshape(-1, dim)
    centres = grid[(grid < n - 1).all(axis=1)] + 0.5
    return grid, centres


def _co_circular() -> tuple[np.ndarray, np.ndarray]:
    """64 sources on a circle whose centre is a target, with other targets
    on the circle and off it.  Integer points of one radius are exactly
    equidistant; the rest are equal only to rounding."""
    exact = [(x, y) for x in range(-65, 66) for y in range(-65, 66) if x * x + y * y == 4225]
    angle = np.linspace(0.0, 2.0 * np.pi, 28, endpoint=False)
    rounded = 65.0 * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    sources = np.concatenate([np.asarray(exact, dtype=np.float64), rounded])[:64]
    targets = np.array([[0.0, 0.0], [65.0, 0.0], [1e-9, -1e-9], [30.0, 40.0], [0.5, 0.25]])
    return sources, targets


def _point_sets() -> dict:
    rng = np.random.default_rng(17)
    sets = {}
    for dim in (1, 2, 3):
        grid, centres = _lattice(dim)
        sets[f"lattice-{dim}d"] = (grid, centres)
        sets[f"lattice-{dim}d-offset-1e9"] = (grid + 1e9, centres + 1e9)
        sets[f"lattice-{dim}d-scale-1e-12"] = (grid * 1e-12, centres * 1e-12)
        sets[f"lattice-{dim}d-scale-0.1"] = (grid * 0.1, centres * 0.1)
        sets[f"lattice-{dim}d-scale-0.1-offset-1e6"] = (grid * 0.1 + 1e6, centres * 0.1 + 1e6)
        sets[f"lattice-{dim}d-scale-1e-160"] = (grid * 1e-160, centres * 1e-160)
        sets[f"lattice-{dim}d-scale-1e-150"] = (grid * 1e-150, centres * 1e-150)
        sets[f"lattice-{dim}d-scale-1e149"] = (grid * 1e149, centres * 1e149)
    grid, centres = _lattice(2)
    shuffled = rng.permutation(np.repeat(grid, 3, axis=0))
    sets["duplicated-sources"] = (shuffled, centres)
    sets["targets-on-the-sources"] = (shuffled, np.concatenate([grid, grid[::-1]]))
    sets["co-circular"] = _co_circular()
    sets["all-sources-identical"] = (np.full((9, 3), 0.25), rng.normal(size=(6, 3)))
    sets["a-single-source"] = (np.array([[0.5, -0.5]]), rng.normal(size=(7, 2)))
    sets["a-single-target"] = (rng.normal(size=(7, 2)), np.array([[0.0, 0.0]]))
    sets["signed-zeros"] = (np.array([0.0, -0.0, 1.0, -1.0, -0.0, 0.0]),
                            np.array([-0.0, 0.0, 0.5, -0.5, 2.0]))
    sets["float32-inputs"] = (rng.normal(size=(40, 2)).astype(np.float32),
                              rng.normal(size=(25, 2)).astype(np.float32))
    sets["integer-inputs"] = (rng.integers(-4, 5, size=(60, 2)),
                              rng.integers(-4, 5, size=(40, 2)))
    for dim in (1, 2, 3, 9, 20):
        sets[f"random-{dim}d"] = (rng.normal(size=(50, dim)), rng.normal(size=(35, dim)))
    # A handful of values per axis: many exact ties, of every multiplicity.
    sets["few-distinct-values"] = (rng.integers(0, 4, size=(300, 3)).astype(np.float64),
                                   rng.integers(0, 4, size=(200, 3)) + 0.5)
    # One ulp either side of an exact tie.
    base = np.array([[3.0, 4.0], [4.0, 3.0], [5.0, 0.0], [0.0, 5.0], [-3.0, 4.0], [-5.0, 0.0]])
    nudged = np.concatenate([base, np.nextafter(base, 0.0), np.nextafter(base, np.inf)])
    sets["one-ulp-from-a-tie"] = (nudged, np.array([[0.0, 0.0], [1e-300, 0.0], [0.5, 0.5]]))
    return sets


POINT_SETS = _point_sets()


def test_the_point_sets_are_built_to_tie():
    """The fixture can express a wrong tie rule: in most sets the dense
    distance table has more than one minimum in a row, and the lowest index
    is not always the first or the last candidate a tree would return."""
    tied = 0
    for name, (src, tgt) in POINT_SETS.items():
        a = np.asarray(src, np.float64).reshape(len(src), -1)
        b = np.asarray(tgt, np.float64).reshape(len(tgt), -1)
        d2 = np.sum((b[:, None, :] - a[None, :, :]) ** 2, axis=-1)
        tied += int(((d2 == d2.min(axis=1, keepdims=True)).sum(axis=1) > 1).any())
    assert tied >= 25, tied
    grid, centres = _lattice(3)
    d2 = np.sum((centres[:, None, :] - grid[None, :, :]) ** 2, axis=-1)
    assert ((d2 == d2.min(axis=1, keepdims=True)).sum(axis=1) == 8).all()


@pytest.mark.parametrize("mode, transpose", FORMS,
                         ids=[f"{m}-{t}" for m, t in FORMS])
@pytest.mark.parametrize("name", sorted(POINT_SETS))
def test_sparse_nearest_neighbour_is_the_dense_matrix_bit_for_bit(name, mode, transpose):
    """Each point set both ways round, in both modes and both conservative
    forms: scattering the rows gives ``nearest_neighbor_mapping(...).H``."""
    a, b = POINT_SETS[name]
    for source, target in ((a, b), (b, a)):
        sparse = sparse_nearest_neighbor_mapping(source, target, mode=mode,
                                                 transpose=transpose)
        dense = nearest_neighbor_mapping(source, target, mode=mode)
        _same_matrix(sparse, dense, f"{name} {mode} {transpose}")
        assert sparse.layout == ("scatter" if transpose == "scatter" else "gather")
        assert sparse.weights.dtype == jnp.float32
        assert sparse.spec.hyperparameters == {"mode": mode, "transpose": transpose}


def test_the_nearest_index_is_the_dense_argmin_on_a_set_with_ties_of_every_size():
    """The rule itself, on 3000 points with few distinct coordinates (so
    most queries tie, between two and dozens of points) and many exact
    duplicates: the lowest index at the minimal dense squared distance."""
    rng = np.random.default_rng(3)
    points = rng.integers(0, 6, size=(3000, 2)).astype(np.float64)
    queries = rng.integers(0, 12, size=(2000, 2)) * 0.5
    d2 = np.sum((queries[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    expected = np.argmin(d2, axis=1)
    ties = (d2 == d2.min(axis=1, keepdims=True)).sum(axis=1)
    assert ties.max() > 50 and (ties > 1).mean() > 0.9
    got = sparse_mapping._nearest_lowest_index("test", points, queries)
    np.testing.assert_array_equal(got, expected)


def test_the_tie_search_runs_in_bounded_batches(monkeypatch):
    """The same answer when the points are searched a few at a time and
    each batch of candidate lists is small: the chunking is not part of
    the rule."""
    rng = np.random.default_rng(4)
    points = rng.integers(0, 5, size=(400, 2)).astype(np.float64)
    queries = rng.integers(0, 10, size=(300, 2)) * 0.5
    expected = np.argmin(np.sum((queries[:, None, :] - points[None, :, :]) ** 2, axis=-1),
                         axis=1)
    monkeypatch.setattr(sparse_mapping, "_TIE_CHUNK", 7)
    monkeypatch.setattr(sparse_mapping, "_TIE_BATCH_CANDIDATES", 5)
    np.testing.assert_array_equal(
        sparse_mapping._nearest_lowest_index("test", points, queries), expected)


class _NoisyTree:
    """A k-d tree whose reported distances are off by a relative *noise*,
    as another build's arithmetic (a fused multiply-add in the squared
    distance) or a pruning bound may make them.  The points it finds within
    a radius are the real tree's."""

    def __init__(self, points, noise):
        from scipy.spatial import KDTree

        self._tree = KDTree(points)
        self._noise = noise
        self._rng = np.random.default_rng(99)

    def query(self, x, k=2):
        distance, index = self._tree.query(x, k=k)
        distance = distance * (1.0 + self._noise * self._rng.uniform(-1.0, 1.0, distance.shape))
        order = np.argsort(distance, axis=1, kind="stable")
        return (np.take_along_axis(distance, order, axis=1),
                np.take_along_axis(index, order, axis=1))

    def query_ball_point(self, x, r, **kwargs):
        return self._tree.query_ball_point(x, r, **kwargs)


@pytest.mark.parametrize("scale, noise", [
    (1.0, 200 * np.finfo(np.float64).eps),
    (1e9, 200 * np.finfo(np.float64).eps),
    (1e-156, 1e-9),
    (1e-158, 1e-6),
], ids=["rounding at scale one", "rounding at scale 1e9", "squares gone subnormal",
        "squares almost gone"])
def test_the_tree_only_proposes_candidates_whatever_its_distances_round_to(
        monkeypatch, scale, noise):
    """The rule is the dense expression's, not the tree's.  A tree whose
    distances are off -- by a few hundred epsilons anywhere, or by far more
    where the squared distance is subnormal and one build's rounding of it
    need not be another's -- still gives the dense ``argmin``: every point
    it cannot tell from the nearest is handed to the dense expression."""
    grid, centres = _lattice(3, n=4)
    points = np.random.default_rng(5).permutation(np.concatenate([grid, grid])) * scale
    queries = np.concatenate([centres, grid[::3] + 0.25]) * scale
    d2 = np.sum((queries[:, None, :] - points[None, :, :]) ** 2, axis=-1)
    expected = np.argmin(d2, axis=1)
    assert ((d2 == d2.min(axis=1, keepdims=True)).sum(axis=1) > 1).all(), "premise: every query ties"
    monkeypatch.setattr(sparse_mapping, "_kdtree",
                        lambda: (lambda pts: _NoisyTree(pts, noise)))
    np.testing.assert_array_equal(
        sparse_mapping._nearest_lowest_index("test", points, queries), expected)
    # the premise: this tree, taken at its word, would choose otherwise
    tree = _NoisyTree(np.unique(points, axis=0), noise)
    distance, _index = tree.query(queries)
    assert (distance[:, 1] > distance[:, 0]).any(), "premise: the noise separates tied points"


def test_the_weights_are_float32_ones_under_x64_as_the_dense_kinds_are():
    src, tgt = POINT_SETS["random-2d"]
    with x64(True):
        for mode, transpose in FORMS:
            sparse = sparse_nearest_neighbor_mapping(src, tgt, mode=mode, transpose=transpose)
            assert sparse.weights.dtype == jnp.float32
            _same_matrix(sparse, nearest_neighbor_mapping(src, tgt, mode=mode))


def test_the_projection_weights_follow_the_boundaries_dtype_under_x64():
    """Both projection factories, as ``rbf_mapping``: float64 weights from
    float64 boundaries in an x64 process, float32 from float32 ones and
    with x64 off.  They were float32 in an x64 graph, where a row summed
    to one and the integral was preserved to float32 rounding only
    (2.98e-8 and 3.7e-9 on these grids)."""
    rng = np.random.default_rng(12)
    source = np.concatenate([[0.0], np.cumsum(rng.uniform(0.1, 1.0, 23))])
    target = np.linspace(0.0, source[-1], 8)                # covers the source grid exactly
    field64 = rng.uniform(-2.0, 2.0, 23)
    widths_s, widths_t = np.diff(source), np.diff(target)
    # The reference: the overlap formula, in float64, on the host.
    ref = np.zeros((7, 23))
    for i in range(7):
        for j in range(23):
            ref[i, j] = max(0.0, min(target[i + 1], source[j + 1]) - max(target[i], source[j])
                            ) / widths_t[i]
    eps = np.finfo(np.float64).eps
    with x64(True):
        dense = projection_1d_mapping(source, target)
        sparse = sparse_projection_1d_mapping(source, target)
        assert dense.H.dtype == jnp.float64 and sparse.weights.dtype == jnp.float64
        _same_matrix(sparse, dense)
        np.testing.assert_array_equal(np.asarray(dense.H), ref)
        assert np.max(np.abs(np.asarray(dense.H).sum(axis=1) - 1.0)) <= 8 * eps
        field = jnp.asarray(field64)
        for mapping in (dense, sparse):
            out = np.asarray(jax.jit(mapping.apply)(field))
            assert out.dtype == np.float64
            np.testing.assert_allclose(out, ref @ field64, rtol=0, atol=64 * eps)
            integral, want = float(widths_t @ out), float(widths_s @ field64)
            assert abs(integral - want) <= 64 * eps * float(widths_s @ np.abs(field64))
        # float32 boundaries are a float32 mapping in an x64 process too.
        dense32 = projection_1d_mapping(source.astype(np.float32), target.astype(np.float32))
        sparse32 = sparse_projection_1d_mapping(source.astype(np.float32),
                                                target.astype(np.float32))
        assert dense32.H.dtype == jnp.float32 and sparse32.weights.dtype == jnp.float32
        _same_matrix(sparse32, dense32)
    # With x64 off the weights are float32 whatever was passed.
    assert projection_1d_mapping(source, target).H.dtype == jnp.float32
    assert sparse_projection_1d_mapping(source, target).weights.dtype == jnp.float32


def test_conservative_nearest_neighbour_preserves_the_total_in_both_forms():
    """Every source adds to exactly one target: the column sums of the
    operator are one, and the total of a mapped field is the total of the
    field (to the rounding of the two sums)."""
    rng = np.random.default_rng(8)
    src, tgt = rng.uniform(size=(400, 2)), rng.uniform(size=(37, 2))
    field = jnp.asarray(rng.uniform(1.0, 2.0, size=400), jnp.float32)
    results = {}
    for transpose in ("gather", "scatter"):
        m = sparse_nearest_neighbor_mapping(src, tgt, mode="conservative",
                                            transpose=transpose)
        np.testing.assert_array_equal(densify(m).sum(axis=0), np.ones(400, np.float32))
        out = jax.jit(m.apply)(field)
        assert_within_rows(out, m, field, what=transpose)
        results[transpose] = np.asarray(out, np.float64)
        assert abs(results[transpose].sum() - float(np.sum(np.asarray(field, np.float64)))) \
            <= 400 * np.finfo(np.float32).eps * float(np.sum(np.asarray(field, np.float64)))
    dense = np.asarray(nearest_neighbor_mapping(src, tgt, mode="conservative").H @ field,
                       np.float64)
    # three evaluations of the same sums of positive terms
    bound = 64 * np.finfo(np.float32).eps * np.abs(dense)
    assert np.all(np.abs(results["gather"] - dense) <= bound)
    assert np.all(np.abs(results["scatter"] - dense) <= bound)
    assert np.all(np.abs(results["gather"] - results["scatter"]) <= bound)
    # the gather form lists a target's sources in ascending order
    g = sparse_nearest_neighbor_mapping(src, tgt, mode="conservative")
    for row, count in zip(g.indices, g.counts):
        assert (np.diff(row[:count]) > 0).all()


NN_REFUSED = {
    "no source points": (lambda: (np.zeros((0, 2)), np.zeros((3, 2)), {}),
                         "source_points holds no points"),
    "no target points": (lambda: (np.zeros((3, 2)), np.zeros((0, 2)), {}),
                         "target_points holds no points"),
    "points without coordinates": (lambda: (np.zeros((3, 0)), np.zeros((3, 0)), {}),
                                   "its points have no coordinates"),
    "a three-dimensional array": (lambda: (np.zeros((3, 2, 2)), np.zeros((3, 2)), {}),
                                  r"source_points must be an \(n,\) or \(n, d\) array"),
    "a scalar": (lambda: (np.float64(1.0), np.zeros(3), {}),
                 r"source_points must be an \(n,\) or \(n, d\) array"),
    "two dimensions": (lambda: (np.zeros((3, 2)), np.zeros((3, 3)), {}),
                       "must share a dimension, got 2 and 3"),
    "a NaN source": (lambda: (np.array([0.0, np.nan, 1.0]), np.zeros(2), {}),
                     "source_points holds a non-finite coordinate at index 1"),
    "an infinite target": (lambda: (np.zeros(2), np.array([[0.0, 1.0], [np.inf, 0.0]]), {}),
                           "target_points holds a non-finite coordinate at index 1"),
    "a coordinate past 1e150": (lambda: (np.array([0.0, 2e150]), np.zeros(2), {}),
                                "source_points holds a coordinate of magnitude above 1e\\+150"),
    "complex points": (lambda: (np.zeros(3, np.complex128), np.zeros(2), {}),
                       "source_points must be a real numeric array"),
    "text points": (lambda: (np.array(["a", "b"]), np.zeros(2), {}),
                    "source_points must be a real numeric array"),
    "object points": (lambda: (np.array([1.0, None], dtype=object), np.zeros(2), {}),
                      "source_points must be a real numeric array"),
    "an unknown mode": (lambda: (np.zeros(2), np.zeros(2), dict(mode="sideways")),
                        "mode='sideways' not in"),
    "an unknown transpose": (lambda: (np.zeros(2), np.zeros(2),
                                      dict(mode="conservative", transpose="segment")),
                             "transpose='segment' not in"),
    "scatter without a transpose": (lambda: (np.zeros(2), np.zeros(2),
                                             dict(transpose="scatter")),
                                    "mode='consistent' has none and takes only "
                                    "transpose='gather'"),
}


@pytest.mark.parametrize("name", sorted(NN_REFUSED))
def test_sparse_nearest_neighbour_refuses_what_it_cannot_map(name):
    make, message = NN_REFUSED[name]
    source, target, kwargs = make()
    with pytest.raises(ValueError, match=message):
        sparse_nearest_neighbor_mapping(source, target, **kwargs)


def test_extended_precision_and_traced_points_are_refused_not_narrowed():
    if np.dtype(np.longdouble).itemsize > 8:
        with pytest.raises(PointReferenceError, match="extended-precision"):
            sparse_nearest_neighbor_mapping(np.zeros(3, np.longdouble), np.zeros(2))
        with pytest.raises(PointReferenceError, match="extended-precision"):
            sparse_projection_1d_mapping(np.arange(3, dtype=np.longdouble), np.arange(3.0))
        with pytest.raises(PointReferenceError, match="extended-precision"):
            sparse_matrix_mapping(np.zeros((2, 1), np.int64), np.ones((2, 1), np.longdouble),
                                  n_source=1)

    def traced(points):
        return sparse_nearest_neighbor_mapping(points, np.zeros(2)).weights

    with pytest.raises(Exception, match="traced value|Tracer"):
        jax.jit(traced)(jnp.zeros(3))


def test_a_missing_scipy_is_an_import_error_that_names_it(monkeypatch):
    """scipy is a base dependency; if it cannot be imported the builder
    says so, and a loader reports it against the edge like any other
    rebuild failure."""
    monkeypatch.setitem(sys.modules, "scipy.spatial", None)
    src, tgt = POINT_SETS["random-2d"]
    with pytest.raises(ImportError, match="needs scipy"):
        sparse_nearest_neighbor_mapping(src, tgt)
    config = _config({"kind": "sparse_nearest_neighbor",
                      "points": {"source_points": [0.0, 0.5, 1.0],
                                 "target_points": [0.1, 0.9]}}, 3, 2)
    with pytest.raises(MappingRebuildError, match=r"edge a.v -> b.inp.*needs scipy"):
        GraphManager.from_dict(config, REGISTRY)
    # a single distinct source needs no tree
    assert sparse_nearest_neighbor_mapping(np.zeros((4, 2)), tgt).n_source == 4


# ---------------------------------------------------------------------------
# 1-D projection
# ---------------------------------------------------------------------------

def _grids() -> dict:
    rng = np.random.default_rng(23)

    def rising(n, low=0.0, high=1.0):
        return np.sort(rng.uniform(low, high, size=n))

    uniform = np.linspace(0.0, 1.0, 13)
    grids = {
        "random": (rising(30), rising(17)),
        "random-fine-to-coarse": (rising(200), rising(9)),
        "random-coarse-to-fine": (rising(9), rising(200)),
        "identical": (uniform, uniform.copy()),
        "nested-refinement": (uniform, np.linspace(0.0, 1.0, 37)),
        "nested-coarsening": (np.linspace(0.0, 1.0, 37), uniform),
        "shared-boundaries": (np.array([0.0, 0.25, 0.5, 1.0, 2.0]),
                              np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])),
        "target-wider-than-source": (rising(20, 0.3, 0.6), rising(25, 0.0, 1.0)),
        "source-wider-than-target": (rising(25, 0.0, 1.0), rising(20, 0.3, 0.6)),
        "disjoint-target-left": (rising(10, 2.0, 3.0), rising(10, 0.0, 1.0)),
        "disjoint-target-right": (rising(10, 0.0, 1.0), rising(10, 2.0, 3.0)),
        "touching": (np.array([0.0, 0.5, 1.0]), np.array([1.0, 1.5, 2.0])),
        "one-cell-each": (np.array([0.0, 1.0]), np.array([0.25, 0.5])),
        "one-target-over-everything": (rising(50), np.array([-1.0, 2.0])),
        "one-source-under-everything": (np.array([-1.0, 2.0]), rising(50)),
        "float32-boundaries": (rising(30).astype(np.float32), rising(17).astype(np.float32)),
        "integer-boundaries": (np.arange(0, 20, 2), np.arange(1, 18, 3)),
        "offset-1e9": (1e9 + rising(30), 1e9 + rising(17)),
        "tiny-cells": (rising(30) * 1e-200, rising(17) * 1e-200),
        "negative": (-rising(30)[::-1], -rising(17)[::-1]),
        "adjacent-floats": (np.nextafter(np.float64(1.0), 2.0) ** np.arange(6),
                            np.array([1.0, np.nextafter(1.0, 2.0), 1.0 + 1e-15, 1.0 + 2e-15])),
    }
    return grids


GRIDS = _grids()


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_the_sparse_projection_is_the_dense_double_loop_bit_for_bit(name):
    """Both ways round: every entry is the dense factory's expression, a
    row lists its source cells in ascending order, and a target cell
    outside the source grid has an empty row."""
    a, b = GRIDS[name]
    for source, target in ((a, b), (b, a)):
        sparse = sparse_projection_1d_mapping(source, target)
        dense = projection_1d_mapping(source, target)
        _same_matrix(sparse, dense, name)
        assert sparse.kind == "sparse_projection_1d" and sparse.layout == "gather"
        assert sparse.weights.dtype == jnp.float32
        counts = (np.full(sparse.n_target, sparse.k) if sparse.counts is None
                  else np.asarray(sparse.counts))
        np.testing.assert_array_equal(counts, np.count_nonzero(np.asarray(dense.H), axis=1))
        for row, count in zip(sparse.indices, counts):
            assert (np.diff(row[:count]) == 1).all(), "consecutive source cells, ascending"


def test_the_grids_cover_empty_rows_shared_boundaries_and_padding():
    counts = []
    for name, (source, target) in GRIDS.items():
        m = sparse_projection_1d_mapping(source, target)
        counts.append(np.full(m.n_target, m.k) if m.counts is None else np.asarray(m.counts))
    assert any((c == 0).all() for c in counts), "a grid pair with no overlap"
    assert any((c == 0).any() and (c > 0).any() for c in counts), "an empty row beside others"
    assert any(len(set(c.tolist())) > 2 for c in counts), "rows of several lengths"
    assert any(c.max() >= 40 for c in counts), "a long row"


def test_the_projection_preserves_the_integral_of_a_field_it_covers():
    rng = np.random.default_rng(31)
    source, target = np.sort(rng.uniform(0.2, 0.8, 40)), np.linspace(0.0, 1.0, 12)
    source[0], source[-1] = 0.2, 0.8
    m = sparse_projection_1d_mapping(source, target)
    cell_values = rng.uniform(1.0, 2.0, size=39)
    mapped = np.asarray(jax.jit(m.apply)(jnp.asarray(cell_values, jnp.float32)), np.float64)
    integral = float(np.sum(cell_values * np.diff(source)))
    assert abs(float(np.sum(mapped * np.diff(target))) - integral) <= 1e-5 * integral
    assert m.mode == "conservative" and m.spec.hyperparameters == {}


def _replace(array, index, value):
    out = np.array(array, dtype=np.float64)
    out[index] = value
    return out


RISING = np.linspace(0.0, 1.0, 6)

PROJECTION_REFUSED = {
    "a descending source": (RISING[::-1], RISING,
                            r"source_boundaries must be strictly increasing, but "
                            r"source_boundaries\[1\] = 0.8 is not greater than "
                            r"source_boundaries\[0\] = 1.0"),
    "a descending target": (RISING, RISING[::-1],
                            "target_boundaries must be strictly increasing"),
    "a source that turns back": (_replace(RISING, 3, 0.1), RISING,
                                 r"source_boundaries\[3\] = 0.1 is not greater than"),
    "a repeated boundary": (_replace(RISING, 2, RISING[1]), RISING,
                            r"source_boundaries\[2\] = 0.2 is not greater than "
                            r"source_boundaries\[1\] = 0.2"),
    "one boundary": (np.array([0.5]), RISING,
                     "source_boundaries needs at least two boundaries"),
    "no boundary": (np.zeros(0), RISING, "source_boundaries needs at least two boundaries"),
    "a two-dimensional array": (RISING.reshape(2, 3), RISING,
                                "source_boundaries must be a one-dimensional array"),
    "a NaN boundary": (_replace(RISING, 2, np.nan), RISING,
                       "source_boundaries holds a non-finite value at index 2"),
    "an infinite boundary": (RISING, _replace(RISING, 5, np.inf),
                             "target_boundaries holds a non-finite value at index 5"),
    "complex boundaries": (RISING.astype(np.complex128), RISING,
                           "source_boundaries must be a real numeric array"),
}


@pytest.mark.parametrize("name", sorted(PROJECTION_REFUSED))
def test_the_sparse_projection_refuses_boundaries_that_do_not_rise(name):
    """The dense kind computes an operator from these (every row zero for a
    descending source); the sparse kind refuses them, and does not sort or
    reverse them, because the field keeps its cell order."""
    source, target, message = PROJECTION_REFUSED[name]
    with pytest.raises(ValueError, match=message):
        sparse_projection_1d_mapping(source, target)


# ---------------------------------------------------------------------------
# Your own rows
# ---------------------------------------------------------------------------

def _rows(seed=0, n_target=9, n_source=6, k=4):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_source, size=(n_target, k))
    values = rng.normal(size=(n_target, k)).astype(np.float32)
    unused = rng.uniform(size=(n_target, k)) < 0.4
    unused[0] = False
    unused[1] = True
    indices[unused] = -1
    values[unused] = 0.0
    return indices, values


def test_sparse_matrix_keeps_each_rows_entries_in_the_order_given():
    indices, values = _rows()
    m = sparse_matrix_mapping(indices, values, n_source=6)
    used = indices >= 0
    assert (used[:, :-1] < used[:, 1:]).any(), "premise: an unused slot before a used one"
    for i in range(indices.shape[0]):
        count = int(used[i].sum())
        np.testing.assert_array_equal(m.indices[i, :count], indices[i][used[i]])
        np.testing.assert_array_equal(np.asarray(m.weights)[i, :count], values[i][used[i]])
        assert not m.indices[i, count:].any() and not np.asarray(m.weights)[i, count:].any()
    np.testing.assert_array_equal(m.counts, used.sum(axis=1))
    assert m.kind == "sparse_matrix" and m.layout == "gather" and m.n_source == 6


@pytest.mark.parametrize("dtype, enabled", [("float32", False), ("float64", True),
                                            ("float16", False)])
def test_sparse_matrix_is_matrix_mapping_of_the_same_rows(dtype, enabled):
    """``sparse_matrix`` against ``matrix_mapping`` of the densified rows:
    the same matrix, and results within the rounding of a row sum."""
    with x64(enabled):
        indices, values = _rows(seed=2)
        indices[0] = np.arange(4)              # no index repeated: a slot is a matrix entry
        m = sparse_matrix_mapping(indices, values.astype(dtype), n_source=6)
        assert str(m.weights.dtype) == dtype
        H = np.zeros((9, 6), dtype)
        used = indices >= 0
        rows = np.broadcast_to(np.arange(9)[:, None], indices.shape)
        np.add.at(H, (rows[used], indices[used]), values.astype(dtype)[used])
        dense = matrix_mapping(H)
        field = jnp.asarray(np.random.default_rng(5).normal(size=(6, 2)), dtype)
        out = jax.jit(m.apply)(field)
        assert out.dtype == dense.apply(field).dtype
        assert_within_rows(out, m, field)


def test_a_repeated_index_in_a_row_adds():
    m = sparse_matrix_mapping(np.array([[1, 1, 0]]), np.array([[2.0, 3.0, 1.0]], np.float32),
                              n_source=2)
    np.testing.assert_array_equal(np.asarray(m.apply(jnp.asarray([10.0, 1.0]))), [15.0])


def test_sparse_matrix_takes_unsigned_and_narrow_index_dtypes_and_a_negative_zero():
    values = np.array([[1.0, -0.0], [0.5, 0.25]], np.float32)
    for dtype in (np.uint8, np.uint64, np.int8, np.int32):
        m = sparse_matrix_mapping(np.array([[3, 0], [1, 2]], dtype), values, n_source=4)
        assert m.indices.dtype == np.int32 and m.counts is None
    # -0.0 in an unused slot is zero
    m = sparse_matrix_mapping(np.array([[3, -1], [1, 2]]), values, n_source=4)
    np.testing.assert_array_equal(m.counts, [1, 2])


def _matrix_args(**changes):
    kwargs = dict(indices=np.array([[0, 2, -1], [1, -1, -1]]),
                  values=np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]], np.float32), n_source=3)
    kwargs.update(changes)
    return kwargs


MATRIX_REFUSED = {
    "a value in an unused slot": (
        dict(values=np.array([[1.0, 2.0, 0.5], [3.0, 0.0, 0.0]], np.float32)),
        r"values\[0, 2\] = 0.5 belongs to an unused slot \(indices\[0, 2\] = -1\); it "
        r"would be dropped"),
    "an index below -1": (dict(indices=np.array([[0, -2, -1], [1, -1, -1]])),
                          r"indices\[0, 1\] = -2 is neither a source index"),
    "an index of n_source": (dict(indices=np.array([[0, 3, -1], [1, -1, -1]])),
                             r"indices\[0, 1\] = 3 is neither a source index in "
                             r"\[0, n_source\) = \[0, 3\)"),
    "a float index": (dict(indices=np.array([[0.0, 2.0, -1.0], [1.0, -1.0, -1.0]])),
                      "indices must be an integer array, got dtype float64"),
    "a boolean index": (dict(indices=np.zeros((2, 3), bool)),
                        "indices must be an integer array, got dtype bool"),
    "integer values": (dict(values=np.ones((2, 3), np.int64)),
                       "values must be a floating-point array, got dtype int64"),
    "complex values": (dict(values=np.ones((2, 3), np.complex64)),
                       "values must be a floating-point array, got dtype complex64"),
    "a NaN value": (dict(values=np.array([[1.0, np.nan, 0.0], [3.0, 0.0, 0.0]], np.float32)),
                    r"values holds a non-finite value at index \(0, 1\) \(nan\)"),
    "an infinite value": (dict(values=np.array([[1.0, 2.0, 0.0], [np.inf, 0.0, 0.0]],
                                               np.float32)),
                          r"values holds a non-finite value at index \(1, 0\) \(inf\)"),
    "values of another shape": (dict(values=np.ones((2, 2), np.float32)),
                                "one value per slot"),
    "a one-dimensional index": (dict(indices=np.array([0, 1]), values=np.ones(2, np.float32)),
                                r"indices must have shape \(n_target, k\)"),
    "no rows": (dict(indices=np.zeros((0, 3), np.int64), values=np.zeros((0, 3), np.float32)),
                r"indices must have shape \(n_target, k\)"),
    "no slots": (dict(indices=np.zeros((2, 0), np.int64), values=np.zeros((2, 0), np.float32)),
                 r"indices must have shape \(n_target, k\)"),
    "n_source zero": (dict(n_source=0), "n_source must be between 1 and"),
    "n_source past int32": (dict(n_source=2 ** 31), "n_source must be between 1 and"),
    "n_source a float": (dict(n_source=3.0), "n_source must be an integer"),
    "n_source a bool": (dict(n_source=True), "n_source must be an integer"),
    "an unknown mode": (dict(mode="sideways"), "mode='sideways' not in"),
    "a name that is not text": (dict(name=7), "name must be a string label"),
    "a node reference": (dict(indices_asset={"node": "a", "field": "v"}),
                         "indices_asset= must name a .npy/.npz file"),
    "an inline reference": (dict(values_asset={"inline": [[1.0]]}),
                            "values_asset= must name a .npy/.npz file"),
    "an asset that is not a NumPy file": (dict(values_asset="values.csv"),
                                          "must be a .npy or .npz file"),
    "an asset outside the directory": (dict(indices_asset="../indices.npy"),
                                       "must not contain '..'"),
}


@pytest.mark.parametrize("name", sorted(MATRIX_REFUSED))
def test_sparse_matrix_refuses_rows_it_would_have_to_guess_at(name):
    changes, message = MATRIX_REFUSED[name]
    with pytest.raises(ValueError, match=message):
        sparse_matrix_mapping(**_matrix_args(**changes))


def test_n_source_is_required_and_sizes_nothing():
    """It cannot be inferred, so it is asked for; and it bounds the index
    check only, so the largest value an int32 index allows builds at once
    (a hyper-parameter a config holds cannot allocate)."""
    args = _matrix_args()
    del args["n_source"]
    with pytest.raises(TypeError, match="n_source"):
        sparse_matrix_mapping(**args)
    m = sparse_matrix_mapping(**_matrix_args(n_source=2 ** 31 - 1))
    assert m.n_source == 2 ** 31 - 1 and m.indices.shape == (2, 3)


def test_bfloat16_values_build_but_cannot_be_named_as_an_asset():
    """A ``.npy`` file stores a bfloat16 array as raw bytes, so a reference
    to one could be written and never read back."""
    values = np.asarray(jnp.ones((2, 3), jnp.bfloat16) * jnp.asarray(
        [[1.0, 2.0, 0.0], [3.0, 0.0, 0.0]], jnp.bfloat16))
    m = sparse_matrix_mapping(np.array([[0, 2, -1], [1, -1, -1]]), values, n_source=3)
    assert m.weights.dtype == jnp.bfloat16
    with pytest.raises(ValueError, match="which a .npy asset cannot hold"):
        sparse_matrix_mapping(np.array([[0, 2, -1], [1, -1, -1]]), values, n_source=3,
                              values_asset="values.npy")


# ---------------------------------------------------------------------------
# Limits, through a factory and through a config
# ---------------------------------------------------------------------------

class Vec(SimulationNode):
    """n-vector integrating its boundary input."""

    def __init__(self, name, timestep, n=3):
        super().__init__(name, timestep, n=n)

    def initial_state(self):
        return {"v": jnp.arange(1, self.params["n"] + 1, dtype=jnp.float32)}

    def update(self, s, bi, dt):
        return {"v": s["v"] + dt * bi.get("inp", jnp.zeros_like(s["v"]))}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=(self.params["n"],), description="i")}


REGISTRY = {"Vec": Vec}


def _config(mapping, n_source, n_target) -> dict:
    return {"nodes": [{"type": "Vec", "name": "a", "timestep": 1.0, "params": {"n": n_source}},
                      {"type": "Vec", "name": "b", "timestep": 1.0, "params": {"n": n_target}}],
            "edges": [{"source_node": "a", "target_node": "b", "source_field": "v",
                       "target_field": "inp", "mapping": mapping}],
            "external_inputs": []}


def _skewed(n_source=40, n_target=10):
    """Every source nearest to target 0: the gather form's first row is
    ``n_source`` long and the other rows are empty."""
    source = np.linspace(0.0, 1e-3, n_source)
    target = np.concatenate([[0.0], np.linspace(10.0, 20.0, n_target - 1)])
    return source, target


def test_a_skewed_conservative_pattern_is_refused_and_the_error_names_scatter(monkeypatch):
    """The padded gather needs ``n_target * longest row`` slots.  Past the
    byte cap it is refused from the row counts, naming the sizes and the
    scatter form -- which builds the same operator in one slot per source."""
    source, target = _skewed()
    monkeypatch.setattr(sparse_mapping, "MAX_SPARSE_STRUCTURE_BYTES", 10 * 40 * 8 - 1)
    with pytest.raises(SparseMappingLimitError) as refused:
        sparse_nearest_neighbor_mapping(source, target, mode="conservative")
    message = str(refused.value)
    for part in ("10 rows of 40 slots at 8 bytes a slot = 3200 bytes",
                 "MAX_SPARSE_STRUCTURE_BYTES=3199", "n_target=10", "largest row 40",
                 "median row 0", "40 entries", "transpose='scatter'",
                 "not reproducible run to run on a GPU",
                 "sparse_mapping.MAX_SPARSE_STRUCTURE_BYTES"):
        assert part in message, part
    assert isinstance(refused.value, ValueError)
    scattered = sparse_nearest_neighbor_mapping(source, target, mode="conservative",
                                                transpose="scatter")
    assert scattered.indices.shape == (40, 1)
    _same_matrix(scattered, nearest_neighbor_mapping(source, target, mode="conservative"))
    # one byte more and the gather form is taken
    monkeypatch.setattr(sparse_mapping, "MAX_SPARSE_STRUCTURE_BYTES", 10 * 40 * 8)
    gathered = sparse_nearest_neighbor_mapping(source, target, mode="conservative")
    assert gathered.indices.shape == (10, 40)
    field = jnp.arange(40.0)
    np.testing.assert_array_equal(np.asarray(gathered.apply(field)),
                                  np.asarray(scattered.apply(field)))


@pytest.mark.parametrize("form", ["consistent", "scatter", "projection", "matrix"])
def test_every_builder_checks_the_row_structure_against_the_byte_cap(form, monkeypatch):
    """One slot per row is still a structure: each builder refuses one over
    the cap, and takes one exactly at it."""
    points = np.linspace(0.0, 1.0, 50)
    builds = {
        "consistent": (lambda: sparse_nearest_neighbor_mapping(points[:7], points), 50 * 8),
        "scatter": (lambda: sparse_nearest_neighbor_mapping(
            points, points[:7], mode="conservative", transpose="scatter"), 50 * 8),
        # 8 target cells, each over 6 of the 49 source cells
        "projection": (lambda: sparse_projection_1d_mapping(points, points[::6]), 8 * 6 * 8),
        "matrix": (lambda: sparse_matrix_mapping(np.zeros((50, 3), np.int64),
                                                 np.ones((50, 3), np.float32), n_source=1),
                   50 * 3 * 8),
    }
    build, needed = builds[form]
    monkeypatch.setattr(sparse_mapping, "MAX_SPARSE_STRUCTURE_BYTES", needed - 1)
    with pytest.raises(SparseMappingLimitError, match=f"= {needed} bytes, more than "
                                                      f"MAX_SPARSE_STRUCTURE_BYTES={needed - 1}"):
        build()
    monkeypatch.setattr(sparse_mapping, "MAX_SPARSE_STRUCTURE_BYTES", needed)
    assert build().nnz > 0


def test_the_default_byte_cap_is_the_asset_cap():
    from maddening.core.coupling.mapping_spec import MAX_ASSET_BYTES

    assert sparse_mapping.MAX_SPARSE_STRUCTURE_BYTES == MAX_ASSET_BYTES == 256 * 1024 * 1024
    assert sparse_mapping.TIE_CANDIDATES_PER_POINT == 8
    assert sparse_mapping.TIE_CANDIDATES_FLOOR == 1_000_000
    assert sparse_mapping._TIE_CHUNK == 4096        # the guide's "4096 points at a time"


def _degenerate(n_target=30):
    """Every target at the centre of a circle of integer points: each is at
    one distance from all of them."""
    exact = [(x, y) for x in range(-65, 66) for y in range(-65, 66) if x * x + y * y == 4225]
    return np.asarray(exact, dtype=np.float64), np.zeros((n_target, 2))


def test_a_degenerate_tie_set_is_refused_before_its_candidates_are_collected(monkeypatch):
    source, target = _degenerate()
    n_candidates = len(source) * len(target)
    assert len(source) >= 30
    monkeypatch.setattr(sparse_mapping, "TIE_CANDIDATES_PER_POINT", 2)
    monkeypatch.setattr(sparse_mapping, "TIE_CANDIDATES_FLOOR", n_candidates - 2 * 30 - 1)
    with pytest.raises(SparseMappingLimitError) as refused:
        sparse_nearest_neighbor_mapping(source, target)
    message = str(refused.value)
    for part in ("the point set is degenerate", f"more than {n_candidates - 1} candidates",
                 "TIE_CANDIDATES_PER_POINT=2 for each of 30 searched points",
                 f"the largest tie being {len(source)} points"):
        assert part in message, part
    # exactly at the bound it is resolved, by the dense rule
    monkeypatch.setattr(sparse_mapping, "TIE_CANDIDATES_FLOOR", n_candidates - 2 * 30)
    _same_matrix(sparse_nearest_neighbor_mapping(source, target),
                 nearest_neighbor_mapping(source, target))


class _CountingTree:
    """The real k-d tree, recording how many points each search was asked
    for and how many each collection of candidate lists was."""

    def __init__(self, points, searched, collected):
        from scipy.spatial import KDTree

        self._tree = KDTree(points)
        self._searched = searched
        self._collected = collected

    def query(self, x, k=2):
        self._searched.append(len(x))
        return self._tree.query(x, k=k)

    def query_ball_point(self, x, r, **kwargs):
        if not kwargs.get("return_length"):
            self._collected.append(len(x))
        return self._tree.query_ball_point(x, r, **kwargs)


def test_a_degenerate_set_is_refused_after_the_chunk_that_passes_the_bound(monkeypatch):
    """The search runs a chunk of points at a time and counts each chunk's
    candidates against one bound for the whole search.  A set on which the
    tree cannot prune -- every search measures every point -- is refused
    as soon as the count passes the bound: the points after that chunk are
    never searched, and that chunk's candidates are counted, not collected."""
    source, target = _degenerate()          # 30 targets, each tied with every source
    per_chunk = 7 * len(source)
    searched: list = []
    collected: list = []
    monkeypatch.setattr(sparse_mapping, "_kdtree",
                        lambda: (lambda pts: _CountingTree(pts, searched, collected)))
    monkeypatch.setattr(sparse_mapping, "_TIE_CHUNK", 7)
    monkeypatch.setattr(sparse_mapping, "TIE_CANDIDATES_PER_POINT", 0)
    # one candidate short of the first chunk's: refused after it
    monkeypatch.setattr(sparse_mapping, "TIE_CANDIDATES_FLOOR", per_chunk - 1)
    with pytest.raises(SparseMappingLimitError,
                       match=f"after 7 of 30 searched points there were {per_chunk},"):
        sparse_nearest_neighbor_mapping(source, target)
    assert searched == [7] and collected == []
    # one chunk fits and two do not: the count is of the whole search
    searched.clear()
    monkeypatch.setattr(sparse_mapping, "TIE_CANDIDATES_FLOOR", 2 * per_chunk - 1)
    with pytest.raises(SparseMappingLimitError,
                       match=f"after 14 of 30 searched points there were {2 * per_chunk},"):
        sparse_nearest_neighbor_mapping(source, target)
    assert searched == [7, 7] and collected == [7]
    # with room for every candidate, the chunks give the dense matrix
    searched.clear()
    collected.clear()
    monkeypatch.setattr(sparse_mapping, "TIE_CANDIDATES_FLOOR", 30 * len(source))
    _same_matrix(sparse_nearest_neighbor_mapping(source, target),
                 nearest_neighbor_mapping(source, target))
    assert searched == [7, 7, 7, 7, 2] and collected == [7, 7, 7, 7, 2]


def test_the_tie_bound_counts_the_points_searched_from_in_each_mode(monkeypatch):
    """Consistent searches from the targets, conservative from the sources:
    the bound scales with whichever set is searched from."""
    circle, centre = _degenerate(n_target=1)
    monkeypatch.setattr(sparse_mapping, "TIE_CANDIDATES_FLOOR", 0)
    monkeypatch.setattr(sparse_mapping, "TIE_CANDIDATES_PER_POINT", len(circle) - 1)
    with pytest.raises(SparseMappingLimitError, match="for each of 1 searched points"):
        sparse_nearest_neighbor_mapping(circle, centre)
    with pytest.raises(SparseMappingLimitError, match="for each of 1 searched points"):
        sparse_nearest_neighbor_mapping(centre, circle, mode="conservative")
    monkeypatch.setattr(sparse_mapping, "TIE_CANDIDATES_PER_POINT", len(circle))
    _same_matrix(sparse_nearest_neighbor_mapping(circle, centre),
                 nearest_neighbor_mapping(circle, centre))


def test_a_limit_met_while_loading_a_config_names_the_edge(monkeypatch, tmp_path):
    """The same builders run inside ``from_dict``: each cap is a
    ``MappingRebuildError`` naming the edge and the kind, with the limit
    error chained."""
    source, target = _skewed()
    np.save(tmp_path / "source.npy", source)
    np.save(tmp_path / "target.npy", target)
    spec = {"kind": "sparse_nearest_neighbor", "mode": "conservative",
            "points": {"source_points": {"asset": "source.npy"},
                       "target_points": {"asset": "target.npy"}}}
    monkeypatch.setattr(sparse_mapping, "MAX_SPARSE_STRUCTURE_BYTES", 1000)
    with pytest.raises(MappingRebuildError, match=r"edge a.v -> b.inp: cannot rebuild "
                                                  r"interface mapping \(kind "
                                                  r"'sparse_nearest_neighbor'\): "
                                                  r"SparseMappingLimitError") as refused:
        GraphManager.from_dict(_config(spec, 40, 10), REGISTRY, base_dir=tmp_path)
    assert isinstance(refused.value.__cause__, SparseMappingLimitError)
    gm = GraphManager.from_dict(_config({**spec, "transpose": "scatter"}, 40, 10), REGISTRY,
                                base_dir=tmp_path)
    assert gm.edges[0].mapping.layout == "scatter"

    circle, centre = _degenerate()
    np.save(tmp_path / "circle.npy", circle)
    np.save(tmp_path / "centre.npy", centre)
    monkeypatch.setattr(sparse_mapping, "TIE_CANDIDATES_PER_POINT", 0)
    monkeypatch.setattr(sparse_mapping, "TIE_CANDIDATES_FLOOR", 100)
    tied = {"kind": "sparse_nearest_neighbor",
            "points": {"source_points": {"asset": "circle.npy"},
                       "target_points": {"asset": "centre.npy"}}}
    with pytest.raises(MappingRebuildError, match="the point set is degenerate"):
        GraphManager.from_dict(_config(tied, len(circle), 30), REGISTRY, base_dir=tmp_path)


def test_the_asset_cap_bounds_the_arrays_of_every_sparse_kind(monkeypatch, tmp_path):
    """The reference resolver reads a header before it allocates, for every
    kind: an asset over ``MAX_ASSET_BYTES`` never reaches a sparse builder."""
    from maddening.core.coupling import mapping_spec

    np.save(tmp_path / "big.npy", np.zeros(2000))
    np.save(tmp_path / "small.npy", np.linspace(0.0, 1.0, 5))
    np.save(tmp_path / "rows.npy", np.zeros((2000, 1), np.int64))
    np.save(tmp_path / "values.npy", np.ones((2000, 1), np.float32))
    monkeypatch.setattr(mapping_spec, "MAX_ASSET_BYTES", 2000 * 8 - 1)
    for mapping in (
        {"kind": "sparse_nearest_neighbor",
         "points": {"source_points": {"asset": "big.npy"},
                    "target_points": {"asset": "small.npy"}}},
        {"kind": "sparse_projection_1d",
         "points": {"source_boundaries": {"asset": "small.npy"},
                    "target_boundaries": {"asset": "big.npy"}}},
        {"kind": "sparse_matrix", "n_source": 3,
         "points": {"indices": {"asset": "rows.npy"}, "values": {"asset": "values.npy"}}},
    ):
        with pytest.raises(MappingRebuildError, match="more than MAX_ASSET_BYTES"):
            GraphManager.from_dict(_config(mapping, 3, 3), REGISTRY, base_dir=tmp_path)


# ---------------------------------------------------------------------------
# In a process whose address space is capped
# ---------------------------------------------------------------------------

_GIB = 1024 ** 3


def _capped(script: str, *arguments: str, address_space: int, timeout: int = 900) -> dict:
    """Run *script* in a fresh interpreter whose address space is capped
    (``RLIMIT_AS``, what ``prlimit --as`` sets) and return what it printed
    after ``RESULT``.  An allocation the cap does not allow is a
    ``MemoryError`` there, not a machine out of memory."""
    def cap():
        resource.setrlimit(resource.RLIMIT_AS, (address_space, address_space))

    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "PYTHONPATH": os.pathsep.join([str(SRC), str(REPO_ROOT)]),
           # one thread pool per library: the cap counts every thread's arena
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MALLOC_ARENA_MAX": "2"}
    done = subprocess.run([sys.executable, "-c", textwrap.dedent(script), *arguments],
                          capture_output=True, text=True, timeout=timeout, env=env,
                          preexec_fn=cap)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.rsplit("RESULT", 1)[1])


_REFUSALS_UNDER_A_CAP = '''
    import json, sys
    import numpy as np
    from maddening.core.coupling import sparse_mapping as sm

    out = {}

    def outcome(build):
        try:
            build()
        except sm.SparseMappingLimitError as exc:
            return "refused: " + str(exc)[:160]
        except MemoryError:
            return "MemoryError"
        return "built"

    # Four hundred thousand sources nearest to one of fifty thousand
    # targets: the padded gather would be 5e4 x 4e5 slots, 1.6e11 bytes.
    source = np.linspace(0.0, 1e-3, 400_000)
    target = np.concatenate([[0.0], np.linspace(10.0, 20.0, 49_999)])
    out["skewed gather"] = outcome(lambda: sm.sparse_nearest_neighbor_mapping(
        source, target, mode="conservative"))
    scattered = sm.sparse_nearest_neighbor_mapping(source, target, mode="conservative",
                                                   transpose="scatter")
    out["skewed scatter"] = [int(s) for s in scattered.indices.shape]

    # Thirty thousand targets at the centre of a circle of thirty thousand
    # sources: 9e8 candidates, each a Python integer in a list.
    angle = np.linspace(0.0, 2.0 * np.pi, 30_000, endpoint=False)
    circle = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    out["degenerate ties"] = outcome(lambda: sm.sparse_nearest_neighbor_mapping(
        circle, np.zeros((30_000, 2))))

    # One target cell over five million source cells, among fifty thousand.
    fine = np.linspace(0.0, 1.0, 5_000_001)
    coarse = np.concatenate([[0.0], np.linspace(1.0, 2.0, 50_000)])
    out["wide projection row"] = outcome(lambda: sm.sparse_projection_1d_mapping(fine, coarse))

    # The largest n_source there is sizes nothing.
    hostile = sm.sparse_matrix_mapping(np.zeros((4, 2), np.int64), np.ones((4, 2), np.float32),
                                       n_source=2 ** 31 - 1)
    out["hostile n_source"] = hostile.n_source
    print("RESULT" + json.dumps(out))
'''


def test_the_allocation_bounds_hold_in_a_process_with_a_capped_address_space():
    """Each bound is checked before the allocation it bounds: in a process
    capped at 6 GiB of address space, a pattern whose padded gather would
    need 160 GB, a tie set of 9e8 candidates and a projection row of five
    million slots across fifty thousand rows are each a refusal, not a
    ``MemoryError``."""
    report = _capped(_REFUSALS_UNDER_A_CAP, address_space=6 * _GIB)
    assert report["skewed gather"].startswith("refused: "), report
    assert "50000 rows of 400000 slots" in report["skewed gather"]
    assert report["skewed scatter"] == [400_000, 1]
    assert report["degenerate ties"].startswith("refused: "), report
    assert "degenerate" in report["degenerate ties"]
    assert report["wide projection row"].startswith("refused: "), report
    assert report["hostile n_source"] == 2 ** 31 - 1


_SCALE = '''
    import json, sys, time
    import numpy as np
    n = int(sys.argv[1])
    import jax, jax.numpy as jnp
    from maddening.core.coupling import sparse_mapping as sm

    out = {"n": n, "seconds": {}}
    rng = np.random.default_rng(0)

    def timed(name, build):
        start = time.perf_counter()
        result = build()
        out["seconds"][name] = round(time.perf_counter() - start, 3)
        return result

    def valid(m):
        rows, k = m.indices.shape
        return np.ones((rows, k), bool) if m.counts is None else (
            np.arange(k)[None, :] < m.counts[:, None])

    # --- nearest neighbour, n points onto n points in three dimensions
    source, target = rng.uniform(size=(n, 3)), rng.uniform(size=(n, 3))
    consistent = timed("nearest consistent", lambda: sm.sparse_nearest_neighbor_mapping(
        source, target))
    gathered = timed("nearest conservative gather", lambda: sm.sparse_nearest_neighbor_mapping(
        source, target, mode="conservative"))
    scattered = timed("nearest conservative scatter",
                      lambda: sm.sparse_nearest_neighbor_mapping(
                          source, target, mode="conservative", transpose="scatter"))
    sample = rng.choice(n, size=200, replace=False)
    brute = np.array([int(np.argmin(np.sum((target[i] - source) ** 2, axis=-1))) for i in sample])
    out["consistent sample agrees"] = bool((consistent.indices[sample, 0] == brute).all())
    brute_t = np.array([int(np.argmin(np.sum((source[j] - target) ** 2, axis=-1)))
                        for j in sample])
    out["scatter sample agrees"] = bool((scattered.indices[sample, 0] == brute_t).all())
    # the gather form holds the same entries: target t lists exactly its sources
    rows = np.repeat(np.arange(n), gathered.indices.shape[1]).reshape(gathered.indices.shape)
    mask = valid(gathered)
    back = np.empty(n, np.int64)
    back[gathered.indices[mask]] = rows[mask]
    out["gather is the scatter transposed"] = bool(
        mask.sum() == n and (back == scattered.indices[:, 0]).all())
    out["gather k"] = int(gathered.indices.shape[1])

    field = jnp.asarray(rng.normal(size=n), jnp.float32)
    for name, m in (("consistent", consistent), ("gather", gathered), ("scatter", scattered)):
        apply = jax.jit(m.apply)
        compiled = timed(f"compile and apply {name}", lambda: np.asarray(apply(field)))
        again = timed(f"apply {name}", lambda: np.asarray(apply(field)))
        out[f"{name} is one result"] = bool(compiled.tobytes() == again.tobytes())
    np_field = np.asarray(field)
    out["consistent applies the index"] = bool(
        (np.asarray(jax.jit(consistent.apply)(field)) == np_field[consistent.indices[:, 0]]).all())
    expected = np.zeros(n, np.float64)
    np.add.at(expected, scattered.indices[:, 0], np_field.astype(np.float64))
    for name, m in (("gather", gathered), ("scatter", scattered)):
        got = np.asarray(jax.jit(m.apply)(field), np.float64)
        out[f"{name} is the transpose"] = bool(np.allclose(got, expected, rtol=1e-4, atol=1e-4))

    # --- projection, n cells onto 0.7 n cells
    fine = np.sort(rng.uniform(size=n + 1))
    coarse = np.sort(rng.uniform(size=(7 * n) // 10 + 1))
    projection = timed("projection", lambda: sm.sparse_projection_1d_mapping(fine, coarse))
    w = np.asarray(projection.weights, np.float64)
    inside = (coarse[:-1] >= fine[0]) & (coarse[1:] <= fine[-1])
    out["projection rows sum to one"] = bool(np.allclose(w.sum(axis=1)[inside], 1.0, atol=1e-4))
    i = int(sample[0] % (coarse.size - 1))
    count = int(projection.counts[i]) if projection.counts is not None else projection.k
    j = projection.indices[i, :count].astype(np.int64)
    dense_row = (np.minimum(coarse[i + 1], fine[j + 1]) - np.maximum(coarse[i], fine[j])) / (
        coarse[i + 1] - coarse[i])
    out["projection row is the dense expression"] = bool(
        (np.asarray(projection.weights)[i, :count] == dense_row.astype(np.float32)).all())
    out["projection k"] = int(projection.k)

    # --- your own rows, n by 8, applied and differentiated
    indices = rng.integers(0, n, size=(n, 8))
    values = rng.normal(size=(n, 8)).astype(np.float32)
    matrix = timed("matrix", lambda: sm.sparse_matrix_mapping(indices, values, n_source=n))
    apply = jax.jit(matrix.apply)
    got = timed("compile and apply matrix", lambda: np.asarray(apply(field)))
    out["seconds"]["apply matrix"] = round(min(
        (lambda t: (np.asarray(apply(field)), time.perf_counter() - t)[1])(time.perf_counter())
        for _ in range(3)), 4)
    rows8 = sample[:50]
    want = np.sum(values[rows8].astype(np.float64) * np_field[indices[rows8]], axis=1)
    out["matrix sample agrees"] = bool(np.allclose(got[rows8], want, rtol=1e-4, atol=1e-4))
    grad = timed("compile and differentiate matrix", lambda: np.asarray(jax.jit(jax.grad(
        lambda w: jnp.sum(matrix.apply(field, {"W": w}) ** 2)))(matrix.weights)))
    out["gradient is finite"] = bool(np.isfinite(grad).all() and np.abs(grad).max() > 0)
    out["digest"] = len(timed("digest", matrix.structure_digest))

    # This process image's own high-water mark.  ru_maxrss is not that:
    # after a fork and an exec it still holds the parent's peak, so it
    # reports the test session's memory, not this script's.
    with open("/proc/self/status") as status:
        high_water = [line for line in status if line.startswith("VmHWM:")]
    out["peak_rss_mib"] = int(high_water[0].split()[1]) // 1024
    print("RESULT" + json.dumps(out))
'''

_SCALE_CHECKS = ("consistent sample agrees", "scatter sample agrees",
                 "gather is the scatter transposed", "consistent is one result",
                 "gather is one result", "scatter is one result",
                 "consistent applies the index", "gather is the transpose",
                 "scatter is the transpose", "projection rows sum to one",
                 "projection row is the dense expression", "matrix sample agrees",
                 "gradient is finite")


def _scale(n: int) -> dict:
    report = _capped(_SCALE, str(n), address_space=8 * _GIB)
    failed = [name for name in _SCALE_CHECKS if report[name] is not True]
    assert not failed, (failed, report)
    assert report["digest"] == 64 and report["gather k"] >= 2 and report["projection k"] >= 2
    print(f"\nsparse mappings at n = {n}: {json.dumps(report)}")
    return report


def test_the_three_builders_at_a_hundred_thousand_points_in_a_capped_process():
    """A hundred thousand points each side, built, applied under ``jit``,
    differentiated and checked against brute force on a sample, in a
    process capped at 8 GiB of address space.  A dense mapping of this
    size would be 40 GB."""
    report = _scale(100_000)
    assert report["peak_rss_mib"] < 2048, report


# Per push: tests/core/test_sparse_mapping_builders.py::test_the_three_builders_at_a_hundred_thousand_points_in_a_capped_process
# (the same script at a tenth of the size)
@pytest.mark.slow
def test_the_three_builders_at_a_million_points_in_a_capped_process():
    """A million points each side: the size the limits are documented at."""
    report = _scale(1_000_000)
    assert report["peak_rss_mib"] < 6144, report
