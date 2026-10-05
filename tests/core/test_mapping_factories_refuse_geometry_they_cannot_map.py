"""A mapping factory refuses geometry it cannot map, and maps the rest as before.

Every factory of ``maddening.core.coupling.mapping`` and every closure
factory of ``maddening.core.coupling.interface_mapping`` builds its
operator once from coordinates.  Each used to return *an* operator for
coordinates its formula does not cover (MADD-ANO-174):

* the 1-D projection assumes increasing cell boundaries: a descending
  array gave a matrix of zeros, a non-monotone one rows summing to 4/3;
* nearest neighbour takes ``argmin`` of the distances, and ``argmin``
  returns a NaN: every target read the one point without a position;
* an ``(n, 1)`` point set against an ``(m, 2)`` one was broadcast;
* an RBF system with a NaN point solved to a matrix of NaN, and an
  infinite target to a row of zeros.

The battery below gives every such input to every factory that takes it.
Each must raise a ``ValueError`` naming the argument and the first
offending index; a mapping rebuilt from a config or a USD stage must
raise ``MappingRebuildError``.  Accepted input must give the operator it
gave before the checks existed, bit for bit: ``_BASE_DIGESTS`` holds the
operators of ``release/0.4.0`` at ``a7e69509``.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import interface_mapping as closures
from maddening.core.coupling.mapping import (
    matrix_mapping,
    nearest_neighbor_mapping,
    projection_1d_mapping,
    rbf_mapping,
    rbf_matrix,
)
from maddening.core.coupling.mapping_spec import MappingRebuildError, point_array_digest
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

F32 = jnp.float32

# ---------------------------------------------------------------------------
# Cell boundaries: the projection, as a mapping and as a closure
# ---------------------------------------------------------------------------

GOOD_SOURCE = [0.0, 1.0, 2.0, 3.0]
GOOD_TARGET = [0.0, 1.5, 3.0]

BOUNDARY_FACTORIES = {
    "projection_1d_mapping": projection_1d_mapping,
    "conservative_projection_1d": closures.conservative_projection_1d,
}

#: label -> (boundaries, what the message says, the index it names or None)
BAD_BOUNDARIES = {
    "descending": ([3.0, 2.0, 1.0, 0.0], "strictly increasing", 1),
    "non-monotone": ([0.0, 2.0, 1.0, 3.0], "strictly increasing", 2),
    "a repeated boundary": ([0.0, 1.0, 1.0, 3.0], "strictly increasing", 2),
    "NaN": ([0.0, float("nan"), 2.0, 3.0], "non-finite", 1),
    "+inf at the end": ([0.0, 1.0, 2.0, float("inf")], "non-finite", 3),
    "-inf at the start": ([float("-inf"), 1.0, 2.0, 3.0], "non-finite", 0),
    "one value": ([0.0], "at least two", None),
    "empty": ([], "at least two", None),
    "two-dimensional": ([[0.0, 1.0], [2.0, 3.0]], "one-dimensional", None),
    "a column": ([[0.0], [1.0], [3.0]], "one-dimensional", None),
    "a scalar": (1.0, "one-dimensional", None),
}


@pytest.mark.parametrize("factory", sorted(BOUNDARY_FACTORIES))
@pytest.mark.parametrize("argument", ["source_boundaries", "target_boundaries"])
@pytest.mark.parametrize("case", sorted(BAD_BOUNDARIES))
def test_boundaries_that_are_not_strictly_increasing_are_refused_by_name(
        factory, argument, case):
    bad, says, index = BAD_BOUNDARIES[case]
    args = (bad, GOOD_TARGET) if argument == "source_boundaries" else (GOOD_SOURCE, bad)
    with pytest.raises(ValueError, match=says) as caught:
        BOUNDARY_FACTORIES[factory](*args)
    message = str(caught.value)
    assert message.startswith(argument), message
    if index is not None:
        assert (f"{argument}[{index}]" in message) or (f"index {index} " in message), message


@pytest.mark.parametrize("source,target", [
    ([0.0, 1.0, 2.0], [10.0, 11.0, 12.0]),      # the target lies beyond the source
    ([10.0, 11.0, 12.0], [0.0, 1.0, 2.0]),      # and before it
    ([0.0, 1.0], [1.0, 2.0]),                   # they touch at one point
])
def test_grids_that_share_no_interval_give_a_zero_matrix(source, target):
    """Not refused: it is what the formula says of cells that do not meet,
    the limit of grids that overlap in part, and the docstring says so."""
    P = np.asarray(projection_1d_mapping(source, target).H)
    np.testing.assert_array_equal(P, np.zeros((len(target) - 1, len(source) - 1)))
    values = jnp.ones(len(source) - 1, F32)
    np.testing.assert_array_equal(
        np.asarray(closures.conservative_projection_1d(source, target)(values)), 0.0)


def test_the_boundaries_are_not_sorted_or_reversed_for_the_caller():
    """Reversing the boundaries without reversing the field would map the
    wrong cells; the refusal says so instead."""
    with pytest.raises(ValueError, match="not sorted or reversed"):
        projection_1d_mapping(GOOD_SOURCE[::-1], GOOD_TARGET)


def _weights(sb, tb):
    return np.asarray(projection_1d_mapping(sb, tb).H, np.float64)


def test_the_projection_preserves_the_integral_when_the_target_covers_the_source():
    sb, tb = np.array([0.0, 0.3, 1.0, 1.25]), np.array([-1.0, 0.5, 1.0, 4.0])
    P = _weights(sb, tb)
    # sum_i |target_i| P[i, j] = |source_j| for every source cell.
    np.testing.assert_allclose(np.diff(tb) @ P, np.diff(sb), rtol=1e-6)
    # ... and a target cell partly outside the source averages in zeros,
    # where the one inside it ([0.5, 1.0]) reproduces a constant.
    np.testing.assert_allclose(P.sum(axis=1), [1.0 / 3.0, 1.0, 0.25 / 3.0], rtol=1e-6)


def test_the_projection_reproduces_a_constant_on_target_cells_the_source_covers():
    sb, tb = np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 1.5, 2.5])
    P = _weights(sb, tb)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, rtol=1e-6)
    # The source outside the target is dropped: the integral is not kept.
    assert (np.diff(tb) @ P).sum() == pytest.approx(1.5) and np.diff(sb).sum() == 3.0


def test_a_target_cell_outside_the_source_is_zero_and_the_overlap_is_kept():
    """Grids that overlap in part: ``sum_i |target_i| P[i, j] = |source_j ∩ target|``."""
    sb, tb = np.array([0.0, 1.0, 2.0, 3.0]), np.array([2.5, 3.0, 4.0, 5.0])
    P = _weights(sb, tb)
    np.testing.assert_allclose(np.diff(tb) @ P, [0.0, 0.0, 0.5], atol=1e-7)
    np.testing.assert_array_equal(P[1:], 0.0)


# ---------------------------------------------------------------------------
# Point sets: RBF and nearest neighbour, as mappings and as closures
# ---------------------------------------------------------------------------

SRC2 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.2]])
TGT2 = np.array([[0.5, 0.5], [0.25, 0.75], [0.9, 0.1], [0.1, 0.3]])

POINT_FACTORIES = {
    "rbf_mapping[consistent]": lambda s, t: rbf_mapping(s, t),
    "rbf_mapping[conservative]": lambda s, t: rbf_mapping(s, t, mode="conservative"),
    "rbf_mapping[no polynomial]": lambda s, t: rbf_mapping(s, t, polynomial=False),
    "rbf_matrix": lambda s, t: rbf_matrix(s, t),
    "nearest_neighbor_mapping[consistent]": lambda s, t: nearest_neighbor_mapping(s, t),
    "nearest_neighbor_mapping[conservative]":
        lambda s, t: nearest_neighbor_mapping(s, t, mode="conservative"),
    "rbf_interpolation": lambda s, t: closures.rbf_interpolation(s, t),
    "rbf_interpolation_2d": lambda s, t: closures.rbf_interpolation_2d(s, t),
    "nearest_neighbor_2d": lambda s, t: closures.nearest_neighbor_2d(s, t),
}


def _with(points, row, col, value):
    out = np.array(points, np.float64)
    out[row, col] = value
    return out


@pytest.mark.parametrize("factory", sorted(POINT_FACTORIES))
@pytest.mark.parametrize("argument", ["source_points", "target_points"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_point_is_refused_by_name_and_index(factory, argument, value):
    if argument == "source_points":
        args = (_with(SRC2, 3, 1, value), TGT2)
    else:
        args = (SRC2, _with(TGT2, 3, 1, value))
    with pytest.raises(ValueError, match="non-finite coordinate at index 3 ") as caught:
        POINT_FACTORIES[factory](*args)
    assert str(caught.value).startswith(argument)


@pytest.mark.parametrize("factory", sorted(POINT_FACTORIES))
@pytest.mark.parametrize("target", [
    np.zeros((4, 3)),            # more columns: used to fail in the broadcast, unnamed
    np.linspace(0.0, 1.0, 4).reshape(-1, 1),     # one column: used to broadcast
    np.linspace(0.0, 1.0, 4),                    # (n,): read as one column
], ids=["2 against 3", "2 against (n, 1)", "2 against (n,)"])
def test_point_sets_of_different_dimension_are_refused(factory, target):
    with pytest.raises(ValueError, match="must share a dimension, got 2 and "):
        POINT_FACTORIES[factory](SRC2, target)


@pytest.mark.parametrize("factory", sorted(POINT_FACTORIES))
@pytest.mark.parametrize("argument", ["source_points", "target_points"])
def test_a_point_array_with_more_than_two_axes_is_refused(factory, argument):
    bad = np.zeros((2, 2, 2))
    args = (bad, TGT2) if argument == "source_points" else (SRC2, bad)
    with pytest.raises(ValueError, match=r"\(n,\) or \(n, d\) array") as caught:
        POINT_FACTORIES[factory](*args)
    assert str(caught.value).startswith(argument)


@pytest.mark.parametrize("factory", sorted(POINT_FACTORIES))
@pytest.mark.parametrize("bad,says", [(np.zeros((3, 0)), "no coordinates"),
                                      (np.float64(1.0), r"\(n,\) or \(n, d\) array")],
                         ids=["no columns", "a scalar"])
def test_a_point_set_without_coordinates_is_refused(factory, bad, says):
    """Zero columns put every point at distance zero from every other."""
    with pytest.raises(ValueError, match=says) as caught:
        POINT_FACTORIES[factory](bad, TGT2)
    assert str(caught.value).startswith("source_points")


@pytest.mark.parametrize("factory,argument", [
    ("rbf_mapping[consistent]", "source_points"),
    ("rbf_mapping[no polynomial]", "source_points"),
    ("rbf_mapping[conservative]", "target_points"),
    ("rbf_matrix", "source_points"),
    ("nearest_neighbor_mapping[consistent]", "source_points"),
    ("nearest_neighbor_mapping[conservative]", "target_points"),
    ("rbf_interpolation", "source_points"),
    ("rbf_interpolation_2d", "source_points"),
])
def test_an_empty_set_to_map_from_is_refused_by_name(factory, argument):
    """The set the operator interpolates from: the source in consistent
    mode, the target in conservative mode (the transpose of the reverse)."""
    empty = np.zeros((0, 2))
    args = (empty, TGT2) if argument == "source_points" else (SRC2, empty)
    with pytest.raises(ValueError, match="holds no points") as caught:
        POINT_FACTORIES[factory](*args)
    assert str(caught.value).startswith(argument)


@pytest.mark.parametrize("build,shape", [
    (lambda e: nearest_neighbor_mapping(SRC2, e), (0, 5)),
    (lambda e: nearest_neighbor_mapping(e, TGT2, mode="conservative"), (4, 0)),
    (lambda e: rbf_mapping(SRC2, e), (0, 5)),
    (lambda e: rbf_mapping(e, TGT2, mode="conservative", polynomial=False), (4, 0)),
])
def test_an_empty_set_to_evaluate_at_gives_an_empty_operator(build, shape):
    """Nothing to refuse: the operator has no rows (or no columns), as before."""
    assert build(np.zeros((0, 2))).H.shape == shape


@pytest.mark.parametrize("factory", ["nearest_neighbor_1d", "linear_interpolation_1d"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_the_1d_closures_refuse_a_non_finite_source_coordinate(factory, value):
    with pytest.raises(ValueError, match="source_x holds a non-finite value at index 1 "):
        getattr(closures, factory)([0.0, value, 2.0, 3.0], [0.5, 2.5])


def test_nearest_neighbor_1d_refuses_a_non_finite_target_coordinate():
    """Its distances are all NaN, and ``argmin`` of them is source 0."""
    with pytest.raises(ValueError, match="target_x holds a non-finite value at index 0 "):
        closures.nearest_neighbor_1d([0.0, 1.0, 2.0], [float("nan"), 1.6])


@pytest.mark.parametrize("source,index", [([3.0, 2.0, 1.0, 0.0], 1), ([0.0, 2.0, 1.0, 3.0], 2)])
def test_linear_interpolation_refuses_source_coordinates_out_of_order(source, index):
    """``searchsorted`` assumes ascending coordinates; on a descending grid
    every target read an end value (x -> [3, 0] where it is [0.5, 2.5])."""
    with pytest.raises(ValueError, match="ascending") as caught:
        closures.linear_interpolation_1d(source, [0.5, 2.5])
    assert f"source_x[{index}]" in str(caught.value)


def test_linear_interpolation_keeps_a_repeated_coordinate_and_clamps_outside():
    """Neither was wrong, so neither is refused: a zero-width interval is
    skipped, and a target outside the grid reads the end value."""
    interp = closures.linear_interpolation_1d(jnp.array([0.0, 1.0, 1.0, 3.0]),
                                              jnp.array([0.5, 2.0, -5.0, float("inf")]))
    out = np.asarray(interp(jnp.array([0.0, 1.0, 1.0, 3.0], F32)))
    np.testing.assert_allclose(out, [0.5, 2.0, 0.0, 3.0], rtol=1e-6)


def test_nearest_neighbour_takes_the_lowest_index_among_equidistant_points():
    """Coincident and equidistant sources are a defined case, not a refusal."""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    tgt = np.array([[1.0, 0.0], [0.5, 0.5]])            # on the pair; equidistant from all four
    H = np.asarray(nearest_neighbor_mapping(src, tgt).H)
    np.testing.assert_array_equal(H, [[0, 1, 0, 0], [1, 0, 0, 0]])
    # Conservative mode: each source adds to its nearest target (the lower on a tie).
    Hc = np.asarray(nearest_neighbor_mapping(tgt, src, mode="conservative").H)
    np.testing.assert_array_equal(Hc, H.T)


def test_rbf_shares_the_weight_between_coincident_sources_and_keeps_the_patch_test():
    """With the default ridge a repeated source point is regularised: the
    two columns are equal and constants and linear fields still cross."""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    H = np.asarray(rbf_mapping(src, TGT2).H, np.float64)
    assert np.all(np.isfinite(H))
    np.testing.assert_allclose(H[:, 1], H[:, 2], atol=1e-5)
    np.testing.assert_allclose(H @ np.ones(5), 1.0, atol=1e-5)
    linear = lambda p: 0.5 + p @ np.array([0.3, -1.2])                 # noqa: E731
    np.testing.assert_allclose(H @ linear(src), linear(TGT2), atol=1e-4)


@pytest.mark.parametrize("polynomial", [True, False])
def test_rbf_on_coincident_sources_without_a_ridge_is_a_value_error(polynomial):
    """The system is exactly singular and the solve says so."""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="Singular"):
        rbf_mapping(src, TGT2, ridge=0.0, polynomial=polynomial)


# ---------------------------------------------------------------------------
# A given matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.float16])
def test_a_non_finite_matrix_is_refused_by_index(value, dtype):
    H = np.eye(3, 4, dtype=dtype)
    H[2, 1] = value
    with pytest.raises(ValueError, match=r"H holds a non-finite value at index \(2, 1\)"):
        matrix_mapping(H)
    with pytest.raises(ValueError, match="non-finite"):
        matrix_mapping(jnp.asarray(H))
    with pytest.raises(ValueError, match="non-finite"):
        matrix_mapping(H.tolist(), kind="supermesh")


def test_a_matrix_built_inside_a_transform_is_taken_as_given():
    """A traced ``H`` has no values to read; the mapping still differentiates."""
    v = jnp.array([1.0, 2.0, 3.0], F32)

    def loss(H):
        return jnp.sum(matrix_mapping(H).apply(v) ** 2)

    g = jax.grad(loss)(jnp.ones((2, 3), F32))
    np.testing.assert_allclose(np.asarray(g), 2 * 6.0 * np.tile(np.asarray(v), (2, 1)))


def test_a_complex_point_set_is_still_refused_by_the_reference_check():
    """Unchanged: the dtype is refused by name where the reference is
    recorded, whatever its real parts would have looked like here."""
    from maddening.core.coupling.mapping_spec import PointReferenceError  # noqa: PLC0415

    with pytest.warns(np.exceptions.ComplexWarning):
        with pytest.raises(PointReferenceError, match="source_boundaries: dtype 'complex128'"):
            projection_1d_mapping(np.array([0.0, 1.0j, 2.0]), GOOD_TARGET)
    with pytest.warns(np.exceptions.ComplexWarning):
        with pytest.raises(PointReferenceError, match="source_points: dtype 'complex128'"):
            nearest_neighbor_mapping(np.array([0.0, 1.0j, 2.0]), [0.5, 1.5])


def test_integer_and_empty_matrices_are_still_accepted():
    assert matrix_mapping(np.eye(2, dtype=np.int32)).H.dtype == jnp.int32
    assert matrix_mapping(np.zeros((0, 3), np.float32)).H.shape == (0, 3)


def test_closure_coordinates_passed_as_tracers_are_taken_as_given():
    """A closure factory called inside ``jit`` cannot read its coordinates
    on the host; it builds the transform it built before."""
    values = jnp.array([10.0, 20.0, 30.0], F32)

    @jax.jit
    def nearest(xs):
        return closures.nearest_neighbor_1d(xs, jnp.array([0.1, 1.9]))(values)

    np.testing.assert_array_equal(np.asarray(nearest(jnp.array([0.0, 1.0, 2.0]))), [10.0, 30.0])

    @jax.jit
    def projected(sb):
        return closures.conservative_projection_1d(sb, jnp.array([0.0, 1.5, 3.0]))(values)

    np.testing.assert_allclose(np.asarray(projected(jnp.array([0.0, 1.0, 2.0, 3.0]))),
                               [(10 + 0.5 * 20) / 1.5, (0.5 * 20 + 30) / 1.5], rtol=1e-6)


# ---------------------------------------------------------------------------
# Accepted input: the operators of the base tree, bit for bit
# ---------------------------------------------------------------------------


def _f32(values) -> jax.Array:
    """A float32 JAX array of host-computed values (``jnp.linspace`` is the
    backend's own arithmetic, and need not agree between jaxlib builds)."""
    return jnp.asarray(np.asarray(values, np.float32))


def _operators() -> dict:
    """Operators whose arithmetic is single IEEE operations or a selection,
    so their bits do not depend on the machine or the jaxlib.  A closure is
    read through the identity: ``transform(I)`` is its matrix, each entry
    one product with 1.0 plus zeros."""
    lattice = np.array([[x, y] for x in (0.0, 1.0, 2.0) for y in (0.0, 1.0)])
    centres = np.array([[0.5, 0.5], [1.5, 0.5], [1.0, 0.0], [2.0, 1.0], [0.4, 0.9]])
    out = {
        "projection coarsen": projection_1d_mapping(np.linspace(0, 1, 11), np.linspace(0, 1, 6)).H,
        "projection refine": projection_1d_mapping(np.linspace(0, 1, 6), np.linspace(0, 1, 11)).H,
        "projection non-uniform": projection_1d_mapping(
            [0.0, 0.1, 0.35, 0.4, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0]).H,
        "projection target inside": projection_1d_mapping(GOOD_SOURCE, [1.0, 1.5, 2.0]).H,
        "projection target wider": projection_1d_mapping(GOOD_SOURCE, [-1.0, 1.5, 4.0]).H,
        "projection partial overlap": projection_1d_mapping(GOOD_SOURCE, [2.0, 3.0, 4.0, 5.0]).H,
        "projection no overlap": projection_1d_mapping(GOOD_SOURCE, [3.0, 4.0, 6.0]).H,
        "projection float32 boundaries": projection_1d_mapping(
            np.linspace(0, 1, 8, dtype=np.float32), np.linspace(0, 1, 4, dtype=np.float32)).H,
        "projection integer boundaries": projection_1d_mapping(np.arange(5), np.array([0, 3, 4])).H,
        "projection jax boundaries": projection_1d_mapping(
            _f32(np.linspace(0, 1, 11)), _f32(np.linspace(0, 1, 6))).H,
        "projection negative and small": projection_1d_mapping(
            [-2e-3, -1e-3, 0.0, 5e-4], [-2e-3, 0.0, 5e-4]).H,
        "nearest 1d": nearest_neighbor_mapping([0.0, 1.0, 2.0], [0.1, 1.9, 0.9]).H,
        "nearest 1d conservative": nearest_neighbor_mapping(
            [0.0, 1.0, 2.0], [0.1, 1.9, 0.9, 1.1], mode="conservative").H,
        "nearest 2d with ties": nearest_neighbor_mapping(lattice, centres).H,
        "nearest 2d conservative": nearest_neighbor_mapping(
            lattice, centres, mode="conservative").H,
        "nearest coincident sources": nearest_neighbor_mapping(
            np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]), centres).H,
        "matrix float32": matrix_mapping(np.arange(12, dtype=np.float32).reshape(3, 4) / 7).H,
        "matrix int32": matrix_mapping(np.eye(3, 2, dtype=np.int32)).H,
        "matrix float16 labelled": matrix_mapping(
            np.linspace(0, 1, 6, dtype=np.float16).reshape(2, 3), kind="supermesh").H,
        "closure projection": closures.conservative_projection_1d(
            _f32(np.linspace(0, 1, 11)), _f32(np.linspace(0, 1, 6)))(jnp.eye(10, dtype=F32)),
        "closure projection partial overlap": closures.conservative_projection_1d(
            _f32(GOOD_SOURCE), _f32([2.0, 3.0, 4.0, 5.0]))(jnp.eye(3, dtype=F32)),
        "closure nearest 1d": closures.nearest_neighbor_1d(
            _f32(np.linspace(0, 1, 10)), _f32([0.04, 0.5, 0.96]))(jnp.eye(10, dtype=F32)),
        "closure nearest 2d": closures.nearest_neighbor_2d(
            _f32(lattice), _f32(centres))(jnp.eye(6, dtype=F32)),
        "closure linear": closures.linear_interpolation_1d(
            _f32([0.0, 1.0, 2.0, 4.0]), _f32([0.5, 1.0, 3.0, 9.0]))(jnp.eye(4, dtype=F32)),
    }
    return {k: np.asarray(a) for k, a in out.items()}


#: ``point_array_digest`` (dtype, shape and bytes) of each operator of
#: ``_operators()`` on ``release/0.4.0`` at ``a7e69509``, before any check
#: existed (jax 0.11.0, numpy 2.4.6, CPU).
_BASE_DIGESTS: dict = {
    "closure linear":
        "3de70562d5c0410667c93200cb2ac2cfdcb80e581dc1cb95dfe08507c8d08adc",
    "closure nearest 1d":
        "5b6a459193708c0d9b2498d735aef27996f8e90a52cc0a29aaf7afd71ca278c6",
    "closure nearest 2d":
        "d2fa93e90eee0d02252224057828c9fd256d8d9ac945d45b0502c6dea7b8cc21",
    "closure projection":
        "f60e912d8badb519aab6af660ab3f31a3fe42413c1c220be7f395d71b8279a64",
    "closure projection partial overlap":
        "fdf8ac4dbc1a4a3d6ae9a20fa5201b7d878aa0458623025a345076cfdbef6a9b",
    "matrix float16 labelled":
        "a322d8fa2df426235b8b126dcadc7a248ab0b6a493d0854fc9b6dd4bb5c53725",
    "matrix float32":
        "a892eb0feb9e01d91e1bc3da67b4b3470d4ccb281ea4eaef752be6bd5a2c05b3",
    "matrix int32":
        "850e459fb5ab13e13861c7c67fd524723cdf222c6d110eeea9df4469e80f12d0",
    "nearest 1d":
        "bea2288f4d3c78d67387c3142ad6cf350e797d425e86f92f209b7cbc76aae62c",
    "nearest 1d conservative":
        "cdb1dd1a3c4959d9e913d68eb0268bbb6f4c0f2d41e6f1e249701f5a2fad583a",
    "nearest 2d conservative":
        "eeff34c9d1f606a4180836a2b9638876d104d651887ab21fa98007a4f25844b6",
    "nearest 2d with ties":
        "d2fa93e90eee0d02252224057828c9fd256d8d9ac945d45b0502c6dea7b8cc21",
    "nearest coincident sources":
        "e20297f4a9278423c255dbb3bf240a24db559d10a1be06a3a7cf942baf4f5cfb",
    "projection coarsen":
        "91af5df86872bfd19d1536c591b0ad25a470269d2152c978748dd3d0f5da8428",
    "projection float32 boundaries":
        "8bcfb2ca2796518781e52f17398490e18b728941713d44e62e5f84fa9eba56f0",
    "projection integer boundaries":
        "b014c24674f68894e84ae7a6598319d822c2bf10d4664a088e84d18215c51db7",
    "projection jax boundaries":
        "f60e912d8badb519aab6af660ab3f31a3fe42413c1c220be7f395d71b8279a64",
    "projection negative and small":
        "53809049d6267a1590a82794256a51133097bf05399a9d22b2251a014be5f86d",
    "projection no overlap":
        "3f4eb10c903e8c984347b3eb750e66cc6cd14a5cc07ff01e674e03fce5adb8b1",
    "projection non-uniform":
        "8b0662ff2e9f0f3569580dd7c8bf6702b9a2ac1f7b5f3b9bf06f60c13cbc4d95",
    "projection partial overlap":
        "fdf8ac4dbc1a4a3d6ae9a20fa5201b7d878aa0458623025a345076cfdbef6a9b",
    "projection refine":
        "e27b0b36cf703ffd708b6998e57555f7d0fbe71903f3f07b11cbcd867b5f50ae",
    "projection target inside":
        "577cc9fbedc44dada4076a2e30fbd06fdd4974a66bd889a2595799a482f15c64",
    "projection target wider":
        "4fa5c380ca53bee137d004ac878c18904641598bb8c6363ec5dd1781e2fc35fb",
}


@pytest.fixture(scope="module")
def operators():
    return _operators()


def test_the_case_table_and_the_captured_digests_name_the_same_operators(operators):
    assert sorted(operators) == sorted(_BASE_DIGESTS)


@pytest.mark.parametrize("name", sorted(_BASE_DIGESTS))
def test_an_accepted_input_gives_the_operator_it_gave_before_the_checks(operators, name):
    assert point_array_digest(operators[name]) == _BASE_DIGESTS[name]


def _rbf_matrix_before_the_checks(source_points, target_points, *, kernel, epsilon,
                                  polynomial, ridge):
    """``rbf_matrix`` of the base tree, transcribed: the same NumPy calls in
    the same order, so on one machine the same bits.  (An RBF operator
    goes through a LAPACK solve, whose last bit differs between builds;
    it cannot be pinned by a digest taken elsewhere.)"""
    def as_points(x):
        x = np.asarray(x, dtype=np.float64)
        return x.reshape(-1, 1) if x.ndim == 1 else x

    def pairwise(a, b):
        return np.sqrt(np.maximum(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1), 0.0))

    def phi(r):
        if kernel == "gaussian":
            return np.exp(-(epsilon * r) ** 2)
        if kernel == "multiquadric":
            return np.sqrt(1.0 + (epsilon * r) ** 2)
        if kernel == "inverse_multiquadric":
            return 1.0 / np.sqrt(1.0 + (epsilon * r) ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(r > 1e-300, r ** 2 * np.log(np.where(r > 1e-300, r, 1.0)), 0.0)

    src, tgt = as_points(source_points), as_points(target_points)
    n, d = src.shape
    phi_ss, phi_ts = phi(pairwise(src, src)), phi(pairwise(tgt, src))
    phi_ss = phi_ss + (ridge * max(float(np.max(np.abs(phi_ss))), 1e-300)) * np.eye(n)
    if not polynomial:
        return jnp.asarray(np.linalg.solve(phi_ss.T, phi_ts.T).T, dtype=jnp.float32)
    q = 1 + d
    p_s = np.concatenate([np.ones((n, 1)), src], axis=1)
    p_t = np.concatenate([np.ones((tgt.shape[0], 1)), tgt], axis=1)
    a = np.block([[phi_ss, p_s], [p_s.T, np.zeros((q, q))]])
    b = np.concatenate([np.eye(n), np.zeros((q, n))])
    return jnp.asarray(np.concatenate([phi_ts, p_t], axis=1) @ np.linalg.solve(a, b),
                       dtype=jnp.float32)


@pytest.mark.parametrize("kernel", ["gaussian", "multiquadric", "inverse_multiquadric",
                                    "thin_plate_spline"])
@pytest.mark.parametrize("polynomial", [True, False])
@pytest.mark.parametrize("mode", ["consistent", "conservative"])
def test_an_rbf_operator_is_the_one_the_unchecked_construction_gives(kernel, polynomial, mode):
    kw = dict(kernel=kernel, epsilon=1.7, polynomial=polynomial, ridge=1e-8)
    got = np.asarray(rbf_mapping(SRC2, TGT2, mode=mode, **kw).H)
    if mode == "consistent":
        want = _rbf_matrix_before_the_checks(SRC2, TGT2, **kw)
    else:
        want = _rbf_matrix_before_the_checks(TGT2, SRC2, **kw).T
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, np.asarray(want))
    # ... and the closure shares it.
    if mode == "consistent":
        v = jnp.arange(1.0, 6.0, dtype=F32)
        np.testing.assert_array_equal(
            np.asarray(closures.rbf_interpolation(SRC2, TGT2, **kw)(v)), np.asarray(want @ v))


# ---------------------------------------------------------------------------
# Rebuilt from a config: the refusal is a MappingRebuildError naming the edge
# ---------------------------------------------------------------------------


class _Vec(SimulationNode):
    """``n`` values that integrate their boundary input."""

    def __init__(self, name, timestep, n=3):
        super().__init__(name, timestep, n=n)

    def initial_state(self):
        return {"v": jnp.arange(1, self.params["n"] + 1, dtype=F32)}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=(self.params["n"],), description="input")}

    def update(self, state, boundary_inputs, dt):
        return {"v": state["v"] + dt * boundary_inputs.get("inp", jnp.zeros_like(state["v"]))}


REGISTRY = {"_Vec": _Vec}
EDGE = "edge a.v -> b.inp"


def _config(mapping: dict, n_source=3, n_target=2) -> dict:
    gm = GraphManager()
    gm.add_node(_Vec("a", 0.1, n=n_source))
    gm.add_node(_Vec("b", 0.1, n=n_target))
    config = gm.to_dict()
    config["edges"] = [{"source_node": "a", "target_node": "b", "source_field": "v",
                        "target_field": "inp", "mapping": mapping}]
    return config


def _refused(config, tmp_path, says):
    with pytest.raises(MappingRebuildError, match=says) as caught:
        GraphManager.from_dict(config, REGISTRY, base_dir=tmp_path)
    assert str(caught.value).startswith(EDGE)
    assert type(caught.value.__cause__) is ValueError
    return caught.value


def test_a_valid_projection_config_rebuilds(tmp_path):
    """The control for the refusals below: this config loads."""
    config = _config({"kind": "projection_1d", "points": {
        "source_boundaries": GOOD_SOURCE, "target_boundaries": GOOD_TARGET}})
    gm = GraphManager.from_dict(config, REGISTRY, base_dir=tmp_path)
    np.testing.assert_array_equal(np.asarray(gm.edges[0].mapping.H),
                                  np.asarray(projection_1d_mapping(GOOD_SOURCE, GOOD_TARGET).H))


@pytest.mark.parametrize("argument", ["source_boundaries", "target_boundaries"])
@pytest.mark.parametrize("case", ["descending", "non-monotone", "a repeated boundary"])
def test_a_config_with_boundaries_out_of_order_does_not_rebuild(tmp_path, argument, case):
    points = {"source_boundaries": GOOD_SOURCE, "target_boundaries": GOOD_TARGET}
    points[argument] = BAD_BOUNDARIES[case][0]
    if argument == "target_boundaries":
        points[argument] = points[argument][:3]
    error = _refused(_config({"kind": "projection_1d", "points": points}), tmp_path,
                     f"{argument} must be strictly increasing")
    assert error.kind == "projection_1d"


def test_a_config_whose_boundary_asset_holds_a_nan_does_not_rebuild(tmp_path):
    """An inline reference cannot hold a NaN; a file can."""
    np.save(tmp_path / "bounds.npy", np.array([0.0, np.nan, 2.0, 3.0]))
    _refused(_config({"kind": "projection_1d", "points": {
        "source_boundaries": {"asset": "bounds.npy"}, "target_boundaries": GOOD_TARGET}}),
        tmp_path, "source_boundaries holds a non-finite value at index 1 ")


@pytest.mark.parametrize("kind,extra", [
    ("nearest_neighbor", {"mode": "consistent"}),
    ("nearest_neighbor", {"mode": "conservative"}),
    ("rbf", {"mode": "consistent"}),
    ("rbf", {"mode": "conservative", "polynomial": False}),
])
@pytest.mark.parametrize("argument", ["source_points", "target_points"])
def test_a_config_whose_point_asset_holds_a_nan_does_not_rebuild(tmp_path, kind, extra, argument):
    np.save(tmp_path / "source.npy", np.array([0.0, 0.5, 1.0]))
    np.save(tmp_path / "target.npy", np.array([0.25, 0.75]))
    bad = np.array([0.0, np.nan, 1.0]) if argument == "source_points" else np.array([0.25, np.nan])
    np.save(tmp_path / f"{argument.split('_')[0]}.npy", bad)
    _refused(_config({"kind": kind, **extra, "points": {
        "source_points": {"asset": "source.npy"}, "target_points": {"asset": "target.npy"}}}),
        tmp_path, f"{argument} holds a non-finite coordinate at index 1 ")


def test_a_config_whose_point_sets_differ_in_dimension_does_not_rebuild(tmp_path):
    """Three points in the plane against two on a line used to broadcast."""
    _refused(_config({"kind": "nearest_neighbor", "points": {
        "source_points": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        "target_points": [0.25, 0.75]}}), tmp_path, "must share a dimension, got 2 and 1")


def test_a_config_whose_matrix_asset_holds_a_nan_does_not_rebuild(tmp_path):
    H = np.zeros((2, 3), np.float32)
    H[1, 2] = np.nan
    np.save(tmp_path / "H.npy", H)
    _refused(_config({"kind": "matrix", "points": {"H": {"asset": "H.npy"}}}), tmp_path,
             r"H holds a non-finite value at index \(1, 2\)")
