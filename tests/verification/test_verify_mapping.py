"""`verify_mapping`: the battery passes every shipped kind and fails broken ones.

Two halves, because a battery that cannot fail verifies nothing:

* every shipped mapping kind, with the claims the interface-mapping guide
  makes for it, passes every check (:data:`SHIPPED`);
* each seeded broken kind fails the one check that should catch it, and
  no other (:data:`BROKEN`).

The kinds here are small (a dozen entries a side): the properties are
identities of linear algebra, which a small operator shows as well as a
large one.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import contextlib  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from hypothesis.extra.numpy import arrays  # noqa: E402

from maddening.core.coupling import mapping_registry  # noqa: E402
from maddening.core.coupling.grid_mapping import (  # noqa: E402
    MultilinearGridMapping,
    multilinear_grid_mapping,
)
from maddening.core.coupling.mapping import (  # noqa: E402
    StaticLinearMapping,
    matrix_mapping,
    nearest_neighbor_mapping,
    projection_1d_mapping,
    rbf_mapping,
    register_mapping,
)
from maddening.core.coupling.mapping_spec import MappingSpec, reference_for_array  # noqa: E402
from maddening.core.coupling.sparse_mapping import (  # noqa: E402
    StaticSparseMapping,
    sparse_matrix_mapping,
    sparse_nearest_neighbor_mapping,
    sparse_projection_1d_mapping,
)
from maddening.core.edge import EdgeSpec  # noqa: E402
from maddening.testing.mapping import (  # noqa: E402
    DEFAULT_MAPPING_CHECKS,
    assert_mapping_verified,
    verify_mapping,
)

# Twenty draws a check, derandomised: every property here is an identity
# of a fixed linear operator, so a draw differs from the next only in the
# field's values, and the per-push lane runs some thirty batteries of ten
# checks.  The slow test below runs the default depth.
KW = dict(max_examples=20, derandomize=True)

_RNG = np.random.default_rng(3)
XS = np.sort(_RNG.uniform(0.0, 1.0, 9))
XT = np.sort(_RNG.uniform(0.05, 0.95, 7))
SB = np.linspace(0.0, 1.0, 7)
TB = np.linspace(0.0, 1.0, 10)
COORDS = dict(source_coordinates=XS, target_coordinates=XT)
MEASURES = dict(source_measure=np.diff(SB), target_measure=np.diff(TB))

ORIGIN, SPACING, SHAPE = (0.0, -1.0), (0.25, 0.5), (5, 4)
LOWER = np.asarray(ORIGIN)
UPPER = LOWER + np.asarray(SPACING) * (np.asarray(SHAPE) - 1)
N_POINTS = 6


def _lattice(origin, spacing, shape):
    axes = [origin[a] + spacing[a] * np.arange(shape[a]) for a in range(len(shape))]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(shape))


def _positions(lower, upper, n_points=N_POINTS, dtype=np.float32):
    """Positions inside the hull, where the kind's claims hold."""
    lower, upper = np.asarray(lower, np.float64), np.asarray(upper, np.float64)
    unit = arrays(np.float32, (n_points, lower.size),
                  elements=st.floats(0.0, 1.0, width=32))
    return unit.map(lambda u: (lower + u * (upper - lower)).astype(dtype))


def _grid(mode, layout="flat", origin=ORIGIN, spacing=SPACING, shape=SHAPE, dtype=np.float32):
    lower = np.asarray(origin, np.float64)
    upper = lower + np.asarray(spacing) * (np.asarray(shape) - 1)
    mapping = multilinear_grid_mapping(origin, spacing, shape, n_points=N_POINTS, mode=mode,
                                       layout=layout)
    lattice = _lattice(origin, spacing, shape)
    points = dict(source_coordinates=lattice, target_coordinates=lambda g: g)
    if mode == "conservative":
        points = dict(source_coordinates=lambda g: g, target_coordinates=lattice)
    return mapping, dict(geometry_strategy=_positions(lower, upper, dtype=dtype),
                         polynomial_order=1, hull=(lower, upper), outside="clamp", **points)


def _row_stochastic():
    h = np.abs(np.random.default_rng(5).normal(size=(7, 9))).astype(np.float32)
    return h / h.sum(axis=1, keepdims=True)


def _scatter_rows():
    """Each of 9 sources shares its value between two of 7 targets."""
    rng = np.random.default_rng(6)
    indices = np.stack([rng.integers(0, 7, 9), rng.integers(0, 7, 9)], axis=1)
    first = rng.uniform(0.2, 0.8, 9).astype(np.float32)
    return indices, np.stack([first, (1 - first).astype(np.float32)], axis=1)


#: Every shipped kind with the claims the guide makes for it:
#: ``id -> () -> (mapping, verify_mapping keywords)``.
SHIPPED = {
    # rbf: constants and linear fields reproduced with polynomial=True, in
    # every kernel; conservative mode preserves the total and, as the
    # transpose of an interpolant that reproduces linear fields, the
    # first moment.
    **{f"rbf-{kernel}-consistent": (
        lambda kernel=kernel: (rbf_mapping(XS, XT, kernel=kernel),
                               dict(polynomial_order=1, **COORDS)))
       for kernel in ("gaussian", "thin_plate_spline")},
    **{f"rbf-{kernel}-conservative": (
        lambda kernel=kernel: (rbf_mapping(XS, XT, kernel=kernel, mode="conservative"),
                               dict(polynomial_order=1, **COORDS)))
       for kernel in ("gaussian", "thin_plate_spline")},
    "nearest-consistent": lambda: (nearest_neighbor_mapping(XS, XT), {}),
    "nearest-conservative": lambda: (
        nearest_neighbor_mapping(XS, XT, mode="conservative"), {}),
    # A cell average: constants reproduced on covered cells, the integral
    # (cell sizes as the measure) preserved.
    "projection_1d": lambda: (projection_1d_mapping(SB, TB),
                              dict(consistent=True, conservative=True, **MEASURES)),
    # Bring your own matrix: the mode is a label, the claims are the matrix's.
    "matrix-consistent": lambda: (matrix_mapping(_row_stochastic()), {}),
    "matrix-conservative": lambda: (
        matrix_mapping(_row_stochastic().T, mode="conservative"), {}),
    "sparse_nearest-consistent": lambda: (sparse_nearest_neighbor_mapping(XS, XT), {}),
    "sparse_nearest-conservative-gather": lambda: (
        sparse_nearest_neighbor_mapping(XS, XT, mode="conservative"), {}),
    "sparse_nearest-conservative-scatter": lambda: (
        sparse_nearest_neighbor_mapping(XS, XT, mode="conservative", transpose="scatter"),
        {}),
    "sparse_projection_1d": lambda: (
        sparse_projection_1d_mapping(SB, TB),
        dict(consistent=True, conservative=True, **MEASURES)),
    "sparse_matrix-gather": lambda: (
        sparse_matrix_mapping(np.asarray([[0, 1], [1, 2], [2, -1]]),
                              np.asarray([[0.25, 0.75], [0.5, 0.5], [1.0, 0.0]], np.float32),
                              n_source=3), {}),
    "sparse_matrix-scatter": lambda: (
        StaticSparseMapping(_scatter_rows()[0], _scatter_rows()[1], n_source=9,
                            mode="conservative", layout="scatter", n_target=7), {}),
    # Both modes -- each is the other's transpose, so both directions of
    # the grid-and-points pair -- and both layouts of the grid field.
    "multilinear_grid-consistent-flat": lambda: _grid("consistent"),
    "multilinear_grid-conservative-shaped": lambda: _grid("conservative", "shaped"),
}

#: What the per-push table leaves out: the two other RBF kernels, the two
#: other mode-and-layout pairs of the grid kind, and grids of one and
#: three axes.
SHIPPED_SLOW = {
    **{f"rbf-{kernel}-consistent": (
        lambda kernel=kernel: (rbf_mapping(XS, XT, kernel=kernel),
                               dict(polynomial_order=1, **COORDS)))
       for kernel in ("multiquadric", "inverse_multiquadric")},
    "multilinear_grid-consistent-shaped": lambda: _grid("consistent", "shaped"),
    "multilinear_grid-conservative-flat": lambda: _grid("conservative"),
    **{f"multilinear_grid-{mode}-{d}d": (
        lambda mode=mode, d=d: _grid(mode, origin=(0.5, -1.0, 2.0)[:d],
                                     spacing=(0.25, 0.5, 0.125)[:d], shape=(5, 4, 3)[:d]))
       for mode in ("consistent", "conservative") for d in (1, 3)},
}

#: What a serialisable kind's round trip must have been: checked, not skipped.
_NOT_SERIALISABLE = ("matrix-consistent", "matrix-conservative", "sparse_matrix-gather",
                     "sparse_matrix-scatter")


def _failed(results):
    return sorted(name for name, r in results.items() if not r.passed)


def _report(results):
    return "\n".join(str(r) for r in results.values() if not r.passed)


@contextlib.contextmanager
def _float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def float64():
    with _float64():
        yield


# ---------------------------------------------------------------------------
# Every shipped kind passes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(SHIPPED))
def test_shipped_kind_passes_the_battery_with_the_claims_the_guide_makes(case):
    mapping, claims = SHIPPED[case]()
    results = verify_mapping(mapping, **claims, **KW)
    assert not _failed(results), _report(results)
    # Nothing passed by not being looked at: the claimed property, the
    # float32 run and (for a kind a config can carry) the round trip ran.
    mode = mapping.mode
    assert results[mode].status == "PASS"
    assert results["dtype_float32"].status == "PASS"
    assert results["dtype_float64"].skipped  # x64 is off in the default lane
    expected = "SKIP" if case in _NOT_SERIALISABLE else "PASS"
    assert results["round_trip"].status == expected, results["round_trip"].detail
    geometric = case.startswith("multilinear_grid")
    for name in ("geometry_derivative", "outside_hull"):
        assert results[name].status == ("PASS" if geometric else "SKIP")
    if geometric:
        assert "draws compared" in results["geometry_derivative"].detail


# Per push: tests/verification/test_verify_mapping.py::test_shipped_kind_passes_the_battery_with_the_claims_the_guide_makes (two RBF kernels in both modes; the two-axis grid in both modes, one layout each)
@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(SHIPPED_SLOW))
def test_the_other_kernels_layouts_and_grid_dimensions_pass_the_battery(case):
    mapping, claims = SHIPPED_SLOW[case]()
    results = verify_mapping(mapping, **claims, **KW)
    assert not _failed(results), _report(results)
    assert results[mapping.mode].status == "PASS"
    if case.startswith("multilinear_grid"):
        assert results["geometry_derivative"].status == "PASS"


# Per push: tests/verification/test_verify_mapping.py::test_shipped_kind_passes_the_battery_with_the_claims_the_guide_makes
@pytest.mark.slow
@pytest.mark.parametrize("case", sorted({**SHIPPED, **SHIPPED_SLOW}))
def test_shipped_kind_passes_at_the_default_depth_in_float64(case, float64):
    """The default number of draws, double-precision fields, and for a
    grid double-precision positions.  The RBF factories solve a kernel
    system with a relative ridge of 1e-8, so in float64 they reproduce
    and conserve to 1e-8, not to rounding (the guide says so): that is
    the tolerance they are held to here."""
    mapping, claims = {**SHIPPED, **SHIPPED_SLOW}[case]()
    if case.startswith("multilinear_grid") and not case.endswith("d"):
        mode, layout = case.split("-")[1:]
        mapping, claims = _grid(mode, layout, dtype=np.float64)
    if case.startswith("rbf"):
        claims["rounding_units"] = 1e-8 / float(np.finfo(np.float64).eps)
    results = verify_mapping(mapping, dtype=np.float64, derandomize=True, **claims)
    assert not _failed(results), _report(results)
    assert results["dtype_float64"].status == "PASS"


def test_a_registered_kind_of_the_callers_own_passes_and_is_round_tripped(registered_kind):
    mapping = registered_kind(XS, XT, power=3.0)
    results = verify_mapping(mapping, **KW)
    assert not _failed(results), _report(results)
    assert results["round_trip"].status == "PASS"
    assert_mapping_verified(mapping, require=("round_trip", "consistent"), **KW)


def test_a_hand_written_unregistered_kind_passes_and_its_round_trip_is_a_skip_with_the_reason():
    results = verify_mapping(_HandWritten(_row_stochastic()), **KW)
    assert not _failed(results), _report(results)
    assert results["consistent"].status == "PASS"
    assert results["round_trip"].skipped
    assert "not serialisable" in results["round_trip"].detail
    with pytest.raises(AssertionError, match="round_trip: SKIP"):
        assert_mapping_verified(_HandWritten(_row_stochastic()), require=("round_trip",), **KW)


def test_the_weights_a_graph_would_pass_are_the_ones_verified():
    mapping = matrix_mapping(_row_stochastic())
    spoiled = {"H": mapping.H.at[0, 0].add(0.5)}
    assert verify_mapping(mapping, checks=["consistent"], **KW)["consistent"].passed
    assert verify_mapping(mapping, weights=spoiled, checks=["consistent"],
                          **KW)["consistent"].failed


# ---------------------------------------------------------------------------
# Seeded broken kinds: each fails the check that should catch it, alone
# ---------------------------------------------------------------------------


class _HandWritten:
    """A dense kind written by hand: the protocol's members, no spec, no
    registration, no weights in the parameter tree."""

    kind = "hand_written"

    def __init__(self, matrix, mode="consistent"):
        self.matrix = jnp.asarray(matrix)
        self.mode = mode

    @property
    def n_source(self):
        return int(self.matrix.shape[1])

    @property
    def n_target(self):
        return int(self.matrix.shape[0])

    def params_pytree(self):
        return {}

    def apply(self, field, weights=None, geom=None):
        return self.matrix @ field

    def apply_T(self, field, weights=None, geom=None):
        return self.matrix.T @ field


def _unnormalised():
    """A reverse interpolant (targets to sources) whose rows do not sum to
    one: its transpose scatters more, or less, than it is given."""
    r = np.abs(XS[:, None] - XT[None, :])
    return (1.0 / (r + 0.1) ** 2).astype(np.float32) / 40.0


class _NonConservativeTranspose(_HandWritten):
    """Conservative by label: the transpose of an interpolant that does
    not reproduce constants."""

    def __init__(self):
        super().__init__(_unnormalised().T, mode="conservative")


class _TransposeNotTheAdjoint(_HandWritten):
    """A consistent gather whose ``apply_T`` scatters with other weights."""

    def __init__(self):
        super().__init__(_row_stochastic())

    def apply_T(self, field, weights=None, geom=None):
        return (self.matrix ** 2).T @ field


class _NonLinear(_HandWritten):
    def __init__(self):
        super().__init__(_row_stochastic())

    def apply(self, field, weights=None, geom=None):
        return self.matrix @ (field + 0.01 * field * jnp.abs(field))


class _MutatesItsInput(_HandWritten):
    def __init__(self):
        super().__init__(_row_stochastic())

    def apply(self, field, weights=None, geom=None):
        out = self.matrix @ jnp.asarray(field)
        if isinstance(field, np.ndarray):
            field[...] = 0.0  # a "scratch buffer" that is the caller's array
        return out


class _WrongOutputDtype(_HandWritten):
    def __init__(self):
        super().__init__(_row_stochastic())

    def apply(self, field, weights=None, geom=None):
        return (self.matrix @ field).astype(jnp.float32)


class _KeepsStateBetweenCalls(_HandWritten):
    def __init__(self):
        super().__init__(_row_stochastic())
        self.calls = 0

    def apply(self, field, weights=None, geom=None):
        self.calls += 1
        return self.matrix @ field * (1.0 + 0.001 * (self.calls % 2))


class _DerivativeUnderStopGradient(MultilinearGridMapping):
    """The shipped kernel with its positions detached: the forward value
    is right and the derivative with respect to a position is zero."""

    def __init__(self, mode="consistent"):
        super().__init__(ORIGIN, SPACING, SHAPE, N_POINTS, mode, "flat", None)

    def _stencil(self, geom):
        return super()._stencil(jax.lax.stop_gradient(jnp.asarray(geom)))


def _grid_claims():
    return dict(geometry_strategy=_positions(LOWER, UPPER), hull=(LOWER, UPPER),
                outside="clamp")


#: ``id -> (() -> mapping, keywords, the one check that must fail)``.
BROKEN = {
    "a non-conservative transpose": (_NonConservativeTranspose, {}, "conservative"),
    "a transpose that is not the adjoint": (_TransposeNotTheAdjoint, {}, "adjoint"),
    "a derivative under stop_gradient": (
        _DerivativeUnderStopGradient, _grid_claims, "geometry_derivative"),
    # Its claims are withdrawn: a non-linear map reproduces no constant.
    "a non-linear apply": (_NonLinear, dict(consistent=False), "linearity"),
    "a kind that mutates its input": (_MutatesItsInput, {}, "structure"),
    "a kind that keeps state between calls": (
        _KeepsStateBetweenCalls, dict(consistent=False, checks=["structure"]), "structure"),
}


@pytest.mark.parametrize("case", sorted(BROKEN))
def test_seeded_broken_kind_fails_the_check_that_should_catch_it_and_no_other(case):
    build, claims, caught_by = BROKEN[case]
    claims = claims() if callable(claims) else claims
    results = verify_mapping(build(), **claims, **KW)
    assert _failed(results) == [caught_by], _report(results) or "nothing failed"
    assert results[caught_by].failed and results[caught_by].counterexample


def test_wrong_output_dtype_fails_the_dtype_check_and_no_other(float64):
    results = verify_mapping(_WrongOutputDtype(), **KW)
    assert _failed(results) == ["dtype_float64"], _report(results) or "nothing failed"
    assert "promote to float64" in results["dtype_float64"].detail


class _AnotherResultUnderJit(_HandWritten):
    """Branches in Python on whether it is being traced."""

    def __init__(self):
        super().__init__(_row_stochastic())

    def apply(self, field, weights=None, geom=None):
        if isinstance(field, jax.core.Tracer):
            return 1.5 * (self.matrix @ field)
        return self.matrix @ field


def test_a_kind_that_computes_another_result_under_jit_fails_jit_consistent():
    results = verify_mapping(_AnotherResultUnderJit(), consistent=False,
                             checks=["structure", "jit_consistent", "dtype"], **KW)
    assert _failed(results) == ["jit_consistent"], _report(results) or "nothing failed"
    assert "compiled delivery differs" in results["jit_consistent"].detail


def test_a_kind_that_does_not_clamp_where_it_was_declared_to_fails_the_hull_check():
    """The shipped grid, declared to clamp at a box one cell short of its
    own: between the two it interpolates, which is not a clamp."""
    mapping = multilinear_grid_mapping(ORIGIN, SPACING, SHAPE, n_points=N_POINTS)
    short = UPPER - np.asarray(SPACING)
    results = verify_mapping(mapping, geometry_strategy=_positions(LOWER, short),
                             hull=(LOWER, short), outside="clamp",
                             checks=["outside_hull"], **KW)
    assert results["outside_hull"].failed
    assert "declared to clamp" in results["outside_hull"].detail


def test_the_adjoint_identity_is_not_judged_on_a_map_that_is_not_linear():
    results = verify_mapping(_NonLinear(), consistent=False, **KW)
    assert results["adjoint"].skipped and "not linear" in results["adjoint"].detail


def test_a_claim_the_kind_does_not_keep_fails():
    """Nearest neighbour reproduces constants and nothing more."""
    mapping = nearest_neighbor_mapping(XS, XT)
    results = verify_mapping(mapping, polynomial_order=1, **COORDS, **KW)
    assert _failed(results) == ["consistent"]
    assert "exponents (1,)" in results["consistent"].detail
    conservative = nearest_neighbor_mapping(XS, XT, mode="conservative")
    results = verify_mapping(conservative, polynomial_order=1, checks=["conservative"],
                             **COORDS, **KW)
    assert results["conservative"].failed and "moment" in results["conservative"].detail
    # A cell average reproduces constants at the cell centres, not linear
    # fields (the guide's worked example turns on this).
    centres = dict(source_coordinates=0.5 * (SB[1:] + SB[:-1]),
                   target_coordinates=0.5 * (TB[1:] + TB[:-1]))
    results = verify_mapping(projection_1d_mapping(SB, TB), consistent=True,
                             polynomial_order=1, checks=["consistent"], **centres, **KW)
    assert results["consistent"].failed


def test_a_factory_that_does_not_record_what_it_was_given_fails_the_round_trip(
        registered_kind):
    mapping = registered_kind(XS, XT, power=3.0, forget_power=True)
    results = verify_mapping(mapping, **KW)
    assert _failed(results) == ["round_trip"], _report(results) or "nothing failed"


def test_an_object_without_the_protocols_members_fails_structure_and_nothing_else_runs():
    class Half:
        kind, mode = "half", "consistent"
        n_source = n_target = 3

        def apply(self, field, weights=None, geom=None):
            return field

    results = verify_mapping(Half(), **KW)
    assert results["structure"].failed
    assert "params_pytree" in results["structure"].detail
    assert all(r.skipped for name, r in results.items() if name != "structure")
    with pytest.raises(AssertionError, match="structure: FAIL"):
        assert_mapping_verified(Half(), **KW)


def test_a_params_pytree_add_edge_would_refuse_fails_structure():
    class Nested(_HandWritten):
        def params_pytree(self):
            return {"H": {"inner": self.matrix}}

    results = verify_mapping(Nested(_row_stochastic()), checks=["structure"], **KW)
    assert results["structure"].failed
    assert "add_edge refuses" in results["structure"].detail


# ---------------------------------------------------------------------------
# A whole edge: the mapping, then the transform
# ---------------------------------------------------------------------------


def _times(factor):
    def transform(value):
        return factor * value
    return transform


def _clamp(value):
    return jnp.clip(value, -1.0, 1.0)


def test_an_edge_is_verified_on_what_it_delivers_with_its_declared_scale():
    edge = EdgeSpec("fluid", "solid", "traction", "force",
                    mapping=nearest_neighbor_mapping(XS, XT, mode="conservative"),
                    transform=_times(-2.5))
    results = verify_mapping(edge, scale=-2.5, **KW)
    assert not _failed(results), _report(results)
    assert results["conservative"].status == "PASS"
    # The same through a bare mapping and transform=.
    results = verify_mapping(nearest_neighbor_mapping(XS, XT), transform=_times(-2.5),
                             scale=-2.5, checks=["linearity", "consistent", "adjoint"], **KW)
    assert not _failed(results), _report(results)
    assert results["consistent"].status == "PASS"


def test_an_undeclared_scale_fails_conservation_and_names_the_transform():
    edge = EdgeSpec("fluid", "solid", "traction", "force",
                    mapping=nearest_neighbor_mapping(XS, XT, mode="conservative"),
                    transform=_times(-2.5))
    results = verify_mapping(edge, **KW)
    assert _failed(results) == ["adjoint", "conservative"]
    assert "_times.<locals>.transform" in results["conservative"].detail


def test_a_clamp_fails_conservation_visibly_with_the_reason():
    edge = EdgeSpec("fluid", "solid", "traction", "force",
                    mapping=nearest_neighbor_mapping(XS, XT, mode="conservative"),
                    transform=_clamp)
    results = verify_mapping(edge, **KW)
    assert results["conservative"].failed and results["linearity"].failed
    detail = results["conservative"].detail
    assert "is not preserved" in detail and "_clamp" in detail and "a clamp" in detail
    # The mapping alone is conservative: the edge is what loses the total.
    assert verify_mapping(edge.mapping, checks=["conservative"], **KW)["conservative"].passed


def test_an_edge_with_a_geometry_dependent_mapping_is_verified_through_the_edge_rule():
    mapping, claims = _grid("conservative")
    edge = EdgeSpec("body", "fluid", "force", "body_force", mapping=mapping,
                    transform=_times(4.0), geometry=("source", "marker_position"))
    results = verify_mapping(edge, scale=4.0, **claims, **KW, checks=[
        "structure", "conservative", "adjoint", "geometry_derivative", "outside_hull"])
    assert not _failed(results), _report(results)
    assert all(r.status == "PASS" for r in results.values())


# ---------------------------------------------------------------------------
# Positions: fixed, drawn, and on a kink
# ---------------------------------------------------------------------------


def test_fixed_sample_positions_are_enough_for_a_geometry_kind():
    mapping = multilinear_grid_mapping(ORIGIN, SPACING, SHAPE, n_points=N_POINTS)
    inside = (LOWER + np.random.default_rng(8).uniform(0.1, 0.9, (N_POINTS, 2))
              * (UPPER - LOWER)).astype(np.float32)
    results = verify_mapping(mapping, geometry=inside, **KW, checks=[
        "consistent", "adjoint", "geometry_derivative", "outside_hull"])
    assert not _failed(results), _report(results)
    assert results["geometry_derivative"].status == "PASS"
    assert results["outside_hull"].skipped and "not claimed" in results["outside_hull"].detail


def test_a_check_in_which_no_draw_could_be_compared_is_a_skip_and_never_a_pass():
    """A step far below what float32 positions resolve: no stencil can be
    laid, so nothing is compared, and the result says that."""
    mapping = multilinear_grid_mapping(ORIGIN, SPACING, SHAPE, n_points=N_POINTS)
    inside = (LOWER + np.random.default_rng(8).uniform(0.1, 0.9, (N_POINTS, 2))
              * (UPPER - LOWER)).astype(np.float32)
    kw = dict(geometry=inside, geometry_step=1e-12, checks=["geometry_derivative"], **KW)
    result = verify_mapping(mapping, **kw)["geometry_derivative"]
    assert result.skipped and "no draw could be compared" in result.detail
    assert "cannot hold a stencil" in result.detail
    with pytest.raises(AssertionError, match="no draw could be compared"):
        assert_mapping_verified(mapping, require=("geometry_derivative",), **kw)


def test_a_drawn_kink_is_counted_beside_the_draws_that_were_compared():
    """Half the points on a lattice plane of the first axis: the draws
    that move one of them along that axis are kinks, the rest compare."""
    mapping = multilinear_grid_mapping(ORIGIN, SPACING, SHAPE, n_points=N_POINTS)
    rng = np.random.default_rng(9)
    positions = (LOWER + rng.uniform(0.1, 0.9, (N_POINTS, 2)) * (UPPER - LOWER))
    positions[:3, 0] = ORIGIN[0] + 2 * SPACING[0]
    result = verify_mapping(mapping, geometry=positions.astype(np.float32),
                            checks=["geometry_derivative"], max_examples=60,
                            derandomize=True)["geometry_derivative"]
    assert result.status == "PASS"
    compared, kinks = (int(part.split()[0]) for part in result.detail.split(";"))
    assert compared > 0 and kinks > 0 and compared + kinks == result.n_examples
    assert "on a kink of the kernel" in result.detail


def test_a_geometry_strategy_that_raises_is_refused_not_reported_as_the_mappings_failure():
    mapping = multilinear_grid_mapping(ORIGIN, SPACING, SHAPE, n_points=N_POINTS)
    broken = st.integers(0, 3).map(lambda i: [1.0][i + 1])
    with pytest.raises(ValueError, match="geometry_strategy raised IndexError"):
        verify_mapping(mapping, geometry_strategy=broken, checks=["linearity"], **KW)
    with pytest.raises(ValueError, match="must yield an array of positions"):
        verify_mapping(mapping, geometry_strategy=st.just("here"), checks=["linearity"], **KW)


# ---------------------------------------------------------------------------
# Options and refusals
# ---------------------------------------------------------------------------


def test_a_property_that_is_not_claimed_is_not_checked_and_is_listed_as_not_claimed():
    only = dict(checks=["consistent", "conservative"], **KW)
    results = verify_mapping(nearest_neighbor_mapping(XS, XT), **only)
    assert results["conservative"].skipped
    assert "not claimed" in results["conservative"].detail
    both = verify_mapping(nearest_neighbor_mapping(XS, XT), conservative=True, **only)
    assert both["conservative"].failed  # stated, checked, and not true of a selection
    with pytest.raises(AssertionError, match="conservative: SKIP"):
        assert_mapping_verified(nearest_neighbor_mapping(XS, XT), require=("conservative",),
                                **only)


@pytest.mark.parametrize("keywords, message", [
    (dict(checks=["lineraity"]), "unknown checks"),
    (dict(polynomial_order=1), "pass source_coordinates= and target_coordinates="),
    (dict(polynomial_order=-1), "non-negative integer"),
    (dict(scale=2.0), "there is no transform"),
    (dict(transform=_times(2.0), scale=0.0), "finite, non-zero"),
    (dict(geometry=np.zeros((7, 1), np.float32)), "is a static mapping"),
    (dict(hull=(0.0, 1.0)), "go together"),
    (dict(hull=(0.0, 1.0), outside="zero"), "only 'clamp'"),
    (dict(rounding_units=0), "must be positive"),
    (dict(source_measure=np.ones(3)), "one weight per entry"),
])
def test_contradictory_options_are_refused(keywords, message):
    keywords.setdefault("conservative", "source_measure" in keywords)
    with pytest.raises(ValueError, match=message):
        verify_mapping(nearest_neighbor_mapping(XS, XT), **keywords, **KW)


def test_a_geometry_kind_without_positions_and_an_edge_without_a_mapping_are_refused():
    mapping = multilinear_grid_mapping(ORIGIN, SPACING, SHAPE, n_points=N_POINTS)
    with pytest.raises(ValueError, match="pass sample positions as geometry="):
        verify_mapping(mapping, **KW)
    positions = np.zeros((N_POINTS, 2), np.float32)
    with pytest.raises(ValueError, match="pass one or the other"):
        verify_mapping(mapping, geometry=positions,
                       geometry_strategy=_positions(LOWER, UPPER), **KW)
    with pytest.raises(TypeError, match="must be a Hypothesis strategy"):
        verify_mapping(mapping, geometry_strategy=positions, **KW)
    with pytest.raises(ValueError, match="has no mapping"):
        verify_mapping(EdgeSpec("a", "b", "x", "y", transform=_times(2.0)), **KW)
    edge = EdgeSpec("a", "b", "x", "y", mapping=nearest_neighbor_mapping(XS, XT))
    with pytest.raises(ValueError, match="carries its own"):
        verify_mapping(edge, transform=_times(2.0), **KW)
    with pytest.raises(ValueError, match="names no geometry"):
        verify_mapping(EdgeSpec("a", "b", "x", "y", mapping=mapping), geometry=positions, **KW)
    with pytest.raises(ValueError, match="not among the results"):
        assert_mapping_verified(edge, require=("conservation",), **KW)


def test_the_default_checks_are_the_documented_ones_in_order():
    results = verify_mapping(nearest_neighbor_mapping(XS, XT), **KW)
    names = [n for check in DEFAULT_MAPPING_CHECKS
             for n in (("dtype_float32", "dtype_float64") if check == "dtype" else (check,))]
    assert list(results) == names
    only = verify_mapping(nearest_neighbor_mapping(XS, XT), checks=["adjoint", "dtype"],
                          dtypes=(np.float32,), **KW)
    assert list(only) == ["adjoint", "dtype_float32"]


def test_a_sixteen_bit_dtype_is_checked_only_when_it_is_asked_for():
    mapping = nearest_neighbor_mapping(XS, XT)
    assert "dtype_float16" not in verify_mapping(mapping, checks=["dtype"], **KW)
    claimed = verify_mapping(mapping, checks=["dtype"], dtypes=(np.float16,), **KW)
    # A float32 selection matrix promotes a float16 field to float32:
    # the kind does not claim float16, and the check says what it does.
    assert list(claimed) == ["dtype_float16"] and claimed["dtype_float16"].passed


# ---------------------------------------------------------------------------
# A kind registered by the caller (and removed again)
# ---------------------------------------------------------------------------


@pytest.fixture
def registered_kind():
    kind = "verify_mapping_inverse_distance"

    @register_mapping(
        kind, arrays=("source_points", "target_points"), hyperparameters={"power": float},
        references={"source_points": "source_ref", "target_points": "target_ref"})
    def inverse_distance_mapping(source_points, target_points, *, power=2.0,
                                 source_ref=None, target_ref=None, forget_power=False):
        src = np.asarray(source_points, np.float64).reshape(-1, 1)
        tgt = np.asarray(target_points, np.float64).reshape(-1, 1)
        weight = 1.0 / (np.abs(tgt - src.T) + 1e-3) ** power
        matrix = (weight / weight.sum(axis=1, keepdims=True)).astype(np.float32)
        spec = MappingSpec(kind, {} if forget_power else {"power": float(power)}, {
            "source_points": reference_for_array(source_points, source_ref,
                                                 name="source_points"),
            "target_points": reference_for_array(target_points, target_ref,
                                                 name="target_points")})
        return StaticLinearMapping(jnp.asarray(matrix), kind=kind, spec=spec)

    try:
        yield inverse_distance_mapping
    finally:
        mapping_registry._unregister(kind)
