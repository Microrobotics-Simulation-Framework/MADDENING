"""``StaticSparseMapping``: a row structure and one weight per slot.

What is held here:

* **Sparse equals dense.**  The rows of a sparse mapping, scattered into a
  zero matrix, are a dense operator; ``apply`` and ``apply_T`` must give
  what that operator gives, within the rounding of one row sum, in both
  layouts, for vector and matrix fields, in float32, float64, mixed and
  16-bit dtypes, eagerly, under ``jit``, ``grad`` and ``vmap``.
* **The index is structure.**  It is validated once, frozen, kept out of
  the parameter tree (whose one leaf is the floating-point ``W``), and
  described by a digest that changes with the pattern and with nothing
  else.
* **What is refused**: every malformed structure when the mapping is
  constructed, and a field or a weight array of another size when it is
  applied (a gather would clamp the index and answer).
* **Reproducibility on the CPU, as measured**: one result per compiled
  program for the row sum, and for the scatter-add the in-order sum.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import copy
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.mapping import (
    Mapping,
    StaticLinearMapping,
    _params_contract_problem,
)
from maddening.core.coupling.mapping_spec import MappingSpec
from maddening.core.coupling.sparse_mapping import (
    StaticSparseMapping,
    sparse_matrix_mapping,
)
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.inspection import _mapping_text
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.sparse_mapping_support import (
    CASES,
    assert_within_dense,
    assert_within_rows,
    densify,
    interface_points,
    valid_slots,
    x64,
)

N_SOURCE, N_TARGET, K = 7, 5, 4


def _pattern(seed: int = 0, *, n_rows: int = N_TARGET, n_columns: int = N_SOURCE,
             k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """A ragged pattern: a full row, an empty row, a row that repeats an
    index, and rows of every length between.  ``(indices, counts)``, the
    padded slots holding index 0."""
    rng = np.random.default_rng(seed)
    counts = np.array([k, 0] + [int(rng.integers(1, k)) for _ in range(n_rows - 2)])
    indices = np.zeros((n_rows, k), dtype=np.int64)
    for r, count in enumerate(counts):
        indices[r, :count] = rng.integers(0, n_columns, size=count)
    # The full row: distinct entries but for one repeat (the two add).
    indices[0] = [(3 + j) % n_columns for j in range(k)]
    indices[0, 1] = indices[0, 0]
    return indices, counts


def _weights(indices, counts, dtype, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=indices.shape)
    valid = np.arange(indices.shape[1])[None, :] < counts[:, None]
    return np.where(valid, values, 0.0).astype(dtype)


def _mapping(layout: str = "gather", dtype="float32", seed: int = 0) -> StaticSparseMapping:
    """A ragged mapping ``N_SOURCE -> N_TARGET`` in *layout*."""
    if layout == "gather":
        indices, counts = _pattern(seed)
        return StaticSparseMapping(indices, jnp.asarray(_weights(indices, counts, dtype)),
                                   n_source=N_SOURCE, counts=counts)
    indices, counts = _pattern(seed, n_rows=N_SOURCE, n_columns=N_TARGET)
    return StaticSparseMapping(indices, jnp.asarray(_weights(indices, counts, dtype)),
                               n_source=N_SOURCE, n_target=N_TARGET, counts=counts,
                               layout="scatter")


def _field(n: int, dtype, *, channels: int = 0, seed: int = 2):
    rng = np.random.default_rng(seed)
    shape = (n, channels) if channels else (n,)
    return jnp.asarray(rng.normal(size=shape), dtype=dtype)


# ---------------------------------------------------------------------------
# Sparse equals dense
# ---------------------------------------------------------------------------

#: ``(weights dtype, field dtype, x64)``: float32, float64, both mixes of
#: the two, and the two 16-bit dtypes.
DTYPES = [
    pytest.param("float32", "float32", False, id="f32"),
    pytest.param("float64", "float64", True, id="f64"),
    pytest.param("float64", "float32", True, id="mixed-f64-weights"),
    pytest.param("float32", "float64", True, id="mixed-f64-field"),
    pytest.param("float16", "float16", False, id="float16"),
    pytest.param("bfloat16", "bfloat16", False, id="bfloat16"),
]


@pytest.mark.parametrize("channels", [0, 3], ids=["vector", "matrix"])
@pytest.mark.parametrize("layout", ["gather", "scatter"])
@pytest.mark.parametrize("w_dtype, f_dtype, enabled", DTYPES)
def test_apply_and_apply_T_are_the_dense_operator_to_rounding(w_dtype, f_dtype, enabled,
                                                               layout, channels):
    """``apply`` is ``H @ field`` and ``apply_T`` is ``H.T @ field`` for the
    ``H`` the rows are, at the dtype ``H @ field`` has, within ``(k_i + 2)
    eps sum |terms|`` per entry -- eagerly and under ``jit``."""
    with x64(enabled):
        m = _mapping(layout, w_dtype)
        assert str(m.weights.dtype) == w_dtype
        H = densify(m)
        dense = StaticLinearMapping(jnp.asarray(H))
        f = _field(N_SOURCE, f_dtype, channels=channels)
        g = _field(N_TARGET, f_dtype, channels=channels, seed=3)
        for fn in (lambda a, b: a(b), lambda a, b: jax.jit(a)(b)):
            out = fn(m.apply, f)
            assert out.dtype == dense.apply(f).dtype and out.shape == dense.apply(f).shape
            assert_within_rows(out, m, f, what=f"apply {layout}")
            back = fn(m.apply_T, g)
            assert back.dtype == dense.apply_T(g).dtype
            assert back.shape == dense.apply_T(g).shape
            assert_within_rows(back, m, g, transpose=True, what=f"apply_T {layout}")
            # ... and the dense mapping holding the same matrix is within the
            # rounding of its own, longer, row sum of the same exact value
            assert_within_dense(dense.apply(f), H, f, N_SOURCE, what="the dense apply",
                                extra=4.0)


def test_the_fixture_has_a_full_row_an_empty_row_a_repeat_and_padding():
    """The fixture can express the defects: a mask that is dropped (padded
    slots exist, and the field is not zero at index 0), a row that is not
    summed (rows of several entries), a repeated index, an empty row."""
    for layout in ("gather", "scatter"):
        m = _mapping(layout)
        counts = np.asarray(m.counts)
        assert counts.max() == K and counts.min() == 0 and len(set(counts)) >= 3
        assert m.indices[0, 0] == m.indices[0, 1]
        assert not valid_slots(m).all() and m.nnz == counts.sum() < m.indices.size
    assert float(_field(N_SOURCE, "float32")[0]) != 0.0


@pytest.mark.parametrize("layout", ["gather", "scatter"])
def test_one_entry_per_target_is_the_dense_result_exactly(layout):
    """With one term per row the sparse result and the dense one are the
    same number: the dense sum adds exact zeros to the one product."""
    rng = np.random.default_rng(5)
    picks = rng.integers(0, N_SOURCE, size=N_TARGET)
    values = rng.normal(size=N_TARGET).astype(np.float32)
    if layout == "gather":
        m = StaticSparseMapping(picks[:, None], jnp.asarray(values[:, None]),
                                n_source=N_SOURCE)
    else:
        # per source, the targets that picked it
        lists = [np.nonzero(picks == j)[0] for j in range(N_SOURCE)]
        k = max(len(entries) for entries in lists)
        indices = np.zeros((N_SOURCE, k), np.int64)
        weights = np.zeros((N_SOURCE, k), np.float32)
        for j, entries in enumerate(lists):
            indices[j, :len(entries)] = entries
            weights[j, :len(entries)] = values[entries]
        m = StaticSparseMapping(indices, jnp.asarray(weights), n_source=N_SOURCE,
                                n_target=N_TARGET, layout="scatter",
                                counts=[len(entries) for entries in lists])
    H = densify(m)
    assert (np.count_nonzero(H, axis=1) == 1).all()
    for channels in (0, 2):
        f = _field(N_SOURCE, "float32", channels=channels)
        np.testing.assert_array_equal(np.asarray(jax.jit(m.apply)(f)),
                                      np.asarray(jax.jit(lambda x: jnp.asarray(H) @ x)(f)))


@pytest.mark.parametrize("layout", ["gather", "scatter"])
def test_the_gradient_with_respect_to_the_weights_and_the_field_is_the_dense_one(layout):
    """``jax.grad`` of a loss through ``apply``: with respect to ``W`` it is
    the dense gradient read through the pattern (and exactly zero in a
    padded slot); with respect to the field it is the dense one."""
    m = _mapping(layout)
    H = jnp.asarray(densify(m))
    f = _field(N_SOURCE, "float32", channels=2)
    target = _field(N_TARGET, "float32", channels=2, seed=9)

    def loss(apply, w, x):
        return jnp.sum((apply(x, w) - target) ** 2)

    gw, gx = jax.jit(jax.grad(lambda w, x: loss(lambda a, b: m.apply(a, {"W": b}), w, x),
                              argnums=(0, 1)))(m.weights, f)
    dw, dx = jax.grad(lambda w, x: loss(lambda a, b: b @ a, w, x), argnums=(0, 1))(H, f)
    valid = valid_slots(m)
    rows = np.broadcast_to(np.arange(m.indices.shape[0])[:, None], m.indices.shape)
    tgt, src = (rows, m.indices) if layout == "gather" else (m.indices, rows)
    expected = np.where(valid, np.asarray(dw)[tgt, src], 0.0)
    np.testing.assert_allclose(np.asarray(gw), expected, rtol=2e-5, atol=1e-6)
    assert not np.asarray(gw)[~valid].any(), "a padded weight has no derivative"
    np.testing.assert_allclose(np.asarray(gx), np.asarray(dx), rtol=2e-5, atol=1e-6)
    assert float(jnp.max(jnp.abs(gw))) > 0.0 and float(jnp.max(jnp.abs(gx))) > 0.0


def test_a_padded_slot_contributes_an_exact_zero_whatever_the_field_holds_at_index_0():
    """The mask is on the gathered values: an infinity at index 0 -- where
    every padded slot points -- reaches only the rows that list index 0,
    and no padded weight gets a ``0 * inf`` derivative."""
    indices = np.array([[1, 2, 0], [3, 0, 0], [0, 0, 0], [4, 4, 0]])
    counts = np.array([2, 1, 1, 2])
    weights = np.where(np.arange(3)[None, :] < counts[:, None], 1.5, 0.0).astype(np.float32)
    m = StaticSparseMapping(indices, jnp.asarray(weights), n_source=5, counts=counts)
    f = jnp.asarray([np.inf, 1.0, 2.0, 3.0, 4.0], jnp.float32)
    out = np.asarray(jax.jit(m.apply)(f))
    assert np.isinf(out[2]) and np.isfinite(out[[0, 1, 3]]).all()
    np.testing.assert_array_equal(out[[0, 1, 3]], np.float32([4.5, 4.5, 12.0]))
    grad = np.asarray(jax.grad(
        lambda w: jnp.sum(m.apply(f, {"W": w})[jnp.asarray([0, 1, 3])]))(m.weights))
    padded = ~valid_slots(m)
    assert padded.sum() == 6 and np.isfinite(grad[padded]).all() and not grad[padded].any()
    np.testing.assert_array_equal(grad[[0, 0, 1, 3, 3], [0, 1, 0, 0, 1]], [1, 2, 3, 4, 4])
    # the dense mapping is not finite anywhere: a zero entry meets the infinity
    assert not np.isfinite(np.asarray(jnp.asarray(densify(m)) @ f)).any()


@pytest.mark.parametrize("layout", ["gather", "scatter"])
def test_vmap_over_fields_and_over_weights_is_each_member_alone(layout):
    """A batch of fields, and a batch of weight arrays, through one
    ``jax.vmap``: each member is the mapping applied to it alone, to
    rounding (a batched row sum may round in another order, as a batched
    ``H @ field`` may)."""
    m = _mapping(layout)
    fields = jnp.stack([_field(N_SOURCE, "float32", seed=s) for s in range(4)])
    scales = jnp.asarray([1.0, -0.5, 2.0, 0.25], jnp.float32)
    weights = scales[:, None, None] * m.weights
    eps = float(np.finfo(np.float32).eps)

    batched = np.asarray(jax.jit(jax.vmap(m.apply))(fields))
    alone = np.stack([np.asarray(m.apply(f)) for f in fields])
    np.testing.assert_allclose(batched, alone, rtol=8 * eps, atol=8 * eps)

    by_weight = np.asarray(jax.jit(jax.vmap(lambda w: m.apply(fields[0], {"W": w})))(weights))
    alone = np.stack([np.asarray(m.apply(fields[0], {"W": w})) for w in weights])
    np.testing.assert_allclose(by_weight, alone, rtol=8 * eps, atol=8 * eps)
    assert len({row.tobytes() for row in by_weight}) == 4          # the weights were read


def test_the_weights_passed_to_apply_are_the_ones_used():
    m = _mapping()
    f = _field(N_SOURCE, "float32")
    doubled = np.asarray(m.apply(f, {"W": 2.0 * m.weights}))
    np.testing.assert_allclose(doubled, 2.0 * np.asarray(m.apply(f)), rtol=1e-6)
    assert np.asarray(m.apply(f, None)).tobytes() == np.asarray(m.apply(f)).tobytes()


# ---------------------------------------------------------------------------
# What is refused when a mapping is applied
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["gather", "scatter"])
@pytest.mark.parametrize("size", [N_SOURCE - 1, N_SOURCE + 1, 1])
def test_a_field_of_another_size_is_refused_rather_than_clamped(layout, size):
    """A gather clamps an index past the end of its operand and a longer
    field is simply not read to its end: either way ``apply`` would return
    a result.  The size is static, so it is refused when the step is
    traced."""
    m = _mapping(layout)
    f = jnp.ones(size, jnp.float32)
    with pytest.raises(ValueError, match=f"first axis must be n_source = {N_SOURCE}"):
        m.apply(f)
    with pytest.raises(ValueError, match=f"first axis must be n_source = {N_SOURCE}"):
        jax.jit(m.apply)(f)
    with pytest.raises(ValueError, match=f"first axis must be n_target = {N_TARGET}"):
        m.apply_T(jnp.ones(N_TARGET + 1, jnp.float32))
    with pytest.raises(ValueError, match="first axis must be n_source"):
        m.apply(jnp.asarray(1.0))


@pytest.mark.parametrize("layout", ["gather", "scatter"])
@pytest.mark.parametrize("shape", ["one row", "one slot", "transposed", "scalar"])
def test_weights_of_another_shape_are_refused_rather_than_broadcast(layout, shape):
    """``W`` of shape ``(1, k)`` or ``(n_rows, 1)`` broadcasts against the
    gathered values: one weight per slot is checked, not assumed."""
    m = _mapping(layout)
    rows, k = m.indices.shape
    w = {"one row": jnp.ones((1, k)), "one slot": jnp.ones((rows, 1)),
         "transposed": jnp.ones((k, rows)), "scalar": jnp.asarray(1.0)}[shape]
    f = _field(N_SOURCE, "float32")
    with pytest.raises(ValueError, match="one weight per slot"):
        m.apply(f, {"W": w})
    with pytest.raises(ValueError, match="one weight per slot"):
        jax.jit(lambda x: m.apply_T(x, {"W": w}))(jnp.ones(N_TARGET))


# ---------------------------------------------------------------------------
# Construction: what a row structure may be
# ---------------------------------------------------------------------------

def _good(**changes):
    indices, counts = _pattern()
    kwargs = dict(indices=indices, weights=jnp.asarray(_weights(indices, counts, "float32")),
                  n_source=N_SOURCE, counts=counts)
    kwargs.update(changes)
    return StaticSparseMapping(**kwargs)


def _stray_index():
    indices, counts = _pattern()
    indices[1, 2] = 3                  # row 1 is empty: every slot is padding
    return dict(indices=indices)


def _stray_weight():
    indices, counts = _pattern()
    weights = _weights(indices, counts, "float32")
    weights[1, 0] = 0.25
    return dict(weights=jnp.asarray(weights))


def _bad_index(value):
    indices, _counts = _pattern()
    indices[0, 2] = value
    return dict(indices=indices)


def _nonfinite(value):
    indices, counts = _pattern()
    weights = _weights(indices, counts, "float32")
    weights[0, 0] = value
    return dict(weights=jnp.asarray(weights))


REFUSED = {
    "a float index": (lambda: dict(indices=_pattern()[0].astype(np.float64)),
                      "indices must be an integer array, got dtype float64"),
    "a boolean index": (lambda: dict(indices=np.ones((N_TARGET, K), bool)),
                        "indices must be an integer array, got dtype bool"),
    "a one-dimensional index": (lambda: dict(indices=np.zeros(N_TARGET, np.int64)),
                                r"indices must have shape \(n_rows, k\)"),
    "no rows": (lambda: dict(indices=np.zeros((0, K), np.int64)),
                r"indices must have shape \(n_rows, k\)"),
    "no slots": (lambda: dict(indices=np.zeros((N_TARGET, 0), np.int64)),
                 r"indices must have shape \(n_rows, k\)"),
    "an index of n_source": (lambda: _bad_index(N_SOURCE),
                             rf"indices\[0, 2\] = {N_SOURCE} is outside \[0, n_source\)"),
    "a negative index": (lambda: _bad_index(-1),
                         r"indices\[0, 2\] = -1 is outside \[0, n_source\)"),
    "an index past int32": (lambda: _bad_index(2 ** 40),
                            rf"indices\[0, 2\] = {2 ** 40} is outside"),
    "a padded slot with an index": (_stray_index,
                                    r"indices\[1, 2\] = 3 is a padded slot \(row 1 has 0"),
    "a padded slot with a weight": (_stray_weight,
                                    r"weights\[1, 0\] = 0.25 is in a padded slot"),
    "a NaN weight": (lambda: _nonfinite(np.nan), r"weights\[0, 0\] is not finite \(nan\)"),
    "an infinite weight": (lambda: _nonfinite(np.inf),
                           r"weights\[0, 0\] is not finite \(inf\)"),
    "integer weights": (lambda: dict(weights=jnp.ones((N_TARGET, K), jnp.int32)),
                        "weights must be a floating-point array, got dtype int32"),
    "weights of another shape": (lambda: dict(weights=jnp.ones((N_TARGET, K + 1))),
                                 "one weight per slot"),
    "counts of another length": (lambda: dict(counts=np.ones(N_TARGET + 1, np.int64)),
                                 "counts must be an integer array of shape"),
    "float counts": (lambda: dict(counts=np.ones(N_TARGET)),
                     "counts must be an integer array of shape"),
    "a count above k": (lambda: dict(counts=np.array([K + 1, 0, 1, 1, 1])),
                        rf"counts\[0\] = {K + 1} is outside \[0, {K}\]"),
    "a negative count": (lambda: dict(counts=np.array([K, -1, 1, 1, 1])),
                         rf"counts\[1\] = -1 is outside \[0, {K}\]"),
    "n_source zero": (lambda: dict(n_source=0), "n_source must be between 1 and"),
    "n_source past int32": (lambda: dict(n_source=2 ** 31), "n_source must be between 1 and"),
    "n_source a float": (lambda: dict(n_source=7.0), "n_source must be an integer"),
    "n_source a bool": (lambda: dict(n_source=True), "n_source must be an integer"),
    "n_target that is not the rows": (lambda: dict(n_target=N_TARGET + 1),
                                      "does not match the 5 rows of the index"),
    "an unknown layout": (lambda: dict(layout="coo"), "layout='coo' not in"),
    "an unknown mode": (lambda: dict(mode="sideways"), "mode='sideways' not in"),
    "an empty kind": (lambda: dict(kind=""), "kind must be a non-empty string"),
}


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_a_malformed_structure_is_refused_when_the_mapping_is_constructed(name):
    changes, message = REFUSED[name]
    with pytest.raises(ValueError, match=message):
        _good(**changes())


def test_the_scatter_layout_needs_both_sizes():
    """A row is a source there: ``n_source`` is the rows, and ``n_target``
    cannot be inferred (the last target need not be used)."""
    indices, counts = _pattern(n_rows=N_SOURCE, n_columns=N_TARGET)
    weights = jnp.asarray(_weights(indices, counts, "float32"))
    with pytest.raises(ValueError, match="the scatter layout needs n_target"):
        StaticSparseMapping(indices, weights, n_source=N_SOURCE, counts=counts,
                            layout="scatter")
    with pytest.raises(ValueError, match="n_source=6 does not match the 7 rows"):
        StaticSparseMapping(indices, weights, n_source=N_SOURCE - 1, n_target=N_TARGET,
                            counts=counts, layout="scatter")
    with pytest.raises(ValueError, match=r"is outside \[0, n_target\) = \[0, 3\)"):
        StaticSparseMapping(indices, weights, n_source=N_SOURCE, n_target=3,
                            counts=counts, layout="scatter")
    m = StaticSparseMapping(indices, weights, n_source=N_SOURCE, n_target=N_TARGET + 2,
                            counts=counts, layout="scatter")
    assert (m.n_target, m.n_source) == (N_TARGET + 2, N_SOURCE)
    assert m.apply(_field(N_SOURCE, "float32")).shape == (N_TARGET + 2,)


def test_a_traced_index_or_traced_weights_are_refused():
    indices, counts = _pattern()
    weights = jnp.asarray(_weights(indices, counts, "float32"))

    def with_traced_weights(w):
        return StaticSparseMapping(indices, w, n_source=N_SOURCE, counts=counts).weights

    def with_traced_index(i):
        return StaticSparseMapping(i, weights, n_source=N_SOURCE, counts=counts).weights

    with pytest.raises(ValueError, match="weights is a traced value"):
        jax.jit(with_traced_weights)(weights)
    with pytest.raises(ValueError, match="indices is a traced value"):
        jax.jit(with_traced_index)(jnp.asarray(indices))


# ---------------------------------------------------------------------------
# The index is structure
# ---------------------------------------------------------------------------

def test_the_index_is_frozen_and_is_the_mappings_own_copy():
    indices, counts = _pattern()
    m = StaticSparseMapping(indices, jnp.asarray(_weights(indices, counts, "float32")),
                            n_source=N_SOURCE, counts=counts)
    assert isinstance(m.indices, np.ndarray) and m.indices.dtype == np.int32
    assert m.indices.flags.c_contiguous and not m.indices.flags.writeable
    assert m.counts.dtype == np.int32 and not m.counts.flags.writeable
    before = m.indices.copy()
    indices[:] = 0
    counts[:] = 0
    np.testing.assert_array_equal(m.indices, before)
    with pytest.raises(ValueError, match="read-only"):
        m.indices[0, 0] = 1
    with pytest.raises(ValueError, match="read-only"):
        m.counts[0] = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.indices = before
    # a copy would have a writable index under a step that baked this one
    assert copy.copy(m) is m and copy.deepcopy(m) is m


def test_the_parameter_tree_holds_one_floating_point_leaf_and_no_index():
    """What the registry asks of any mapping class: a plain dict from an
    identifier to a concrete floating-point JAX array, the same on every
    call.  The index, the counts and the sizes stay on the mapping."""
    for layout in ("gather", "scatter"):
        m = _mapping(layout)
        tree = m.params_pytree()
        assert type(tree) is dict and list(tree) == ["W"]
        assert isinstance(tree["W"], jax.Array) and tree["W"] is m.weights
        assert jnp.issubdtype(tree["W"].dtype, jnp.floating)
        assert tree["W"].shape == m.indices.shape
        assert _params_contract_problem(m) is None
        assert isinstance(m, Mapping) and m.needs_geometry is False
        leaves = jax.tree.leaves(tree)
        assert len(leaves) == 1 and all(not jnp.issubdtype(leaf.dtype, jnp.integer)
                                        for leaf in leaves)


class _Vec(SimulationNode):
    """``v <- v + dt * inp``, from ``1 .. n`` or from zero."""

    def __init__(self, name, timestep, n=3, zero=False):
        super().__init__(name, timestep, n=n, zero=zero)

    def initial_state(self):
        start = jnp.arange(1, self.params["n"] + 1, dtype=jnp.float32)
        return {"v": jnp.zeros_like(start) if self.params["zero"] else start}

    def update(self, s, bi, dt):
        return {"v": s["v"] + dt * bi.get("inp", jnp.zeros_like(s["v"]))}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=(self.params["n"],), description="i")}


@pytest.mark.parametrize("layout", ["gather", "scatter"])
def test_an_edge_takes_a_sparse_mapping_and_the_step_reads_its_live_weights(layout):
    m = _mapping(layout)
    assert hash(EdgeSpec("a", "b", "v", "inp", mapping=m)) is not None
    gm = GraphManager()
    gm.add_node(_Vec("a", 1.0, n=N_SOURCE))
    gm.add_node(_Vec("b", 1.0, n=N_TARGET, zero=True))
    gm.add_edge("a", "b", "v", "inp", mapping=m)
    gm.compile()
    key = "a.v->b.inp"
    assert list(gm.params["mappings"][key]) == ["W"]
    assert gm.params["mappings"][key]["W"].dtype == jnp.float32
    source = np.arange(1, N_SOURCE + 1, dtype=np.float64)
    H = densify(m)
    # b starts at zero and adds its mapped input once: the step is the mapping
    first = np.asarray(gm.step()["b"]["v"])
    assert np.abs(first).max() > 0.0
    assert_within_rows(first, m, source)
    np.testing.assert_allclose(first, H @ source, rtol=1e-5, atol=1e-5)
    gm.reset_state()
    tripled = 3.0 * gm.params["mappings"][key]["W"]
    gm.params["mappings"][key]["W"] = tripled
    moved = np.asarray(gm.step()["b"]["v"])
    assert_within_rows(moved, m, source, weights=tripled)
    with pytest.raises(ValueError, match="does not match"):
        gm.add_edge("b", "a", "v", "inp", mapping=m)
    # a write may replace the weights, never the shape of the structure
    gm.params["mappings"][key]["W"] = jnp.ones((1, m.k))
    with pytest.raises(ValueError, match=rf"has shape \(1, {m.k}\), expected"):
        gm.step()


def test_describe_adds_no_key_a_reload_would_read_as_a_hyper_parameter():
    """``MappingSpec.from_dict`` reads every key but ``kind``, ``points``
    and ``shape`` as a hyper-parameter: ``describe()`` writes the dense
    form and nothing else, so each kind reads back as its own spec."""
    hand = _mapping()
    assert hand.describe() == {"kind": "sparse_matrix", "mode": "consistent",
                               "shape": [N_TARGET, N_SOURCE]}
    points = interface_points(6, 9)
    for case in CASES.values():
        m = case.build(points, {name: f"{name}.npy" for name in case.assets(points)})
        described = m.describe()
        assert described["kind"] == case.kind and described["mode"] == case.mode
        assert described["shape"] == [m.n_target, m.n_source]
        assert set(described) == {"kind", "mode", "shape", "points",
                                  *m.spec.hyperparameters}
        assert MappingSpec.from_dict(described) == m.spec
        assert "W" not in described and "k" not in described and "layout" not in described


def test_the_sizes_the_text_and_the_inspection_line():
    m = _mapping()
    assert (m.n_target, m.n_source, m.k) == (N_TARGET, N_SOURCE, K)
    assert m.nnz == int(np.asarray(m.counts).sum())
    assert repr(m) == f"StaticSparseMapping(sparse_matrix, consistent, 5x7, k={K})"
    assert _mapping_text(m) == f"StaticSparseMapping 7->5 (k={K}, nnz={m.nnz})"
    s = _mapping("scatter")
    assert repr(s).endswith(f"k={K}, scatter)")
    full = StaticSparseMapping(np.zeros((3, 2), np.int64), jnp.ones((3, 2)), n_source=4)
    assert full.counts is None and full.nnz == 6
    # the dense mapping's line is what it was
    assert _mapping_text(StaticLinearMapping(jnp.eye(3))) == "StaticLinearMapping 3->3"


# ---------------------------------------------------------------------------
# The structure digest
# ---------------------------------------------------------------------------

def _digest_of(**changes) -> str:
    return _good(**changes).structure_digest()


def test_the_digest_is_of_the_pattern_and_of_nothing_else():
    indices, counts = _pattern()
    base = _digest_of()
    assert len(base) == 64 and int(base, 16) >= 0
    assert _digest_of() == base                                   # a function of its inputs
    # the weights, the kind, the mode and the description do not enter
    assert _digest_of(weights=jnp.asarray(_weights(indices, counts, "float32", seed=7))) == base
    assert _digest_of(mode="conservative", kind="other", meta={"a": 1}) == base
    with x64(True):
        assert _digest_of(weights=jnp.asarray(_weights(indices, counts, "float64"))) == base
    # the index dtype the caller used does not enter either
    assert _digest_of(indices=indices.astype(np.int16), counts=counts.astype(np.uint8)) == base


def test_the_digest_changes_with_every_part_of_the_structure():
    indices, counts = _pattern()
    base = _digest_of()
    seen = {base}

    moved = indices.copy()
    moved[0, 0] = (moved[0, 0] + 1) % N_SOURCE
    swapped = indices.copy()
    swapped[0, [2, 3]] = swapped[0, [3, 2]]
    assert swapped[0, 2] != swapped[0, 3], "premise: the swap changes the row"
    shorter = counts.copy()
    shorter[0] -= 1
    trimmed = indices.copy()
    trimmed[0, K - 1] = 0
    weights = _weights(indices, counts, "float32")
    cut = weights.copy()
    cut[0, K - 1] = 0.0
    variants = {
        "an index moved": dict(indices=moved),
        "two slots swapped": dict(indices=swapped),
        "a larger n_source": dict(n_source=N_SOURCE + 1),
        "a row shortened": dict(indices=trimmed, counts=shorter, weights=jnp.asarray(cut)),
    }
    for name, changes in variants.items():
        digest = _digest_of(**changes)
        assert digest not in seen, name
        seen.add(digest)

    # the same rows read the other way are another operator
    square = np.array([[0, 1], [1, 0], [2, 2]])
    w = jnp.ones((3, 2))
    gather = StaticSparseMapping(square, w, n_source=3)
    scatter = StaticSparseMapping(square, w, n_source=3, n_target=3, layout="scatter")
    wider = StaticSparseMapping(square, w, n_source=3, n_target=4, layout="scatter")
    assert len({gather.structure_digest(), scatter.structure_digest(),
                wider.structure_digest()}) == 3


def test_counts_that_fill_every_row_are_the_structure_without_counts():
    indices = np.array([[0, 1], [2, 0]])
    w = jnp.ones((2, 2))
    full = StaticSparseMapping(indices, w, n_source=3)
    counted = StaticSparseMapping(indices, w, n_source=3, counts=[2, 2])
    assert counted.counts is None
    assert counted.structure_digest() == full.structure_digest()
    padded = StaticSparseMapping(np.array([[0, 1], [2, 0]]), jnp.asarray([[1.0, 1.0], [1.0, 0.0]]),
                                 n_source=3, counts=[2, 1])
    assert padded.structure_digest() != full.structure_digest()


# ---------------------------------------------------------------------------
# Reproducibility on the CPU, as measured
# ---------------------------------------------------------------------------

def _twelve_decades(shape, seed):
    """Values spanning twelve decades with random signs: a sum of them
    shows the order it was taken in."""
    rng = np.random.default_rng(seed)
    return (rng.choice([-1.0, 1.0], size=shape) * 10.0 ** rng.uniform(-6, 6, size=shape)
            ).astype(np.float32)


def test_a_scatter_add_with_repeated_indices_is_the_in_order_sum_on_the_cpu():
    """Measured on the CPU: ``zeros.at[index].add(terms)`` with repeated
    indices is the sequential float32 sum in the order of the index array
    (row-major), bit for bit, and the same on every call.  The data are
    chosen so that another order gives other bits.  Not claimed on a GPU,
    where the same program gave a different result on every run."""
    n_rows, k, n_columns = 4000, 8, 97
    rng = np.random.default_rng(11)
    indices = rng.integers(0, n_columns, size=(n_rows, k))
    weights = _twelve_decades((n_rows, k), 12)
    field = _twelve_decades(n_rows, 13)
    m = StaticSparseMapping(indices, jnp.asarray(weights), n_source=n_rows,
                            n_target=n_columns, layout="scatter")
    in_order = np.zeros(n_columns, np.float32)
    np.add.at(in_order, indices.ravel(), (weights * field[:, None]).ravel())
    reversed_order = np.zeros(n_columns, np.float32)
    np.add.at(reversed_order, indices.ravel()[::-1], (weights * field[:, None]).ravel()[::-1])
    assert in_order.tobytes() != reversed_order.tobytes(), "premise: the order shows"

    apply = jax.jit(m.apply)
    results = {np.asarray(apply(jnp.asarray(field))).tobytes() for _ in range(10)}
    assert results == {in_order.tobytes()}
    # the transpose of the gather layout is the same scatter
    g = StaticSparseMapping(indices, jnp.asarray(weights), n_source=n_columns)
    assert np.asarray(jax.jit(g.apply_T)(jnp.asarray(field))).tobytes() == in_order.tobytes()


def test_the_row_sum_of_one_compiled_program_is_one_result():
    """A gather-and-sum reduces every row on its own: one compiled program
    returns the same bits on every call."""
    n_rows, k, n_columns = 4000, 8, 97
    rng = np.random.default_rng(21)
    indices = rng.integers(0, n_columns, size=(n_rows, k))
    m = StaticSparseMapping(indices, jnp.asarray(_twelve_decades((n_rows, k), 22)),
                            n_source=n_columns)
    field = jnp.asarray(_twelve_decades(n_columns, 23))
    apply = jax.jit(m.apply)
    assert len({np.asarray(apply(field)).tobytes() for _ in range(10)}) == 1
    assert_within_rows(apply(field), m, field)


# ---------------------------------------------------------------------------
# sparse_matrix builds the class from rows as a user writes them
# ---------------------------------------------------------------------------

def test_sparse_matrix_rows_become_the_structure_the_class_describes():
    indices = np.array([[2, -1, 0], [-1, -1, 4], [1, 1, 3], [-1, -1, -1]])
    values = np.array([[1.0, 0.0, 2.0], [0.0, -0.0, 3.0], [4.0, 5.0, 6.0], [0.0, 0.0, 0.0]],
                      np.float32)
    m = sparse_matrix_mapping(indices, values, n_source=5)
    np.testing.assert_array_equal(m.indices, [[2, 0, 0], [4, 0, 0], [1, 1, 3], [0, 0, 0]])
    np.testing.assert_array_equal(m.counts, [2, 1, 3, 0])
    np.testing.assert_array_equal(np.asarray(m.weights),
                                  [[1, 2, 0], [3, 0, 0], [4, 5, 6], [0, 0, 0]])
    assert not np.signbit(np.asarray(m.weights)).any()
    expected = np.zeros((4, 5), np.float32)
    expected[0, 2], expected[0, 0], expected[1, 4] = 1.0, 2.0, 3.0
    expected[2, 1], expected[2, 3] = 9.0, 6.0
    np.testing.assert_array_equal(densify(m), expected)
