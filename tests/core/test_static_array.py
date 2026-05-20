"""Tests for v0.2 #3 follow-up: StaticArray typed wrapper.

Pins:
  - The dataclass contract (replication / shard_axis rules).
  - Nested-structure rejection at construction.
  - Bare-array coercion path emits a FutureWarning.
  - The static_data_hash() contract: array values hash by
    (shape, dtype, replication, shard_axis); scalars by repr.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.node import SimulationNode
from maddening.core.static_data import StaticArray, coerce_static_data_value


# ---------------------------------------------------------------------------
# Dataclass construction rules
# ---------------------------------------------------------------------------


class TestStaticArrayConstruction:
    def test_replicate_default(self):
        a = StaticArray(jnp.zeros(4))
        assert a.replication == "replicate"
        assert a.shard_axis is None

    def test_shard_with_axis(self):
        a = StaticArray(jnp.zeros((4, 5)), replication="shard", shard_axis=1)
        assert a.replication == "shard"
        assert a.shard_axis == 1

    def test_shard_requires_axis(self):
        with pytest.raises(ValueError, match="requires an explicit shard_axis"):
            StaticArray(jnp.zeros(4), replication="shard")

    def test_replicate_forbids_axis(self):
        with pytest.raises(ValueError, match="shard_axis must be None"):
            StaticArray(jnp.zeros(4), replication="replicate", shard_axis=0)

    def test_unknown_replication_rejected(self):
        with pytest.raises(ValueError, match="must be 'replicate' or 'shard'"):
            StaticArray(jnp.zeros(4), replication="broadcast")  # type: ignore

    def test_shard_axis_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            StaticArray(jnp.zeros((4,)), replication="shard", shard_axis=2)

    def test_negative_shard_axis_rejected(self):
        # Negative axis is also out-of-range; we don't support reverse indexing.
        with pytest.raises(ValueError, match="out of range"):
            StaticArray(jnp.zeros((4, 5)), replication="shard", shard_axis=-1)


# ---------------------------------------------------------------------------
# Nested-structure rejection
# ---------------------------------------------------------------------------


class TestStaticArrayRejectsNested:
    def test_list_of_arrays_rejected(self):
        with pytest.raises(TypeError, match="nested|unfold|array-like"):
            StaticArray([jnp.zeros(3), jnp.ones(3)])

    def test_tuple_of_arrays_rejected(self):
        with pytest.raises(TypeError, match="nested|unfold|array-like"):
            StaticArray((jnp.zeros(3), jnp.ones(3)))

    def test_dict_of_arrays_rejected(self):
        with pytest.raises(TypeError, match="nested|unfold|array-like"):
            StaticArray({"a": jnp.zeros(3)})

    def test_set_rejected(self):
        with pytest.raises(TypeError, match="nested|unfold|array-like"):
            StaticArray(set())  # type: ignore

    def test_plain_python_int_rejected(self):
        with pytest.raises(TypeError, match="array-like"):
            StaticArray(42)  # type: ignore

    def test_plain_python_string_rejected(self):
        with pytest.raises(TypeError, match="array-like"):
            StaticArray("hello")  # type: ignore


# ---------------------------------------------------------------------------
# Shape / dtype proxies
# ---------------------------------------------------------------------------


class TestStaticArrayProxies:
    def test_shape_proxy(self):
        a = StaticArray(jnp.zeros((4, 5)))
        assert a.shape == (4, 5)

    def test_dtype_proxy(self):
        a = StaticArray(jnp.zeros(3, dtype=jnp.int32))
        assert str(a.dtype) == "int32"

    def test_value_accessible(self):
        x = jnp.arange(4)
        a = StaticArray(x)
        # Same object (no copy)
        assert a.value is x


# ---------------------------------------------------------------------------
# Bare-array coercion path
# ---------------------------------------------------------------------------


class TestCoercion:
    def test_bare_array_warns_and_coerces(self):
        with pytest.warns(FutureWarning, match="StaticArray"):
            out = coerce_static_data_value(
                jnp.zeros(3), node_name="x", key="k",
            )
        assert isinstance(out, StaticArray)
        assert out.replication == "replicate"

    def test_already_wrapped_passes_through(self):
        wrapped = StaticArray(jnp.zeros(3))
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            out = coerce_static_data_value(
                wrapped, node_name="x", key="k",
            )
        assert out is wrapped

    def test_scalars_pass_through_unchanged(self):
        for v in (42, 3.14, "hello", (1, 2, 3)):
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                out = coerce_static_data_value(v, node_name="x", key="k")
            assert out == v


# ---------------------------------------------------------------------------
# static_data_hash contract: shape+dtype+replication+shard_axis
# ---------------------------------------------------------------------------


class _Node(SimulationNode):
    def __init__(self, name, timestep, sd):
        super().__init__(name, timestep)
        self._sd = sd

    @property
    def static_data(self):
        return self._sd

    def initial_state(self):
        return {"x": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt):
        return state


class TestHashIncludesShardingPolicy:
    def test_same_array_different_replication_distinct_hash(self):
        a = StaticArray(jnp.zeros(4), replication="replicate")
        b = StaticArray(jnp.zeros(4), replication="shard", shard_axis=0)
        n1 = _Node("a", timestep=0.01, sd={"k": a})
        n2 = _Node("b", timestep=0.01, sd={"k": b})
        assert n1.static_data_hash() != n2.static_data_hash()

    def test_same_array_different_shard_axis_distinct_hash(self):
        a = StaticArray(jnp.zeros((4, 5)), replication="shard", shard_axis=0)
        b = StaticArray(jnp.zeros((4, 5)), replication="shard", shard_axis=1)
        n1 = _Node("a", timestep=0.01, sd={"k": a})
        n2 = _Node("b", timestep=0.01, sd={"k": b})
        assert n1.static_data_hash() != n2.static_data_hash()

    def test_same_policy_same_shape_same_hash(self):
        # Same hashing inputs → identical hash, even if the *values*
        # differ.  The shape+dtype+policy is the key, not the contents.
        n1 = _Node("a", timestep=0.01,
                   sd={"k": StaticArray(jnp.zeros(4))})
        n2 = _Node("b", timestep=0.01,
                   sd={"k": StaticArray(jnp.ones(4))})
        assert n1.static_data_hash() == n2.static_data_hash()

    def test_bare_array_hash_matches_replicate_wrapping(self):
        # The coercion path treats a bare array as
        # StaticArray(value=arr, replication="replicate"), so the
        # hashes match — but the bare path emits FutureWarning.
        bare_node = _Node("a", timestep=0.01, sd={"k": jnp.zeros(4)})
        wrapped_node = _Node(
            "b", timestep=0.01,
            sd={"k": StaticArray(jnp.zeros(4), replication="replicate")},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            h_bare = bare_node.static_data_hash()
        assert h_bare == wrapped_node.static_data_hash()
