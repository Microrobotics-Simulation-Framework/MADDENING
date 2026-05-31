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
        with pytest.raises(
            ValueError,
            match="must be 'replicate', 'shard', or 'partition'",
        ):
            StaticArray(jnp.zeros(4), replication="broadcast")  # type: ignore

    def test_shard_axis_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            StaticArray(jnp.zeros((4,)), replication="shard", shard_axis=2)

    def test_negative_shard_axis_rejected(self):
        # Negative axis is also out-of-range; we don't support reverse indexing.
        with pytest.raises(ValueError, match="out of range"):
            StaticArray(jnp.zeros((4, 5)), replication="shard", shard_axis=-1)


class TestStaticArrayPartitionVariant:
    """The replication='partition' variant added in v0.3.0 (§A6)."""

    def test_partition_requires_assignment(self):
        with pytest.raises(
            ValueError, match="requires partition_assignment",
        ):
            StaticArray(jnp.zeros(4), replication="partition")

    def test_partition_assignment_must_be_array_like(self):
        with pytest.raises(TypeError, match="array-like"):
            StaticArray(
                jnp.zeros(4), replication="partition",
                partition_assignment="not-an-array",  # type: ignore[arg-type]
            )

    def test_partition_assignment_must_be_1d(self):
        with pytest.raises(ValueError, match="must be 1-D"):
            StaticArray(
                jnp.zeros((4, 2)), replication="partition",
                partition_assignment=jnp.zeros((4, 2), dtype=jnp.int32),
            )

    def test_partition_assignment_must_be_integer(self):
        with pytest.raises(TypeError, match="integer dtype"):
            StaticArray(
                jnp.zeros(4), replication="partition",
                partition_assignment=jnp.zeros(4, dtype=jnp.float32),
            )

    def test_partition_assignment_length_must_match_value(self):
        with pytest.raises(ValueError, match="must equal value.shape"):
            StaticArray(
                jnp.zeros(4), replication="partition",
                partition_assignment=jnp.zeros(8, dtype=jnp.int32),
            )

    def test_partition_with_shard_axis_nonzero_rejected(self):
        with pytest.raises(ValueError, match="shard_axis must be None or 0"):
            StaticArray(
                jnp.zeros((4, 5)),
                replication="partition",
                partition_assignment=jnp.zeros(4, dtype=jnp.int32),
                shard_axis=1,
            )

    def test_replicate_with_assignment_rejected(self):
        with pytest.raises(
            ValueError, match="partition_assignment must be None",
        ):
            StaticArray(
                jnp.zeros(4),
                partition_assignment=jnp.zeros(4, dtype=jnp.int32),
            )

    def test_shard_with_assignment_rejected(self):
        with pytest.raises(
            ValueError, match="partition_assignment must be None",
        ):
            StaticArray(
                jnp.zeros((4, 5)), replication="shard", shard_axis=0,
                partition_assignment=jnp.zeros(4, dtype=jnp.int32),
            )

    def test_valid_partition_construction(self):
        sa = StaticArray(
            jnp.arange(12, dtype=jnp.float32),
            replication="partition",
            partition_assignment=jnp.arange(12, dtype=jnp.int32) % 4,
        )
        assert sa.replication == "partition"
        assert sa.partition_assignment is not None
        assert sa.partition_assignment.shape == (12,)


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
    def test_bare_array_raises_migration_error(self):
        """v0.3.0 hard-removes the FutureWarning-coerce path; bare arrays
        in static_data now raise MigrationError immediately."""
        from maddening.warnings import MigrationError
        with pytest.raises(MigrationError) as exc:
            coerce_static_data_value(
                jnp.zeros(3), node_name="x", key="k",
            )
        assert "static_data" in exc.value.api_name
        assert exc.value.replacement is not None
        assert "StaticArray" in exc.value.replacement

    def test_already_wrapped_passes_through(self):
        wrapped = StaticArray(jnp.zeros(3))
        out = coerce_static_data_value(
            wrapped, node_name="x", key="k",
        )
        assert out is wrapped

    def test_scalars_pass_through_unchanged(self):
        for v in (42, 3.14, "hello", (1, 2, 3)):
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

    def test_bare_array_hash_raises_migration_error(self):
        """v0.3.0 removed the bare-array coercion path; computing the
        hash on a bare-array static_data dict now raises MigrationError.
        """
        from maddening.warnings import MigrationError
        bare_node = _Node("a", timestep=0.01, sd={"k": jnp.zeros(4)})
        with pytest.raises(MigrationError):
            bare_node.static_data_hash()
