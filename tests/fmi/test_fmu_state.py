"""Tests for the FMU state serialization helpers."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.fmi.fmu_state import (
    FMUState,
    deserialize_fmu_state,
    serialize_fmu_state,
)


class TestRoundTrip:

    def test_basic_round_trip(self):
        state = {
            "ball": {
                "position": jnp.array(1.5, dtype=jnp.float32),
                "velocity": jnp.array(-0.2, dtype=jnp.float32),
            },
            "table": {
                "position": jnp.array(0.0, dtype=jnp.float32),
            },
        }
        snapshot = serialize_fmu_state(
            state=state, schema_token="token-abc-123",
        )
        assert isinstance(snapshot, FMUState)
        assert snapshot.schema_token == "token-abc-123"

        restored = deserialize_fmu_state(
            snapshot, expected_schema_token="token-abc-123",
        )
        assert set(restored) == set(state)
        np.testing.assert_allclose(
            restored["ball"]["position"], 1.5,
        )
        np.testing.assert_allclose(
            restored["ball"]["velocity"], -0.2,
        )

    def test_schema_mismatch_rejected(self):
        state = {"ball": {"position": jnp.array(1.0)}}
        snapshot = serialize_fmu_state(state=state, schema_token="A")
        with pytest.raises(ValueError, match="schema mismatch"):
            deserialize_fmu_state(snapshot, expected_schema_token="B")

    def test_round_trip_preserves_array_shape(self):
        state = {
            "node": {
                "field": jnp.array([[1.0, 2.0], [3.0, 4.0]],
                                   dtype=jnp.float32),
            },
        }
        snapshot = serialize_fmu_state(state=state, schema_token="T")
        restored = deserialize_fmu_state(
            snapshot, expected_schema_token="T",
        )
        np.testing.assert_array_equal(
            np.asarray(restored["node"]["field"]),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )


class TestStabilityTagging:

    def test_serialize_tagged_evolving(self):
        from maddening.core.compliance.metadata import StabilityLevel
        assert serialize_fmu_state._stability_level == \
            StabilityLevel.EVOLVING

    def test_deserialize_tagged_evolving(self):
        from maddening.core.compliance.metadata import StabilityLevel
        assert deserialize_fmu_state._stability_level == \
            StabilityLevel.EVOLVING
