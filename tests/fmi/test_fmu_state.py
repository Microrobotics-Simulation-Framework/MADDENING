"""Tests for the FMU state serialization helpers."""

import io
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


class TestNoPickle:
    """The payload is an ``npz`` of plain arrays, not a pickle.

    ``deserialize_fmu_state`` used to ``pickle.loads`` whatever it was
    given, which made every transport that reached it an arbitrary-code
    door for whoever supplied the bytes.  (Audit params-io 2026-09-19.)
    """

    def test_the_payload_is_an_archive_and_carries_no_pickle_opcodes(self):
        state = {"ball": {"position": jnp.array([1.0, 2.0], jnp.float32)}}
        snapshot = serialize_fmu_state(state=state, schema_token="T")
        assert snapshot.payload[:4] == b"PK\x03\x04"
        with np.load(io.BytesIO(snapshot.payload), allow_pickle=False) as data:
            assert "_paths" in data.files

    def test_params_round_trip_beside_the_state(self):
        state = {"ball": {"position": jnp.array(1.0, jnp.float32)}}
        params = {"nodes": {"ball": {"mass": jnp.array(2.5, jnp.float32)}},
                  "mappings": {}}
        snapshot = serialize_fmu_state(state=state, schema_token="T",
                                       params=params)
        got_state, got_params = deserialize_fmu_state(
            snapshot, expected_schema_token="T", return_params=True)
        np.testing.assert_allclose(got_state["ball"]["position"], 1.0)
        np.testing.assert_allclose(got_params["nodes"]["ball"]["mass"], 2.5)
        assert got_params["mappings"] == {}

    def test_a_pickle_payload_is_refused_rather_than_unpickled(self, tmp_path):
        import pickle

        class _Evil:
            def __reduce__(self):
                return (open, (str(tmp_path / "pwned"), "w"))

        snapshot = FMUState(payload=pickle.dumps(_Evil()), schema_token="T")
        with pytest.raises(ValueError, match="not a MADDENING snapshot"):
            deserialize_fmu_state(snapshot, expected_schema_token="T")
        assert not (tmp_path / "pwned").exists()

    def test_a_leaf_that_is_not_a_plain_array_is_refused_at_snapshot_time(self):
        state = {"ball": {"provider": object()}}
        with pytest.raises(TypeError, match="not a plain array"):
            serialize_fmu_state(state=state, schema_token="T")

    def test_a_non_string_key_is_refused_at_snapshot_time(self):
        state = {"ball": {7: jnp.array(1.0)}}
        with pytest.raises(TypeError, match="must be strings"):
            serialize_fmu_state(state=state, schema_token="T")
