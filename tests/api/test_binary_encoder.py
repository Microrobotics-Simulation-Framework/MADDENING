"""Tests for the binary state encoder."""

import struct

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.api.binary_encoder import BinaryStateEncoder


@pytest.fixture
def sample_state():
    return {
        "ball": {
            "position": jnp.array(5.0),
            "velocity": jnp.array(-3.2),
        },
        "heat_rod": {
            "temperature": jnp.ones(20) * 20.0,
        },
        "table": {
            "position": jnp.array(0.0),
        },
    }


class TestBinaryStateEncoder:
    def test_schema_structure(self, sample_state):
        enc = BinaryStateEncoder(sample_state)
        schema = enc.schema()

        assert schema["type"] == "schema"
        assert schema["total_floats"] == 2 + 20 + 1  # ball(2) + heat(20) + table(1)
        assert schema["frame_bytes"] == 8 + 23 * 4
        assert len(schema["fields"]) == 4  # ball(2) + heat(1) + table(1) = 4 entries

        # Fields should be sorted by node then field name
        nodes = [f["node"] for f in schema["fields"]]
        assert nodes == sorted(nodes)

    def test_schema_field_offsets(self, sample_state):
        enc = BinaryStateEncoder(sample_state)
        schema = enc.schema()

        offset = 0
        for f in schema["fields"]:
            assert f["offset"] == offset
            offset += f["count"]
        assert offset == schema["total_floats"]

    def test_encode_decode_roundtrip(self, sample_state):
        enc = BinaryStateEncoder(sample_state)
        schema = enc.schema()

        sim_time = 1.234
        frame = enc.encode(sim_time, sample_state)

        # Check frame size
        assert len(frame) == schema["frame_bytes"]

        # Decode sim_time
        t = struct.unpack_from("d", frame, 0)[0]
        assert abs(t - sim_time) < 1e-10

        # Decode values
        values = np.frombuffer(frame, dtype=np.float32, offset=8)
        assert len(values) == schema["total_floats"]

    def test_encode_values_correct(self, sample_state):
        enc = BinaryStateEncoder(sample_state)
        schema = enc.schema()

        frame = enc.encode(0.0, sample_state)
        values = np.frombuffer(frame, dtype=np.float32, offset=8)

        # Check specific values via schema
        for f in schema["fields"]:
            if f["node"] == "ball" and f["field"] == "position":
                assert abs(values[f["offset"]] - 5.0) < 1e-6
            elif f["node"] == "ball" and f["field"] == "velocity":
                assert abs(values[f["offset"]] - (-3.2)) < 1e-5
            elif f["node"] == "table" and f["field"] == "position":
                assert abs(values[f["offset"]] - 0.0) < 1e-6
            elif f["node"] == "heat_rod" and f["field"] == "temperature":
                for i in range(f["count"]):
                    assert abs(values[f["offset"] + i] - 20.0) < 1e-6

    def test_encode_multiple_frames(self, sample_state):
        enc = BinaryStateEncoder(sample_state)

        # Encode the same state at different times
        frames = [enc.encode(t * 0.01, sample_state) for t in range(10)]
        assert len(set(len(f) for f in frames)) == 1  # all same size

        # Different sim_times
        times = [struct.unpack_from("d", f, 0)[0] for f in frames]
        for i, t in enumerate(times):
            assert abs(t - i * 0.01) < 1e-10

    def test_scalar_only_state(self):
        state = {"node": {"x": jnp.array(1.0), "y": jnp.array(2.0)}}
        enc = BinaryStateEncoder(state)
        assert enc.total_floats == 2
        assert enc.frame_bytes == 8 + 2 * 4

    def test_empty_state(self):
        enc = BinaryStateEncoder({})
        assert enc.total_floats == 0
        assert enc.frame_bytes == 8

        frame = enc.encode(0.5, {})
        assert len(frame) == 8
