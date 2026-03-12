"""Tests for field subscription filtering in BinaryStateEncoder."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import struct

import numpy as np
import pytest

from maddening.api.binary_encoder import BinaryStateEncoder


# -- Example state used across tests --

def _make_state():
    return {
        "ball": {
            "position": np.array(5.0, dtype=np.float32),
            "velocity": np.array(-1.0, dtype=np.float32),
        },
        "fluid": {
            "density": np.ones((4, 4, 4), dtype=np.float32),
            "velocity": np.zeros((4, 4, 4, 3), dtype=np.float32),
            "tracer": np.ones((4, 4, 4), dtype=np.float32) * 0.5,
        },
    }


class TestFieldSubscriptionEncoder:
    """BinaryStateEncoder with field filtering."""

    def test_full_state_no_filter(self):
        state = _make_state()
        enc = BinaryStateEncoder(state)
        # Should include all fields from both nodes
        schema = enc.schema()
        field_names = [(f["node"], f["field"]) for f in schema["fields"]]
        assert ("ball", "position") in field_names
        assert ("ball", "velocity") in field_names
        assert ("fluid", "density") in field_names
        assert ("fluid", "velocity") in field_names
        assert ("fluid", "tracer") in field_names
        assert enc.subscription is None

    def test_filter_single_node(self):
        state = _make_state()
        enc = BinaryStateEncoder(state, fields={"ball": ["position"]})
        schema = enc.schema()
        field_names = [(f["node"], f["field"]) for f in schema["fields"]]
        assert field_names == [("ball", "position")]
        assert enc.total_floats == 1
        assert enc.subscription == {"ball": ["position"]}

    def test_filter_multiple_fields(self):
        state = _make_state()
        enc = BinaryStateEncoder(
            state, fields={"fluid": ["density", "tracer"]},
        )
        schema = enc.schema()
        field_names = [(f["node"], f["field"]) for f in schema["fields"]]
        assert ("fluid", "density") in field_names
        assert ("fluid", "tracer") in field_names
        assert ("fluid", "velocity") not in field_names
        assert ("ball", "position") not in field_names
        # density=4*4*4=64, tracer=64 → 128 floats
        assert enc.total_floats == 128

    def test_filter_multiple_nodes(self):
        state = _make_state()
        enc = BinaryStateEncoder(
            state,
            fields={"ball": ["position"], "fluid": ["density"]},
        )
        schema = enc.schema()
        field_names = [(f["node"], f["field"]) for f in schema["fields"]]
        assert len(field_names) == 2
        assert ("ball", "position") in field_names
        assert ("fluid", "density") in field_names

    def test_filter_nonexistent_node_ignored(self):
        state = _make_state()
        enc = BinaryStateEncoder(
            state, fields={"nonexistent": ["foo"]},
        )
        assert enc.total_floats == 0

    def test_filter_nonexistent_field_ignored(self):
        state = _make_state()
        enc = BinaryStateEncoder(
            state, fields={"ball": ["nonexistent_field"]},
        )
        assert enc.total_floats == 0

    def test_encode_filtered(self):
        state = _make_state()
        enc = BinaryStateEncoder(
            state, fields={"ball": ["position"]},
        )
        frame = enc.encode(1.5, state)
        # 8 bytes sim_time + 4 bytes (1 float32)
        assert len(frame) == 12
        sim_time = struct.unpack_from("d", frame, 0)[0]
        assert sim_time == pytest.approx(1.5)
        val = struct.unpack_from("f", frame, 8)[0]
        assert val == pytest.approx(5.0)

    def test_encode_filtered_large_field(self):
        state = _make_state()
        enc = BinaryStateEncoder(
            state, fields={"fluid": ["density"]},
        )
        frame = enc.encode(0.0, state)
        # 8 + 64*4 = 264
        assert len(frame) == 264

    def test_bandwidth_reduction(self):
        """Filtered encoder should be much smaller than full encoder."""
        state = _make_state()
        full = BinaryStateEncoder(state)
        filtered = BinaryStateEncoder(
            state, fields={"ball": ["position", "velocity"]},
        )
        # Full includes fluid arrays (64+192+64 = 320 + 2 ball = 322)
        # Filtered: just 2 floats
        assert filtered.frame_bytes < full.frame_bytes
        ratio = filtered.frame_bytes / full.frame_bytes
        assert ratio < 0.05  # >95% reduction
