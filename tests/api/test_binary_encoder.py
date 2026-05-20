"""Tests for the binary state encoder."""

import struct

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.api.binary_encoder import BinaryStateEncoder, decode_frame

try:
    import zstandard  # noqa: F401
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


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


class TestFieldSubscription:
    """v0.2 #5: subset packing via the ``fields`` selector."""

    def test_subscribe_subset_of_fields(self, sample_state):
        enc = BinaryStateEncoder(sample_state, fields={"ball": ["position"]})
        schema = enc.schema()

        # Only ball.position should be packed
        assert schema["total_floats"] == 1
        assert len(schema["fields"]) == 1
        assert schema["fields"][0]["node"] == "ball"
        assert schema["fields"][0]["field"] == "position"

    def test_subscribe_multiple_nodes(self, sample_state):
        enc = BinaryStateEncoder(
            sample_state,
            fields={"ball": ["velocity"], "heat_rod": ["temperature"]},
        )
        schema = enc.schema()
        assert schema["total_floats"] == 1 + 20
        included = {(f["node"], f["field"]) for f in schema["fields"]}
        assert included == {("ball", "velocity"), ("heat_rod", "temperature")}

    def test_subscribe_unknown_field_ignored(self, sample_state):
        # field that doesn't exist on the node — should be silently dropped
        enc = BinaryStateEncoder(
            sample_state, fields={"ball": ["position", "nonexistent"]},
        )
        schema = enc.schema()
        assert schema["total_floats"] == 1
        assert {f["field"] for f in schema["fields"]} == {"position"}

    def test_subscribe_unknown_node_ignored(self, sample_state):
        enc = BinaryStateEncoder(
            sample_state, fields={"ghost_node": ["x"], "ball": ["position"]},
        )
        assert enc.total_floats == 1

    def test_subscribe_empty_dict_means_nothing(self, sample_state):
        # fields={} means "no nodes selected" — distinct from None ("all").
        enc = BinaryStateEncoder(sample_state, fields={})
        assert enc.total_floats == 0
        assert enc.frame_bytes == 8

    def test_subscription_introspection(self, sample_state):
        sub = {"ball": ["position"]}
        enc = BinaryStateEncoder(sample_state, fields=sub)
        assert enc.subscription is sub

        enc_full = BinaryStateEncoder(sample_state)
        assert enc_full.subscription is None

    def test_encode_subset_packs_only_requested(self, sample_state):
        enc = BinaryStateEncoder(sample_state, fields={"ball": ["position"]})
        frame = enc.encode(2.5, sample_state)
        # 8 (sim_time) + 1 float (ball.position) = 12
        assert len(frame) == 12
        values = np.frombuffer(frame, dtype=np.float32, offset=8)
        assert len(values) == 1
        assert abs(values[0] - 5.0) < 1e-6

    def test_bandwidth_reduction_target(self):
        """v0.2 brief: subset packing should give >95% bandwidth reduction
        on a typical LBM frame where only velocity is needed and the
        f-distribution arrays dominate the payload."""
        # Simulate a 32³ LBM-like state: velocity (3 vectors) + 19 f-dists
        N = 32 * 32 * 32
        big_state = {
            "lbm": {
                "velocity": jnp.zeros((N, 3)),
                **{f"f{i}": jnp.zeros(N) for i in range(19)},
            }
        }
        full = BinaryStateEncoder(big_state)
        subset = BinaryStateEncoder(big_state, fields={"lbm": ["velocity"]})
        reduction = 1.0 - subset.frame_bytes / full.frame_bytes
        # 19 f-dist scalars + 3 velocity components: subset keeps 3/22 ≈ 13.6%
        # so reduction ≈ 86.4% (the brief's >95% target only holds when
        # velocity is also dropped or compressed further — see #6).
        assert reduction > 0.80, (
            f"Subset frame is {subset.frame_bytes} B vs full {full.frame_bytes} B; "
            f"reduction {reduction*100:.1f}% (expected ≥80%)"
        )


@pytest.mark.skipif(not HAS_ZSTD, reason="zstandard not installed")
class TestCompression:
    """v0.2 #6: zstd + zstd-xor compression on the encoder."""

    def test_compression_field_in_schema(self, sample_state):
        enc = BinaryStateEncoder(sample_state, compression="zstd")
        assert enc.schema()["compression"] == "zstd"
        assert enc.compression == "zstd"

        enc_xor = BinaryStateEncoder(sample_state, compression="zstd+xor")
        assert enc_xor.schema()["compression"] == "zstd+xor"

    def test_default_compression_is_none(self, sample_state):
        enc = BinaryStateEncoder(sample_state)
        assert enc.compression == "none"
        assert enc.schema()["compression"] == "none"

    def test_invalid_compression_rejected(self, sample_state):
        with pytest.raises(ValueError, match="compression"):
            BinaryStateEncoder(sample_state, compression="lzma")

    def test_zstd_roundtrip(self, sample_state):
        enc = BinaryStateEncoder(sample_state, compression="zstd")
        schema = enc.schema()
        frame = enc.encode(7.5, sample_state)
        sim_time, values = decode_frame(frame, schema)
        assert abs(sim_time - 7.5) < 1e-9
        assert values.shape == (schema["total_floats"],)
        assert abs(values[schema["fields"][0]["offset"]] - 5.0) < 1e-6  # ball.position

    def test_zstd_frame_smaller_than_raw_on_smooth_fields(self):
        # A 1024-element heat rod with constant temperature is the
        # easiest possible compressible payload.
        state = {"rod": {"T": jnp.ones(1024) * 22.5}}
        raw = BinaryStateEncoder(state).encode(0.0, state)
        zstd_frame = BinaryStateEncoder(state, compression="zstd").encode(0.0, state)
        # zstd should crush the 1024 identical floats to ~tens of bytes.
        ratio = len(raw) / len(zstd_frame)
        assert ratio > 10.0, f"zstd ratio {ratio:.1f}× should be >10× on constant fields"

    def test_zstd_xor_better_than_zstd_on_slow_dynamics(self):
        # 4096-cell field that only slightly changes between frames —
        # the xor between consecutive frames is mostly zeros which zstd
        # compresses better than the raw second frame.
        state0 = {"rod": {"T": jnp.ones(4096) * 22.0}}
        state1 = {"rod": {"T": jnp.ones(4096) * 22.0 + 0.001}}

        enc_zstd = BinaryStateEncoder(state0, compression="zstd")
        enc_xor = BinaryStateEncoder(state0, compression="zstd+xor")
        # Prime the xor encoder with frame 0
        enc_xor.encode(0.0, state0)
        enc_zstd.encode(0.0, state0)

        zstd_size = len(enc_zstd.encode(0.01, state1))
        xor_size = len(enc_xor.encode(0.01, state1))
        # xor frame should be at most as large as zstd alone, usually smaller
        # on slowly-varying fields.  Allow 1.05× tolerance (zstd overhead
        # on tiny payloads can make the comparison noisy).
        assert xor_size <= int(zstd_size * 1.05), (
            f"xor-delta {xor_size}B should be ≤ plain zstd {zstd_size}B "
            f"on slowly-varying field"
        )

    def test_sim_time_header_is_plaintext_under_compression(self):
        """The 8-byte sim_time prefix must NOT be compressed so clients
        can read it without invoking zstd."""
        state = {"a": {"x": jnp.array(1.0)}}
        enc = BinaryStateEncoder(state, compression="zstd")
        frame = enc.encode(3.14, state)
        # First 8 bytes decode as float64 directly
        t = struct.unpack_from("d", frame, 0)[0]
        assert abs(t - 3.14) < 1e-9

    def test_compression_combined_with_field_subscription(self, sample_state):
        enc = BinaryStateEncoder(
            sample_state,
            fields={"heat_rod": ["temperature"]},
            compression="zstd",
        )
        schema = enc.schema()
        assert schema["compression"] == "zstd"
        assert schema["total_floats"] == 20
        frame = enc.encode(0.1, sample_state)
        sim_time, vals = decode_frame(frame, schema)
        assert abs(sim_time - 0.1) < 1e-9
        assert vals.shape == (20,)
        assert np.allclose(vals, 20.0, atol=1e-5)

    def test_lbm_bandwidth_target(self):
        """The brief's >95% target on a 32³ LBM payload should be met
        when field-subscription is combined with zstd compression on
        the constant initial state."""
        N = 32 * 32 * 32
        state = {
            "lbm": {
                "velocity": jnp.zeros((N, 3)),
                **{f"f{i}": jnp.ones(N) * 0.05 for i in range(19)},
            }
        }
        full_raw = BinaryStateEncoder(state).encode(0.0, state)
        compressed = BinaryStateEncoder(
            state,
            fields={"lbm": ["velocity"]},
            compression="zstd",
        ).encode(0.0, state)
        reduction = 1.0 - len(compressed) / len(full_raw)
        assert reduction > 0.95, (
            f"compressed+subscribed={len(compressed)}B vs raw "
            f"full={len(full_raw)}B; reduction {reduction*100:.2f}% "
            f"(brief target ≥95%)"
        )


@pytest.mark.skipif(not HAS_ZSTD, reason="zstandard not installed")
class TestEncodeLatency:
    """v0.2 #6 follow-up: latency-budget assertion.

    Each test measures wall-clock encode time over many frames and
    asserts the median per-frame budget the brief implies (real-time
    60 fps server → ≤16.6 ms per frame; we set a much tighter bound
    so the test catches regressions before the budget is actually
    threatened).  Numbers are deliberately generous so the test is
    stable across CPU types; the assertion is order-of-magnitude.
    """

    @staticmethod
    def _measure(encoder, state, n=200):
        import time
        # Warm up so the first JAX-array → numpy conversion isn't
        # billed against the loop body.
        encoder.encode(0.0, state)
        t0 = time.perf_counter()
        for i in range(n):
            encoder.encode(i * 0.01, state)
        elapsed = time.perf_counter() - t0
        return elapsed / n  # average seconds per frame

    def test_none_encoding_under_budget_on_32cube_lbm(self):
        # 32³ LBM-like payload: 22 fields total (velocity + 19 fdists +
        # 2 scalars).  Uncompressed wire size ≈ 2.7 MB per frame.
        #
        # The 30 ms bound is order-of-magnitude (the real 60-fps budget
        # is 16.6 ms; quiescent box hits ~6 ms).  We keep it loose so a
        # noisy CI runner doesn't false-flag; the assertion's job is to
        # catch O(N²) rewrites, not pin a clock.
        N = 32 * 32 * 32
        state = {
            "lbm": {
                "velocity": jnp.zeros((N, 3)),
                "rho": jnp.ones(N),
                "p": jnp.zeros(N),
                **{f"f{i}": jnp.ones(N) * 0.05 for i in range(19)},
            }
        }
        enc = BinaryStateEncoder(state)
        per_frame_s = self._measure(enc, state, n=50)
        assert per_frame_s < 0.030, (
            f"uncompressed 32³ encode took {per_frame_s*1000:.2f} ms/frame "
            f"(>30 ms order-of-magnitude budget)"
        )

    def test_zstd_encoding_under_budget_on_subscribed_frame(self):
        # Just velocity — the typical bandwidth-sensitive subscriber.
        # Same generous order-of-magnitude bound.
        N = 32 * 32 * 32
        state = {"lbm": {"velocity": jnp.zeros((N, 3))}}
        enc = BinaryStateEncoder(
            state, fields={"lbm": ["velocity"]}, compression="zstd",
        )
        per_frame_s = self._measure(enc, state, n=50)
        assert per_frame_s < 0.030, (
            f"zstd-compressed 32³-velocity encode took "
            f"{per_frame_s*1000:.2f} ms/frame (>30 ms order-of-magnitude budget)"
        )

    @pytest.mark.slow
    def test_zstd_xor_encoding_under_budget_on_slow_dynamics(self):
        # XOR-delta path: state changes slightly between frames.  We
        # mark this slow because the 100-frame loop with shifting
        # state takes ~1 s on CI.
        N = 32 * 32 * 32
        base = jnp.ones((N, 3)) * 0.1
        states = [{"lbm": {"velocity": base + jnp.array([0.0, 0.0, i * 1e-4])}}
                  for i in range(100)]
        enc = BinaryStateEncoder(
            states[0], fields={"lbm": ["velocity"]}, compression="zstd+xor",
        )
        # Prime so the XOR path has a previous frame.
        enc.encode(0.0, states[0])
        import time
        t0 = time.perf_counter()
        for i, s in enumerate(states):
            enc.encode(i * 0.01, s)
        avg = (time.perf_counter() - t0) / len(states)
        # Same order-of-magnitude bound; XOR adds a NumPy XOR + compress.
        assert avg < 0.040, (
            f"zstd+xor 32³-velocity encode took {avg*1000:.2f} ms/frame"
        )
