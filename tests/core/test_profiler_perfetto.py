"""Tests for v0.2 #9 profiler perfetto export + jax.profiler integration."""

from __future__ import annotations

import json
import os
import tarfile
import io
from pathlib import Path

import pytest

from maddening.core.simulation.profiler import (
    JaxProfilerSession,
    ProfileReport,
    jax_trace_active,
    profile_graph,
    profile_report_to_perfetto,
    start_jax_trace,
    stop_jax_trace,
    tar_trace_dir,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_graph():
    """A 2-node bouncing ball graph; cheap enough for profiling tests."""
    from maddening.core.graph_manager import GraphManager
    from maddening.nodes.ball import BallNode
    from maddening.nodes.table import TableNode

    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01, position=0.0))
    gm.add_node(BallNode(
        name="ball", timestep=0.01, initial_position=5.0, elasticity=0.7,
    ))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    return gm


@pytest.fixture
def sample_report():
    """A hand-crafted ProfileReport for output-format tests."""
    return ProfileReport(
        graph_name="test",
        n_steps=10,
        n_nodes=2,
        jit_compile_ms=12.5,
        mean_step_ms=1.25,
        std_step_ms=0.05,
        total_run_ms=12.5,
        steps_per_second=800.0,
        node_times_ms={"table": 0.5, "ball": 0.7},
        n_coupling_groups=0,
        coupling_overhead_ms=0.05,
        node_sizes={"table": 1, "ball": 2},
        total_state_elements=3,
        bottleneck="ball (0.70ms, 56% of step)",
        recommendations=["use --gpu for the heat node"],
    )


# ---------------------------------------------------------------------------
# Perfetto-format conversion
# ---------------------------------------------------------------------------


class TestPerfettoExport:
    def test_returns_dict_with_traceEvents(self, sample_report):
        out = profile_report_to_perfetto(sample_report)
        assert isinstance(out, dict)
        assert "traceEvents" in out
        assert isinstance(out["traceEvents"], list)
        assert out["traceEvents"]  # non-empty

    def test_is_json_serialisable(self, sample_report):
        # Perfetto loads JSON; we must not leak numpy/jax types.
        out = profile_report_to_perfetto(sample_report)
        s = json.dumps(out)
        # round-trip
        back = json.loads(s)
        assert back["traceEvents"][0]["ph"] == "X"

    def test_includes_step_event(self, sample_report):
        out = profile_report_to_perfetto(sample_report)
        names = [e["name"] for e in out["traceEvents"]]
        # Should contain a top-level step event referencing n_steps
        assert any("run" in n and "10" in n for n in names)

    def test_includes_one_event_per_node(self, sample_report):
        out = profile_report_to_perfetto(sample_report)
        node_events = [e for e in out["traceEvents"] if e["cat"] == "node"]
        assert {e["args"]["node"] for e in node_events} == {"table", "ball"}

    def test_node_durations_in_microseconds(self, sample_report):
        out = profile_report_to_perfetto(sample_report)
        node_events = {e["args"]["node"]: e for e in out["traceEvents"]
                       if e["cat"] == "node"}
        assert node_events["table"]["dur"] == pytest.approx(0.5 * 1000.0)
        assert node_events["ball"]["dur"] == pytest.approx(0.7 * 1000.0)

    def test_complete_phase_events(self, sample_report):
        """All events must have ph='X' (complete event) so Perfetto
        renders them as duration bars rather than counters."""
        out = profile_report_to_perfetto(sample_report)
        for ev in out["traceEvents"]:
            assert ev["ph"] == "X"
            assert "ts" in ev
            assert "dur" in ev

    def test_coupling_event_appears_when_overhead_positive(self, sample_report):
        out = profile_report_to_perfetto(sample_report)
        names = [e["name"] for e in out["traceEvents"]]
        assert "coupling_overhead" in names

    def test_coupling_event_absent_when_zero(self):
        report = ProfileReport(
            n_steps=10, mean_step_ms=1.0, total_run_ms=10.0,
            node_times_ms={"a": 1.0},
            coupling_overhead_ms=0.0,
        )
        out = profile_report_to_perfetto(report)
        names = [e["name"] for e in out["traceEvents"]]
        assert "coupling_overhead" not in names

    def test_otherData_carries_recommendations(self, sample_report):
        out = profile_report_to_perfetto(sample_report)
        assert "otherData" in out
        assert out["otherData"]["recommendations"] == sample_report.recommendations

    def test_displayTimeUnit_us(self, sample_report):
        out = profile_report_to_perfetto(sample_report)
        assert out["displayTimeUnit"] == "us"

    def test_node_events_layout_is_sequential(self, sample_report):
        """Within the step, per-node events should be laid end-to-end
        so Perfetto shows a flame-graph instead of overlapping bars."""
        out = profile_report_to_perfetto(sample_report)
        node_events = sorted(
            [e for e in out["traceEvents"] if e["cat"] == "node"],
            key=lambda e: e["ts"],
        )
        for prev, nxt in zip(node_events, node_events[1:]):
            assert nxt["ts"] >= prev["ts"] + prev["dur"] - 1e-9, (
                "node events should not overlap"
            )

    def test_share_of_step_pct_sane(self, sample_report):
        out = profile_report_to_perfetto(sample_report)
        for ev in out["traceEvents"]:
            if ev["cat"] == "node":
                pct = ev["args"]["share_of_step_pct"]
                assert 0.0 <= pct <= 200.0  # generous upper bound


# ---------------------------------------------------------------------------
# Round-trip through profile_graph (integration; light step count)
# ---------------------------------------------------------------------------


class TestProfileGraphIntegration:
    def test_profile_real_graph_produces_loadable_perfetto(self, small_graph):
        report = profile_graph(small_graph, n_steps=5, n_warmup=1)
        out = profile_report_to_perfetto(report)
        # Must be valid JSON and have the expected schema keys
        s = json.dumps(out)
        back = json.loads(s)
        assert back["displayTimeUnit"] == "us"
        assert len(back["traceEvents"]) >= 1
        # Steps-per-second should be positive on a working graph
        assert report.steps_per_second > 0

    def test_profile_uncompiled_graph_recompiles(self, small_graph):
        # Dirty the graph so profile_graph has to recompile
        small_graph._dirty = True
        report = profile_graph(small_graph, n_steps=3, n_warmup=1)
        assert report.jit_compile_ms > 0


# ---------------------------------------------------------------------------
# JAX profiler session management
# ---------------------------------------------------------------------------


class TestJaxProfilerSession:
    def test_context_manager_creates_dir(self):
        with JaxProfilerSession() as sess:
            assert sess.log_dir is not None
            assert os.path.isdir(sess.log_dir)
            assert sess.active is True
        assert sess.active is False

    def test_context_manager_uses_supplied_dir(self, tmp_path):
        target = tmp_path / "custom_trace"
        target.mkdir()
        with JaxProfilerSession(log_dir=str(target)) as sess:
            assert sess.log_dir == str(target)
        # The dir we supplied still exists after the session ends
        assert target.is_dir()

    def test_context_manager_handles_exception(self):
        sess_ref = None
        with pytest.raises(RuntimeError, match="boom"):
            with JaxProfilerSession() as sess:
                sess_ref = sess
                raise RuntimeError("boom")
        # Session should have been cleanly stopped on exception
        assert sess_ref is not None and sess_ref.active is False

    def test_start_stop_pair_round_trip(self):
        assert jax_trace_active() is False
        log_dir = start_jax_trace()
        try:
            assert jax_trace_active() is True
            assert os.path.isdir(log_dir)
        finally:
            stop_jax_trace()
        assert jax_trace_active() is False

    def test_start_while_active_raises(self):
        start_jax_trace()
        try:
            with pytest.raises(RuntimeError, match="already active"):
                start_jax_trace()
        finally:
            stop_jax_trace()

    def test_stop_when_inactive_raises(self):
        assert jax_trace_active() is False
        with pytest.raises(RuntimeError, match="No JAX trace"):
            stop_jax_trace()

    def test_start_returns_log_dir_path(self):
        log_dir = start_jax_trace()
        try:
            assert isinstance(log_dir, str)
            assert log_dir.startswith("/")  # absolute path
        finally:
            stop_jax_trace()


# ---------------------------------------------------------------------------
# tar_trace_dir
# ---------------------------------------------------------------------------


class TestTarTraceDir:
    def test_packs_directory(self, tmp_path):
        # Create a fake trace directory with one file
        trace_dir = tmp_path / "trace_xyz"
        trace_dir.mkdir()
        (trace_dir / "profile.xplane.pb").write_bytes(b"\x00\x01\x02")

        tar_bytes = tar_trace_dir(str(trace_dir))
        assert isinstance(tar_bytes, bytes)
        assert len(tar_bytes) > 0

        # Verify it's a valid tar.gz with the expected member
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            names = tar.getnames()
            assert any("profile.xplane.pb" in n for n in names)

    def test_empty_dir_still_produces_tar(self, tmp_path):
        trace_dir = tmp_path / "empty"
        trace_dir.mkdir()
        tar_bytes = tar_trace_dir(str(trace_dir))
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            names = tar.getnames()
            # the directory itself shows up as a tar entry
            assert any("empty" in n for n in names)

    def test_preserves_nested_files(self, tmp_path):
        trace_dir = tmp_path / "trace_nest"
        (trace_dir / "plugins" / "profile" / "ts").mkdir(parents=True)
        (trace_dir / "plugins" / "profile" / "ts" / "xplane.pb").write_bytes(b"data")

        tar_bytes = tar_trace_dir(str(trace_dir))
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
            names = tar.getnames()
            assert any("plugins/profile/ts/xplane.pb" in n for n in names)
