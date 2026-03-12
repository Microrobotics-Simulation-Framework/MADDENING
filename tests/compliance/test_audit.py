"""Tests for AuditLogger, NullSink, JSONFileSink."""

import json
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from maddening.core.audit import AuditLogger, NullSink, JSONFileSink


class TestNullSink:
    def test_write_event_does_nothing(self):
        sink = NullSink()
        sink.write_event({"event_type": "test"})  # Should not raise


class TestJSONFileSink:
    def test_writes_jsonl(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        sink = JSONFileSink(path)
        sink.write_event({"event_type": "test", "data": {"key": "value"}})
        sink.write_event({"event_type": "step", "data": {"n": 42}})

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["event_type"] == "test"
        assert json.loads(lines[1])["data"]["n"] == 42

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "audit.jsonl"
        sink = JSONFileSink(path)
        sink.write_event({"test": True})
        assert path.exists()


class TestAuditLogger:
    def test_default_null_sink(self):
        logger = AuditLogger()
        logger.log("test")  # Should not raise
        assert logger.event_count == 1

    def test_with_json_sink(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        logger = AuditLogger(sink=JSONFileSink(path))
        logger.log("simulation_start", {"n_nodes": 4})
        logger.log("step", {"step": 1, "t": 0.001})

        assert logger.event_count == 2
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

        event1 = json.loads(lines[0])
        assert event1["event_type"] == "simulation_start"
        assert event1["sequence"] == 1
        assert "timestamp" in event1

    def test_sequence_increments(self):
        logger = AuditLogger()
        logger.log("a")
        logger.log("b")
        logger.log("c")
        assert logger.event_count == 3
