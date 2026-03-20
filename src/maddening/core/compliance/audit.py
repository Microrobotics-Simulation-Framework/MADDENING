"""
Audit logging interface (Section 9.6).

Provides an opt-in audit logger that records simulation events for
downstream regulatory traceability.

Pure Python — no JAX dependency.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Protocol, Optional
from pathlib import Path


class AuditSink(Protocol):
    """Protocol for audit event sinks."""

    def write_event(self, event: dict[str, Any]) -> None:
        """Write a single audit event."""
        ...


class NullSink:
    """Sink that discards all events (default when auditing is not needed)."""

    def write_event(self, event: dict[str, Any]) -> None:
        pass


class JSONFileSink:
    """Sink that appends events as JSON lines to a file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_event(self, event: dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")


class AuditLogger:
    """Opt-in audit logger for simulation events.

    The audit logger records structured events (simulation start, step,
    parameter changes, anomalies detected) to a configurable sink.

    Usage::

        logger = AuditLogger(sink=JSONFileSink("audit.jsonl"))
        logger.log("simulation_start", {"n_nodes": 4, "dt": 0.001})
        logger.log("step", {"step": 42, "t": 0.042})

    By default, the logger uses a NullSink and does nothing.
    """

    def __init__(self, sink: Optional[AuditSink] = None):
        self._sink = sink or NullSink()
        self._sequence = 0

    def log(self, event_type: str, data: Optional[dict[str, Any]] = None) -> None:
        """Record an audit event.

        Parameters
        ----------
        event_type : str
            Classification of the event (e.g., "simulation_start", "step",
            "parameter_change", "anomaly_detected").
        data : dict, optional
            Event-specific payload.
        """
        self._sequence += 1
        event = {
            "sequence": self._sequence,
            "timestamp": time.time(),
            "event_type": event_type,
            "data": data or {},
        }
        self._sink.write_event(event)

    @property
    def event_count(self) -> int:
        """Number of events logged so far."""
        return self._sequence
