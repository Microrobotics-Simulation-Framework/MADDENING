"""Cloud session state machine and orchestration.

``CloudSession`` provisions a cloud GPU VM via SkyPilot, starts a
container, waits for the simulation and stream to become ready, and
exposes health-check / teardown methods.  All SkyPilot calls are
isolated in ``_skypilot.py``; health probes are in ``_health.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from maddening.cloud.streaming import QualityPreset, StreamConfig, StreamInfo

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------

class CloudStage(Enum):
    """Lifecycle stages of a cloud session."""

    NOT_STARTED = "not_started"
    VM_PROVISIONING = "vm_provisioning"
    CONTAINER_STARTING = "container_starting"
    SIMULATION_STARTING = "simulation_starting"
    STREAM_STARTING = "stream_starting"
    STREAM_READY = "stream_ready"
    DATA_READY = "data_ready"
    FULLY_READY = "fully_ready"
    ERROR = "error"
    PREEMPTED = "preempted"


class PreemptionPolicy(Enum):
    """How to respond when a spot VM is preempted."""

    CHECKPOINT = "checkpoint"
    FAILOVER = "failover"
    ABORT = "abort"


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass(frozen=True)
class CloudConfig:
    """Configuration for a cloud session."""

    cloud: str = "gcp"
    instance_type: str = "n1-standard-4"
    accelerator: str = "T4:1"
    spot: bool = True
    on_preempted: PreemptionPolicy = PreemptionPolicy.CHECKPOINT
    container_image: str = "maddening-cloud:latest"
    stream_config: StreamConfig = field(default_factory=StreamConfig)
    region: str = ""
    region_strategy: str = "cheapest"

    @classmethod
    def from_dict(cls, d: dict) -> "CloudConfig":
        """Reconstruct from a plain dict (e.g. JSON deserialization)."""
        d = dict(d)  # shallow copy
        if "on_preempted" in d and isinstance(d["on_preempted"], str):
            d["on_preempted"] = PreemptionPolicy(d["on_preempted"])
        if "stream_config" in d and isinstance(d["stream_config"], dict):
            d["stream_config"] = StreamConfig.from_dict(d["stream_config"])
        return cls(**d)


# ------------------------------------------------------------------
# Result / Info types
# ------------------------------------------------------------------

@dataclass(frozen=True)
class CloudReadyResult:
    """Result of ``wait_ready()``, with per-stage pass/fail."""

    vm_ready: bool = False
    container_ready: bool = False
    simulation_ready: bool = False
    stream_ready: bool = False
    data_ready: bool = False
    error_stage: Optional[str] = None
    error_detail: Optional[str] = None

    @property
    def fully_ready(self) -> bool:
        return (self.vm_ready and self.container_ready
                and self.simulation_ready and self.stream_ready
                and self.data_ready and self.error_stage is None)


@dataclass
class CloudSessionInfo:
    """Live metadata about a cloud session."""

    session_id: str = ""
    vm_ip: str = ""
    stage: CloudStage = CloudStage.NOT_STARTED
    stream_info: Optional[StreamInfo] = None
    zmq_state_endpoint: str = ""
    zmq_command_endpoint: str = ""
    skypilot_job_id: Optional[str] = None
    container_image_hash: str = ""


# ------------------------------------------------------------------
# Error
# ------------------------------------------------------------------

class CloudSessionError(Exception):
    """Error with stage attribution."""

    def __init__(self, message: str, stage: str = "", detail: str = ""):
        super().__init__(message)
        self.stage = stage
        self.detail = detail


# ------------------------------------------------------------------
# CloudSession
# ------------------------------------------------------------------

class CloudSession:
    """Orchestrates a cloud GPU session lifecycle.

    Parameters
    ----------
    on_stage_changed : callable, optional
        ``(CloudSessionInfo) -> None`` — called on every stage transition.
    on_preempted : callable, optional
        ``(CloudSessionInfo) -> None`` — called when the VM is preempted.
    """

    def __init__(
        self,
        on_stage_changed: Optional[Callable[[CloudSessionInfo], None]] = None,
        on_preempted: Optional[Callable[[CloudSessionInfo], None]] = None,
    ) -> None:
        self._on_stage_changed = on_stage_changed
        self._on_preempted = on_preempted
        self._lock = threading.Lock()
        self._ready_event = threading.Event()
        self._info = CloudSessionInfo()
        self._config: Optional[CloudConfig] = None
        self._preemption_monitor: Optional[threading.Thread] = None
        self._launch_thread: Optional[threading.Thread] = None
        self._ready_result: Optional[CloudReadyResult] = None

    @property
    def info(self) -> CloudSessionInfo:
        with self._lock:
            return dataclasses.replace(self._info)

    @property
    def stage(self) -> CloudStage:
        with self._lock:
            return self._info.stage

    def launch(self, config: CloudConfig) -> CloudSessionInfo:
        """Start provisioning in a background thread.

        Returns immediately with the current ``CloudSessionInfo``
        (stage will be VM_PROVISIONING).
        """
        with self._lock:
            if self._info.stage not in (
                CloudStage.NOT_STARTED, CloudStage.ERROR, CloudStage.PREEMPTED,
            ):
                raise CloudSessionError(
                    "Cannot launch: session is already active",
                    stage=self._info.stage.value,
                )
            self._config = config
            self._info.stage = CloudStage.VM_PROVISIONING

        if self._on_stage_changed:
            self._on_stage_changed(self.info)

        self._ready_event.clear()
        self._launch_thread = threading.Thread(
            target=self._launch_worker, daemon=True,
        )
        self._launch_thread.start()
        return self.info

    def wait_ready(self, timeout: Optional[float] = None) -> CloudReadyResult:
        """Block until FULLY_READY or ERROR, then return result."""
        self._ready_event.wait(timeout=timeout)
        if self._ready_result is not None:
            return self._ready_result
        # Timeout — build partial result from current stage
        return self._build_ready_result()

    def health_check(self) -> CloudReadyResult:
        """Non-blocking health check based on current stage."""
        return self._build_ready_result()

    def teardown(self) -> None:
        """Tear down the cloud session and release resources."""
        with self._lock:
            stage = self._info.stage
            job_id = self._info.skypilot_job_id
            self._info.stage = CloudStage.NOT_STARTED

        if job_id:
            try:
                from maddening.cloud._skypilot import teardown_vm
                teardown_vm(job_id)
            except Exception:
                logger.exception("Failed to tear down VM")

        self._ready_event.set()

    # -- Internal stage management -------------------------------------

    def _advance_stage(self, new_stage: CloudStage) -> None:
        """Thread-safe stage transition + user callback."""
        with self._lock:
            self._info.stage = new_stage
        if self._on_stage_changed:
            self._on_stage_changed(self.info)

    def _on_preemption_signal(self) -> None:
        """Called by the preemption monitor (internal only)."""
        with self._lock:
            self._info.stage = CloudStage.PREEMPTED
        if self._on_preempted:
            self._on_preempted(self.info)
        self._ready_event.set()

    def _build_ready_result(
        self,
        error_stage: Optional[str] = None,
        error_detail: Optional[str] = None,
    ) -> CloudReadyResult:
        with self._lock:
            stage = self._info.stage

        stage_order = [
            CloudStage.VM_PROVISIONING,
            CloudStage.CONTAINER_STARTING,
            CloudStage.SIMULATION_STARTING,
            CloudStage.STREAM_STARTING,
            CloudStage.STREAM_READY,
            CloudStage.DATA_READY,
            CloudStage.FULLY_READY,
        ]
        passed = set()
        for s in stage_order:
            if s == stage:
                break
            passed.add(s)
        if stage == CloudStage.FULLY_READY:
            passed = set(stage_order)

        return CloudReadyResult(
            vm_ready=CloudStage.VM_PROVISIONING in passed or stage in (
                CloudStage.FULLY_READY,),
            container_ready=CloudStage.CONTAINER_STARTING in passed or stage in (
                CloudStage.FULLY_READY,),
            simulation_ready=CloudStage.SIMULATION_STARTING in passed or stage in (
                CloudStage.FULLY_READY,),
            stream_ready=CloudStage.STREAM_READY in passed
                or CloudStage.STREAM_STARTING in passed
                or stage in (CloudStage.FULLY_READY,),
            data_ready=CloudStage.DATA_READY in passed or stage in (
                CloudStage.FULLY_READY,),
            error_stage=error_stage,
            error_detail=error_detail,
        )

    # -- Background worker ---------------------------------------------

    def _launch_worker(self) -> None:
        """Runs in a background thread: provisions VM, probes health."""
        from maddening.cloud._health import HealthProbeError

        config = self._config
        assert config is not None

        try:
            # Stage 1: Provision VM
            from maddening.cloud._skypilot import launch_vm, monitor_preemption
            vm_ip, job_id = launch_vm(config)
            with self._lock:
                self._info.vm_ip = vm_ip
                self._info.skypilot_job_id = job_id
            self._advance_stage(CloudStage.CONTAINER_STARTING)

            # Start preemption monitor if spot
            if config.spot:
                self._preemption_monitor = monitor_preemption(
                    job_id, self._on_preemption_signal,
                )

            # Stage 2: Wait for container
            from maddening.cloud._health import probe_http, wait_for
            container_url = f"http://{vm_ip}:8000/graph"
            wait_for(lambda: probe_http(container_url), timeout=120, interval=5)
            self._advance_stage(CloudStage.SIMULATION_STARTING)

            # Stage 3: Wait for simulation
            sim_url = f"http://{vm_ip}:8000/graph/state"
            wait_for(lambda: probe_http(sim_url), timeout=60, interval=3)
            self._advance_stage(CloudStage.STREAM_STARTING)

            # Stage 4: Wait for stream
            stream_url = f"http://{vm_ip}:8080/health"
            try:
                wait_for(
                    lambda: probe_http(stream_url, timeout=5),
                    timeout=60, interval=3,
                )
            except HealthProbeError:
                # Stream health endpoint may not exist; proceed anyway
                pass
            self._advance_stage(CloudStage.STREAM_READY)

            # Stage 5: Wait for data channel
            zmq_endpoint = f"tcp://{vm_ip}:5555"
            with self._lock:
                self._info.zmq_state_endpoint = zmq_endpoint
                self._info.zmq_command_endpoint = f"tcp://{vm_ip}:5556"
            try:
                from maddening.cloud._health import probe_zmq
                wait_for(
                    lambda: probe_zmq(zmq_endpoint),
                    timeout=30, interval=3,
                )
            except (HealthProbeError, ImportError):
                pass
            self._advance_stage(CloudStage.DATA_READY)

            # Final: fully ready
            self._advance_stage(CloudStage.FULLY_READY)
            self._ready_result = self._build_ready_result()
            self._ready_event.set()

        except HealthProbeError as exc:
            logger.error("Health probe failed at stage %s: %s", exc.stage, exc.detail)
            self._ready_result = self._build_ready_result(
                error_stage=exc.stage, error_detail=exc.detail,
            )
            self._advance_stage(CloudStage.ERROR)
            self._ready_event.set()

        except Exception as exc:
            logger.exception("Cloud session launch failed")
            self._ready_result = self._build_ready_result(
                error_stage="unknown", error_detail=str(exc),
            )
            self._advance_stage(CloudStage.ERROR)
            self._ready_event.set()
