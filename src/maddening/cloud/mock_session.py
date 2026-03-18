"""Mock cloud session for testing (depends on session.py types only).

``MockCloudSession`` simulates the ``CloudSession`` state machine with
configurable delays and failure injection, without touching SkyPilot or
real infrastructure.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from typing import Callable, Optional

from maddening.cloud.session import (
    CloudConfig,
    CloudReadyResult,
    CloudSessionError,
    CloudSessionInfo,
    CloudStage,
)

logger = logging.getLogger(__name__)


class MockCloudSession:
    """Simulated cloud session for testing.

    Parameters
    ----------
    on_stage_changed : callable, optional
        ``(CloudSessionInfo) -> None`` — called on every stage transition.
    on_preempted : callable, optional
        ``(CloudSessionInfo) -> None`` — called when preemption is simulated.
    stage_delay : float
        Seconds to wait between stage transitions (simulates provisioning).
    fail_at_stage : CloudStage, optional
        If set, the session will fail when reaching this stage.
    fail_detail : str
        Error detail when failing at the configured stage.
    preempt_after : float, optional
        If set, simulate preemption after this many seconds.
    """

    def __init__(
        self,
        on_stage_changed: Optional[Callable[[CloudSessionInfo], None]] = None,
        on_preempted: Optional[Callable[[CloudSessionInfo], None]] = None,
        stage_delay: float = 0.0,
        fail_at_stage: Optional[CloudStage] = None,
        fail_detail: str = "Simulated failure",
        preempt_after: Optional[float] = None,
    ) -> None:
        self._on_stage_changed = on_stage_changed
        self._on_preempted = on_preempted
        self._stage_delay = stage_delay
        self._fail_at_stage = fail_at_stage
        self._fail_detail = fail_detail
        self._preempt_after = preempt_after
        self._lock = threading.Lock()
        self._ready_event = threading.Event()
        self._info = CloudSessionInfo()
        self._config: Optional[CloudConfig] = None
        self._ready_result: Optional[CloudReadyResult] = None
        self._launch_thread: Optional[threading.Thread] = None

    @property
    def info(self) -> CloudSessionInfo:
        with self._lock:
            return dataclasses.replace(self._info)

    @property
    def stage(self) -> CloudStage:
        with self._lock:
            return self._info.stage

    def launch(self, config: CloudConfig) -> CloudSessionInfo:
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
            self._info.session_id = "mock-session-001"
            self._info.vm_ip = "10.0.0.42"

        if self._on_stage_changed:
            self._on_stage_changed(self.info)

        self._ready_event.clear()
        self._launch_thread = threading.Thread(
            target=self._launch_worker, daemon=True,
        )
        self._launch_thread.start()
        return self.info

    def wait_ready(self, timeout: Optional[float] = None) -> CloudReadyResult:
        self._ready_event.wait(timeout=timeout)
        if self._ready_result is not None:
            return self._ready_result
        return self._build_ready_result()

    def health_check(self) -> CloudReadyResult:
        return self._build_ready_result()

    def teardown(self) -> None:
        with self._lock:
            self._info.stage = CloudStage.NOT_STARTED
        self._ready_event.set()

    # -- Internal helpers ----------------------------------------------

    def _advance_stage(self, new_stage: CloudStage) -> None:
        with self._lock:
            self._info.stage = new_stage
        if self._on_stage_changed:
            self._on_stage_changed(self.info)

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
            vm_ready=CloudStage.VM_PROVISIONING in passed or stage == CloudStage.FULLY_READY,
            container_ready=CloudStage.CONTAINER_STARTING in passed or stage == CloudStage.FULLY_READY,
            simulation_ready=CloudStage.SIMULATION_STARTING in passed or stage == CloudStage.FULLY_READY,
            stream_ready=(CloudStage.STREAM_READY in passed
                          or CloudStage.STREAM_STARTING in passed
                          or stage == CloudStage.FULLY_READY),
            data_ready=CloudStage.DATA_READY in passed or stage == CloudStage.FULLY_READY,
            error_stage=error_stage,
            error_detail=error_detail,
        )

    def _launch_worker(self) -> None:
        """Simulate stage progression in a background thread."""
        stages = [
            CloudStage.CONTAINER_STARTING,
            CloudStage.SIMULATION_STARTING,
            CloudStage.STREAM_STARTING,
            CloudStage.STREAM_READY,
            CloudStage.DATA_READY,
            CloudStage.FULLY_READY,
        ]

        # Optional preemption timer
        preempt_event = threading.Event()
        if self._preempt_after is not None:
            def _trigger_preemption():
                time.sleep(self._preempt_after)
                preempt_event.set()
            pt = threading.Thread(target=_trigger_preemption, daemon=True)
            pt.start()

        for target_stage in stages:
            if self._stage_delay > 0:
                time.sleep(self._stage_delay)

            # Check for preemption
            if preempt_event.is_set():
                with self._lock:
                    self._info.stage = CloudStage.PREEMPTED
                if self._on_preempted:
                    self._on_preempted(self.info)
                self._ready_event.set()
                return

            # Check for configured failure
            if self._fail_at_stage == target_stage:
                error_stage = target_stage.value
                self._ready_result = self._build_ready_result(
                    error_stage=error_stage,
                    error_detail=self._fail_detail,
                )
                self._advance_stage(CloudStage.ERROR)
                self._ready_event.set()
                return

            self._advance_stage(target_stage)

        self._ready_result = self._build_ready_result()
        self._ready_event.set()
