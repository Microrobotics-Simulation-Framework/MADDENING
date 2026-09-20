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
    """Configuration for a cloud session.

    Attributes
    ----------
    ports : tuple of int
        Container ports to publish on the VM's interfaces, and to open
        in the provider's firewall.  Empty by default, so a launch
        exposes nothing and is reached over an SSH tunnel.  The health
        probes in :meth:`CloudSession._launch_worker` talk to
        *api_port* over the network, so a session that expects them to
        pass must publish it (or run the container with host
        networking, where nothing needs publishing).
    api_port : int
        The port the container's HTTP API listens on, which is what the
        container's own ``MADDENING_PORT`` decides.  Configurable
        because the health probes address it, and a probe of a port
        nothing serves is a 120-second silence.
    """

    cloud: str = "gcp"
    instance_type: str = "n1-standard-4"
    accelerator: str = "T4:1"
    spot: bool = True
    on_preempted: PreemptionPolicy = PreemptionPolicy.CHECKPOINT
    container_image: str = "maddening-cloud:latest"
    stream_config: StreamConfig = field(default_factory=StreamConfig)
    region: str = ""
    region_strategy: str = "cheapest"
    ports: tuple[int, ...] = ()
    api_port: int = 8000

    @classmethod
    def from_dict(cls, d: dict) -> "CloudConfig":
        """Reconstruct from a plain dict (e.g. JSON deserialization)."""
        d = dict(d)  # shallow copy
        if "on_preempted" in d and isinstance(d["on_preempted"], str):
            d["on_preempted"] = PreemptionPolicy(d["on_preempted"])
        if "stream_config" in d and isinstance(d["stream_config"], dict):
            d["stream_config"] = StreamConfig.from_dict(d["stream_config"])
        if "ports" in d and d["ports"] is not None:
            d["ports"] = tuple(int(p) for p in d["ports"])
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

    The session owns the container's credentials.  It resolves them
    before the VM is provisioned, hands them to the container as
    environment variables, and presents them on its own health probes,
    so the probes measure the container's health rather than their own
    lack of a credential.  Before 0.4.0 the container generated a token
    that only its log held and the probes sent none, which meant
    :meth:`wait_ready` could not succeed against it at all.

    Parameters
    ----------
    on_stage_changed : callable, optional
        ``(CloudSessionInfo) -> None`` — called on every stage transition.
    on_preempted : callable, optional
        ``(CloudSessionInfo) -> None`` — called when the VM is preempted.
    api_token : str, optional
        The HTTP bearer token for the container.  ``None`` reads
        ``MADDENING_API_TOKEN`` from the launching environment and, if
        that is unset, **generates** one.  Either way the same value
        reaches the container and the probes.
    transport_token : str, optional
        The secret the ZeroMQ CURVE keys are derived from.  ``None``
        reads ``MADDENING_TRANSPORT_TOKEN`` and, if that is unset,
        leaves it unset in the container, where
        :class:`maddening.transport_auth.TransportAuth` falls back to
        the API token.  Set it to keep the streams' key off the
        cleartext HTTP credential.

    Attributes
    ----------
    api_token : str
        The bearer token the container was given.  A caller that did not
        supply one reads it here to talk to the API: a generated token
        is deliberately not logged.
    transport_token : str or None
        The transport secret the container was given, or ``None`` when
        the container falls back to :attr:`api_token`.
    """

    def __init__(
        self,
        on_stage_changed: Optional[Callable[[CloudSessionInfo], None]] = None,
        on_preempted: Optional[Callable[[CloudSessionInfo], None]] = None,
        api_token: Optional[str] = None,
        transport_token: Optional[str] = None,
    ) -> None:
        import os
        import secrets

        from maddening.api.auth import TOKEN_ENV
        from maddening.transport_auth import TRANSPORT_TOKEN_ENV

        self._on_stage_changed = on_stage_changed
        self._on_preempted = on_preempted
        self._lock = threading.Lock()
        self._ready_event = threading.Event()
        self._info = CloudSessionInfo()
        self._config: Optional[CloudConfig] = None
        self._preemption_monitor: Optional[threading.Thread] = None
        self._launch_thread: Optional[threading.Thread] = None
        self._ready_result: Optional[CloudReadyResult] = None
        #: Seconds each health stage may spend retrying.  A dict rather
        #: than four literals so a test can drive the real probe path
        #: without waiting out a production retry window.
        self._stage_timeouts = {
            "container": 120.0, "simulation": 60.0, "data_channel": 30.0,
        }

        source = api_token if api_token is not None else os.environ.get(TOKEN_ENV)
        if source is not None and not source.strip():
            raise CloudSessionError(
                f"{TOKEN_ENV} is set but blank. A blank token is a "
                f"configuration error, not a request to disable "
                f"authentication; unset it to have one generated.",
                stage="not_started",
            )
        self._api_token_generated = source is None
        self.api_token = secrets.token_urlsafe(32) if source is None else source
        transport = (
            transport_token if transport_token is not None
            else os.environ.get(TRANSPORT_TOKEN_ENV)
        )
        if transport is not None and not transport.strip():
            raise CloudSessionError(
                f"{TRANSPORT_TOKEN_ENV} is set but blank; unset it to fall "
                f"back to {TOKEN_ENV}, or set it to a value.",
                stage="not_started",
            )
        self.transport_token = transport
        if self._api_token_generated:
            logger.info(
                "No %s in this process's environment; this session generated "
                "one for the container. Read it from CloudSession.api_token "
                "-- it is deliberately not logged.", TOKEN_ENV,
            )

    def container_env(self) -> dict:
        """The credentials this session hands to the container.

        Returns
        -------
        dict
            ``{env var: value}``.  ``MADDENING_TRANSPORT_TOKEN`` appears
            only when one was configured; without it the container's
            transports fall back to the API token, which is the
            documented single-variable setup.
        """
        from maddening.api.auth import TOKEN_ENV
        from maddening.transport_auth import TRANSPORT_TOKEN_ENV

        env = {TOKEN_ENV: self.api_token}
        if self.transport_token:
            env[TRANSPORT_TOKEN_ENV] = self.transport_token
        return env

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
            # Stage 1: Provision VM.  The container is given this
            # session's credentials, so the probes below can present
            # them; a container left to generate its own token would be
            # unprobeable by anything but /healthz.
            from maddening.cloud import _skypilot
            launch_vm = _skypilot.launch_vm
            monitor_preemption = _skypilot.monitor_preemption
            vm_ip, job_id = launch_vm(config, envs=self.container_env())
            with self._lock:
                self._info.vm_ip = vm_ip
                self._info.skypilot_job_id = job_id
            self._advance_stage(CloudStage.CONTAINER_STARTING)

            # Start preemption monitor if spot
            if config.spot:
                self._preemption_monitor = monitor_preemption(
                    job_id, self._on_preemption_signal,
                )

            # Stage 2: Wait for container.
            #
            # /healthz, not /graph.  The container binds 0.0.0.0 (it
            # must), which turns its bearer token on for every route but
            # /healthz and /viz/*, and this probe is the one that has to
            # work before anything is known about credentials: it says
            # "the process is up and this is its version" and nothing
            # about the graph, which is exactly what this stage means.
            # Probing /graph here sent no Authorization header, got 401,
            # retried for the full 120 s and ended every real launch in
            # CloudStage.ERROR.
            from maddening.cloud._health import probe_http, wait_for
            api_port = getattr(config, "api_port", 8000)
            published = tuple(getattr(config, "ports", None) or ())
            if api_port not in published:
                logger.warning(
                    "The health probes address http://%s:%s but %s is not in "
                    "CloudConfig.ports, so the container's port is published "
                    "on the VM only if the container uses host networking. "
                    "Add %s to CloudConfig.ports if these probes time out.",
                    vm_ip, api_port, api_port, api_port,
                )
            container_url = f"http://{vm_ip}:{api_port}/healthz"
            wait_for(
                lambda: probe_http(container_url),
                timeout=self._stage_timeouts["container"], interval=5,
            )
            self._advance_stage(CloudStage.SIMULATION_STARTING)

            # Stage 3: Wait for simulation.  This one really does need
            # the graph, so it presents the token the container was
            # launched with.
            sim_url = f"http://{vm_ip}:{api_port}/graph/state"
            wait_for(
                lambda: probe_http(sim_url, token=self.api_token),
                timeout=self._stage_timeouts["simulation"], interval=3,
            )
            self._advance_stage(CloudStage.STREAM_STARTING)

            # Stage 4: Wait for stream
            #
            # This probed http://<vm>:8080/health for sixty seconds and
            # then swallowed the failure.  Nothing in MADDENING has ever
            # served port 8080 -- the WebRTC signaling server listens on
            # 8443 (``cloud.selkies_session``) and the HTTP API on 8000 --
            # so the probe could only ever time out, and launch_vm was
            # publishing 8080 to the VM's public interface to support it.
            # Both are gone; the stage is kept so the reported sequence of
            # stages is unchanged.
            self._advance_stage(CloudStage.STREAM_READY)

            # Stage 5: Wait for data channel
            #
            # Reachable only if the job's config asked for 5555/5556 in
            # CloudConfig.ports; launch_vm no longer publishes them by
            # default.  The probe failure is swallowed below, so an
            # unpublished port just leaves the endpoints advertised for a
            # caller who has set up an SSH tunnel -- but the probe is
            # skipped outright in that case rather than spending 30 s
            # discovering that nothing is listening.
            #
            # A published relay port is by definition non-loopback, so
            # the relay runs CURVE; the probe therefore carries the
            # transport secret.  A plain SUB socket (what this used) can
            # never complete that handshake, so the stage could only
            # ever time out.
            zmq_endpoint = f"tcp://{vm_ip}:5555"
            with self._lock:
                self._info.zmq_state_endpoint = zmq_endpoint
                self._info.zmq_command_endpoint = f"tcp://{vm_ip}:5556"
            if 5555 in published:
                probe_token = self.transport_token or self.api_token
                try:
                    from maddening.cloud._health import probe_zmq
                    wait_for(
                        lambda: probe_zmq(zmq_endpoint, token=probe_token),
                        timeout=self._stage_timeouts["data_channel"],
                        interval=3,
                    )
                except (HealthProbeError, ImportError):
                    logger.info(
                        "State stream probe of %s did not receive a frame; "
                        "the endpoint is advertised anyway.", zmq_endpoint,
                    )
            else:
                logger.info(
                    "5555 is not published, so the state stream is reachable "
                    "only through a tunnel; skipping the data-channel probe "
                    "rather than waiting 30s for a port nobody opened.",
                )
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
