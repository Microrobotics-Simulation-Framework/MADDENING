"""CloudLauncher — user-facing cloud job orchestration.

Two separate paths to cloud execution exist in MADDENING:

CloudLauncher  — User-facing, script/CLI path.  Loads credentials from
                 ~/.maddening/cloud_credentials.yaml.  Calls sky.* directly.
                 Future basis for CloudSweep and CloudGroup.

CloudSession   — Server-side orchestration path.  Credentials assumed
                 pre-configured on the machine.  Uses _skypilot.py wrapper.
                 Future basis for cloud API endpoints in MICROBOTICA.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import yaml

from maddening.cloud.providers import PROVIDERS, CloudProvider

logger = logging.getLogger(__name__)

_DEFAULT_CREDENTIALS_PATH = Path.home() / ".maddening" / "cloud_credentials.yaml"

# Env vars injected by CloudLauncher or reserved for future CloudGroup.
# JobConfig.envs must not contain these.
_RESERVED_ENV_VARS = frozenset({
    "MADDENING_CLOUD_CONFIG",
    "COORDINATOR_ADDR",
    "SUBGRAPH_ID",
})


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------

class CredentialError(Exception):
    """Missing/invalid credentials file or provider block."""
    pass


class CostLimitError(Exception):
    """Cost guard rejected the launch.

    Attributes
    ----------
    limit : float
        The configured limit that was exceeded.
    actual : float
        The actual value that exceeded the limit.
    guard_type : str
        ``'hourly'`` or ``'budget'``.
    """

    def __init__(self, message: str, limit: float, actual: float,
                 guard_type: str):
        super().__init__(message)
        self.limit = limit
        self.actual = actual
        self.guard_type = guard_type


class LaunchError(Exception):
    """SkyPilot launch failed.  Wraps the underlying sky exception."""
    pass


# ------------------------------------------------------------------
# Enums and config dataclasses
# ------------------------------------------------------------------

class JobPhase(Enum):
    """Execution phase of a cloud job."""

    PROVISIONING = "provisioning"
    WAITING = "waiting"       # At rendezvous barrier (multi-job only)
    EXECUTING = "executing"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class CostPolicy:
    """Cost safety guardrails."""

    max_cost_per_hour: float = 2.00
    max_total_budget: float = 5.00
    autostop_minutes: int = 15
    auto_teardown: bool = True
    spot_fallback: bool = False  # If True, fall back to on-demand when spot unavailable


@dataclass(frozen=True)
class JobConfig:
    """Parsed job configuration (no secrets).

    ``gpu_type`` is a provider-specific SkyPilot accelerator name.
    It is NOT portable across providers — ``"A4000"`` on RunPod is not
    the same string as on Lambda Labs.  Run
    ``sky show-gpus --cloud <provider>`` to see available options.

    Reserved env var names (must not appear in ``envs``)::

        MADDENING_CLOUD_CONFIG  — injected by CloudLauncher
        COORDINATOR_ADDR        — injected by future CloudGroup
        SUBGRAPH_ID             — injected by future CloudGroup
    """

    provider: str
    gpu_type: str
    gpu_count: int = 1
    use_spot: bool = True
    region: str = ""
    disk_size: int = 50
    cost: CostPolicy = field(default_factory=CostPolicy)
    container_image: str = "ghcr.io/microrobotics-simulation-framework/maddening-cloud:latest"
    stream_preset: str = "standard"
    setup: str = ""    # Shell commands to run during VM setup (pip install, etc.)
    run: str = ""      # Shell commands to run as the job
    workdir: str = ""  # Local directory to sync to VM (SkyPilot workdir)
    ports: list[int] = field(default_factory=lambda: [8000])  # Ports to expose via RunPod NAT
    envs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Check for reserved env vars in user-provided envs.
        # CloudGroup injects reserved vars internally via _with_envs().
        bad = _RESERVED_ENV_VARS & set(self.envs)
        if bad and not getattr(self, "_skip_env_check", False):
            raise ValueError(
                f"envs contains reserved key(s): {bad}. "
                f"These are injected automatically by CloudLauncher/CloudGroup."
            )

    def _with_envs(self, extra_envs: dict[str, str]) -> "JobConfig":
        """Return a copy with extra env vars, bypassing reserved-var check.

        Used internally by CloudGroup to inject COORDINATOR_ADDR etc.
        """
        merged = {**self.envs, **extra_envs}
        new = JobConfig(
            provider=self.provider, gpu_type=self.gpu_type,
            gpu_count=self.gpu_count, use_spot=self.use_spot,
            region=self.region, disk_size=self.disk_size,
            cost=self.cost, container_image=self.container_image,
            stream_preset=self.stream_preset, setup=self.setup,
            run=self.run, workdir=self.workdir,
            ports=self.ports,
            envs={k: v for k, v in merged.items() if k not in _RESERVED_ENV_VARS},
        )
        # Store the full envs (including reserved) for _do_launch to use
        object.__setattr__(new, "_extra_envs", extra_envs)
        return new

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "JobConfig":
        """Load from a YAML file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> "JobConfig":
        """Reconstruct from a plain dict."""
        d = dict(d)
        d.pop("version", None)
        if "provider" not in d:
            raise ValueError("Job config must contain 'provider'")
        if "gpu_type" not in d:
            raise ValueError("Job config must contain 'gpu_type'")
        if "cost" in d and isinstance(d["cost"], dict):
            d["cost"] = CostPolicy(**d["cost"])
        if "envs" not in d:
            d["envs"] = {}
        return cls(**d)


# ------------------------------------------------------------------
# Credential context manager
# ------------------------------------------------------------------

@contextlib.contextmanager
def _credential_context(provider: CloudProvider, creds: dict):
    """Write provider credentials, yield, delete on exit.

    If the credential file already exists on disk (user-managed), we
    do NOT overwrite it and do NOT delete it on exit.  We only manage
    files we create.
    """
    path = provider.credential_file_path()
    we_wrote = False
    saved_env: dict[str, Optional[str]] = {}

    try:
        if not path.exists():
            provider.write_credentials(creds)
            we_wrote = True
        saved_env = provider.set_env_vars(creds)
        yield
    finally:
        if we_wrote:
            provider.delete_credentials()
        provider.restore_env_vars(saved_env)


# ------------------------------------------------------------------
# CloudJob
# ------------------------------------------------------------------

class CloudJob:
    """Handle to a single running cloud job.

    Independently status-checkable and teardown-able.  A future
    ``CloudSweep`` or ``CloudGroup`` can hold a ``list[CloudJob]``.

    Not constructed directly — created by ``CloudLauncher.launch()``.
    """

    def __init__(
        self,
        cluster_name: str,
        *,
        request_id: Any = None,
        provider: Optional[CloudProvider] = None,
        creds: Optional[dict] = None,
        vm_ip: Optional[str] = None,
        ssh_port: int = 22,
        ports: Optional[dict[str, int]] = None,
        hourly_cost: float = 0.0,
    ) -> None:
        self._cluster_name = cluster_name
        self._request_id = request_id
        self._provider = provider
        self._creds = creds
        self._vm_ip = vm_ip
        self._ssh_port = ssh_port
        self._ports = ports or {}
        self._hourly_cost = hourly_cost
        self._phase = (
            JobPhase.DONE if cluster_name == "dry-run"
            else JobPhase.PROVISIONING
        )
        self._torn_down = False

    @property
    def cluster_name(self) -> str:
        return self._cluster_name

    @property
    def phase(self) -> JobPhase:
        """Current execution phase.

        For single-job use, ``WAITING`` is never entered.  It exists
        for future ``CloudGroup`` use.
        """
        return self._phase

    @property
    def vm_ip(self) -> Optional[str]:
        """Public IP of the provisioned VM, or ``None`` if not yet available.

        Used by future ``CloudGroup`` to build the ZeroMQ topology
        descriptor.
        """
        return self._vm_ip

    @property
    def ports(self) -> dict[str, int]:
        """ZeroMQ ports this job exposes, keyed by service name.

        Empty for single-job use.  Populated by ``CloudGroup`` from
        ``SubgraphSpec.zmq_ports``.
        """
        return dict(self._ports)

    @property
    def ssh_port(self) -> int:
        """SSH port on the public IP (may be remapped by RunPod NAT)."""
        return self._ssh_port

    def ssh_run(
        self,
        command: str,
        timeout: Optional[int] = None,
        check: bool = True,
        capture: bool = False,
    ) -> "subprocess.CompletedProcess":
        """Run a shell command on the remote VM via SSH.

        Bypasses SkyPilot's Ray job scheduler — runs directly on the
        system Python with full GPU visibility.

        Parameters
        ----------
        command : str
            Shell command to execute.
        timeout : int, optional
            Seconds before the SSH command times out.
        check : bool
            If True, raise ``subprocess.CalledProcessError`` on non-zero exit.
        capture : bool
            If True, capture stdout/stderr instead of printing to terminal.
        """
        import subprocess

        if not self._vm_ip:
            raise LaunchError("No VM IP available — cluster may not be UP yet")

        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-p", str(self._ssh_port),
            f"root@{self._vm_ip}",
            command,
        ]

        kwargs: dict[str, Any] = {"timeout": timeout, "check": check}
        if capture:
            kwargs["capture_output"] = True
            kwargs["text"] = True

        return subprocess.run(ssh_cmd, **kwargs)

    def ssh_run_background(self, command: str) -> None:
        """Start a command on the remote VM in the background via SSH.

        Uses ``nohup ... &`` so the process survives after SSH disconnects.
        """
        # Wrap in nohup and redirect output
        bg_cmd = f"nohup bash -c {_shell_quote(command)} > /tmp/bg_cmd.log 2>&1 &"
        self.ssh_run(bg_cmd, check=False)

    def get_runpod_endpoint(self, private_port: int = 8000) -> Optional[str]:
        """Query RunPod API for the public endpoint of a private port.

        RunPod uses NAT — internal ports are mapped to
        ``public_ip:public_port``.  Returns ``"http://host:port"``
        or ``None`` if not found.
        """
        if self._provider is None or self._creds is None:
            return None
        try:
            import yaml as _yaml
            import runpod
            # Load API key from our credentials
            creds_path = Path.home() / ".maddening" / "cloud_credentials.yaml"
            if creds_path.exists():
                with open(creds_path) as f:
                    all_creds = _yaml.safe_load(f)
                if "runpod" in all_creds:
                    runpod.api_key = all_creds["runpod"]["api_key"]
            elif self._creds and "api_key" in self._creds:
                runpod.api_key = self._creds["api_key"]

            for pod in runpod.get_pods():
                if self._cluster_name in pod.get("name", ""):
                    runtime = pod.get("runtime") or {}
                    for port_info in (runtime.get("ports") or []):
                        if (port_info.get("privatePort") == private_port
                                and port_info.get("isIpPublic")):
                            host = port_info["ip"]
                            pub_port = port_info["publicPort"]
                            return f"http://{host}:{pub_port}"
        except Exception:
            logger.debug("Failed to query RunPod port mapping", exc_info=True)
        return None

    @classmethod
    def from_cluster_name(
        cls,
        name: str,
        credentials_path: Optional[Union[str, Path]] = None,
    ) -> "CloudJob":
        """Reconnect to an existing cluster after process restart.

        ``cluster_name`` is the durable identifier; ``request_id`` is
        transient and unavailable after reconnect.

        Without *credentials_path*, the returned handle supports
        ``status()`` and ``cost_so_far()`` but ``teardown()`` will
        raise ``CredentialError``.  Pass *credentials_path* to enable
        teardown.
        """
        provider = None
        creds = None
        if credentials_path is not None:
            # Try to detect provider from cluster status
            provider, creds = cls._load_creds_for_cluster(
                name, credentials_path,
            )
        job = cls(
            name, provider=provider, creds=creds,
        )
        job._phase = JobPhase.EXECUTING
        return job

    def stream_logs(self) -> None:
        """Block and stream provisioning + job logs to stdout.

        Only available immediately after ``launch()`` (requires
        ``request_id``).
        """
        if self._request_id is None:
            logger.warning("No request_id — cannot stream logs")
            return
        import sky
        sky.stream_and_get(self._request_id)
        self._phase = JobPhase.EXECUTING

    def status(self) -> dict:
        """Non-blocking status check.

        Returns dict with keys: ``cluster_status``, ``job_id``,
        ``vm_ip``, ``hourly_cost``.
        """
        if self._cluster_name == "dry-run":
            return {"cluster_status": "dry-run", "job_id": None,
                    "vm_ip": None, "hourly_cost": 0.0}
        import sky
        req = sky.status(cluster_names=[self._cluster_name])
        clusters = sky.get(req)
        if not clusters:
            self._phase = JobPhase.FAILED
            return {"cluster_status": "not_found", "job_id": None,
                    "vm_ip": None, "hourly_cost": 0.0}
        c = clusters[0]
        status_str = str(c.get("status", "unknown"))
        if hasattr(c.get("status"), "value"):
            status_str = c["status"].value
        # Update phase based on cluster status
        if status_str == "UP":
            if self._phase == JobPhase.PROVISIONING:
                self._phase = JobPhase.EXECUTING
        elif status_str in ("INIT",):
            self._phase = JobPhase.PROVISIONING
        elif status_str in ("STOPPED", "ERROR"):
            self._phase = JobPhase.FAILED
        # Extract IP if available
        handle = c.get("handle")
        if handle is not None and hasattr(handle, "head_ip"):
            self._vm_ip = handle.head_ip
        return {
            "cluster_status": status_str,
            "job_id": c.get("job_id"),
            "vm_ip": self._vm_ip,
            "hourly_cost": self._hourly_cost,
        }

    def cost_so_far(self) -> float:
        """Best-effort estimated cost for this cluster.

        Calculated from SkyPilot's local cache:
        ``hourly_rate * uptime``.  **NOT real-time billing data** — may
        lag or be inaccurate for spot instances or clusters managed
        outside this SkyPilot instance.  Do not use as a hard budget
        guarantee.
        """
        if self._cluster_name == "dry-run":
            return 0.0
        import sky
        req = sky.cost_report()
        records = sky.get(req)
        for r in records:
            if r.get("name") == self._cluster_name:
                return float(r.get("total_cost", 0.0))
        return 0.0

    def teardown(self) -> None:
        """Tear down the cluster.  Idempotent — safe to call multiple times."""
        if self._torn_down or self._cluster_name == "dry-run":
            return
        if self._provider is not None and self._creds is not None:
            with _credential_context(self._provider, self._creds):
                self._do_teardown()
        else:
            # Try without credential context — may work if creds are
            # already on disk, or raise if cloud API needs them.
            self._do_teardown()
        self._torn_down = True
        self._phase = JobPhase.DONE

    def _do_teardown(self) -> None:
        import sky
        try:
            req = sky.down(self._cluster_name, purge=False)
            sky.get(req)
        except Exception as exc:
            logger.warning("Teardown of %s failed: %s",
                           self._cluster_name, exc)
            # Try with purge as fallback
            try:
                req = sky.down(self._cluster_name, purge=True)
                sky.get(req)
            except Exception:
                logger.exception("Purge teardown also failed for %s",
                                 self._cluster_name)

    def is_done(self) -> bool:
        """True if the job has finished or been torn down."""
        return self._phase in (JobPhase.DONE, JobPhase.FAILED)

    @staticmethod
    def _load_creds_for_cluster(
        cluster_name: str,
        credentials_path: Union[str, Path],
    ) -> tuple[Optional[CloudProvider], Optional[dict]]:
        """Try to detect the provider from cluster status and load creds."""
        try:
            import sky
            req = sky.status(cluster_names=[cluster_name])
            clusters = sky.get(req)
            if clusters:
                handle = clusters[0].get("handle")
                if handle is not None and hasattr(handle, "launched_resources"):
                    cloud = handle.launched_resources.cloud
                    cloud_name = str(cloud).lower()
                    for pname, prov in PROVIDERS.items():
                        if prov.skypilot_cloud_name() == cloud_name:
                            creds_data = _load_credentials_file(
                                Path(credentials_path),
                            )
                            if pname in creds_data:
                                return prov, creds_data[pname]
        except Exception:
            pass
        return None, None


# ------------------------------------------------------------------
# CloudLauncher
# ------------------------------------------------------------------

class CloudLauncher:
    """Loads config, manages credentials, enforces cost policy, launches jobs.

    Parameters
    ----------
    credentials_path : str or Path, optional
        Path to ``cloud_credentials.yaml``.
        Default: ``~/.maddening/cloud_credentials.yaml``
    """

    def __init__(
        self,
        credentials_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._credentials_path = Path(
            credentials_path or _DEFAULT_CREDENTIALS_PATH,
        )
        self._creds_data: Optional[dict] = None

    @classmethod
    def from_config(
        cls,
        job_config: Union[JobConfig, str, Path],
        credentials_path: Optional[Union[str, Path]] = None,
    ) -> "CloudLauncher":
        """Convenience: create a launcher with a pre-loaded job config."""
        launcher = cls(credentials_path=credentials_path)
        # Pre-validate
        if not isinstance(job_config, JobConfig):
            job_config = JobConfig.from_yaml(job_config)
        launcher._resolve_provider(job_config.provider)
        return launcher

    def validate(
        self,
        job_config: Union[JobConfig, str, Path],
    ) -> dict:
        """Run all pre-launch checks without spending money.

        Returns
        -------
        dict
            ``{provider, gpu_type, instance_type, hourly_cost,
            budget_used, budget_remaining}``

        Raises
        ------
        CredentialError, CostLimitError, LaunchError
        """
        if not isinstance(job_config, JobConfig):
            job_config = JobConfig.from_yaml(job_config)

        provider, creds = self._resolve_provider(job_config.provider)
        instance_type, hourly_cost = self._resolve_resources(
            provider, job_config,
        )
        budget_used = self._get_budget_used()
        self._check_cost_guards(job_config.cost, hourly_cost, budget_used)

        return {
            "provider": job_config.provider,
            "gpu_type": job_config.gpu_type,
            "instance_type": instance_type,
            "hourly_cost": hourly_cost,
            "budget_used": budget_used,
            "budget_remaining": job_config.cost.max_total_budget - budget_used,
        }

    def launch(
        self,
        job_config: Union[JobConfig, str, Path],
        dry_run: bool = False,
    ) -> CloudJob:
        """Launch a single cloud job.

        Non-blocking after the SkyPilot request is accepted.  Call
        ``job.stream_logs()`` to follow provisioning output.

        If ``job_config.use_spot`` is True and ``job_config.cost.spot_fallback``
        is True, a spot failure will automatically retry with on-demand
        instances (subject to the same cost guards).

        Parameters
        ----------
        job_config : JobConfig or path
            Job configuration.
        dry_run : bool
            If ``True``, SkyPilot resolves resources but does not
            provision.  Returns a ``CloudJob`` with
            ``cluster_name='dry-run'``.
        """
        if not isinstance(job_config, JobConfig):
            job_config = JobConfig.from_yaml(job_config)

        provider, creds = self._resolve_provider(job_config.provider)
        instance_type, hourly_cost = self._resolve_resources(
            provider, job_config,
        )
        budget_used = self._get_budget_used()
        self._check_cost_guards(job_config.cost, hourly_cost, budget_used)

        try:
            return self._do_launch(
                job_config, provider, creds, hourly_cost, dry_run,
                use_spot=job_config.use_spot,
            )
        except LaunchError as exc:
            # If spot failed and fallback is enabled, retry on-demand
            if (job_config.use_spot
                    and job_config.cost.spot_fallback
                    and _is_spot_unavailable_error(exc)):
                logger.info(
                    "Spot instances unavailable, falling back to on-demand"
                )
                # Re-resolve cost for on-demand pricing
                od_instance, od_cost = self._resolve_resources(
                    provider, job_config, use_spot_override=False,
                )
                self._check_cost_guards(
                    job_config.cost, od_cost, budget_used,
                )
                return self._do_launch(
                    job_config, provider, creds, od_cost, dry_run,
                    use_spot=False,
                )
            raise

    def _do_launch(
        self,
        job_config: JobConfig,
        provider: CloudProvider,
        creds: dict,
        hourly_cost: float,
        dry_run: bool,
        use_spot: bool,
    ) -> CloudJob:
        """Internal: execute a single SkyPilot launch attempt."""
        import sky

        envs = dict(job_config.envs)
        envs["MADDENING_CLOUD_CONFIG"] = json.dumps({
            "stream_preset": job_config.stream_preset,
        })
        # Merge any internal extra envs (from CloudGroup._with_envs)
        extra = getattr(job_config, "_extra_envs", None)
        if extra:
            envs.update(extra)

        run_cmd = (
            job_config.run
            or "echo 'MADDENING cloud job running on $HOSTNAME'"
        )
        setup_cmd = job_config.setup or None

        task = sky.Task(
            name=f"maddening-{int(time.time())}",
            setup=setup_cmd,
            run=run_cmd,
            workdir=job_config.workdir or None,
            envs=envs,
        )

        accelerators = (
            f"{job_config.gpu_type}:{job_config.gpu_count}"
            if job_config.gpu_type else None
        )

        cloud_name = provider.skypilot_cloud_name()
        cloud_cls = _resolve_sky_cloud_class(sky, cloud_name)
        if cloud_cls is None:
            raise LaunchError(
                f"Could not find SkyPilot cloud class for '{cloud_name}'"
            )

        resources = sky.Resources(
            cloud=cloud_cls(),
            accelerators=accelerators,
            use_spot=use_spot,
            region=job_config.region or None,
            disk_size=job_config.disk_size,
            image_id=(
                f"docker:{job_config.container_image}"
                if job_config.container_image else None
            ),
            ports=job_config.ports or [8000],
        )
        task.set_resources(resources)

        cluster_name = f"maddening-{int(time.time())}"

        # Credentials must be on disk during sky.launch() AND
        # sky.stream_and_get() (the API server reads creds during
        # provisioning, which happens inside stream_and_get).
        with _credential_context(provider, creds):
            try:
                request_id = sky.launch(
                    task,
                    cluster_name=cluster_name,
                    retry_until_up=not use_spot,
                    idle_minutes_to_autostop=(
                        job_config.cost.autostop_minutes
                    ),
                    down=job_config.cost.auto_teardown,
                    dryrun=dry_run,
                )
                if dry_run:
                    return CloudJob("dry-run")

                # stream_and_get blocks until provisioning + setup + run
                # start, streaming logs to stdout. More reliable than
                # sky.get() which can raise spurious AssertionError.
                result = sky.stream_and_get(request_id)
                job_id = None
                handle = None
                if result is not None:
                    job_id, handle = result

            except Exception as exc:
                # Check if the cluster actually came up despite the error
                try:
                    poll_handle = self._poll_until_up(
                        cluster_name, timeout=30,
                    )
                    if poll_handle is not None:
                        logger.warning(
                            "sky.stream_and_get raised %s but cluster %s "
                            "is UP — continuing", type(exc).__name__,
                            cluster_name,
                        )
                        handle = poll_handle
                        job_id = None
                    else:
                        raise
                except LaunchError:
                    raise LaunchError(
                        _format_launch_error(exc)
                    ) from exc

        vm_ip = None
        ssh_port = 22
        if handle is not None:
            if hasattr(handle, "head_ip"):
                vm_ip = handle.head_ip
            # SkyPilot stores the mapped SSH port for RunPod
            if hasattr(handle, "stable_ssh_ports") and handle.stable_ssh_ports:
                ssh_port = handle.stable_ssh_ports[0]

        spot_str = "spot" if use_spot else "on-demand"
        logger.info("Launched %s (%s, %s, $%.2f/hr)",
                     cluster_name, job_config.gpu_type, spot_str, hourly_cost)

        job = CloudJob(
            cluster_name,
            request_id=request_id,
            provider=provider,
            creds=creds,
            vm_ip=vm_ip,
            ssh_port=ssh_port,
            hourly_cost=hourly_cost,
        )
        job._phase = JobPhase.EXECUTING
        return job

    def list_gpu_types(
        self,
        provider: Optional[str] = None,
    ) -> list[dict]:
        """List available GPU types from SkyPilot catalog.

        Wraps ``sky show-gpus --cloud <provider>`` equivalent.
        Helps users fill in ``job_config.gpu_type`` correctly.
        """
        import sky
        from sky import catalog

        clouds = (
            [provider] if provider
            else list(PROVIDERS.keys())
        )
        results = []
        for cloud_name in clouds:
            prov = PROVIDERS.get(cloud_name)
            if prov is None:
                continue
            sky_cloud = prov.skypilot_cloud_name()
            try:
                df = catalog.list_accelerators(
                    gpus_only=True, clouds=sky_cloud,
                )
                if hasattr(df, 'iterrows'):
                    for _, row in df.iterrows():
                        results.append({
                            "provider": cloud_name,
                            "gpu_type": row.get("AcceleratorName", ""),
                            "count": row.get("AcceleratorCount", 1),
                            "hourly_cost": row.get("Price", 0.0),
                        })
                elif isinstance(df, dict):
                    for name, entries in df.items():
                        for entry in (entries if isinstance(entries, list)
                                      else [entries]):
                            results.append({
                                "provider": cloud_name,
                                "gpu_type": name,
                                "hourly_cost": 0.0,
                            })
            except Exception:
                logger.debug("Failed to list GPUs for %s", cloud_name,
                             exc_info=True)
        return results

    # -- Internal helpers ----------------------------------------------

    @staticmethod
    def _poll_until_up(cluster_name: str, timeout: float = 900):
        """Poll sky.status() until the cluster reaches UP or timeout."""
        import sky as _sky
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                clusters = _sky.get(_sky.status(
                    cluster_names=[cluster_name],
                ))
                if clusters:
                    status = clusters[0].get("status")
                    if str(status) == "ClusterStatus.UP":
                        return clusters[0].get("handle")
            except Exception:
                pass
            time.sleep(10)
        raise LaunchError(
            f"Cluster {cluster_name} did not reach UP within {timeout}s"
        )

    def _load_credentials(self) -> dict:
        """Load and cache the credentials file."""
        if self._creds_data is not None:
            return self._creds_data
        self._creds_data = _load_credentials_file(self._credentials_path)
        return self._creds_data

    def _resolve_provider(
        self,
        provider_name: str,
    ) -> tuple[CloudProvider, dict]:
        """Look up provider and extract its credentials."""
        if provider_name not in PROVIDERS:
            known = ", ".join(sorted(PROVIDERS.keys()))
            raise CredentialError(
                f"Unknown provider '{provider_name}'. "
                f"Known providers: {known}"
            )
        provider = PROVIDERS[provider_name]
        creds_data = self._load_credentials()
        if provider_name not in creds_data:
            raise CredentialError(
                f"No credentials for provider '{provider_name}' in "
                f"{self._credentials_path}. "
                f"Add a '{provider_name}:' block with your API key."
            )
        creds = creds_data[provider_name]
        provider.validate_creds_dict(creds)
        return provider, creds

    def _resolve_resources(
        self,
        provider: CloudProvider,
        job_config: JobConfig,
        use_spot_override: Optional[bool] = None,
    ) -> tuple[str, float]:
        """Resolve gpu_type → instance_type → hourly_cost.

        Two-step resolution: accelerator name → instance type via
        ``sky.catalog.get_instance_type_for_accelerator()``, then
        instance type → hourly cost.
        """
        from sky import catalog

        cloud_name = provider.skypilot_cloud_name()
        acc = job_config.gpu_type
        count = job_config.gpu_count
        use_spot = (
            use_spot_override if use_spot_override is not None
            else job_config.use_spot
        )

        instance_list, _ = catalog.get_instance_type_for_accelerator(
            acc, count,
            use_spot=use_spot,
            region=job_config.region or None,
            clouds=cloud_name,
        )
        if not instance_list:
            raise LaunchError(
                f"No instance type found for {acc}:{count} on {cloud_name}. "
                f"Run 'sky show-gpus --cloud {cloud_name}' to see available types."
            )
        instance_type = instance_list[0]

        hourly_cost = catalog.get_hourly_cost(
            instance_type,
            use_spot=use_spot,
            region=job_config.region or None,
            zone=None,
            clouds=cloud_name,
        )
        return instance_type, hourly_cost

    def _get_budget_used(self) -> float:
        """Query SkyPilot cost_report for cumulative local spend.

        This is a best-effort estimate from SkyPilot's local cache.
        It tracks spend on THIS machine via local cluster records only.
        It does NOT reflect spend from other machines or clusters managed
        outside SkyPilot.  Do not treat this as a real-time budget guard.
        """
        try:
            import sky
            req = sky.cost_report()
            records = sky.get(req)
            return sum(float(r.get("total_cost", 0.0)) for r in records)
        except Exception:
            return 0.0

    def _check_cost_guards(
        self,
        policy: CostPolicy,
        hourly_cost: float,
        budget_used: float,
    ) -> None:
        """Enforce cost policy.  Raises ``CostLimitError`` on violation."""
        if hourly_cost > policy.max_cost_per_hour:
            raise CostLimitError(
                f"Instance hourly cost ${hourly_cost:.2f} exceeds limit "
                f"${policy.max_cost_per_hour:.2f}",
                limit=policy.max_cost_per_hour,
                actual=hourly_cost,
                guard_type="hourly",
            )
        if budget_used >= policy.max_total_budget:
            raise CostLimitError(
                f"Cumulative SkyPilot-tracked spend ${budget_used:.2f} "
                f"exceeds budget ${policy.max_total_budget:.2f}. "
                f"Note: this is a local estimate only — see cost_so_far() docs.",
                limit=policy.max_total_budget,
                actual=budget_used,
                guard_type="budget",
            )


def _is_spot_unavailable_error(exc: LaunchError) -> bool:
    """Return True if the error is specifically about spot capacity."""
    # Check both the LaunchError message and the original cause
    marker = "Failed to provision all possible launchable resources"
    if marker in str(exc):
        return True
    if exc.__cause__ is not None and marker in str(exc.__cause__):
        return True
    if "Spot instances unavailable" in str(exc):
        return True
    return False


def _format_launch_error(exc: Exception) -> str:
    """Format a SkyPilot launch exception into a concise message.

    Truncates the verbose per-region table from
    ``ResourcesUnavailableError`` while preserving other errors in full.
    """
    msg = str(exc)
    # The "Failed to provision" error includes a huge table of regions.
    # Truncate it to just the summary line.
    marker = "Failed to provision all possible launchable resources"
    if marker in msg:
        # Extract just the first line (the summary) and the resource spec
        lines = msg.split("\n")
        summary_parts = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            summary_parts.append(stripped)
            if stripped.startswith("To keep retrying") or len(summary_parts) >= 3:
                break
        return (
            "Spot instances unavailable across all regions. "
            "Use spot_fallback=True in CostPolicy to auto-retry on-demand, "
            "or set use_spot=False."
        )
    return f"SkyPilot launch failed: {exc}"


def _shell_quote(s: str) -> str:
    """Quote a string for safe shell use."""
    import shlex
    return shlex.quote(s)


def _resolve_sky_cloud_class(sky_module, cloud_name: str):
    """Resolve a SkyPilot cloud class by name (e.g. 'runpod' → sky.RunPod).

    Searches sky module attributes for a Cloud subclass whose ``_REPR``
    matches *cloud_name* (case-insensitive).  Falls back to title-cased
    ``getattr(sky, name.title())`` for common clouds.
    """
    # Try attribute scan first
    try:
        for attr in dir(sky_module):
            obj = getattr(sky_module, attr, None)
            if (isinstance(obj, type)
                    and hasattr(sky_module, 'clouds')
                    and hasattr(sky_module.clouds, 'Cloud')
                    and issubclass(obj, sky_module.clouds.Cloud)
                    and getattr(obj, '_REPR', '').lower() == cloud_name):
                return obj
    except (TypeError, AttributeError):
        pass

    # Fallback: try common capitalisations
    for name_variant in [cloud_name.title(), cloud_name.upper(), cloud_name]:
        cls = getattr(sky_module, name_variant, None)
        if cls is not None:
            return cls

    return None


def _load_credentials_file(path: Path) -> dict:
    """Load a cloud_credentials.yaml file."""
    if not path.exists():
        raise CredentialError(
            f"Credentials file not found: {path}\n"
            f"Create it from the example:\n"
            f"  cp src/maddening/examples/cloud/config/cloud_credentials.example.yaml {path}"
        )
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise CredentialError(
            f"Credentials file is not a valid YAML mapping: {path}"
        )
    data.pop("version", None)
    return data
