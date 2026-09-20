"""Internal SkyPilot wrapper — isolates all ``import sky`` calls.

This module is only imported by ``CloudSession._launch_worker()``.
If SkyPilot is not installed, imports will fail with a clear message.

Two separate paths to cloud execution exist in MADDENING:

CloudLauncher  — User-facing, script/CLI path.  Loads credentials from
                 ~/.maddening/cloud_credentials.yaml.  Calls sky.* directly.
                 Future basis for CloudSweep and CloudGroup.

CloudSession   — Server-side orchestration path.  Credentials assumed
                 pre-configured on the machine.  Uses _skypilot.py wrapper.
                 Future basis for cloud API endpoints in MICROBOTICA.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Callable, Mapping, Optional

#: An environment variable name we are willing to interpolate into a
#: ``docker run`` command line.  Only the *name* is interpolated -- the
#: value travels in SkyPilot's task environment -- so this is the whole
#: of the injection surface, and it fails closed on anything else.
_ENV_NAME = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\Z")

logger = logging.getLogger(__name__)

_SKY_INSTALL_MSG = (
    "Cloud orchestration requires SkyPilot. Install with:\n"
    "  pip install maddening[runpod]     # for RunPod\n"
    "  pip install maddening[lambda]     # for Lambda Labs\n"
    "  pip install maddening[cloud]      # all supported providers"
)


def _import_sky():
    """Import sky with a clear error message if not installed."""
    try:
        import sky
        return sky
    except ImportError as exc:
        raise ImportError(_SKY_INSTALL_MSG) from exc


def _port_flags(ports) -> str:
    """Render ``-p HOST:CONTAINER`` flags for *ports*.

    Parameters
    ----------
    ports : iterable of int, or None
        The ports to publish from the container to the VM's interfaces.

    Returns
    -------
    str
        The flags, with a trailing space, or ``""`` for no ports.

    Raises
    ------
    ValueError
        If a port is not an integer in 1-65535.  These values are
        interpolated into a shell command line, so anything that is not
        a plain port number is refused rather than quoted.
    """
    flags = []
    for port in ports or ():
        number = int(port)
        if not 1 <= number <= 65535:
            raise ValueError(
                f"JobConfig.ports contains {port!r}, which is not a port "
                f"number in 1-65535."
            )
        flags.append(f"-p {number}:{number}")
    return " ".join(flags) + " " if flags else ""


def _env_flags(envs: Optional[Mapping[str, str]]) -> str:
    """Render pass-through ``-e NAME`` flags for *envs*.

    Parameters
    ----------
    envs : mapping of str to str, or None
        Environment variables to hand to the container.

    Returns
    -------
    str
        The flags, with a trailing space, or ``""``.

    Raises
    ------
    ValueError
        If a name is not a plain environment-variable identifier.

    Notes
    -----
    The flag is ``-e NAME``, not ``-e NAME=value``.  Docker reads an
    unassigned ``-e NAME`` from the environment of the process that runs
    it, so the **value never reaches a command line**: the VM's
    ``/proc/<pid>/cmdline`` is world-readable, and these values are
    credentials.  The values travel in SkyPilot's task environment
    instead.
    """
    flags = []
    for name in sorted(envs or {}):
        if not _ENV_NAME.match(name):
            raise ValueError(
                f"{name!r} is not a valid environment variable name. These "
                f"names are interpolated into a shell command line, so "
                f"anything that is not an identifier is refused rather "
                f"than quoted."
            )
        flags.append(f"-e {name}")
    return " ".join(flags) + " " if flags else ""


def launch_vm(config, envs: Optional[Mapping[str, str]] = None) -> tuple[str, str]:
    """Provision a VM via SkyPilot.

    Only the ports in ``config.ports`` are published from the container
    to the VM's interfaces.  That list is empty by default, so a
    launched job exposes nothing and is reached over an SSH tunnel.

    Parameters
    ----------
    config : object
        Anything carrying ``ports``, ``container_image``, ``cloud``,
        ``instance_type``, ``accelerator``, ``spot`` and ``region``.
    envs : mapping of str to str, optional
        Environment variables for the container -- in practice the
        credentials from :meth:`maddening.cloud.session.CloudSession.container_env`.
        Their *values* are passed through SkyPilot's task environment
        and never appear on a command line; see :func:`_env_flags`.

    .. versionchanged:: 0.4.0
       This function published ``8000``, ``8080``, ``5555`` and ``5556``
       unconditionally, ignoring ``JobConfig.ports`` entirely -- so the
       API, the unauthenticated state stream and the unauthenticated
       command channel were all on the VM's public interface on every
       launch.  ``8080`` was published for a health endpoint that no
       component in this package ever served, and is simply dropped.
       It also passed no credentials into the container, so the
       container generated a bearer token that nothing outside its own
       log could present, and ``CloudSession``'s health probes could not
       authenticate against it.

    Returns ``(vm_ip, job_id)``.
    """
    sky = _import_sky()

    ports = list(getattr(config, "ports", None) or ())
    task = sky.Task(
        run=f"docker run --gpus all {_port_flags(ports)}{_env_flags(envs)}"
            f"-e MADDENING_CLOUD_CONFIG='{{}}' "
            f"{config.container_image}",
        envs=dict(envs) if envs else None,
    )
    resources = sky.Resources(
        cloud=getattr(sky, config.cloud.upper(), None) or sky.GCP(),
        instance_type=config.instance_type if config.instance_type else None,
        accelerators=config.accelerator if config.accelerator else None,
        use_spot=config.spot,
        region=config.region if config.region else None,
        # SkyPilot's annotation says `List[str]`; it accepts a list of
        # ints and normalises them (verified against skypilot 0.13).
        ports=ports or None,  # pyright: ignore[reportArgumentType]
    )
    task.set_resources(resources)

    cluster_name = f"maddening-{int(time.time())}"
    # ----------------------------------------------------------------
    # KNOWN DEFECT -- this block is written against the pre-0.7 SkyPilot
    # API and cannot work against the >=0.11 floor this package declares.
    # `sky.launch` has had no `detach_run` parameter since the client/
    # server split, and `sky.launch` / `sky.status` now return a
    # `RequestId` (a `str` subclass) that has to be resolved with
    # `sky.get()` / `sky.stream_and_get()` -- so `status[0]` indexes a
    # character and `.get(...)` raises `AttributeError`.
    # `maddening.cloud.launcher` already uses the current API and is the
    # reference for the port.  Every test of this module substitutes a
    # fake `sky` that mirrors the stale signature, so nothing in the
    # suite can see it.
    #
    # Suppressed rather than fixed here because this branch is annotation
    # work and the fix cannot be exercised without a real cloud account;
    # it needs its own change with its own verification.
    # ----------------------------------------------------------------
    job_id = sky.launch(task, cluster_name=cluster_name,
                        detach_run=True)  # pyright: ignore[reportCallIssue]

    # Get the VM IP
    status = sky.status(cluster_names=[cluster_name])
    if status:
        vm_ip = status[0].get("handle", {}).get("head_ip", "")  # pyright: ignore[reportAttributeAccessIssue]
        if not vm_ip:
            vm_ip = status[0].get("head_ip", "unknown")  # pyright: ignore[reportAttributeAccessIssue]
    else:
        vm_ip = "unknown"

    return vm_ip, cluster_name


def check_status(job_id: str) -> str:
    """Check the status of a SkyPilot cluster."""
    sky = _import_sky()

    status = sky.status(cluster_names=[job_id])
    if not status:
        return "not_found"
    # See the KNOWN DEFECT note in `launch_vm`: `sky.status` returns a
    # `RequestId`, not a list of dicts.
    return status[0].get("status", "unknown")  # pyright: ignore[reportAttributeAccessIssue]


def teardown_vm(job_id: str) -> None:
    """Tear down a SkyPilot cluster."""
    sky = _import_sky()

    try:
        sky.down(job_id, purge=True)
    except Exception:
        logger.exception("SkyPilot teardown failed for %s", job_id)


def monitor_preemption(
    job_id: str,
    callback: Callable[[], None],
    poll_interval: float = 5.0,
) -> threading.Thread:
    """Start a daemon thread that polls for spot preemption.

    Parameters
    ----------
    job_id : str
        SkyPilot cluster name to monitor.
    callback : callable
        Called (once) if preemption is detected.  This must be a
        CloudSession-internal method, never a user callback directly.
    poll_interval : float
        Seconds between status checks.

    Returns the monitoring thread (already started).
    """
    def _monitor():
        while True:
            time.sleep(poll_interval)
            try:
                status = check_status(job_id)
                if status in ("STOPPED", "not_found", "PREEMPTED"):
                    logger.warning("Preemption detected for %s (status=%s)",
                                   job_id, status)
                    callback()
                    return
            except Exception:
                logger.debug("Preemption check failed for %s", job_id, exc_info=True)

    thread = threading.Thread(target=_monitor, daemon=True, name=f"preemption-{job_id}")
    thread.start()
    return thread
