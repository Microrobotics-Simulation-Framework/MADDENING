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
import threading
import time
from typing import Callable, Optional

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


def launch_vm(config) -> tuple[str, str]:
    """Provision a VM via SkyPilot.

    Only the ports in ``config.ports`` are published from the container
    to the VM's interfaces.  That list is empty by default, so a
    launched job exposes nothing and is reached over an SSH tunnel.

    .. versionchanged:: 0.4.0
       This function published ``8000``, ``8080``, ``5555`` and ``5556``
       unconditionally, ignoring ``JobConfig.ports`` entirely -- so the
       API, the unauthenticated state stream and the unauthenticated
       command channel were all on the VM's public interface on every
       launch.  ``8080`` was published for a health endpoint that no
       component in this package ever served, and is simply dropped.

    Returns ``(vm_ip, job_id)``.
    """
    sky = _import_sky()

    ports = list(getattr(config, "ports", None) or ())
    task = sky.Task(
        run=f"docker run --gpus all {_port_flags(ports)}"
            f"-e MADDENING_CLOUD_CONFIG='{{}}' "
            f"{config.container_image}",
    )
    resources = sky.Resources(
        cloud=getattr(sky, config.cloud.upper(), None) or sky.GCP(),
        instance_type=config.instance_type if config.instance_type else None,
        accelerators=config.accelerator if config.accelerator else None,
        use_spot=config.spot,
        region=config.region if config.region else None,
        ports=ports or None,
    )
    task.set_resources(resources)

    cluster_name = f"maddening-{int(time.time())}"
    job_id = sky.launch(task, cluster_name=cluster_name, detach_run=True)

    # Get the VM IP
    status = sky.status(cluster_names=[cluster_name])
    if status:
        vm_ip = status[0].get("handle", {}).get("head_ip", "")
        if not vm_ip:
            vm_ip = status[0].get("head_ip", "unknown")
    else:
        vm_ip = "unknown"

    return vm_ip, cluster_name


def check_status(job_id: str) -> str:
    """Check the status of a SkyPilot cluster."""
    sky = _import_sky()

    status = sky.status(cluster_names=[job_id])
    if not status:
        return "not_found"
    return status[0].get("status", "unknown")


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
