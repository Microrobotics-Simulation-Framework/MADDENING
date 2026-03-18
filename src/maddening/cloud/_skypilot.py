"""Internal SkyPilot wrapper — isolates all ``import sky`` calls.

This module is only imported by ``CloudSession._launch_worker()``.
If SkyPilot is not installed, imports will fail with a clear message.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def launch_vm(config) -> tuple[str, str]:
    """Provision a VM via SkyPilot.

    Returns ``(vm_ip, job_id)``.
    """
    import sky

    task = sky.Task(
        run=f"docker run --gpus all -p 8000:8000 -p 8080:8080 -p 5555:5555 -p 5556:5556 "
            f"-e MADDENING_CLOUD_CONFIG='{{}}' "
            f"{config.container_image}",
    )
    resources = sky.Resources(
        cloud=getattr(sky, config.cloud.upper(), None) or sky.GCP(),
        instance_type=config.instance_type if config.instance_type else None,
        accelerators=config.accelerator if config.accelerator else None,
        use_spot=config.spot,
        region=config.region if config.region else None,
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
    import sky

    status = sky.status(cluster_names=[job_id])
    if not status:
        return "not_found"
    return status[0].get("status", "unknown")


def teardown_vm(job_id: str) -> None:
    """Tear down a SkyPilot cluster."""
    import sky

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
