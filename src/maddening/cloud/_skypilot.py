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

.. versionchanged:: 0.4.0
   Ported to SkyPilot's client-server API (MADD-ANO-016).  This module was
   byte-identical from v0.1.0 and written against the pre-0.7 API, while the
   declared floor has always been ``skypilot>=0.11``: ``sky.launch`` lost
   ``detach_run``, and ``launch``/``status``/``down`` return a ``RequestId``
   (a ``str`` subclass) that must be resolved with ``sky.get`` or
   ``sky.stream_and_get``.  Indexing the ``RequestId`` therefore yielded a
   character, and calling ``.get()`` on it raised ``AttributeError``.
   ``maddening.cloud.launcher`` has used the current API since v0.2.0 and is
   the reference this port follows.

   **The signatures are now verified against the installed SkyPilot**
   (``tests/cloud/test_skypilot_api_contract.py``).  The end-to-end behaviour
   — a teardown that really releases the VM, a preemption callback that
   really fires — needs a live provider and is *not* verified; MADD-ANO-016
   stays open for that.
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
    from maddening.cloud.launcher import _resolve_sky_cloud_class

    ports = list(getattr(config, "ports", None) or ())
    task = sky.Task(
        run=f"docker run --gpus all {_port_flags(ports)}{_env_flags(envs)}"
            f"-e MADDENING_CLOUD_CONFIG='{{}}' "
            f"{config.container_image}",
        envs=dict(envs) if envs else None,
    )
    # ``getattr(sky, config.cloud.upper())`` used to live here with an
    # ``or sky.GCP()`` fallback.  No SkyPilot cloud class is spelled in
    # upper case -- it is ``sky.RunPod``, not ``sky.RUNPOD`` -- so the
    # lookup always missed and *every* CloudSession launch silently went to
    # GCP, whatever the config said.  The resolver is launcher.py's, and an
    # unresolvable name is refused rather than redirected.
    cloud_name = str(getattr(config, "cloud", "") or "").lower()
    cloud_cls = _resolve_sky_cloud_class(sky, cloud_name)
    if cloud_cls is None:
        raise ValueError(
            f"No SkyPilot cloud class for {cloud_name!r}. Refusing to launch: "
            f"this used to fall back to GCP, which bills a provider the "
            f"caller did not ask for."
        )
    resources = sky.Resources(
        cloud=cloud_cls(),
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
    # ``sky.launch`` returns a RequestId; the work happens when it is
    # resolved.  ``stream_and_get`` rather than ``get`` for the same reason
    # launcher.py gives: ``get`` can raise a spurious AssertionError during
    # provisioning.  It returns ``(job_id, handle)`` once the cluster is up
    # and the run has been submitted -- which is what the old
    # ``detach_run=True`` asked for, and is now the only behaviour.
    request_id = sky.launch(task, cluster_name=cluster_name)
    result = sky.stream_and_get(request_id)
    handle = result[1] if result is not None else None

    vm_ip = getattr(handle, "head_ip", None)
    if not vm_ip:
        # The handle did not carry an IP (some backends fill it in late).
        # Ask for the cluster record, resolving the RequestId as before.
        records = sky.get(sky.status(cluster_names=[cluster_name]))
        if records:
            status_handle = records[0].get("handle")
            vm_ip = getattr(status_handle, "head_ip", None)

    return vm_ip or "unknown", cluster_name


def check_status(job_id: str) -> str:
    """The SkyPilot cluster status for *job_id*, as a bare status name.

    Parameters
    ----------
    job_id : str
        The SkyPilot *cluster* name (what :func:`launch_vm` returns).

    Returns
    -------
    str
        ``"UP"``, ``"INIT"``, ``"STOPPED"``, ... — the ``ClusterStatus``
        value, not its ``repr`` — or ``"not_found"`` when SkyPilot knows no
        such cluster.

    Notes
    -----
    ``str(ClusterStatus.UP)`` is ``"ClusterStatus.UP"`` and ``.value`` is
    ``"UP"``; :func:`monitor_preemption` compares against the bare names, so
    the ``.value`` is what this returns.  The pre-port version returned
    ``status[0].get("status", "unknown")`` on a ``RequestId``, which raised
    ``AttributeError`` rather than returning anything at all.
    """
    sky = _import_sky()

    records = sky.get(sky.status(cluster_names=[job_id]))
    if not records:
        return "not_found"
    status = records[0].get("status")
    if status is None:
        return "unknown"
    return str(getattr(status, "value", status))


class TeardownError(RuntimeError):
    """A SkyPilot teardown did not complete, so the VM may still be running.

    Raised by :func:`teardown_vm`.  The previous version logged the failure
    and returned normally, which is indistinguishable from success to every
    caller — and the call it was hiding could never have worked, so a
    ``CloudSession.teardown()`` reported a released VM while the instance
    kept billing.
    """


def teardown_vm(job_id: str) -> None:
    """Tear down a SkyPilot cluster, or raise.

    Parameters
    ----------
    job_id : str
        The SkyPilot cluster name.

    Raises
    ------
    TeardownError
        The teardown request failed.  **The VM may still be running and
        still be billing**; the cluster name is in the message so it can be
        torn down by hand (``sky down <name>``) or retried.

    Notes
    -----
    This used to be ``except Exception: logger.exception(...)``.  A broad
    except around a call that cannot succeed turns a leaked VM into a log
    line, so the failure is now the caller's to handle.
    """
    sky = _import_sky()

    try:
        sky.get(sky.down(job_id, purge=True))
    except Exception as exc:
        raise TeardownError(
            f"SkyPilot teardown of cluster {job_id!r} failed: {exc}. "
            f"The VM may still be running and billing -- check with "
            f"`sky status` and tear it down with `sky down {job_id}`."
        ) from exc


#: Consecutive failing status checks before the preemption monitor gives up.
#: One transient error is normal; a run of them means the check is broken,
#: and a monitor that cannot read the status is not monitoring anything.
MAX_CONSECUTIVE_STATUS_ERRORS = 3


def monitor_preemption(
    job_id: str,
    callback: Callable[[], None],
    poll_interval: float = 5.0,
    max_consecutive_errors: int = MAX_CONSECUTIVE_STATUS_ERRORS,
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
    max_consecutive_errors : int
        Give up after this many status checks in a row have raised.

    Returns
    -------
    threading.Thread
        The monitoring thread, already started.

    Notes
    -----
    The loop used to swallow every exception at ``logger.debug`` and carry
    on forever.  Because :func:`check_status` could not work at all against
    the supported SkyPilot versions, that meant the callback could never
    fire and nothing above ``DEBUG`` ever said so — a spot VM could be
    preempted and the session would wait for a container that no longer
    existed.

    A failing check is now logged at ``ERROR`` the first time, and after
    *max_consecutive_errors* in a row the thread stops with a final
    ``ERROR`` saying preemption is no longer being watched.  It does **not**
    invoke *callback* on an error: a broken status check is not evidence of
    preemption, and treating it as such would tear down healthy sessions.
    """
    def _monitor():
        consecutive_errors = 0
        while True:
            time.sleep(poll_interval)
            try:
                status = check_status(job_id)
            except Exception:
                consecutive_errors += 1
                logger.error(
                    "Preemption check %d/%d failed for %s",
                    consecutive_errors, max_consecutive_errors, job_id,
                    exc_info=True,
                )
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(
                        "Giving up on the preemption monitor for %s after %d "
                        "consecutive failures: preemption is NOT being "
                        "watched for this cluster.",
                        job_id, consecutive_errors,
                    )
                    return
                continue
            consecutive_errors = 0
            # SkyPilot has no PREEMPTED ClusterStatus (0.12): a preempted
            # spot instance shows up as STOPPED or disappears entirely.
            # PREEMPTED is kept in case a provider backend grows it.
            if status in ("STOPPED", "not_found", "PREEMPTED"):
                logger.warning("Preemption detected for %s (status=%s)",
                               job_id, status)
                callback()
                return

    thread = threading.Thread(target=_monitor, daemon=True, name=f"preemption-{job_id}")
    thread.start()
    return thread
