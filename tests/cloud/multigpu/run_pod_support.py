"""Running ``benchmarks/multigpu/run_pod.py`` from a test, offline.

The runner launches nothing itself -- it runs on the machine that has the
GPUs -- but no test may be able to reach a cloud path, even by mistake.
So every run a test starts goes through :func:`run_pod`, which:

* refuses, before anything runs, any argument list that runs a goal
  without ``--dry-run`` (``--summarise`` reads JSON and runs no goal);
* points ``HOME`` at an empty directory, so no cloud credentials or
  SkyPilot state are visible;
* drops every cloud credential variable from the environment.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

#: Environment variables that could carry a cloud credential or select a
#: cloud account; dropped from every run.
CLOUD_ENV_PREFIXES = ("RUNPOD_", "AWS_", "GOOGLE_", "GCLOUD_", "CLOUDSDK_", "GCP_", "AZURE_",
                      "LAMBDA_", "SKYPILOT_", "SKY_", "OCI_", "IBM_")

_HOME: Path | None = None


def empty_home() -> Path:
    """One empty directory per test process, to stand in for ``HOME``."""
    global _HOME
    if _HOME is None:
        _HOME = Path(tempfile.mkdtemp(prefix="run_pod_home_"))
    # A run may leave a cache behind; never a credential or cloud state.
    held = {p.name for p in _HOME.iterdir()} & {".runpod", ".sky", ".aws", ".maddening",
                                                ".lambda_cloud", ".azure", ".oci"}
    assert not held, f"cloud state appeared in the test HOME {_HOME}: {sorted(held)}"
    return _HOME


def offline_env(pythonpath: str, n_devices: int = 4) -> dict:
    """The environment of a run: CPU, ``n_devices`` virtual devices,
    ``pythonpath`` first, an empty ``HOME``, no cloud variables."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("XLA_FLAGS", "JAX_PLATFORMS", "MADDENING_VIRTUAL_DEVICES", "HOME")
           and not k.upper().startswith(CLOUD_ENV_PREFIXES)}
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n_devices}"
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = pythonpath
    env["HOME"] = str(empty_home())
    return env


def checked_argv(argv) -> list[str]:
    """``argv`` unchanged, or ``AssertionError`` if it would run a goal
    without ``--dry-run``."""
    argv = [str(a) for a in argv]
    if "--summarise" not in argv or "--goal" in argv:
        if "--dry-run" not in argv:
            raise AssertionError(
                f"refusing to run run_pod.py without --dry-run: {argv}.  A test "
                "never runs a goal for real")
    return argv


def run_pod(runner: Path, argv, *, pythonpath: str, timeout: float,
            n_devices: int = 4) -> subprocess.CompletedProcess:
    """Run the runner offline (see the module docstring); not checked."""
    return subprocess.run([sys.executable, str(runner), *checked_argv(argv)],
                          env=offline_env(pythonpath, n_devices), capture_output=True,
                          text=True, timeout=timeout, check=False)
