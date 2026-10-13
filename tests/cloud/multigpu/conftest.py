"""Virtual CPU devices for the multi-device tests -- only when no
accelerator is going to be used.

``XLA_FLAGS=--xla_force_host_platform_device_count=N`` must be set
before JAX is imported; this conftest runs before any test module in
this directory, and must therefore decide *without importing JAX*
whether the tests are about to run on real accelerators.

Policy (:func:`virtual_device_count`), first rule that applies wins:

1. ``MADDENING_VIRTUAL_DEVICES=N`` forces ``N`` virtual host devices
   (``0`` disables the forcing altogether; anything but a non-negative
   integer is a usage error that stops collection with a message).  The
   explicit override for running the suite on virtual devices on a GPU
   host, or the other way round.
2. ``XLA_FLAGS`` already carries ``--xla_force_host_platform_device_count``:
   left untouched.
3. ``JAX_PLATFORMS`` names an accelerator (``cuda``, ``gpu``, ``rocm``,
   ``tpu``, possibly in a list such as ``cuda,cpu``): nothing is forced,
   the tests see the real devices and skip themselves where fewer than
   they need are visible.
4. ``JAX_PLATFORMS=cpu``: 16 virtual devices (2-D pencil tests use up to
   4x4 meshes; the others pick a subset).
5. ``JAX_PLATFORMS`` unset: an accelerator is expected only when all
   three hold -- a CUDA jaxlib plugin is importable, ``nvidia-smi -L``
   lists a GPU, and ``CUDA_VISIBLE_DEVICES`` does not hide every GPU
   (``""`` or ``-1``).  Then nothing is forced; otherwise JAX would fall
   back to CPU anyway, so 16 virtual devices are forced.

A GPU laptop with a CPU-only jaxlib therefore still gets the 16 virtual
devices (rule 5, no plugin), and a pod with ``jax[cuda12]`` runs the
multi-device tests on its GPUs.  Note that the root ``tests/conftest.py``
defaults ``JAX_PLATFORMS`` to ``cpu`` (rule 4) before this file runs, so
under pytest the GPUs are used only when ``JAX_PLATFORMS=cuda`` is
exported explicitly -- rule 5 covers direct imports and any future
conftest that stops setting the default.

This conftest is the only place under ``tests/cloud/multigpu`` that sets
``XLA_FLAGS`` (checked by ``test_conftest_device_policy.py``); a test
module that set its own default would silently override rules 1 and 3.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess

HOST_FLAG = "--xla_force_host_platform_device_count"
DEFAULT_VIRTUAL_DEVICES = 16
OVERRIDE_VAR = "MADDENING_VIRTUAL_DEVICES"
ACCELERATOR_PLATFORMS = ("cuda", "gpu", "rocm", "tpu")
CUDA_PLUGIN_MODULES = ("jax_cuda12_plugin", "jax_cuda13_plugin",
                       "jax_plugins.xla_cuda12", "jax_plugins.xla_cuda13")


def cuda_jaxlib_installed() -> bool:
    """A CUDA plugin for jaxlib is importable (does not import JAX)."""
    for name in CUDA_PLUGIN_MODULES:
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except (ImportError, ValueError):
            continue
    return False


def nvidia_gpu_present() -> bool:
    """``nvidia-smi -L`` runs and lists at least one GPU."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True,
                             timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and any(
        line.startswith("GPU ") for line in out.stdout.splitlines())


def gpus_hidden(environ) -> bool:
    """``CUDA_VISIBLE_DEVICES`` is set to hide every GPU."""
    value = environ.get("CUDA_VISIBLE_DEVICES")
    return value is not None and value.strip() in ("", "-1")


def virtual_device_count(environ, *, cuda_jaxlib: bool, nvidia_gpu: bool) -> int | None:
    """How many virtual host devices to force, or ``None`` to leave JAX alone.

    Pure function of the environment and the two host probes so the
    policy is testable without a GPU (see the module docstring for the
    rules).
    """
    override = environ.get(OVERRIDE_VAR)
    if override is not None and override.strip() != "":
        n = parse_override(override)
        return n if n > 0 else None
    if HOST_FLAG in environ.get("XLA_FLAGS", ""):
        return None
    platforms = [p.strip().lower() for p in environ.get("JAX_PLATFORMS", "").split(",")
                 if p.strip()]
    if any(p in ACCELERATOR_PLATFORMS for p in platforms):
        return None
    if platforms:                       # cpu (or anything else non-accelerator)
        return DEFAULT_VIRTUAL_DEVICES
    accelerator_expected = cuda_jaxlib and nvidia_gpu and not gpus_hidden(environ)
    return None if accelerator_expected else DEFAULT_VIRTUAL_DEVICES


def parse_override(value: str) -> int:
    """``MADDENING_VIRTUAL_DEVICES`` as an integer, or a clear error.

    Accepted: a non-negative integer (``0`` = force nothing).  Anything
    else -- text, a negative number -- raises ``ValueError`` naming the
    variable; :func:`apply` turns that into a pytest usage error so the
    directory fails collection with one line instead of a traceback.
    """
    try:
        n = int(value.strip())
    except ValueError:
        n = -1
    if n < 0:
        raise ValueError(
            f"{OVERRIDE_VAR} must be a non-negative integer (N>0 forces N virtual "
            f"host devices, 0 forces nothing); got {value!r}")
    return n


def apply(environ=os.environ) -> int | None:
    """Set ``XLA_FLAGS`` per the policy; returns the count forced (or None).

    A malformed override is reported as ``pytest.UsageError`` when pytest
    is importable (collection stops with the message), else re-raised.
    """
    unset = not environ.get("JAX_PLATFORMS", "").strip()
    needs_probe = (environ.get(OVERRIDE_VAR, "").strip() == ""
                   and HOST_FLAG not in environ.get("XLA_FLAGS", "") and unset)
    try:
        count = virtual_device_count(
            environ,
            cuda_jaxlib=cuda_jaxlib_installed() if needs_probe else False,
            nvidia_gpu=nvidia_gpu_present() if needs_probe else False,
        )
    except ValueError as e:
        try:
            import pytest  # noqa: PLC0415
        except ImportError:  # pragma: no cover - conftest imported outside pytest
            raise
        raise pytest.UsageError(str(e)) from e
    if count is not None:
        environ["XLA_FLAGS"] = (environ.get("XLA_FLAGS", "") + f" {HOST_FLAG}={count}").strip()
    return count


apply()
