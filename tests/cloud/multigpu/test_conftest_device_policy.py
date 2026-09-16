"""The multigpu conftest forces virtual host devices only when no
accelerator is about to be used, and its override is honoured.

The policy is a pure function of the environment plus two host probes
(CUDA jaxlib plugin importable, ``nvidia-smi`` lists a GPU), checked here
for every rule; the end-to-end cases import the conftest in a fresh
subprocess with a controlled environment and, where the CPU backend is
selected, import JAX to confirm the device count JAX actually sees.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_CONFTEST = Path(__file__).with_name("conftest.py")


def _policy():
    spec = importlib.util.spec_from_file_location("multigpu_conftest_under_test", _CONFTEST)
    module = importlib.util.module_from_spec(spec)
    # Loading runs ``apply()`` on a copy of the environment, not on os.environ.
    saved = dict(os.environ)
    try:
        spec.loader.exec_module(module)
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return module


@pytest.mark.parametrize(
    "env, cuda_jaxlib, nvidia_gpu, expected",
    [
        # rule 1: explicit override wins over everything
        ({"MADDENING_VIRTUAL_DEVICES": "8", "JAX_PLATFORMS": "cuda"}, True, True, 8),
        ({"MADDENING_VIRTUAL_DEVICES": "0", "JAX_PLATFORMS": "cpu"}, False, False, None),
        # rule 2: a pre-set flag is respected
        ({"XLA_FLAGS": "--xla_force_host_platform_device_count=4", "JAX_PLATFORMS": "cpu"},
         False, False, None),
        # rule 3: an accelerator platform, alone or in a list
        ({"JAX_PLATFORMS": "cuda"}, True, True, None),
        ({"JAX_PLATFORMS": "gpu"}, False, False, None),
        ({"JAX_PLATFORMS": "cuda,cpu"}, True, True, None),
        ({"JAX_PLATFORMS": "TPU"}, False, False, None),
        # rule 4: cpu requested
        ({"JAX_PLATFORMS": "cpu"}, True, True, 16),
        # rule 5: unset -- all three probes must agree for an accelerator
        ({}, True, True, None),
        ({}, False, True, 16),                       # GPU laptop, CPU-only jaxlib
        ({}, True, False, 16),                       # plugin but no driver / no GPU
        ({"CUDA_VISIBLE_DEVICES": ""}, True, True, 16),
        ({"CUDA_VISIBLE_DEVICES": "-1"}, True, True, 16),
        ({"CUDA_VISIBLE_DEVICES": "0,1"}, True, True, None),
    ],
)
def test_virtual_device_count_rules(env, cuda_jaxlib, nvidia_gpu, expected):
    policy = _policy()
    assert policy.virtual_device_count(env, cuda_jaxlib=cuda_jaxlib,
                                       nvidia_gpu=nvidia_gpu) == expected


def test_apply_appends_to_existing_xla_flags_without_clobbering():
    policy = _policy()
    env = {"JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_cpu_enable_fast_math=false"}
    assert policy.apply(env) == 16
    assert env["XLA_FLAGS"] == (
        "--xla_cpu_enable_fast_math=false --xla_force_host_platform_device_count=16")
    # a second application is a no-op (rule 2)
    assert policy.apply(env) is None
    assert env["XLA_FLAGS"].count("--xla_force_host_platform_device_count") == 1


def _run(env_overrides: dict, code: str) -> str:
    env = {k: v for k, v in os.environ.items()
           if k not in ("XLA_FLAGS", "JAX_PLATFORMS", "MADDENING_VIRTUAL_DEVICES",
                        "CUDA_VISIBLE_DEVICES")}
    env.update(env_overrides)
    prelude = (
        "import importlib.util, os, sys\n"
        f"spec = importlib.util.spec_from_file_location('mgc', {str(_CONFTEST)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
    )
    out = subprocess.run([sys.executable, "-c", prelude + code], env=env,
                         capture_output=True, text=True, timeout=180, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip()


@pytest.mark.parametrize(
    "env, expected_devices",
    [
        ({"JAX_PLATFORMS": "cpu"}, 16),
        ({"JAX_PLATFORMS": "cpu", "MADDENING_VIRTUAL_DEVICES": "0"}, 1),
        ({"JAX_PLATFORMS": "cpu", "MADDENING_VIRTUAL_DEVICES": "6"}, 6),
        ({"JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=4"}, 4),
    ],
)
def test_cpu_backend_sees_the_forced_device_count(env, expected_devices):
    got = _run(env, "import jax; print(len(jax.devices()))")
    assert int(got) == expected_devices


def test_accelerator_platform_leaves_xla_flags_alone():
    # JAX is not imported here: a requested cuda backend may not exist on
    # the test host, and the point is what the conftest does *before* JAX.
    got = _run({"JAX_PLATFORMS": "cuda", "XLA_FLAGS": "--xla_gpu_autotune_level=0"},
               "print(os.environ['XLA_FLAGS'])")
    assert got == "--xla_gpu_autotune_level=0"
