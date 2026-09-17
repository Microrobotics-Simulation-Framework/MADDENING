"""The multigpu conftest forces virtual host devices only when no
accelerator is about to be used, and its override is honoured.

The policy is a pure function of the environment plus two host probes
(CUDA jaxlib plugin importable, ``nvidia-smi`` lists a GPU), checked here
for every rule; the end-to-end cases import the conftest in a fresh
subprocess with a controlled environment and, where the CPU backend is
selected, import JAX to confirm the device count JAX actually sees.  The
conftest must also be the *only* module in this directory that touches
``XLA_FLAGS``: a per-module default would win whenever the policy decides
to force nothing.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_CONFTEST = _HERE / "conftest.py"


def _policy():
    spec = importlib.util.spec_from_file_location("multigpu_conftest_under_test", _CONFTEST)
    module = importlib.util.module_from_spec(spec)
    # Loading runs ``apply()`` on the real ``os.environ`` (a no-op under
    # pytest, where the flag is already present); the environment is
    # snapshotted and restored so nothing leaks either way.
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


@pytest.mark.parametrize("bad", ["abc", "-3", "1.5", "4 devices"])
def test_override_must_be_a_non_negative_integer(bad):
    policy = _policy()
    env = {"MADDENING_VIRTUAL_DEVICES": bad, "JAX_PLATFORMS": "cpu"}
    with pytest.raises(ValueError, match=r"MADDENING_VIRTUAL_DEVICES must be a non-negative integer"):
        policy.virtual_device_count(env, cuda_jaxlib=False, nvidia_gpu=False)
    # under pytest the same mistake is a usage error (collection stops
    # with the message), not a traceback out of int()
    with pytest.raises(pytest.UsageError, match=re.escape(repr(bad))):
        policy.apply(dict(env))
    assert "XLA_FLAGS" not in env
    # a negative value is never "disable"; only 0 is
    assert policy.parse_override(" 0 ") == 0
    assert policy.parse_override("6") == 6


def test_apply_appends_to_existing_xla_flags_without_clobbering():
    policy = _policy()
    env = {"JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_cpu_enable_fast_math=false"}
    assert policy.apply(env) == 16
    assert env["XLA_FLAGS"] == (
        "--xla_cpu_enable_fast_math=false --xla_force_host_platform_device_count=16")
    # a second application is a no-op (rule 2)
    assert policy.apply(env) is None
    assert env["XLA_FLAGS"].count("--xla_force_host_platform_device_count") == 1


# an assignment to XLA_FLAGS (setdefault / item assignment / putenv), not a read
_SETS_XLA_FLAGS = re.compile(
    r"""os\.environ\s*\.setdefault\s*\(\s*["']XLA_FLAGS["']"""
    r"""|os\.environ\s*\[\s*["']XLA_FLAGS["']\s*\]\s*=(?!=)"""
    r"""|os\.putenv\s*\(\s*["']XLA_FLAGS["']""")


def test_only_the_conftest_sets_the_host_device_flag():
    """A per-module ``setdefault("XLA_FLAGS", ...)`` runs after the conftest
    and before the backend initialises, so it silently overrides rule 1
    (``MADDENING_VIRTUAL_DEVICES=0``) and rule 3 (accelerator requested)."""
    offenders = []
    for path in sorted(_HERE.glob("*.py")):
        if path.name == "conftest.py":
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _SETS_XLA_FLAGS.search(line):
                offenders.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


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
                         capture_output=True, text=True, timeout=300, check=False)
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


def test_override_zero_yields_single_device_even_after_test_modules_import():
    """Rule 1's ``0`` must survive the import of every test module in the
    directory (pytest's order: conftest, then the modules, then the
    backend initialises on first use)."""
    modules = sorted(p for p in _HERE.glob("test_*.py"))
    assert modules
    code = (
        f"for i, p in enumerate({[str(p) for p in modules]!r}):\n"
        "    s = importlib.util.spec_from_file_location(f'mg_mod_{i}', p)\n"
        "    mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod)\n"
        "import jax\n"
        "print(len(jax.devices()), os.environ.get('XLA_FLAGS', ''))\n"
    )
    got = _run({"JAX_PLATFORMS": "cpu", "MADDENING_VIRTUAL_DEVICES": "0"}, code)
    n_devices, _, flags = got.partition(" ")
    assert int(n_devices) == 1, got
    assert "--xla_force_host_platform_device_count" not in flags, got


def test_accelerator_platform_leaves_xla_flags_alone():
    # JAX is not imported here: a requested cuda backend may not exist on
    # the test host, and the point is what the conftest does *before* JAX.
    got = _run({"JAX_PLATFORMS": "cuda", "XLA_FLAGS": "--xla_gpu_autotune_level=0"},
               "print(os.environ['XLA_FLAGS'])")
    assert got == "--xla_gpu_autotune_level=0"
