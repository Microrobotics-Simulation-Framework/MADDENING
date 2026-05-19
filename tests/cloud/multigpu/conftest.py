"""Configure virtual CPU devices for multi-GPU tests.

The XLA_FLAGS env var MUST be set before JAX is imported.  This conftest
runs before any test module in this directory.

We default to 16 virtual devices so 2-D pencil tests (which use 2x4 = 8
or 4x4 = 16 device meshes) can run locally without real GPUs.  Tests
that need fewer devices simply pick a subset.

Users can override the count via ``XLA_FLAGS`` in the environment before
launching pytest -- this conftest only appends the device-count flag
when one is not already present.
"""

import os

_HOST_FLAG = "--xla_force_host_platform_device_count"
_existing = os.environ.get("XLA_FLAGS", "")

if _HOST_FLAG not in _existing:
    os.environ["XLA_FLAGS"] = (_existing + f" {_HOST_FLAG}=16").strip()
