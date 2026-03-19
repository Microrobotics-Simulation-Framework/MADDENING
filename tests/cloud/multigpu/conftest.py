"""Configure 2 CPU devices for multi-GPU tests.

The XLA_FLAGS env var MUST be set before JAX is imported.
This conftest runs before any test module in this directory.
"""

import os

# Force 2 CPU devices for testing multi-GPU logic without real GPUs.
# This must happen before JAX is first imported anywhere in the process.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") +
    " --xla_force_host_platform_device_count=2"
).strip()
