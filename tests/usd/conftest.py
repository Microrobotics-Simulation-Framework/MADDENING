"""Shared fixtures for USD tests.

All tests in this directory require the ``pxr`` package (OpenUSD).
If it is not installed, every test is skipped automatically.
"""

import os
import pytest

# Force CPU backend for tests
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Skip the entire directory if pxr is not installed
pxr = pytest.importorskip("pxr", reason="usd-core not installed")

# CRITICAL: import maddening.usd BEFORE any other USD operations
# to register the codeless schemas.
import maddening.usd  # noqa: F401, E402
