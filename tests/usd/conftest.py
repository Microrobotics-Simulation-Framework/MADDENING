"""Shared fixtures for USD tests."""

import os
# Force CPU backend for tests
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# CRITICAL: import maddening.usd BEFORE any other USD operations
# to register the codeless schemas.
import maddening.usd  # noqa: F401, E402
