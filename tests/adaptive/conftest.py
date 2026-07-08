"""Scope ``jax_enable_x64`` per-test for the solver-utils / adaptive suite.

These tests compare ``jax.grad`` against finite differences to tight tolerances
(down to ~1e-9), which requires float64.  We toggle ``jax_enable_x64`` around
each individual test rather than session-wide, so the setting does not leak into
other test modules that rely on the default float32 dtype.
"""

from __future__ import annotations

import jax
import pytest


@pytest.fixture(autouse=True)
def _enable_x64_for_each_adaptive_test():
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)
