"""Scope ``jax_enable_x64`` per-test for the adaptive-node suite.

The adaptive tests require float64 to compare ``jax.grad`` against
finite differences to ~1e-5 (the round-7 spike tolerance).  We
toggle ``jax_enable_x64`` around each individual test rather than
session-wide, so the setting does not leak into other test modules
that rely on the default float32 dtype.

The mock subclass and the public toys (``TopKAdaptiveNode``,
``HierarchicalHatAdaptiveNode``) all build their static basis
arrays lazily on first construction, so this fixture's
just-in-time x64 enable is sufficient.
"""

from __future__ import annotations

import jax
import pytest


@pytest.fixture(autouse=True)
def _enable_x64_for_each_adaptive_test():
    """Enable ``jax_enable_x64`` for the duration of one test, then
    restore the prior value."""
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)
