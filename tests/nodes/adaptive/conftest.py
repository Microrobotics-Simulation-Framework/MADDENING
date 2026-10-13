"""Enable ``jax_enable_x64`` per test for the adaptive-node suite.

The gradient checks compare ``jax.grad`` against central finite
differences to 1e-5, which needs float64.  The flag is toggled around
each test (not at import time) so it never leaks into the modules that
rely on the default float32.
"""

from __future__ import annotations

import jax
import pytest


@pytest.fixture(autouse=True)
def _x64_per_test():
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)
