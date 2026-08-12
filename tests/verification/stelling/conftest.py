"""Enable float64 for stelling verification tests.

Stelling operates in real arithmetic semantics and requires float64
for meaningful interval bounds. This conftest enables x64 for the
duration of each test and restores the prior state afterward.
"""

import jax
import pytest


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)
