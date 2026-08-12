"""Hypothesis verification test configuration.

JAX JIT compilation causes the first execution of any test to take
hundreds of milliseconds. We disable Hypothesis's deadline globally
for these tests rather than annotating each one.
"""

from hypothesis import settings

settings.register_profile(
    "jax", deadline=None, print_blob=True,
)
settings.load_profile("jax")
