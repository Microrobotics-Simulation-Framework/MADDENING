"""The adaptive steppers' error norm divides each entry by its own scale.

``_tree_error_norm`` measures ``|fine - coarse| / (atol + rtol * max(|fine|,
|coarse|))`` and guarded the division with ``max(scale, 1e-300)``, an absolute
floor in the state's units.  In float32 it was inert (``1e-300`` rounds to 0),
but in float64 an entry below about ``1e-297`` (``1e-300 / rtol``) with
``atol=0`` was divided by ``1e-300`` instead of its scale: its error read up to
``rtol * |x| / 1e-300`` times too small, and the controller accepted steps it
should have rejected.  A zero scale still contributes nothing.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.simulation.adaptive import _tree_error_norm


@pytest.fixture
def x64():
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


@pytest.mark.parametrize("magnitude", [1.0, 1e-200, 1e-302])
def test_the_error_of_an_entry_does_not_depend_on_its_magnitude(x64, magnitude):
    fine = {"n": {"x": jnp.asarray([2.0 * magnitude], jnp.float64)}}
    coarse = {"n": {"x": jnp.asarray([1.0 * magnitude], jnp.float64)}}
    err = float(_tree_error_norm(fine, coarse, 0.0, 1e-3))
    # |2m - m| / (1e-3 * 2m) = 500, whatever m is.
    assert err == pytest.approx(500.0, rel=1e-12), magnitude


def test_a_zero_scale_contributes_nothing_and_counts_as_an_element():
    fine = {"n": {"x": jnp.zeros(3, jnp.float32), "y": jnp.asarray([2.0], jnp.float32)}}
    coarse = {"n": {"x": jnp.zeros(3, jnp.float32), "y": jnp.asarray([1.0], jnp.float32)}}
    err = float(_tree_error_norm(fine, coarse, 0.0, 1e-3))
    assert np.isfinite(err)
    assert err == pytest.approx(np.sqrt(500.0 ** 2 / 4), rel=1e-6)


def test_in_float32_the_norm_is_unchanged_at_every_magnitude():
    rng = np.random.default_rng(1)
    for mag in (1e-30, 1.0, 1e30):
        f = jnp.asarray(rng.standard_normal(16) * mag, jnp.float32)
        c = jnp.asarray(np.asarray(f) * (1 + 1e-3 * rng.standard_normal(16)), jnp.float32)
        got = float(_tree_error_norm({"n": {"x": f}}, {"n": {"x": c}}, 0.0, 1e-3))
        scale = 1e-3 * jnp.maximum(jnp.abs(f), jnp.abs(c))
        want = float(jnp.sqrt(jnp.mean((jnp.abs(f - c) / scale) ** 2)))
        assert got == want, mag
