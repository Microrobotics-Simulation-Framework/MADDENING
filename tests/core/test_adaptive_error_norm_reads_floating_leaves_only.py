"""The adaptive steppers' error norm measures floating state fields only.

``_tree_error_norm`` subtracted every leaf of the user state, so a ``bool``
raised ``TypeError`` in ``run_adaptive`` and ``run_adaptive_scan``, a
``uint32`` wrapped modulo ``2**32`` and the controller rejected almost
every step, and an ``int32`` counter -- ``k + 2`` after the two half steps
and ``k + 1`` after the full one, by construction -- added a term and an
element to the RMS and moved the accepted steps.  Integer, boolean and
PRNG-key leaves are now skipped: no term, no element.  The differential
harness pins the steppers end to end
(``tests/property/test_differential_schedules.py``); this pins the norm,
the legacy ``build_adaptive_step`` path that shares it, and an
all-floating state measured exactly as before.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.simulation.adaptive import (
    AdaptiveConfig,
    _tree_error_norm,
    build_adaptive_step,
)

ATOL, RTOL = 1e-4, 1e-3


def _reference_norm(fine, coarse):
    """The formula, restated in NumPy over the floating leaves only."""
    terms = []
    for f, c in zip(jax.tree.leaves(fine), jax.tree.leaves(coarse)):
        f, c = np.asarray(f), np.asarray(c)
        if not np.issubdtype(f.dtype, np.inexact):
            continue
        scale = ATOL + RTOL * np.maximum(np.abs(f), np.abs(c))
        terms.append((np.abs(f - c) / scale).ravel())
    t = np.concatenate(terms)
    return float(np.sqrt(np.mean(t.astype(np.float64) ** 2)))


def _floats():
    fine = {"a": {"x": jnp.array([1.0, 2.0], jnp.float32), "v": jnp.float32(0.5)}}
    coarse = {"a": {"x": jnp.array([1.0001, 1.9995], jnp.float32), "v": jnp.float32(0.5002)}}
    return fine, coarse


def _with(leaf_fine, leaf_coarse):
    fine, coarse = _floats()
    fine["a"]["leaf"] = leaf_fine
    coarse["a"]["leaf"] = leaf_coarse
    return fine, coarse


_LEAVES = {
    "bool": (jnp.asarray(True), jnp.asarray(False)),
    "uint32-wraps": (jnp.asarray(1, jnp.uint32), jnp.asarray(2, jnp.uint32)),
    "int32-counter": (jnp.asarray(2, jnp.int32), jnp.asarray(1, jnp.int32)),
    "int32-from-zero": (jnp.asarray(0, jnp.int32), jnp.asarray(1, jnp.int32)),
    "prng-key": (jax.random.key(0), jax.random.key(1)),
    "uint32-key-data": (jnp.array([0xDEADBEEF, 7], jnp.uint32),
                        jnp.array([0xDEADBEEF, 8], jnp.uint32)),
}


@pytest.mark.parametrize("kind", sorted(_LEAVES))
def test_a_non_floating_leaf_adds_neither_a_term_nor_an_element(kind):
    base = float(_tree_error_norm(*_floats(), ATOL, RTOL))
    got = float(_tree_error_norm(*_with(*_LEAVES[kind]), ATOL, RTOL))
    assert got == base
    assert got == pytest.approx(_reference_norm(*_floats()), rel=1e-6)


def test_an_all_floating_state_is_measured_by_the_formula():
    """Every floating element counts, including one equal in both estimates."""
    fine, coarse = _floats()
    fine["a"]["same"] = coarse["a"]["same"] = jnp.float32(3.0)
    got = float(_tree_error_norm(fine, coarse, ATOL, RTOL))
    assert got == pytest.approx(_reference_norm(fine, coarse), rel=1e-6)
    assert got < float(_tree_error_norm(*_floats(), ATOL, RTOL))   # diluted, as documented


def test_a_complex_leaf_is_still_measured():
    fine = {"a": {"z": jnp.array([1 + 1j], jnp.complex64)}}
    coarse = {"a": {"z": jnp.array([1 + 1.001j], jnp.complex64)}}
    assert float(_tree_error_norm(fine, coarse, ATOL, RTOL)) > 0.0


def test_the_norm_traces_under_jit_with_every_leaf_kind():
    fine, coarse = _with(*_LEAVES["bool"])
    fine["a"]["tag"], coarse["a"]["tag"] = _LEAVES["uint32-wraps"]
    got = jax.jit(lambda f, c: _tree_error_norm(f, c, ATOL, RTOL))(fine, coarse)
    # Compiled against eager: the same arithmetic, fused differently.
    assert float(got) == pytest.approx(float(_tree_error_norm(*_floats(), ATOL, RTOL)),
                                       rel=1e-6)


def test_build_adaptive_step_steps_a_state_with_a_flag_and_a_counter():
    """The legacy builder shares the norm: the leaves change nothing it decides."""
    def dt_step(state, ext, dt):
        x = state["n"]["x"]
        new = {"x": x + dt * (-x), "flag": jnp.logical_not(state["n"]["flag"]),
               "count": state["n"]["count"] + jnp.int32(1)}
        return {"n": new}

    step = build_adaptive_step(None, AdaptiveConfig(atol=1e-4, rtol=1e-4), ["n"])
    plain = {"n": {"x": jnp.float32(1.0)}}

    def dt_plain(state, ext, dt):
        return {"n": {"x": state["n"]["x"] + dt * (-state["n"]["x"])}}

    s = {"n": {"x": jnp.float32(1.0), "flag": jnp.asarray(True),
               "count": jnp.asarray(2**24 + 1, jnp.int32)}}
    got = step(s, jnp.float32(0.1), {}, dt_step)
    ref = step(plain, jnp.float32(0.1), {}, dt_plain)
    for i in (1, 2, 3):        # dt_next, error, accepted
        assert np.asarray(got[i]).tobytes() == np.asarray(ref[i]).tobytes()
    assert np.asarray(got[0]["n"]["x"]).tobytes() == np.asarray(ref[0]["n"]["x"]).tobytes()
    assert bool(got[0]["n"]["flag"]) is True          # two half steps: flipped twice
    assert int(got[0]["n"]["count"]) == 2**24 + 3
