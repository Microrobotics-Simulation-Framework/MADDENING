"""CPL-047 and CPL-048 under ``jax_enable_x64``: the rate and error estimates.

``tests/core/test_coupling_claims_helpers.py`` pins both formulas on
float32 residuals.  A float64 group hands the same helpers float64
residuals, and a float32 group under x64 hands them float32 ones in a
process whose Python floats are float64.  The claims are the same:
``error_amplification`` is ``1 / (1 - max(r/r1, sqrt(r/r2)))`` and ``0.0``
at every documented rejection; ``estimated_error`` is
``residual * max(step_scale * amplification, 1)``, never below the
residual.  What float64 adds is the edge float32 cannot express: a rate
within ``2**-40`` of one, whose amplification is ``2**40``.
"""

from __future__ import annotations

import contextlib
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import error_amplification, estimated_error


@contextlib.contextmanager
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _amp(dtype, r, rp, rpp=None):
    rpp = None if rpp is None else jnp.asarray(rpp, dtype)
    out = error_amplification(jnp.asarray(r, dtype), jnp.asarray(rp, dtype), rpp)
    assert out.dtype == dtype, out.dtype
    return float(out)


#: Every documented rejection: rho >= 1, a zero or non-finite predecessor,
#: a non-finite residual.
_REJECTED = [(1.0, 1.0), (2.0, 1.0), (0.5, 0.0), (0.5, 1.0, 0.0), (math.nan, 1.0),
             (0.5, math.inf), (0.5, 1.0, math.nan), (math.inf, 1.0), (0.0, 0.0)]


def _formula_holds(dtype, rel):
    assert _amp(dtype, 0.5, 1.0) == pytest.approx(1.0 / (1.0 - math.sqrt(0.5)), rel=rel)
    assert _amp(dtype, 0.25, 0.5, 1.0) == pytest.approx(2.0, rel=rel)
    assert _amp(dtype, 0.25, 2.5, 0.5) == pytest.approx(1.0 / (1.0 - math.sqrt(0.5)), rel=rel)
    assert _amp(dtype, 0.0, 1.0, 1.0) == 1.0
    for args in _REJECTED:
        assert _amp(dtype, *args) == 0.0, args
    rng = np.random.default_rng(0)
    for _ in range(200):
        r1, r2 = rng.uniform(1e-3, 1.0, size=2)
        r = rng.uniform(0.0, 1.0) * min(r1, r2)
        a = _amp(dtype, r, r1, r2)
        assert a == 0.0 or a >= 1.0, (r, r1, r2, a)


def test_error_amplification_follows_its_formula_in_float64():
    """CPL-047 on float64 residuals, and at a rate float32 cannot tell from one."""
    with _x64():
        _formula_holds(jnp.float64, rel=1e-14)
        # A geometric sequence at rho = 1 - 2**-40: r2 = 1/rho, r1 = 1, r = rho,
        # so the one-step and two-step rates are both rho.
        r = 1.0 - 2.0 ** -40
        amp = _amp(jnp.float64, r, 1.0, 1.0 / r)
        assert amp == pytest.approx(2.0 ** 40, rel=1e-6), amp
        # The same sequence in float32 cannot see the contraction at all.
        assert _amp(jnp.float32, np.float32(r), np.float32(1.0)) == 0.0


def test_error_amplification_follows_its_formula_on_float32_residuals_under_x64():
    """A float32 group's residuals under x64: float32 out, the float32 numbers."""
    with _x64():
        under = [_amp(jnp.float32, *args) for args in [(0.5, 1.0), (0.25, 0.5, 1.0)] + _REJECTED]
    without = [_amp(jnp.float32, *args) for args in [(0.5, 1.0), (0.25, 0.5, 1.0)] + _REJECTED]
    assert under == without
    with _x64():
        _formula_holds(jnp.float32, rel=1e-6)


def _never_below(dtype):
    r = jnp.asarray(1e-3, dtype)
    assert estimated_error(r, jnp.asarray(0.0, dtype)).dtype == dtype
    assert float(estimated_error(r, jnp.asarray(0.0, dtype))) == float(r)          # rejected
    assert float(estimated_error(r, jnp.asarray(1.5, dtype), 0.5)) == float(r)     # floored
    rng = np.random.default_rng(1)
    for _ in range(300):
        res = jnp.asarray(rng.uniform(0.0, 10.0), dtype)
        amp = jnp.asarray(rng.choice([0.0, rng.uniform(0.0, 1e4)]), dtype)
        scale = float(rng.uniform(0.01, 2.0))
        assert float(estimated_error(res, amp, scale)) >= float(res)


def test_estimated_error_is_never_below_the_residual_in_float64():
    """CPL-048 on float64 residuals, the amplification at ``2**40`` included."""
    with _x64():
        _never_below(jnp.float64)
        r = jnp.asarray(2.0 ** -30, jnp.float64)
        est = estimated_error(r, jnp.asarray(2.0 ** 40, jnp.float64), 1.0)
        assert float(est) == 2.0 ** 10 and est.dtype == jnp.float64


def test_estimated_error_is_never_below_the_residual_on_float32_residuals_under_x64():
    with _x64():
        _never_below(jnp.float32)
        r = jnp.asarray(1e-3, jnp.float32)
        assert estimated_error(r, jnp.asarray(3.0, jnp.float32), 1.3).dtype == jnp.float32
