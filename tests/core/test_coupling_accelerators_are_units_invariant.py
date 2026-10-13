"""Aitken's omega and IQN's step do not depend on the units the state is written in.

Unit-level companions of the units oracle in
``tests/property/test_differential_coupling_units.py``: the accelerator
functions themselves, fed the same vectors multiplied by a power of two,
must return the same omega and the same step multiplied by that power,
bit for bit.  Each guards one constant that used to be absolute:

* ``aitken_relaxation`` froze omega when ``sum(delta_r**2) <= 1e-30`` --
  at 1e-12 of a group's units part-way through a solve, at 1e-16 from the
  first pass -- and the squares underflow (or, near 1e19, overflow) in
  float32 whatever the threshold;
* ``iqn_ils_update`` accepted any correction below an absolute 1e-6 once
  the residual fell below 1e-12, so a blow-up a group at scale one rejects
  was taken at small scales; its norms overflowed near 1e19, which rejected
  every step; and LAPACK rescaled its least-squares solve at extreme
  scales by a factor that is not a power of two.

The functions now rescale by an exact power of two
(``maddening.core._pow2_frame.pow2_frame``), which leaves every result at
ordinary scales as it was.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import aitken_relaxation, iqn_ils_update

#: 1e-20, 1e-16, 1e-12, 1e6, 7e19 -- as exact powers of two.
SCALES = (2.0 ** -66, 2.0 ** -53, 2.0 ** -40, 2.0 ** 20, 2.0 ** 66)
F32 = np.float32


def _f(v):
    return jnp.asarray(np.asarray(v, F32))


@pytest.mark.parametrize("s", SCALES)
def test_aitken_omega_is_the_same_at_every_scale(s):
    """A residual sequence and its rescaled copy give the same omega and step."""
    x_old = np.array([1.0, 2.0, -0.5], F32)
    x_raw = np.array([1.001, 1.9985, -0.4996], F32)
    prev_r = np.array([0.0012, -0.0018, 0.0005], F32)
    omega = jnp.float32(1.0)
    x1, w1, r1 = aitken_relaxation(_f(x_old), _f(x_raw), _f(prev_r), omega)
    xs, ws, rs = aitken_relaxation(_f(x_old * F32(s)), _f(x_raw * F32(s)),
                                   _f(prev_r * F32(s)), omega)
    assert float(w1) != 1.0, "fixture premise: omega is updated at scale one"
    assert float(ws) == float(w1), (s, float(w1), float(ws))
    np.testing.assert_array_equal(np.asarray(xs) / F32(s), np.asarray(x1))
    np.testing.assert_array_equal(np.asarray(rs) / F32(s), np.asarray(r1))


def _iqn(scale, residual, V, W):
    n, cols = V.shape
    x_old = np.zeros(n, F32)
    x_raw = residual                       # residual = x_raw - x_old
    sc = F32(scale)
    out = iqn_ils_update(
        _f(x_raw * sc), _f(x_old * sc), _f(np.zeros(n, F32)), _f(np.zeros(n, F32)),
        _f(V * sc), _f(W * sc), jnp.int32(cols), jnp.float32(1.0),
        _f(np.zeros(n, F32)), have_prev=False,
    )
    return np.asarray(out[0]) / sc


@pytest.mark.parametrize("s", SCALES)
def test_an_iqn_step_is_the_same_at_every_scale(s):
    """A well-posed secant step: taken at every scale, the same step rescaled."""
    residual = np.array([0.01, -0.02], F32)
    V = np.array([[0.004, 0.001], [-0.006, 0.003]], F32)
    W = np.array([[0.002, -0.001], [0.001, 0.004]], F32)
    want = _iqn(1.0, residual, V, W)
    assert not np.array_equal(want, residual), "fixture premise: the secant step is taken"
    np.testing.assert_array_equal(_iqn(s, residual, V, W), want)


@pytest.mark.parametrize("s", SCALES)
def test_an_iqn_blow_up_is_rejected_at_every_scale(s):
    """A correction 1e7 times the residual falls back to Aitken at every scale.

    The secant column is ten million times smaller than the residual, so
    ``c = -1e7`` and the correction is ``1e7`` residuals -- past the 1e6
    blow-up cap.  At scale one it falls back to the Aitken step (here the
    plain step, ``x_raw``); with the old absolute floor a group at 1e-20 of
    its units accepted it.
    """
    residual = np.array([1.0, 0.0], F32)
    V = np.array([[1e-7], [0.0]], F32)
    W = np.array([[1.0], [1.0]], F32)
    want = _iqn(1.0, residual, V, W)
    np.testing.assert_array_equal(want, residual)   # the fallback, not the blow-up
    np.testing.assert_array_equal(_iqn(s, residual, V, W), want)
