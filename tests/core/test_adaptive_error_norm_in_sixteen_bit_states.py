"""A bfloat16 or float16 state's step-doubling error is measured and summed in float32.

``_tree_error_norm`` forms ``|fine - coarse| / (atol + rtol * max(|fine|,
|coarse|))`` per element, squares it, sums the squares and divides by the
element count.  For a 16-bit state all of that ran in the state's own
dtype -- the defect the coupling norms had (their round-6 audit), at the
other norm the library takes:

* the running sum was a weakly typed ``jnp.array(0.0)``, which took the
  leaf's float16, and the int32 element count became float16 ``inf`` above
  65 504 elements: the norm of a finite sum read ``0.0``, and ``NaN`` once
  the sum overflowed too -- ``run_adaptive`` on a 70 000-entry float16
  field returned a NaN state after one accepted step;
* a ratio above 256 squares past float16's 65 504, so the norm read
  ``inf`` on an ordinary rejected attempt;
* the 16-bit norm made the next timestep 16-bit, which the carry of
  ``run_adaptive_scan`` refuses: a scan-carry ``TypeError`` on any 16-bit
  state.

Each leaf is now widened to float32 before its difference is formed
(``coupling.acceleration._widened``) and the sum starts from a float32
zero.  The oracles are the closed forms in float64; the three routes that
share the norm (``run_adaptive``, ``run_adaptive_scan`` and the legacy
``build_adaptive_step``) each take the failing state.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.simulation.adaptive import (
    AdaptiveConfig,
    _tree_error_norm,
    build_adaptive_step,
)

SIXTEEN = (jnp.float16, jnp.bfloat16)
#: More elements than float16 can count (its largest finite value is
#: 65 504): two leaves of this many.
BIG = 40_000


@pytest.mark.parametrize("dtype", SIXTEEN)
@pytest.mark.parametrize("atol, rtol", [(0.0, 1e-2), (1e-6, 1e-3), (0.0, 1e-6)])
def test_a_sixteen_bit_error_norm_over_more_than_65504_elements_reads_its_rms(dtype, atol, rtol):
    """``2 * BIG`` elements each eight ulps apart: the closed form, in float32.

    Before, float16 read ``0.0`` at ``rtol=1e-2`` (a finite sum over a
    count of ``inf``) and NaN at the smaller tolerances.
    """
    eps = float(jnp.finfo(dtype).eps)
    fine = {nm: {"u": jnp.full((BIG,), 1.0, dtype)} for nm in ("A", "B")}
    coarse = {nm: {"u": jnp.full((BIG,), 1.0 + 8 * eps, dtype)} for nm in ("A", "B")}
    got = _tree_error_norm(fine, coarse, atol, rtol)
    assert float(got) == pytest.approx(8 * eps / (atol + rtol * (1.0 + 8 * eps)), rel=1e-5)
    assert got.dtype == jnp.float32


def test_a_float16_error_norm_is_finite_where_a_squared_ratio_passes_65504():
    """One element a thousand scales apart reads 1000; it read ``inf``."""
    fine = {"n": {"x": jnp.asarray([2.0], jnp.float16)}}
    coarse = {"n": {"x": jnp.asarray([1.0], jnp.float16)}}
    # |2 - 1| / (5e-4 * max(2, 1)) = 1000, whose square is past float16's range.
    assert float(_tree_error_norm(fine, coarse, 0.0, 5e-4)) == pytest.approx(1000.0, rel=1e-6)


def test_a_float32_state_is_measured_in_float32_as_it_was():
    """The control: nothing is widened, and the closed form is float32's own."""
    f = jnp.asarray([1.0, -3.0, 0.25], jnp.float32)
    c = jnp.asarray([1.001, -3.0, 0.2501], jnp.float32)
    got = _tree_error_norm({"n": {"x": f}}, {"n": {"x": c}}, 0.0, 1e-3)
    scale = 1e-3 * jnp.maximum(jnp.abs(f), jnp.abs(c))
    assert got.dtype == jnp.float32
    assert float(got) == float(jnp.sqrt(jnp.sum((jnp.abs(f - c) / scale) ** 2) / 3))


class _Decay(SimulationNode):
    """``u <- u - dt * 4 u`` (explicit Euler) on an ``n``-entry field of a chosen dtype."""

    def __init__(self, name, n, dtype):
        super().__init__(name, 0.01)
        self._a = (n, dtype)

    def initial_state(self):
        n, dtype = self._a
        return {"u": jnp.full((n,), 1.0, dtype)}

    def update(self, state, boundary_inputs, dt):
        _n, dtype = self._a
        u = state["u"]
        return {"u": (u - jnp.asarray(dt, dtype) * jnp.asarray(4.0, dtype) * u).astype(dtype)}


def _decay(dtype, n):
    """A fresh graph per run: the steppers leave the graph at its final state."""
    gm = GraphManager()
    gm.add_node(_Decay("d", n, dtype))
    gm.compile()
    return gm


_RUN = dict(dt_initial=0.05, atol=0.0, rtol=1e-2, dt_min=1e-4, dt_max=0.25)
T_END = 0.5


def test_run_adaptive_on_a_float16_state_above_65504_elements_steps_as_the_float32_state_does():
    """70 000 float16 entries take the float32 state's steps, not one step to a NaN state."""
    out = {}
    for dtype in (jnp.float16, jnp.float32):
        final, info = _decay(dtype, 70_000).run_adaptive(T_END, **_RUN)
        out[jnp.dtype(dtype).name] = (np.asarray(final["d"]["u"], np.float64), info)
    u16, info16 = out["float16"]
    u32, info32 = out["float32"]
    assert np.all(np.isfinite(u16)), u16[:3]
    assert info16["t_history"][-1] == pytest.approx(T_END, rel=1e-6), info16["t_history"][-3:]
    # The float16 state rounds each Euler step in its own dtype, so its
    # error estimates differ from float32's by that rounding.
    assert abs(info16["n_steps"] - info32["n_steps"]) <= 2, (info16["n_steps"], info32["n_steps"])
    assert u16[0] == pytest.approx(u32[0], rel=2e-2)


@pytest.mark.parametrize("dtype", SIXTEEN)
def test_run_adaptive_scan_steps_a_sixteen_bit_state_to_its_end_time(dtype):
    """Was a scan-carry ``TypeError``: the 16-bit norm made the next timestep 16-bit.

    The scan reaches ``t_end`` with the state in its own dtype, in about
    the steps the host loop takes (they share one acceptance rule; the
    host loop reads the norm as a Python float).  Where the default float
    is float32: under ``jax_enable_x64`` the scan refuses every state
    narrower than float64 for another reason (MADD-ANO-190, below).
    """
    rtol = 1e-2 if dtype == jnp.float16 else 1e-1      # bfloat16 resolves 0.8%
    run = dict(_RUN, rtol=rtol)
    final, _history, info = _decay(dtype, 8).run_adaptive_scan(T_END, max_steps=400, **run)
    assert final["d"]["u"].dtype == dtype
    assert float(info["final_t"]) == pytest.approx(T_END, rel=1e-6)
    u = np.asarray(final["d"]["u"], np.float64)
    assert np.all(np.isfinite(u)) and u[0] == pytest.approx(math.exp(-4.0 * T_END), rel=0.3)
    host_final, host_info = _decay(dtype, 8).run_adaptive(T_END, **run)
    assert abs(int(info["n_steps"]) - int(host_info["n_steps"])) <= 2
    assert u[0] == pytest.approx(float(np.asarray(host_final["d"]["u"], np.float64)[0]), rel=5e-2)


@contextlib.contextmanager
def _x64():
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


@pytest.mark.xfail(strict=True, raises=TypeError, reason=(
    "MADD-ANO-190: under jax_enable_x64 run_adaptive_scan's clock carry is float64 and its "
    "next timestep takes the error norm's dtype, so a float32 or 16-bit state raises a "
    "scan-carry TypeError; deferred to 0.5.0"))
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
def test_run_adaptive_scan_steps_a_state_narrower_than_float64_under_x64(dtype):
    """Under x64 the scan must step a float32 or 16-bit state as the host loop does.

    It steps only an all-float64 one: the weakly typed float64 clock times
    the controller's float32 factor comes out float32, and ``lax.scan``
    refuses the carry.  ``run_adaptive`` steps all three.
    """
    with _x64():
        _final, host_info = _decay(dtype, 8).run_adaptive(T_END, **_RUN)
        assert host_info["t_history"][-1] == pytest.approx(T_END, rel=1e-6)     # the control
        _final, _history, info = _decay(dtype, 8).run_adaptive_scan(T_END, max_steps=400, **_RUN)
        assert float(info["final_t"]) == pytest.approx(T_END, rel=1e-6)


@pytest.mark.parametrize("dtype", SIXTEEN)
def test_build_adaptive_step_reports_a_sixteen_bit_states_error_in_float32(dtype):
    """The legacy step builder shares the norm: 70 000 entries, the closed form, float32."""
    rtol = 1e-2
    step = build_adaptive_step(None, AdaptiveConfig(atol=0.0, rtol=rtol), ["n"])

    def dt_step(state, _external, dt):
        x = state["n"]["x"]
        return {"n": {"x": (x - (4.0 * dt).astype(dtype) * x).astype(dtype)}}

    state = {"n": {"x": jnp.full((70_000,), 1.0, dtype)}}
    dt = jnp.float32(0.2)
    half, dt_next, err, _accepted = step(state, dt, {}, dt_step)
    full = np.asarray(dt_step(state, {}, dt)["n"]["x"], np.float64)
    two_halves = np.asarray(half["n"]["x"], np.float64)
    expected = abs(two_halves[0] - full[0]) / (rtol * max(abs(two_halves[0]), abs(full[0])))
    assert expected > 1.0                              # the fixture premise: a rejected attempt
    assert float(err) == pytest.approx(expected, rel=1e-5)
    assert err.dtype == jnp.float32 and dt_next.dtype == jnp.float32
