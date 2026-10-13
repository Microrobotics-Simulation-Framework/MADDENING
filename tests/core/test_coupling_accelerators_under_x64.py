"""Coupling accelerators on a float32 group under ``jax_enable_x64``.

The accelerator carries -- Aitken's omega and previous residual, IQN's
secant matrices -- were seeded at canonical precision (``jnp.zeros(n)``,
``jnp.array(1.0)``), which is float64 under x64.  Against a group whose
state stays float32 (float32 parameters, as a node that pins its dtype has)
``solver="fori"`` with ``"aitken"``, ``"iqn-ils"`` or ``"iqn-imvj"`` raised a
``fori_loop`` carry ``TypeError`` from ``step()``, and ``solver="ift"``
with IQN narrowed a float64 step into the float32 iterate through a scatter
JAX warns will become an error (``FutureWarning``, fatal under this suite's
``filterwarnings = error``).  They are now seeded in the interface
vector's dtype (at least float32), as ``compile()`` seeds IQN-IMVJ's warm
start.  MADD-ANO-017's item (3) used to say the fori path "adds no failure
of its own" under x64; that was measured on nodes whose float64 parameters
promote the state, and is corrected.

``jax.config.update`` is process-global, so every test runs inside
:func:`_x64`, which restores the previous setting (the pattern of
``tests/core/test_x64_graph_scan_limitation.py``).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

KEY = "a+b"
_ACCELERATIONS = ["none", "fixed", "aitken", "iqn-ils", "iqn-imvj"]


@contextlib.contextmanager
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


class _Relay(SimulationNode):
    """``x <- gain * u + bias`` with parameters and state in one fixed dtype."""

    def __init__(self, name, gain, bias, dtype):
        super().__init__(name, 1.0, gain=jnp.asarray(gain, dtype),
                         bias=jnp.asarray(bias, dtype))
        self._dtype = dtype

    def initial_state(self):
        return {"x": jnp.zeros(2, self._dtype)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(2,), dtype=self._dtype,
                                       default=jnp.zeros(2, self._dtype))}

    def update(self, state, boundary_inputs, dt):
        return {"x": self.params["gain"] * boundary_inputs["u"] + self.params["bias"]}


def _pair(solver, acceleration, dtype):
    gm = GraphManager()
    gm.add_node(_Relay("a", 0.6, [1.0, 2.0], dtype))
    gm.add_node(_Relay("b", 0.9, [0.0, 1.0], dtype))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    kw = {"jacobian_reuse": 2} if acceleration == "iqn-imvj" else {}
    kw.update({"relaxation": 0.8} if acceleration == "fixed" else {})
    with pytest.warns(DeprecationWarning) if solver == "fori" else contextlib.nullcontext():
        gm.add_coupling_group(["a", "b"], solver=solver, acceleration=acceleration,
                              max_iterations=30, tolerance=1e-4, **kw)
    gm.compile()
    return gm


@pytest.mark.parametrize("acceleration", _ACCELERATIONS)
@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_a_float32_group_steps_and_scans_under_x64(solver, acceleration):
    """``step()`` and ``run_scan`` run, warn nothing, and keep the state float32."""
    with _x64():
        gm = _pair(solver, acceleration, jnp.float32)
        gm.step()
        assert gm.get_node_state("a")["x"].dtype == jnp.float32
        gm.run_scan(3)
        assert gm.get_node_state("a")["x"].dtype == jnp.float32
        if acceleration == "iqn-imvj":
            assert gm._state["_meta"][f"coupling_{KEY}_V"].dtype == jnp.float32
        d = gm.coupling_diagnostics()[KEY] if solver == "ift" else None
        if d is not None:
            assert np.isfinite(d["residual"]), dict(d)


@pytest.mark.parametrize("acceleration", ["aitken", "iqn-imvj"])
@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_the_accelerated_state_agrees_with_the_same_group_without_x64(solver, acceleration):
    """The seed dtype is the only thing x64 moved: the float32 answer is the same.

    Not bit for bit -- under x64 a Python scalar in a node's arithmetic is a
    weak float64, which can round a product differently -- but to a few
    float32 ulps of the state, after the same number of passes.
    """
    gm = _pair(solver, acceleration, jnp.float32)
    gm.run_scan(2)
    want = np.asarray(gm.get_node_state("a")["x"])
    with _x64():
        gm = _pair(solver, acceleration, jnp.float32)
        gm.run_scan(2)
        got = np.asarray(gm.get_node_state("a")["x"])
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=0.0)


@pytest.mark.parametrize("acceleration", ["aitken", "iqn-imvj"])
def test_a_float64_group_keeps_float64_carries_under_x64(acceleration):
    """The canonical case, unchanged: a float64 group's seeds are float64."""
    with _x64():
        gm = _pair("fori", acceleration, jnp.float64)
        gm.run_scan(2)
        assert gm.get_node_state("a")["x"].dtype == jnp.float64
        if acceleration == "iqn-imvj":
            assert gm._state["_meta"][f"coupling_{KEY}_V"].dtype == jnp.float64


def test_the_x64_context_manager_restores_the_previous_setting():
    """Defined last: the rest of the module left the session as it found it."""
    assert jax.config.jax_enable_x64 is False or os.environ.get("JAX_ENABLE_X64")
