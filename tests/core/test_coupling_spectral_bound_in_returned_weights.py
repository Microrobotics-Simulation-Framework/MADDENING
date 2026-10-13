"""``spectral_error_bound`` bounds the distance in the returned state's weights, as documented.

The bound is ``residual`` times a resolvent factor.  The factor is measured
in the group's norm at the returned state -- each field divided by its own
``max|field|`` there -- but ``residual`` divides each field by the larger of
its magnitude at the returned state and after one more pass (the pair the
pass compared).  On a group still growing toward its fixed point the two
differ by ``max|F(x)| / max|x|``, and the bound read 0.94x the true distance
in the returned state's weights with ``spectral_usable=True`` (the round-5
audit, CPL-088).  The factor now carries the ratio of the two weightings of
``r = F(x) - x``, so the bound holds in the norm it names.

Exact reference: the float64 fixed point of the linear pair.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode


class _Rel(SimulationNode):
    def __init__(self, name, g, b):
        super().__init__(name, 1.0)
        self._g, self._b = g, b

    def initial_state(self):
        return {"x": jnp.zeros(1, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(1,), dtype=jnp.float32,
                                       default=jnp.zeros(1, jnp.float32))}

    def update(self, state, boundary_inputs, dt):
        return {"x": jnp.float32(self._g) * boundary_inputs["u"] + jnp.float32(self._b)}


@pytest.mark.parametrize("gain, cap", [(0.9, 2), (0.999, 2), (0.99, 3), (0.999, 5)])
def test_the_bound_holds_in_the_returned_states_weights_on_a_growing_group(gain, cap):
    """``A <- B + 1``, ``B <- gain * A`` from zero: every field grows toward ``x*``.

    Before the fix the bound read 0.94x (gain 0.9 and 0.999, cap 2).
    """
    gm = GraphManager()
    gm.add_node(_Rel("A", 1.0, 1.0))
    gm.add_node(_Rel("B", gain, 0.0))
    gm.add_edge("B", "A", "x", "u")
    gm.add_edge("A", "B", "x", "u")
    gm.add_coupling_group(["A", "B"], max_iterations=cap, tolerance=1e-6, diagnostics=True)
    gm.compile()
    s = gm.step()
    d = gm.coupling_diagnostics()["A+B"]
    g32 = float(np.float32(gain))
    xa_star = 1.0 / (1.0 - g32)
    xs = {"A": xa_star, "B": g32 * xa_star}
    dist = float(np.sqrt(sum(((float(s[m]["x"][0]) - xs[m]) / abs(float(s[m]["x"][0]))) ** 2
                             for m in "AB")))
    assert d["spectral_usable"], dict(d)
    assert d["spectral_error_bound"] >= dist, (d["spectral_error_bound"], dist)
