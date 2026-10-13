"""``CouplingGroup`` refuses a count or a threshold outside the range it means anything in.

``waveform_iterations=0`` (or negative) on a sub-cycling group ran no
waveform sweep at all: the members never stepped, the state stayed where
it started, and ``coupling_diagnostics()`` had no entry for the group --
silently (the round-5 audit of the coupling claims).  The other knobs had
the same gap: a ``max_iterations`` below one, a negative
``jacobian_reuse``, a negative or non-finite ``tolerance`` / ``rtol`` /
``atol``, and a ``relaxation`` that is zero, negative or not finite were
all accepted.  Each is now a ``ValueError`` at construction, naming the
knob and its range.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.group import CouplingGroup
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

BAD = [
    ("max_iterations", 0), ("max_iterations", -3), ("max_iterations", 2.0), ("max_iterations", True),
    ("waveform_iterations", 0), ("waveform_iterations", -1), ("waveform_iterations", 1.5),
    ("jacobian_reuse", -1), ("jacobian_reuse", 0.5),
    ("tolerance", -1e-6), ("tolerance", math.nan), ("tolerance", math.inf), ("tolerance", "1e-6"),
    ("rtol", -1e-3), ("rtol", math.nan), ("atol", -1.0), ("atol", math.inf),
    ("relaxation", 0.0), ("relaxation", -0.5), ("relaxation", math.nan), ("relaxation", math.inf),
]


@pytest.mark.parametrize("name, value", BAD, ids=[f"{n}={v!r}" for n, v in BAD])
def test_an_out_of_range_number_is_refused_naming_its_knob(name, value):
    with pytest.raises(ValueError, match=rf"CouplingGroup\.{name}="):
        CouplingGroup(nodes=frozenset({"a", "b"}), **{name: value})


GOOD = [
    ("max_iterations", 1), ("max_iterations", np.int64(7)), ("waveform_iterations", 1),
    ("jacobian_reuse", 0), ("tolerance", 0.0), ("tolerance", 1), ("rtol", 0.0), ("atol", 0.0),
    ("relaxation", 1e-3), ("relaxation", 1.7), ("relaxation", np.float32(0.5)),
]


@pytest.mark.parametrize("name, value", GOOD, ids=[f"{n}={v!r}" for n, v in GOOD])
def test_every_value_in_range_is_accepted(name, value):
    # ``relaxation`` away from 1 is inert without acceleration="fixed", and
    # ``rtol`` under the L2 norm: each warns so (a separate rule), not refused.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        g = CouplingGroup(nodes=frozenset({"a", "b"}), **{name: value})
    assert getattr(g, name) == value


class _Relay(SimulationNode):
    def __init__(self, name, timestep, gain, bias):
        super().__init__(name, timestep)
        self._g, self._b = gain, bias

    def initial_state(self):
        return {"x": jnp.zeros(1, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(1,), dtype=jnp.float32,
                                       default=jnp.zeros(1, jnp.float32))}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] + jnp.float32(dt) * (self._g * boundary_inputs["u"] + self._b)}


@pytest.mark.parametrize("sweeps", [0, -1])
def test_a_sub_cycling_group_with_no_waveform_sweep_is_refused_before_it_freezes(sweeps):
    """The audit's case: a sub-cycling pair whose members never stepped."""
    gm = GraphManager()
    gm.add_node(_Relay("A", 0.1, 0.5, 1.0))
    gm.add_node(_Relay("B", 0.05, 0.5, 2.0))
    gm.add_edge("A", "B", "x", "u")
    gm.add_edge("B", "A", "x", "u")
    with pytest.raises(ValueError, match="waveform_iterations"):
        gm.add_coupling_group(["A", "B"], subcycling=True, waveform_iterations=sweeps)
