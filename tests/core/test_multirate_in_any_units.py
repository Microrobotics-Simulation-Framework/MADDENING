"""A multi-rate graph schedules the same way whatever unit its timesteps are in.

The base timestep is the GCD of the scheduled timesteps, found by Euclid's
algorithm with a tolerance for float noise.  The tolerance was an absolute
``1e-9``, so a graph whose timesteps were themselves near a nanosecond (a
circuit, a molecular model, any graph in seconds at that rate) stopped the
algorithm before its first step: nodes at ``1e-9`` and ``2e-9`` got a base step
of ``2e-9``, the fast node advanced ``1e-9`` per base step while the slow one
advanced ``2e-9``, and after four steps their clocks read ``4e-9`` and ``8e-9``,
with no warning.  The tolerance is now relative to the largest timestep.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager, _float_gcd, _multi_gcd
from maddening.core.node import SimulationNode


class _Clock(SimulationNode):
    """Accumulates the ``dt`` it is handed and counts its updates."""

    def halo_width(self) -> dict[int, int]:
        return {}

    def initial_state(self) -> dict:
        return {"t": jnp.zeros(()), "n": jnp.zeros((), jnp.int32)}

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        return {"t": state["t"] + dt, "n": state["n"] + 1}


#: Timestep sets whose GCD is known exactly, at a reference scale of one.
CASES = [((1.0, 2.0), 1.0), ((2.0, 5.0), 1.0), ((1.0, 2.0, 5.0), 1.0),
         ((4.0, 6.0), 2.0), ((0.5, 3.0), 0.5)]


@pytest.mark.parametrize("unit", [1.0, 1e-3, 1e-6, 1e-9, 1e-12, 1e3])
@pytest.mark.parametrize("steps, gcd", CASES)
def test_the_base_timestep_is_the_gcd_in_any_unit(steps, gcd, unit):
    got = _multi_gcd([s * unit for s in steps])
    assert got == pytest.approx(gcd * unit, rel=1e-9), (steps, unit, got)
    a, b = steps[0] * unit, steps[1] * unit
    assert _float_gcd(a, b) == pytest.approx(_multi_gcd([a, b]), rel=1e-12)


@pytest.mark.parametrize("unit", [1e-3, 1e-9])
def test_a_nanosecond_graph_keeps_its_nodes_clocks_together(unit):
    gm = GraphManager()
    gm.add_node(_Clock("fast", timestep=1.0 * unit))
    gm.add_node(_Clock("slow", timestep=2.0 * unit))
    gm.compile()
    assert gm.timestep == pytest.approx(1.0 * unit, rel=1e-9)
    for _ in range(4):
        gm.step()
    s = gm._state
    assert int(s["fast"]["n"]) == 4 and int(s["slow"]["n"]) == 2
    assert float(s["fast"]["t"]) == pytest.approx(4.0 * unit, rel=1e-6)
    assert float(s["slow"]["t"]) == pytest.approx(4.0 * unit, rel=1e-6)


def test_float_noise_in_decimal_timesteps_is_still_ignored():
    # 0.3 % 0.1 is 0.0999...98 and then 2.8e-17: noise, not a common step.
    assert _multi_gcd([0.3, 0.1]) == pytest.approx(0.1, rel=1e-12)
    assert _multi_gcd([0.01, 0.003]) == pytest.approx(0.001, rel=1e-9)
    assert _multi_gcd([3e-12, 1e-12]) == pytest.approx(1e-12, rel=1e-9)
