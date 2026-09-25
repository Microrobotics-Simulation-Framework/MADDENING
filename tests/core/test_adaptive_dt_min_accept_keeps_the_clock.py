"""An attempt ``run_adaptive`` accepts at ``dt_min`` advances the clock by what it covered.

When an attempt's error is too large and the shrunk step would fall to
``dt_min``, the stepper warns and accepts the attempt it just made -- whose
two half steps covered the attempted step, up to ``dt_min / min_factor``.
It advanced its clock (``t``, ``dt_history``, ``t_history``, the callback's
``dt``) by ``dt_min`` instead, so the state ran ahead of the time reported
for it, and the run went on past ``t_end`` in the state's own time
(MADD-ANO-053, since 0.1.0).  A node integrating its own clock shows it.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.spring import SpringDamperNode


class Clock(SimulationNode):
    def initial_state(self):
        return {"t": jnp.array(0.0, jnp.float32)}

    def update(self, s, bi, dt):
        return {"t": s["t"] + dt}


def _graph():
    """A clock beside a spring pair whose coupling diverges at ``dt = 0.01``."""
    gm = GraphManager()
    gm.add_node(Clock("clock", 0.01))
    gm.add_node(SpringDamperNode("a", 0.01, stiffness=30000.0, damping=50.0,
                                 initial_position=-1.0))
    gm.add_node(SpringDamperNode("b", 0.01, stiffness=30000.0, damping=50.0,
                                 initial_position=1.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=30, tolerance=1e-5)
    gm.compile()
    return gm


def test_the_clock_advances_by_the_attempt_a_dt_min_accept_keeps():
    """The first attempt (0.01) is rejected, the shrunk step would be below
    ``dt_min`` (0.005), so the 0.01 attempt is accepted: 0.01 of clock."""
    seen = []
    with pytest.warns(UserWarning, match="hit dt_min"):
        state, info = _graph().run_adaptive(
            0.02, dt_initial=0.01, dt_max=0.01, dt_min=0.005, atol=1e-3, rtol=1e-3,
            callback=lambda t, dt, s: seen.append((t, dt, float(s["clock"]["t"]))),
        )
    assert info["dt_history"][0] == pytest.approx(0.01)
    assert info["t_history"][-1] == pytest.approx(sum(info["dt_history"]))
    assert float(state["clock"]["t"]) == pytest.approx(info["t_history"][-1], abs=1e-6)
    # The callback sees the state at the time it is told.
    for t, _dt, clock in seen:
        assert clock == pytest.approx(t, abs=1e-6)
