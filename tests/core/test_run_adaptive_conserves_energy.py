"""``run_adaptive`` keeps an undamped spring's energy, on every push.

``tests/core/test_adaptive.py::TestAdaptivePhysics::test_conservation``
checks this over two seconds at ``atol=1e-8`` -- about 15,000 accepted
steps, each a host round trip -- and is slow-marked.  This is the same
property over a fifth of a second at looser tolerances (87 steps): the
spring swings through its rest length, so its energy moves from the
spring into the mass and back, and the total must stay within the same
5% (measured: 0.44%).

It can fail: with the error control loosened to ``atol=1e-3``,
``rtol=0.1`` the same run drifts by 15%, so a controller that stopped
honouring its tolerances, or an extrapolation that stopped cancelling the
integrator's leading error, shows here.
"""

from __future__ import annotations

import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

K, M, REST, X0 = 100.0, 1.0, 1.0, 2.0


def test_run_adaptive_conserves_an_undamped_springs_energy():
    gm = GraphManager()
    gm.add_node(TableNode(name="anchor", timestep=0.001, position=0.0))
    gm.add_node(SpringDamperNode(name="spring", timestep=0.001, stiffness=K, damping=0.0,
                                 mass=M, rest_length=REST, initial_position=X0))
    gm.add_edge("anchor", "spring", "position", "anchor_position")
    gm.compile()
    e0 = 0.5 * K * (X0 - REST) ** 2

    state, info = gm.run_adaptive(t_end=0.2, dt_initial=0.001, atol=1e-6, rtol=1e-4,
                                  dt_max=0.05)

    x = float(state["spring"]["position"])
    v = float(state["spring"]["velocity"])
    # Non-vacuity: the run reached its end time and the spring swung
    # through its rest length (a quarter period is 0.157 s).
    assert float(info["t_history"][-1]) == pytest.approx(0.2, abs=1e-6)
    assert x < REST, x
    assert 0.5 * K * (x - REST) ** 2 + 0.5 * M * v ** 2 == pytest.approx(e0, rel=0.05)
