"""Every window of ``windowed_loss`` restarts on its own multi-rate phase.

A window that starts at sample ``k`` restarts the simulation at base step
``k * sample_every``, and on a multi-rate graph the sub-step counter in
``_meta`` decides which nodes fire on which base step.  ``windowed_loss``
restores that counter per window (``_state_from_obs``).  Restarting every
window at step 0 instead runs a slower node on the wrong phase in every
window that starts on an odd base step; it survived every sysid test
(audit_040_p4_4/fmu-sysid/repro_mutation_survivors.py, S10): the loss at the
true parameters, teacher-forced from the graph's own exact record, stayed
0.0 for even windows and rose to 0.0167 for ``window=3``.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode
from maddening.sysid import observations_from_history, windowed_loss

DT = 0.01
N_STEPS = 30


def _multirate():
    """The ball steps every other base step; the table and spring every one."""
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=DT))
    gm.add_node(BallNode(name="ball", timestep=2 * DT, initial_position=1.0, elasticity=0.7))
    gm.add_node(SpringDamperNode(name="spring", timestep=DT, stiffness=30.0, damping=2.0,
                                 initial_position=0.5))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    return gm


@pytest.fixture(scope="module")
def exact_record():
    source = _multirate()
    assert source._rate_dividers == {"table": 1, "ball": 2, "spring": 1}  # noqa: SLF001
    s0 = source._user_state(source._state)  # noqa: SLF001
    _, history = source.run_scan_with_history(N_STEPS)
    return observations_from_history(s0, history)


@pytest.mark.parametrize("sample_every, window", [
    (1, 1), (1, 2), (1, 3), (1, 5), (3, 1), (3, 5),
])
def test_a_window_on_an_odd_base_step_runs_the_slow_node_on_its_own_phase(
    exact_record, sample_every, window,
):
    """At the true parameters, teacher-forced from the graph's own record,
    every window reproduces the record -- including those that start on an
    odd base step (all of them but ``window=2`` here), where the ball does
    not fire on the window's first step."""
    obs = jax.tree.map(lambda x: x[::sample_every], exact_record)
    gm = _multirate()
    loss = float(windowed_loss(gm, gm.params, obs, obs_fn=lambda s: s["ball"]["position"],
                               window=window, sample_every=sample_every))
    # Exactly 0.0 on this box; the bound leaves room for another jaxlib's
    # rounding and is five decades below the wrong phase's 0.0167.
    assert loss <= 1e-7, loss
