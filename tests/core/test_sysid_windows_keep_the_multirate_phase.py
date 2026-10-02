"""Every window of ``windowed_loss`` restarts on its own multi-rate phase.

A window that starts at sample ``k`` restarts the simulation at base step
``start_step + k * sample_every``, and on a multi-rate graph the sub-step
counter in ``_meta`` decides which nodes fire on which base step.
``windowed_loss`` restores that counter per window (``_state_from_obs``).

Three faults this pins, each of which survived every sysid test in turn:

* restarting every window at step 0 (the slow node runs on the wrong phase
  in every window that starts on an odd base step: 0.0167 at the truth for
  ``window=3``; audit_040_p4_4/fmu-sysid/repro_mutation_survivors.py, S10);
* ``k`` in place of ``k * sample_every`` (equivalent at ``sample_every`` 1
  and 3 against a divider of 2, so only an even ``sample_every`` sees it;
  audit_040_p4_5/fmu-sysid/repro_mutation_M21_is_live.py);
* a record that did not start at base step 0: the observations are user
  state and cannot carry the recording's step counter, so the loss at the
  generating parameters was 1.3e-2 for a record starting one step in, with
  nothing to say why (repro_windowed_loss_phase_offset.py).  ``start_step``
  says where it began, and a multi-rate graph without it warns.
"""

import os
import warnings

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


def _floor_driven():
    """The ball (every 2nd base step) reads a floor that moves every base step,
    so the phase decides what it sees (the audit's M21 graph)."""
    gm = GraphManager()
    gm.add_node(BallNode(name="ball", timestep=2 * DT, initial_position=0.6, elasticity=0.7))
    gm.add_node(SpringDamperNode(name="spring", timestep=DT, stiffness=300.0, damping=0.5,
                                 initial_position=0.5))
    gm.add_edge("spring", "ball", "position", "table_position")
    gm.compile()
    return gm


def _record(build, n_steps, warmup=0):
    source = build()
    if warmup:
        source.run(warmup)
    s0 = source._user_state(source._state)  # noqa: SLF001
    _, history = source.run_scan_with_history(n_steps)
    return observations_from_history(s0, history)


def _ball(s):
    return s["ball"]["position"]


@pytest.fixture(scope="module")
def exact_record():
    source = _multirate()
    assert source._rate_dividers == {"table": 1, "ball": 2, "spring": 1}  # noqa: SLF001
    return _record(_multirate, N_STEPS)


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
    loss = float(windowed_loss(gm, gm.params, obs, obs_fn=_ball, window=window,
                               sample_every=sample_every, start_step=0))
    # Exactly 0.0 on this box; the bound leaves room for another jaxlib's
    # rounding and is five decades below the wrong phase's 0.0167.
    assert loss <= 1e-7, loss


@pytest.mark.parametrize("window", [1, 3])
def test_an_even_sample_every_restarts_each_window_at_k_times_sample_every(window):
    """``sample_every=2`` against a divider of 2: every window starts on an
    even base step, ``2k``.  Restarting at ``k`` instead puts every
    odd-numbered window on the wrong phase (1.6e-2 at the truth)."""
    obs = jax.tree.map(lambda x: x[::2], _record(_floor_driven, 24))
    gm = _floor_driven()
    loss = float(windowed_loss(gm, gm.params, obs, obs_fn=_ball, window=window,
                               sample_every=2, start_step=0))
    assert loss <= 1e-7, loss


@pytest.mark.parametrize("warmup", [1, 2, 3])
def test_a_record_that_starts_mid_schedule_is_replayed_on_its_own_phase(warmup):
    """A record taken after ``gm.run(warmup)``: with ``start_step=warmup``
    every window is on the phase it was recorded on."""
    obs = _record(_multirate, 24, warmup=warmup)
    gm = _multirate()
    loss = float(windowed_loss(gm, gm.params, obs, obs_fn=_ball, window=4,
                               start_step=warmup))
    assert loss <= 1e-7, loss


def test_an_odd_start_read_as_zero_is_off_phase_and_the_default_warns():
    """Non-vacuity, and the warning: the same odd-start record read as
    starting at 0 is replayed on the wrong phase (the audit's 1.3e-2), and
    leaving ``start_step`` out on a multi-rate graph says it is assuming 0."""
    obs = _record(_multirate, 24, warmup=1)
    gm = _multirate()
    assert float(windowed_loss(gm, gm.params, obs, obs_fn=_ball, window=4,
                               start_step=0)) > 1e-3
    with pytest.warns(UserWarning, match="start_step="):
        windowed_loss(gm, gm.params, obs, obs_fn=_ball, window=4)


def test_a_single_rate_graph_needs_no_start_step():
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="spring", timestep=DT, stiffness=30.0, damping=2.0,
                                 initial_position=0.5))
    gm.compile()
    s0 = gm._user_state(gm._state)  # noqa: SLF001
    obs = observations_from_history(s0, gm.run_scan_with_history(8)[1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loss = windowed_loss(gm, gm.params, obs, obs_fn=lambda s: s["spring"]["position"],
                             window=4)
    assert float(loss) <= 1e-12


@pytest.mark.parametrize("bad", [-1, 1.5, True, "0"])
def test_a_start_step_that_is_not_a_step_is_refused(exact_record, bad):
    gm = _multirate()
    with pytest.raises(ValueError, match="start_step"):
        windowed_loss(gm, gm.params, exact_record, obs_fn=_ball, window=5, start_step=bad)
