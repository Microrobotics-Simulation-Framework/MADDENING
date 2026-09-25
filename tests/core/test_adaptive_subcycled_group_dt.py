"""``run_adaptive*`` advances every member of a sub-cycled group by the step.

A coupling group with ``subcycling=True`` over nodes of different
timesteps is not multi-rate -- ``compile()`` schedules every member at
the group's macro timestep -- so the adaptive steppers accept it.  They
hand the coupled block one ``runtime_dt`` for every node, and the block
still sub-steps the fast node ``round(macro_dt / node_dt)`` times per
pass.  Each sub-step used to be the whole ``runtime_dt``, so the fast
node advanced ``divider * dt`` per adaptive step: a clock on it read
0.5 at ``t_end = 0.05`` for a 10:1 group, 0.2 for a 4:1 one.  Each
sub-step is now ``dt * node_dt / macro_dt``.

The probe is a node that integrates its own clock, ``t <- t + dt``: after
any run to ``t_end`` every node's clock must read ``t_end``.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

T_END = 0.05
MACRO_DT = 0.01


class _Clock(SimulationNode):
    """``t <- t + dt``; ``x <- gain * u + bias`` (``u`` from the partner)."""

    def __init__(self, name, timestep, gain, bias):
        super().__init__(name=name, timestep=timestep, gain=gain, bias=bias)

    def initial_state(self):
        return {"t": jnp.float32(0.0), "x": jnp.float32(0.0)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0))}

    def update(self, state, boundary_inputs, dt):
        u = boundary_inputs.get("u", jnp.float32(0.0))
        return {"t": state["t"] + dt,
                "x": self.params["gain"] * u + self.params["bias"]}


def _subcycled_pair(fast_dt):
    gm = GraphManager()
    gm.add_node(_Clock("fast", fast_dt, gain=0.5, bias=0.0))
    gm.add_node(_Clock("slow", MACRO_DT, gain=0.5, bias=1.0))
    gm.add_edge("slow", "fast", "x", "u")
    gm.add_edge("fast", "slow", "x", "u")
    gm.add_coupling_group(["fast", "slow"], max_iterations=20,
                          tolerance=1e-6, subcycling=True)
    gm.compile()
    return gm


@pytest.fixture(scope="module", params=[0.001, 0.0025], ids=["10:1", "4:1"])
def fast_dt(request):
    return request.param


def test_a_subcycled_group_is_accepted_by_the_adaptive_steppers(fast_dt):
    """The precondition: the group is not multi-rate, so nothing refuses it."""
    gm = _subcycled_pair(fast_dt)
    assert not gm.is_multirate


def test_run_adaptive_advances_every_member_by_the_step(fast_dt):
    gm = _subcycled_pair(fast_dt)
    final, info = gm.run_adaptive(T_END, dt_initial=MACRO_DT, dt_max=MACRO_DT)
    elapsed = sum(info["dt_history"])
    assert elapsed == pytest.approx(T_END)
    for node in ("fast", "slow"):
        assert float(final[node]["t"]) == pytest.approx(elapsed, rel=1e-5), (
            node, float(final[node]["t"]), elapsed)


def test_run_adaptive_scan_advances_every_member_by_the_step(fast_dt):
    gm = _subcycled_pair(fast_dt)
    _final, _history, info = gm.run_adaptive_scan(
        T_END, max_steps=8, dt_initial=MACRO_DT, dt_max=MACRO_DT)
    elapsed = float(info["final_t"])
    assert elapsed == pytest.approx(T_END)
    for node in ("fast", "slow"):
        assert float(gm.get_node_state(node)["t"]) == pytest.approx(
            elapsed, rel=1e-5), node


def test_the_adaptive_step_at_the_macro_timestep_is_the_compiled_step(fast_dt):
    """At ``dt == macro_dt`` each sub-step is the node's own timestep.

    So one adaptive step reproduces the step ``compile()`` built, to the
    float32 rounding of ``dt * node_dt / macro_dt`` against ``node_dt``.
    A step that is not the compiled one -- the whole ``dt`` per sub-step
    -- is ``divider`` times off in the fast clock.
    """
    gm = _subcycled_pair(fast_dt)
    state = gm._state
    ext = gm._default_external_inputs()
    compiled = gm._compiled_step(state, ext)
    adaptive = jax.jit(gm._build_dt_step_fn())(state, ext, jnp.float32(MACRO_DT))
    for node in ("fast", "slow"):
        for field in ("t", "x"):
            np.testing.assert_allclose(
                np.asarray(adaptive[node][field]), np.asarray(compiled[node][field]),
                rtol=1e-6, err_msg=f"{node}.{field}")


def test_the_multirate_refusal_says_what_it_enforces():
    """A multi-rate graph is refused, by a message that matches the rule.

    It used to say "All nodes must share the same timestep", which the
    sub-cycled group above contradicts: the refusal is of multi-rate
    graphs, and a sub-cycled group of differing timesteps is not one.
    """
    gm = GraphManager()
    gm.add_node(_Clock("fast", 0.001, gain=0.0, bias=0.0))
    gm.add_node(_Clock("slow", MACRO_DT, gain=0.0, bias=0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # disconnected nodes: expected
        gm.compile()
    assert gm.is_multirate
    for run in (lambda: gm.run_adaptive(T_END),
                lambda: gm.run_adaptive_scan(T_END, max_steps=4)):
        with pytest.raises(RuntimeError, match="multi-rate") as excinfo:
            run()
        message = str(excinfo.value)
        assert "All nodes must share the same timestep" not in message
        assert "subcycling=True" in message
        assert str({"fast": 1, "slow": 10}) in message, message
