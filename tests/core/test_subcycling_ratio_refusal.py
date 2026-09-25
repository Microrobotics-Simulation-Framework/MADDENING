"""A sub-cycled member whose timestep does not divide the macro timestep is refused.

A node in a ``subcycling=True`` group takes ``round(macro_dt / node_dt)``
sub-steps of its own timestep per coupling pass.  Unless that ratio is a
whole number it covers ``divider * node_dt`` per macro step, not
``macro_dt``: a clock on a node at 0.003 s in a group whose macro timestep
is 0.01 s read 0.09 after ten steps while the graph's read 0.1 -- and
``run_adaptive`` kept the same drift.  Nothing refused it.  ``compile()``
now does, naming the node, both timesteps and the nearest timesteps that
divide.
"""

import warnings

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

MACRO = 0.01


class _Clock(SimulationNode):
    """``t <- t + dt``; ``x <- gain * u + bias``."""

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


def _group(fast_dt, *, extra=None):
    gm = GraphManager()
    gm.add_node(_Clock("fast", fast_dt, 0.5, 0.0))
    gm.add_node(_Clock("slow", MACRO, 0.5, 1.0))
    gm.add_edge("slow", "fast", "x", "u")
    gm.add_edge("fast", "slow", "x", "u")
    members = ["fast", "slow"]
    if extra is not None:
        gm.add_node(_Clock("mid", extra, 0.0, 0.0))
        gm.add_edge("slow", "mid", "x", "u")
        gm.add_edge("mid", "slow", "x", "u")
        members.append("mid")
    gm.add_coupling_group(members, max_iterations=20, tolerance=1e-6,
                          subcycling=True)
    return gm


@pytest.mark.parametrize("fast_dt, covered, nearest", [
    (0.003, "0.009", ("0.00333333 (0.01/3)", "0.0025 (0.01/4)")),
    (0.004, "0.008", ("0.005 (0.01/2)", "0.00333333 (0.01/3)")),
    # A divider of one is no exception: 0.009 took one step of 0.009.
    (0.009, "0.009", ("0.01 (0.01/1)", "0.005 (0.01/2)")),
])
def test_a_member_whose_timestep_does_not_divide_is_refused(fast_dt, covered, nearest):
    gm = _group(fast_dt)
    with pytest.raises(RuntimeError, match="does not divide") as excinfo:
        gm.compile()
    message = str(excinfo.value)
    assert f"node 'fast' has timestep {fast_dt:g}" in message, message
    assert f"macro timestep {MACRO:g}" in message, message
    assert f"cover {covered} per macro step" in message, message
    for timestep in nearest:
        assert timestep in message, (timestep, message)
    # ``validate()`` names it too, without compiling.
    assert any("does not divide" in issue for issue in gm.validate())


def test_every_member_that_does_not_divide_is_named():
    gm = _group(0.003, extra=0.004)
    with pytest.raises(RuntimeError) as excinfo:
        gm.compile()
    message = str(excinfo.value)
    assert "node 'fast'" in message and "node 'mid'" in message, message


@pytest.mark.parametrize("fast_dt", [0.001, 0.0025, 0.005, MACRO / 3, MACRO / 7])
def test_a_member_that_divides_is_accepted_and_keeps_the_graphs_clock(fast_dt):
    """Decimal timesteps whose ratio is whole up to float64 noise compile.

    ``0.01 / 0.001`` is ``10.000000000000002`` in float64; ``MACRO / 7``
    round-trips to a ratio a few ulps from 7.  Both are admitted, and the
    sub-cycled clock keeps time with the graph's.
    """
    gm = _group(fast_dt)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.compile()
    gm.run_scan(10)
    assert float(gm.get_node_state("fast")["t"]) == pytest.approx(10 * MACRO, rel=1e-5)
    assert float(gm.get_node_state("slow")["t"]) == pytest.approx(10 * MACRO, rel=1e-5)


def test_the_tolerance_admits_float_noise_and_nothing_coarser():
    """``1e-9`` relative: a ratio 1e-12 off whole passes, 1e-6 off is refused."""
    from maddening.core.graph_manager import _SUBCYCLING_RATIO_RTOL

    assert _SUBCYCLING_RATIO_RTOL == 1e-9
    near = MACRO / (4 * (1 + 1e-12))
    far = MACRO / (4 * (1 + 1e-6))
    _group(near).compile()
    with pytest.raises(RuntimeError, match="does not divide"):
        _group(far).compile()
