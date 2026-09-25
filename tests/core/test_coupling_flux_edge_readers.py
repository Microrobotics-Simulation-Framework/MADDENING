"""A group-internal flux edge is refused where the loop reads it from the state.

A flux edge carries what ``compute_boundary_fluxes`` returns, which is not
a field of the producer's state.  Two parts of the coupling loop read an
internal edge's value from the state -- the interface norm, and a
sub-cycled member's linear (or quadratic) boundary interpolation -- and
both failed inside the trace with a bare ``KeyError`` naming the flux, at
every release since 0.1.0 (MADD-ANO-052).  ``compile()`` now refuses them
by name and says which setting works; the settings it names are checked
to work here too.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from maddening.core.coupling.helpers import add_flux_coupling
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.nodes.spring import SpringDamperNode


class Follower(SimulationNode):
    def initial_state(self):
        return {"y": jnp.array(0.0, jnp.float32)}

    def boundary_input_spec(self):
        return {"x": BoundaryInputSpec(shape=(), description="drive")}

    def update(self, s, bi, dt, *, params=None):
        x = bi.get("x", jnp.array(0.0, jnp.float32))
        return {"y": s["y"] + dt * (x - s["y"])}


def _flux_pair(follower_dt=0.01, **group):
    """``s.spring_force`` (a flux) drives ``k``; ``k.y`` (a state field) drives ``s``."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.add_node(Follower("k", follower_dt))
    add_flux_coupling(gm, "s", "k", "spring_force", "x")
    gm.add_edge("k", "s", "y", "anchor_position")
    gm.add_coupling_group(["s", "k"], max_iterations=8, **group)
    return gm


def test_the_interface_norm_on_a_flux_edge_is_refused_by_name():
    gm = _flux_pair(convergence_norm="interface", rtol=1e-4)
    with pytest.raises(RuntimeError) as info:
        gm.compile()
    message = str(info.value)
    assert "convergence_norm='interface'" in message
    assert "'spring_force'" in message and "s.spring_force->k.x" in message
    assert "compute_boundary_fluxes" in message
    assert "KeyError" not in type(info.value).__name__


@pytest.mark.parametrize("norm", ["mixed", "l2"])
def test_the_norms_the_refusal_names_step_the_same_group(norm):
    kw = {"rtol": 1e-4} if norm == "mixed" else {"tolerance": 1e-6}
    gm = _flux_pair(convergence_norm=norm, **kw)
    gm.compile()
    gm.step()
    assert gm.coupling_diagnostics()["k+s"]["converged"] is True


@pytest.mark.parametrize("interp", ["linear", "quadratic"])
def test_a_sub_cycled_consumer_of_a_flux_edge_needs_constant_interpolation(interp):
    kw = {} if interp == "linear" else {"boundary_interpolation": interp}
    gm = _flux_pair(follower_dt=0.001, tolerance=1e-6, subcycling=True, **kw)
    with pytest.raises(RuntimeError) as info:
        gm.compile()
    message = str(info.value)
    assert "node 'k' is sub-cycled (10 sub-steps per pass)" in message
    assert f"boundary_interpolation={interp!r}" in message
    assert "boundary_interpolation='constant'" in message


def test_constant_interpolation_steps_the_sub_cycled_flux_consumer():
    gm = _flux_pair(follower_dt=0.001, tolerance=1e-6, subcycling=True,
                    boundary_interpolation="constant")
    gm.compile()
    gm.step()
    assert gm.coupling_diagnostics()["k+s"]["converged"] is True


def test_the_slow_member_may_read_a_flux_under_linear_interpolation():
    """Only a member that sub-steps interpolates; the refusal is that narrow.

    Here the flux producer is the fast node and the consumer the slow one,
    which reads its inputs once per pass through the flux-aware path.
    """
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.001, stiffness=30.0, damping=2.0,
                                 initial_position=1.0))
    gm.add_node(Follower("k", 0.01))
    add_flux_coupling(gm, "s", "k", "spring_force", "x")
    gm.add_edge("k", "s", "y", "anchor_position")
    gm.add_coupling_group(["s", "k"], max_iterations=8, tolerance=1e-6,
                          subcycling=True)
    assert not [i for i in gm.validate() if i.startswith("ERROR")]
    gm.compile()
    gm.step()
    assert gm.coupling_diagnostics()["k+s"]["converged"] is True
