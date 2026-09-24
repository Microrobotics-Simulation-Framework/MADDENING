"""``run_adaptive`` / ``run_adaptive_scan`` resolve flux edges.

Both adaptive runners step through ``GraphManager._build_dt_step_fn``,
whose boundary-input resolution read ``state[source][field]`` only, so
any edge from a ``compute_boundary_fluxes`` output was a bare
``KeyError: 'spring_force'`` -- with or without ``params`` -- while
``step`` / ``run_scan`` resolved the same edge from the fluxes the source
node produced earlier in the step.  The dt step now resolves it the same
way; an edge it genuinely cannot resolve (a flux on a back edge) is a
``ValueError`` naming the edge.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.nodes.spring import SpringDamperNode


class _Relay(SimulationNode):
    """Integrates the force it is fed: ``dy/dt = 0.01 * inp``.  An ODE and
    not an algebraic relay, so the step-doubling error of the adaptive
    runners shrinks with ``dt`` and the controller can do its job."""

    def __init__(self, name="R"):
        super().__init__(name, 0.01)

    def initial_state(self):
        return {"y": jnp.asarray(0.1, jnp.float32)}

    def update(self, state, bi, dt):
        return {"y": state["y"] + dt * 0.01 * jnp.asarray(bi.get("inp", 0.0), jnp.float32)}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=())}


def _graph(k=40.0, *, relay_first=False):
    gm = GraphManager()
    spring = SpringDamperNode("s", 0.01, stiffness=k, damping=0.5, rest_length=0.6,
                              initial_position=0.2)
    for node in ([_Relay(), spring] if relay_first else [spring, _Relay()]):
        gm.add_node(node)
    gm.add_edge("s", "R", "spring_force", "inp")
    gm.add_edge("R", "s", "y", "anchor_position")
    gm.compile()
    return gm


def test_the_dt_step_resolves_a_flux_edge_exactly_as_the_fixed_step():
    """At the node's own timestep the two step functions are one step."""
    gm = _graph()
    ext = gm._default_external_inputs()
    fixed = gm._compiled_step(gm._state, ext, gm.params)
    adaptive = gm._build_dt_step_fn()(gm._state, ext, jnp.asarray(0.01, jnp.float32), gm.params)
    for n in ("s", "R"):
        for f in fixed[n]:
            np.testing.assert_array_equal(np.asarray(adaptive[n][f]), np.asarray(fixed[n][f]))
    # The relay read the spring's force, not a default of zero.
    assert float(adaptive["R"]["y"]) != pytest.approx(0.1, abs=1e-6)


@pytest.mark.parametrize("runner", ["run_adaptive", "run_adaptive_scan"])
def test_both_adaptive_runners_run_a_flux_edge_graph_with_injected_params(runner):
    """They run, and an injected stiffness acts like a constructed one."""
    def final(gm, params=None):
        out = getattr(gm, runner)(0.1, params=params)[0]
        return np.asarray(out["R"]["y"]), np.asarray(out["s"]["position"])

    injected = final(_graph(40.0), {"nodes": {"s": {"stiffness": jnp.asarray(52.0, jnp.float32)}}})
    constructed = final(_graph(52.0))
    default = final(_graph(40.0))
    np.testing.assert_array_equal(injected[0], constructed[0])
    np.testing.assert_array_equal(injected[1], constructed[1])
    assert not np.allclose(default[1], constructed[1])


def test_a_flux_on_a_back_edge_is_refused_by_name():
    """Scheduled relay-first, the flux edge is the back edge: its value
    would be the previous step's flux, which the dt step does not keep."""
    gm = _graph(relay_first=True)
    back = {e.key for e in gm._back_edges}
    assert "s.spring_force->R.inp" in back
    with pytest.raises(ValueError, match=r"s\.spring_force->R\.inp.*back edge"):
        gm.run_adaptive(0.05)
    with pytest.raises(ValueError, match=r"s\.spring_force->R\.inp.*back edge"):
        jax.block_until_ready(gm.run_adaptive_scan(0.05))
