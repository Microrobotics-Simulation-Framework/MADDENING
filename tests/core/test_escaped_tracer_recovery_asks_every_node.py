"""The graph is put back after ``jax.grad`` whatever its first entry holds.

``step`` / ``run_scan`` assign their result into the graph, so a loss that
calls one leaves tracers in it when ``jax.grad`` returns, and every entry
point puts the graph back first (``_recover_from_escaped_tracers``).  It
recognised a traced state by its *first* entry only.  A transform's output
is traced only where it depends on what is differentiated, and a state that
has been through a ``lax.scan`` has its keys sorted, so two ordinary graphs
defeated it and their next ``step()`` raised JAX's ``UnexpectedTracerError``:

* a coupled graph whose node names sort after ``"_"``: ``_meta`` comes
  first, and its slots come back concrete;
* a first node that reads no parameter (a clock): concrete beside traced
  nodes.

``coupling_diagnostics()`` did not put the graph back at all.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.spring import SpringDamperNode


class Clock(SimulationNode):
    """Reads no parameter, so a gradient over the params leaves it concrete."""

    def initial_state(self):
        return {"t": jnp.array(0.0, jnp.float32)}

    def update(self, s, bi, dt):
        return {"t": s["t"] + dt}


def _graph(first=None, coupled=True, names=("a", "b")):
    gm = GraphManager()
    if first is not None:
        gm.add_node(first)
    a, b = names
    gm.add_node(SpringDamperNode(a, 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.add_node(SpringDamperNode(b, 0.01, stiffness=30.0, damping=2.0, initial_position=0.0))
    gm.add_edge(a, b, "position", "anchor_position")
    gm.add_edge(b, a, "position", "anchor_position")
    if coupled:
        gm.add_coupling_group([a, b], max_iterations=8, tolerance=1e-6)
    gm.compile()
    return gm


def _differentiate_through_run_scan(gm, node):
    def loss(p):
        return jnp.sum(gm.run_scan(3, params=p)[node]["position"] ** 2)

    grad = jax.grad(loss)(gm.params)
    assert np.isfinite(float(grad["nodes"][node]["stiffness"]))


_BUILDERS = {
    "meta-sorts-first": lambda: _graph(),
    "param-free-first-node": lambda: _graph(first=Clock("a0", 0.01)),
    "param-free-first-node-uncoupled": lambda: _graph(first=Clock("a0", 0.01),
                                                      coupled=False),
}


@pytest.mark.parametrize("case", sorted(_BUILDERS))
def test_step_after_grad_of_run_scan_puts_the_graph_back(case):
    gm = _BUILDERS[case]()
    _differentiate_through_run_scan(gm, "a")
    assert gm._state_traced, list(gm._state)
    with pytest.warns(RuntimeWarning, match="tracers"):
        gm.step()
    assert not gm._state_traced
    # Put back to the state before the transform, then stepped once: the
    # state a fresh graph reaches in one step.
    fresh = _BUILDERS[case]()
    fresh.step()
    for n in fresh.node_names:
        for f, v in fresh.get_node_state(n).items():
            np.testing.assert_array_equal(np.asarray(gm.get_node_state(n)[f]),
                                          np.asarray(v), err_msg=f"{n}.{f}")


def _as_text(report):
    return {k: repr(v) for k, v in dict(report).items()}


def test_coupling_diagnostics_after_grad_of_run_scan_puts_the_graph_back():
    gm = _graph()
    gm.step()
    report_before = _as_text(gm.coupling_diagnostics()["a+b"])
    _differentiate_through_run_scan(gm, "a")
    with pytest.warns(RuntimeWarning, match="tracers"):
        report = gm.coupling_diagnostics()
    # The report of the state the graph was put back to: the one step
    # taken before the transform.
    assert _as_text(report["a+b"]) == report_before


def test_a_graph_whose_first_node_sorts_before_meta_was_already_recovered():
    """The case the first-entry check did see, kept as the control."""
    gm = _graph(names=("A", "B"))
    def loss(p):
        return jnp.sum(gm.run_scan(3, params=p)["A"]["position"] ** 2)
    jax.grad(loss)(gm.params)
    assert list(gm._state)[0] == "A" and gm._state_traced
    with pytest.warns(RuntimeWarning, match="tracers"):
        gm.step()
