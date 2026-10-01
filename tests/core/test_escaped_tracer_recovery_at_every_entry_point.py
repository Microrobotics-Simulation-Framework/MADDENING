"""Every public entry point that touches the state puts the graph back first.

``jax.grad`` of a loss that calls ``run_scan`` leaves the traced final
state in the graph; ``_recover_from_escaped_tracers`` is documented as
"called from every entry point", and its warning names ``set_node_state``
/ ``reset_state`` / ``load_state`` as the remedy.  ``reset_state`` did not
call it: it sized a predictor group's history seed from the live, traced
slot and raised ``UnexpectedTracerError`` -- the remedy failed.
``set_node_state`` did not either: its write went into the traced state,
and the next entry point put the graph back over it, silently.  The same
held for ``get_node_state`` (which handed back escaped tracers),
``resolve_boundary_inputs``, ``validate``, ``add_node`` and ``remove_node``.
"""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

_RECOVERED = "the graph held JAX tracers"


def _graph(**group):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.add_node(SpringDamperNode("b", 0.01, stiffness=30.0, damping=2.0, initial_position=0.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], max_iterations=8, tolerance=1e-6, **group)
    gm.compile()
    return gm


def _differentiated(**group):
    """A graph left holding the tracers of ``jax.grad`` through ``run_scan``.

    One concrete step first, so the state the graph is put back to is
    distinguishable from both ``initial_state()`` and the traced one.
    """
    gm = _graph(**group)
    gm.step()
    before = {n: np.asarray(gm.get_node_state(n)["position"]) for n in ("a", "b")}

    def loss(p):
        return jnp.sum(gm.run_scan(3, params=p)["a"]["position"] ** 2)

    jax.grad(loss)(gm.params)
    assert gm._state_traced  # noqa: SLF001 - the premise of every test here
    return gm, before


@pytest.mark.parametrize("predictor", ["none", "linear", "quadratic"])
def test_reset_state_after_grad_resets_quietly(predictor):
    """No ``UnexpectedTracerError``, and no warning: the reset replaces it all."""
    gm, _before = _differentiated(predictor=predictor)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.reset_state()
    assert not gm._state_traced  # noqa: SLF001
    assert float(gm.get_node_state("a")["position"]) == 1.0
    assert float(gm.get_node_state("b")["position"]) == 0.0
    gm.step()      # the compiled step still runs, from the fresh state
    assert gm.trace_count == 1


def test_set_node_state_after_grad_survives_the_next_step():
    """The write lands in the state that is kept, not in the traced one."""
    gm, _before = _differentiated()
    with pytest.warns(RuntimeWarning, match=_RECOVERED):
        gm.set_node_state("a", {"position": jnp.float32(0.25), "velocity": jnp.float32(0.0)})
    assert not gm._state_traced  # noqa: SLF001
    assert float(gm.get_node_state("a")["position"]) == 0.25
    ref = _graph()
    ref.set_node_state("a", {"position": jnp.float32(0.25), "velocity": jnp.float32(0.0)})
    ref.set_node_state("b", dict(gm.get_node_state("b")))
    gm.step()
    ref.step()
    for n in ("a", "b"):
        assert np.asarray(gm.get_node_state(n)["position"]).tobytes() == \
            np.asarray(ref.get_node_state(n)["position"]).tobytes()


def test_get_node_state_after_grad_returns_the_recovered_state():
    gm, before = _differentiated()
    with pytest.warns(RuntimeWarning, match=_RECOVERED):
        s = gm.get_node_state("a")
    # ``np.asarray`` of an escaped tracer raises; the recovered state is concrete.
    assert np.asarray(s["position"]).tobytes() == before["a"].tobytes()


def test_resolve_boundary_inputs_after_grad_reads_the_recovered_state():
    gm, before = _differentiated()
    with pytest.warns(RuntimeWarning, match=_RECOVERED):
        bi = gm.resolve_boundary_inputs("a")
    assert np.asarray(bi["anchor_position"]).tobytes() == before["b"].tobytes()


def test_validate_after_grad_reads_the_recovered_state():
    gm, _before = _differentiated()
    with pytest.warns(RuntimeWarning, match=_RECOVERED):
        issues = gm.validate()
    assert not [i for i in issues if i.startswith("ERROR")]
    assert not gm._state_traced  # noqa: SLF001


def test_add_node_after_grad_keeps_the_new_node():
    """Added to the traced state, the node's state vanished on the next recovery."""
    gm, _before = _differentiated()
    with pytest.warns(RuntimeWarning, match=_RECOVERED):
        gm.add_node(SpringDamperNode("c", 0.01, initial_position=2.0))
    gm.step()
    assert float(gm.get_node_state("c")["position"]) == pytest.approx(2.0, abs=1e-3)


def test_remove_node_after_grad_leaves_no_state_behind():
    gm = _graph()
    gm.add_node(SpringDamperNode("c", 0.01, initial_position=2.0))
    gm.compile()

    def loss(p):
        return jnp.sum(gm.run_scan(2, params=p)["a"]["position"])

    jax.grad(loss)(gm.params)
    with pytest.warns(RuntimeWarning, match=_RECOVERED):
        gm.remove_node("c")
    assert "c" not in gm._state  # noqa: SLF001
    gm.step()


def test_inside_a_transform_nothing_is_put_back():
    """A traced multi-step loop still works: the entry points do nothing there."""
    gm = _graph(predictor="linear")

    def loss(p):
        gm.step(params=p)
        gm.set_node_state("b", dict(gm.get_node_state("b")))
        return jnp.sum(gm.run_scan(2, params=p)["a"]["position"])

    g = jax.grad(loss)(gm.params)
    assert np.isfinite(float(g["nodes"]["a"]["stiffness"]))
