"""A write to a ``gm.params`` leaf the compiled step cannot read is refused.

``HeatNode.grid_points`` (non-uniform grid), ``WaveletAdaptiveNode.mass``,
the ``initial_*`` leaves and geometry a node consumes in ``__init__`` are
leaves of ``gm.params`` -- ``params_pytree`` exposes every float -- but the
step never reads them: the first two are baked into a static when the node
is constructed (``static_data_deps`` says so), the others are read by
nothing in the step at all.  An edit to one was accepted by ``gm.params``,
``check_params`` and every run method, ignored by the step, and then
serialised by ``to_dict()``, so the saved graph reloaded as a different
model (0.093 K after one heat step; 0.071 in the wavelet's ``c``).
``docs/user_guide/parameters.md`` promises "not a silently ignored leaf".

The graph now refuses such a leaf wherever it would be used -- every run
method, ``check_params``, ``to_dict``, ``save_state`` -- naming the leaf and
why.  Two sources of "cannot read", declared first: a ``static_data_deps``
entry, and a structural walk of the traced step (a leaf no operation of the
step takes as an input).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core import graph_manager as gm_mod
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes import HeatNode
from maddening.nodes.spring import SpringDamperNode

GP0 = [0.05, 0.2, 0.35, 0.6, 0.8, 0.95]
GP1 = [0.1, 0.25, 0.4, 0.55, 0.7, 0.9]


def _heat(gp=GP0):
    gm = GraphManager()
    gm.add_node(HeatNode("h", 1e-2, n_cells=6, thermal_diffusivity=0.02,
                         initial_temperature=[300, 320, 340, 330, 310, 305], grid_points=gp))
    gm.compile()
    return gm


def _spring():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=40.0, rest_length=0.6,
                                 initial_position=0.2))
    gm.compile()
    return gm


_ENTRY_POINTS = {
    "step": lambda gm: gm.step(),
    "run": lambda gm: gm.run(2),
    "run_scan": lambda gm: gm.run_scan(2),
    "run_scan_with_history": lambda gm: gm.run_scan_with_history(2),
    "check_params": lambda gm: gm.check_params(),
    "to_dict": lambda gm: gm.to_dict(),
}


@pytest.mark.parametrize("entry", list(_ENTRY_POINTS))
def test_a_declared_baked_leaf_edit_is_refused_by_name(entry):
    """``grid_points`` is baked into ``static_data['grid_x']``."""
    gm = _heat()
    gm.params["nodes"]["h"]["grid_points"] = jnp.asarray(GP1, jnp.float32)
    with pytest.raises(ValueError, match=r"\['h'\]\['grid_points'\].*static_data \['grid_x'\]"):
        _ENTRY_POINTS[entry](gm)


def test_the_refused_edit_is_never_serialised(tmp_path):
    gm = _heat()
    gm.params["nodes"]["h"]["grid_points"] = jnp.asarray(GP1, jnp.float32)
    with pytest.raises(ValueError, match="grid_points"):
        gm.save_state(tmp_path / "ckpt.npz")
    assert not (tmp_path / "ckpt.npz").exists()


def test_an_initial_condition_edit_is_refused_as_never_read():
    """No declaration: the traced step has no path from the leaf."""
    gm = _spring()
    gm.params["nodes"]["s"]["initial_position"] = jnp.asarray(0.9, jnp.float32)
    with pytest.raises(ValueError, match=r"'initial_position'.*node cannot read it"):
        gm.step()


def test_a_leaf_only_this_graph_does_not_exercise_is_carried_not_refused():
    """A ball reads ``elasticity`` only when a ``table_position`` edge
    exists.  Without one the step never reads the leaf -- but the value in
    gm.params is latent, not ignored: the node reads it as soon as the input
    arrives, and the carried value is then the one used, which is what
    ``to_dict()`` records.  Refusing it would refuse a calibration of a
    graph that is merely missing an edge."""
    from maddening.nodes.ball import BallNode
    from maddening.nodes.table import TableNode

    def graph(e, edge):
        gm = GraphManager()
        gm.add_node(BallNode("b", 0.01, initial_position=0.05, initial_velocity=-2.0,
                             elasticity=e))
        gm.add_node(TableNode("t", 0.01, position=0.0))
        if edge:
            gm.add_edge("t", "b", "position", "table_position")
        gm.compile()
        return gm

    gm = graph(1.0, edge=False)
    gm.params["nodes"]["b"]["elasticity"] = jnp.asarray(0.5, jnp.float32)
    gm.run(3)
    assert [n for n in gm.to_dict()["nodes"] if n["name"] == "b"][0]["params"]["elasticity"] == 0.5
    gm.add_edge("t", "b", "position", "table_position")
    gm.reset_state()
    got = gm.run_scan(5)["b"]["velocity"]
    np.testing.assert_array_equal(np.asarray(got), np.asarray(graph(0.5, edge=True).run_scan(5)["b"]["velocity"]))
    assert not np.allclose(np.asarray(got), np.asarray(graph(1.0, edge=True).run_scan(5)["b"]["velocity"]))


def test_an_explicit_params_argument_is_not_refused_but_the_live_leaves_under_it_are():
    """A caller's ``params=`` is never serialised, and a leaf the step
    ignores may be one the caller's own code consumes (a residual seeding
    the initial state from ``initial_velocity``); it runs.  The live leaves
    a partial pytree is completed from are ``gm.params`` and are checked."""
    gm = _spring()
    gm.run_scan(2, params={"nodes": {"s": {"initial_position": jnp.asarray(0.9, jnp.float32)}}})
    assert float(gm.params["nodes"]["s"]["initial_position"]) == pytest.approx(0.2)
    gm.params["nodes"]["s"]["initial_velocity"] = jnp.asarray(0.3, jnp.float32)
    with pytest.raises(ValueError, match=r"gm\.params\['nodes'\]\['s'\]\['initial_velocity'\]"):
        gm.run_scan(2, params={"nodes": {"s": {"stiffness": jnp.asarray(50.0, jnp.float32)}}})


def test_an_edit_to_a_leaf_the_step_reads_is_honoured():
    """The refusal is for ignored leaves only: a live edit runs as before,
    and so does restoring a refused leaf."""
    gm = _spring()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(52.0, jnp.float32)
    ref = GraphManager()
    ref.add_node(SpringDamperNode("s", 0.01, stiffness=52.0, rest_length=0.6, initial_position=0.2))
    ref.compile()
    np.testing.assert_array_equal(np.asarray(gm.run_scan(3)["s"]["position"]),
                                  np.asarray(ref.run_scan(3)["s"]["position"]))

    gm.params["nodes"]["s"]["initial_position"] = jnp.asarray(0.9, jnp.float32)
    with pytest.raises(ValueError):
        gm.step()
    gm.params["nodes"]["s"]["initial_position"] = jnp.asarray(0.2, jnp.float32)
    gm.step()


def test_reset_params_is_the_way_back():
    gm = _heat()
    gm.params["nodes"]["h"]["grid_points"] = jnp.asarray(GP1, jnp.float32)
    gm.reset_params()
    gm.step()
    assert gm.to_dict()["nodes"][0]["params"]["grid_points"] == pytest.approx(GP0)


def test_a_traced_leaf_is_not_compared():
    """A fit or an FIM differentiates the whole pytree, frozen leaves
    included; a tracer cannot be compared and must pass untouched."""
    gm = _spring()
    g = jax.grad(lambda p: gm.run_scan(3, params=p)["s"]["position"])(gm.params)
    assert float(g["nodes"]["s"]["stiffness"]) != 0.0
    assert float(g["nodes"]["s"]["initial_position"]) == 0.0


def test_a_rest_style_write_that_reaches_the_node_is_not_refused():
    """``PUT /graph/params`` writes both the leaf and ``node.params``; the
    node's own value is the reference, so the pair is consistent."""
    gm = _spring()
    gm.params["nodes"]["s"]["initial_position"] = jnp.asarray(0.9, jnp.float32)
    gm._nodes["s"].node.params["initial_position"] = 0.9
    gm.step()


def test_a_steady_run_does_not_retrace_for_the_check():
    """The structural trace is taken only when a leaf differs from its
    node, once per compile, and is not counted as a step trace."""
    gm = _spring()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(52.0, jnp.float32)
    calls = []
    real = gm_mod._param_leaves_read
    gm_mod._param_leaves_read = lambda *a: (calls.append(1), real(*a))[1]
    try:
        for _ in range(3):
            gm.step()
    finally:
        gm_mod._param_leaves_read = real
    assert len(calls) == 1
    assert gm.trace_count == 1


# ------------------------------------------------------------------
# The structural walk itself
# ------------------------------------------------------------------

class _Branchy(SimulationNode):
    """``a`` read inside a while loop, ``b`` in one cond branch, ``c`` inside
    a scan and a nested ``jax.jit``; ``dead`` is routed through all four --
    a discarded while carry, a discarded scan carry, a cond operand no
    branch reads, a jitted function's unused argument -- and so read by
    nothing.  A walk that stopped at any of the four (called every input
    of an unknown equation live) would report ``dead`` read.

    ``e_scan`` and ``e_while`` are the other direction: each reaches the
    output only on its loop's *second* iteration, through carries that are
    themselves discarded (``w -> u -> x``), and ``k_cond`` rides a while
    carry that only the loop's condition reads.  A walk that read a loop body as a single
    call -- no fixed point over the carries -- or ignored the condition
    would report them dead, and the graph would refuse a write the step
    does read.  One parameter per loop, so each loop is pinned alone."""

    def __init__(self):
        super().__init__("n", 0.1, a=1.0, b=2.0, c=3.0, e_scan=5.0, e_while=6.0,
                         k_cond=2.0, dead=4.0)

    def initial_state(self):
        return {"x": jnp.asarray(1.0, jnp.float32)}

    def update(self, state, bi, dt, *, params=None):
        p = {**self.params, **(params or {})}
        zero = jnp.zeros((), jnp.float32)
        x = jax.lax.while_loop(
            lambda v: v[1] < 3,
            lambda v: (v[0] * p["a"], v[1] + 1, v[2] * 2.0),
            (state["x"], 0, p["dead"]))[0]
        x = jax.lax.cond(x > 0, lambda v, d: v + p["b"], lambda v, d: v, x, p["dead"])
        x = jax.lax.scan(lambda carry, _: ((carry[0] + p["c"], carry[1] + 1.0), None),
                         (x, p["dead"]), None, length=2)[0][0]
        x = jax.jit(lambda q, v: v + q["c"])(p, x)
        x = jax.lax.scan(lambda v, _: ((v[0] + v[1], v[2] * 1.0, v[2] + 0.0), None),
                         (x, zero, p["e_scan"]), None, length=2)[0][0]
        x = jax.lax.while_loop(
            lambda v: v[3] < 2,
            lambda v: (v[0] + v[1], v[2] * 1.0, v[2] + 0.0, v[3] + 1),
            (x, zero, p["e_while"], 0))[0]
        # ``k_cond`` rides a carry that only the condition reads.
        x = jax.lax.while_loop(lambda v: v[1] < v[2],
                               lambda v: (v[0] * 1.5, v[1] + 1.0, v[2] * 1.0),
                               (x, zero, p["k_cond"]))[0]
        return {"x": x}


def test_the_structural_walk_follows_loops_and_branches():
    gm = GraphManager()
    gm.add_node(_Branchy())
    gm.compile()
    reads = gm_mod._param_leaves_read(gm._raw_step_fn, gm._state,
                                      gm._default_external_inputs(), gm.params)
    assert reads == {("n", k) for k in ("a", "b", "c", "e_scan", "e_while", "k_cond")}
