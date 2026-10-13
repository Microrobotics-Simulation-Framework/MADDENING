"""A ``node.params`` write made after compile reaches the step at the next compile.

``compile()`` builds the step with ``gm.params``, and used to keep every
live leaf over the node's own value -- so a write to ``node.params`` after
the first compile was dropped, even across a recompile, with no sign: a
``HeartPumpNode`` set to 144 bpm and recompiled went on at 72.  (0.3.x, which
had no ``gm.params``, took such a write at the recompile.)

The rule now, per leaf: the later write wins.  A ``node.params`` write
reaches ``gm.params`` at the next read of it, the next run or the next
compile; a ``gm.params`` write reads ``gm.params`` first, so the two are
always ordered.  A calibration written into ``gm.params`` survives a
recompile; a node write after it replaces it.
"""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.ball import BallNode
from maddening.nodes.heart_pump import HeartPumpNode
from maddening.nodes.spring import SpringDamperNode


def _heart():
    gm = GraphManager()
    gm.add_node(HeartPumpNode(name="heart", timestep=0.001))
    gm.compile()
    return gm


def _phase_after(gm, n=100):
    gm.run_scan(n)
    return float(gm.get_node_state("heart")["phase"])


def _spring():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.5,
                                 initial_position=0.5))
    gm.compile()
    return gm


def _f32(x):
    return jnp.asarray(x, jnp.float32)


@pytest.mark.parametrize("recompile", ["dirty", "compile"])
def test_a_node_params_write_after_compile_takes_effect_at_the_next_compile(recompile):
    """The examples' own pattern: write ``node.params``, mark the graph dirty
    (or call ``compile()``), step.  The phase advances at ``rate / 60`` per
    second, so 144 bpm covers twice the phase 72 does."""
    at_72 = _phase_after(_heart())
    gm = _heart()
    gm.get_node("heart").params["heart_rate"] = 144.0
    if recompile == "dirty":
        gm._dirty = True  # noqa: SLF001
    else:
        gm.compile()
    assert _phase_after(gm) == pytest.approx(2.0 * at_72, rel=1e-5)
    assert float(gm.params["nodes"]["heart"]["heart_rate"]) == 144.0


def test_a_calibrated_leaf_survives_a_recompile_its_node_did_not_change():
    """The other half of the rule, pinned: a fit written into ``gm.params``
    is not discarded by a recompile (an added node, here)."""
    gm = _spring()
    gm.params["nodes"]["s"]["stiffness"] = _f32(35.0)
    gm.add_node(BallNode("b", 0.01))
    gm.compile()
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 35.0
    gm.compile()                                     # and again
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 35.0


def test_the_newer_write_wins_across_compiles():
    """A calibration, a compile, then a node write: the node's is newer."""
    gm = _spring()
    gm.params["nodes"]["s"]["stiffness"] = _f32(35.0)
    gm.compile()
    gm.get_node("s").params["stiffness"] = 50.0
    gm.compile()
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 50.0
    # ... and a later calibration is newer than that.
    gm.params["nodes"]["s"]["stiffness"] = _f32(41.0)
    gm.compile()
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 41.0


def test_the_later_of_two_disagreeing_writes_wins():
    """A ``gm.params`` write reads ``gm.params`` first, which takes in any
    pending ``node.params`` write, so the order of the two is exact: the
    later wins, either way round, with nothing to warn about.  (0.4.0
    development builds kept ``gm.params`` and warned, which lost a
    ``gm.params`` write of the value the node had before -- reverting a
    node write through ``gm.params``.)"""
    gm = _spring()
    gm.get_node("s").params["stiffness"] = 50.0
    gm.params["nodes"]["s"]["stiffness"] = _f32(41.0)      # later
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.compile()
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 41.0
    gm.params["nodes"]["s"]["stiffness"] = _f32(43.0)
    gm.get_node("s").params["stiffness"] = 52.0              # later
    gm.compile()
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 52.0
    gm.get_node("s").params["stiffness"] = 60.0
    gm.params["nodes"]["s"]["stiffness"] = _f32(52.0)      # back to what it was: still a write
    gm.compile()
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 52.0


def test_two_agreeing_writes_are_no_conflict():
    """``PUT /graph/params`` writes both the node and ``gm.params``."""
    gm = _spring()
    gm.get_node("s").params["stiffness"] = 50.0
    gm.params["nodes"]["s"]["stiffness"] = _f32(50.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.compile()
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 50.0


def test_a_compile_that_raises_does_not_lose_the_write(monkeypatch):
    """A compile that fails after the params merge commits nothing, so its
    view of "what changed" must not be kept either: the next compile still
    sees the node write as new."""
    gm = _heart()
    gm.get_node("heart").params["heart_rate"] = 144.0

    def broken(*args, **kwargs):
        raise RuntimeError("step build failed")

    monkeypatch.setattr(gm, "_build_step_fn", broken)
    with pytest.raises(RuntimeError, match="step build failed"):
        gm.compile()
    monkeypatch.undo()
    gm.compile()
    assert float(gm.params["nodes"]["heart"]["heart_rate"]) == 144.0


def test_an_initial_condition_written_on_the_node_is_taken_not_refused():
    """A path the rule newly enables.  An ``initial_*`` leaf is read by
    ``initial_state()`` from the node, not by the step from ``gm.params``;
    a ``gm.params`` value that differs from the node's is refused at the next
    run (it would be ignored and then saved).  The kept stale leaf used to
    differ from a node written after compile, so the next run raised for a
    write the user made the documented way.  Now the leaf follows the node,
    the run goes ahead, and ``reset_state()`` starts from the new value."""
    gm = _spring()
    gm.get_node("s").params["initial_position"] = 0.9
    gm.compile()
    assert float(gm.params["nodes"]["s"]["initial_position"]) == pytest.approx(0.9)
    gm.step()
    gm.reset_state()
    assert float(gm.get_node_state("s")["position"]) == pytest.approx(0.9)


def test_reset_params_after_a_node_write_is_no_conflict():
    gm = _spring()
    gm.get_node("s").params["stiffness"] = 50.0
    gm.reset_params()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.compile()
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 50.0


class _Counted(SimulationNode):
    """``x += count * gain * dt``: ``count`` is an int, so structural (not in
    ``gm.params``; read when the step is traced), ``gain`` a float leaf."""

    def initial_state(self):
        return {"x": jnp.array(0.0, jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": state["x"] + int(self.params["count"]) * p["gain"] * dt}


def test_a_structural_write_and_a_leaf_write_take_the_same_path():
    """A structural value has always reached the step through a recompile;
    a float leaf now does too, so one ``node.params`` habit covers both."""
    gm = GraphManager()
    gm.add_node(_Counted("c", 1.0, count=1, gain=1.0))
    gm.compile()
    assert set(gm.params["nodes"]["c"]) == {"gain"}
    node = gm.get_node("c")
    node.params["count"] = 3
    node.params["gain"] = 2.0
    gm.compile()
    gm.step()
    assert float(gm.get_node_state("c")["x"]) == 6.0
