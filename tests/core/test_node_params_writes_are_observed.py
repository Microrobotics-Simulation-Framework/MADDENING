"""A ``node.params`` write is observed as a write, and every entry point
runs the model it leaves.

Before: the graph inferred a write from "the node's value changed since the
last compile", so

* a write of the value the node already held -- reverting a calibration
  written into ``gm.params`` -- was dropped;
* a write made before ``load_state`` won over the restored value, or not,
  depending on whether ``load_state`` found the graph dirty (it compiles a
  dirty graph first);
* a write with no compile reached whichever entry point traced next: a new
  ``run_scan`` length or ``windowed_loss`` picked it up while ``gm.step``
  kept its cached trace and ran the old model
  (audit_040_p4_5/fmu-sysid/repro_node_params_writes.py,
  repro_node_params_retrace.py).

Now ``node.params`` counts its writes (``_ParamsDict``); every entry point
checks the counts first and recompiles for a write the compiled step has not
taken; and ``load_state`` supersedes earlier writes to the constants it
restores.
"""

import copy
import json
import os
import pickle
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode, _ParamsDict
from maddening.nodes.spring import SpringDamperNode


def _spring():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.5,
                                 initial_position=0.5))
    gm.compile()
    return gm


def _k(gm):
    return float(gm.params["nodes"]["s"]["stiffness"])


def _k_stepped(gm):
    """The stiffness the next ``gm.step()`` integrates with, from one step."""
    st = gm.get_node_state("s")
    x, v = float(st["position"]), float(st["velocity"])
    gm.step()
    v2 = float(gm.get_node_state("s")["velocity"])
    return (-(v2 - v) / 0.01 * 1.5 - 2.0 * v) / (x - 1.0)


class _Counted(SimulationNode):
    """``x += count * gain * dt``: ``count`` is structural (an int, read when
    the step is traced), ``gain`` a float leaf of ``gm.params``."""

    def initial_state(self):
        return {"x": jnp.array(0.0, jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": state["x"] + int(self.params["count"]) * p["gain"] * dt}


def _counted(**kw):
    gm = GraphManager()
    gm.add_node(_Counted("c", 1.0, **{"count": 1, "gain": 1.0, **kw}))
    gm.compile()
    return gm


def _increment(gm):
    before = float(gm.get_node_state("c")["x"])
    gm.step()
    return float(gm.get_node_state("c")["x"]) - before


# ---------------------------------------------------------------------------
# The mapping counts its writes, and is a dict everywhere else
# ---------------------------------------------------------------------------


def test_every_mutation_through_the_mapping_counts():
    p = _ParamsDict(a=1.0, b=2.0)
    assert p._writes == 0
    p["a"] = 1.0                     # the value it already held: still a write
    p.update(b=3.0)
    p.setdefault("c", 4.0)
    p.setdefault("c", 5.0)           # present: not a write
    p.pop("c")
    p.pop("missing", None)           # absent: not a write
    p |= {"d": 1.0}
    del p["d"]
    assert p._key_writes == {"a": 1, "b": 1, "c": 2, "d": 2}
    assert p._writes == 6
    p.clear()
    assert p._writes == 8 and p == {}


def test_a_copy_is_a_fresh_mapping_and_the_rest_sees_a_dict():
    p = _ParamsDict(a=1.0, b=[1, 2])
    p["a"] = 2.0
    for twin in (copy.deepcopy(p), pickle.loads(pickle.dumps(p))):
        assert type(twin) is _ParamsDict and twin == p and twin._writes == 0
    assert type(dict(p)) is dict and type(p.copy()) is dict
    assert isinstance(p, dict) and json.loads(json.dumps(p)) == {"a": 2.0, "b": [1, 2]}
    mapped = jax.tree.map(lambda x: x * 2, _ParamsDict(a=1.0, b=2.0))
    assert mapped == {"a": 2.0, "b": 4.0}
    assert [jax.tree_util.keystr(k) for k, _ in
            jax.tree_util.tree_flatten_with_path(p)[0]] == ["['a']", "['b'][0]", "['b'][1]"]


def test_a_node_holds_one():
    gm = _spring()
    assert type(gm.get_node("s").params) is _ParamsDict


# ---------------------------------------------------------------------------
# A write is a write, even of the value the node held
# ---------------------------------------------------------------------------


def test_writing_the_nodes_own_value_back_reverts_a_calibration():
    gm = _spring()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(42.0, jnp.float32)
    gm.compile()                                  # the calibration survives
    assert _k(gm) == 42.0
    gm.get_node("s").params["stiffness"] = 30.0   # the value the node held all along
    gm.compile()
    assert _k(gm) == 30.0
    assert _k_stepped(gm) == pytest.approx(30.0, rel=1e-3)


@pytest.mark.parametrize("dirty", [False, True])
def test_load_state_supersedes_an_earlier_node_write(tmp_path, dirty):
    """The restore is the later of the two writes, whether or not
    ``load_state`` found the graph dirty (it compiles a dirty graph first,
    which takes the write in before restoring)."""
    gm = _spring()
    gm.run(3)
    path = tmp_path / "k30.npz"
    gm.save_state(str(path))                          # saved with k = 30
    gm.run(2)
    gm.get_node("s").params["stiffness"] = 50.0       # node write ...
    if dirty:
        gm.add_external_input("s", "anchor_position")
    gm.load_state(str(path))                          # ... then the restore
    assert _k(gm) == 30.0
    gm.run(1)
    gm.compile()
    assert _k(gm) == 30.0
    assert _k_stepped(gm) == pytest.approx(30.0, rel=1e-3)


def test_load_state_leaves_a_structural_write_it_cannot_restore_pending(tmp_path):
    """A checkpoint holds ``gm.params`` and the state, not a structural
    constant: a write to one made before the restore still takes effect."""
    gm = _counted()
    path = tmp_path / "c.npz"
    gm.save_state(str(path))
    node = gm.get_node("c")
    node.params["count"] = 3
    node.params["gain"] = 5.0
    gm.load_state(str(path))
    assert float(gm.params["nodes"]["c"]["gain"]) == 1.0          # restored
    assert _increment(gm) == 3.0                                  # count 3, gain 1


# ---------------------------------------------------------------------------
# Every entry point runs the model the write leaves
# ---------------------------------------------------------------------------


def test_a_structural_write_reaches_gm_step_without_a_compile():
    """``gm.step`` used to keep the trace it had cached (factor 1) while a new
    ``run_scan`` length traced the write in (factor 3)."""
    gm = _counted()
    assert _increment(gm) == 1.0
    gm.get_node("c").params["count"] = 3
    assert _increment(gm) == 3.0
    x0 = float(gm.get_node_state("c")["x"])
    gm.run_scan(1)                                   # a new scan length
    assert float(gm.get_node_state("c")["x"]) - x0 == 3.0
    assert _increment(gm) == 3.0


def test_a_leaf_write_reaches_every_entry_point_too():
    gm = _counted()
    gm.run_scan(2)
    gm.get_node("c").params["gain"] = 4.0
    assert _increment(gm) == 4.0
    x0 = float(gm.get_node_state("c")["x"])
    gm.run_scan(2)                                   # the cached length
    assert float(gm.get_node_state("c")["x"]) - x0 == 8.0
    assert float(gm.params["nodes"]["c"]["gain"]) == 4.0


def test_a_write_gm_params_already_reflects_costs_no_recompile():
    """``PUT /graph/params`` writes both the node and ``gm.params``: the
    step already runs that value, so the write is absorbed, not recompiled,
    and the next compile does not read it as a pending node write."""
    gm = _counted()
    gm.step()
    generation = gm._compile_generation  # noqa: SLF001
    gm.params["nodes"]["c"]["gain"] = jnp.asarray(6.0, jnp.float32)
    gm.get_node("c").params["gain"] = 6.0
    assert _increment(gm) == 6.0
    assert gm._compile_generation == generation  # noqa: SLF001
    gm.params["nodes"]["c"]["gain"] = jnp.asarray(7.0, jnp.float32)   # a later calibration
    with warnings.catch_warnings():
        warnings.simplefilter("error")                               # no conflict warning
        gm.compile()
    assert float(gm.params["nodes"]["c"]["gain"]) == 7.0


def test_windowed_loss_runs_the_written_model_as_gm_step_does():
    from maddening.sysid import windowed_loss

    gm = _counted()
    gm.step()
    gm.get_node("c").params["count"] = 2
    s0 = gm._user_state(gm._state)  # noqa: SLF001
    _, hist = gm.run_scan_with_history(4)
    obs = jax.tree.map(lambda a, h: jnp.concatenate([jnp.asarray(a)[None], h]), s0, hist)
    loss = windowed_loss(gm, gm.params, obs, obs_fn=lambda s: s["c"]["x"], window=4)
    assert float(loss) == 0.0


@pytest.mark.parametrize("fitter", ["fit", "fit_lm", "fit_multiple_shooting"])
def test_a_fitter_starts_from_the_written_value(fitter):
    """A fit started after a node write, with no compile, starts where the
    graph now is: it used to read ``gm.params`` before the write had reached
    it and fit from the old value."""
    from maddening import sysid

    gm = _counted()
    gm.step()
    gm.get_node("c").params["gain"] = 4.0
    mask = jax.tree.map(lambda _: False, gm.trainable_mask(gm.params))
    mask["nodes"]["c"]["gain"] = True
    if fitter == "fit":
        res = sysid.fit(gm, lambda p: jnp.sum(p["nodes"]["c"]["gain"] ** 2), mask=mask,
                        n_iter=0)
    elif fitter == "fit_lm":
        res = sysid.fit_lm(gm, lambda p: p["nodes"]["c"]["gain"][None], mask=mask, n_iter=0)
    else:
        obs = {"c": {"x": jnp.zeros(3, jnp.float32)}}
        res, _ = sysid.fit_multiple_shooting(gm, obs, obs_fn=lambda s: s["c"]["x"],
                                             window=1, mask=mask, n_iter=0)
    assert float(res.params["nodes"]["c"]["gain"]) == 4.0
