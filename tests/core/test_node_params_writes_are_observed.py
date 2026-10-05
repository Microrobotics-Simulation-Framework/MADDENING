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


def test_the_fmu_export_reads_a_pending_node_write():
    """The documented export wiring -- a model description and a sidecar
    built from ``gm.params`` and ``gm._compiled_step`` -- after a node write
    with nothing run: it used to export, and step, the value from before."""
    from maddening.fmi import build_model_description
    from maddening.fmi.package import MODEL_IDENTIFIER
    from maddening.fmi.sidecar import FmuSidecar, SidecarConfig

    gm = _spring()
    gm.get_node("s").params["stiffness"] = 45.0
    md = build_model_description(gm, model_name="S", model_identifier=MODEL_IDENTIFIER)
    start = next(v.start for v in md.variables if v.name == "s.params.stiffness")
    assert float(start) == 45.0
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,  # noqa: SLF001
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),  # noqa: SLF001
        fixed_params=md.fixed_parameters, input_resolver=gm._resolve_external_inputs))  # noqa: SLF001
    assert float(sidecar.params["nodes"]["s"]["stiffness"]) == 45.0


def test_assigning_gm_params_is_later_than_a_node_write_before_it():
    """``gm.params = tree`` is a write of every leaf, later than any
    ``node.params`` write before it -- the profiler restores the caller's
    tree this way after its own recompiles.  The tree was taken before the
    node write, so it holds the old value, and that value must stay."""
    gm = _spring()
    saved = jax.tree.map(lambda x: x, gm.params)
    gm.get_node("s").params["stiffness"] = 50.0
    gm.params = saved
    assert _k(gm) == 30.0
    assert _k_stepped(gm) == pytest.approx(30.0, rel=1e-3)


# ---------------------------------------------------------------------------
# A replaced mapping keeps counting, and an in-place write is a write
# (MADD-ANO-175)
# ---------------------------------------------------------------------------


def _saved_value(gm, node, key):
    return next(d for d in gm.to_dict()["nodes"] if d["name"] == node)["params"][key]


def test_assigning_node_params_stores_a_counting_mapping():
    """``node.params = {...}`` stores the items in a fresh ``_ParamsDict``;
    one that already counts (a sharded wrapper shares its node's) is stored
    as it is, and anything that is not a mapping is left alone."""
    node = SpringDamperNode("s", 0.01, stiffness=30.0)
    plain = {**node.params, "stiffness": 40.0}
    node.params = plain
    assert type(node.params) is _ParamsDict and node.params == plain
    assert node.params is not plain and node.params._writes == 0
    shared = _ParamsDict(a=1.0)
    node.params = shared
    assert node.params is shared
    assert "params" in vars(node) and vars(node)["params"] is shared
    other = SpringDamperNode("t", 0.01)
    del other.__dict__["params"]
    assert getattr(other, "params", None) is None          # AttributeError, not KeyError


def test_a_replaced_mapping_counts_the_writes_made_into_it_afterwards():
    """The idiom ``node.params = {**node.params, "k": v}`` then a write into
    the new mapping: the write used to be lost to ``gm.params``, every run,
    ``compile`` and ``to_dict`` (audit_040_p4_10/fmu-sysid/
    repro_params_sync_lost_writes.py, case A)."""
    gm = _spring()
    node = gm.get_node("s")
    node.params = {**node.params, "stiffness": 40.0}
    assert _k(gm) == 40.0
    node.params["stiffness"] = 99.0
    assert _k(gm) == 99.0
    assert _k_stepped(gm) == pytest.approx(99.0, rel=1e-3)
    gm.compile()
    assert _k(gm) == 99.0
    assert _saved_value(gm, "s", "stiffness") == 99.0


def test_a_replaced_mapping_with_no_later_write_reaches_the_graph():
    """The replacement alone is a write of the keys it changes, including
    one never written before -- the replaced mapping's own write counts
    start again at zero, so the replacement itself counts them
    (``_ParamsDict._succeed``)."""
    gm = _spring()
    node = gm.get_node("s")
    node.params = {**node.params, "damping": 7.0}
    assert float(gm.params["nodes"]["s"]["damping"]) == 7.0
    gm.compile()
    assert float(gm.params["nodes"]["s"]["damping"]) == 7.0
    gm2 = _counted()
    gm2.get_node("c").params = {"count": 4, "gain": 1.0}     # a structural value
    assert _increment(gm2) == 4.0


class _VecDecay(SimulationNode):
    """``x' = -rates * x`` with ``rates`` a list of numbers, a parameter leaf;
    ``shape`` a nested dict, structural (read when the step is traced)."""

    def initial_state(self):
        return {"x": jnp.ones(2, dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        gain = float(self.params["shape"]["gain"])
        return {"x": state["x"] - dt * gain * jnp.asarray(p["rates"]) * state["x"]}


def _vec(rates):
    gm = GraphManager()
    gm.add_node(_VecDecay("v", 0.1, rates=rates, shape={"gain": 1.0, "tag": float("nan")}))
    gm.compile()
    return gm


@pytest.mark.parametrize("spelling", ["list", "ndarray"])
def test_an_in_place_write_into_a_parameter_reaches_every_reader(spelling):
    """``node.params["rates"][0] = 5.0`` goes through no method of the
    mapping.  It used to be lost to ``gm.params``, every run, ``compile`` and
    ``to_dict`` (case B of the reproducer); the graph now compares each list
    and array with its copy from the last sync."""
    import numpy as np

    rates = [1.0, 2.0] if spelling == "list" else np.asarray([1.0, 2.0], np.float32)
    gm = _vec(rates)
    gm.get_node("v").params["rates"][0] = 5.0
    assert gm.params["nodes"]["v"]["rates"].tolist() == [5.0, 2.0]
    gm.step()
    assert gm.get_node_state("v")["x"].tolist() == pytest.approx([0.5, 0.8])
    gm.compile()
    assert gm.params["nodes"]["v"]["rates"].tolist() == [5.0, 2.0]
    assert list(_saved_value(gm, "v", "rates")) == [5.0, 2.0]


def test_an_in_place_write_into_a_structural_value_recompiles():
    gm = _vec([1.0, 1.0])
    generation = gm._compile_generation  # noqa: SLF001
    gm.get_node("v").params["shape"]["gain"] = 3.0
    gm.step()
    assert gm._compile_generation > generation  # noqa: SLF001
    assert gm.get_node_state("v")["x"].tolist() == pytest.approx([0.7, 0.7])
    taken = gm._compile_generation  # noqa: SLF001
    gm.step()
    gm.step()
    assert gm._compile_generation == taken  # noqa: SLF001 - taken in once, not on every sync


def test_an_unchanged_mutable_value_is_not_a_write():
    """A list or dict holding ``NaN`` compares equal to its own copy: a
    comparison by ``==`` would read it as written on every sync and
    recompile the graph on every step."""
    gm = _vec([1.0, float("nan")])
    gm.step()
    generation = gm._compile_generation  # noqa: SLF001
    for _ in range(3):
        gm.step()
        _ = gm.params
    assert gm._compile_generation == generation  # noqa: SLF001
    assert not gm._dirty  # noqa: SLF001


def test_values_compare_bit_for_bit():
    import numpy as np

    from maddening.core.node import _mutable_snapshot, _mutated_keys, _same_value

    assert _same_value([float("nan"), 1.0], [float("nan"), 1.0])
    assert not _same_value([0.0], [-0.0])
    assert not _same_value([1.0], (1.0,))
    a = np.asarray([1.0, np.nan], np.float32)
    assert _same_value(a, a.copy())
    assert not _same_value(a, a.astype(np.float64))
    params = _ParamsDict(k=1.0, xs=[1.0, 2.0], t=(1.0, 2.0), nested=({"a": [1]},))
    snap = _mutable_snapshot(params)
    assert set(snap) == {"xs", "nested"}                    # floats and tuples of numbers are not copied
    params["xs"][1] = 3.0
    params["nested"][0]["a"].append(2)
    assert _mutated_keys(params, snap) == {"xs", "nested"}


def test_an_in_place_write_is_taken_in_once_and_a_later_calibration_wins():
    """After the sync takes an in-place write in, its copy is renewed: a
    ``gm.params`` write made afterwards is later and stays, rather than being
    overwritten at every later read by a write already taken in."""
    gm = _vec([1.0, 2.0])
    gm.get_node("v").params["rates"][0] = 5.0
    assert gm.params["nodes"]["v"]["rates"].tolist() == [5.0, 2.0]
    gm.params["nodes"]["v"]["rates"] = jnp.asarray([7.0, 2.0], jnp.float32)
    assert gm.params["nodes"]["v"]["rates"].tolist() == [7.0, 2.0]
    gm.step()
    assert gm.params["nodes"]["v"]["rates"].tolist() == [7.0, 2.0]


# ---------------------------------------------------------------------------
# A replaced mapping writes the keys it changes, and no others (MADD-ANO-179)
# ---------------------------------------------------------------------------


def _c(gm):
    return float(gm.params["nodes"]["s"]["damping"])


def test_replacing_the_mapping_leaves_the_other_calibrated_constants_alone():
    """``node.params = {**node.params, "damping": 5.0}`` names one key.  Read
    as a write of every key, it put the node's own stiffness (30) back over
    the 45 a fit had calibrated in ``gm.params``, with no word
    (audit_040_p4_12/fmu-sysid/repro_F6_low_wording_items.py); the keyed
    write ``node.params["damping"] = 5.0`` never did."""
    for write in (lambda node: node.params.__setitem__("damping", 5.0),
                  lambda node: setattr(node, "params", {**node.params, "damping": 5.0})):
        gm = _spring()
        gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(45.0, jnp.float32)   # the calibration
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            write(gm.get_node("s"))
            assert (_k(gm), _c(gm)) == (45.0, 5.0)
            assert _k_stepped(gm) == pytest.approx(45.0, rel=1e-3)
            gm.compile()
        assert (_k(gm), _c(gm)) == (45.0, 5.0)
        assert _saved_value(gm, "s", "stiffness") == 45.0


def test_a_replacement_that_hands_a_key_back_unchanged_is_not_a_write_of_it():
    """The rule, both ways: handed the value the node already held, a key is
    not written and its calibration stays; the keyed write of that same
    value is a write and reverts it (``test_writing_the_nodes_own_value_back_
    reverts_a_calibration``)."""
    gm = _spring()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(42.0, jnp.float32)
    node = gm.get_node("s")
    node.params = {**node.params, "stiffness": 30.0}      # the node's own value, handed back
    assert _k(gm) == 42.0
    node.params = dict(node.params)                       # a copy changes nothing
    assert _k(gm) == 42.0 and node.params._write_counts() == {}
    node.params["stiffness"] = 30.0                       # the keyed write is a write
    assert _k(gm) == 30.0


def test_a_replacement_carries_the_writes_the_old_mapping_had_pending():
    """A keyed write the graph has not yet taken, then the mapping replaced:
    the write is not lost with the mapping that counted it -- including a
    write of the value the node already held, which no comparison of the two
    mappings can see."""
    gm = _spring()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(42.0, jnp.float32)
    gm.params["nodes"]["s"]["damping"] = jnp.asarray(9.0, jnp.float32)
    node = gm.get_node("s")
    node.params["stiffness"] = 30.0                       # pending: reverts the calibration
    node.params = {**node.params, "mass": 2.5}            # ... and the mapping goes
    node.params = {**node.params, "rest_length": 1.25}    # ... twice
    assert node.params._writes == 0                       # its own counts start at zero
    assert node.params._write_counts() == {"stiffness": 1, "mass": 1, "rest_length": 1}
    leaves = gm.params["nodes"]["s"]
    assert (float(leaves["stiffness"]), float(leaves["mass"]), float(leaves["rest_length"])) \
        == (30.0, 2.5, 1.25)
    assert float(leaves["damping"]) == 9.0                # handed back twice: still calibrated


@pytest.mark.parametrize("spelling", ["list", "ndarray"])
def test_a_replacement_and_an_in_place_write_do_not_hide_each_other(spelling):
    """A list value is shared by ``{**node.params}``: an element written in
    place before or after the replacement is a write of its key; a new list
    of the same numbers is not, and one of other numbers is."""
    import numpy as np

    def rates(*values):
        return list(values) if spelling == "list" else np.asarray(values, np.float32)

    gm = _vec(rates(1.0, 2.0))
    node = gm.get_node("v")
    node.params["rates"][0] = 5.0                         # in place, pending
    node.params = {**node.params}                         # replaced: same list object
    assert gm.params["nodes"]["v"]["rates"].tolist() == [5.0, 2.0]
    gm.params["nodes"]["v"]["rates"] = jnp.asarray([7.0, 2.0], jnp.float32)   # a calibration
    node.params = {**node.params, "rates": rates(5.0, 2.0)}                  # equal: no write
    assert gm.params["nodes"]["v"]["rates"].tolist() == [7.0, 2.0]
    node.params = {**node.params, "rates": rates(5.0, 3.0)}                  # changed: a write
    assert gm.params["nodes"]["v"]["rates"].tolist() == [5.0, 3.0]
    node.params = {**node.params}
    node.params["rates"][1] = 4.0                         # in place, after the replacement
    assert gm.params["nodes"]["v"]["rates"].tolist() == [5.0, 4.0]


def test_a_counting_mapping_from_elsewhere_writes_the_keys_it_changes():
    """A ``_ParamsDict`` assigned as it is (another node's, a sharded
    wrapper's) has counts that say nothing about this graph: its values do."""
    gm = _spring()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(42.0, jnp.float32)
    node = gm.get_node("s")
    node.params["mass"] = 1.75                            # pending on the old mapping
    foreign = _ParamsDict({**node.params, "damping": 6.0})
    foreign["damping"] = 6.0                              # counted before it arrives
    node.params = foreign
    assert node.params is foreign
    leaves = gm.params["nodes"]["s"]
    assert (float(leaves["stiffness"]), float(leaves["damping"]), float(leaves["mass"])) \
        == (42.0, 6.0, 1.75)
    foreign["stiffness"] = 30.0                           # and it counts from here on
    assert _k(gm) == 30.0


_ORACLE_KEYS = ("stiffness", "damping", "mass")
#: Three values a key, so that a write of the value already held is common.
_ORACLE_VALUES = {"stiffness": (30.0, 40.0, 50.0), "damping": (2.0, 3.0, 4.0),
                  "mass": (1.5, 2.5, 3.5)}


class _LaterWriteWins:
    """The reference model of "the later of a node.params write and a
    gm.params write wins": what the node holds, what the graph holds, and
    the keys the node has written since the graph last looked."""

    def __init__(self):
        self.node = {"stiffness": 30.0, "damping": 2.0, "mass": 1.5}
        self.live = dict(self.node)
        self.pending: set = set()

    def sync(self):
        for key in self.pending:
            self.live[key] = self.node[key]
        self.pending.clear()

    def gm_write(self, key, value):          # reads gm.params first
        self.sync()
        self.live[key] = value

    def node_write(self, key, value):        # a write, whatever the value
        self.node[key] = value
        self.pending.add(key)

    def replace(self, mapping):              # writes the keys it changes
        self.pending |= {k for k in mapping if mapping[k] != self.node[k]}
        self.node = dict(mapping)


def test_the_later_write_wins_over_generated_sequences_of_writes():
    """400 generated sequences of keyed node writes, mapping replacements
    (one key changed, several, none), ``gm.params`` leaf writes, whole-tree
    assignments and reads, held to the reference model after every read:
    whichever of a ``node.params`` write and a ``gm.params`` write came
    later is the value the graph holds, a replacement being a write of the
    keys it changed and of no others."""
    import numpy as np

    rng = np.random.default_rng(2026)
    gm = _spring()
    node = gm.get_node("s")
    replacements = 0
    for sequence in range(400):
        model = _LaterWriteWins()
        # Every sequence starts from the same state, by writes the graph takes.
        for key, value in model.node.items():
            node.params[key] = value
        for key in _ORACLE_KEYS:
            gm.params["nodes"]["s"][key] = jnp.asarray(model.node[key], jnp.float32)
        trace = []
        for _ in range(int(rng.integers(3, 10))):
            op = rng.choice(["node", "replace", "replace-many", "gm", "gm-all", "read"])
            key = str(rng.choice(_ORACLE_KEYS))
            value = float(rng.choice(_ORACLE_VALUES[key]))
            trace.append((str(op), key, value))
            if op == "node":
                node.params[key] = value
                model.node_write(key, value)
            elif op == "replace":
                node.params = {**node.params, key: value}
                model.replace({**model.node, key: value})
                replacements += 1
            elif op == "replace-many":
                new = {k: float(rng.choice(_ORACLE_VALUES[k])) for k in _ORACLE_KEYS}
                trace[-1] = ("replace-many", new)
                node.params = {**node.params, **new}
                model.replace({**model.node, **new})
                replacements += 1
            elif op == "gm":
                gm.params["nodes"]["s"][key] = jnp.asarray(value, jnp.float32)
                model.gm_write(key, value)
            elif op == "gm-all":
                gm.params = jax.tree.map(lambda x: x, gm.params)
                model.sync()
            else:
                model.sync()
                got = {k: float(gm.params["nodes"]["s"][k]) for k in _ORACLE_KEYS}
                assert got == model.live, (sequence, trace)
        model.sync()
        got = {k: float(gm.params["nodes"]["s"][k]) for k in _ORACLE_KEYS}
        assert got == model.live, (sequence, trace)
        assert {k: node.params[k] for k in _ORACLE_KEYS} == model.node, (sequence, trace)
    assert replacements > 300                             # the oracle saw what it is about
    # And the graph runs what it holds, after all of it.
    node.params = {**node.params, "stiffness": 40.0, "damping": 2.0, "mass": 1.5}
    assert _k_stepped(gm) == pytest.approx(40.0, rel=1e-3)
