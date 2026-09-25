"""``PUT /graph/params`` refuses a value the running node cannot use.

The graph refuses a ``gm.params`` write to a leaf its step cannot read, by
comparing against the node's own value.  The REST route writes both
``gm.params`` and ``node.params``, so that comparison cannot see it -- and
writing ``node.params`` rebuilds nothing the node derived from the value
when it was constructed.  ``WaveletAdaptiveNode.mass`` (baked into the
operator, declared in ``static_data_deps``) and ``LBMPipeNode.pipe_radius``
(baked into the wall mask) were answered 200, served by ``GET``, ignored by
every step even after ``POST /graph/compile``, and saved by ``to_dict()``
and ``save_state()``: the reloaded graph ran a different model (a factor of
two in the wavelet objective).  These tests drive the real server and pin
the refusal, that nothing is written, that ``to_dict()`` still reproduces
the running graph, and that the writes which do take effect still do.

Originally written from the confirmation audit of the 0.4.0 tree
(reproducers under
``benchmarks/results/audit_040_phase3_confirm/params-fim-nodes/``).
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager, _node_with_params
from maddening.core.node import SimulationNode
from maddening.nodes.adaptive.wavelet import WaveletAdaptiveNode
from maddening.nodes.ball import BallNode
from maddening.nodes.lbm_pipe import LBMPipeNode

REGISTRY = {
    "WaveletAdaptiveNode": WaveletAdaptiveNode,
    "LBMPipeNode": LBMPipeNode,
    "BallNode": BallNode,
}

PIPE = dict(nx=8, ny=10, nz=10, pipe_radius=0.8, propeller_x=2,
            propeller_strength=0.01)


def _client(gm, tmp_path=None):
    server = SimulationServer(
        node_registry=REGISTRY, graph_manager=gm,
        checkpoint_root=str(tmp_path) if tmp_path is not None else None,
    )
    return TestClient(server.create_app(), raise_server_exceptions=False)


def _pipe(compile=True, **overrides):
    gm = GraphManager()
    gm.add_node(LBMPipeNode("p", 1.0, **{**PIPE, **overrides}))
    if compile:
        gm.compile()
    return gm


def _velocity(gm, n=20):
    return np.asarray(gm.run_scan(n)["p"]["velocity"])


def _reloaded(gm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = GraphManager.from_dict(gm.to_dict(), REGISTRY)
        again.compile()
    return again


def _node_params(gm, name):
    return dict(gm._nodes[name].node.params)


def _live(gm, name):
    return {k: np.asarray(v).copy() for k, v in gm.params["nodes"].get(name, {}).items()}


def _assert_nothing_written(gm, name, node_before, live_before, dirty_before):
    assert _node_params(gm, name) == node_before
    live_after = _live(gm, name)
    assert live_after.keys() == live_before.keys()
    for key, value in live_before.items():
        np.testing.assert_array_equal(live_after[key], value)
    assert gm._dirty is dirty_before


# ---------------------------------------------------------------------------
# The two audited nodes
# ---------------------------------------------------------------------------


def test_wavelet_mass_is_refused_and_the_saved_graph_still_matches_the_running_one(tmp_path):
    """``mass`` is baked into the operator (``static_data_deps``)."""
    gm = GraphManager()
    gm.add_node(WaveletAdaptiveNode("w", 1.0, n_levels=3, mass=1.0, blindness_gate=False))
    gm.compile()
    client = _client(gm)
    node_before, live_before = _node_params(gm, "w"), _live(gm, "w")

    resp = client.put("/graph/params/w", json={"params": {"mass": 0.5}})

    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail.startswith("mass:") and "static_data_deps" in detail
    assert "Nothing was written" in detail
    _assert_nothing_written(gm, "w", node_before, live_before, False)
    assert client.get("/graph/params/w").json()["mass"] == pytest.approx(1.0)
    saved = gm.to_dict()
    assert saved["nodes"][0]["params"]["mass"] == pytest.approx(1.0)
    gm.save_state(tmp_path / "ck.npz")
    with np.load(tmp_path / "ck.npz") as ck:
        assert float(ck["_params/w/mass"]) == pytest.approx(1.0)
    # The reloaded graph runs the model the server is running.
    again = _reloaded(gm)
    np.testing.assert_array_equal(
        np.asarray(again.run_scan(2)["w"]["c"]), np.asarray(gm.run_scan(2)["w"]["c"]))


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "before_compile"])
def test_pipe_radius_is_refused_and_the_saved_graph_still_matches_the_running_one(tmp_path, compiled):
    """``pipe_radius`` is baked into the wall mask in ``__init__``.  The
    liveness walk (compiled) and the node's hooks (before the first compile)
    found it while the pipe was full; since the initial fill of a part-full
    pipe reads it too, ``LBMPipeNode`` declares it in ``static_data_deps``."""
    gm = _pipe(compile=compiled)
    client = _client(gm)
    node_before, live_before = _node_params(gm, "p"), _live(gm, "p")
    dirty_before = gm._dirty

    resp = client.put("/graph/params/p", json={"params": {"pipe_radius": 0.5}})

    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail.startswith("pipe_radius:") and "static_data_deps" in detail
    _assert_nothing_written(gm, "p", node_before, live_before, dirty_before)
    assert gm.to_dict()["nodes"][0]["params"]["pipe_radius"] == pytest.approx(0.8)
    np.testing.assert_array_equal(_velocity(_reloaded(gm)), _velocity(gm))
    gm.save_state(tmp_path / "ck.npz")
    with np.load(tmp_path / "ck.npz") as ck:
        assert float(ck["_params/p/pipe_radius"]) == pytest.approx(0.8)


def test_a_refused_key_leaves_the_accepted_keys_of_the_same_request_unwritten():
    gm = _pipe()
    client = _client(gm)
    node_before, live_before = _node_params(gm, "p"), _live(gm, "p")
    resp = client.put("/graph/params/p", json={
        "params": {"propeller_strength": 0.02, "pipe_radius": 0.5}})
    assert resp.status_code == 400
    _assert_nothing_written(gm, "p", node_before, live_before, False)


def test_a_structural_value_consumed_at_construction_is_refused():
    """``propeller_x`` (an int) would go the recompile path, and the
    recompile would trace the mask ``__init__`` built from the old value.
    ``LBMPipeNode`` declares it in ``static_data_deps``, so it is refused
    before any trace; the undeclared form of the same case (the hooks trace
    identically) is ``test_a_structural_value_the_node_copied_in_init_is_refused``."""
    gm = _pipe()
    client = _client(gm)
    node_before, live_before = _node_params(gm, "p"), _live(gm, "p")
    resp = client.put("/graph/params/p", json={"params": {"propeller_x": 5}})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"].startswith("propeller_x:")
    assert "static_data_deps" in resp.json()["detail"]
    _assert_nothing_written(gm, "p", node_before, live_before, False)


@pytest.mark.parametrize("key, value", [("pipe_radius", 0.5), ("propeller_radius", 0.5),
                                        ("propeller_x", 5)])
def test_pipe_geometry_the_initial_fill_also_reads_is_refused_and_the_reload_matches(key, value):
    """A part-full pipe: ``initial_state()`` reads ``pipe_radius`` again (the
    fill mask), which the undeclared rule took for "takes effect at the next
    reset".  ``pipe_radius`` was answered 200; after ``/sim/reset`` the step
    kept the wall mask of the old radius, bit for bit, while ``to_dict()``
    saved the new one, and the reload ran a pipe 5.667e-03 away."""
    gm = _pipe(fill_fraction=0.5)
    client = _client(gm)
    node_before, live_before = _node_params(gm, "p"), _live(gm, "p")

    resp = client.put("/graph/params/p", json={"params": {key: value}})

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"].startswith(f"{key}:")
    assert "static_data_deps" in resp.json()["detail"]
    _assert_nothing_written(gm, "p", node_before, live_before, False)
    assert client.post("/sim/reset").status_code == 200
    np.testing.assert_array_equal(_velocity(_reloaded(gm), n=5), _velocity(gm, n=5))


def test_an_initial_condition_copied_at_construction_is_refused():
    """``initial_rho_liquid`` is copied in ``__init__`` and ``initial_state``
    reads the copy, so a reset would not use the new value either."""
    gm = _pipe(G=-5.0, rho_liquid=2.0, rho_gas=0.5, fill_fraction=0.5)
    client = _client(gm)
    resp = client.put("/graph/params/p", json={"params": {"initial_rho_liquid": 3.0}})
    assert resp.status_code == 400, resp.text
    assert gm._nodes["p"].node.params["initial_rho_liquid"] == 2.0


# ---------------------------------------------------------------------------
# What still takes effect
# ---------------------------------------------------------------------------


def test_a_leaf_the_step_reads_is_written_and_used():
    gm = _pipe()
    client = _client(gm)
    resp = client.put("/graph/params/p", json={"params": {"propeller_strength": 0.03}})
    assert resp.status_code == 200, resp.text
    np.testing.assert_array_equal(_velocity(gm), _velocity(_pipe(propeller_strength=0.03)))


def test_rewriting_the_constructed_value_of_a_baked_leaf_is_accepted():
    gm = _pipe()
    client = _client(gm)
    resp = client.put("/graph/params/p", json={"params": {"pipe_radius": 0.8}})
    assert resp.status_code == 200, resp.text


def test_an_initial_condition_the_node_reads_takes_effect_at_the_next_reset():
    gm = GraphManager()
    gm.add_node(BallNode(name="b", timestep=0.01, initial_position=5.0))
    gm.compile()
    client = _client(gm)
    assert client.put("/graph/params/b", json={
        "params": {"initial_position": 2.5}}).status_code == 200
    reset = client.post("/sim/reset")
    assert reset.json()["state"]["b"]["position"] == pytest.approx(2.5)
    assert gm.to_dict()["nodes"][0]["params"]["initial_position"] == pytest.approx(2.5)


class _Legacy(SimulationNode):
    """3-argument contract: every constant is a structural write."""

    def __init__(self, name="n", timestep=0.1, k=2.0, baked=3.0):
        super().__init__(name, timestep, k=k, baked=baked)
        self._baked = float(baked)

    def initial_state(self):
        return {"x": jnp.asarray(1.0, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        rate = self.params["k"] + self._baked
        return {"x": state["x"] * (1.0 - dt * rate)}


def test_a_structural_value_the_trace_reads_is_written_and_recompiled():
    gm = GraphManager()
    gm.add_node(_Legacy())
    gm.compile()
    client = _client(gm)
    resp = client.put("/graph/params/n", json={"params": {"k": 4.0}})
    assert resp.status_code == 200, resp.text
    assert gm._dirty
    gm.step()
    assert float(gm.get_node_state("n")["x"]) == pytest.approx(1.0 - 0.1 * 7.0)


class _TraceTimeMask(SimulationNode):
    """Builds an array from ``self.params`` while it is traced (a derived
    mask rebuilt for a new scale): the new value changes a *constant* of the
    trace, not its text."""

    def __init__(self, name="n", timestep=0.1, scale=1.0):
        super().__init__(name, timestep, scale=scale)

    def initial_state(self):
        return {"x": jnp.ones(4, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        mask = np.linspace(0.5, 1.5, 4, dtype=np.float32) * np.float32(self.params["scale"])
        return {"x": state["x"] * (1.0 - dt * mask)}


def test_a_structural_value_that_changes_a_trace_constant_is_written_and_recompiled():
    gm = GraphManager()
    gm.add_node(_TraceTimeMask())
    gm.compile()
    resp = _client(gm).put("/graph/params/n", json={"params": {"scale": 2.0}})
    assert resp.status_code == 200, resp.text
    gm.step()
    expected = 1.0 - 0.1 * np.linspace(0.5, 1.5, 4, dtype=np.float32) * 2.0
    np.testing.assert_allclose(np.asarray(gm.get_node_state("n")["x"]), expected, rtol=1e-6)


def test_a_structural_value_the_node_copied_in_init_is_refused():
    gm = GraphManager()
    gm.add_node(_Legacy())
    gm.compile()
    resp = _client(gm).put("/graph/params/n", json={"params": {"baked": 9.0}})
    assert resp.status_code == 400, resp.text
    assert gm._nodes["n"].node.params["baked"] == 3.0 and not gm._dirty


# ---------------------------------------------------------------------------
# The probe copy: faithful where the write lands, never the node itself
# ---------------------------------------------------------------------------


class _SelfBound(_Legacy):
    """Reads ``k`` through a bound method it stored at construction: a
    shallow copy would call the original's method and read the original's
    params, so no faithful copy exists."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._rate = self._rate_impl

    def _rate_impl(self):
        return self.params["k"] + self._baked

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] * (1.0 - dt * self._rate())}


def test_a_node_no_copy_can_be_made_of_is_refused_rather_than_assumed_to_use_the_write():
    """Whether the step reads ``k`` can only be told on a copy that reads
    the new value, and none can be made of this node.  "Cannot tell" used
    to mean "accept": the write was answered 200 whatever the node did with
    it.  It is refused now, saying why, and nothing is written."""
    node = _SelfBound()
    assert _node_with_params(node, {**node.params, "k": 4.0}) is None
    gm = GraphManager()
    gm.add_node(node)
    gm.compile()
    node_before, live_before = _node_params(gm, "n"), _live(gm, "n")
    resp = _client(gm).put("/graph/params/n", json={"params": {"k": 4.0}})
    assert resp.status_code == 400, resp.text
    assert "no copy of _SelfBound that reads the new value can be made" in resp.json()["detail"]
    _assert_nothing_written(gm, "n", node_before, live_before, False)
    gm.step()
    assert float(gm.get_node_state("n")["x"]) == pytest.approx(1.0 - 0.1 * 5.0)


class _Inner(SimulationNode):
    def __init__(self, name="n", timestep=0.1, initial_x=1.0):
        super().__init__(name, timestep, initial_x=initial_x)

    def initial_state(self):
        return {"x": jnp.asarray(self.params["initial_x"], jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] * 0.5}


class _Wrapper(SimulationNode):
    """Shares its wrapped node's params dict, like the sharded wrappers."""

    def __init__(self, inner):
        super().__init__(inner.name, inner.delta_t)
        self._inner = inner
        self.params = inner.params

    def initial_state(self):
        return self._inner.initial_state()

    def update(self, state, boundary_inputs, dt):
        return self._inner.update(state, boundary_inputs, dt)


def test_a_wrapper_sharing_its_inner_params_is_probed_where_the_write_lands():
    """The copy of the wrapper must hold a copy of the inner node reading
    the new value: probing only the outer object would see ``initial_state``
    unchanged and refuse a write that does take effect at the reset."""
    inner = _Inner()
    wrapper = _Wrapper(inner)
    probe = _node_with_params(wrapper, {**wrapper.params, "initial_x": 4.0})
    assert probe is not None and probe._inner is not inner
    assert float(probe.initial_state()["x"]) == 4.0
    assert float(wrapper.initial_state()["x"]) == 1.0      # original untouched
    gm = GraphManager()
    gm.add_node(wrapper)
    gm.compile()
    client = _client(gm)
    assert client.put("/graph/params/n", json={
        "params": {"initial_x": 4.0}}).status_code == 200
    assert client.post("/sim/reset").json()["state"]["n"]["x"] == pytest.approx(4.0)


def test_the_probe_copy_never_mutates_the_node():
    node = LBMPipeNode("p", 1.0, **PIPE)
    attrs = dict(vars(node))
    params = node.params
    snapshot = dict(params)
    probe = _node_with_params(node, {**params, "pipe_radius": 0.5})
    assert probe is not None and probe.params["pipe_radius"] == 0.5
    probe.initial_state()
    assert node.params is params and node.params == snapshot
    assert all(vars(node)[k] is v for k, v in attrs.items())
    assert vars(node).keys() == attrs.keys()


# ---------------------------------------------------------------------------
# POST /checkpoint/load writes gm.params too
# ---------------------------------------------------------------------------


def test_a_checkpoint_carrying_another_constructed_value_is_refused_and_not_loaded(tmp_path):
    """A checkpoint of a pipe built with another radius carries that radius
    in gm.params.  Loaded, the graph refuses the leaf at every step, and the
    API has no reset_params: every /sim/step was a 500."""
    source = _pipe(pipe_radius=0.5)
    source.run_scan(3)
    source.save_state(tmp_path / "other.npz")

    gm = _pipe()
    client = _client(gm, tmp_path)
    state_before = {k: np.asarray(v).copy() for k, v in gm.get_node_state("p").items()}
    live_before = _live(gm, "p")

    resp = client.post("/checkpoint/load", params={"path": "other.npz"})

    assert resp.status_code == 400, resp.text
    assert "pipe_radius" in resp.json()["detail"] and "nothing was loaded" in resp.json()["detail"]
    for key, value in state_before.items():
        np.testing.assert_array_equal(np.asarray(gm.get_node_state("p")[key]), value)
    for key, value in live_before.items():
        np.testing.assert_array_equal(np.asarray(gm.params["nodes"]["p"][key]), value)
    assert client.post("/sim/step").status_code == 200


def test_a_checkpoint_of_the_same_graph_still_loads(tmp_path):
    gm = _pipe()
    client = _client(gm, tmp_path)
    gm.run_scan(3)
    assert client.post("/checkpoint/save", params={"path": "same.npz"}).status_code == 200
    expected = np.asarray(gm.get_node_state("p")["velocity"]).copy()
    gm.run_scan(2)
    assert client.post("/checkpoint/load", params={"path": "same.npz"}).status_code == 200
    np.testing.assert_array_equal(np.asarray(gm.get_node_state("p")["velocity"]), expected)


# ---------------------------------------------------------------------------
# The hooks' answer is cached per compile *and* per node object
# ---------------------------------------------------------------------------


class _Reads(SimulationNode):
    def __init__(self, name="n", timestep=0.1, k=2.0):
        super().__init__(name, timestep, k=k)
        self._k0 = float(k)

    def initial_state(self):
        return {"x": jnp.asarray(1.0, jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = {**self.params, **(params or {})}
        return {"x": state["x"] * (1.0 - dt * p["k"])}


class _Bakes(_Reads):
    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": state["x"] * (1.0 - dt * self._k0)}


def test_a_node_replaced_under_the_same_name_before_compiling_is_asked_again():
    """Before the first compile the generation does not move, so a cache
    keyed on it alone answered for the node that used to hold the name."""
    gm = GraphManager()
    gm.add_node(_Reads())
    client = _client(gm)
    assert client.put("/graph/params/n", json={"params": {"k": 3.0}}).status_code == 200
    gm.remove_node("n")
    gm.add_node(_Bakes())
    resp = client.put("/graph/params/n", json={"params": {"k": 5.0}})
    assert resp.status_code == 400, resp.text
    assert gm._nodes["n"].node.params["k"] == 2.0
