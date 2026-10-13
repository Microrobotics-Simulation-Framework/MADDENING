"""A config keeps what a node's ``to_dict`` override adds under ``params``,
for a node whose ``update`` takes ``params`` too.

A class that keeps a constructor argument outside ``params`` saves it by
overriding ``to_dict`` and adding it to the entry's ``params``, as the
node-authoring guide says.  ``GraphManager.to_dict`` writes, for a node
whose ``update`` takes ``params``, the live (calibrated) values of the
node's parameter tree; it used to write them IN PLACE of the override's
``params``, so the added argument was not saved, the reloaded graph ran
the constructor's default, and nothing said so.  Now the keys of
``node.params`` take their live values and the override's other entries
are kept: the reloaded graph has both, and steps bit for bit like the
graph that was saved.

What did not move is pinned too: a node that adds nothing writes exactly
the live values, in the order of ``node.params``; a key of
``node.params`` an override leaves out is written all the same; an
entry that is not a dict is replaced by the live values.
"""
from __future__ import annotations

import contextlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition
from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode, ShardedStencilNode
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.simulation.hybrid_node import HybridNode

DT = 0.01
RATE, CALIBRATED, GAIN = 0.5, 0.37, 7.0


class KeepsGain(SimulationNode):
    """``rate`` is in ``params`` (the node's parameter tree, which a
    calibration writes); ``gain`` is kept as an attribute, and saved by
    the override below."""

    def __init__(self, name="n", timestep=DT, rate=RATE, gain=2.0):
        super().__init__(name, timestep, rate=rate)
        self.gain = gain

    def initial_state(self):
        return {"x": jnp.linspace(1.0, 2.0, 8)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        rate = {**self.params, **(params or {})}["rate"]
        return {"x": state["x"] * (1.0 - dt * rate * self.gain)}

    def update_padded(self, state_padded, boundary_inputs, dt, *, params=None):
        return self.update(state_padded, boundary_inputs, dt, params=params)


class SavesGain(KeepsGain):
    """The guide's way to save it: the base entry, and ``gain`` added to
    its ``params``."""

    def to_dict(self):
        d = super().to_dict()
        d["params"] = {**d["params"], "gain": self.gain}
        return d


class SavesGainStencil(SavesGain):
    def halo_width(self):
        return {0: 1}


REGISTRY = {"SavesGain": SavesGain, "SavesGainStencil": SavesGainStencil}


@contextlib.contextmanager
def _x64(on: bool):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", on)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _calibrated(node, rate=CALIBRATED) -> GraphManager:
    """A graph of ``node`` with ``rate`` written through ``gm.params``."""
    gm = GraphManager()
    gm.add_node(node)
    gm.compile()
    params = gm.params
    leaf = params["nodes"][node.name]["rate"]
    params["nodes"][node.name] = {**params["nodes"][node.name],
                                  "rate": jnp.asarray(rate, jnp.asarray(leaf).dtype)}
    gm.params = params
    return gm


def _run(gm: GraphManager, name="n", steps=4) -> np.ndarray:
    gm.compile()
    for _ in range(steps):
        gm.step()
    return np.asarray(gm.get_node_state(name)["x"])


@pytest.mark.parametrize("x64", [False, True], ids=["float32", "x64"])
def test_a_reloaded_config_has_the_added_argument_and_the_calibrated_value(x64):
    with _x64(x64):
        gm = _calibrated(SavesGain(gain=GAIN))
        config = gm.to_dict()                              # warnings are errors here
        entry = config["nodes"][0]["params"]
        assert entry["gain"] == GAIN                       # the override's entry is kept
        assert entry["rate"] == float(jnp.asarray(CALIBRATED, gm.params["nodes"]["n"]["rate"].dtype))
        reloaded = GraphManager.from_dict(json.loads(json.dumps(config)), REGISTRY)
        assert reloaded.get_node("n").gain == GAIN
        want, got = _run(gm), _run(reloaded)
        assert got.dtype == want.dtype == (np.float64 if x64 else np.float32)
        assert got.tobytes() == want.tobytes()
        # Both are needed for that: the default gain, or the uncalibrated
        # rate, runs another trajectory.
        assert want.tobytes() != _run(_calibrated(SavesGain())).tobytes()
        assert want.tobytes() != _run(_calibrated(SavesGain(gain=GAIN), RATE)).tobytes()


def _wrapped():
    mesh = create_device_mesh(shape=(1,))
    layout = build_unstructured_partition(
        partition_assignment=np.zeros(8, np.int32),
        edges=np.array([[i, (i + 1) % 8] for i in range(8)], np.int32), n_devices=1)
    return {
        "HybridNode": lambda: HybridNode(SavesGain(gain=GAIN), lambda *a: {}),
        "ShardedPointwiseNode": lambda: ShardedPointwiseNode(SavesGain(gain=GAIN), mesh),
        "ShardedStencilNode": lambda: ShardedStencilNode(
            SavesGainStencil(gain=GAIN), mesh, {"devices": 0}),
        "ShardedUnstructuredNode": lambda: ShardedUnstructuredNode(
            SavesGain(gain=GAIN), mesh, layout),
    }


@pytest.mark.parametrize("kind", sorted(_wrapped()))
def test_behind_each_wrapper_the_entry_has_both(kind):
    """A wrapper's entry is the node's own (with the wrapper's keys): the
    same merge applies, and a sharded entry reloads unsharded with both."""
    gm = _calibrated(_wrapped()[kind]())
    entry = gm.to_dict()["nodes"][0]
    rate = float(jnp.asarray(CALIBRATED, gm.params["nodes"]["n"]["rate"].dtype))
    assert entry["params"] == {"rate": rate, "gain": GAIN}
    if kind == "HybridNode":
        return                                             # from_dict cannot rebuild it
    with pytest.warns(UserWarning, match="unsharded"):
        reloaded = GraphManager.from_dict(json.loads(json.dumps(gm.to_dict())), REGISTRY)
    node = reloaded.get_node("n")
    assert (node.gain, node.params["rate"]) == (GAIN, rate)
    plain = _calibrated(SavesGain(gain=GAIN))
    assert _run(reloaded).tobytes() == _run(plain).tobytes()


class SaysSo(KeepsGain):
    """Saves ``gain`` another way, says so, and adds nothing."""

    def to_dict(self):
        return super().to_dict()


class Drops(KeepsGain):
    def to_dict(self):
        return {**super().to_dict(), "params": {}}


class NotADict(KeepsGain):
    def to_dict(self):
        return {**super().to_dict(), "params": None}


class Reorders(KeepsGain):
    def to_dict(self):
        d = super().to_dict()
        d["params"] = dict(reversed(list({**d["params"], "gain": self.gain}.items())))
        return d


@pytest.mark.parametrize("node", [KeepsGain(), SaysSo(gain=GAIN), Drops(gain=GAIN),
                                  NotADict(gain=GAIN)],
                         ids=["no override", "adds nothing", "drops rate", "not a dict"])
def test_an_entry_that_adds_nothing_is_the_live_values_as_before(node):
    """Unchanged by the merge (and so passing before it too): a node that
    adds nothing to ``params`` writes the live values exactly, in the order
    of ``node.params``; a key the override leaves out is written all the
    same; an entry that is not a dict is replaced by the live values."""
    gm = _calibrated(node)
    entry = gm.to_dict()["nodes"][0]["params"]
    live = gm.effective_node_params("n")
    assert list(entry.items()) == list(live.items())
    assert list(entry) == ["rate"] and entry["rate"] != RATE


def test_an_overrides_other_keys_follow_the_live_ones():
    """The keys of ``node.params`` first, in its order and at their live
    values, whatever order the override wrote; its other keys after them."""
    gm = _calibrated(Reorders(gain=GAIN))
    entry = gm.to_dict()["nodes"][0]["params"]
    assert list(entry.items()) == [*gm.effective_node_params("n").items(), ("gain", GAIN)]
