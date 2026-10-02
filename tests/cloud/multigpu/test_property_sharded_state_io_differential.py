"""Differential oracles 1-4 on a sharded node: writes, reloads, checkpoints.

The state-and-I/O harness (``tests/property/test_differential_*.py``) runs on
one device.  This module states the same oracles for a built-in node behind
a sharded wrapper -- a ``HeatNode`` in a :class:`ShardedStencilNode` over two
and four virtual devices -- where the route and the graph reach the
parameters through the wrapper (``_params_holders``) and a config reloads
the node unsharded (MADD-ANO-036).

The reload is therefore compared *re-wrapped*: ``from_dict`` (which must
warn, naming the wrapper), then the ``replace_node`` call its warning names,
then the checkpoint.  That isolates "the write is the reload" from "the
sharded step is the unsharded one", which is the sharding harness's oracle,
not this one, and keeps the comparison exact.

Tolerance: none -- the original and the re-wrapped reload run the same
sharded program.
"""

from __future__ import annotations

import json
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.api.server import SimulationServer
from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.graph_manager import GraphManager
from maddening.nodes import HeatNode
from maddening.surrogates.replace import replace_node

from tests.conftest import EXAMPLES_COSTLY
from tests.property.differential import (
    note,
    assert_nothing_written,
    assert_trees_identical,
    canonical,
    checkpoint_path,
    full_state,
    graph_snapshot,
    no_cloud_launch,
    rollout,
    tmp_dir,
)
from tests.property.node_catalogue import KINDS, floats32, writes

_N_DEVICES = len(jax.devices())
#: Per push on two devices; four in the slow lane (each example compiles a
#: fresh ``shard_map`` twice, ~5 s for twenty on 3 cores).  Each slow case's
#: per-push sibling is the ``n_devices=2`` case of the same test.
DEVICES = tuple(
    pytest.param(n, marks=pytest.mark.slow) if n > 2 else n
    for n in (2, 4) if n <= _N_DEVICES)
N_STEPS = 3
REGISTRY = {"HeatNode": HeatNode}

pytestmark = pytest.mark.skipif(not DEVICES, reason="needs >=2 CPU-virtual devices")


@pytest.fixture(scope="module", autouse=True)
def _no_cloud():
    with no_cloud_launch():
        yield


def _wrap(node: HeatNode, n_devices: int) -> ShardedStencilNode:
    mesh = create_device_mesh(shape=(n_devices,))
    return ShardedStencilNode(node, mesh, axis_map={"devices": 0}, boundary="edge")


def _sharded_rod(kwargs: dict, n_devices: int) -> GraphManager:
    gm = GraphManager()
    gm.add_node(_wrap(HeatNode("rod", 0.01, **kwargs), n_devices))
    gm.add_external_input("rod", "heat_source", shape=(int(kwargs["n_cells"]),))
    gm.compile()
    return gm


def _rewrapped_reload(config: dict, n_devices: int) -> GraphManager:
    """``from_dict`` -- which warns, naming the wrapper -- then the
    ``replace_node`` its warning names."""
    with pytest.warns(UserWarning, match="ShardedStencilNode"):
        gm = GraphManager.from_dict(config, REGISTRY)
    replace_node(gm, "rod", _wrap(gm.get_node("rod"), n_devices))
    gm.compile()
    return gm


@st.composite
def _rod_kwargs(draw, n_devices: int):
    n = n_devices * draw(st.integers(min_value=2, max_value=3))
    length = draw(st.sampled_from([1.0, 2.0]))
    dx = length / n
    return {
        "n_cells": n,
        "length": length,
        "thermal_diffusivity": draw(floats32(1e-3, min(1e-2, 0.1 * dx * dx / 0.01))),
        "initial_temperature": draw(st.lists(floats32(0.0, 5.0), min_size=n, max_size=n)),
    }


def _check(gm: GraphManager, n_devices: int, *, after_write: str) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*live mapping weights.*")
        config = json.loads(json.dumps(gm.to_dict(), allow_nan=True))
    with tmp_dir() as tmp:
        ckpt = gm.save_state(checkpoint_path(tmp))
        reloaded = _rewrapped_reload(config, n_devices)
        reloaded.load_state(ckpt)
    assert_trees_identical(full_state(gm), full_state(reloaded),
                           what=f"{after_write}: restored state")
    assert_trees_identical(rollout(gm, N_STEPS), rollout(reloaded, N_STEPS),
                           what=f"{after_write}: continued")
    gm.reset_state()
    reloaded.reset_state()
    assert_trees_identical(rollout(gm, N_STEPS), rollout(reloaded, N_STEPS),
                           what=f"{after_write}: after a reset")


# Per push: tests/cloud/multigpu/test_property_sharded_state_io_differential.py::test_a_rest_write_into_a_sharded_rod_is_refused_whole_or_runs_as_its_reload[2] (two devices; four in the slow lane).
@pytest.mark.parametrize("n_devices", DEVICES)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_rest_write_into_a_sharded_rod_is_refused_whole_or_runs_as_its_reload(
        n_devices, data):
    kwargs = data.draw(_rod_kwargs(n_devices), label="kwargs")
    write = data.draw(writes(KINDS["heat_uniform"], kwargs), label="write")
    gm = _sharded_rod(kwargs, n_devices)
    gm.run(2)
    with tmp_dir() as root:
        client = TestClient(SimulationServer(node_registry=REGISTRY, graph_manager=gm,
                                             checkpoint_root=root).create_app(),
                            raise_server_exceptions=False)
        before = graph_snapshot(gm)
        get_before = client.get("/graph/params/rod").json()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="overflow encountered in cast",
                                    category=RuntimeWarning)
            resp = client.put("/graph/params/rod", content=write.body(),
                              headers={"content-type": "application/json"})
        note(f"PUT {write.params!r} -> {resp.status_code} {resp.text[:300]}")
        assert resp.status_code < 500, resp.text
        if resp.status_code >= 400:
            assert_nothing_written(gm, before, what="refused sharded write")
            assert canonical(client.get("/graph/params/rod").json()) == canonical(get_before)
            return
        assert canonical(resp.json()["params"]) == canonical(
            client.get("/graph/params/rod").json())
    _check(gm, n_devices, after_write=f"REST {write.params!r}")


# Per push: tests/cloud/multigpu/test_property_sharded_state_io_differential.py::test_a_param_write_into_a_sharded_rod_runs_as_its_reload[2] (two devices; four in the slow lane).
@pytest.mark.parametrize("n_devices", DEVICES)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_param_write_into_a_sharded_rod_runs_as_its_reload(n_devices, data):
    """``gm.params`` through the wrapper: a diffusivity or a length moved
    inside its stability limit, as a calibration would."""
    kwargs = data.draw(_rod_kwargs(n_devices), label="kwargs")
    key = data.draw(st.sampled_from(["thermal_diffusivity", "length"]), label="key")
    factor = data.draw(st.sampled_from([0.5, 0.8, 1.25]), label="factor")
    gm = _sharded_rod(kwargs, n_devices)
    gm.run(2)
    live = gm.params["nodes"]["rod"]
    live[key] = jnp.asarray(np.asarray(live[key]) * np.float32(factor), jnp.float32)
    _check(gm, n_devices, after_write=f"gm.params {key} x{factor}")


# Per push: tests/cloud/multigpu/test_property_sharded_state_io_differential.py::test_a_checkpoint_of_a_sharded_rod_resumes_the_uninterrupted_run[2] (two devices; four in the slow lane).
@pytest.mark.parametrize("n_devices", DEVICES)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_checkpoint_of_a_sharded_rod_resumes_the_uninterrupted_run(n_devices, data):
    kwargs = data.draw(_rod_kwargs(n_devices), label="kwargs")
    split = data.draw(st.integers(min_value=0, max_value=4), label="split")
    original = _sharded_rod(kwargs, n_devices)
    original.run(split)
    with tmp_dir() as tmp:
        path = original.save_state(checkpoint_path(tmp))
        resumed = _sharded_rod(kwargs, n_devices)
        resumed.run(split + 1)
        resumed.load_state(path)
    assert_trees_identical(full_state(original), full_state(resumed), what="restored")
    assert_trees_identical(rollout(original, N_STEPS), rollout(resumed, N_STEPS),
                           what="continued")


# Per push: tests/cloud/multigpu/test_property_sharded_state_io_differential.py::test_a_sharded_rod_config_reloads_unsharded_with_a_warning_and_rewrapped_bit_for_bit[2] (two devices; four in the slow lane).
@pytest.mark.parametrize("n_devices", DEVICES)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_sharded_rod_config_reloads_unsharded_with_a_warning_and_rewrapped_bit_for_bit(
        n_devices, data):
    """MADD-ANO-036: the plain reload warns and is a ``HeatNode``; re-wrapped as
    the warning says, it is the original bit for bit."""
    kwargs = data.draw(_rod_kwargs(n_devices), label="kwargs")
    gm = _sharded_rod(kwargs, n_devices)
    config = json.loads(json.dumps(gm.to_dict(), allow_nan=True))
    with pytest.warns(UserWarning, match="unsharded"):
        plain = GraphManager.from_dict(config, REGISTRY)
    assert type(plain.get_node("rod")) is HeatNode
    reloaded = _rewrapped_reload(config, n_devices)
    assert_trees_identical(full_state(gm), full_state(reloaded), what="initial state")
    assert_trees_identical(rollout(gm, N_STEPS), rollout(reloaded, N_STEPS),
                           what="trajectory")
