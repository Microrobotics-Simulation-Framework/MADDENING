"""A graph containing a sharded node, through checkpoint, config and USD.

Checkpointing round-trips correctly and is stated here as a property.
The two serialisation formats do not, and the failures are pinned as
strict xfails rather than fixed: both need a decision about the public
representation of a sharded node, which a test branch must not take.

* **W7 (config)** -- ``ShardedStencilNode.to_dict`` delegates to the
  inner node and appends ``sharded``/``axis_map``/``boundary``, so the
  config records ``"type": "HeatNode"`` and ``from_dict`` rebuilds a
  plain, single-device node.  The reload succeeds and quietly gives back
  a different graph.
* **W7 (USD)** -- the USD writer records the *wrapper's* qualified class
  name and the loader calls it as ``cls(name=..., timestep=...,
  **params)``, which no wrapper accepts.  The save succeeds; the load
  raises ``TypeError``.
* **W6 (boundary inputs)** -- ``boundary_input_spec()`` is proxied
  unconditionally, so a graph may declare, validate and connect an edge
  into a field the sharded ``update_padded`` never reads.  The edge is
  silently discarded and the sharded graph runs different physics.

Each xfail reproduces the audit's own case (``HeatNode`` behind a
``ShardedStencilNode``) so that the day the representation is decided,
the strict marker fails the run and the test becomes the regression
test for the fix.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode
from tests.cloud.multigpu.property_support import (
    WRAPPER_FAMILY,
    device_counts,
    param_values,
    wrapper_labels,
)
from tests.conftest import EXAMPLES_COSTLY

_N_DEVICES = min(4, len(jax.devices()))


def _graph_for(case) -> GraphManager:
    gm = GraphManager()
    gm.add_node(case.wrapped)
    for field, value in case.boundary_inputs.items():
        gm.add_external_input(target_node=case.wrapped.name,
                              target_field=field,
                              shape=tuple(jnp.shape(value)))
    gm.compile()
    return gm


def _external(case):
    return ({case.wrapped.name: dict(case.boundary_inputs)}
            if case.boundary_inputs else None)


def _state_of(gm, name) -> dict:
    return {k: np.asarray(jax.device_get(v))
            for k, v in gm.get_node_state(name).items()}


# ---------------------------------------------------------------------------
# Checkpoints: the round trip that works
# ---------------------------------------------------------------------------


@given(label=wrapper_labels(), n_devices=device_counts(),
       steps=st.integers(min_value=1, max_value=3), rate=param_values())
@settings(max_examples=EXAMPLES_COSTLY)
def test_a_checkpoint_round_trip_restores_a_sharded_graph(
        label, n_devices, steps, rate, tmp_path_factory):
    """State and calibrated constants come back, and the run continues.

    Restoring is not the same as resuming: the property asserts both
    that the restored state matches and that the next step from it
    matches the step the original graph would have taken, which is what
    a preempted cloud run actually depends on.
    """
    case = WRAPPER_FAMILY[label](n_devices=n_devices, rate=0.5)
    gm = _graph_for(case)
    gm.params["nodes"][case.wrapped.name][case.param_name] = jnp.float32(rate)
    for _ in range(steps):
        gm.step(_external(case))
    saved = _state_of(gm, case.wrapped.name)

    path = tmp_path_factory.mktemp("sharded-ckpt") / "state.npz"
    gm.save_state(path)

    restored_case = WRAPPER_FAMILY[label](n_devices=n_devices, rate=0.5)
    restored = _graph_for(restored_case)
    restored.load_state(path)

    for field, value in saved.items():
        np.testing.assert_allclose(
            _state_of(restored, case.wrapped.name)[field], value,
            rtol=1e-6, atol=1e-7, err_msg=field)
    assert float(restored.params["nodes"][case.wrapped.name][case.param_name]) == (
        pytest.approx(rate, rel=1e-6))

    gm.step(_external(case))
    restored.step(_external(restored_case))
    for field, value in _state_of(gm, case.wrapped.name).items():
        np.testing.assert_allclose(
            _state_of(restored, case.wrapped.name)[field], value,
            rtol=1e-5, atol=1e-6, err_msg=f"after resuming: {field}")


# ---------------------------------------------------------------------------
# W7 -- the two serialisation formats
# ---------------------------------------------------------------------------


def _sharded_heat_graph(n_cells: int = 8) -> GraphManager:
    mesh = create_device_mesh(shape=(_N_DEVICES,))
    rod = HeatNode("rod", 0.01, n_cells=n_cells, thermal_diffusivity=0.02,
                   initial_temperature=1.0)
    gm = GraphManager()
    gm.add_node(ShardedStencilNode(rod, mesh, axis_map={"devices": 0},
                                   boundary="edge"))
    gm.add_external_input(target_node="rod", target_field="heat_source",
                          shape=(n_cells,))
    gm.compile()
    return gm


@pytest.mark.skipif(_N_DEVICES < 2, reason="needs >=2 CPU-virtual devices")
@pytest.mark.xfail(strict=True, reason=(
    "whole-tree audit W7: ShardedStencilNode.to_dict() records the INNER "
    "node's type plus a 'sharded' flag, and GraphManager.from_dict ignores "
    "the flag, so a config round trip silently returns a single-device "
    "graph.  Fixing it is a decision about the public config schema (how a "
    "wrapper and its mesh are represented, and what an older loader should "
    "do with it), which this test branch must not take."))
def test_a_config_round_trip_keeps_a_sharded_node_sharded():
    gm = _sharded_heat_graph()
    config = json.loads(json.dumps(gm.to_dict()))
    assert config["nodes"][0]["sharded"] is True     # recorded...
    reloaded = GraphManager.from_dict(config, {"HeatNode": HeatNode})
    assert isinstance(reloaded.get_node("rod"), ShardedStencilNode), (
        "...and dropped on reload: the graph is now single-device")


@pytest.mark.skipif(_N_DEVICES < 2, reason="needs >=2 CPU-virtual devices")
@pytest.mark.xfail(strict=True, reason=(
    "whole-tree audit W7: save_graph_to_usd writes the WRAPPER's qualified "
    "class name and load_graph_from_usd calls it as cls(name=..., "
    "timestep=..., **params), which raises TypeError -- the stage saves "
    "fine and cannot be read back.  The fix is the same schema decision as "
    "the config half."))
def test_a_usd_round_trip_reloads_a_graph_containing_a_sharded_node(tmp_path):
    pytest.importorskip("pxr", reason="USD round trip needs the pxr bindings")
    from pxr import Usd

    from maddening.usd.serialization import (
        load_graph_from_usd,
        save_graph_to_usd,
    )

    gm = _sharded_heat_graph()
    path = tmp_path / "sharded.usda"
    stage = Usd.Stage.CreateNew(str(path))
    save_graph_to_usd(gm, stage)
    stage.Save()
    reloaded = load_graph_from_usd(Usd.Stage.Open(str(path)),
                                   base_dir=str(tmp_path))
    assert isinstance(reloaded.get_node("rod"), ShardedStencilNode)


# ---------------------------------------------------------------------------
# W6 -- an edge into a boundary field the sharded path ignores
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_N_DEVICES < 2, reason="needs >=2 CPU-virtual devices")
@pytest.mark.xfail(strict=True, reason=(
    "whole-tree audit W6: ShardedStencilNode proxies boundary_input_spec() "
    "unconditionally, but HeatNode.update_padded reads only 'heat_source'.  "
    "An edge into 'left_temperature' therefore validates clean, compiles, "
    "and is silently discarded -- the sharded graph runs different physics "
    "from the unsharded one.  Refusing it needs a new node-level "
    "declaration of which boundary inputs the padded path honours (the "
    "audit suggests update_padded_input_fields()), i.e. a public API "
    "addition this test branch must not make."))
def test_an_edge_into_a_boundary_field_the_sharded_path_ignores_is_refused():
    mesh = create_device_mesh(shape=(_N_DEVICES,))
    gm = GraphManager()
    gm.add_node(SpringDamperNode("src", 0.01, stiffness=10.0,
                                 initial_position=3.0))
    rod = HeatNode("rod", 0.01, n_cells=8, thermal_diffusivity=0.02,
                   initial_temperature=1.0)
    gm.add_node(ShardedStencilNode(rod, mesh, axis_map={"devices": 0},
                                   boundary="edge"))
    gm.add_edge(source="src", target="rod", source_field="position",
                target_field="left_temperature")

    issues = [issue for issue in gm.validate() if "left_temperature" in issue]
    try:
        gm.compile()
    except Exception as exc:                 # a fix may refuse it here instead
        refusal = str(exc)
    else:
        refusal = ""
    assert (any(issue.startswith("ERROR") for issue in issues)
            or "left_temperature" in refusal), (
        "an edge into a boundary field the sharded path never reads was "
        f"accepted without a diagnostic (validate said {issues}, compile "
        "said nothing); the graph now runs different physics from the "
        "unsharded one")
