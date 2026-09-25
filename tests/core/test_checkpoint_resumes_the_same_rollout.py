"""A checkpoint restores state and params, and the run continues as if uninterrupted.

``tests/property/test_round_trips.py::test_a_checkpoint_restores_state_and_params_and_continues_the_same_rollout``
states this over generated graphs and is slow-marked.  This is the same
property on one fixed graph, on every push: stop at a step, checkpoint,
load into a freshly built graph, and require the parameter pytree and the
state back bit for bit and the continued trajectory identical to the
uninterrupted one.  A parameter is changed before the checkpoint, so a
restore that kept the fresh graph's own parameters would fail.

(Its companion, a checkpoint beating the config for trained mapping
weights, is checked on every push by
``tests/core/test_mapping_spec_serialisation.py::test_checkpoint_weights_win_over_the_rebuilt_spec``.)
"""

from __future__ import annotations

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode

from tests.property.invariants import assert_params_identical, assert_states_identical


def _build() -> GraphManager:
    gm = GraphManager()
    gm.add_node(BallNode("rod", 0.01, initial_position=0.0, initial_velocity=0.0,
                         elasticity=0.0, gravity=-3.0))
    gm.add_node(BallNode("node_1", 0.01, initial_position=0.0, initial_velocity=0.0,
                         elasticity=0.0, gravity=-2.181640625))
    gm.add_edge("rod", "node_1", "position", "table_position")
    gm.compile()
    return gm


def _states(gm: GraphManager) -> dict:
    return {name: gm.get_node_state(name) for name in gm.node_names}


def test_a_checkpoint_restores_state_and_params_and_continues_the_same_rollout(tmp_path):
    gm = _build()
    gm.params["nodes"]["node_1"]["gravity"] = jnp.asarray(-1.5, jnp.float32)
    gm.run(2)
    path = gm.save_state(tmp_path / "checkpoint.npz")

    resumed = _build()
    assert float(resumed.params["nodes"]["node_1"]["gravity"]) != -1.5, "nothing to restore"
    resumed.load_state(path)
    assert_params_identical(gm.params, resumed.params)
    assert_states_identical(_states(gm), _states(resumed), what="restored state")

    # ``run`` re-enters one compiled step, so the split rollout is exactly
    # the unsplit one (a fused ``run_scan`` of another length need not be;
    # see test_round_trips.py::test_a_split_rollout_is_step_for_step_identical).
    gm.run(3)
    resumed.run(3)
    assert_states_identical(_states(gm), _states(resumed), what="continued trajectory")
