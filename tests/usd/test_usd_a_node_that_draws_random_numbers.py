"""A USD stage carries a noise node's seed, and a loaded graph restarts its stream.

The convention and its other doors are in
``tests/core/test_a_node_that_draws_random_numbers.py``.  A stage, like a
config, holds the node and not its state: the seed is a constructor
parameter (an ``int`` in ``paramsJson``), and the graph loaded from the
stage draws the seed's stream from its first sample.
"""

from __future__ import annotations

import json

from pxr import Usd

from maddening.core.graph_manager import GraphManager
from maddening.usd.serialization import load_graph_from_usd, save_graph_to_usd
from tests.core.test_a_node_that_draws_random_numbers import (
    AMP,
    DT,
    SEED,
    NoisySensor,
    assert_draws,
    stream,
)


def test_a_stage_carries_the_seed_and_a_loaded_graph_restarts_the_stream():
    want, _ = stream(SEED, 4)
    gm = GraphManager()
    gm.add_node(NoisySensor("n", DT, seed=SEED, amplitude=AMP))
    gm.compile()
    gm.run_scan(3)                       # the stage holds the node, not its state
    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    params = json.loads(
        stage.GetPrimAtPath("/Simulation/nodes/n").GetAttribute("maddening:paramsJson").Get())
    assert params["seed"] == SEED and isinstance(params["seed"], int)

    loaded = load_graph_from_usd(stage, node_registry={
        f"{NoisySensor.__module__}.{NoisySensor.__qualname__}": NoisySensor})
    loaded.compile()
    seed = loaded.get_node("n").params["seed"]
    assert seed == SEED and isinstance(seed, int)
    assert sorted(loaded.params["nodes"]["n"]) == ["amplitude", "gain"]
    _, history = loaded.run_scan_with_history(4)
    assert_draws(history["n"]["noise"], want)
