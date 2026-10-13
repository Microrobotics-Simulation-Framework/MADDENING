"""``POST /surrogate/train``'s memory estimate sizes a boundary input as delivered.

The estimate counted each incoming edge at its *source field's* size.
That was the size the dataset held while ``DatasetGenerator`` ignored
interface mappings (MADD-ANO-193); now that the dataset holds what the
edge delivers, a mapped edge is ``mapping.n_target`` entries, and an
estimate left at the source's size would under-count a refining mapping
by ``n_target / n_source``.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="the REST server needs the api extra")

from maddening.api import server as server_module  # noqa: E402
from maddening.api.server import TrainSurrogateRequest  # noqa: E402
from maddening.core.coupling.mapping import matrix_mapping  # noqa: E402
from maddening.core.graph_manager import GraphManager  # noqa: E402
from maddening.nodes import HeatNode  # noqa: E402
from maddening.surrogates.dataset import DatasetGenerator  # noqa: E402

N_SOURCE, N_TARGET = 4, 12


def _rods(mapped: bool) -> GraphManager:
    gm = GraphManager()
    gm.add_node(HeatNode("coarse", 1e-6, n_cells=N_SOURCE))
    gm.add_node(HeatNode("fine", 1e-6, n_cells=N_TARGET))
    if mapped:
        H = np.repeat(np.eye(N_SOURCE, dtype=np.float32), N_TARGET // N_SOURCE, axis=0)
        gm.add_edge("coarse", "fine", "temperature", "heat_source", mapping=matrix_mapping(H))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


def test_a_mapped_edge_is_counted_at_the_size_it_delivers():
    request = TrainSurrogateRequest(node_name="fine", n_data_steps=10, hidden_sizes=[8],
                                    batch_size=1)
    estimate = server_module._surrogate_training_bytes
    without, mapped = estimate(_rods(False), "fine", request), estimate(_rods(True), "fine", request)
    # 16 conditions x 9 samples of the input, held three times; the network's
    # input layer (8 wide) and one batch's activations grow by it too: float32.
    per_input_value = 4 * (3 * 16 * 9 + 12 * 8 + 3)
    assert mapped - without == N_TARGET * per_input_value

    # ... which is the size the dataset holds for it.
    ds = DatasetGenerator.from_graph(_rods(True), "fine", 10)
    assert ds.boundary_inputs["heat_source"].shape == (9, N_TARGET)
