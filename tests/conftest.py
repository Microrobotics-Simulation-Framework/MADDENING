"""Shared fixtures for MADDENING tests."""

import os
# Force CPU backend for tests -- jaxlib 0.5.1 has a segfault in the CUDA XLA
# compiler triggered by matmul/einsum on 3D+ arrays.  Must be set before JAX
# is imported.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


@pytest.fixture
def ball_node():
    """A ball starting at height 5 with zero velocity."""
    return BallNode(name="ball", timestep=0.01, initial_position=5.0,
                    initial_velocity=0.0, elasticity=0.7)


@pytest.fixture
def table_node():
    """A table at height 0."""
    return TableNode(name="table", timestep=0.01, position=0.0)


@pytest.fixture
def bouncing_ball_graph(ball_node, table_node):
    """A compiled bouncing-ball graph (table -> ball)."""
    gm = GraphManager()
    gm.add_node(table_node)
    gm.add_node(ball_node)
    gm.add_edge(source="table", target="ball",
                source_field="position", target_field="table_position")
    gm.compile()
    return gm
