"""The guides show the lines ``validate()`` gives for an edge from a node to itself.

``docs/user_guide/quickstart.md`` and
``docs/developer_guide/coupling_algorithm_guide.md`` print, whole, the
``INFO:`` line ``GraphManager.validate()`` returns for an edge from a node
to itself: outside a coupling group, in one, and for a boundary flux,
which cannot be read outside one (MADD-ANO-157).  That line is how a user
learns which of its own values the node reads, so a page that shows
another one misleads.  The docs-snippet gate runs the pages' Python blocks
and does not read their ``text`` blocks.  Here each line is asked of the
library, for the graph the page describes, and must be on the page; a
line break on the page is taken as the space it stands for.

It is in ``tests/compliance`` because a pull request that changes only
the guides runs no other tests.  What the lines say, and that the graph
does what they say, is ``tests/core/test_an_edge_from_a_node_to_itself.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.nodes.heat import HeatNode

REPO_ROOT = Path(__file__).resolve().parents[2]


class _A(SimulationNode):
    """The guide's ``a``: a state ``x``, an input ``u`` and a boundary flux ``q``."""

    def initial_state(self):
        return {"x": jnp.ones(())}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), default=jnp.zeros(()))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": state["x"] + dt * boundary_inputs.get("u", 0.0)}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        return {"q": 2.0 * state["x"]}


def _rod(grouped):
    """The quickstart's graph: a rod whose temperature feeds its own heat source."""
    gm = GraphManager()
    gm.add_node(HeatNode("rod", 0.01, n_cells=4, thermal_diffusivity=0.1))
    gm.add_edge("rod", "rod", "temperature", "heat_source", transform=lambda T: -T)
    if grouped:
        gm.add_coupling_group(["rod"], max_iterations=50)
    return gm


def _a(field, grouped):
    """The coupling guide's ``add_edge("a", "a", ...)``."""
    gm = GraphManager()
    gm.add_node(_A("a", 0.01))
    gm.add_edge("a", "a", field, "u")
    if grouped:
        gm.add_coupling_group(["a"])
    return gm


@pytest.mark.parametrize("page, graphs", [
    ("docs/user_guide/quickstart.md", [lambda: _rod(False), lambda: _rod(True)]),
    ("docs/developer_guide/coupling_algorithm_guide.md",
     [lambda: _a("x", False), lambda: _a("x", True), lambda: _a("q", False)]),
], ids=["quickstart", "coupling guide"])
def test_the_page_shows_each_line_validate_gives_for_the_graph_it_describes(page, graphs):
    text = " ".join((REPO_ROOT / page).read_text(encoding="utf-8").split())
    lines = []
    for build in graphs:
        (line,) = build().validate()
        assert line.startswith("INFO: edge ") and " to itself. " in line, line
        assert line in text, f"{page} does not show: {line}"
        lines.append(line)
    assert len(set(lines)) == len(graphs), lines      # each graph has its own line
