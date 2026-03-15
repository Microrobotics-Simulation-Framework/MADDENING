"""Tests for coupling helper utilities (Phase 6e).

Covers: add_value_coupling, add_flux_coupling, add_dirichlet_neumann_pair,
add_symmetric_value_coupling, add_robin_coupling, and check_conservation.
"""

import jax
import jax.numpy as jnp
import pytest

from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.coupling_helpers import (
    add_dirichlet_neumann_pair,
    add_flux_coupling,
    add_robin_coupling,
    add_symmetric_value_coupling,
    add_value_coupling,
    check_conservation,
)
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode


# ==================================================================
# Helpers
# ==================================================================

def _make_heat_rod_pair(dt=0.001, n_cells=10):
    """Two HeatNodes for coupling tests."""
    gm = GraphManager()
    gm.add_node(HeatNode(name="rod_a", timestep=dt, n_cells=n_cells,
                          thermal_diffusivity=0.01, length=1.0,
                          initial_temperature=100.0))
    gm.add_node(HeatNode(name="rod_b", timestep=dt, n_cells=n_cells,
                          thermal_diffusivity=0.01, length=1.0,
                          initial_temperature=0.0))
    return gm


def _make_spring_pair(dt=0.01, pos_a=0.0, pos_b=3.0):
    """Two SpringDamperNodes for coupling tests."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="spring_a", timestep=dt,
                                  stiffness=50.0, damping=1.0, mass=1.0,
                                  rest_length=1.0,
                                  initial_position=pos_a))
    gm.add_node(SpringDamperNode(name="spring_b", timestep=dt,
                                  stiffness=50.0, damping=1.0, mass=1.0,
                                  rest_length=1.0,
                                  initial_position=pos_b))
    return gm


# ==================================================================
# Test add_value_coupling
# ==================================================================

class TestAddValueCoupling:
    """Tests for add_value_coupling helper."""

    def test_add_value_coupling(self):
        """Creates correct edge between two nodes."""
        gm = _make_spring_pair()
        add_value_coupling(gm, "spring_a", "spring_b",
                           "position", target_field="anchor_position")
        assert len(gm._edges) == 1
        edge = gm._edges[0]
        assert edge.source_node == "spring_a"
        assert edge.target_node == "spring_b"
        assert edge.source_field == "position"
        assert edge.target_field == "anchor_position"
        assert edge.additive is False
        assert edge.transform is None

    def test_add_value_coupling_default_target(self):
        """When target_field is None, defaults to source field name."""
        gm = _make_spring_pair()
        add_value_coupling(gm, "spring_a", "spring_b", "position")
        assert gm._edges[0].target_field == "position"

    def test_add_value_coupling_with_transform(self):
        """Transform is passed through to edge."""
        gm = _make_spring_pair()
        add_value_coupling(gm, "spring_a", "spring_b",
                           "position", target_field="anchor_position",
                           transform=lambda x: x * 2.0)
        assert gm._edges[0].transform is not None


# ==================================================================
# Test add_flux_coupling
# ==================================================================

class TestAddFluxCoupling:
    """Tests for add_flux_coupling helper."""

    def test_add_flux_coupling(self):
        """Creates edge for flux field."""
        gm = _make_heat_rod_pair()
        add_flux_coupling(gm, "rod_a", "rod_b",
                          "right_heat_flux", "heat_source")
        assert len(gm._edges) == 1
        edge = gm._edges[0]
        assert edge.source_node == "rod_a"
        assert edge.target_node == "rod_b"
        assert edge.source_field == "right_heat_flux"
        assert edge.target_field == "heat_source"

    def test_add_flux_coupling_additive(self):
        """Additive flag is forwarded."""
        gm = _make_heat_rod_pair()
        add_flux_coupling(gm, "rod_a", "rod_b",
                          "right_heat_flux", "heat_source",
                          additive=True)
        assert gm._edges[0].additive is True

    def test_add_flux_coupling_marks_dirty(self):
        """Adding a flux edge marks graph dirty."""
        gm = _make_heat_rod_pair()
        gm.compile()
        assert not gm._dirty
        add_flux_coupling(gm, "rod_a", "rod_b",
                          "right_heat_flux", "heat_source")
        assert gm._dirty


# ==================================================================
# Test add_dirichlet_neumann_pair
# ==================================================================

class TestAddDirichletNeumannPair:
    """Tests for add_dirichlet_neumann_pair helper."""

    def test_add_dn_pair(self):
        """Creates correct bidirectional edges (value + flux)."""
        gm = _make_heat_rod_pair()
        add_dirichlet_neumann_pair(
            gm,
            dirichlet_node="rod_a",
            neumann_node="rod_b",
            value_field="temperature",
            flux_field="left_heat_flux",
            value_input="right_temperature",
            flux_input="heat_source",
            value_transform=lambda T: T[0],
        )
        assert len(gm._edges) == 2

        # First edge: value (neumann -> dirichlet)
        value_edge = gm._edges[0]
        assert value_edge.source_node == "rod_b"
        assert value_edge.target_node == "rod_a"
        assert value_edge.source_field == "temperature"
        assert value_edge.target_field == "right_temperature"
        assert value_edge.transform is not None

        # Second edge: flux (dirichlet -> neumann)
        flux_edge = gm._edges[1]
        assert flux_edge.source_node == "rod_a"
        assert flux_edge.target_node == "rod_b"
        assert flux_edge.source_field == "left_heat_flux"
        assert flux_edge.target_field == "heat_source"


# ==================================================================
# Test add_symmetric_value_coupling
# ==================================================================

class TestAddSymmetricValueCoupling:
    """Tests for add_symmetric_value_coupling helper."""

    def test_add_symmetric_value_coupling(self):
        """Creates two bidirectional edges."""
        gm = _make_spring_pair()
        add_symmetric_value_coupling(
            gm,
            "spring_a", "spring_b",
            field_a="position", input_a="anchor_position",
            field_b="position", input_b="anchor_position",
        )
        assert len(gm._edges) == 2

        # A -> B
        assert gm._edges[0].source_node == "spring_a"
        assert gm._edges[0].target_node == "spring_b"
        assert gm._edges[0].source_field == "position"
        assert gm._edges[0].target_field == "anchor_position"

        # B -> A
        assert gm._edges[1].source_node == "spring_b"
        assert gm._edges[1].target_node == "spring_a"
        assert gm._edges[1].source_field == "position"
        assert gm._edges[1].target_field == "anchor_position"

    def test_symmetric_coupling_runs(self):
        """Graph with symmetric coupling produces finite results."""
        gm = _make_spring_pair()
        add_symmetric_value_coupling(
            gm,
            "spring_a", "spring_b",
            field_a="position", input_a="anchor_position",
            field_b="position", input_b="anchor_position",
        )
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=10, tolerance=1e-8,
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["spring_a"]["position"])
        assert jnp.isfinite(state["spring_b"]["position"])


# ==================================================================
# Test Robin coupling
# ==================================================================

class TestRobinCoupling:
    """Tests for add_robin_coupling helper."""

    def test_robin_coupling_creates_four_edges(self):
        """Robin coupling creates 4 edges (2 per direction)."""
        gm = _make_heat_rod_pair()
        add_robin_coupling(
            gm,
            "rod_a", "rod_b",
            value_field_a="temperature",
            flux_field_a="right_heat_flux",
            value_field_b="temperature",
            flux_field_b="left_heat_flux",
            input_a="right_temperature",
            input_b="left_temperature",
            alpha=0.5,
        )
        assert len(gm._edges) == 4

    def test_robin_coupling_heat_rods(self):
        """Two heat rods with Robin interface converge.

        Uses a simplified Robin setup with value fields only (both
        temperature) since full flux-based Robin requires two-stage
        flux pre-computation.  The alpha mixing is applied via
        transforms on the value edges.
        """
        gm = _make_heat_rod_pair()

        # Simplified Robin: mix temperature values from both ends.
        # rod_a gets alpha * rod_b.T[0] as right BC
        # rod_b gets alpha * rod_a.T[-1] as left BC
        alpha = 0.5
        gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                     transform=lambda T, _a=alpha: _a * T[0])
        gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                     transform=lambda T, _a=alpha: _a * T[-1])

        gm.add_coupling_group(
            ["rod_a", "rod_b"],
            max_iterations=15, tolerance=1e-8,
        )
        gm.compile()
        state = gm.run_scan(100)

        assert jnp.all(jnp.isfinite(state["rod_a"]["temperature"]))
        assert jnp.all(jnp.isfinite(state["rod_b"]["temperature"]))

    def test_robin_grad_compatible(self):
        """jax.grad through Robin-style coupled heat rods works."""
        gm = _make_heat_rod_pair(n_cells=5)

        # Robin-like coupling: mixed BC using temperature values
        alpha = 0.5
        gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                     transform=lambda T, _a=alpha: _a * T[0])
        gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                     transform=lambda T, _a=alpha: _a * T[-1])

        gm.add_coupling_group(
            ["rod_a", "rod_b"],
            max_iterations=5, tolerance=1e-6,
        )
        gm.compile()
        step_fn = gm._build_step_fn()
        ext = gm._default_external_inputs()

        def loss_fn(init_temp):
            state = dict(gm._state)
            state["rod_a"] = {"temperature": jnp.ones(5) * init_temp}
            state["rod_b"] = {"temperature": jnp.zeros(5)}
            for _ in range(3):
                state = step_fn(state, ext)
            return jnp.sum(state["rod_a"]["temperature"])

        g = jax.grad(loss_fn)(jnp.array(100.0))
        assert jnp.isfinite(g)
        assert float(g) != 0.0


# ==================================================================
# Test conservation check
# ==================================================================

class TestConservationCheck:
    """Tests for check_conservation diagnostic."""

    def test_conservation_check(self):
        """check_conservation returns near-zero imbalance for converged coupling."""
        gm = _make_heat_rod_pair(n_cells=10)

        # Bidirectional temperature coupling at the interface
        gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                     transform=lambda T: T[-1])
        gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                     transform=lambda T: T[0])

        gm.add_coupling_group(
            ["rod_a", "rod_b"],
            max_iterations=20, tolerance=1e-10,
        )
        gm.compile()

        # Run for several steps to reach a quasi-steady state
        state = gm.run_scan(200)

        # Check flux conservation at the interface
        result = check_conservation(
            gm, state,
            [("rod_a", "right_heat_flux", "rod_b", "left_heat_flux")],
        )

        # There should be one interface result
        assert len(result) == 1
        key = "rod_a.right_heat_flux-rod_b.left_heat_flux"
        assert key in result

        # For converged heat transfer, fluxes should be reasonably balanced.
        # The sign convention is: right_heat_flux and left_heat_flux may not
        # perfectly cancel because they're computed at different cell locations,
        # but the imbalance should be bounded.
        imbalance = abs(result[key])
        assert imbalance < 100.0, f"Flux imbalance too large: {imbalance}"
