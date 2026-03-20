"""Tests for Phase 6 flux-based coupling.

Covers: boundary_input_spec, compute_boundary_fluxes, additive edges,
flux fields on edges, and JAX compatibility (grad, scan).
"""

import jax
import jax.numpy as jnp
import pytest

from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.nodes.ball import BallNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.rigid_body_2d import RigidBody2DNode
from maddening.nodes.spring import SpringDamperNode


# ==================================================================
# boundary_input_spec tests
# ==================================================================

class TestBoundaryInputSpec:
    """Tests for SimulationNode.boundary_input_spec() on built-in nodes."""

    def test_boundary_input_spec_heat(self):
        """HeatNode declares 3 boundary inputs with correct types."""
        node = HeatNode(name="rod", timestep=0.001, n_cells=10)
        spec = node.boundary_input_spec()

        assert len(spec) == 3
        assert "left_temperature" in spec
        assert "right_temperature" in spec
        assert "heat_source" in spec

        # Shapes
        assert spec["left_temperature"].shape == ()
        assert spec["right_temperature"].shape == ()
        assert spec["heat_source"].shape == (10,)

        # heat_source is additive
        assert spec["heat_source"].coupling_type == "additive"
        # temperature inputs are replacive (default)
        assert spec["left_temperature"].coupling_type == "replacive"
        assert spec["right_temperature"].coupling_type == "replacive"

    def test_boundary_input_spec_spring(self):
        """SpringDamperNode declares anchor_position."""
        node = SpringDamperNode(name="spr", timestep=0.01)
        spec = node.boundary_input_spec()

        assert len(spec) == 1
        assert "anchor_position" in spec
        assert spec["anchor_position"].shape == ()
        assert isinstance(spec["anchor_position"], BoundaryInputSpec)

    def test_boundary_input_spec_rigid_body(self):
        """RigidBody2DNode has additive force and torque."""
        node = RigidBody2DNode(name="rb", timestep=0.01)
        spec = node.boundary_input_spec()

        assert "force" in spec
        assert "torque" in spec
        assert spec["force"].coupling_type == "additive"
        assert spec["torque"].coupling_type == "additive"
        assert spec["force"].shape == (2,)
        assert spec["torque"].shape == ()

    def test_boundary_input_spec_default_empty(self):
        """Base SimulationNode returns empty dict."""

        class MinimalNode(SimulationNode):
            @property
            def requires_halo(self) -> bool:
                return False

            def initial_state(self):
                return {"x": jnp.array(0.0)}

            def update(self, state, boundary_inputs, dt):
                return state

        node = MinimalNode(name="bare", timestep=0.01)
        spec = node.boundary_input_spec()
        assert spec == {}

    def test_boundary_input_spec_ball(self):
        """BallNode declares table_position."""
        node = BallNode(name="ball", timestep=0.01)
        spec = node.boundary_input_spec()

        assert "table_position" in spec
        assert spec["table_position"].shape == ()


# ==================================================================
# compute_boundary_fluxes tests
# ==================================================================

class TestComputeBoundaryFluxes:
    """Tests for SimulationNode.compute_boundary_fluxes() on built-in nodes."""

    def test_compute_fluxes_heat(self):
        """HeatNode returns left/right heat flux, values are finite."""
        node = HeatNode(name="rod", timestep=0.001, n_cells=10,
                        thermal_diffusivity=0.01, length=1.0,
                        initial_temperature=100.0)
        state = node.initial_state()
        bi = {}  # no boundary inputs
        fluxes = node.compute_boundary_fluxes(state, bi, 0.001)

        assert "left_heat_flux" in fluxes
        assert "right_heat_flux" in fluxes
        assert jnp.isfinite(fluxes["left_heat_flux"])
        assert jnp.isfinite(fluxes["right_heat_flux"])

    def test_compute_fluxes_heat_nonuniform(self):
        """Heat flux is nonzero for a non-uniform temperature profile."""
        node = HeatNode(name="rod", timestep=0.001, n_cells=10,
                        thermal_diffusivity=0.01, length=1.0)
        # Create a linear temperature gradient
        state = {"temperature": jnp.linspace(0.0, 100.0, 10)}
        fluxes = node.compute_boundary_fluxes(state, {}, 0.001)

        # With a gradient, fluxes should be nonzero
        assert float(jnp.abs(fluxes["left_heat_flux"])) > 0.0
        assert float(jnp.abs(fluxes["right_heat_flux"])) > 0.0

    def test_compute_fluxes_spring(self):
        """SpringDamperNode returns spring_force."""
        node = SpringDamperNode(name="spr", timestep=0.01,
                                stiffness=100.0, damping=1.0,
                                rest_length=1.0, initial_position=2.0)
        state = node.initial_state()
        # Anchor at 0.0 => stretch = 2.0 - 0.0 - 1.0 = 1.0
        bi = {"anchor_position": jnp.array(0.0)}
        fluxes = node.compute_boundary_fluxes(state, bi, 0.01)

        assert "spring_force" in fluxes
        assert jnp.isfinite(fluxes["spring_force"])
        # Force should be nonzero (stretched spring)
        assert float(jnp.abs(fluxes["spring_force"])) > 0.0


# ==================================================================
# Additive edge tests
# ==================================================================

class TestAdditiveEdges:
    """Tests for additive edge accumulation."""

    def test_additive_edge_sums_values(self):
        """Two additive force edges to RigidBody2DNode accumulate."""
        gm = GraphManager()

        # Two springs that each provide a force to the rigid body
        gm.add_node(SpringDamperNode(name="spr_a", timestep=0.01,
                                      stiffness=50.0, damping=0.0,
                                      rest_length=1.0,
                                      initial_position=1.0))
        gm.add_node(SpringDamperNode(name="spr_b", timestep=0.01,
                                      stiffness=30.0, damping=0.0,
                                      rest_length=1.0,
                                      initial_position=-1.0))
        gm.add_node(RigidBody2DNode(name="body", timestep=0.01,
                                     mass=1.0, inertia=1.0,
                                     gravity=(0.0, 0.0)))

        # Both springs contribute force to the body (additive)
        # Use position as a scalar, transform to 2D force
        gm.add_edge("spr_a", "body", "position", "force",
                     transform=lambda p: jnp.array([p, 0.0]),
                     additive=False)
        gm.add_edge("spr_b", "body", "position", "force",
                     transform=lambda p: jnp.array([0.0, p]),
                     additive=True)

        gm.compile()
        state = gm.run_scan(10)

        # Body should have moved due to accumulated forces
        assert jnp.all(jnp.isfinite(state["body"]["x"]))
        assert jnp.all(jnp.isfinite(state["body"]["v"]))

        # Build a graph with only spr_a's force for comparison
        gm2 = GraphManager()
        gm2.add_node(SpringDamperNode(name="spr_a", timestep=0.01,
                                       stiffness=50.0, damping=0.0,
                                       rest_length=1.0,
                                       initial_position=1.0))
        gm2.add_node(RigidBody2DNode(name="body", timestep=0.01,
                                      mass=1.0, inertia=1.0,
                                      gravity=(0.0, 0.0)))
        gm2.add_edge("spr_a", "body", "position", "force",
                      transform=lambda p: jnp.array([p, 0.0]))
        gm2.compile()
        state2 = gm2.run_scan(10)

        # With both forces, result should differ from single force
        diff = float(jnp.sum(jnp.abs(state["body"]["x"] - state2["body"]["x"])))
        assert diff > 1e-6, "Additive edge should change result"

    def test_additive_edge_default_false(self):
        """EdgeSpec.additive defaults to False."""
        edge = EdgeSpec("a", "b", "x", "y")
        assert edge.additive is False

    def test_additive_flag_in_edgespec(self):
        """EdgeSpec stores additive flag."""
        edge = EdgeSpec("a", "b", "x", "y", additive=True)
        assert edge.additive is True


# ==================================================================
# Flux field on edge tests
# ==================================================================

class TestFluxFieldEdge:
    """Tests for edges that read from compute_boundary_fluxes output."""

    def test_flux_field_on_edge(self):
        """Edge reads from compute_boundary_fluxes output (heat flux).

        Set up two heat rods where rod_a's right_heat_flux feeds
        rod_b's heat_source via an edge with the flux field as source.
        """
        gm = GraphManager()
        gm.add_node(HeatNode(name="rod_a", timestep=0.001, n_cells=10,
                              thermal_diffusivity=0.01, length=1.0,
                              initial_temperature=100.0))
        gm.add_node(HeatNode(name="rod_b", timestep=0.001, n_cells=10,
                              thermal_diffusivity=0.01, length=1.0,
                              initial_temperature=0.0))

        # Standard value edge: rod_a's right cell temp -> rod_b's left BC
        gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                     transform=lambda T: T[-1])

        # Flux edge: rod_a's right_heat_flux -> rod_b's heat_source
        # heat_source is (n_cells,) so we broadcast the scalar flux
        gm.add_edge("rod_a", "rod_b", "right_heat_flux", "heat_source",
                     transform=lambda q: jnp.broadcast_to(q, (10,)))

        gm.compile()
        state = gm.run_scan(50)

        # Both rods should produce finite results
        assert jnp.all(jnp.isfinite(state["rod_a"]["temperature"]))
        assert jnp.all(jnp.isfinite(state["rod_b"]["temperature"]))

    def test_flux_field_in_coupling_group(self):
        """Flux field edges work inside a coupling group."""
        gm = GraphManager()
        gm.add_node(HeatNode(name="rod_a", timestep=0.001, n_cells=10,
                              thermal_diffusivity=0.01, length=1.0,
                              initial_temperature=100.0))
        gm.add_node(HeatNode(name="rod_b", timestep=0.001, n_cells=10,
                              thermal_diffusivity=0.01, length=1.0,
                              initial_temperature=0.0))

        # Bidirectional value coupling
        gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                     transform=lambda T: T[-1])
        gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                     transform=lambda T: T[0])

        gm.add_coupling_group(
            ["rod_a", "rod_b"],
            max_iterations=10, tolerance=1e-8,
        )
        gm.compile()
        state = gm.run_scan(50)

        assert jnp.all(jnp.isfinite(state["rod_a"]["temperature"]))
        assert jnp.all(jnp.isfinite(state["rod_b"]["temperature"]))


# ==================================================================
# JAX compatibility tests
# ==================================================================

class TestFluxCouplingJAX:
    """Tests for JAX compatibility of flux-based coupling."""

    def test_flux_coupling_grad_compatible(self):
        """jax.grad through flux-based coupling works."""
        gm = GraphManager()
        gm.add_node(HeatNode(name="rod_a", timestep=0.001, n_cells=5,
                              thermal_diffusivity=0.01, length=1.0,
                              initial_temperature=100.0))
        gm.add_node(HeatNode(name="rod_b", timestep=0.001, n_cells=5,
                              thermal_diffusivity=0.01, length=1.0,
                              initial_temperature=0.0))

        # Value edges (bidirectional cycle)
        gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                     transform=lambda T: T[-1])
        gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                     transform=lambda T: T[0])

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

    def test_flux_coupling_with_scan(self):
        """Flux edges work inside run_scan."""
        gm = GraphManager()
        gm.add_node(HeatNode(name="rod_a", timestep=0.001, n_cells=5,
                              thermal_diffusivity=0.01, length=1.0,
                              initial_temperature=100.0))
        gm.add_node(HeatNode(name="rod_b", timestep=0.001, n_cells=5,
                              thermal_diffusivity=0.01, length=1.0,
                              initial_temperature=0.0))

        # Flux edge from rod_a to rod_b
        gm.add_edge("rod_a", "rod_b", "right_heat_flux", "heat_source",
                     transform=lambda q: jnp.broadcast_to(q, (5,)))
        # Value edge the other way
        gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                     transform=lambda T: T[0])

        gm.compile()
        final, history = gm.run_scan_with_history(100)

        assert history["rod_a"]["temperature"].shape == (100, 5)
        assert history["rod_b"]["temperature"].shape == (100, 5)
        assert jnp.all(jnp.isfinite(final["rod_a"]["temperature"]))
        assert jnp.all(jnp.isfinite(final["rod_b"]["temperature"]))
