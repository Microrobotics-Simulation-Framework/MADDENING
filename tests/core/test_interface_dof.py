"""Tests for Phase 1a: Interface DOF awareness.

Verifies that:
1. SimulationNode.interface_dof_indices() defaults to empty dict
2. HeatNode declares interface DOFs for left/right temperature
3. DD coupling between two heat rods actually transfers heat
4. Interface overrides work in coupled and non-coupled paths
5. Interface overrides work with Jacobi mode
6. Existing nodes without interface DOFs are unaffected
"""

import jax
import jax.numpy as jnp
import pytest

from maddening.core.coupling import CouplingGroup
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.ball import BallNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode


# ==================================================================
# Tests for the base interface
# ==================================================================

class TestInterfaceDofBase:
    """Tests for the base SimulationNode.interface_dof_indices()."""

    def test_default_returns_empty_dict(self):
        """Base SimulationNode.interface_dof_indices() returns {}."""
        ball = BallNode(name="ball", timestep=0.01)
        assert ball.interface_dof_indices() == {}

    def test_spring_returns_empty_dict(self):
        """SpringDamperNode has no interface DOFs."""
        spring = SpringDamperNode(name="s", timestep=0.01)
        assert spring.interface_dof_indices() == {}

    def test_heat_node_declares_interface_dofs(self):
        """HeatNode declares left/right temperature as interface DOFs."""
        h = HeatNode(name="rod", timestep=0.001, n_cells=10)
        dofs = h.interface_dof_indices()
        assert "left_temperature" in dofs
        assert "right_temperature" in dofs
        assert dofs["left_temperature"] == ("temperature", 0)
        assert dofs["right_temperature"] == ("temperature", -1)


# ==================================================================
# Tests for DD coupling between heat rods
# ==================================================================

class TestHeatRodDDCoupling:
    """Dirichlet-Dirichlet coupling between two heat rods.

    The classic "cold lock" problem: rod_a is hot (100C), rod_b is
    cold (0C).  Connected end-to-end via DD coupling (right end of
    rod_a feeds left BC of rod_b, and vice versa).  Without interface
    DOF overrides, HeatNode's internal Dirichlet BC enforcement
    overwrites the coupled values, preventing heat transfer.
    """

    def _make_dd_coupled_rods(self, dt=0.001, n_cells=10, max_iters=20,
                               tol=1e-8, **coupling_kwargs):
        """Create two heat rods with DD coupling."""
        gm = GraphManager()
        rod_a = HeatNode(
            name="rod_a", timestep=dt, n_cells=n_cells, length=1.0,
            thermal_diffusivity=0.01, initial_temperature=100.0,
        )
        rod_b = HeatNode(
            name="rod_b", timestep=dt, n_cells=n_cells, length=1.0,
            thermal_diffusivity=0.01, initial_temperature=0.0,
        )
        gm.add_node(rod_a)
        gm.add_node(rod_b)

        # rod_a's right end -> rod_b's left BC
        gm.add_edge(
            "rod_a", "rod_b",
            "temperature", "left_temperature",
            transform=lambda T: T[-1],
        )
        # rod_b's left end -> rod_a's right BC
        gm.add_edge(
            "rod_b", "rod_a",
            "temperature", "right_temperature",
            transform=lambda T: T[0],
        )

        # Fixed external BCs: rod_a left = 100, rod_b right = 0
        gm.add_external_input("rod_a", "left_temperature", shape=())
        gm.add_external_input("rod_b", "right_temperature", shape=())

        gm.add_coupling_group(
            ["rod_a", "rod_b"],
            max_iterations=max_iters,
            tolerance=tol,
            **coupling_kwargs,
        )
        gm.compile()

        ext = {
            "rod_a": {"left_temperature": jnp.array(100.0)},
            "rod_b": {"right_temperature": jnp.array(0.0)},
        }
        return gm, ext

    def test_dd_coupling_transfers_heat(self):
        """DD coupling should transfer heat between rods."""
        gm, ext = self._make_dd_coupled_rods(max_iters=30, tol=1e-8)

        # Run for a number of steps
        for _ in range(200):
            gm.step(external_inputs=ext)

        state_a = gm.get_node_state("rod_a")
        state_b = gm.get_node_state("rod_b")

        # rod_a's right end should have cooled below 100 (heat flows out)
        T_a_right = float(state_a["temperature"][-1])
        assert T_a_right < 99.0, (
            f"rod_a right end is {T_a_right}, expected cooling below 99"
        )

        # rod_b's left end should have warmed above 0 (heat flows in)
        T_b_left = float(state_b["temperature"][0])
        assert T_b_left > 1.0, (
            f"rod_b left end is {T_b_left}, expected warming above 1"
        )

    def test_dd_coupling_interface_heat_flow(self):
        """Both sides of the interface should show significant heat transfer."""
        gm, ext = self._make_dd_coupled_rods(max_iters=30, tol=1e-10)

        for _ in range(200):
            gm.step(external_inputs=ext)

        state_a = gm.get_node_state("rod_a")
        state_b = gm.get_node_state("rod_b")

        T_a_right = float(state_a["temperature"][-1])
        T_b_left = float(state_b["temperature"][0])

        # Both sides should show heat transfer (moved from initial values)
        assert T_a_right < 98.0, (
            f"rod_a right end at {T_a_right}, expected cooling"
        )
        assert T_b_left > 2.0, (
            f"rod_b left end at {T_b_left}, expected warming"
        )
        # Temperature should be monotonically decreasing from left to right
        # across the full domain (rod_a is hotter, rod_b is colder)
        assert T_a_right > T_b_left, (
            f"rod_a[-1]={T_a_right} should be > rod_b[0]={T_b_left}"
        )

    def test_dd_coupling_steady_state_profile(self):
        """Two equal rods with 100 and 0 endpoints should produce a
        monotonically decreasing profile across the full domain."""
        n = 10
        gm, ext = self._make_dd_coupled_rods(
            dt=0.0001, n_cells=n, max_iters=30, tol=1e-10,
        )

        # Run to near steady state
        gm.run_scan(100000, external_inputs=ext)

        state_a = gm.get_node_state("rod_a")
        state_b = gm.get_node_state("rod_b")

        T_a = state_a["temperature"]
        T_b = state_b["temperature"]

        # rod_a should be monotonically decreasing from left to right
        assert jnp.all(jnp.diff(T_a) <= 0.0), "rod_a should be monotonically decreasing"
        # rod_b should be monotonically decreasing from left to right
        assert jnp.all(jnp.diff(T_b) <= 0.0), "rod_b should be monotonically decreasing"

        # rod_a[0] should be near 100 (fixed BC), rod_b[-1] near 0
        assert float(T_a[0]) == pytest.approx(100.0, abs=1.0)
        assert float(T_b[-1]) == pytest.approx(0.0, abs=1.0)

        # Heat should transfer: rod_a's right part should be cooled,
        # rod_b's left part should be warmed
        assert float(T_a[-1]) < 95.0, "rod_a right should cool down"
        assert float(T_b[0]) > 5.0, "rod_b left should warm up"

    def test_dd_coupling_jacobi_mode(self):
        """DD coupling should also work in Jacobi iteration mode."""
        gm, ext = self._make_dd_coupled_rods(
            max_iters=30, tol=1e-8, iteration_mode="jacobi",
        )

        for _ in range(200):
            gm.step(external_inputs=ext)

        state_a = gm.get_node_state("rod_a")
        state_b = gm.get_node_state("rod_b")

        T_a_right = float(state_a["temperature"][-1])
        T_b_left = float(state_b["temperature"][0])

        assert T_a_right < 99.0
        assert T_b_left > 1.0

    def test_dd_coupling_with_aitken_acceleration(self):
        """DD coupling with Aitken acceleration should converge."""
        gm, ext = self._make_dd_coupled_rods(
            max_iters=30, tol=1e-8, acceleration="aitken",
        )

        for _ in range(200):
            gm.step(external_inputs=ext)

        state_a = gm.get_node_state("rod_a")
        state_b = gm.get_node_state("rod_b")

        T_a_right = float(state_a["temperature"][-1])
        T_b_left = float(state_b["temperature"][0])

        assert T_a_right < 99.0
        assert T_b_left > 1.0

    def test_dd_coupling_diagnostics(self):
        """DD coupling with diagnostics should report iteration count."""
        gm, ext = self._make_dd_coupled_rods(
            max_iters=30, tol=1e-8, diagnostics=True,
        )

        gm.step(external_inputs=ext)
        diag = gm.coupling_diagnostics()
        key = "rod_a+rod_b"
        assert key in diag
        assert "iterations" in diag[key]
        assert "residual" in diag[key]
        # Should converge in less than max_iters
        assert diag[key]["iterations"] < 30

    def test_dd_coupling_run_scan(self):
        """DD coupling should work with run_scan (lax.scan loop)."""
        gm, ext = self._make_dd_coupled_rods(max_iters=30, tol=1e-8)

        gm.run_scan(200, external_inputs=ext)

        state_a = gm.get_node_state("rod_a")
        state_b = gm.get_node_state("rod_b")

        T_a_right = float(state_a["temperature"][-1])
        T_b_left = float(state_b["temperature"][0])

        assert T_a_right < 99.0
        assert T_b_left > 1.0

    def test_dd_coupling_grad_through(self):
        """Should be differentiable through DD coupling."""
        n = 5

        def loss_fn(T_left):
            gm = GraphManager()
            rod_a = HeatNode(
                name="rod_a", timestep=0.001, n_cells=n, length=1.0,
                thermal_diffusivity=0.01, initial_temperature=50.0,
            )
            rod_b = HeatNode(
                name="rod_b", timestep=0.001, n_cells=n, length=1.0,
                thermal_diffusivity=0.01, initial_temperature=50.0,
            )
            gm.add_node(rod_a)
            gm.add_node(rod_b)
            gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                        transform=lambda T: T[-1])
            gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                        transform=lambda T: T[0])
            gm.add_external_input("rod_a", "left_temperature", shape=())
            gm.add_external_input("rod_b", "right_temperature", shape=())
            gm.add_coupling_group(["rod_a", "rod_b"],
                                  max_iterations=10, tolerance=1e-6)
            gm.compile()

            ext = {
                "rod_a": {"left_temperature": T_left},
                "rod_b": {"right_temperature": jnp.array(0.0)},
            }
            gm.run_scan(50, external_inputs=ext)
            state_b = gm.get_node_state("rod_b")
            return jnp.mean(state_b["temperature"])

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(jnp.array(100.0))
        assert jnp.isfinite(g)
        # Increasing T_left should increase mean temperature of rod_b
        assert float(g) > 0.0


# ==================================================================
# Tests for non-coupled path with interface overrides
# ==================================================================

class TestInterfaceOverrideNonCoupled:
    """Verify non-coupled edges use standard Dirichlet enforcement."""

    def test_one_directional_edge_uses_dirichlet(self):
        """A one-directional edge feeding left_temperature to a heat rod
        should use normal Dirichlet BC enforcement (no interface correction
        since there's no coupling group)."""
        n = 10
        gm = GraphManager()
        rod_a = HeatNode(
            name="rod_a", timestep=0.001, n_cells=n,
            thermal_diffusivity=0.01, initial_temperature=100.0,
        )
        rod_b = HeatNode(
            name="rod_b", timestep=0.001, n_cells=n,
            thermal_diffusivity=0.01, initial_temperature=0.0,
        )
        gm.add_node(rod_a)
        gm.add_node(rod_b)

        # rod_a's right end drives rod_b's left BC (one-directional)
        gm.add_edge(
            "rod_a", "rod_b",
            "temperature", "left_temperature",
            transform=lambda T: T[-1],
        )

        # External BCs
        gm.add_external_input("rod_a", "left_temperature", shape=())
        gm.add_external_input("rod_a", "right_temperature", shape=())
        gm.add_external_input("rod_b", "right_temperature", shape=())
        gm.compile()

        ext = {
            "rod_a": {
                "left_temperature": jnp.array(100.0),
                "right_temperature": jnp.array(100.0),
            },
            "rod_b": {
                "right_temperature": jnp.array(0.0),
            },
        }

        gm.run_scan(500, external_inputs=ext)
        state_b = gm.get_node_state("rod_b")

        # With standard Dirichlet enforcement (no coupling group),
        # rod_b's left end should be pinned to rod_a's T[-1] = 100
        T_b_left = float(state_b["temperature"][0])
        assert T_b_left > 90.0, (
            f"rod_b left end is {T_b_left}, expected ~100 from Dirichlet"
        )


# ==================================================================
# Tests verifying backward compatibility
# ==================================================================

class TestInterfaceOverrideBackwardCompat:
    """Nodes without interface DOFs should behave identically."""

    def test_ball_spring_unaffected(self):
        """Ball-spring coupling should work as before."""
        gm = GraphManager()
        ball = BallNode(name="ball", timestep=0.001, initial_position=5.0)
        spring = SpringDamperNode(
            name="spring", timestep=0.001,
            stiffness=50.0, damping=1.0, mass=1.0,
        )
        gm.add_node(ball)
        gm.add_node(spring)
        gm.add_edge("ball", "spring", "position", "anchor_position")
        gm.compile()

        # Run a few steps
        for _ in range(100):
            gm.step()

        state = gm.get_node_state("ball")
        # Ball should have fallen from 5.0
        assert float(state["position"]) < 5.0

    def test_coupled_springs_unaffected(self):
        """Bidirectional spring coupling should work as before."""
        gm = GraphManager()
        a = SpringDamperNode(
            name="spring_a", timestep=0.001,
            stiffness=50.0, damping=1.0, mass=1.0,
            rest_length=1.0, initial_position=0.0,
        )
        b = SpringDamperNode(
            name="spring_b", timestep=0.001,
            stiffness=50.0, damping=1.0, mass=1.0,
            rest_length=1.0, initial_position=2.0,
        )
        gm.add_node(a)
        gm.add_node(b)
        gm.add_edge("spring_a", "spring_b", "position", "anchor_position")
        gm.add_edge("spring_b", "spring_a", "position", "anchor_position")
        gm.add_coupling_group(["spring_a", "spring_b"],
                              max_iterations=10, tolerance=1e-6)
        gm.compile()

        for _ in range(100):
            gm.step()

        # Springs should be moving toward equilibrium
        state_a = gm.get_node_state("spring_a")
        state_b = gm.get_node_state("spring_b")
        # Distance between them should be approaching rest length
        dist = abs(float(state_b["position"]) - float(state_a["position"]))
        assert dist < 3.0  # Started at 2.0, rest_length is 1.0
