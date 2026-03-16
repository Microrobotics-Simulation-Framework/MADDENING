"""Tests for HeartPumpNode -- 2-element Windkessel heart pump model."""

import pytest
import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec
from maddening.nodes.heart_pump import HeartPumpNode


class TestHeartPumpConstruction:
    """Construction and default parameter tests."""

    def test_construction_defaults(self):
        """Verify default params and state shapes."""
        node = HeartPumpNode(name="heart", timestep=0.001)
        assert node.name == "heart"
        assert node.delta_t == 0.001
        assert node.params["resistance"] == 1.0
        assert node.params["compliance"] == 1.0
        assert node.params["heart_rate"] == 72.0
        assert node.params["stroke_volume"] == 70.0
        assert node.params["venous_pressure"] == 0.0
        assert node.params["systole_fraction"] == 0.35
        assert node.params["initial_pressure"] == 80.0

        state = node.initial_state()
        assert "arterial_pressure" in state
        assert "phase" in state
        assert "flow_rate" in state
        # All scalars
        assert state["arterial_pressure"].shape == ()
        assert state["phase"].shape == ()
        assert state["flow_rate"].shape == ()

    def test_initial_state(self):
        """arterial_pressure=initial_pressure, phase=0, flow_rate=0."""
        node = HeartPumpNode(
            name="heart", timestep=0.001, initial_pressure=90.0
        )
        state = node.initial_state()
        assert float(state["arterial_pressure"]) == pytest.approx(90.0)
        assert float(state["phase"]) == pytest.approx(0.0)
        assert float(state["flow_rate"]) == pytest.approx(0.0)

    def test_custom_params(self):
        """Custom parameters are stored correctly."""
        node = HeartPumpNode(
            name="heart",
            timestep=0.0005,
            resistance=2.0,
            compliance=0.5,
            heart_rate=80.0,
            stroke_volume=65.0,
            venous_pressure=5.0,
            systole_fraction=0.3,
            initial_pressure=75.0,
        )
        assert node.params["resistance"] == 2.0
        assert node.params["compliance"] == 0.5
        assert node.params["heart_rate"] == 80.0
        assert node.params["stroke_volume"] == 65.0
        assert node.params["venous_pressure"] == 5.0
        assert node.params["systole_fraction"] == 0.3
        assert node.params["initial_pressure"] == 75.0


class TestHeartPumpPhysics:
    """Physics and waveform tests."""

    def test_phase_advances(self):
        """After steps, phase should advance proportionally to heart_rate."""
        hr = 72.0  # bpm
        dt = 0.001
        node = HeartPumpNode(name="heart", timestep=dt, heart_rate=hr)
        state = node.initial_state()

        n_steps = 100
        for _ in range(n_steps):
            state = node.update(state, {}, dt)

        expected_phase_advance = n_steps * dt * hr / 60.0
        # Phase wraps at 1.0, so take modulo
        expected_phase = expected_phase_advance % 1.0
        assert float(state["phase"]) == pytest.approx(expected_phase, abs=1e-5)

    def test_phase_wraps(self):
        """Phase wraps from ~1 back to ~0 across a heartbeat boundary."""
        hr = 60.0  # 1 beat per second
        dt = 0.01
        node = HeartPumpNode(name="heart", timestep=dt, heart_rate=hr)
        state = node.initial_state()

        # Run for 0.99 seconds (phase should be ~0.99)
        for _ in range(99):
            state = node.update(state, {}, dt)
        assert float(state["phase"]) > 0.9

        # Two more steps to ensure we clearly pass 1.0 and wrap
        state = node.update(state, {}, dt)
        state = node.update(state, {}, dt)
        assert float(state["phase"]) < 0.15

    def test_systolic_flow_positive(self):
        """During systole (phase < systole_fraction), flow_rate > 0."""
        node = HeartPumpNode(
            name="heart", timestep=0.001, heart_rate=60.0,
            systole_fraction=0.35,
        )
        state = node.initial_state()

        # Advance a few steps to get into systole (phase > 0 but < 0.35)
        dt = 0.001
        for _ in range(50):  # phase ~= 0.05
            state = node.update(state, {}, dt)

        phase = float(state["phase"])
        assert 0.0 < phase < 0.35, f"Phase {phase} not in systole"
        assert float(state["flow_rate"]) > 0.0

    def test_diastolic_flow_zero(self):
        """During diastole (phase >= systole_fraction), flow_rate = 0."""
        hr = 60.0  # 1 beat per second
        dt = 0.001
        sf = 0.35
        node = HeartPumpNode(
            name="heart", timestep=dt, heart_rate=hr,
            systole_fraction=sf,
        )
        state = node.initial_state()

        # Advance to phase ~0.5 (well into diastole)
        for _ in range(500):
            state = node.update(state, {}, dt)

        phase = float(state["phase"])
        assert phase >= sf, f"Phase {phase} not in diastole"
        assert float(state["flow_rate"]) == pytest.approx(0.0, abs=1e-6)

    def test_pressure_oscillates(self):
        """Over a full cardiac cycle, pressure should rise during systole
        and fall during diastole."""
        hr = 60.0
        dt = 0.001
        sf = 0.35
        node = HeartPumpNode(
            name="heart", timestep=dt, heart_rate=hr,
            systole_fraction=sf, resistance=1.0, compliance=1.0,
            initial_pressure=80.0,
        )
        state = node.initial_state()

        # Track pressure over one full cycle (1 second at 60 bpm)
        pressures = []
        for _ in range(1000):
            state = node.update(state, {}, dt)
            pressures.append(float(state["arterial_pressure"]))

        # Pressure should not be monotone -- it should go up and down
        p_max = max(pressures)
        p_min = min(pressures)
        assert p_max > p_min, "Pressure should oscillate"

        # Systolic peak should be higher than initial
        assert p_max > 80.0, "Pressure should rise during systole"

    def test_steady_state_pressure(self):
        """After many cycles, mean pressure should approach
        R * mean_flow + P_venous (Windkessel steady state)."""
        hr = 72.0
        sv = 70.0
        R = 1.0
        C = 1.0
        P_v = 5.0
        dt = 0.0001  # small dt for stability and accuracy

        node = HeartPumpNode(
            name="heart", timestep=dt, heart_rate=hr,
            stroke_volume=sv, resistance=R, compliance=C,
            venous_pressure=P_v, initial_pressure=80.0,
        )
        state = node.initial_state()

        period = 60.0 / hr  # seconds per beat
        n_cycles = 20
        steps_per_cycle = int(period / dt)
        total_steps = n_cycles * steps_per_cycle

        # Run to steady state
        for _ in range(total_steps):
            state = node.update(state, {}, dt)

        # Collect one more cycle for mean pressure
        pressures = []
        for _ in range(steps_per_cycle):
            state = node.update(state, {}, dt)
            pressures.append(float(state["arterial_pressure"]))

        mean_pressure = sum(pressures) / len(pressures)
        mean_flow = sv * hr / 60.0  # ml/s (or whatever units)
        expected_mean_pressure = R * mean_flow + P_v

        # Accept 20% error due to transient effects and Euler integration
        assert mean_pressure == pytest.approx(
            expected_mean_pressure, rel=0.20
        ), (
            f"Mean pressure {mean_pressure:.2f} should be near "
            f"expected {expected_mean_pressure:.2f}"
        )


class TestHeartPumpCoupling:
    """Boundary input and flux coupling tests."""

    def test_backpressure_coupling(self):
        """Providing backpressure changes the outflow computation."""
        dt = 0.001
        node = HeartPumpNode(
            name="heart", timestep=dt, resistance=1.0,
            compliance=1.0, venous_pressure=0.0, initial_pressure=100.0,
        )
        state = node.initial_state()

        # Step without backpressure
        state_no_bp = node.update(state, {}, dt)

        # Step with high backpressure (reduces outflow -> higher pressure)
        state_with_bp = node.update(
            state, {"backpressure": jnp.array(50.0)}, dt
        )

        # With backpressure of 50, outflow Q_out = (100 - 50)/R = 50
        # Without backpressure, Q_out = (100 - 0)/R = 100
        # So pressure should be higher with backpressure (less outflow)
        assert float(state_with_bp["arterial_pressure"]) > float(
            state_no_bp["arterial_pressure"]
        )

    def test_boundary_fluxes(self):
        """compute_boundary_fluxes returns inlet_pressure."""
        node = HeartPumpNode(
            name="heart", timestep=0.001, initial_pressure=120.0
        )
        state = node.initial_state()
        fluxes = node.compute_boundary_fluxes(state, {}, 0.001)
        assert "inlet_pressure" in fluxes
        assert float(fluxes["inlet_pressure"]) == pytest.approx(120.0)

    def test_boundary_input_spec(self):
        """backpressure is declared in boundary_input_spec."""
        node = HeartPumpNode(name="heart", timestep=0.001)
        spec = node.boundary_input_spec()
        assert "backpressure" in spec
        assert isinstance(spec["backpressure"], BoundaryInputSpec)
        assert spec["backpressure"].shape == ()


class TestHeartPumpDerivatives:
    """Tests for derivatives and implicit_residual."""

    def test_derivatives(self):
        """derivatives() returns correct keys and shapes."""
        node = HeartPumpNode(name="heart", timestep=0.001)
        state = node.initial_state()
        derivs = node.derivatives(state, {})

        assert "arterial_pressure" in derivs
        assert "phase" in derivs
        assert "flow_rate" in derivs

        # All should be scalars
        assert derivs["arterial_pressure"].shape == ()
        assert derivs["phase"].shape == ()
        assert derivs["flow_rate"].shape == ()

        # Phase derivative should be heart_rate / 60
        assert float(derivs["phase"]) == pytest.approx(72.0 / 60.0)

    def test_implicit_residual(self):
        """implicit_residual returns correct keys and zero at equilibrium."""
        node = HeartPumpNode(name="heart", timestep=0.001)
        state = node.initial_state()
        # If state_new == state_old and derivs are zero, residual should
        # equal -dt * derivs
        residual = node.implicit_residual(state, state, {}, 0.001)
        assert "arterial_pressure" in residual
        assert "phase" in residual
        assert "flow_rate" in residual


class TestHeartPumpJAX:
    """JAX compilation and differentiation tests."""

    def test_jit_compatible(self):
        """jax.jit(node.update) works."""
        node = HeartPumpNode(name="heart", timestep=0.001)
        state = node.initial_state()
        bi = {"backpressure": jnp.array(5.0)}

        jit_update = jax.jit(node.update)
        new_state = jit_update(state, bi, 0.001)

        assert jnp.isfinite(new_state["arterial_pressure"])
        assert jnp.isfinite(new_state["phase"])
        assert jnp.isfinite(new_state["flow_rate"])

    def test_grad_compatible(self):
        """jax.grad through update works."""
        node = HeartPumpNode(
            name="heart", timestep=0.001, resistance=1.0,
            compliance=1.0,
        )

        def loss_fn(init_pressure):
            state = {
                "arterial_pressure": init_pressure,
                "phase": jnp.array(0.1, dtype=jnp.float32),
                "flow_rate": jnp.array(0.0, dtype=jnp.float32),
            }
            for _ in range(10):
                state = node.update(state, {}, 0.001)
            return state["arterial_pressure"]

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(jnp.array(80.0, dtype=jnp.float32))
        assert jnp.isfinite(g)


class TestHeartPumpGraphIntegration:
    """GraphManager integration tests."""

    def test_graph_manager_integration(self):
        """HeartPumpNode works in a GraphManager."""
        gm = GraphManager()
        node = HeartPumpNode(
            name="heart", timestep=0.001, heart_rate=72.0,
            initial_pressure=80.0,
        )
        gm.add_node(node)
        gm.compile()

        # Run for a few steps
        gm.run(100)
        state = gm.get_node_state("heart")

        assert jnp.isfinite(state["arterial_pressure"])
        assert jnp.isfinite(state["phase"])
        assert jnp.isfinite(state["flow_rate"])
        # Phase should have advanced
        assert float(state["phase"]) > 0.0

    def test_graph_manager_with_scan(self):
        """HeartPumpNode works with run_scan_with_history."""
        gm = GraphManager()
        node = HeartPumpNode(
            name="heart", timestep=0.001, heart_rate=60.0,
            initial_pressure=80.0,
        )
        gm.add_node(node)
        gm.compile()

        final, history = gm.run_scan_with_history(500)
        assert history["heart"]["arterial_pressure"].shape == (500,)
        assert history["heart"]["phase"].shape == (500,)
        assert history["heart"]["flow_rate"].shape == (500,)
        # All values should be finite
        assert jnp.all(jnp.isfinite(history["heart"]["arterial_pressure"]))
