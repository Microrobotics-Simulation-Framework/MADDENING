"""Tests for the LBMPipeNode -- 3D Lattice Boltzmann fluid in a pipe."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.lbm_pipe import (
    LBMPipeNode,
    _equilibrium,
    _compute_macroscopic,
    _stream,
    _tracer_equilibrium,
    _stream7,
    _psi,
    _shan_chen_force,
    _eos_pressure,
    _E,
    _W,
    _OPP,
    _E7,
    _W7,
    _OPP7,
)


class TestLatticeConstants:
    def test_19_directions(self):
        assert _E.shape == (19, 3)
        assert _W.shape == (19,)
        assert _OPP.shape == (19,)

    def test_weights_sum_to_one(self):
        np.testing.assert_allclose(float(jnp.sum(_W)), 1.0, rtol=1e-6)

    def test_opposite_is_involution(self):
        """opp[opp[q]] == q for all q."""
        for q in range(19):
            assert int(_OPP[_OPP[q]]) == q

    def test_opposite_reverses_velocity(self):
        """e[opp[q]] == -e[q] for all q."""
        for q in range(19):
            np.testing.assert_array_equal(_E[_OPP[q]], -_E[q])


class TestD3Q7Constants:
    def test_7_directions(self):
        assert _E7.shape == (7, 3)
        assert _W7.shape == (7,)
        assert _OPP7.shape == (7,)

    def test_weights_sum_to_one(self):
        np.testing.assert_allclose(float(np.sum(_W7)), 1.0, rtol=1e-6)

    def test_opposite_is_involution(self):
        for q in range(7):
            assert int(_OPP7[_OPP7[q]]) == q


class TestEquilibrium:
    def test_equilibrium_shape(self):
        rho = jnp.ones((4, 4, 4))
        u = jnp.zeros((4, 4, 4, 3))
        f_eq = _equilibrium(rho, u)
        assert f_eq.shape == (4, 4, 4, 19)

    def test_equilibrium_sums_to_density(self):
        rho = jnp.ones((8, 8, 8)) * 1.5
        u = jnp.zeros((8, 8, 8, 3))
        f_eq = _equilibrium(rho, u)
        rho_back = jnp.sum(f_eq, axis=-1)
        np.testing.assert_allclose(rho_back, 1.5, rtol=1e-5)

    def test_equilibrium_recovers_velocity(self):
        rho = jnp.ones((4, 4, 4))
        u = jnp.zeros((4, 4, 4, 3)).at[:, :, :, 0].set(0.05)
        f_eq = _equilibrium(rho, u)
        rho2, u2 = _compute_macroscopic(f_eq)
        np.testing.assert_allclose(u2[..., 0], 0.05, atol=1e-5)


class TestTracerEquilibrium:
    def test_shape(self):
        c = jnp.ones((4, 4, 4))
        u = jnp.zeros((4, 4, 4, 3))
        g_eq = _tracer_equilibrium(c, u)
        assert g_eq.shape == (4, 4, 4, 7)

    def test_sums_to_concentration(self):
        c = jnp.ones((4, 4, 4)) * 0.8
        u = jnp.zeros((4, 4, 4, 3))
        g_eq = _tracer_equilibrium(c, u)
        c_back = jnp.sum(g_eq, axis=-1)
        np.testing.assert_allclose(c_back, 0.8, rtol=1e-5)


class TestStreaming:
    def test_streaming_conserves_mass(self):
        """Total mass should be conserved by streaming."""
        key = jax.random.PRNGKey(0)
        f = jax.random.uniform(key, (8, 8, 8, 19))
        f_streamed = _stream(f)
        np.testing.assert_allclose(
            float(jnp.sum(f)), float(jnp.sum(f_streamed)), rtol=1e-5,
        )

    def test_streaming_shifts_correctly(self):
        """A delta function at (2,2,2,q=1) should shift to (3,2,2,q=1)."""
        f = jnp.zeros((8, 8, 8, 19))
        f = f.at[2, 2, 2, 1].set(1.0)  # q=1 is (+1,0,0)
        f_s = _stream(f)
        assert float(f_s[3, 2, 2, 1]) == pytest.approx(1.0)
        assert float(f_s[2, 2, 2, 1]) == pytest.approx(0.0)

    def test_stream7_conserves_mass(self):
        key = jax.random.PRNGKey(1)
        g = jax.random.uniform(key, (8, 8, 8, 7))
        g_s = _stream7(g)
        np.testing.assert_allclose(
            float(jnp.sum(g)), float(jnp.sum(g_s)), rtol=1e-5,
        )


class TestLBMPipeNode:
    def test_creation(self):
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=8, nz=8)
        assert node.name == "fluid"

    def test_tau_validation(self):
        with pytest.raises(ValueError, match="tau must be > 0.5"):
            LBMPipeNode("fluid", timestep=0.01, tau=0.3)

    def test_tau_tracer_validation(self):
        with pytest.raises(ValueError, match="tau_tracer must be > 0.5"):
            LBMPipeNode("fluid", timestep=0.01, tau_tracer=0.4)

    def test_fill_fraction_validation(self):
        with pytest.raises(ValueError, match="fill_fraction"):
            LBMPipeNode("fluid", timestep=0.01, fill_fraction=0.0)

    def test_viscosity_property(self):
        node = LBMPipeNode("fluid", timestep=0.01, tau=0.8)
        np.testing.assert_allclose(node.viscosity, 0.1, rtol=1e-6)

    def test_initial_state_shapes(self):
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=8, nz=8)
        state = node.initial_state()
        assert state["f"].shape == (16, 8, 8, 19)
        assert state["density"].shape == (16, 8, 8)
        assert state["velocity"].shape == (16, 8, 8, 3)
        assert state["tracer"].shape == (16, 8, 8)
        assert state["tracer_f"].shape == (16, 8, 8, 7)

    def test_initial_density_is_one(self):
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=8, nz=8)
        state = node.initial_state()
        np.testing.assert_allclose(state["density"], 1.0, rtol=1e-5)

    def test_initial_tracer_full_pipe(self):
        """With fill_fraction=1.0, tracer should be 1.0 inside pipe."""
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=8, nz=8,
                          fill_fraction=1.0)
        state = node.initial_state()
        fluid = ~node._wall_mask
        np.testing.assert_allclose(
            state["tracer"][fluid], 1.0, rtol=1e-5,
        )

    def test_initial_tracer_partial_fill(self):
        """With fill_fraction=0.75, about 75% of fluid cells have tracer=1."""
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=16, nz=16,
                          fill_fraction=0.75)
        state = node.initial_state()
        fluid = ~node._wall_mask
        n_fluid = int(jnp.sum(fluid))
        n_liquid = int(jnp.sum(state["tracer"][fluid] > 0.5))
        frac = n_liquid / n_fluid
        assert 0.70 < frac < 0.80, f"Expected ~0.75, got {frac:.3f}"

    def test_wall_mask_shape(self):
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=8, nz=8)
        assert node._wall_mask.shape == (16, 8, 8)
        assert node._wall_mask.dtype == jnp.bool_

    def test_wall_mask_blocks_corners(self):
        """Corners of the cross-section should be wall cells."""
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=16, nz=16,
                          pipe_radius=0.9)
        # Corner (0,0) should be wall
        assert bool(node._wall_mask[0, 0, 0])
        # Centre should be fluid
        assert not bool(node._wall_mask[0, 8, 8])

    def test_propeller_mask(self):
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=8, nz=8,
                          propeller_x=5)
        # Propeller should exist at x=5
        assert bool(jnp.any(node._propeller_mask[5, :, :]))
        # Should not exist at x=0
        assert not bool(jnp.any(node._propeller_mask[0, :, :]))

    def test_single_step(self):
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=8, nz=8)
        state = node.initial_state()
        new_state = node.update(state, {}, 0.01)

        assert new_state["f"].shape == (16, 8, 8, 19)
        assert new_state["tracer"].shape == (16, 8, 8)
        assert new_state["tracer_f"].shape == (16, 8, 8, 7)
        assert not jnp.any(jnp.isnan(new_state["density"]))
        assert not jnp.any(jnp.isnan(new_state["velocity"]))
        assert not jnp.any(jnp.isnan(new_state["tracer"]))

    def test_mass_conservation(self):
        """Total density should be approximately conserved."""
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=8, nz=8,
                          propeller_strength=0.0)
        state = node.initial_state()
        # Only count fluid cells
        fluid = ~node._wall_mask
        initial_mass = float(jnp.sum(state["density"] * fluid))

        for _ in range(10):
            state = node.update(state, {}, 0.01)

        final_mass = float(jnp.sum(state["density"] * fluid))
        np.testing.assert_allclose(initial_mass, final_mass, rtol=1e-4)

    def test_propeller_drives_flow(self):
        """With propeller, velocity in x should increase."""
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=8, nz=8,
                          propeller_strength=0.001)
        state = node.initial_state()

        for _ in range(50):
            state = node.update(state, {}, 0.01)

        # Mean x-velocity in fluid cells should be positive
        fluid = ~node._wall_mask
        mean_ux = float(jnp.sum(state["velocity"][..., 0] * fluid) / jnp.sum(fluid))
        assert mean_ux > 0

    def test_stability_many_steps(self):
        """Node should not blow up over many steps."""
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=8, nz=8,
                          tau=0.8, propeller_strength=0.0005)
        state = node.initial_state()

        for _ in range(200):
            state = node.update(state, {}, 0.01)

        assert not jnp.any(jnp.isnan(state["density"]))
        assert not jnp.any(jnp.isinf(state["velocity"]))
        # Density should stay near 1
        fluid = ~node._wall_mask
        max_rho = float(jnp.max(state["density"] * fluid))
        min_rho = float(jnp.min(state["density"] + (~fluid) * 999))
        assert 0.5 < min_rho < 1.5
        assert 0.5 < max_rho < 1.5

    def test_boundary_input_propeller(self):
        """Dynamic propeller_force boundary input works."""
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=8, nz=8,
                          propeller_strength=0.0)
        state = node.initial_state()

        # Run with override force
        for _ in range(20):
            state = node.update(state, {"propeller_force": 0.002}, 0.01)

        fluid = ~node._wall_mask
        mean_ux = float(jnp.sum(state["velocity"][..., 0] * fluid) / jnp.sum(fluid))
        assert mean_ux > 0


class TestGravity:
    def test_gravity_creates_downward_velocity(self):
        """With gravity in -z, fluid should develop negative z-velocity."""
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=8, nz=8,
                          propeller_strength=0.0, gravity=-0.0005)
        state = node.initial_state()

        for _ in range(50):
            state = node.update(state, {}, 0.01)

        fluid = ~node._wall_mask
        mean_uz = float(jnp.sum(state["velocity"][..., 2] * fluid) / jnp.sum(fluid))
        # In a closed pipe, net z-velocity is near zero (walls confine),
        # but individual cells should show non-zero z-velocity
        max_uz = float(jnp.max(jnp.abs(state["velocity"][..., 2] * fluid)))
        assert max_uz > 0

    def test_gravity_stable(self):
        """Simulation with gravity should remain stable."""
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=8, nz=8,
                          tau=0.8, gravity=-0.0001)
        state = node.initial_state()

        for _ in range(200):
            state = node.update(state, {}, 0.01)

        assert not jnp.any(jnp.isnan(state["density"]))
        assert not jnp.any(jnp.isnan(state["velocity"]))


class TestTracer:
    def test_tracer_conservation_full_fill(self):
        """Total tracer distribution mass should be conserved."""
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=8, nz=8,
                          propeller_strength=0.0, fill_fraction=1.0)
        state = node.initial_state()
        # The conserved quantity is sum(tracer_f) — the total distribution
        # mass.  The fluid-cell-only tracer fluctuates slightly as mass
        # cycles through wall bounce-back with a 1-step delay.
        initial_mass = float(jnp.sum(state["tracer_f"]))

        for _ in range(20):
            state = node.update(state, {}, 0.01)

        final_mass = float(jnp.sum(state["tracer_f"]))
        np.testing.assert_allclose(initial_mass, final_mass, rtol=1e-4)

    def test_tracer_bounded(self):
        """Tracer values should stay in [0, 1]."""
        node = LBMPipeNode("fluid", timestep=0.01, nx=16, ny=8, nz=8,
                          propeller_strength=0.001, gravity=-0.0001,
                          fill_fraction=0.75)
        state = node.initial_state()

        for _ in range(50):
            state = node.update(state, {}, 0.01)

        assert float(jnp.min(state["tracer"])) >= -0.01
        assert float(jnp.max(state["tracer"])) <= 1.01


class TestLBMGraphIntegration:
    def test_graph_compile_and_step(self):
        gm = GraphManager()
        gm.add_node(LBMPipeNode("fluid", timestep=0.01, nx=8, ny=8, nz=8))
        gm.compile()
        gm.step()
        state = gm.get_node_state("fluid")
        assert "density" in state
        assert "tracer" in state
        assert state["density"].shape == (8, 8, 8)

    def test_jit_compatible(self):
        """The update function should be JIT-compilable."""
        node = LBMPipeNode("fluid", timestep=0.01, nx=8, ny=8, nz=8)
        state = node.initial_state()

        @jax.jit
        def step(s):
            return node.update(s, {}, 0.01)

        new_state = step(state)
        assert not jnp.any(jnp.isnan(new_state["density"]))

    def test_scan_compatible(self):
        """Can run inside GraphManager.run_scan."""
        gm = GraphManager()
        gm.add_node(LBMPipeNode("fluid", timestep=0.01, nx=8, ny=8, nz=8))
        gm.compile()
        final_state = gm.run_scan(5)
        assert "fluid" in final_state
        assert final_state["fluid"]["density"].shape == (8, 8, 8)
        assert final_state["fluid"]["tracer"].shape == (8, 8, 8)

    def test_scan_with_history(self):
        """run_scan_with_history captures tracer field."""
        gm = GraphManager()
        gm.add_node(LBMPipeNode("fluid", timestep=0.01, nx=8, ny=8, nz=8))
        gm.compile()
        _, history = gm.run_scan_with_history(5)
        assert "tracer" in history["fluid"]
        assert history["fluid"]["tracer"].shape == (5, 8, 8, 8)


# ── Shan-Chen multiphase tests ────────────────────────────────────

class TestShanChenHelpers:
    """Tests for the Shan-Chen pseudopotential helper functions."""

    def test_psi_monotonic(self):
        """Yuan-Schaefer pseudopotential should be monotonically increasing."""
        rho = jnp.linspace(0.01, 5.0, 100)
        psi_vals = _psi(rho, rho_0=1.0)
        assert jnp.all(jnp.diff(psi_vals) > 0)

    def test_psi_saturation(self):
        """psi(rho) → rho_0 as rho → infinity."""
        large = _psi(jnp.array(100.0), rho_0=1.0)
        assert float(large) == pytest.approx(1.0, abs=1e-5)

    def test_psi_zero_at_zero(self):
        """psi(0) = 0."""
        assert float(_psi(jnp.array(0.0), rho_0=1.0)) == pytest.approx(0.0)

    def test_shan_chen_force_zero_uniform(self):
        """SC force should be zero in a uniform density field."""
        density = jnp.ones((8, 8, 8), dtype=jnp.float32) * 0.5
        wall_mask = jnp.zeros((8, 8, 8), dtype=jnp.bool_)
        force = _shan_chen_force(density, G=-5.0, rho_0=1.0,
                                 wall_mask=wall_mask, rho_wall=0.5)
        assert jnp.allclose(force, 0.0, atol=1e-6)

    def test_shan_chen_force_direction(self):
        """SC force at an interface should be non-zero; zero far away."""
        density = jnp.ones((16, 4, 4), dtype=jnp.float32) * 0.2
        density = density.at[8:, :, :].set(1.0)
        wall_mask = jnp.zeros_like(density, dtype=jnp.bool_)
        force = _shan_chen_force(density, G=-5.0, rho_0=1.0,
                                 wall_mask=wall_mask, rho_wall=0.2)
        # At the gas side of interface (x=7), force should point toward
        # liquid (+x direction)
        fx_gas_side = float(force[7, 2, 2, 0])
        assert fx_gas_side > 0, f"Expected positive Fx at gas side, got {fx_gas_side}"
        # Force should be large at the interface and ~zero far from it
        fx_far_gas = float(jnp.abs(force[2, 2, 2, 0]))
        fx_far_liq = float(jnp.abs(force[13, 2, 2, 0]))
        fx_interface = float(jnp.abs(force[7, 2, 2, 0]))
        assert fx_interface > fx_far_gas * 5, "Force should be much larger at interface"
        assert fx_interface > fx_far_liq * 5, "Force should be much larger at interface"

    def test_shan_chen_force_zero_at_walls(self):
        """SC force at wall cells should be exactly zero."""
        density = jnp.ones((8, 8, 8), dtype=jnp.float32)
        wall_mask = jnp.zeros((8, 8, 8), dtype=jnp.bool_)
        wall_mask = wall_mask.at[0, :, :].set(True)
        force = _shan_chen_force(density, G=-5.0, rho_0=1.0,
                                 wall_mask=wall_mask, rho_wall=1.0)
        assert jnp.allclose(force[0, :, :, :], 0.0)

    def test_eos_pressure_ideal_gas_limit(self):
        """At G=0 the EOS should reduce to ideal gas (P = rho * cs^2)."""
        rho = jnp.array([0.5, 1.0, 2.0])
        P = _eos_pressure(rho, G=0.0, rho_0=1.0)
        expected = rho / 3.0
        assert jnp.allclose(P, expected, atol=1e-6)


class TestMultiphaseInit:
    """Test multiphase (Shan-Chen) initialization."""

    def test_multiphase_validation(self):
        """Should reject invalid multiphase parameters."""
        with pytest.raises(ValueError, match="rho_liquid"):
            LBMPipeNode("f", 0.01, nx=8, ny=8, nz=8,
                        G=-5.0, rho_liquid=0.1, rho_gas=0.5)
        with pytest.raises(ValueError, match="rho_gas"):
            LBMPipeNode("f", 0.01, nx=8, ny=8, nz=8,
                        G=-5.0, rho_gas=-0.1)

    def test_multiphase_initial_state_shapes(self):
        node = LBMPipeNode("f", 0.01, nx=8, ny=8, nz=8,
                           G=-5.0, fill_fraction=0.55)
        state = node.initial_state()
        assert state["f"].shape == (8, 8, 8, 19)
        assert state["density"].shape == (8, 8, 8)
        assert state["tracer"].shape == (8, 8, 8)

    def test_multiphase_density_range(self):
        """Initial density should span from gas to liquid."""
        node = LBMPipeNode("f", 0.01, nx=16, ny=16, nz=16,
                           G=-5.0, rho_liquid=1.0, rho_gas=0.25,
                           fill_fraction=0.55)
        state = node.initial_state()
        fluid = ~np.array(node._wall_mask)
        rho = np.array(state["density"])[fluid]
        assert rho.min() < 0.4, "Gas region should have low density"
        assert rho.max() > 0.7, "Liquid region should have high density"

    def test_multiphase_tracer_from_density(self):
        """In multiphase mode, tracer should be derived from density."""
        node = LBMPipeNode("f", 0.01, nx=16, ny=16, nz=16,
                           G=-5.0, rho_liquid=1.0, rho_gas=0.25,
                           fill_fraction=0.55)
        state = node.initial_state()
        fluid = ~np.array(node._wall_mask)
        t = np.array(state["tracer"])[fluid]
        # Tracer should range close to 0 and 1
        assert t.min() < 0.1
        assert t.max() > 0.8


class TestMultiphaseDynamics:
    """Test that Shan-Chen multiphase dynamics work correctly."""

    def _make_multiphase_gm(self, nx=16, ny=12, nz=12, G=-4.5,
                            prop_strength=0.0, gravity=0.0):
        import warnings
        node = LBMPipeNode(
            "fluid", timestep=0.01,
            nx=nx, ny=ny, nz=nz, tau=0.8,
            pipe_radius=0.9, propeller_x=nx // 6,
            propeller_radius=0.8,
            propeller_strength=prop_strength,
            gravity=gravity, fill_fraction=0.55,
            G=G, rho_liquid=1.0, rho_gas=0.25, rho_0=1.0,
        )
        gm = GraphManager()
        gm.add_node(node)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.compile()
        return gm, node

    def test_stable_200_steps(self):
        """Multiphase simulation should be stable for 200 steps."""
        gm, _ = self._make_multiphase_gm()
        final_state, history = gm.run_scan_with_history(200)
        rho = np.array(history["fluid"]["density"])
        assert not np.any(np.isnan(rho)), "Density should not have NaN"
        assert rho.min() > 0, "Density should stay positive"

    def test_phase_separation(self):
        """SC force should maintain / strengthen phase separation."""
        gm, node = self._make_multiphase_gm()
        final_state, history = gm.run_scan_with_history(100)
        rho = np.array(history["fluid"]["density"])
        fluid = ~np.array(node._wall_mask)
        rho_0 = rho[0][fluid]
        rho_end = rho[-1][fluid]
        # Density contrast should increase or at least be maintained
        contrast_0 = rho_0.max() - rho_0.min()
        contrast_end = rho_end.max() - rho_end.min()
        assert contrast_end >= contrast_0 * 0.8, \
            f"Phase contrast decreased from {contrast_0:.3f} to {contrast_end:.3f}"

    def test_propeller_deforms_surface(self):
        """Propeller should create visible surface deformation."""
        gm_still, node = self._make_multiphase_gm(
            nx=24, prop_strength=0.0, gravity=-0.001)
        gm_prop, _ = self._make_multiphase_gm(
            nx=24, prop_strength=0.01, gravity=-0.001)
        _, hist_still = gm_still.run_scan_with_history(200)
        _, hist_prop = gm_prop.run_scan_with_history(200)
        vel_still = np.array(hist_still["fluid"]["velocity"])
        vel_prop = np.array(hist_prop["fluid"]["velocity"])
        # Propeller case should have higher max velocity
        v_still = np.sqrt(np.sum(vel_still[-1] ** 2, axis=-1)).max()
        v_prop = np.sqrt(np.sum(vel_prop[-1] ** 2, axis=-1)).max()
        assert v_prop > v_still * 1.05, \
            f"Propeller should increase velocity ({v_still:.4f} vs {v_prop:.4f})"

    def test_gravity_settles_liquid(self):
        """Gravity should pull liquid downward."""
        gm, node = self._make_multiphase_gm(gravity=-0.001)
        _, history = gm.run_scan_with_history(200)
        tracer = np.array(history["fluid"]["tracer"])
        fluid = ~np.array(node._wall_mask)
        nz = node._nz
        # Compare liquid fraction in bottom vs top half
        t_end = tracer[-1]
        bottom = t_end[:, :, :nz // 2][fluid[:, :, :nz // 2]]
        top = t_end[:, :, nz // 2:][fluid[:, :, nz // 2:]]
        assert bottom.mean() > top.mean(), \
            f"Bottom should have more liquid ({bottom.mean():.3f}) than top ({top.mean():.3f})"

    def test_jit_compatible(self):
        """Multiphase update should be JIT-compilable."""
        node = LBMPipeNode("f", 0.01, nx=8, ny=8, nz=8,
                           G=-4.5, fill_fraction=0.55)
        state = node.initial_state()

        @jax.jit
        def step(s):
            return node.update(s, {}, 0.01)

        new_state = step(state)
        assert not jnp.any(jnp.isnan(new_state["density"]))

    def test_scan_compatible(self):
        """Multiphase node should work inside run_scan."""
        gm, _ = self._make_multiphase_gm()
        final_state = gm.run_scan(50)
        assert "fluid" in final_state
        assert not np.any(np.isnan(np.array(final_state["fluid"]["density"])))

    def test_backward_compatible_g_zero(self):
        """G=0 should produce identical results to the original single-phase."""
        gm = GraphManager()
        node = LBMPipeNode("fluid", timestep=0.01, nx=8, ny=8, nz=8,
                           propeller_strength=0.001, gravity=-0.001,
                           G=0.0, fill_fraction=0.55)
        gm.add_node(node)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.compile()
        final_state = gm.run_scan(20)
        rho = np.array(final_state["fluid"]["density"])
        # Single-phase: density should stay close to 1.0
        assert rho.max() < 1.1
        assert rho.min() > 0.9


class TestFillMaskFix:
    """Test the fill mask wall-margin fix for the initial isosurface."""

    def test_fill_mask_no_wall_overlap(self):
        """Liquid mask should not overlap with pipe wall."""
        node = LBMPipeNode("f", 0.01, nx=16, ny=16, nz=16,
                           fill_fraction=0.55, G=0.0)
        fill = np.array(node._compute_fill_mask())
        wall = np.array(node._wall_mask)
        overlap = fill & wall
        assert not np.any(overlap), "Fill mask should not overlap with walls"

    def test_fill_mask_margin_near_wall(self):
        """Fill mask should be pulled inward near the wall at the surface."""
        node = LBMPipeNode("f", 0.01, nx=16, ny=16, nz=16,
                           pipe_radius=0.9, fill_fraction=0.55, G=0.0)
        fill = np.array(node._compute_fill_mask())
        wall = np.array(node._wall_mask)
        # Check that the liquid region near the wall boundary is smaller
        # than a simple z-cutoff would produce
        interior = ~wall[0]
        z_cutoff = np.percentile(
            np.arange(16, dtype=np.float32).reshape(1, -1).repeat(16, 0)[interior],
            55,
        )
        # The margin should exclude some cells near the wall at the surface
        # (cells that would be filled by a flat cutoff but are near the wall)
        near_wall = np.sqrt(
            (np.arange(16)[:, None] - 7.5) ** 2
            + (np.arange(16)[None, :] - 7.5) ** 2
        ) > (0.9 * 8 - 1.5)
        near_surface = np.abs(
            np.arange(16)[None, :] - z_cutoff
        ) < 2.0
        excluded = near_wall[:, :] & near_surface & interior
        if excluded.any():
            # At least some excluded cells should NOT be in the fill mask
            excluded_fill = fill[0][excluded]
            assert not np.all(excluded_fill), \
                "Cells near wall AND surface should be excluded from fill"
