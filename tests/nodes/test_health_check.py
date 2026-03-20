"""Tests for HealthCheckNode."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from maddening.nodes.health_check import HealthCheckNode
from maddening.core.compliance.metadata import StabilityLevel


class TestHealthCheckNodeMeta:
    def test_has_meta(self):
        assert HealthCheckNode.meta is not None

    def test_algorithm_id(self):
        assert HealthCheckNode.meta.algorithm_id == "MADD-NODE-HEALTH-001"

    def test_stability(self):
        assert HealthCheckNode.meta.stability == StabilityLevel.EXPERIMENTAL

    def test_has_hazard_hints(self):
        assert len(HealthCheckNode.meta.hazard_hints) == 3

    def test_algorithmic_diversity_hint(self):
        """First hazard hint must mention Algorithmic Diversity Principle."""
        assert "Algorithmic Diversity" in HealthCheckNode.meta.hazard_hints[0]


class TestHealthCheckNodeBasic:
    def test_initial_state(self):
        node = HealthCheckNode("hc", 0.01, checks={"density": {"finite": True}})
        state = node.initial_state()
        assert "all_passed" in state
        assert "n_checks" in state
        assert state["n_checks"] == 1

    def test_empty_checks(self):
        node = HealthCheckNode("hc", 0.01)
        state = node.initial_state()
        assert state["n_checks"] == 0

    def test_update_no_checks(self):
        node = HealthCheckNode("hc", 0.01)
        state = node.initial_state()
        new_state = node.update(state, {}, 0.01)
        assert bool(new_state["all_passed"])


class TestHealthCheckNaN:
    """The most critical test: detecting NaN in monitored fields."""

    def test_detects_nan(self):
        node = HealthCheckNode("hc", 0.01, checks={
            "temperature": {"finite": True},
        })
        state = node.initial_state()

        # Pass NaN data
        nan_data = jnp.array([1.0, float("nan"), 3.0])
        new_state = node.update(state, {"temperature": nan_data}, 0.01)

        assert not bool(new_state["all_passed"])
        assert int(new_state["n_failed"]) == 1

    def test_detects_inf(self):
        node = HealthCheckNode("hc", 0.01, checks={
            "velocity": {"finite": True},
        })
        state = node.initial_state()

        inf_data = jnp.array([1.0, float("inf"), -1.0])
        new_state = node.update(state, {"velocity": inf_data}, 0.01)

        assert not bool(new_state["all_passed"])

    def test_passes_clean_data(self):
        node = HealthCheckNode("hc", 0.01, checks={
            "temperature": {"finite": True},
        })
        state = node.initial_state()

        clean_data = jnp.array([1.0, 2.0, 3.0])
        new_state = node.update(state, {"temperature": clean_data}, 0.01)

        assert bool(new_state["all_passed"])
        assert int(new_state["n_passed"]) == 1
        assert int(new_state["n_failed"]) == 0


class TestHealthCheckBounds:
    def test_min_violation(self):
        node = HealthCheckNode("hc", 0.01, checks={
            "density": {"min": 0.0},
        })
        state = node.initial_state()

        bad_data = jnp.array([1.0, -0.5, 2.0])  # -0.5 violates min=0
        new_state = node.update(state, {"density": bad_data}, 0.01)

        assert not bool(new_state["all_passed"])

    def test_max_violation(self):
        node = HealthCheckNode("hc", 0.01, checks={
            "density": {"max": 10.0},
        })
        state = node.initial_state()

        bad_data = jnp.array([1.0, 15.0, 2.0])  # 15 > max=10
        new_state = node.update(state, {"density": bad_data}, 0.01)

        assert not bool(new_state["all_passed"])

    def test_max_abs_violation(self):
        node = HealthCheckNode("hc", 0.01, checks={
            "velocity": {"max_abs": 5.0},
        })
        state = node.initial_state()

        bad_data = jnp.array([1.0, -6.0, 2.0])  # |-6| > 5
        new_state = node.update(state, {"velocity": bad_data}, 0.01)

        assert not bool(new_state["all_passed"])

    def test_within_bounds_passes(self):
        node = HealthCheckNode("hc", 0.01, checks={
            "density": {"min": 0.0, "max": 10.0},
        })
        state = node.initial_state()

        good_data = jnp.array([1.0, 5.0, 9.0])
        new_state = node.update(state, {"density": good_data}, 0.01)

        assert bool(new_state["all_passed"])


class TestHealthCheckMultipleFields:
    def test_multiple_checks(self):
        node = HealthCheckNode("hc", 0.01, checks={
            "density": {"finite": True, "min": 0.0},
            "velocity": {"finite": True, "max_abs": 100.0},
        })
        state = node.initial_state()
        assert state["n_checks"] == 2

        new_state = node.update(state, {
            "density": jnp.array([1.0, 2.0]),
            "velocity": jnp.array([10.0, -20.0]),
        }, 0.01)

        assert bool(new_state["all_passed"])
        assert int(new_state["n_passed"]) == 2

    def test_one_fails_of_two(self):
        node = HealthCheckNode("hc", 0.01, checks={
            "density": {"finite": True},
            "velocity": {"finite": True},
        })
        state = node.initial_state()

        new_state = node.update(state, {
            "density": jnp.array([1.0, 2.0]),
            "velocity": jnp.array([float("nan")]),
        }, 0.01)

        assert not bool(new_state["all_passed"])
        assert int(new_state["n_passed"]) == 1
        assert int(new_state["n_failed"]) == 1


class TestHealthCheckDoesNotHalt:
    """HealthCheckNode must not halt — it only reports."""

    def test_returns_state_dict(self):
        """update() always returns a state dict, never raises."""
        node = HealthCheckNode("hc", 0.01, checks={
            "x": {"finite": True, "min": 0, "max": 1},
        })
        state = node.initial_state()

        # Even with NaN, Inf, and out-of-bounds, update() returns normally
        bad_data = jnp.array([float("nan"), float("inf"), -100.0])
        new_state = node.update(state, {"x": bad_data}, 0.01)

        assert isinstance(new_state, dict)
        assert "all_passed" in new_state
        assert not bool(new_state["all_passed"])
