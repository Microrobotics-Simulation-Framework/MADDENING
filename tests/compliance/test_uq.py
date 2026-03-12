"""Tests for UncertaintySpec and UncertainParameter."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from maddening.core.uq import (
    UncertaintySpec,
    UncertainParameter,
    DistributionType,
)


class TestUncertainParameter:
    def test_construction(self):
        up = UncertainParameter(
            name="thermal_diffusivity",
            distribution=DistributionType.UNIFORM,
            nominal=0.01,
            lower_bound=0.005,
            upper_bound=0.02,
            units="m^2/s",
            description="Thermal diffusivity",
        )
        assert up.name == "thermal_diffusivity"
        assert up.distribution == DistributionType.UNIFORM
        assert up.nominal == 0.01

    def test_defaults(self):
        up = UncertainParameter(name="k")
        assert up.distribution == DistributionType.UNIFORM
        assert up.nominal == 0.0
        assert up.lower_bound is None


class TestUncertaintySpec:
    def test_empty(self):
        spec = UncertaintySpec()
        assert spec.parameters == ()
        assert spec.notes == ""

    def test_with_parameters(self):
        params = (
            UncertainParameter(name="k", nominal=100.0),
            UncertainParameter(name="c", nominal=1.0),
        )
        spec = UncertaintySpec(parameters=params, notes="Basic UQ")
        assert len(spec.parameters) == 2
        assert spec.notes == "Basic UQ"


class TestDistributionType:
    def test_all_types(self):
        assert DistributionType.UNIFORM.value == "uniform"
        assert DistributionType.NORMAL.value == "normal"
        assert DistributionType.LOG_NORMAL.value == "log_normal"
        assert DistributionType.TRUNCATED_NORMAL.value == "truncated_normal"


class TestNodeUQInterface:
    def test_default_uncertainty_spec_is_none(self):
        """SimulationNode.uncertainty_spec() returns None by default."""
        from maddening.nodes.ball import BallNode
        node = BallNode("test", 0.01)
        assert node.uncertainty_spec() is None
