"""Tests for NodeMeta, StabilityLevel, UQReadiness, and related schema types."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from maddening.core.compliance.metadata import (
    NodeMeta, EdgeMeta, ValidatedRegime, Reference,
    StabilityLevel, UQReadiness,
    collect_node_metadata, collect_hazard_hints,
)


class TestStabilityLevel:
    def test_values(self):
        assert StabilityLevel.EXPERIMENTAL.value == "experimental"
        assert StabilityLevel.PROVISIONAL.value == "provisional"
        assert StabilityLevel.EVOLVING.value == "evolving"
        assert StabilityLevel.STABLE.value == "stable"
        assert StabilityLevel.INTERNAL.value == "internal"
        assert StabilityLevel.DEPRECATED.value == "deprecated"

    def test_all_six_exist(self):
        # v0.3.0 added EVOLVING + INTERNAL on top of the original four
        # (EXPERIMENTAL, PROVISIONAL, STABLE, DEPRECATED).  See
        # plans/MADDENING_v0.3.0_PLAN.md §A2.
        assert len(StabilityLevel) == 6


class TestUQReadiness:
    def test_values(self):
        assert UQReadiness.NOT_READY.value == "not_ready"
        assert UQReadiness.PARAMETER_SWEEP.value == "parameter_sweep"
        assert UQReadiness.FULL.value == "full"

    def test_all_three_exist(self):
        assert len(UQReadiness) == 3


class TestNodeMeta:
    def test_default_construction(self):
        meta = NodeMeta()
        assert meta.algorithm_id == ""
        assert meta.stability == StabilityLevel.EXPERIMENTAL
        assert meta.hazard_hints == ()
        assert meta.implementation_map == {}

    def test_full_construction(self):
        meta = NodeMeta(
            algorithm_id="MADD-NODE-TEST",
            algorithm_version="1.0.0",
            stability=StabilityLevel.STABLE,
            description="Test node",
            governing_equations="x = x + v*dt",
            discretization="Forward Euler",
            assumptions=("Point mass",),
            limitations=("1st order only",),
            validated_regimes=(
                ValidatedRegime("dt", 0.001, 0.1, "s"),
            ),
            references=(
                Reference("Author2024", "A paper"),
            ),
            uq_readiness=UQReadiness.PARAMETER_SWEEP,
            hazard_hints=("Energy drift at large dt",),
            implementation_map={"x + v*dt": "module.Class.method"},
        )
        assert meta.algorithm_id == "MADD-NODE-TEST"
        assert meta.stability == StabilityLevel.STABLE
        assert len(meta.assumptions) == 1
        assert len(meta.hazard_hints) == 1
        assert "x + v*dt" in meta.implementation_map

    def test_frozen(self):
        meta = NodeMeta(algorithm_id="test")
        with pytest.raises(AttributeError):
            meta.algorithm_id = "changed"

    def test_implementation_map_is_mutable_dict(self):
        """implementation_map uses field(default_factory=dict), so each
        instance gets its own dict even though the dataclass is frozen."""
        m1 = NodeMeta()
        m2 = NodeMeta()
        # They should be separate instances
        assert m1.implementation_map is not m2.implementation_map


class TestValidatedRegime:
    def test_construction(self):
        vr = ValidatedRegime("Re", 0, 100, notes="Laminar only")
        assert vr.parameter == "Re"
        assert vr.min_value == 0
        assert vr.max_value == 100
        assert vr.notes == "Laminar only"


class TestEdgeMeta:
    def test_construction(self):
        em = EdgeMeta(description="Position coupling", units="m")
        assert em.description == "Position coupling"
        assert em.units == "m"


class TestReference:
    def test_construction(self):
        ref = Reference("Crank1975", "The Mathematics of Diffusion")
        assert ref.key == "Crank1975"


class TestHarvesting:
    def test_collect_node_metadata_returns_dict(self):
        """At minimum, our built-in nodes should appear."""
        # Import nodes to ensure they're registered as subclasses
        import maddening.nodes  # noqa: F401
        result = collect_node_metadata()
        assert isinstance(result, dict)
        # Should find at least BallNode since it has meta
        assert "BallNode" in result
        assert result["BallNode"].algorithm_id == "MADD-NODE-001"

    def test_collect_hazard_hints(self):
        import maddening.nodes  # noqa: F401
        result = collect_hazard_hints()
        assert isinstance(result, dict)
        assert "HeatNode" in result
        assert any("CFL" in h for h in result["HeatNode"])

    def test_all_builtin_nodes_have_meta(self):
        """Every built-in SimulationNode subclass must have meta."""
        import maddening.nodes  # noqa: F401
        from maddening.core.node import SimulationNode
        for cls in SimulationNode.__subclasses__():
            # Skip SurrogateNode and test-only helper nodes
            if cls.__module__.startswith("maddening.nodes") is False:
                continue
            meta = getattr(cls, "meta", None)
            assert meta is not None, f"{cls.__name__} missing meta ClassVar"
            assert meta.algorithm_id, f"{cls.__name__} has empty algorithm_id"
            assert meta.description, f"{cls.__name__} has empty description"
