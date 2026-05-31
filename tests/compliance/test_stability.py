"""Tests for the @stability decorator and generate_stability_report()."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import (
    stability,
    generate_stability_report,
    _STABILITY_REGISTRY,
)


class TestStabilityDecorator:
    def test_decorator_returns_original_class(self):
        @stability(StabilityLevel.STABLE)
        class MyClass:
            pass

        assert MyClass.__name__ == "MyClass"
        assert MyClass._stability_level == StabilityLevel.STABLE

    def test_decorator_returns_original_function(self):
        @stability(StabilityLevel.EXPERIMENTAL)
        def my_func():
            return 42

        assert my_func() == 42
        assert my_func._stability_level == StabilityLevel.EXPERIMENTAL

    def test_decorator_registers_in_registry(self):
        @stability(StabilityLevel.PROVISIONAL)
        class RegisteredClass:
            pass

        # Should be in the registry
        found = False
        for name, level in _STABILITY_REGISTRY.items():
            if "RegisteredClass" in name:
                found = True
                assert level == StabilityLevel.PROVISIONAL
                break
        assert found, "RegisteredClass not found in registry"

    def test_different_levels(self):
        @stability(StabilityLevel.DEPRECATED)
        class OldClass:
            pass

        assert OldClass._stability_level == StabilityLevel.DEPRECATED

    def test_evolving_level_v030(self):
        """EVOLVING was added in v0.3.0 (per plans/MADDENING_v0.3.0_PLAN.md §A2)."""
        @stability(StabilityLevel.EVOLVING)
        class GrowingClass:
            pass

        assert GrowingClass._stability_level == StabilityLevel.EVOLVING
        assert StabilityLevel.EVOLVING.value == "evolving"

    def test_internal_level_v030(self):
        """INTERNAL was added in v0.3.0 (per plans/MADDENING_v0.3.0_PLAN.md §A2)."""
        @stability(StabilityLevel.INTERNAL)
        class InternalClass:
            pass

        assert InternalClass._stability_level == StabilityLevel.INTERNAL
        assert StabilityLevel.INTERNAL.value == "internal"

    def test_v030_first_wave_tagged(self):
        """The v0.3.0 §A2 first wave of API surfaces is tagged.

        Reads the registry and asserts each surface the plan names is
        present at the expected level.  If the registry doesn't contain
        a surface, this test fails — surfacing untagged additions
        immediately rather than at v0.4.0 / 1.0.0 freeze time.
        """
        # Import them so the @stability decorators fire.
        import maddening.core.graph_manager  # noqa: F401
        import maddening.core.node  # noqa: F401
        import maddening.core.edge  # noqa: F401
        import maddening.core.static_data  # noqa: F401
        import maddening.cloud.multigpu.sharded_node  # noqa: F401
        import maddening.api.binary_encoder  # noqa: F401
        import maddening.cloud.providers  # noqa: F401

        stable_required = {
            "maddening.core.graph_manager.GraphManager",
            "maddening.core.node.SimulationNode",
            "maddening.core.edge.EdgeSpec",
            "maddening.core.static_data.StaticArray",
            "maddening.cloud.multigpu.sharded_node.ShardedPointwiseNode",
            "maddening.cloud.multigpu.sharded_node.ShardedStencilNode",
        }
        evolving_required = {
            "maddening.api.binary_encoder.BinaryStateEncoder",
            "maddening.cloud.providers.CloudProvider",
        }

        for name in stable_required:
            assert name in _STABILITY_REGISTRY, f"{name} missing from registry"
            assert _STABILITY_REGISTRY[name] == StabilityLevel.STABLE, (
                f"{name} is {_STABILITY_REGISTRY[name]}, expected STABLE"
            )
        for name in evolving_required:
            assert name in _STABILITY_REGISTRY, f"{name} missing from registry"
            assert _STABILITY_REGISTRY[name] == StabilityLevel.EVOLVING, (
                f"{name} is {_STABILITY_REGISTRY[name]}, expected EVOLVING"
            )


class TestStabilityReport:
    def test_report_is_markdown(self):
        report = generate_stability_report()
        assert isinstance(report, str)
        assert "# Stability Report" in report

    def test_report_contains_registered_items(self):
        @stability(StabilityLevel.STABLE)
        class ReportTestClass:
            pass

        report = generate_stability_report()
        assert "ReportTestClass" in report
        assert "stable" in report
