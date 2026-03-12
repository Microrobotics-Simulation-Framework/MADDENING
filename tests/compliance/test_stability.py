"""Tests for the @stability decorator and generate_stability_report()."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from maddening.core.metadata import StabilityLevel
from maddening.core.stability import (
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
