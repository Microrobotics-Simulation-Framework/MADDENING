"""Tests for the @stability decorator and generate_stability_report()."""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

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
        import maddening.cloud.resume  # noqa: F401

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
            # v0.4.0: the resume-from-URL transport moved to the cloud package
            # and was tagged there (it had no tag in the core module).
            "maddening.cloud.resume.download_and_load_state",
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


class TestStabilityReportGeneratorCoverage:
    """``scripts/generate_stability_report.py`` must reach every tagged module.

    The registry is populated at import time, so a ``@stability``-tagged
    module the generator never imports silently drops out of the report
    (the v0.4.0 ``maddening.cloud.resume`` tag went missing that way).
    The generator's module list is compared against a grep of the source
    tree in a fresh interpreter so in-process imports from other tests
    cannot mask a gap.
    """

    REPO_ROOT = Path(__file__).resolve().parents[2]
    SCRIPT = REPO_ROOT / "scripts" / "generate_stability_report.py"
    SRC = REPO_ROOT / "src"

    def _modules_using_stability(self) -> set[str]:
        pattern = re.compile(r"^\s*@stability\(", re.M)
        found = set()
        for path in (self.SRC / "maddening").rglob("*.py"):
            if pattern.search(path.read_text(encoding="utf-8")):
                rel = path.relative_to(self.SRC).with_suffix("")
                parts = list(rel.parts)
                if parts[-1] == "__init__":
                    parts.pop()
                found.add(".".join(parts))
        return found

    def test_generator_module_list_covers_every_module_using_stability(self):
        tagged = self._modules_using_stability()
        assert "maddening.cloud.resume" in tagged  # sanity: the grep sees the tag
        code = (
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('gen', {str(self.SCRIPT)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "mod.import_stability_surfaces()\n"
            "print(repr(sorted(m for m in sys.modules if m.startswith('maddening'))))\n"
        )
        env = dict(os.environ, JAX_PLATFORMS="cpu")
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True, env=env,
        )
        imported = set(ast.literal_eval(out.stdout.strip().splitlines()[-1]))
        missing = sorted(tagged - imported)
        assert not missing, (
            "modules using @stability that scripts/generate_stability_report.py "
            f"never imports (add them to STABILITY_MODULES): {missing}"
        )

    def test_resume_transport_is_registered_evolving_by_generator(self):
        code = (
            "import importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('gen', {str(self.SCRIPT)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "from maddening.core.compliance.stability import _STABILITY_REGISTRY\n"
            "print(_STABILITY_REGISTRY['maddening.cloud.resume.download_and_load_state'].value)\n"
        )
        env = dict(os.environ, JAX_PLATFORMS="cpu")
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, check=True, env=env,
        )
        assert out.stdout.strip().splitlines()[-1] == "evolving"
