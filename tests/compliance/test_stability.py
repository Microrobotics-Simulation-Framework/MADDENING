"""Tests for the @stability decorator and generate_stability_report()."""

import ast
import difflib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

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


_DECORATOR_LINE = re.compile(r"^\s*@stability\(", re.M)


def applies_stability(source: str) -> bool:
    """Does this module source apply ``stability`` to anything, in any form?

    A regex for ``^\\s*@stability(`` saw the decorator form only, so a
    surface tagged ``fn = stability(StabilityLevel.STABLE)(fn)`` in a module
    missing from the generator's list was invisible to the coverage test
    below, and to the report (audit_040_phase3_wave_d, R3).  This walks the
    AST instead and counts any *call* of the decorator -- ``@stability(...)``
    is a call in the decorator list, and the call form is the same node:

    * the bare name ``stability``, or any name an import or an assignment
      binds to it (``import stability as tag``, ``tag = stability``);
    * any attribute ``<x>.stability(...)`` (``compliance.stability``,
      ``import ...stability as S; S.stability``).

    Not seen: a decorator reached through ``getattr`` or a container.  A
    ``@stability(`` in a docstring is prose and does not count; a module
    that does not parse falls back to the old regex rather than to "no".
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return bool(_DECORATOR_LINE.search(source))
    names = {"stability"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "stability" and alias.asname:
                    names.add(alias.asname)
    for _ in range(3):                  # rebinding chains are short
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in names):
                names.update(t.id for t in node.targets
                             if isinstance(t, ast.Name))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in names:
            return True
        if isinstance(func, ast.Attribute) and func.attr == "stability":
            return True
    return False


class TestApplyStabilityDetection:
    """The coverage test is only as good as its detector; pin every form."""

    @pytest.mark.parametrize("source", [
        "@stability(StabilityLevel.STABLE)\ndef f():\n    pass\n",
        "def f():\n    pass\nf = stability(StabilityLevel.STABLE)(f)\n",
        "from maddening.core.compliance.stability import stability as tag\n"
        "@tag(StabilityLevel.EVOLVING)\ndef f():\n    pass\n",
        "tag = stability\n@tag(StabilityLevel.EVOLVING)\nclass C:\n    pass\n",
        "import maddening.core.compliance.stability as S\n"
        "f = S.stability(StabilityLevel.STABLE)(lambda: 1)\n",
    ])
    def test_every_way_of_applying_the_decorator_is_seen(self, source):
        assert applies_stability(source)

    @pytest.mark.parametrize("source", [
        '"""Example::\n\n    @stability(StabilityLevel.STABLE)\n"""\n',
        "from maddening.core.compliance.stability import stability\n",
        "def f():\n    pass\n",
    ])
    def test_a_mention_that_applies_nothing_is_not_counted(self, source):
        assert not applies_stability(source)


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
        found = set()
        for path in (self.SRC / "maddening").rglob("*.py"):
            if applies_stability(path.read_text(encoding="utf-8")):
                rel = path.relative_to(self.SRC).with_suffix("")
                parts = list(rel.parts)
                if parts[-1] == "__init__":
                    parts.pop()
                found.add(".".join(parts))
        return found

    def _listed_modules(self) -> set[str]:
        """``STABILITY_MODULES`` read statically (no import, no optional deps)."""
        tree = ast.parse(self.SCRIPT.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "STABILITY_MODULES":
                return set(ast.literal_eval(node.value))
        raise AssertionError("STABILITY_MODULES not found in the generator")

    def test_generator_module_list_covers_every_module_using_stability(self):
        """Static check: every module under src/maddening that applies
        ``stability`` -- as a decorator or as a call -- is listed in the
        generator, so a newly tagged module cannot silently drop out of the
        report.  Static, so it holds in environments without every optional
        extra."""
        tagged = self._modules_using_stability()
        assert "maddening.cloud.resume" in tagged  # sanity: the scan sees the tag
        # An empty or near-empty scan passes the subset check below
        # vacuously; 57 modules apply the decorator today.
        assert len(tagged) >= 50, sorted(tagged)
        listed = self._listed_modules()
        # Exact membership, deliberately not a prefix match.  The generator
        # imports exactly the names in STABILITY_MODULES, and importing
        # maddening.nodes does not import maddening.nodes.ball, so a listed
        # parent never makes a tagged submodule reachable.  An earlier version
        # accepted "one of its parent packages", and because "maddening" is
        # itself listed every module matched: the assertion could not fail, and
        # 25 tagged modules -- including the whole api.auth surface -- were
        # absent from the report while this test passed.
        missing = sorted(m for m in tagged if m not in listed)
        assert not missing, (
            "modules using @stability that scripts/generate_stability_report.py "
            f"never imports (add them to STABILITY_MODULES): {missing}"
        )

    def test_generator_imports_every_listed_module_or_records_why(self):
        """Loading the generator in a fresh interpreter imports every listed
        module, except those whose optional dependency is absent, which are
        recorded in SKIPPED_MODULES with the reason (never silently dropped)."""
        code = (
            "import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('gen', {str(self.SCRIPT)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "imported = sorted(m for m in sys.modules if m.startswith('maddening'))\n"
            "print(repr((imported, dict(mod.SKIPPED_MODULES))))\n"
        )
        env = dict(os.environ, JAX_PLATFORMS="cpu")
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr[-3000:]
        imported, skipped = ast.literal_eval(out.stdout.strip().splitlines()[-1])
        imported = set(imported)
        for name in self._listed_modules():
            assert name in imported or name in skipped, name
        # a skip is only ever for an optional third-party dependency; the
        # generator re-raises when the missing module is one of ours
        for name, why in skipped.items():
            assert why, name

    def test_generator_refuses_an_incomplete_report_unless_told_otherwise(self, tmp_path, monkeypatch):
        """With a module skipped for a missing optional dependency, ``main()``
        exits 2 and writes nothing unless ``--allow-missing-optional``."""
        code = (
            "import importlib.util, sys, pathlib\n"
            f"spec = importlib.util.spec_from_file_location('gen', {str(self.SCRIPT)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "mod.SKIPPED_MODULES['maddening.fake'] = \"No module named 'fakedep'\"\n"
            f"mod.REPO_ROOT = pathlib.Path({str(tmp_path)!r})\n"
            "rc1 = mod.main([])\n"
            "exists1 = (mod.REPO_ROOT / 'docs' / 'developer_guide' / 'stability_report.md').exists()\n"
            "rc2 = mod.main(['--allow-missing-optional'])\n"
            "exists2 = (mod.REPO_ROOT / 'docs' / 'developer_guide' / 'stability_report.md').exists()\n"
            "print(repr((rc1, exists1, rc2, exists2)))\n"
        )
        env = dict(os.environ, JAX_PLATFORMS="cpu")
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr[-3000:]
        assert ast.literal_eval(out.stdout.strip().splitlines()[-1]) == (2, False, 0, True)
        assert "skipped maddening.fake" in out.stderr

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

    def test_generator_lists_the_adaptive_node_surfaces(self):
        """The adaptive package is not pulled in by ``maddening.nodes``, and
        ``solver_utils`` by nothing else, so both must be listed explicitly or
        their tags never reach the report (audit A6)."""
        listed = self._listed_modules()
        for module in ("maddening.nodes.adaptive", "maddening.core.solver_utils"):
            assert module in listed, (
                f"{module} is missing from STABILITY_MODULES, so its "
                "@stability-decorated surfaces never reach the report"
            )

    def test_the_committed_report_is_what_the_script_emits(self, tmp_path):
        """``docs/developer_guide/stability_report.md`` must equal a fresh run.

        The report is a generated artifact that the file's own header
        promises is "refreshed at release time".  Nothing checked that, so
        it silently rotted for two releases: at the 0.4.0 audit the
        committed file listed 42 surfaces against a registry of 85 -- half
        the public API missing, including both ``deprecated`` entries, so a
        release that deprecated four things published a stability contract
        showing none.  The coverage test above only checks that the
        generator *reaches* every tagged module; a generated artifact with
        no equality check is a green check that cannot fail.

        The generator runs in a fresh interpreter (in-process imports from
        other tests would otherwise seed the registry) and writes into a
        temporary tree, so the test never touches the working copy.
        """
        code = (
            "import importlib.util, sys, pathlib\n"
            f"spec = importlib.util.spec_from_file_location('gen', {str(self.SCRIPT)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            f"mod.REPO_ROOT = pathlib.Path({str(tmp_path)!r})\n"
            "rc = mod.main([])\n"
            "print(repr((rc, dict(mod.SKIPPED_MODULES))))\n"
        )
        env = dict(os.environ, JAX_PLATFORMS="cpu")
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True, env=env)
        assert out.returncode == 0, out.stderr[-3000:]
        rc, skipped = ast.literal_eval(out.stdout.strip().splitlines()[-1])
        if skipped:
            # ``main`` refuses to write an incomplete report, and comparing
            # against one would be meaningless.  The release gate and CI
            # install every extra; this only fires on a partial install.
            pytest.skip(
                "stability report cannot be regenerated here: optional "
                f"dependencies missing for {sorted(skipped)}"
            )
        assert rc == 0, out.stderr[-3000:]

        fresh = (tmp_path / "docs" / "developer_guide" / "stability_report.md")
        committed = self.REPO_ROOT / "docs" / "developer_guide" / "stability_report.md"
        diff = list(difflib.unified_diff(
            committed.read_text(encoding="utf-8").splitlines(),
            fresh.read_text(encoding="utf-8").splitlines(),
            fromfile="committed", tofile="generated", lineterm="", n=1,
        ))
        assert not diff, (
            "docs/developer_guide/stability_report.md is stale; regenerate it "
            "with `python scripts/generate_stability_report.py` and commit the "
            "result.\n" + "\n".join(diff[:60])
        )

    def test_the_adaptive_surfaces_reach_the_generated_report(self):
        import importlib

        importlib.import_module("maddening.nodes.adaptive")
        importlib.import_module("maddening.core.solver_utils")
        report = generate_stability_report()
        for name in (
            "maddening.nodes.adaptive.base.AdaptiveNode",
            "maddening.nodes.adaptive.base.AdaptiveNodeBlindnessError",
            "maddening.core.solver_utils.ift_linear_solve",
        ):
            assert name in report, name
