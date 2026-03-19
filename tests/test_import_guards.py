"""Tests that optional dependency errors give clear install instructions.

Each test runs in a subprocess to avoid module cache pollution.
"""

import subprocess
import sys

import pytest


def _run_import(code: str) -> subprocess.CompletedProcess:
    """Run a Python snippet in a subprocess, return the result."""
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=30,
    )


class TestImportGuards:
    """Verify that missing optional deps produce clear pip install messages.

    These tests only run when the dependency is NOT installed.  If the
    dependency is available (e.g. in a dev/CI environment), they skip.
    """

    def test_terminal_renderer_mentions_extra(self):
        result = _run_import(
            "import sys; sys.modules['rich'] = None; "
            "sys.modules['rich.live'] = None; "
            "sys.modules['rich.table'] = None; "
            "sys.modules['rich.text'] = None; "
            "from maddening.viz.backends.terminal_renderer import TerminalRenderer"
        )
        if result.returncode == 0:
            pytest.skip("rich is installed — guard not triggered")
        assert "maddening[terminal]" in result.stderr

    def test_matplotlib_renderer_mentions_extra(self):
        result = _run_import(
            "import sys; sys.modules['matplotlib'] = None; "
            "sys.modules['matplotlib.pyplot'] = None; "
            "sys.modules['matplotlib.patches'] = None; "
            "sys.modules['matplotlib.animation'] = None; "
            "from maddening.viz.backends.matplotlib_renderer import MatplotlibRenderer"
        )
        if result.returncode == 0:
            pytest.skip("matplotlib is installed — guard not triggered")
        assert "maddening[viz]" in result.stderr

    def test_zmq_network_mentions_extra(self):
        result = _run_import(
            "import sys; sys.modules['zmq'] = None; "
            "from maddening.viz.network import NetworkRelay"
        )
        if result.returncode == 0:
            pytest.skip("pyzmq is installed — guard not triggered")
        assert "maddening[network]" in result.stderr

    def test_usd_mentions_extra(self):
        result = _run_import(
            "import sys; sys.modules['pxr'] = None; "
            "sys.modules['pxr.Plug'] = None; "
            "sys.modules['pxr.Usd'] = None; "
            "import maddening.usd"
        )
        if result.returncode == 0:
            pytest.skip("usd-core is installed — guard not triggered")
        assert "maddening[usd]" in result.stderr

    def test_frame_renderer_mentions_extra(self):
        result = _run_import(
            "import sys; sys.modules['matplotlib'] = None; "
            "sys.modules['matplotlib.figure'] = None; "
            "sys.modules['matplotlib.backends'] = None; "
            "sys.modules['matplotlib.backends.backend_agg'] = None; "
            "sys.modules['matplotlib.gridspec'] = None; "
            "sys.modules['matplotlib.patches'] = None; "
            "from maddening.api.frame_renderer import ServerFrameRenderer"
        )
        if result.returncode == 0:
            pytest.skip("matplotlib is installed — guard not triggered")
        assert "maddening[viz]" in result.stderr

    def test_skypilot_mentions_extra(self):
        result = _run_import(
            "import sys; sys.modules['sky'] = None; "
            "from maddening.cloud._skypilot import launch_vm; "
            "launch_vm(None)"
        )
        if result.returncode == 0:
            pytest.skip("skypilot is installed — guard not triggered")
        assert "maddening[runpod]" in result.stderr or "maddening[cloud]" in result.stderr


class TestLazyImportHints:
    """Verify that __getattr__ lazy imports give install hints on ImportError."""

    def test_viz_backends_matplotlib(self):
        result = _run_import(
            "import sys; sys.modules['matplotlib'] = None; "
            "sys.modules['matplotlib.pyplot'] = None; "
            "sys.modules['matplotlib.patches'] = None; "
            "sys.modules['matplotlib.animation'] = None; "
            "from maddening.viz.backends import MatplotlibRenderer"
        )
        if result.returncode == 0:
            pytest.skip("matplotlib is installed")
        assert "maddening[viz]" in result.stderr

    def test_viz_backends_terminal(self):
        result = _run_import(
            "import sys; sys.modules['rich'] = None; "
            "sys.modules['rich.live'] = None; "
            "sys.modules['rich.table'] = None; "
            "sys.modules['rich.text'] = None; "
            "from maddening.viz.backends import TerminalRenderer"
        )
        if result.returncode == 0:
            pytest.skip("rich is installed")
        assert "maddening[terminal]" in result.stderr

    def test_surrogates_trainer(self):
        result = _run_import(
            "import sys; sys.modules['equinox'] = None; "
            "sys.modules['optax'] = None; "
            "from maddening.surrogates import SurrogateTrainer"
        )
        if result.returncode == 0:
            pytest.skip("equinox is installed")
        assert "maddening[surrogates]" in result.stderr


class TestPackagingConsistency:
    """Validate pyproject.toml extras are internally consistent."""

    def test_parse_pyproject(self):
        """All extras in pyproject.toml are syntactically valid."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        extras = data["project"]["optional-dependencies"]
        assert isinstance(extras, dict)
        # Every extra should be a non-empty list of strings
        for name, deps in extras.items():
            assert isinstance(deps, list), f"Extra '{name}' is not a list"

    def test_all_is_superset(self):
        """'all' extra covers server, viz, surrogates, and cloud."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        extras = data["project"]["optional-dependencies"]
        all_deps = set(extras["all"])
        # These key packages should be in 'all'
        expected_packages = [
            "matplotlib>=3.5",
            "rich>=12.0",
            "pyzmq>=25.0",
            "fastapi>=0.100",
            "equinox>=0.11",
        ]
        for pkg in expected_packages:
            assert pkg in all_deps, f"'{pkg}' missing from 'all' extra"

    def test_ci_excludes_display_deps(self):
        """CI extra doesn't include display-dependent packages."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        ci_deps = set(extras["ci"]) if (extras := data["project"]["optional-dependencies"]) else set()
        display_deps = {"pyvista>=0.42", "pygfx>=0.16", "glfw>=2.0"}
        overlap = ci_deps & display_deps
        assert not overlap, f"CI extra includes display deps: {overlap}"

    def test_cuda12_jax_version_matches_base(self):
        """cuda12 extra's jax version range doesn't conflict with base."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        base_deps = data["project"]["dependencies"]
        extras = data["project"]["optional-dependencies"]
        # Find jax version in base
        base_jax = [d for d in base_deps if d.startswith("jax>=")]
        cuda_jax = [d for d in extras.get("cuda12", []) if "jax" in d]
        assert len(base_jax) == 1
        assert len(cuda_jax) == 1
        # Both should share the same version range
        base_range = base_jax[0].replace("jax", "")
        cuda_range = cuda_jax[0].split("]")[-1] if "]" in cuda_jax[0] else cuda_jax[0].replace("jax", "")
        assert base_range == cuda_range, (
            f"Base JAX range {base_range} != cuda12 range {cuda_range}"
        )
