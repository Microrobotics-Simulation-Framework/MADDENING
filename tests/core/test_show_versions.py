"""``maddening.show_versions()`` and ``python -m maddening info``.

The report has to name what decides how a simulation runs, never print a
secret, compile nothing, and keep its list of extras in step with
``pyproject.toml``.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

import maddening
from maddening import info
from tests.core.inspection_guard_support import compile_events

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

#: Values planted in the subprocess environment; none may appear in the report.
_SECRETS = {
    "MADDENING_API_TOKEN": "tok-should-never-print-1",
    "RUNPOD_API_KEY": "rp-should-never-print-2",
    "AWS_SECRET_ACCESS_KEY": "aws-should-never-print-3",
    "GITHUB_TOKEN": "gh-should-never-print-4",
    "MY_PASSWORD": "pw-should-never-print-5",
    "XLA_FLAGS_BACKUP": "xla-lookalike-should-never-print-6",
}


def _run_cli(*args: str, extra_env: dict | None = None, tmp_path: Path) -> subprocess.CompletedProcess:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(home), "PYTHONPATH": str(SRC),
           "JAX_PLATFORMS": "cpu", **(extra_env or {})}
    return subprocess.run([sys.executable, "-m", "maddening", *args], env=env, cwd=tmp_path,
                          capture_output=True, text=True, timeout=120)


def test_report_names_versions_backend_and_extras():
    buf = io.StringIO()
    assert maddening.show_versions(file=buf) is None
    text = buf.getvalue()
    for needle in ("MADDENING version information", "maddening", "python", "platform",
                   "jax", "jaxlib", "numpy", "scipy", "JAX runtime", "backend",
                   "devices", "x64 enabled", "optional extras", "terminal"):
        assert needle in text, needle
    assert maddening.__version__ in text


def test_report_as_dict_has_every_section():
    report = maddening.show_versions(as_dict=True)
    assert report is not None
    assert set(report) == {"maddening", "python", "platform", "machine", "dependencies",
                           "jax", "environment", "extras"}
    assert report["maddening"]["version"] == maddening.__version__
    assert set(report["dependencies"]) == {"jax", "jaxlib", "numpy", "scipy", "lineax"}
    import jax
    assert report["dependencies"]["jax"] == jax.__version__
    assert report["jax"]["backend"] == jax.default_backend()
    assert sum(d["count"] for d in report["jax"]["devices"]) == len(jax.devices())
    assert report["jax"]["x64_enabled"] is bool(jax.config.read("jax_enable_x64"))
    assert set(report["extras"]) == {name for name, _ in info.EXTRAS}


def test_report_compiles_nothing():
    with compile_events() as events:
        maddening.show_versions(file=io.StringIO())
    assert not events


def test_environment_section_reads_only_the_allowlist(monkeypatch):
    for key, value in _SECRETS.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("XLA_FLAGS", "--xla_cpu_enable_fast_math=false")
    report = maddening.show_versions(as_dict=True)
    assert report is not None
    assert set(report["environment"]) <= set(info.ENV_ALLOWLIST)
    assert report["environment"]["XLA_FLAGS"] == "--xla_cpu_enable_fast_math=false"
    blob = json.dumps(report, default=str)
    for value in _SECRETS.values():
        assert value not in blob


def test_cli_info_prints_the_report_and_no_secret(tmp_path):
    proc = _run_cli("info", extra_env={**_SECRETS, "XLA_FLAGS": "--xla_dump_hlo_as_text"},
                    tmp_path=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "MADDENING version information" in proc.stdout
    assert "XLA_FLAGS = --xla_dump_hlo_as_text" in proc.stdout
    for value in _SECRETS.values():
        assert value not in proc.stdout and value not in proc.stderr
    for key in _SECRETS:
        assert key not in proc.stdout


def test_cli_info_json_is_the_dict_report(tmp_path):
    proc = _run_cli("info", "--json", tmp_path=tmp_path)
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["dependencies"]["numpy"]
    assert "extras" in report and "jax" in report


def test_cli_without_a_command_prints_usage_and_fails(tmp_path):
    proc = _run_cli(tmp_path=tmp_path)
    assert proc.returncode == 2
    assert "usage: python -m maddening" in proc.stderr and "info" in proc.stderr


def test_cli_help_does_not_import_jax(tmp_path):
    """The parser is built without importing a heavy (or broken) dependency."""
    code = ("import sys, maddening.__main__ as m; p = m.build_parser(); "
            "p.format_help(); print('jax' in sys.modules)")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), "PYTHONPATH": str(SRC)}
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_extras_table_matches_pyproject():
    """Every feature extra is reported, and every reported extra exists."""
    extras = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"]
    not_reported = {"server", "client", "all", "ci", "dev", "sbom", "ift", "runpod", "lambda",
                    "aws", "gcp", "cloud-all"}
    reported = {name for name, _ in info.EXTRAS}
    assert reported <= set(extras), reported - set(extras)
    unaccounted = set(extras) - reported - not_reported
    assert not unaccounted, (
        f"pyproject extras {sorted(unaccounted)} are neither reported by "
        "maddening.info.EXTRAS nor listed here as bundles/aliases")


def test_a_failing_probe_is_reported_not_raised(monkeypatch):
    import jax

    def boom():
        raise RuntimeError("Unable to initialize backend 'cuda'")

    monkeypatch.setattr(jax, "devices", boom)
    report = maddening.show_versions(as_dict=True)
    assert report is not None
    assert report["jax"]["devices"].startswith("unavailable (RuntimeError: Unable to initialize")
    buf = io.StringIO()
    maddening.show_versions(file=buf)
    assert "unavailable (RuntimeError" in buf.getvalue()


def test_show_versions_is_lazily_exported():
    assert "show_versions" in maddening.__all__
    assert maddening.show_versions is info.show_versions


@pytest.mark.parametrize("module, present", [("tabnanny", True),
                                             ("definitely_not_a_module_xyz", False)])
def test_extras_are_located_without_being_imported(module, present, monkeypatch):
    monkeypatch.delitem(sys.modules, module, raising=False)
    assert info._found(module) is present
    assert module not in sys.modules
