"""``maddening.__version__`` is the version of the tree it is imported from.

It used to be read from the installed distribution's metadata, falling back
to a literal only when nothing was installed.  A source tree imported beside
another installed version (``PYTHONPATH`` over a stale editable install)
then called itself that version: ``GET /healthz`` on the 0.4.0 tree answered
``0.3.1``.  The package now states its own version, and these tests hold it
to ``pyproject.toml`` -- from which the wheel's metadata is built -- so a
tree and its wheel cannot disagree.
"""

from __future__ import annotations

import importlib.metadata
import os
import tomllib
from pathlib import Path

import pytest

import maddening

ROOT = Path(__file__).resolve().parents[1]


def test_the_package_states_the_version_pyproject_builds_the_wheel_with():
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert "version" not in declared.get("dynamic", ()), "read it where it is now declared"
    assert maddening.__version__ == declared["version"]


def test_the_version_does_not_follow_the_installed_metadata(monkeypatch):
    """Whatever distribution is installed beside the tree: the package is
    re-imported with the metadata answering another version."""
    import importlib

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.0.0+elsewhere")
    reloaded = importlib.reload(maddening)
    try:
        assert reloaded.__version__ == maddening.__version__ != "0.0.0+elsewhere"
    finally:
        monkeypatch.undo()
        importlib.reload(maddening)


def test_healthz_answers_the_packages_version():
    from tests._loopback_client import LoopbackTestClient

    from maddening.api.server import SimulationServer

    with LoopbackTestClient(SimulationServer({}).create_app()) as client:
        reply = client.get("/healthz")
    assert reply.status_code == 200
    assert reply.json()["version"] == maddening.__version__


def test_in_ci_the_installed_distribution_is_this_tree():
    """CI installs the tree it tests (``pip install -e .``), so there the
    wheel's metadata and the package agree.  Elsewhere the installed
    distribution may be another checkout's, which is the case the package
    no longer follows."""
    if not os.environ.get("CI"):
        pytest.skip("asked where the installed distribution is known to be this tree's "
                    "(CI); a workstation's environment may hold another checkout's")
    assert importlib.metadata.version("maddening") == maddening.__version__
