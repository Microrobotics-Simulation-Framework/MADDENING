"""``python -m maddening.compliance check-anomalies`` prints "OK" on a registry it verified nothing in.

The public command is what a downstream project runs as the gate on its own
registry.  It prints ``OK: anomaly registry is valid`` and exits 0 on

* a well-formed registry with no entries (``anomalies: []``), and
* a registry whose only reference names a symbol in a module that cannot be
  imported where the command runs (a package it needs is not installed):
  the validator records "could not be checked" in its ``notes=`` list,
  and the command never asks for that list.

Either way nothing was verified, and nothing says so
(MADD-ANO-298, open).  MADDENING's own
gate, ``scripts/check_anomalies.py``, refuses both: it fails a registry
with no anomalies, and one in which no reference was verified.  The
command does resolve what it can import: a reference to a module that does
not exist is refused (the control below).

What is held here: the two premises as what happens today, the control,
and a strict xfail of the claim "the command does not pass a registry in
which it verified nothing".
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from maddening.compliance import validate_anomaly_registry

_SRC = str(Path(__file__).resolve().parents[2] / "src")
_LABEL = "MADD-ANO-298"
#: A first-party package of the downstream project, made up for the test.
_PACKAGE = "madd_ano_downstream_fixture"
_OK = "OK: anomaly registry is valid"

_HEADER = textwrap.dedent("""\
    schema_version: "1.0"
    downstream_version: "0.1.0"
    generated_date: "2026-10-22"
""")


def _one_reference(component: str) -> str:
    return _HEADER + textwrap.dedent(f"""\
        anomalies:
          - anomaly_id: "LIB-ANO-001"
            title: "t"
            description: "d"
            severity: "minor"
            safety_relevance: "not_safety_relevant"
            safety_relevance_rationale: "r"
            resolution_status: "open"
            affected_components:
              - "{component}"
    """)


#: The registries, by case.
REGISTRIES = {
    "empty": _HEADER + "anomalies: []\n",
    # ``optional`` imports a package that is not installed: the reference
    # cannot be checked.
    "never checked": _one_reference(f"{_PACKAGE}.optional.Thing"),
    # The control: a module that does not exist is a reference that is
    # checked and fails.
    "dangling": _one_reference(f"{_PACKAGE}.nothing_here.Thing"),
}


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """``{case: (registry path, completed process)}``: the command run once
    on each registry, as a user runs it, with the downstream package on the
    path."""
    root = tmp_path_factory.mktemp("downstream")
    package = root / "pkgroot" / _PACKAGE
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "optional.py").write_text(
        "import a_package_this_environment_does_not_have  # noqa: F401\n\n\n"
        "class Thing:\n    pass\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root / "pkgroot"), _SRC] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    out = {}
    for case, text in REGISTRIES.items():
        path = root / (case.replace(" ", "_") + ".yaml")
        path.write_text(text)
        out[case] = (path, subprocess.run(
            [sys.executable, "-m", "maddening.compliance", "check-anomalies", str(path),
             "--prefix", "LIB-ANO-"],
            capture_output=True, text=True, timeout=120, env=env, cwd=str(root)))
    return out


def test_the_command_prints_ok_on_an_empty_registry(ran):
    """A premise, as what happens today: no entry, nothing verified, exit 0."""
    _path, result = ran["empty"]
    assert (result.returncode, result.stdout.strip(), result.stderr) == (0, _OK, "")


def test_the_command_prints_ok_on_a_registry_whose_only_reference_was_never_checked(
        ran, monkeypatch):
    """A premise, as what happens today: the validator knows the reference
    could not be checked (it says so in ``notes=``), and the command prints
    "OK" with nothing about it."""
    path, result = ran["never checked"]
    assert (result.returncode, result.stdout.strip(), result.stderr) == (0, _OK, "")
    monkeypatch.syspath_prepend(str(path.parent / "pkgroot"))
    notes: list[str] = []
    try:
        assert validate_anomaly_registry(str(path), prefix="LIB-ANO-", notes=notes) == []
    finally:
        for name in [m for m in sys.modules if m.split(".")[0] == _PACKAGE]:
            del sys.modules[name]
    assert len(notes) == 1 and f"{_PACKAGE}.optional.Thing" in notes[0], notes


def test_the_command_refuses_a_reference_to_a_module_that_does_not_exist(ran):
    """The control: what the command can look at, it checks."""
    _path, result = ran["dangling"]
    assert result.returncode == 1, (result.stdout, result.stderr)
    assert f"{_PACKAGE}.nothing_here.Thing" in result.stderr and "does not resolve" in result.stderr
    assert _OK not in result.stdout


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    f"{_LABEL} (open): the public check-anomalies command prints OK, exit 0, on a registry "
    "with no entries and on one whose only reference was never checked"))
@pytest.mark.parametrize("case", ["empty", "never checked"])
def test_the_command_does_not_pass_a_registry_it_verified_nothing_in(ran, case):
    """The claim that does not hold today: a gate that verified nothing
    does not report the registry valid."""
    _path, result = ran[case]
    assert result.returncode != 0 and _OK not in result.stdout, (case, result.stdout)
