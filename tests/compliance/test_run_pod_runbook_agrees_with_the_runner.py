"""The multi-GPU runbook and ``run_pod.py`` state the same numbers and keys.

``benchmarks/multigpu/README.md`` is the session's runbook: the maintainer
reads its limits, its schema and its record layout at a metered pod.  A
change to the runbook alone is a docs-only change, which runs
``tests/compliance`` and nothing else, so the rows of
``docs/validation/rest_runpod_claims.yaml`` that compare the runbook with
the runner (``RPD-029``, ``RPD-030``) are checked here.  The runner is
loaded from its file and runs nothing.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = REPO_ROOT / "benchmarks" / "multigpu" / "run_pod.py"
_RUNBOOK = REPO_ROOT / "benchmarks" / "multigpu" / "README.md"
_RECORD = REPO_ROOT / "tests" / "cloud" / "multigpu" / "run_pod_record"


@pytest.fixture(scope="module")
def rp():
    spec = importlib.util.spec_from_file_location("run_pod_for_its_runbook", _RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runbook() -> str:
    return _RUNBOOK.read_text(encoding="utf-8")


def test_the_limits_are_the_runbooks(rp, runbook):
    """The runbook's checklist table states each limit; ``LIMITS`` holds
    them, and the README says so ("The limits live in LIMITS")."""
    assert rp.LIMITS == {"exact": 0.0, "forward": 1e-5, "gradient": 1e-5,
                         "coupled_gradient_ift": 1e-4, "coupled_gradient_fori": 1e-5,
                         "krylov": 1e-3, "model_gradient": 1e-4}
    for fragment in ("rel 1e-5 |", "**0** (bit for bit)", "1e-4 (coupled, IFT)",
                     "1e-3 (`sharded_cg`)", "1e-4 (IFT) / 1e-5 (`\"fori\"`), 1e-4 against "
                     "the model"):
        assert fragment in runbook, fragment


def test_the_runbook_names_the_runners_schema_version(rp, runbook):
    version = rp.SCHEMA_VERSION
    assert f"## Schema of the JSON (schema_version {version})" in runbook
    assert f"be on the current `schema_version` ({version})" in runbook
    for path in sorted(_RECORD.glob("*.json")):
        assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == version


def test_every_record_carries_the_keys_the_runbook_lists(rp, runbook):
    common = ("schema_version", "goal", "dry_run", "allow_fewer_devices", "n_devices",
              "environment", "config", "wall_s", "results", "checks", "passed")
    environment = ("hostname", "timestamp_utc", "python", "jax", "jaxlib", "platform",
                   "devices", "device_kinds", "n_devices_visible", "nvidia_smi", "xla_flags",
                   "jax_platforms", "git_commit")
    schema = runbook[runbook.index("## Schema of the JSON"):]
    for key in common[1:] + environment:
        assert f"`{key}`" in schema, key
    files = sorted(_RECORD.glob("*.json"))
    assert {p.stem for p in files} == set(rp.ALL_GOALS)
    for path in files:
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert set(common) <= set(doc), (path.name, set(common) - set(doc))
        assert set(environment) <= set(doc["environment"]), path.name
        for c in doc["checks"]:
            assert {"name", "value", "limit", "sense", "passed"} <= set(c), (path.name, c)


def test_the_runbooks_exit_statuses_are_the_runners(rp, runbook):
    """Section 2b reads each goal's exit status and section 3 stops on
    every non-zero one; both name the runner's own statuses for a refusal
    and a crash (RPD-009 for the summary's).  The runbook used to read exit
    1 as "a check failed" when a refusal and a crash exited 1 as well."""
    assert (rp.EXIT_REFUSED, rp.EXIT_CRASHED) == (2, 5)
    section_2b = runbook.split("### 2b.", 1)[1].split("```", 1)[0]
    for fragment in ("**0** = no check failed", "**1** = a check failed",
                     f"**{rp.EXIT_REFUSED}** = the runner refused the run",
                     f"**{rp.EXIT_CRASHED}** = a goal raised", "**124** = the time box ran out"):
        assert fragment in " ".join(section_2b.split()), fragment
    section_3 = " ".join(runbook.split("## 3. Stop condition", 1)[1].split("## 4.", 1)[0].split())
    for fragment in ("`1` (a check failed)", f"`{rp.EXIT_REFUSED}` (the runner refused the run)",
                     f"`{rp.EXIT_CRASHED}` (a goal raised)", "`124` (its time box ran out)"):
        assert fragment in section_3, fragment
