"""The jobs that watch the dependencies, and the file that pins them.

``ci/constraints.txt`` is what per-push CI installs against.  A constraints
file is only evidence while it says what the workflows think it says, so
these tests pin, with nothing installed and no network:

* the file is current: it records the digest of the requirement strings of
  ``pyproject.toml`` that it was resolved from, every requirement CI
  installs is pinned in it, and each pin satisfies its range;
* every ``pip install`` of the project in ``ci.yml`` reads it, and the
  weekly check that installs the newest versions reads nothing;
* the generator and the drift report do what they say, on hand-made input;
* ``lineax-floor`` installs the lowest lineax ``pyproject.toml`` allows and
  runs every test file that names lineax;
* the exploring searches run every file that runs a seeded hunt, with the
  entropy switch the hunts read, and the replay it prints is the one the
  hunts accept.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
CONSTRAINTS = REPO_ROOT / "ci" / "constraints.txt"


def _load(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(f"_ci_dependency_jobs_{name}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


@pytest.fixture(scope="module")
def gen():
    return _load("generate_ci_constraints")


@pytest.fixture(scope="module")
def drift():
    return _load("report_constraint_drift")


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _pins(text: str) -> dict[str, str]:
    return dict(line.split("==", 1) for line in text.splitlines() if line and not line.startswith("#"))


# ---------------------------------------------------------------------------
# The constraints file
# ---------------------------------------------------------------------------

def test_the_constraints_file_was_resolved_from_the_requirements_pyproject_declares(gen):
    """An edited range, a new extra entry or a changed JAX version fails here until regenerated."""
    header = [ln for ln in CONSTRAINTS.read_text(encoding="utf-8").splitlines() if ln.startswith(gen.DIGEST_PREFIX)]
    assert len(header) == 1, header
    assert header[0][len(gen.DIGEST_PREFIX):] == gen.requirements_digest(_pyproject()), (
        "ci/constraints.txt was resolved from other requirements than pyproject.toml's now says; "
        "regenerate it: python scripts/generate_ci_constraints.py")


def test_the_digest_moves_with_a_range_an_extra_and_the_jax_versions(gen, monkeypatch):
    base = _pyproject()
    digest = gen.requirements_digest(base)
    for mutate in (
        lambda p: p["project"]["dependencies"].__setitem__(2, "lineax>=0.0.8"),
        lambda p: p["project"]["optional-dependencies"]["ci"].append("newpackage>=1"),
        lambda p: p["project"]["optional-dependencies"]["usd"].clear(),
    ):
        changed = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        mutate(changed)
        assert gen.requirements_digest(changed) != digest
    # An extra CI does not install changes nothing the file holds.
    other = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    other["project"]["optional-dependencies"]["cloud"].append("anything>=1")
    assert gen.requirements_digest(other) == digest
    monkeypatch.setattr(gen, "JAX_VERSIONS", ("0.10.2",))
    assert gen.requirements_digest(base) != digest


def test_the_constraints_file_is_a_sorted_list_of_exact_pins_without_jax(gen):
    text = CONSTRAINTS.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert len(lines) > 100, len(lines)
    names = []
    for line in lines:
        match = re.fullmatch(r"([a-z0-9][a-z0-9-]*)==([0-9A-Za-z.!+_-]+)", line)
        assert match, f"not a canonical name==version pin: {line!r}"
        names.append(match.group(1))
    assert names == sorted(names) and len(set(names)) == len(names)
    # Each lane pins its own; a pin here would fight the other lane's.
    assert not set(names) & set(gen.LANE_PINNED)
    assert "maddening" not in names


def test_every_requirement_ci_installs_is_pinned_inside_its_range(gen):
    pins = _pins(CONSTRAINTS.read_text(encoding="utf-8"))
    project = _pyproject()["project"]
    requirements = list(project["dependencies"])
    for extra in gen.EXTRAS:
        requirements += project["optional-dependencies"][extra]
    missing, outside = [], []
    for text in requirements:
        requirement = Requirement(text)
        name = gen.canonical(requirement.name)
        if name in gen.LANE_PINNED:
            continue            # the lanes pin these (below)
        if name not in pins:
            missing.append(text)
        elif not requirement.specifier.contains(Version(pins[name]), prereleases=True):
            outside.append(f"{text} against {name}=={pins[name]}")
    assert not missing, f"not in ci/constraints.txt: {missing}"
    assert not outside, f"pinned outside the declared range: {outside}"


def test_the_constraints_are_resolved_for_the_jax_versions_the_test_matrix_runs(gen):
    matrix = _workflow("ci.yml")["jobs"]["test"]["strategy"]["matrix"]["jax-version"]
    assert list(gen.JAX_VERSIONS) == matrix
    jax = Requirement(next(r for r in _pyproject()["project"]["dependencies"] if r.startswith("jax>")))
    for version in matrix:
        assert jax.specifier.contains(Version(version)), version


def test_the_generator_pins_what_a_pip_report_installs_and_leaves_out_the_project(gen):
    report = {"install": [
        {"metadata": {"name": "Some_Package", "version": "1.2"}},
        {"metadata": {"name": "jax", "version": "0.10.2"}},
        {"metadata": {"name": "maddening", "version": "0.4.1"}},
    ]}
    assert gen.pins_from_report(report) == {"some-package": "1.2", "jax": "0.10.2"}
    with pytest.raises(SystemExit):
        gen.pins_from_report({"install": [{"metadata": {"name": "x", "version": "1"}},
                                          {"metadata": {"name": "X", "version": "2"}}]})


def test_one_file_is_written_only_when_every_jax_version_resolves_the_same(gen):
    ten = {"jax": "0.10.2", "jaxlib": "0.10.2", "numpy": "2.5.4", "scipy": "1.18.1"}
    eleven = {**ten, "jax": "0.11.2", "jaxlib": "0.11.2"}
    assert gen.agree({"0.10.2": ten, "0.11.2": eleven}) == {"numpy": "2.5.4", "scipy": "1.18.1"}
    with pytest.raises(SystemExit, match=r"numpy \(2.5.4 / 2.6.0\)"):
        gen.agree({"0.10.2": ten, "0.11.2": {**eleven, "numpy": "2.6.0"}})
    with pytest.raises(SystemExit, match="newdep"):
        gen.agree({"0.10.2": ten, "0.11.2": {**eleven, "newdep": "1"}})


def test_the_generator_check_exits_1_on_a_stale_file_and_0_on_a_current_one(gen, tmp_path, monkeypatch):
    monkeypatch.setattr(gen, "CONSTRAINTS_FILE", tmp_path / "constraints.txt")
    assert gen.main(["--check"]) == 1                      # no file
    (tmp_path / "constraints.txt").write_text(f"{gen.DIGEST_PREFIX}0000\nnumpy==2.5.4\n", encoding="utf-8")
    assert gen.main(["--check"]) == 1                      # another digest
    digest = gen.requirements_digest(_pyproject())
    (tmp_path / "constraints.txt").write_text(f"{gen.DIGEST_PREFIX}{digest}\nnumpy==2.5.4\n", encoding="utf-8")
    assert gen.main(["--check"]) == 0
    assert gen.render({"numpy": "2.5.4"}, digest, "2026-10-14").splitlines()[-2:] == [
        f"{gen.DIGEST_PREFIX}{digest}", "numpy==2.5.4"]
    with pytest.raises(SystemExit):
        gen.render({"numpy": "2.5.4 ; evil"}, digest, "2026-10-14")


def _install_steps(workflow: dict):
    for job_id, job in workflow["jobs"].items():
        for step in job.get("steps", ()):
            if re.search(r"pip install[^\n]*-e ", step.get("run", "")):
                yield job_id, step


def test_every_install_of_the_project_in_ci_reads_the_constraints_file():
    """The per-push jobs resolve against ci/constraints.txt, not against whatever is newest today."""
    steps = list(_install_steps(_workflow("ci.yml")))
    assert {job for job, _ in steps} == {"test", "test-usd", "typecheck", "compliance", "verify-hypothesis",
                                         "lineax-floor"}, steps
    for job, step in steps:
        if job == "lineax-floor":
            # The same file with lineax's own pin taken out, which the floor job replaces.
            assert "PIP_CONSTRAINT" not in (step.get("env") or {})
            assert "grep -v '^lineax==' ci/constraints.txt" in step["run"]
            assert 'export PIP_CONSTRAINT="$RUNNER_TEMP/constraints-without-lineax.txt"' in step["run"]
        else:
            assert (step.get("env") or {}).get("PIP_CONSTRAINT") == "ci/constraints.txt", (job, step["name"])


def test_the_weekly_newest_versions_check_reads_no_constraints():
    workflow = _workflow("scheduled-checks.yml")
    raw = (WORKFLOWS / "scheduled-checks.yml").read_text(encoding="utf-8")
    job = workflow["jobs"]["newest-dependencies"]
    installs = [s for s in job["steps"] if "pip install -e" in s.get("run", "")]
    assert len(installs) == 1
    assert "PIP_CONSTRAINT" not in (installs[0].get("env") or {}) and "-c " not in installs[0]["run"]
    assert "PIP_CONSTRAINT" not in json.dumps(job)
    assert not re.search(r"jax==|jaxlib==", json.dumps(job)), "the newest check pins no JAX"
    # ... and it says what is newer than the file.
    assert any("report_constraint_drift.py ci/constraints.txt" in s.get("run", "") for s in job["steps"])
    assert "ci/constraints.txt" in raw


def test_the_scheduled_checks_run_on_a_schedule_and_on_demand_and_never_on_a_push():
    on = _workflow("scheduled-checks.yml")[True]      # YAML reads the key `on` as True
    assert set(on) == {"schedule", "workflow_dispatch"}
    assert [e["cron"] for e in on["schedule"]] == ["23 4 * * 1", "41 4 * * 4"]
    jobs = _workflow("scheduled-checks.yml")["jobs"]
    # Each job's condition names its own cron (or a dispatch asking for it).
    assert "github.event.schedule == '23 4 * * 1'" in jobs["newest-dependencies"]["if"]
    assert "github.event.schedule == '41 4 * * 4'" in jobs["explore-searches"]["if"]
    assert "inputs.check != 'explore-searches'" in jobs["newest-dependencies"]["if"]
    assert "inputs.check != 'newest-dependencies'" in jobs["explore-searches"]["if"]
    assert on["workflow_dispatch"]["inputs"]["check"]["options"] == ["both", "newest-dependencies", "explore-searches"]


def test_the_newest_versions_check_runs_the_per_push_split():
    job = _workflow("scheduled-checks.yml")["jobs"]["newest-dependencies"]
    n = 6
    assert job["strategy"]["matrix"]["shard"] == list(range(1, n + 1))
    (run,) = [s for s in job["steps"] if s.get("name") == "Run tests"]
    assert run["env"]["MADDENING_TEST_SHARD"] == "${{ matrix.shard }}/6:weighted-push"
    assert "tests/ -v -rs --tb=short --ignore=tests/viz" in run["run"]


# ---------------------------------------------------------------------------
# The drift report
# ---------------------------------------------------------------------------

def test_the_drift_report_lists_what_is_newer_new_and_absent(drift):
    pinned = drift.read_constraints("# header\nnumpy==2.5.4\nlineax==0.1.1\nsome-package==1.0\n")
    have = {"numpy": "2.6.0", "lineax": "0.1.1", "extra-thing": "3", "maddening": "0.4.1", "jax": "0.11.2"}
    found = drift.drift(pinned, have)
    assert found == {"changed": [("numpy", "2.5.4", "2.6.0")], "new": [("extra-thing", "3")],
                     "absent": [("some-package", "1.0")]}
    text = drift.markdown(found, "ci/constraints.txt")
    assert "| numpy | 2.5.4 | 2.6.0 |" in text and "extra-thing 3" in text
    assert "some-package" not in text, "a distribution the job did not install is not drift"
    assert "Nothing differs" in drift.markdown({"changed": [], "new": [], "absent": [("x", "1")]}, "f")
    with pytest.raises(SystemExit):
        drift.read_constraints("numpy>=2\n")


# ---------------------------------------------------------------------------
# lineax at its floor
# ---------------------------------------------------------------------------

def _lineax_floor_job() -> dict:
    return _workflow("ci.yml")["jobs"]["lineax-floor"]


def _names_lineax(path: Path) -> bool:
    return "lineax" in path.read_text(encoding="utf-8")


def test_the_lineax_floor_job_installs_the_floor_pyproject_declares():
    (spec,) = [Requirement(r) for r in _pyproject()["project"]["dependencies"] if r.startswith("lineax")]
    floors = [s.version for s in spec.specifier if s.operator == ">="]
    assert len(floors) == 1, spec
    job = _lineax_floor_job()
    (install,) = [s for s in job["steps"] if "pip install -e" in s.get("run", "")]
    assert f'"lineax=={floors[0]}"' in install["run"], (floors[0], install["run"])
    assert f"assert v == '{floors[0]}'" in install["run"]
    # Both JAX versions of the matrix, like every test lane.
    assert job["strategy"]["matrix"]["jax-version"] == _workflow("ci.yml")["jobs"]["test"]["strategy"]["matrix"]["jax-version"]
    assert job["strategy"]["fail-fast"] is False
    # The newest lineax is what every other job installs: the floor job is the other end.
    assert Version(_pins(CONSTRAINTS.read_text(encoding="utf-8"))["lineax"]) > Version(floors[0])


def _lineax_targets() -> list[str]:
    (step,) = [s for s in _lineax_floor_job()["steps"] if s.get("name") == "Run the tests that name lineax"]
    return re.findall(r"(tests/\S+\.py)", re.sub(r"\\\n", " ", step["run"]))


def test_the_lineax_floor_job_runs_every_test_file_that_names_lineax():
    """A new test of the lineax paths that is not listed would run only on the newest lineax."""
    named = sorted(p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "tests").rglob("*.py")
                   if "compliance" not in p.relative_to(REPO_ROOT / "tests").parts and _names_lineax(p)
                   and p.name.startswith("test_"))
    targets = _lineax_targets()
    assert targets == sorted(set(targets)), "list each file once, sorted"
    assert sorted(set(named) - set(targets)) == [], "name lineax and are not run at its floor"
    assert sorted(set(targets) - set(named)) == [], "listed, but no longer name lineax (or do not exist)"
    for target in targets:
        assert (REPO_ROOT / target).is_file(), target


def test_the_lineax_floor_job_cannot_pass_by_running_nothing():
    job = _lineax_floor_job()
    (step,) = [s for s in job["steps"] if s.get("name") == "Run the tests that name lineax"]
    assert "continue-on-error" not in step and "||" not in step["run"]
    assert not re.search(r"\s-m\s+(?!pytest\b)", step["run"]), "no marker expression: the default lane's `not slow` applies"


# ---------------------------------------------------------------------------
# The exploring searches
# ---------------------------------------------------------------------------

def _explore_job() -> dict:
    return _workflow("scheduled-checks.yml")["jobs"]["explore-searches"]


def _files_that_run_a_seeded_hunt() -> list[str]:
    return sorted(p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "tests").rglob("test_*.py")
                  if ".seeded(" in p.read_text(encoding="utf-8")
                  and "targeted_search" in p.read_text(encoding="utf-8")
                  and p.name != "test_targeted_search.py"
                  and "compliance" not in p.parts)


def test_the_exploring_job_runs_every_file_that_hunts_under_a_seed():
    files = _explore_job()["strategy"]["matrix"]["file"]
    assert sorted(files) == _files_that_run_a_seeded_hunt()


def test_the_exploring_job_uses_the_switch_the_hunts_read():
    ts = (REPO_ROOT / "tests" / "property" / "targeted_search.py").read_text(encoding="utf-8")
    switch = re.search(r'^ENTROPY_VARIABLE = "([A-Z_]+)"', ts, re.M).group(1)
    job = _explore_job()
    (run,) = [s for s in job["steps"] if s.get("name") == "Explore"]
    assert run["env"][switch] == "fresh"
    assert "-m slow" in run["run"] and "set -o pipefail" in run["run"]
    # One file per process: the entropy is drawn once per process.
    assert run["run"].count("${{ matrix.file }}") == 1
    assert "continue-on-error" not in run
    # The other steps never set it: a find is reproducible from the summary alone.
    others = [s for s in job["steps"] if s is not run]
    assert switch not in json.dumps(others) or "MADDENING_SEARCH_ENTROPY=$entropy" in json.dumps(others)
    # The job installs against the constraints, as per-push CI does: the newest-versions job is the other one.
    (install,) = [s for s in job["steps"] if "pip install -e" in s.get("run", "")]
    assert install["env"]["PIP_CONSTRAINT"] == "ci/constraints.txt"


def test_the_summary_step_names_the_entropy_and_its_replay_from_what_a_hunt_prints(tmp_path):
    """Run the step's own shell on a log with the line ``targeted_search`` prints."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from tests.property import targeted_search as ts
    finally:
        sys.path.remove(str(REPO_ROOT))
    line = (f"cell: exploring under {ts.ENTROPY_VARIABLE}=1234567890 (the hunt's own seed is 1007): "
            "the same command with that value in place of 'fresh' draws these examples again")
    log = tmp_path / "explore.log"
    log.write_text(f"collected 3 items\n{line}\nFAILED tests/property/x.py::test_y - AssertionError: bound\n")
    (step,) = [s for s in _explore_job()["steps"] if s.get("name") == "Report the find and how to replay it"]
    script = step["run"].replace("${{ matrix.file }}", "tests/property/x.py")
    summary = tmp_path / "summary.md"
    proc = subprocess.run(["bash", "-e", "-c", script], cwd=tmp_path, capture_output=True, text=True,
                          env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary)}, timeout=60)
    assert proc.returncode == 0, proc.stderr
    text = summary.read_text()
    assert "MADDENING_SEARCH_ENTROPY=1234567890 python -m pytest tests/property/x.py -m slow -rP" in text
    assert "FAILED tests/property/x.py::test_y" in text
    # The replay accepts what the summary prints: an integer is the entropy of that run.
    old = os.environ.get(ts.ENTROPY_VARIABLE)
    os.environ[ts.ENTROPY_VARIABLE] = "1234567890"
    try:
        assert ts.search_entropy() == 1234567890
    finally:
        if old is None:
            del os.environ[ts.ENTROPY_VARIABLE]
        else:
            os.environ[ts.ENTROPY_VARIABLE] = old
    # No hunt reached: it says so rather than printing an empty replay.
    log.write_text("ERROR collecting\n")
    summary.unlink()
    proc = subprocess.run(["bash", "-e", "-c", script], cwd=tmp_path, capture_output=True, text=True,
                          env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary)}, timeout=60)
    assert proc.returncode == 0 and "did not reach a hunt" in summary.read_text()
