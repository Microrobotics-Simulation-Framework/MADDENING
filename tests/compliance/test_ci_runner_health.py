"""Runner health: the pin, the instrumentation, the canary and its wiring.

``scripts/ci_runner_health.py`` logs what each test job's runner is, runs a
canary after the install, and fails a degraded runner at once with exit code
75; ``.github/workflows/ci.yml`` and ``slow-tests.yml`` call it.  A workflow
that calls it wrongly still goes green, so these tests pin both halves:

* every ``runs-on`` is ``ubuntu-24.04`` (``ubuntu-latest`` becomes 26.04 on
  2026-10-19);
* the script reads the kernel's files right (CPU model, vCPUs, memory, steal
  time) -- on canned ``/proc`` text -- and decides as documented;
* a tripped canary exits 75 and leaves the annotation, the summary line, the
  output and the marker file; a healthy one leaves none of them;
* the limits are set, and sit above the healthy maxima they were derived from;
* in both lanes the steps are in the order the script needs, no step writes a
  runner file beside the JUnit report (the report is the one carrier of the
  runner's fields for the weights tooling: ``tests/_sharding.py``), and no
  step that runs after a failure runs after a tripped canary (a re-run could
  not upload the same artifact twice).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location("ci_runner_health", REPO_ROOT / "scripts" / "ci_runner_health.py")
crh = importlib.util.module_from_spec(_spec)
sys.modules["ci_runner_health"] = crh
_spec.loader.exec_module(crh)

CANARY_STEP = "Runner canary (degraded runner check)"
MARKER_STEP = "Mark a degraded runner"
BEFORE_STEP = "Runner health (before the tests)"
AFTER_STEP = "Runner health (after the tests)"
MACHINE_STEP = "Runner health (machine)"
#: (workflow file, job, the step that runs pytest, the artifact-upload step)
LANES = [
    ("ci.yml", "test", "Run tests", "Upload test durations"),
    ("slow-tests.yml", "slow", "Run full suite (slow lane)", "Upload test durations"),
]


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _steps(workflow: str, job: str) -> list[dict]:
    return _workflow(workflow)["jobs"][job]["steps"]


def _names(steps: list[dict]) -> list[str]:
    return [s.get("name", "") for s in steps]


# ---------------------------------------------------------------------------
# the runner image pin
# ---------------------------------------------------------------------------

def test_every_job_of_every_workflow_runs_on_ubuntu_24_04():
    """Not ``ubuntu-latest`` (26.04 from 2026-10-19), and not another image either."""
    seen = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for job_name, job in yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"].items():
            seen.append((path.name, job_name, job.get("runs-on")))
    assert len(seen) >= 12, seen            # the scan found the jobs
    assert [s for s in seen if s[2] != "ubuntu-24.04"] == []


def test_no_workflow_names_ubuntu_latest_as_a_runner_or_in_a_comment_that_says_what_it_is():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        # The pin's own explanation may say the word; nothing may *use* it.
        uses = [line for line in path.read_text(encoding="utf-8").splitlines()
                if re.search(r"runs-on:\s*ubuntu-latest|`ubuntu-latest`\s+(runner|image)", line)]
        assert uses == [], (path.name, uses)


def test_the_soup_tables_read_the_pin_from_ci_yml():
    spec = importlib.util.spec_from_file_location("gen_soup", REPO_ROOT / "scripts" / "generate_soup_tables.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert gen.read_ci()["runners"] == ["ubuntu-24.04"]


# ---------------------------------------------------------------------------
# reading the machine
# ---------------------------------------------------------------------------

CPUINFO = """processor\t: 0
vendor_id\t: AuthenticAMD
model name\t: AMD EPYC 7763 64-Core Processor
cpu MHz\t\t: 2445.426

processor\t: 1
model name\t: AMD EPYC 7763 64-Core Processor

processor\t: 2
model name\t: AMD EPYC 7763 64-Core Processor

processor\t: 3
model name\t: AMD EPYC 7763 64-Core Processor
"""

MEMINFO = """MemTotal:       16379276 kB
MemFree:         9000000 kB
MemAvailable:   15028880 kB
Buffers:          100000 kB
"""


def test_cpu_model_and_vcpus_come_from_cpuinfo():
    assert crh.cpu_model(CPUINFO) == "AMD EPYC 7763 64-Core Processor"
    assert crh.vcpu_count(CPUINFO) == 4
    assert crh.cpu_model("processor : 0\n") == "unknown"       # an arm64 box has no `model name`


def test_memory_is_total_and_available_not_free():
    assert crh.meminfo(MEMINFO) == {"total_kib": 16379276, "available_kib": 15028880}
    with pytest.raises(ValueError, match="MemAvailable|available_kib"):
        crh.meminfo("MemTotal: 1 kB\n")


def test_steal_is_the_eighth_column_and_its_share_is_of_the_first_eight():
    before = crh.cpu_ticks("cpu  100 0 50 800 10 0 5 3 0 0\ncpu0 1 1 1 1 1 1 1 1 1 1\n")
    after = crh.cpu_ticks("cpu  400 0 150 1500 30 0 15 53 99 0\n")     # guest (99) is inside user: not added
    assert before == {"total": 968, "steal": 3}
    assert after == {"total": 2148, "steal": 53}
    delta = crh.steal_between(before, after)
    assert delta["steal_s"] == pytest.approx(0.5) and delta["cpu_s"] == pytest.approx(11.8)
    assert delta["steal_pct"] == pytest.approx(100 * 50 / 1180)
    assert crh.steal_between(before, before) == {"steal_s": 0.0, "cpu_s": 0.0, "steal_pct": 0.0}


def test_a_short_cpu_line_without_a_steal_column_reads_as_no_steal():
    assert crh.cpu_ticks("cpu  1 2 3 4\n") == {"total": 10, "steal": 0}
    with pytest.raises(ValueError):
        crh.cpu_ticks("intr 1 2 3\n")


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------

def test_the_limits_are_set_and_sit_above_the_healthy_runners_they_were_derived_from():
    # benchmarks/results/050_p0_3b_ci/thresholds.md: the slowest model's medians and the healthy maxima.
    assert 2.83 * 3 - 0.1 <= crh.CANARY_LIMIT_S <= 2.83 * 3 + 0.1
    assert 68 * 3 - 1 <= crh.INSTALL_LIMIT_S <= 68 * 3 + 1
    assert crh.CANARY_LIMIT_S > 3.09 * 2 and crh.INSTALL_LIMIT_S > 131          # the worst healthy runs seen
    assert crh.INSTALL_LIMIT_S < 540                                             # the degraded install of 2026-10-13


@pytest.mark.parametrize("canary, install, force, expected", [
    (2.8, 70.0, "", 0),
    (None, None, "", 0),                                   # nothing measured, nothing to trip on
    (crh.CANARY_LIMIT_S + 0.1, 70.0, "", 1),
    (2.8, crh.INSTALL_LIMIT_S + 1, "", 1),
    (30.0, 540.0, "", 2),
    (crh.CANARY_LIMIT_S, crh.INSTALL_LIMIT_S, "", 0),      # at the limit is not over it
    (2.8, 70.0, "canary", 1),
    (2.8, 70.0, "install", 1),
    (2.8, 70.0, "both", 2),
    (2.8, 70.0, "1", 2),
])
def test_the_decision_trips_over_either_limit_or_when_forced(canary, install, force, expected):
    reasons = crh.degraded_reasons(canary, install, crh.CANARY_LIMIT_S, crh.INSTALL_LIMIT_S, force)
    assert len(reasons) == expected, reasons


def test_the_exit_code_is_not_one_pytest_or_timeout_or_a_signal_uses():
    assert crh.DEGRADED_EXIT == 75
    assert crh.DEGRADED_EXIT not in {0, 1, 2, 3, 4, 5, 124, 125, 126, 127} and crh.DEGRADED_EXIT < 128


# ---------------------------------------------------------------------------
# the commands, run as a step runs them
# ---------------------------------------------------------------------------

def test_machine_before_after_write_the_log_lines_and_the_summary(tmp_path, capsys, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "cpuinfo").write_text(CPUINFO)
    (proc / "meminfo").write_text(MEMINFO)
    (proc / "stat").write_text("cpu  100 0 50 800 10 0 5 3 0 0\n")
    monkeypatch.setattr(crh, "PROC_ROOT", str(proc))
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    monkeypatch.delenv("MADDENING_RUNNER_STATE", raising=False)

    assert crh.main(["machine"]) == 0
    assert crh.main(["before-tests"]) == 0
    (proc / "meminfo").write_text(MEMINFO.replace("15028880", "9000000"))
    (proc / "stat").write_text("cpu  4100 0 150 1500 30 0 15 1203 0 0\n")
    capsys.readouterr()
    assert crh.main(["after-tests", "--label", "lane 3"]) == 0

    line = capsys.readouterr().out.strip()
    assert line.startswith("runner: after the tests, memory 8.6 GiB available of 15.6 GiB; "), line
    assert "cpu steal 12.0 s of 60 cpu-s" in line and "(19.90%)" in line, line
    summary = (tmp_path / "summary").read_text()
    assert "lane 3" in summary and "cpu steal 12.0 s" in summary
    # The weights tooling reads the runner from the JUnit report: no file here for it.
    assert sorted(f.name for f in tmp_path.iterdir()) == ["maddening-runner-health.json", "proc", "summary"]


def test_the_log_lines_are_the_ones_the_variance_report_reads(tmp_path, capsys, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "cpuinfo").write_text(CPUINFO)
    (proc / "meminfo").write_text(MEMINFO)
    (proc / "stat").write_text("cpu  1 0 1 1 0 0 0 0 0 0\n")
    monkeypatch.setattr(crh, "PROC_ROOT", str(proc))
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.delenv("MADDENING_RUNNER_STATE", raising=False)
    crh.main(["machine"])
    line = capsys.readouterr().out.strip()
    assert line == "runner: cpu 'AMD EPYC 7763 64-Core Processor', 4 vCPU, memory 15.6 GiB total, 14.3 GiB available"
    spec = importlib.util.spec_from_file_location("cvr", REPO_ROOT / "scripts" / "ci_variance_report.py")
    cvr = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "cvr", cvr)           # a dataclass looks its module up here
    spec.loader.exec_module(cvr)
    assert cvr.cpu_from_log(line) == "AMD EPYC 7763 64-Core Processor"


@pytest.fixture
def tiny_canary(monkeypatch):
    """The real benchmark at a size a unit test can afford (JAX still imports and compiles)."""
    monkeypatch.setattr(crh, "MATMUL_N", 64)
    monkeypatch.setattr(crh, "MATMUL_REPS", 1)
    monkeypatch.setattr(crh, "JIT_LAYERS", 3)
    monkeypatch.setattr(crh, "JIT_WIDTH", 8)
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")          # the canary sets it by default; leave no trace


def test_a_healthy_canary_exits_zero_and_leaves_no_marker(tmp_path, monkeypatch, capsys, tiny_canary):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    monkeypatch.delenv("MADDENING_FORCE_DEGRADED", raising=False)
    monkeypatch.delenv("MADDENING_RUNNER_STATE", raising=False)
    assert crh.main(["canary", "--label", "lane 1"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"runner: canary numpy matmul [\d.]+ s \+ jit compile [\d.]+ s = [\d.]+ s \(limit 8\.5 s\)", out)
    assert "::error" not in out and not (tmp_path / "degraded-runner.json").exists()
    assert "runner healthy" in (tmp_path / "summary").read_text()
    assert not (tmp_path / "output").exists()


def test_a_forced_canary_exits_75_with_the_annotation_summary_output_and_marker(tmp_path, monkeypatch, capsys, tiny_canary):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "summary"))
    monkeypatch.setenv("MADDENING_FORCE_DEGRADED", "canary")
    monkeypatch.delenv("MADDENING_RUNNER_STATE", raising=False)
    assert crh.main(["canary", "--label", "per-push JAX 0.10.2 shard 3/6"]) == crh.DEGRADED_EXIT == 75
    out = capsys.readouterr().out
    assert "::error title=Degraded runner::" in out and "forced by MADDENING_FORCE_DEGRADED" in out
    assert "stops before the tests" in out
    assert (tmp_path / "output").read_text() == "degraded=true\n"
    assert "**degraded runner**" in (tmp_path / "summary").read_text()
    marker = json.loads((tmp_path / "degraded-runner.json").read_text())
    assert marker["degraded"] is True and marker["exit_code"] == 75 and marker["label"].startswith("per-push")
    assert marker["canary_limit_s"] == crh.CANARY_LIMIT_S and marker["score_s"] > 0


def test_a_slow_install_alone_trips_it(tmp_path, monkeypatch, capsys, tiny_canary):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path))
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "output"))
    monkeypatch.delenv("MADDENING_FORCE_DEGRADED", raising=False)
    monkeypatch.delenv("MADDENING_RUNNER_STATE", raising=False)
    state = {"marks": {"install_start": __import__("time").time() - 540}, "cpu_model": "X", "vcpus": 4}
    (tmp_path / "maddening-runner-health.json").write_text(json.dumps(state))
    assert crh.main(["canary"]) == 75
    out = capsys.readouterr().out
    assert "the install took 540 s (limit 204 s)" in out and "CPU canary took" not in out


def test_the_script_hands_the_commands_status_to_the_process():
    """A step fails on the process's exit status: ``main()``'s return value must be what exits."""
    text = (REPO_ROOT / "scripts" / "ci_runner_health.py").read_text(encoding="utf-8")
    assert text.rstrip().endswith('if __name__ == "__main__":\n    sys.exit(main())')
    assert "return DEGRADED_EXIT" in text


# ---------------------------------------------------------------------------
# the wiring in both lanes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("workflow, job, pytest_step, upload_step", LANES)
def test_the_steps_are_in_the_order_the_script_needs(workflow, job, pytest_step, upload_step):
    names = _names(_steps(workflow, job))
    for step in (MACHINE_STEP, CANARY_STEP, MARKER_STEP, BEFORE_STEP, pytest_step, AFTER_STEP, upload_step):
        assert names.count(step) == 1, (workflow, step, names)
    i = names.index
    assert i("Set up Python ${{ matrix.python-version }}") < i(MACHINE_STEP) < i("Install dependencies") < i(CANARY_STEP)
    assert i(CANARY_STEP) < i(MARKER_STEP) < i(BEFORE_STEP)
    assert i(BEFORE_STEP) < i(pytest_step) < i(AFTER_STEP) < i(upload_step)
    # Nothing between the before-snapshot and pytest but the cache restore (per-push) -- the
    # steal and memory figures are of the test step, not of setup.
    between = names[i(BEFORE_STEP) + 1:i(pytest_step)]
    assert between == [], between


@pytest.mark.parametrize("workflow, job, pytest_step, upload_step", LANES)
def test_the_steps_call_the_script_with_the_arguments_it_takes(workflow, job, pytest_step, upload_step):
    steps = {s.get("name"): s for s in _steps(workflow, job)}
    assert steps[MACHINE_STEP]["run"].strip() == "python scripts/ci_runner_health.py machine"
    assert steps[BEFORE_STEP]["run"].strip() == "python scripts/ci_runner_health.py before-tests"
    assert steps[CANARY_STEP]["id"] == "canary"
    assert re.fullmatch(r'python scripts/ci_runner_health\.py canary --label "[^"]+"', steps[CANARY_STEP]["run"].strip())
    after = steps[AFTER_STEP]["run"]
    assert re.fullmatch(r'python scripts/ci_runner_health\.py after-tests --label "[^"]+"', after.strip()), after
    # One carrier for the runner's fields: the JUnit report (tests/_sharding.py).
    # The durations artifact carries no runner file for the weights tooling to
    # read instead, so the two can never disagree.
    assert "runner-shard" not in steps[upload_step]["with"]["path"]
    # The script has every subcommand the workflow names.
    script = (REPO_ROOT / "scripts" / "ci_runner_health.py").read_text(encoding="utf-8")
    for command in ("machine", "canary", "before-tests", "after-tests"):
        assert f'sub.add_parser("{command}"' in script


@pytest.mark.parametrize("workflow, job, pytest_step, upload_step", LANES)
def test_the_marker_artifact_is_named_by_the_job_and_uploaded_only_when_the_canary_tripped(
        workflow, job, pytest_step, upload_step):
    marker = {s.get("name"): s for s in _steps(workflow, job)}[MARKER_STEP]
    assert marker["with"]["name"] == "degraded-runner-${{ job.check_run_id }}"
    assert marker["with"]["path"] == "${{ runner.temp }}/degraded-runner.json"
    assert marker["if"] == "${{ always() && steps.canary.outputs.degraded == 'true' }}"
    assert marker["uses"].startswith("actions/upload-artifact@v4")


@pytest.mark.parametrize("workflow, job, pytest_step, upload_step", LANES)
def test_no_step_that_runs_after_a_failure_runs_after_a_tripped_canary(workflow, job, pytest_step, upload_step):
    """``!cancelled()`` and ``always()`` steps run after a failed step; after the canary they would
    run the budget gate on a report that does not exist, or upload an artifact name that a re-run
    of the job could not upload again."""
    unguarded = []
    for step in _steps(workflow, job):
        condition = str(step.get("if", ""))
        if not re.search(r"!cancelled\(\)|always\(\)|failure\(\)", condition):
            continue
        guarded = ("steps.canary.outcome != 'failure'" in condition
                   or "steps.health-before.outcome == 'success'" in condition
                   or "steps.canary.outputs.degraded == 'true'" in condition
                   or "steps.tests.outcome == 'success'" in condition
                   or "hashFiles('test-results-shard*.xml') != ''" in condition)
        # Steps that read the test report need it to exist; those guards say the tests ran.
        if not guarded:
            unguarded.append(step.get("name"))
    assert unguarded == [], f"{workflow}: runs after a tripped canary: {unguarded}"


@pytest.mark.parametrize("workflow, job, pytest_step, upload_step", LANES)
def test_the_canary_runs_after_the_install_and_before_anything_that_needs_the_tests(workflow, job, pytest_step, upload_step):
    steps = _steps(workflow, job)
    names = _names(steps)
    canary = names.index(CANARY_STEP)
    # No step between the canary and the pytest step may run pytest or the budget script
    # (they would run on a runner the canary has not cleared).
    for step in steps[canary + 1:names.index(pytest_step)]:
        assert "pytest" not in str(step.get("run", "")), step.get("name")


def test_the_two_lanes_use_one_canary_step_name_and_the_rerun_workflow_reads_it():
    rerun = (WORKFLOWS / "rerun-degraded.yml").read_text(encoding="utf-8")
    assert f'CANARY_STEP = "{CANARY_STEP}"' in rerun
    assert 'MARKER_PREFIX = "degraded-runner-"' in rerun
