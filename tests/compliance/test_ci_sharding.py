"""The CI test shards must be a stable partition of the suite.

Shard ``i`` of a pull request has to hold the same test files as shard
``i`` of the base branch, because the XLA compilation cache a shard
restores was saved by that shard on the base branch.  These tests pin the
properties that make that true: the assignment is a function of the file
path alone, every test lands on exactly one shard, the pins are real and
in range, and CI runs the job count the pins were balanced for -- every
shard of it (no matrix leg excluded or added).

They also pin what the sharded lanes select, in each place a selection can
be written, since a narrower one drops tests from CI while every other
check here still passes:

* the pytest command line of both lanes (no extra ``--ignore``, ``-k``,
  ``-m`` or ``--deselect``);
* the environment pytest reads more arguments or plugins from
  (``PYTEST_ADDOPTS``, ``PYTEST_PLUGINS``), at workflow, job and step
  level, and in any ``run`` script;
* ``[tool.pytest.ini_options]`` in ``pyproject.toml`` (``addopts`` exactly
  ``-m 'not slow'``, no key this file does not know) and the absence of
  any other pytest configuration file;
* every ``conftest.py`` under ``tests/`` (no collection hook, no
  ``collect_ignore``);
* the root ``conftest.py`` in action: a real pytest run through it, with
  ``MADDENING_TEST_SHARD=1/2``, keeps one of two files and deselects the
  other.

What none of it sees: a test that skips itself, and a ``pytestmark`` or
fixture in a test module that deselects or skips its own tests.
"""

import ast
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests import _jax_timing, _sharding

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_assignment_is_a_fixed_function_of_the_path():
    # Golden values: a change here re-deals every file, which invalidates
    # every shard's cache on the base branch.  It has to be deliberate.
    assert _sharding.shard_of("tests/core/test_graph_manager.py", 4) == 3
    assert _sharding.shard_of("tests/nodes/test_spring.py", 4) == 4
    assert _sharding.shard_of("tests/fmi/test_c_unit.py", 4) == 1
    assert _sharding.shard_of("tests/nodes/test_spring.py", 3) == 2


def test_pins_apply_only_at_the_job_count_they_were_balanced_for():
    # At any other count a pinned file hashes like every other file: a pin
    # that leaked would put it on a shard the count may not even have.
    for path, pinned in _sharding.PINS.items():
        assert _sharding.shard_of(path, _sharding.PINS_FOR) == pinned
        for n in (1, 2, 3, 5, 8):
            if n == _sharding.PINS_FOR:
                continue
            by_hash = int(hashlib.sha256(path.encode("utf-8")).hexdigest(), 16) % n + 1
            assert _sharding.shard_of(path, n) == by_hash, (path, n)


def test_every_pin_names_a_real_file_and_a_real_shard():
    # A pin to a renamed file silently stops balancing anything.
    for path, shard in _sharding.PINS.items():
        assert (REPO_ROOT / path).is_file(), f"pinned file {path} does not exist"
        assert 1 <= shard <= _sharding.PINS_FOR, (path, shard)


@pytest.mark.parametrize("spec", ["0/4", "5/4", "a/b", "4", "1/4/2", ""])
def test_a_malformed_shard_spec_is_refused(spec):
    with pytest.raises(pytest.UsageError):
        _sharding.parse(spec)


def test_the_shards_partition_the_tests_and_keep_files_together():
    nodeids = [f"tests/{d}/test_{f}.py::test_{t}" for d in "abc" for f in range(7) for t in range(3)]
    nodeids += list(_sharding.PINS)  # pinned files too
    n = _sharding.PINS_FOR
    seen = {}
    for shard in range(1, n + 1):
        items = [SimpleNamespace(nodeid=nid) for nid in nodeids]
        dropped = []
        config = SimpleNamespace(hook=SimpleNamespace(
            pytest_deselected=lambda items: dropped.extend(items)))
        _sharding.Plugin(shard, n).pytest_collection_modifyitems(config, items)
        assert len(items) + len(dropped) == len(nodeids)
        for item in items:
            assert item.nodeid not in seen, f"{item.nodeid} on shards {seen[item.nodeid]} and {shard}"
            seen[item.nodeid] = shard
    assert set(seen) == set(nodeids)
    by_file = {}
    for nid, shard in seen.items():
        by_file.setdefault(nid.split("::")[0], set()).add(shard)
    assert all(len(s) == 1 for s in by_file.values()), "a file was split across shards"


@pytest.mark.parametrize("workflow", ["ci.yml", "slow-tests.yml"])
def test_ci_runs_the_job_count_the_pins_were_balanced_for(workflow):
    ci = (REPO_ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    shards = re.search(r"^\s*shard:\s*\[([0-9, ]+)\]", ci, re.M)
    assert shards, f"{workflow}: the test matrix has no shard axis"
    listed = [int(x) for x in shards.group(1).split(",")]
    n = _sharding.PINS_FOR
    assert listed == list(range(1, n + 1)), workflow
    assert f'MADDENING_TEST_SHARD: "${{{{ matrix.shard }}}}/{n}"' in ci, workflow


def _workflow(name):
    return yaml.safe_load((REPO_ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def _pytest_args(job, step_name):
    (step,) = [s for s in job["steps"] if s.get("name") == step_name]
    # Continuation lines joined, so the invocation reads as one command.
    lines = [ln.strip() for ln in re.sub(r"\\\n", " ", step["run"]).splitlines()
             if "-m pytest" in ln and not ln.strip().startswith("#")]
    assert len(lines) == 1, lines
    # `${{ matrix.shard }}` -> `{{matrix.shard}}`, one shell word.
    tokens = shlex.split(re.sub(r"\$\{\{\s*(.*?)\s*\}\}", r"{{\1}}", lines[0]))
    return tokens[tokens.index("pytest") + 1:]


@pytest.mark.parametrize("workflow, job", [("ci.yml", "test"), ("slow-tests.yml", "slow")])
def test_no_matrix_leg_drops_or_adds_a_shard(workflow, job):
    # `exclude: [{shard: 4}]` would leave the shard axis intact -- so the
    # count check above passes -- while a quarter of the suite runs nowhere.
    matrix = _workflow(workflow)["jobs"][job]["strategy"]["matrix"]
    assert matrix["shard"] == list(range(1, _sharding.PINS_FOR + 1))
    for key in ("exclude", "include"):
        for leg in matrix.get(key) or []:
            assert "shard" not in leg, f"{workflow}: matrix {key} touches a shard: {leg}"


#: Every argument the default lane's pytest takes that does not change
#: which tests run.  Anything else fails the test below: an added
#: ``--ignore``, ``-k``, ``-m`` or ``--deselect`` would silently drop tests
#: from every shard at once.
REPORTING_ARGS = {"-v", "-rs", "--tb=short", "--durations=0", "--durations-min=1.0",
                  "-o", "junit_family=xunit1",
                  "--junitxml=test-results-shard{{matrix.shard}}.xml"}


@pytest.mark.parametrize("workflow, job, step, selection", [
    ("ci.yml", "test", "Run tests", ["tests/", "--ignore=tests/viz"]),
    ("slow-tests.yml", "slow", "Run full suite (slow lane)",
     ["tests/", "-m", "slow or not slow", "--ignore=tests/viz"]),
])
def test_the_sharded_lanes_run_the_whole_suite_but_viz(workflow, job, step, selection):
    args = _pytest_args(_workflow(workflow)["jobs"][job], step)
    chosen = [a for a in args if a not in REPORTING_ARGS]
    assert chosen == selection, (
        f"{workflow} {step!r}: pytest selects {chosen}, expected exactly {selection}; "
        "if a new argument only changes reporting, add it to REPORTING_ARGS")
    assert len(args) == len(set(args)), args


# ---------------------------------------------------------------------------
# Selection written anywhere but the command line
# ---------------------------------------------------------------------------

#: Environment variables pytest takes more arguments (``PYTEST_ADDOPTS``) or
#: plugins (``PYTEST_PLUGINS``) from.  Set in a workflow, either narrows
#: every shard's selection where no check of the command line looks.
PYTEST_ARGUMENT_ENV = ("PYTEST_ADDOPTS", "PYTEST_PLUGINS")


def _env_blocks(workflow: dict):
    """``(where, env)`` for the workflow, each job and each step."""
    yield "the workflow", workflow.get("env") or {}
    for job_id, job in workflow["jobs"].items():
        yield f"job {job_id}", job.get("env") or {}
        for step in job.get("steps", []):
            yield f"job {job_id} step {step.get('name')!r}", step.get("env") or {}


@pytest.mark.parametrize("workflow", ["ci.yml", "slow-tests.yml"])
def test_no_workflow_environment_adds_pytest_arguments(workflow):
    """``PYTEST_ADDOPTS: "--deselect tests/core"`` in a step's env drops tests/core
    from every shard, and the command-line check above never sees it.

    Every env block is read (workflow, job, step), and every ``run`` script
    too: ``echo PYTEST_ADDOPTS=... >> "$GITHUB_ENV"`` or an ``export`` sets
    it for the steps after.
    """
    wf = _workflow(workflow)
    offenders = [f"{where}: {key}" for where, env in _env_blocks(wf)
                 for key in env if key in PYTEST_ARGUMENT_ENV]
    for job_id, job in wf["jobs"].items():
        for step in job.get("steps", []):
            offenders += [f"job {job_id} step {step.get('name')!r}: {var} in its script"
                          for var in PYTEST_ARGUMENT_ENV if var in step.get("run", "")]
    assert not offenders, (
        f"{workflow} sets pytest arguments outside the command line, where the selection "
        f"checks do not look: {offenders}")


def test_the_environment_check_sees_every_place_a_variable_can_be_set():
    # The scan above is only a guard if it can fire.
    wf = {"env": {"PYTEST_ADDOPTS": "-x"},
          "jobs": {"j": {"env": {"PYTEST_PLUGINS": "p"},
                         "steps": [{"name": "s", "env": {"PYTEST_ADDOPTS": "-k x"}}]}}}
    found = [(where, key) for where, env in _env_blocks(wf) for key in env
             if key in PYTEST_ARGUMENT_ENV]
    assert found == [("the workflow", "PYTEST_ADDOPTS"), ("job j", "PYTEST_PLUGINS"),
                     ("job j step 's'", "PYTEST_ADDOPTS")]


#: ``[tool.pytest.ini_options]``, key by key.  ``addopts`` reaches every
#: lane's pytest, so anything added to it (``--ignore=tests/fmi``, ``-k``, a
#: wider ``-m``) changes what every shard runs; a new key (``norecursedirs``,
#: ``python_files``, ``junit_duration_report``) can do the same, so an
#: unknown one fails here until it is added with what it is for.
EXPECTED_INI = {
    "addopts": "-m 'not slow'",
    "testpaths": ["tests"],
    "filterwarnings": None,   # reporting, not selection: any value
    "markers": None,          # reporting, not selection: any value
}
#: Keys that may be absent, with the one value each may take when present.
#: ``junit_duration_report = "call"`` would drop setup and teardown -- where
#: a module fixture's compile is paid -- from every time the budget judges.
OPTIONAL_INI = {"junit_duration_report": "total"}


def test_the_pytest_configuration_selects_nothing_beyond_the_slow_mark():
    ini = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "tool"]["pytest"]["ini_options"]
    assert set(ini) - set(OPTIONAL_INI) == set(EXPECTED_INI), (
        f"[tool.pytest.ini_options] keys are {sorted(ini)}; a new key can change what every "
        "lane selects or what the time budget measures -- add it to EXPECTED_INI with why")
    for key, want in EXPECTED_INI.items():
        if want is not None:
            assert ini[key] == want, f"[tool.pytest.ini_options] {key} = {ini[key]!r}, expected {want!r}"
    for key, want in OPTIONAL_INI.items():
        assert ini.get(key, want) == want, f"[tool.pytest.ini_options] {key} = {ini[key]!r}, expected {want!r}"
    # pytest reads the first of these it finds before pyproject.toml, and a
    # `pytest.ini` wins even when it is empty: every setting above would go.
    for name, section in (("pytest.ini", ""), (".pytest.ini", ""), ("tox.ini", "[pytest]"),
                          ("setup.cfg", "[tool:pytest]")):
        path = REPO_ROOT / name
        if path.is_file():
            assert section and section not in path.read_text(encoding="utf-8"), (
                f"{name} holds pytest configuration, which pytest reads instead of pyproject.toml")


#: Names that, defined in a ``conftest.py``, choose which tests are collected
#: or kept.  The root conftest registers ``tests/_sharding.py``'s
#: ``pytest_collection_modifyitems`` as a plugin; it defines none itself.
COLLECTION_HOOKS = {"collect_ignore", "collect_ignore_glob", "pytest_ignore_collect",
                    "pytest_collection_modifyitems", "pytest_deselected", "pytest_collect_file",
                    "pytest_pycollect_makemodule", "pytest_plugins"}


def _collection_hooks(source: str) -> list[str]:
    found = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in COLLECTION_HOOKS:
            found.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            found += [t.id for t in targets if isinstance(t, ast.Name) and t.id in COLLECTION_HOOKS]
    return found


def test_no_conftest_chooses_which_tests_are_collected():
    """``collect_ignore = ["fmi"]`` in a conftest drops a directory from every shard."""
    offenders = {}
    for path in sorted((REPO_ROOT / "tests").rglob("conftest.py")):
        hooks = _collection_hooks(path.read_text(encoding="utf-8"))
        if hooks:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = hooks
    assert not offenders, (
        f"these conftests choose which tests run, where no selection check looks: {offenders}")


def test_the_conftest_scan_finds_what_it_is_for():
    source = ("collect_ignore = ['fmi']\n"
              "collect_ignore_glob: list = ['*.py']\n"
              "def pytest_collection_modifyitems(config, items):\n    pass\n"
              "def pytest_configure(config):\n    pass\n"
              "def ball_node():\n    pass\n")
    assert _collection_hooks(source) == [
        "collect_ignore", "collect_ignore_glob", "pytest_collection_modifyitems"]


# ---------------------------------------------------------------------------
# The root conftest switches on what CI asks for
# ---------------------------------------------------------------------------

#: Time the kept test spends in a fixture's setup, which the JUnit time the
#: budget judges must include (``junit_duration_report = "total"``, the
#: default; ``"call"`` would report the test body alone).
SETUP_SECONDS = 0.2


def _files_on_each_shard_of_two() -> tuple[str, str]:
    """A test file path on shard 1 of 2 and one on shard 2 of 2."""
    found: dict[int, str] = {}
    for i in range(100):
        rel = f"tests/test_scratch_{i}.py"
        found.setdefault(_sharding.shard_of(rel, 2), rel)
        if len(found) == 2:
            return found[1], found[2]
    raise AssertionError("no two of 100 names hash to different shards of 2")


@pytest.fixture(scope="module")
def shard_run(tmp_path_factory):
    """One pytest run of a two-file tree through the real root conftest and ini.

    The scratch tree holds copies of ``tests/conftest.py`` (and the two
    modules it registers) and of ``pyproject.toml``, and runs as one CI
    shard does: ``MADDENING_TEST_SHARD=1/2`` and ``MADDENING_TEST_JAX_TIMING=1``.
    """
    root = tmp_path_factory.mktemp("shard_run")
    (root / "tests").mkdir()
    for name in ("__init__.py", "conftest.py", "_sharding.py", "_jax_timing.py"):
        shutil.copy(REPO_ROOT / "tests" / name, root / "tests" / name)
    shutil.copy(REPO_ROOT / "pyproject.toml", root / "pyproject.toml")
    kept, dropped = _files_on_each_shard_of_two()
    (root / kept).write_text(
        "import time\n\nimport jax\nimport jax.numpy as jnp\nimport pytest\n\n\n"
        "@pytest.fixture\n"
        "def compiled():\n"
        f"    time.sleep({SETUP_SECONDS})\n"
        "    return jax.jit(lambda x: x * 1.6180339 + 0.5772156)(jnp.ones(3))\n\n\n"
        "def test_kept(compiled):\n"
        "    assert compiled.shape == (3,)\n")
    (root / dropped).write_text("def test_dropped():\n    pass\n")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("MADDENING_TEST_", "PYTEST_", "JAX_COMPILATION_CACHE",
                                "JAX_PERSISTENT_CACHE"))}
    env.update(MADDENING_TEST_SHARD="1/2", MADDENING_TEST_JAX_TIMING="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", JAX_PLATFORMS="cpu")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-p", "no:cacheprovider",
         "--junitxml=report.xml", "-o", "junit_family=xunit1"],
        cwd=root, env=env, capture_output=True, text=True, timeout=300)
    cases = list(ET.parse(root / "report.xml").getroot().iter("testcase")) \
        if (root / "report.xml").is_file() else []
    return SimpleNamespace(proc=proc, cases=cases, kept=kept, dropped=dropped)


def test_the_root_conftest_keeps_only_this_shards_files(shard_run):
    """``MADDENING_TEST_SHARD`` must reach ``tests/_sharding.py`` through the conftest.

    If the conftest stopped registering the plugin, every CI shard would
    run the whole suite: four times the CI time, no test dropped, and every
    check of ``_sharding.py`` itself still green.
    """
    out = shard_run.proc.stdout + shard_run.proc.stderr
    assert shard_run.proc.returncode == 0, out[-3000:]
    assert "test shard: 1/2 (by file; see tests/_sharding.py)" in out, out[-3000:]
    assert re.search(r"\b1 passed, 1 deselected\b", out), out[-3000:]
    assert [c.get("file") for c in shard_run.cases] == [shard_run.kept], (
        f"the run kept {[c.get('file') for c in shard_run.cases]}, expected only "
        f"{shard_run.kept} ({shard_run.dropped} is on shard 2)")


def test_the_root_conftest_records_jax_timing_into_the_junit_report(shard_run):
    """``MADDENING_TEST_JAX_TIMING=1`` must attach the timing properties.

    Without them the budget's summary has no compile / tracing split, no
    cache hit rates, and cannot check a shard's warm label.  The kept test
    compiles a program in its fixture, so its trace and compile times are
    non-zero -- the listeners are really registered, not just the names.
    """
    assert shard_run.proc.returncode == 0, shard_run.proc.stdout[-3000:]
    (case,) = shard_run.cases
    props = {p.get("name"): float(p.get("value")) for p in case.iter("property")}
    assert set(props) == set(_jax_timing.PROPERTIES), sorted(props)
    assert props["jax_trace_s"] > 0 and props["jax_compile_s"] > 0, props
    assert props["subprocesses"] == 0, props


def test_a_shards_junit_time_includes_fixture_setup(shard_run):
    """The budget judges setup + call + teardown, as it says.

    A module fixture's compile is charged to the first test that needs it,
    in its setup.  ``junit_duration_report = "call"`` would report the test
    body alone, and every such compile would vanish from the budget.
    """
    assert shard_run.proc.returncode == 0, shard_run.proc.stdout[-3000:]
    (case,) = shard_run.cases
    assert float(case.get("time")) >= SETUP_SECONDS, (
        f"the kept test's JUnit time is {case.get('time')} s, less than the "
        f"{SETUP_SECONDS} s its fixture's setup took")
