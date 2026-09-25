"""The CI workflows do what their comments and the testing standards say.

Each test pins one property of ``.github/workflows/*.yml`` that nothing
else would notice breaking, because a workflow that runs less than it
claims still goes green:

* the slow lane's schedule covers ``main`` only, and every description of
  it says so;
* a pull request that touches only documentation skips the test lanes,
  so no test outside ``tests/compliance`` (the one job that always runs)
  may read a path the change classifier calls documentation;
* a push always runs every lane;
* a pull request whose diff can move a compile cost onto another test runs
  cold, as does a push and a ``[cold-ci]`` pull request; any other pull
  request runs warm;
* the compilation cache is saved only after a passing test step, and only
  once its unreadable entries are gone;
* verify-hypothesis runs the slow-marked properties too;
* the slow lane installs the tools its C tests self-skip without, and
  files an abort as an abort, not as out-of-memory;
* every test that needs ``usd-core`` runs in the one job that installs it,
  and none of them is slow-marked (that job does not run slow tests).

Workflow steps are read with ``yaml.safe_load`` and, where it is their
behaviour that matters, run: the shell scripts under ``bash`` and the
issue classifier under ``node``, with the ``${{ }}`` expressions they use
filled in (an expression the tests do not know fails them, rather than
running with a guess).
"""

from __future__ import annotations

import ast
import fnmatch
import functools
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> set[str]:
    # YAML 1.1 reads a bare `on:` key as the boolean True.
    on = workflow["on"] if "on" in workflow else workflow[True]
    return {on} if isinstance(on, str) else set(on)


def _step(job: dict, name: str) -> dict:
    found = [s for s in job["steps"] if s.get("name") == name]
    assert len(found) == 1, f"expected one step named {name!r}, found {len(found)}"
    return found[0]


def _render(text: str, values: dict[str, str]) -> str:
    """Fill in ``${{ expr }}``; an expression not in ``values`` fails the test."""
    def sub(m):
        expr = m.group(1).strip()
        assert expr in values, f"the test does not know the expression {expr!r}; add it"
        return values[expr]
    return re.sub(r"\$\{\{(.*?)\}\}", sub, text)


def _literals(tree: ast.AST) -> tuple[list[tuple[int, str]], set[str]]:
    """A module's string constants ``(line, value)`` and the module names it imports.

    ``from a import b`` records both ``a`` and ``a.b``, so ``from maddening
    import usd`` reads as an import of ``maddening.usd``.
    """
    strings, imports = [], set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.append((node.lineno, node.value))
        elif isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.add(node.module)
            imports.update(f"{node.module}.{a.name}" for a in node.names)
    return strings, imports


@functools.lru_cache(maxsize=None)
def _test_modules() -> dict[str, tuple[list[tuple[int, str]], set[str]]]:
    """``_literals`` of every test module outside tests/compliance, parsed once."""
    out = {}
    for path in sorted((REPO_ROOT / "tests").rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if not rel.startswith("tests/compliance/"):
            out[rel] = _literals(ast.parse(path.read_text(encoding="utf-8"), filename=rel))
    return out


def _logical_lines(script: str) -> list[str]:
    """A shell script's commands: continuations joined, comments and blanks dropped."""
    joined = re.sub(r"\\\n", " ", script)
    return [line.strip() for line in joined.splitlines()
            if line.strip() and not line.strip().startswith("#")]


# ---------------------------------------------------------------------------
# The slow lane's schedule
# ---------------------------------------------------------------------------

#: A description of the slow lane's schedule.
SCHEDULE_CLAIM = re.compile(
    r"three times a week|\b3x a week"
    r"|\bMon(?:day)?\b[\s,/]+(?:and\s+)?Wed(?:nesday)?\b[\s,/]+(?:and\s+)?Fri(?:day)?\b",
    re.IGNORECASE)
#: Where such a description could be written; every text file here is read.
CLAIM_PLACES = (".github", "docs", "scripts", "tests", "src",
                "README.md", "CONTRIBUTING.md", "CHANGELOG.md")
CLAIM_SUFFIXES = {".md", ".py", ".yml", ".yaml", ".txt", ".rst", ".toml"}


def _sentences(text: str):
    """Sentences, with comment leaders removed so a wrapped comment reads as one."""
    lines = [re.sub(r"^\s*(?:#+|//+|\*)\s?", "", line) for line in text.splitlines()]
    for paragraph in re.split(r"\n\s*\n", "\n".join(lines)):
        flat = " ".join(paragraph.split())
        for cell in flat.split(" | "):             # one Markdown table cell at a time
            yield from re.split(r"(?<=[.!?])\s+", cell)


def test_every_description_of_the_slow_lane_schedule_says_it_covers_main_only():
    """GitHub runs a ``schedule:`` workflow on the default branch and nowhere else.

    So "the slow tests run three times a week" is true of ``main`` only; a
    release branch gets a slow-lane run when someone dispatches one.  Five
    places said otherwise, one of them calling those runs the basis for
    triage.  Any sentence that states the schedule must name ``main`` (or
    the default branch).  If the triggers change, this fails first: re-read
    every claim it finds, then update the expected triggers here.
    """
    triggers = _triggers(_workflow("slow-tests.yml"))
    assert triggers == {"schedule", "workflow_dispatch"}, (
        f"slow-tests.yml's triggers are now {sorted(triggers)}: re-check what the docs "
        "claim about when the slow lane runs, then update this test")
    offenders = []
    for place in CLAIM_PLACES:
        root = REPO_ROOT / place
        if not root.exists():
            continue
        files = [root] if root.is_file() else sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix in CLAIM_SUFFIXES and "__pycache__" not in p.parts)
        for path in files:
            if path == Path(__file__).resolve():
                continue                            # the pattern above is not a claim
            for sentence in _sentences(path.read_text(encoding="utf-8", errors="replace")):
                if SCHEDULE_CLAIM.search(sentence) and not re.search(
                        r"\bmain\b|default branch", sentence, re.IGNORECASE):
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {sentence[:200]}")
    assert not offenders, (
        "these describe the slow lane's schedule without saying it covers `main` only "
        "(release branches run it only when dispatched by hand):\n" + "\n".join(offenders))


# ---------------------------------------------------------------------------
# The change classifier: docs-only pull requests, and pushes
# ---------------------------------------------------------------------------

CLASSIFY = "Classify changed files"

#: ``"<file>: <literal>" -> reason`` for a literal outside tests/compliance
#: that looks like a documentation path but is not read as one.  Add an
#: entry only with the reason; prefer moving the test.
NOT_A_DOCS_READ: dict[str, str] = {
    "tests/api/test_bearer_auth.py: /docs":
        "the URL route of the API server's generated docs page, not a file",
}


def _docs_patterns() -> list[str]:
    """The ``case`` patterns the classifier files as documentation."""
    script = _step(_workflow("ci.yml")["jobs"]["changes"], CLASSIFY)["run"]
    m = re.search(r"^\s*([^\s()]+)\)\s*\n\s*echo \"  \$f  \(docs\)\"", script, re.M)
    assert m, "the classifier has no `case` arm that files a path as (docs)"
    return m.group(1).split("|")


@functools.lru_cache(maxsize=None)
def _as_regex(patterns: tuple[str, ...]) -> re.Pattern:
    # POSIX `case` semantics, which fnmatch shares: `*` matches `/` too.
    return re.compile("|".join(fnmatch.translate(p) for p in patterns))


def _names_a_docs_path(literal: str, patterns: list[str], docs_only_names: set[str]) -> bool:
    if not literal or len(literal) > 300 or any(c.isspace() for c in literal):
        return False                                # prose, not a path
    docs = _as_regex(tuple(patterns))
    parts = [p for p in literal.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    for i in range(len(parts)):
        tail = "/".join(parts[i:])
        # `tail + "/"` catches a directory component: `ROOT / "docs" / ...`.
        if docs.match(tail) or docs.match(tail + "/"):
            return True
    return bool(parts) and parts[-1] in docs_only_names


def test_no_test_outside_compliance_reads_a_path_the_classifier_calls_docs():
    """A docs-only pull request runs ``tests/compliance`` and no other test.

    The ``changes`` job in ``ci.yml`` skips every test lane when a pull
    request touches only paths it files as documentation.  A test elsewhere
    that reads one of those paths therefore never meets the change that
    breaks it -- two did (the anomaly registry's key list, and a scan of
    ``docs/`` and ``README.md`` for stale jax pins).  This scans every
    string literal in ``tests/`` outside ``tests/compliance`` for a path the
    classifier's own patterns match, or for the name of a file that exists
    only under ``docs/`` (``known_anomalies.yaml``, ``stable_api.json``).

    What it cannot see: a path assembled from no literal at all (a
    ``rglob("*")`` over the repository root, a path from an environment
    variable or a config file), and documentation read by code a test
    calls -- a module in ``src/``, a script under ``scripts/``, a
    subprocess.  When this was written none of those opens a docs path
    for a test outside ``tests/compliance``.  Literals containing
    whitespace are taken as prose (assertion messages, docstrings), not
    paths.
    """
    patterns = _docs_patterns()
    docs_only_names = {p.name for p in (REPO_ROOT / "docs").rglob("*")
                       if p.is_file() and not any(fnmatch.fnmatchcase(p.name, pat)
                                                  for pat in patterns)}
    offenders = []
    for rel, (strings, _) in _test_modules().items():
        for lineno, value in strings:
            if (_names_a_docs_path(value, patterns, docs_only_names)
                    and f"{rel}: {value}" not in NOT_A_DOCS_READ):
                offenders.append(f"{rel}:{lineno}: {value!r}")
    assert not offenders, (
        "these tests name a path the change classifier calls documentation, so a "
        "docs-only pull request would skip them; move the part that reads it into "
        "tests/compliance/:\n" + "\n".join(offenders))


def test_the_docs_path_scan_recognises_what_a_test_would_write():
    # The scan above is only a guard if it can fire.
    patterns = _docs_patterns()
    names = {"known_anomalies.yaml"}
    for literal in ("docs", "docs/validation/known_anomalies.yaml", "README.md", "*.md",
                    ".md", "../docs/x", "known_anomalies.yaml", "LICENSE", "CITATION.cff",
                    "plans", ".claude/settings.json", "src/maddening/examples/README.md"):
        assert _names_a_docs_path(literal, patterns, names), literal
    for literal in ("src", "tests/core/test_x.py", "pyproject.toml", "a docs/ path in prose",
                    "documentation", "docstring", "typing_tiers.json", ""):
        assert not _names_a_docs_path(literal, patterns, names), literal


_GIT_ENV = {
    "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


def _git(repo: Path, *args: str) -> str:
    env = {"PATH": os.environ["PATH"], "HOME": str(repo), **_GIT_ENV}
    return subprocess.run(["git", *args], cwd=repo, env=env, check=True,
                          capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, rel: str, text: str) -> str:
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text(text)
    _git(repo, "add", rel)
    _git(repo, "commit", "-q", "-m", rel)
    return _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("event, change, lanes", [
    ("push", "docs", "true"),
    ("push", "code", "true"),
    ("pull_request", "docs", "false"),
    ("pull_request", "code", "true"),
])
def test_a_push_runs_every_lane_and_only_a_docs_only_pull_request_skips_them(
        tmp_path, event, change, lanes):
    """The classifier, run as GitHub runs it, in a scratch repository.

    A push must run every lane whatever its diff: push runs share a
    concurrency group that replaces a *pending* run with the next one, so
    after merges A (code), B (code) and C (docs), C's diff from
    ``github.event.before`` (B) is docs only although B's code was never
    tested -- C would skip every lane and save no cache.  Here ``before``
    is the commit just behind a docs-only head, exactly C's case.
    """
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.fail("git and bash are needed to run the classifier; CI runners have both")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    base = _commit(repo, "src/m.py", "x = 1\n")
    _commit(repo, "docs/d.md" if change == "docs" else "src/m.py", "changed\n")
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    step = _step(_workflow("ci.yml")["jobs"]["changes"], CLASSIFY)
    context = {"github.event_name": event,
               "github.base_ref": "main" if event == "pull_request" else "",
               "github.event.before": base if event == "push" else ""}
    out = tmp_path / "github_output"
    env = {"PATH": os.environ["PATH"], "HOME": str(repo), "GITHUB_OUTPUT": str(out), **_GIT_ENV,
           **{k: _render(str(v), context) for k, v in (step.get("env") or {}).items()}}
    proc = subprocess.run(["bash", "-e", "-c", _render(step["run"], context)], cwd=repo,
                          env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    outputs = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert outputs["code"] == lanes, f"{event} of a {change} change: code={outputs['code']}\n{proc.stdout}"


#: The test file every "move a compile cost" case below edits.
_SHARED = """import pytest


@pytest.mark.slow
def test_marked_slow():
    pass


def test_pays_for_the_shared_program():
    x = 1


def test_reuses_the_shared_program():
    x = 2
"""

#: ``case -> (path, new text, expected cold)``, each a PR commit on top of a
#: base holding ``_SHARED``, an allowlist and a shard assignment.
_COLD_CASES = {
    "slow mark added": ("tests/test_shared.py", _SHARED.replace(
        "\ndef test_pays", "\n@pytest.mark.slow\ndef test_pays"), "true"),
    "slow mark removed": ("tests/test_shared.py", _SHARED.replace(
        "@pytest.mark.slow\n", ""), "true"),
    "slow mark in a parameter": ("tests/test_shared.py", _SHARED + (
        "\n\n@pytest.mark.parametrize('n', [1, pytest.param(2, marks=pytest.mark.slow)])\n"
        "def test_new(n):\n    pass\n"), "true"),
    "test deleted": ("tests/test_shared.py", _SHARED.split("\n\ndef test_pays")[0] + "\n", "true"),
    "test moved after its sibling": ("tests/test_shared.py", _SHARED.replace(
        "def test_pays_for_the_shared_program():\n    x = 1\n\n\n", "") + (
        "\n\ndef test_pays_for_the_shared_program():\n    x = 1\n"), "true"),
    "test renamed": ("tests/test_shared.py", _SHARED.replace("test_pays_for", "test_now_pays_for"),
                     "true"),
    "test file deleted": ("tests/test_shared.py", None, "true"),
    "allowlist edited": ("tests/duration_allowlist.txt", "t # kept: why\n", "true"),
    "shard assignment edited": ("tests/_sharding.py", "PINS = {'tests/test_shared.py': 2}\n",
                                "true"),
    "test added at the end": ("tests/test_shared.py", _SHARED + (
        "\n\ndef test_new():\n    pass\n"), "false"),
    "test body changed": ("tests/test_shared.py", _SHARED.replace("x = 2", "x = 3"), "false"),
    "library code changed": ("src/m.py", "x = 2\n", "false"),
}


def _run_classifier(repo: Path, tmp_path: Path, event: str) -> dict[str, str]:
    step = _step(_workflow("ci.yml")["jobs"]["changes"], CLASSIFY)
    context = {"github.event_name": event,
               "github.base_ref": "main" if event == "pull_request" else ""}
    out = tmp_path / "github_output"
    env = {"PATH": os.environ["PATH"], "HOME": str(repo), "GITHUB_OUTPUT": str(out), **_GIT_ENV,
           **{k: _render(str(v), context) for k, v in (step.get("env") or {}).items()}}
    proc = subprocess.run(["bash", "-e", "-c", _render(step["run"], context)], cwd=repo,
                          env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return dict(line.split("=", 1) for line in out.read_text().splitlines())


@pytest.mark.parametrize("case", sorted(_COLD_CASES))
def test_a_pull_request_that_can_move_a_compile_cost_runs_cold(tmp_path, case):
    """A warm run cannot see a compile cost that moves from one test to another.

    The first test to compile a shared program pays for every later test
    that reuses it.  When a PR slow-marks, deletes or moves that test, the
    next test inherits the compile; on the PR's warm run it reads the
    program from the base branch's cache and looks fast, and the cold run
    after the merge fails it (commit 04cad05 found one by hand).  The
    classifier therefore asks for a cold run when the diff adds or removes
    a slow mark, removes a test function, or edits the allowlist or the
    shard assignment -- and only then, or every PR would lose the cache.
    """
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.fail("git and bash are needed to run the classifier; CI runners have both")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, "src/m.py", "x = 1\n")
    _commit(repo, "tests/duration_allowlist.txt", "# empty\n")
    _commit(repo, "tests/_sharding.py", "PINS = {}\n")
    base = _commit(repo, "tests/test_shared.py", _SHARED)
    rel, text, cold = _COLD_CASES[case]
    if text is None:
        _git(repo, "rm", "-q", rel)
        _git(repo, "commit", "-q", "-m", f"rm {rel}")
    else:
        _commit(repo, rel, text)
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    outputs = _run_classifier(repo, tmp_path, "pull_request")
    assert outputs["code"] == "true"
    assert outputs["cold"] == cold, f"{case}: cold={outputs['cold']}"


def test_the_cold_verdict_reaches_the_test_job():
    ci = _workflow("ci.yml")
    assert ci["jobs"]["changes"]["outputs"]["cold"] == "${{ steps.classify.outputs.cold }}"
    assert ci["jobs"]["test"]["needs"] == "changes"
    step = _step(ci["jobs"]["test"], "Compilation cache mode")
    assert step["env"]["COLD_DIFF"] == "${{ needs.changes.outputs.cold }}"


#: Everything the "Compilation cache mode" step reads from GitHub.
def _cache_mode_context(event: str) -> dict[str, str]:
    return {"github.event_name": event, "github.repository": "o/r",
            "github.event.pull_request.head.sha": "abc123" if event == "pull_request" else "",
            "github.token": "t", "runner.os": "Linux", "matrix.python-version": "3.12",
            "matrix.jax-version": "0.10.2", "matrix.shard": "3"}


@pytest.mark.parametrize("event, message, cold_diff, mode", [
    ("push", "a merge", "", "cold"),
    ("push", "a merge [cold-ci]", "", "cold"),
    ("pull_request", "fix: a thing", "false", "warm"),
    ("pull_request", "perf: before/after [cold-ci]", "false", "cold"),
    ("pull_request", "test: slow-mark a test", "true", "cold"),
])
def test_pushes_start_cold_and_a_pull_request_restores_the_base_cache_unless_told_not_to(
        tmp_path, event, message, cold_diff, mode):
    """The mode step, run under bash as GitHub runs it.

    A push must start empty: its cache is saved and becomes what every PR
    restores, so a push that restored one would save the base's programs
    plus its own and report none of their compile time.  A PR restores
    that cache unless its head commit says ``[cold-ci]`` or the diff can
    move a compile cost (``changes.outputs.cold``).
    """
    if shutil.which("bash") is None:
        pytest.fail("bash is needed to run the step; CI runners have it")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # The step asks the API for the head commit's message; answer with this case's.
    (bin_dir / "gh").write_text('#!/bin/sh\nprintf "%s" "$FAKE_COMMIT_MESSAGE"\n')
    (bin_dir / "gh").chmod(0o755)
    step = _step(_workflow("ci.yml")["jobs"]["test"], "Compilation cache mode")
    context = {**_cache_mode_context(event), "needs.changes.outputs.cold": cold_diff}
    out = tmp_path / "github_output"
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "GITHUB_OUTPUT": str(out),
           "FAKE_COMMIT_MESSAGE": message,
           **{k: _render(str(v), context) for k, v in step["env"].items()}}
    proc = subprocess.run(["bash", "-e", "-c", _render(step["run"], context)], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    outputs = dict(line.split("=", 1) for line in out.read_text().splitlines())
    assert outputs["mode"] == mode, f"{event} {message!r} cold={cold_diff}: {outputs}"
    # One cache per shard and lane (whatever else the key holds).
    assert re.fullmatch(r"jaxcc-v1-Linux-.*py3\.12-jax0\.10\.2-shard3of4", outputs["key"]), outputs["key"]


def test_only_a_warm_run_restores_a_cache_and_only_a_push_saves_one():
    steps = _workflow("ci.yml")["jobs"]["test"]["steps"]
    restore = _step({"steps": steps}, "Restore compilation cache")
    assert restore["if"] == "steps.cc.outputs.mode == 'warm'"
    assert restore["uses"].startswith("actions/cache/restore@")
    assert not any(s.get("uses", "").startswith("actions/cache@") for s in steps), (
        "actions/cache (restore and save) in the test job: PRs must never save")
    save = _step({"steps": steps}, "Save compilation cache")
    assert "github.event_name == 'push'" in save["if"]


# ---------------------------------------------------------------------------
# The compilation cache is saved only when it can be trusted
# ---------------------------------------------------------------------------


def test_the_cache_is_saved_only_after_a_passing_test_step_and_a_prune():
    """A crashed run may have killed a writer mid-entry; JAX never repairs one.

    Saving after a failed test step, or before the prune, would hand a
    truncated entry to every pull request that restores the cache.
    """
    steps = _workflow("ci.yml")["jobs"]["test"]["steps"]
    names = [s.get("name") for s in steps]
    run_tests = _step({"steps": steps}, "Run tests")
    prune = _step({"steps": steps}, "Drop unreadable compilation-cache entries")
    save = _step({"steps": steps}, "Save compilation cache")
    assert run_tests.get("id") == "tests"
    assert names.index("Run tests") < names.index(prune["name"]) < names.index(save["name"])
    assert "scripts/prune_jax_cache.py" in prune["run"]
    assert "${{ runner.temp }}/jax-cache" in prune["run"]
    assert prune.get("id") == "prune-cache"
    for step in (prune, save):
        cond = step["if"]
        assert "steps.tests.outcome == 'success'" in cond, (step["name"], cond)
        assert "github.event_name == 'push'" in cond, (step["name"], cond)
        assert "||" not in cond, (step["name"], cond)
    assert "steps.prune-cache.outcome == 'success'" in save["if"]
    assert save["with"]["path"] == "${{ runner.temp }}/jax-cache"


# ---------------------------------------------------------------------------
# verify-hypothesis runs the slow-marked properties
# ---------------------------------------------------------------------------


def _marker_selects(expression: str, markers: set[str]) -> bool:
    from _pytest.mark.expression import Expression
    return Expression.compile(expression).evaluate(lambda name, **_: name in markers)


def test_verify_hypothesis_runs_the_slow_properties_too():
    """The one job that runs ``tests/verification/hypothesis`` at ``ci`` depth.

    pyproject's ``addopts`` carries ``-m 'not slow'``; the job overrides it
    with ``-m "slow or not slow"``, passed through the draw-rejection
    audit's ``--pytest-arg``.  Without it the properties slow-marked to
    keep the default lane in budget -- eight of them, each saying "still in
    verify-hypothesis" at its mark -- run in no job on any push.
    """
    job = _workflow("ci.yml")["jobs"]["verify-hypothesis"]
    step = _step(job, "Run hypothesis property tests (with draw-rejection audit)")
    (line,) = [ln for ln in _logical_lines(step["run"]) if "audit_property_rejection.py" in ln]
    tokens = shlex.split(line)
    args = tokens[tokens.index("scripts/audit_property_rejection.py") + 1:]
    pytest_args, paths, it = [], [], iter(args)
    for a in it:
        if a.startswith("--pytest-arg="):
            pytest_args.append(a.split("=", 1)[1])
        elif a == "--pytest-arg":
            pytest_args.append(next(it))
        elif not a.startswith("-"):
            paths.append(a)
    assert paths == ["tests/verification/hypothesis/"], paths
    assert "--check" in args, "the rejection audit no longer gates"
    expressions = [pytest_args[i + 1] for i, a in enumerate(pytest_args[:-1]) if a == "-m"]
    expressions += [a[2:] for a in pytest_args if a.startswith("-m") and len(a) > 2]
    assert expressions, f"no -m reaches pytest, so addopts' -m 'not slow' applies: {pytest_args}"
    # pytest takes the last -m it is given.
    assert _marker_selects(expressions[-1], {"slow"}) and _marker_selects(expressions[-1], set()), (
        f"-m {expressions[-1]!r} does not select both slow and other tests")
    assert step["env"]["MADDENING_HYPOTHESIS_PROFILE"] == "ci"


def test_the_marker_check_can_tell_a_selection_that_drops_slow_tests():
    assert _marker_selects("slow or not slow", {"slow"}) and _marker_selects("slow or not slow", set())
    assert not _marker_selects("not slow", {"slow"})
    assert not _marker_selects("slow", set())


# ---------------------------------------------------------------------------
# The C tests' tools are installed where the tests run
# ---------------------------------------------------------------------------


def _apt_installs(job: dict, before: str) -> set[str]:
    """Packages ``apt-get install``ed by unconditional steps before the step ``before``."""
    names = [s.get("name") for s in job["steps"]]
    packages = set()
    for s in job["steps"][:names.index(before)]:
        if "if" in s or s.get("continue-on-error"):
            continue
        for line in _logical_lines(s.get("run", "")):
            for command in re.split(r"&&|;|\|\|", line):
                tokens = shlex.split(command)
                if "apt-get" in tokens and "install" in tokens:
                    packages.update(t for t in tokens[tokens.index("install") + 1:]
                                    if not t.startswith("-"))
    return packages


@pytest.mark.parametrize("workflow, job, run_step, needed", [
    # test_fuzz_under_valgrind runs per push and self-skips without valgrind.
    ("ci.yml", "test", "Run tests", {"valgrind"}),
    # The unit binary under valgrind and the libFuzzer campaign are
    # slow-marked; this is the only lane that runs them, and each self-skips
    # without its tool.
    ("slow-tests.yml", "slow", "Run full suite (slow lane)", {"valgrind", "clang"}),
])
def test_the_lanes_install_the_tools_the_c_tests_skip_without(workflow, job, run_step, needed):
    installed = _apt_installs(_workflow(workflow)["jobs"][job], run_step)
    assert needed <= installed, (
        f"{workflow} {job}: {sorted(needed - installed)} not installed before {run_step!r}; "
        "tests/fmi/test_c_unit.py would skip and run nowhere")
    c_tests = (REPO_ROOT / "tests" / "fmi" / "test_c_unit.py").read_text(encoding="utf-8")
    for tool in needed:
        assert f'shutil.which("{tool}")' in c_tests, f"test_c_unit.py no longer looks for {tool}"


# ---------------------------------------------------------------------------
# The slow lane's failure classifier
# ---------------------------------------------------------------------------

CLASSIFY_FAILURE = "Classify failure and open tracking issue"


def _classify_failure(exit_code: str, elapsed_sec: int) -> list[dict]:
    step = _step(_workflow("slow-tests.yml")["jobs"]["slow"], CLASSIFY_FAILURE)
    script = _render(step["with"]["script"], {
        "steps.pytest.outputs.exit_code": exit_code,
        "steps.pytest.outputs.elapsed_sec": str(elapsed_sec),
        "matrix.python-version": "3.12", "matrix.jax-version": "0.10.2", "matrix.shard": "1",
    })
    harness = (
        "const out = [];\n"
        "const context = {sha: '0123456789abcdef', serverUrl: 'https://github.invalid',\n"
        "  repo: {owner: 'o', repo: 'r'}, runId: 1};\n"
        "const core = {info: (m) => out.push({info: m})};\n"
        "const github = {rest: {issues: {create: async (x) => out.push(x)}}};\n"
        "(async () => {\n" + script + "\n})().then(() => console.log(JSON.stringify(out)));\n"
    )
    proc = subprocess.run(["node", "-e", harness], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_the_slow_lane_files_an_abort_as_an_abort_and_names_both_causes():
    """Exit 134 is SIGABRT: out of memory, or a native assertion / heap corruption.

    The slow lane loads native code into the pytest process (the FMU C
    wrapper, through ctypes and fmpy), and glibc aborts on a corrupted
    heap -- so filing every 134 as out-of-memory sends the reader after the
    wrong bug.  The memory figure is the runner's, 16 GB.
    """
    step = _step(_workflow("slow-tests.yml")["jobs"]["slow"], CLASSIFY_FAILURE)
    text = step["with"]["script"]
    assert "7 GB" not in text and "16 GB" in text
    if shutil.which("node") is None:
        pytest.skip("node is not installed here, so the classifier's JavaScript cannot "
                    "run; GitHub's runners have it, and the static checks above ran")
    (abort,) = _classify_failure("134", 600)
    assert "abort" in abort["title"].lower() and "out-of-memory" not in abort["title"]
    body = abort["body"]
    assert "out-of-memory" in body and "heap-corruption" in body and "assertion" in body
    assert "valgrind" in body
    (oom,) = _classify_failure("137", 600)
    assert "out-of-memory" in oom["title"] and "16 GB" in oom["body"]
    (timeout,) = _classify_failure("137", 175 * 60)
    assert "timeout" in timeout["title"]
    (timeout,) = _classify_failure("124", 175 * 60)
    assert "timeout" in timeout["title"]
    for ordinary in ("1", "139"):
        (info,) = _classify_failure(ordinary, 600)
        assert "ordinary failure" in info["info"]


# ---------------------------------------------------------------------------
# Tests that need usd-core run in the job that installs it
# ---------------------------------------------------------------------------

#: Files outside tests/usd and tests/compliance that mention pxr or
#: maddening.usd but do not need usd-core installed, with the reason.
USD_NOT_NEEDED = {
    "tests/core/test_solver_ift_lineax_is_core.py":
        "lists pxr among the modules that importing the core must not pull in",
}


def _needs_usd(literals: tuple[list[tuple[int, str]], set[str]]) -> bool:
    """Imports pxr / maddening.usd, or names one as a string (importorskip, find_spec)."""
    def is_usd(name: str) -> bool:
        return name in ("pxr", "maddening.usd") or name.startswith(("pxr.", "maddening.usd."))

    strings, imports = literals
    return any(is_usd(m) for m in imports) or any(is_usd(v) for _, v in strings)


def _examples_needing_usd() -> list[str]:
    root = REPO_ROOT / "src" / "maddening" / "examples"
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py")
                  if _needs_usd(_literals(ast.parse(p.read_text(encoding="utf-8")))))


def _usd_job_targets() -> list[str]:
    job = _workflow("ci.yml")["jobs"]["test-usd"]
    installs = " ".join(s.get("run", "") for s in job["steps"])
    assert '".[ci,usd]"' in installs, "test-usd no longer installs the usd extra"
    runs = [line for s in job["steps"] for line in _logical_lines(s.get("run", ""))
            if "-m pytest" in line]
    assert len(runs) == 1, runs
    tokens = shlex.split(runs[0])
    args = tokens[tokens.index("pytest") + 1:]
    assert not any(a in ("-m", "-k") or a.startswith(("-m=", "-k=", "--deselect", "--ignore"))
                   for a in args), f"test-usd narrows its selection: {args}"
    return [a for a in args if not a.startswith("-")]


def _selected(rel: str, targets: list[str]) -> bool:
    return any(rel == t or t.startswith(rel + "::") or rel.startswith(t.rstrip("/") + "/")
               for t in targets)


def test_every_test_that_needs_usd_core_runs_in_the_job_that_installs_it():
    """Only ``test-usd`` installs ``usd-core`` and runs tests with it.

    The sharded lanes and the slow lane install ``.[ci]``, so a test that
    needs ``pxr`` skips there; outside ``tests/usd`` twelve such tests ran
    in no job at all.  Every test file (outside ``tests/usd`` and
    ``tests/compliance``, whose job installs the extra) that imports or
    skips on ``pxr`` or ``maddening.usd`` must be in ``test-usd``'s
    selection, or in ``USD_NOT_NEEDED`` with the reason.  The examples are
    checked too: ``tests/test_examples_smoke.py`` resolves their imports
    and skips an example whose ``maddening.usd`` cannot be imported.
    """
    targets = _usd_job_targets()
    assert _selected("tests/usd/test_x.py", targets), "test-usd no longer runs tests/usd/"
    missing = [rel for rel, literals in _test_modules().items()
               if not rel.startswith("tests/usd/") and rel not in USD_NOT_NEEDED
               and _needs_usd(literals) and not _selected(rel, targets)]
    assert not missing, f"these need usd-core but test-usd does not run them: {missing}"
    if _examples_needing_usd():
        smoke = "tests/test_examples_smoke.py::test_example_imports_resolve_against_the_library"
        assert _selected("tests/test_examples_smoke.py", targets) and any(
            t in ("tests/test_examples_smoke.py", smoke) for t in targets), (
            f"examples {_examples_needing_usd()} import maddening.usd; test-usd must run "
            f"{smoke}")
    for rel in USD_NOT_NEEDED:
        assert (REPO_ROOT / rel).is_file(), f"USD_NOT_NEEDED names a file that is gone: {rel}"



def _is_slow_mark(node: ast.AST) -> bool:
    """``pytest.mark.slow`` / ``mark.slow``, however it is spelled around."""
    return any(isinstance(n, ast.Attribute) and n.attr == "slow"
               and isinstance(n.value, ast.Attribute) and n.value.attr == "mark"
               for n in ast.walk(node))


def _bound_names(stmt: ast.stmt) -> set[str]:
    """The module-level names a top-level statement binds."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {stmt.name}
    names = set()
    for n in ast.walk(stmt):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
        elif isinstance(n, ast.alias):
            names.add((n.asname or n.name).split(".")[0])
    return names


def _names_used(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _usd_names(body: list[ast.stmt]) -> set[str]:
    """Module-level names whose definition needs usd-core, directly or through another.

    Covers helpers (``def _reload_usd``), skip markers built from a probe
    (``usd_only = pytest.mark.skipif(not _usd_installed(), ...)``) and
    names bound by a guarded import (``try: from pxr import Usd``).
    """
    defs: dict[str, list[ast.stmt]] = {}
    for stmt in body:
        for name in _bound_names(stmt):
            defs.setdefault(name, []).append(stmt)
    needs = {n for n, stmts in defs.items() if any(_needs_usd(_literals(d)) for d in stmts)}
    while True:
        more = {n for n, stmts in defs.items() if n not in needs
                and any(_names_used(d) & needs for d in stmts)}
        if not more:
            return needs
        needs |= more


def _slow_tests_needing_usd(tree: ast.Module) -> list[str]:
    usd = _usd_names(tree.body)

    def needs(node):
        return _needs_usd(_literals(node)) or bool(_names_used(node) & usd)

    def is_pytestmark(stmt):
        return isinstance(stmt, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets)

    top_level = [s for s in tree.body
                 if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    # The whole module needs usd-core when it imports it unguarded, skips on
    # it at import (importorskip), or marks every test with a usd condition.
    # A usd name inside module-level *data* (a table of extras) does not.
    module_needs = any(
        (isinstance(s, (ast.Import, ast.ImportFrom)) and _needs_usd(_literals(s)))
        or (isinstance(s, (ast.Expr, ast.Assign)) and _needs_usd(_literals(s)) and any(
            isinstance(c, ast.Call) and isinstance(c.func, (ast.Attribute, ast.Name))
            and (getattr(c.func, "attr", None) or getattr(c.func, "id", None)) == "importorskip"
            for c in ast.walk(s)))
        or (is_pytestmark(s) and needs(s))
        for s in top_level)
    module_slow = any(is_pytestmark(s) and _is_slow_mark(s) for s in top_level)

    found = []
    for stmt in tree.body:
        if isinstance(stmt, ast.ClassDef):
            class_marks = [s for s in stmt.body if is_pytestmark(s)] + stmt.decorator_list
            class_slow = any(_is_slow_mark(m) for m in class_marks)
            class_needs = any(needs(m) for m in class_marks)
            funcs = [(f, stmt.name, class_slow, class_needs) for f in stmt.body]
        else:
            funcs = [(stmt, None, False, False)]
        for func, cls, cls_slow, cls_needs in funcs:
            if not (isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and func.name.startswith("test")):
                continue
            slow = module_slow or cls_slow or any(_is_slow_mark(d) for d in func.decorator_list)
            if slow and (module_needs or cls_needs or needs(func)):
                found.append(f"{cls}::{func.name}" if cls else func.name)
    return found


def test_no_slow_marked_test_needs_usd_core():
    """A slow-marked test that needs ``usd-core`` would run in no job.

    ``test-usd`` is the only job that installs ``usd-core`` and it runs
    without ``-m`` (``test_every_test_that_needs_usd_core_runs_in_the_job_
    that_installs_it`` forbids one), so addopts' ``-m 'not slow'`` drops
    slow tests there; the slow lane runs them but installs ``.[ci]``, where
    they skip.  A test needs ``usd-core`` when it, or a module-level helper
    it names (followed through other helpers), imports or names ``pxr`` or
    ``maddening.usd``, or when its module does so at the top level.
    """
    offenders = []
    for rel, literals in _test_modules().items():
        if rel in USD_NOT_NEEDED or not _needs_usd(literals):
            continue
        tree = ast.parse((REPO_ROOT / rel).read_text(encoding="utf-8"), filename=rel)
        offenders += [f"{rel}::{name}" for name in _slow_tests_needing_usd(tree)]
    assert not offenders, (
        "slow-marked, but they need usd-core, which only test-usd installs, and it does not "
        f"run slow tests: {offenders}")


def test_the_usd_slow_scan_finds_what_it_is_for():
    source = """
import importlib.util
import pytest

def _usd_installed():
    return importlib.util.find_spec("pxr") is not None

usd_only = pytest.mark.skipif(not _usd_installed(), reason="no usd")

def _reload(gm):
    from maddening.usd.serialization import save_graph_to_usd
    return save_graph_to_usd

@pytest.mark.slow
@usd_only
def test_slow_by_skip_marker():
    pass

@pytest.mark.slow
def test_slow_through_a_helper():
    _reload(None)

@pytest.mark.slow
def test_slow_importing_itself():
    from pxr import Usd

@pytest.mark.parametrize("n", [pytest.param(1, marks=pytest.mark.slow)])
def test_slow_parameter(n):
    pytest.importorskip("pxr")

class TestSlowClass:
    pytestmark = pytest.mark.slow
    def test_in_a_slow_class(self):
        _reload(None)

@pytest.mark.slow
def test_slow_without_usd():
    pass

@usd_only
def test_usd_not_slow():
    pass

class TestPlain:
    def test_usd_in_a_plain_class(self):
        _reload(None)
"""
    assert sorted(_slow_tests_needing_usd(ast.parse(source))) == sorted([
        "test_slow_by_skip_marker", "test_slow_through_a_helper", "test_slow_importing_itself",
        "test_slow_parameter", "TestSlowClass::test_in_a_slow_class"])
    whole_module = "import pytest\nfrom pxr import Usd\n\n@pytest.mark.slow\ndef test_x():\n    pass\n"
    assert _slow_tests_needing_usd(ast.parse(whole_module)) == ["test_x"]
    module_slow = ("import pytest\npytestmark = pytest.mark.slow\n\n"
                   "def test_y():\n    pytest.importorskip('pxr')\n")
    assert _slow_tests_needing_usd(ast.parse(module_slow)) == ["test_y"]
    guarded = ("import pytest\ntry:\n    from pxr import Usd\nexcept ImportError:\n    Usd = None\n\n"
               "@pytest.mark.slow\ndef test_z():\n    Usd.Stage\n\n"
               "@pytest.mark.slow\ndef test_no_usd():\n    pass\n")
    assert _slow_tests_needing_usd(ast.parse(guarded)) == ["test_z"]
    # A usd module named in a table of optional extras is data, not a need.
    table = ("import pytest\nEXTRAS = {'maddening.usd': 'usd'}\n\n"
             "@pytest.mark.slow\ndef test_runs_an_example():\n    pass\n")
    assert _slow_tests_needing_usd(ast.parse(table)) == []
