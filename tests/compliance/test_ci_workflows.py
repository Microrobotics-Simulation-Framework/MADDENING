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
* the compilation cache is saved only after a passing test step, and only
  once its unreadable entries are gone;
* the slow lane files an abort as an abort, not as out-of-memory;
* every test that needs ``usd-core`` runs in the one job that installs it.

Workflow steps are read with ``yaml.safe_load`` and, where it is their
behaviour that matters, run: the shell scripts under ``bash`` and the
issue classifier under ``node``, with the ``${{ }}`` expressions they use
filled in (an expression the tests do not know fails them, rather than
running with a guess).
"""

from __future__ import annotations

import ast
import fnmatch
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


def _names_a_docs_path(literal: str, patterns: list[str], docs_only_names: set[str]) -> bool:
    if not literal or len(literal) > 300 or any(c.isspace() for c in literal):
        return False                                # prose, not a path
    parts = [p for p in literal.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    for i in range(len(parts)):
        tail = "/".join(parts[i:])
        # POSIX `case` semantics, which fnmatch shares: `*` matches `/` too.
        # `tail + "/"` catches a directory component: `ROOT / "docs" / ...`.
        if any(fnmatch.fnmatchcase(tail, p) or fnmatch.fnmatchcase(tail + "/", p)
               for p in patterns):
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
    for path in sorted((REPO_ROOT / "tests").rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.startswith("tests/compliance/"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and _names_a_docs_path(node.value, patterns, docs_only_names)
                    and f"{rel}: {node.value}" not in NOT_A_DOCS_READ):
                offenders.append(f"{rel}:{node.lineno}: {node.value!r}")
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


def _needs_usd(tree: ast.AST) -> bool:
    """Imports pxr / maddening.usd, or skips on them, anywhere in the module."""
    def is_usd(name: str) -> bool:
        return name in ("pxr", "maddening.usd") or name.startswith(("pxr.", "maddening.usd."))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(is_usd(a.name) for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module and (
                is_usd(node.module)
                or (node.module == "maddening" and any(a.name == "usd" for a in node.names))):
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and is_usd(node.value):
            return True                             # importorskip("pxr"), find_spec("pxr"), ...
    return False


def _examples_needing_usd() -> list[str]:
    root = REPO_ROOT / "src" / "maddening" / "examples"
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*.py")
                  if _needs_usd(ast.parse(p.read_text(encoding="utf-8"))))


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
    missing = []
    for path in sorted((REPO_ROOT / "tests").rglob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel.startswith(("tests/usd/", "tests/compliance/")) or rel in USD_NOT_NEEDED:
            continue
        if _needs_usd(ast.parse(path.read_text(encoding="utf-8"))) and not _selected(rel, targets):
            missing.append(rel)
    assert not missing, f"these need usd-core but test-usd does not run them: {missing}"
    if _examples_needing_usd():
        smoke = "tests/test_examples_smoke.py::test_example_imports_resolve_against_the_library"
        assert _selected("tests/test_examples_smoke.py", targets) and any(
            t in ("tests/test_examples_smoke.py", smoke) for t in targets), (
            f"examples {_examples_needing_usd()} import maddening.usd; test-usd must run "
            f"{smoke}")
    for rel in USD_NOT_NEEDED:
        assert (REPO_ROOT / rel).is_file(), f"USD_NOT_NEEDED names a file that is gone: {rel}"
