"""The captured step-program digests are a capture of their recorded commit, never edited.

``tests/core/data/step_program_digests.json`` is what
``tests/core/test_step_program_digests.py`` compares the compiled programs
of graphs without a geometry-dependent mapping with.  It is worth
something only as a record of the tree *before* the change it guards, so
a change must not be able to move it:

* the file names the commit it was captured on, and that commit is an
  ancestor of the tree under test which does not itself contain the file
  (a capture is taken *on* a commit, and committed after it);
* since that commit the file has only grown: a later version of it may
  add the entry of another jax version, and changes neither the recorded
  commit nor any digest already there -- the uncommitted copy in the
  working tree included;
* (slow) recomputed on an archive of the recorded commit, with that
  commit's own capture script, the programs are the ones the file holds.

Until the file exists there is nothing to hold still, and
``tests/core/test_step_program_digests.py`` fails for the missing capture.

History is needed: on a shallow clone that does not reach the recorded
commit the checks are skipped with that reason (the compliance job fetches
the whole history and runs them).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RELATIVE = "tests/core/data/step_program_digests.json"
DIGESTS = REPO_ROOT / RELATIVE


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                          text=True, check=check)


def grew_only(older: dict, newer: dict) -> list:
    """Why *newer* is not *older* with entries added; empty when it is."""
    problems = []
    if newer.get("commit") != older.get("commit"):
        problems.append(f"the recorded commit changed from {older.get('commit')} to "
                        f"{newer.get('commit')}")
    for version, graphs in older.get("captures", {}).items():
        if version not in newer.get("captures", {}):
            problems.append(f"the capture for jax {version} was removed")
            continue
        if newer["captures"][version] != graphs:
            changed = sorted(g for g in set(graphs) | set(newer["captures"][version])
                             if graphs.get(g) != newer["captures"][version].get(g))
            problems.append(f"the capture for jax {version} was edited: {changed}")
    return problems


def _captured() -> dict:
    if not DIGESTS.exists():
        pytest.skip(f"{RELATIVE} has not been captured yet; "
                    f"tests/core/test_step_program_digests.py fails for it")
    data = json.loads(DIGESTS.read_text())
    commit = str(data.get("commit"))
    if _git("cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode != 0:
        shallow = _git("rev-parse", "--is-shallow-repository").stdout.strip() == "true"
        assert shallow, f"{RELATIVE} names commit {commit}, which this repository does not have"
        pytest.skip(f"shallow clone: the recorded commit {commit[:12]} is not in the history "
                    f"fetched here (the compliance job fetches all of it)")
    return data


def test_grew_only_tells_an_added_entry_from_an_edit():
    """The comparison itself: adding a jax version is growth; a changed
    digest, a removed version and another recorded commit are not."""
    base = {"commit": "a" * 40, "captures": {"1.0": {"g": {"step": "0" * 64}}}}
    grown = {"commit": "a" * 40, "captures": {"1.0": {"g": {"step": "0" * 64}},
                                               "2.0": {"g": {"step": "1" * 64}}}}
    assert grew_only(base, base) == [] and grew_only(base, grown) == []
    assert grew_only(grown, base) == ["the capture for jax 2.0 was removed"]
    edited = {"commit": "a" * 40, "captures": {"1.0": {"g": {"step": "2" * 64}}}}
    assert grew_only(base, edited) == ["the capture for jax 1.0 was edited: ['g']"]
    extra = {"commit": "a" * 40, "captures": {"1.0": {"g": {"step": "0" * 64}, "h": {}}}}
    assert grew_only(base, extra) == ["the capture for jax 1.0 was edited: ['h']"]
    moved = dict(base, commit="b" * 40)
    assert grew_only(base, moved)[0].startswith("the recorded commit changed")


def test_the_digests_were_captured_on_an_ancestor_that_does_not_hold_them():
    data = _captured()
    commit = data["commit"]
    assert _git("merge-base", "--is-ancestor", commit, "HEAD", check=False).returncode == 0, (
        f"{RELATIVE} was captured on {commit}, which is not an ancestor of the tree under test")
    assert _git("cat-file", "-e", f"{commit}:{RELATIVE}", check=False).returncode != 0, (
        f"{commit} already contains {RELATIVE}: a capture is taken on a commit and committed "
        f"after it, so the file cannot be a capture of that commit")


def test_the_digests_have_only_grown_since_their_capture():
    data = _captured()
    commit = data["commit"]
    history = _git("log", "--reverse", "--format=%H", f"{commit}..HEAD", "--",
                   RELATIVE).stdout.split()
    versions = []
    for sha in history:
        shown = _git("show", f"{sha}:{RELATIVE}", check=False)
        assert shown.returncode == 0, f"{sha} removed {RELATIVE}"
        versions.append((sha[:12], json.loads(shown.stdout)))
    versions.append(("the working tree", data))
    assert versions[0][1].get("commit") == commit
    for (older_name, older), (newer_name, newer) in zip(versions, versions[1:]):
        problems = grew_only(older, newer)
        assert not problems, (
            f"{RELATIVE} was edited between {older_name} and {newer_name}: {problems}.  The "
            f"capture is a record of {commit[:12]}; it is added to, never changed.")


# Slow: every gate graph lowered again, in a fresh process, from an archive.
# Per push: tests/compliance/test_step_program_digests_are_frozen.py::test_the_digests_have_only_grown_since_their_capture
@pytest.mark.slow
def test_the_digests_are_the_programs_of_the_commit_they_name(tmp_path):
    """The recorded commit's own tree and capture script, recomputed under
    the running jax, give the digests the file holds for it."""
    data = _captured()
    commit = data["commit"]
    archive = tmp_path / "base.tar"
    # What the capture script reads: the library, and the topology harness
    # that builds the gate graphs.
    subprocess.run(["git", "-C", str(REPO_ROOT), "archive", "-o", str(archive), commit,
                    "src", "tests/__init__.py", "tests/property",
                    "scripts/capture_step_programs.py"], check=True)
    tree = tmp_path / "base"
    with tarfile.open(archive) as tar:
        tar.extractall(tree, filter="data")
    env = dict(os.environ, PYTHONPATH=str(tree / "src"), JAX_PLATFORMS="cpu",
               HOME=str(tmp_path), PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, str(tree / "scripts" / "capture_step_programs.py"),
                          "--check", "--file", str(DIGESTS)], env=env, capture_output=True,
                         text=True, timeout=1800, cwd=str(tree))
    assert out.returncode == 0, (
        f"{RELATIVE} does not hold the programs of {commit}:\n{out.stdout[-3000:]}"
        f"\n{out.stderr[-1500:]}")
