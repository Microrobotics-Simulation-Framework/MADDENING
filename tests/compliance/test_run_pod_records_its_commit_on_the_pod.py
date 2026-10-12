"""On the pod, the runner can record the commit it ran with.

Every file ``benchmarks/multigpu/run_pod.py`` writes records the commit
``git rev-parse HEAD`` names, and the summary closes a checklist item only on
files that record one.  The session's image is built from
``docker/Dockerfile.cloud``; an image without ``git`` (the one built from
``46de0bab`` on 2026-10-12, measured in the container) records no commit in
any file, and the runbook's dirty-tree check, ``git status --porcelain``,
prints nothing there and passes.  So the Dockerfile installs ``git``, and the
runbook's section 2a stops a pod where ``git rev-parse`` fails before the
first goal.  Files are read as text; nothing is built or run.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNBOOK = REPO / "benchmarks" / "multigpu" / "README.md"
DOCKERFILE = REPO / "docker" / "Dockerfile.cloud"


def _apt_packages(dockerfile: str) -> set:
    """The packages of the Dockerfile's ``apt-get install`` commands (comment
    lines inside a continued ``RUN`` are dropped, as Docker drops them)."""
    lines = [line for line in dockerfile.splitlines() if not line.lstrip().startswith("#")]
    text = "\n".join(lines).replace("\\\n", " ")
    found = set()
    for command in re.findall(r"apt-get install([^&\n]*)", text):
        found |= {t for t in command.split() if not t.startswith("-")}
    return found


def problems(runbook: str, dockerfile: str) -> list:
    out = []
    if "git" not in _apt_packages(dockerfile):
        out.append("docker/Dockerfile.cloud does not install git: no record on the pod names a commit")
    block = runbook.split("### 2a.", 1)[-1].split("```sh", 1)[-1].split("```", 1)[0]
    first_goal = block.find("run_pod.py")
    stop = re.search(r"git rev-parse --verify HEAD \|\| echo \"[^\"]*stop", block)
    if stop is None or (first_goal != -1 and stop.start() > first_goal):
        out.append("section 2a does not stop a pod where `git rev-parse --verify HEAD` fails, "
                   "before the first goal")
    return out


def _texts():
    return RUNBOOK.read_text(encoding="utf-8"), DOCKERFILE.read_text(encoding="utf-8")


def test_the_image_has_git_and_the_runbook_stops_without_it():
    assert problems(*_texts()) == []


def test_the_check_fails_on_each_way_it_can_part():
    runbook, dockerfile = _texts()
    no_git = re.sub(r"^\s*git \\\n", "", dockerfile, flags=re.M)
    assert no_git != dockerfile
    assert any("does not install git" in p for p in problems(runbook, no_git))
    commented = dockerfile.replace("    git \\\n", "    # git \\\n")
    assert any("does not install git" in p for p in problems(runbook, commented))
    no_stop = runbook.replace('git rev-parse --verify HEAD || echo "no git, or no git checkout: '
                              'stop (section 3)"\n', "")
    assert no_stop != runbook
    assert any("does not stop a pod" in p for p in problems(no_stop, dockerfile))
