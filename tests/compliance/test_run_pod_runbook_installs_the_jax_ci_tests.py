"""The multi-GPU runbook installs the JAX CI tests, on an image that can take it.

``benchmarks/multigpu/README.md`` section 2a installs JAX on the pod and
section 1 says which image to launch: one built from
``docker/Dockerfile.cloud`` at the session's commit, with the build argument
``JAX_SPEC``.  The launcher's default image (``:latest``) was built from an
older Dockerfile on Python 3.10 and ``jax<0.6``, where this tree does not
install; and pip leaves an installed JAX that satisfies a range alone, so a
range in section 2a would keep whatever the image carries.  So the runbook
pins one version, and this test holds every place that names it to the same
one: section 2a's install line, section 1's build argument, a version CI's
test matrix runs, a version inside the Dockerfile's default range, which is
the range of pyproject's ``cuda12`` extra (RPD-069 in
``docs/validation/rest_runpod_claims.yaml``).  Files are read as text;
nothing is built or run.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNBOOK = REPO / "benchmarks" / "multigpu" / "README.md"
DOCKERFILE = REPO / "docker" / "Dockerfile.cloud"
PYPROJECT = REPO / "pyproject.toml"
CI = REPO / ".github" / "workflows" / "ci.yml"
IMAGE = "ghcr.io/microrobotics-simulation-framework/maddening-cloud"


def _version(text: str) -> tuple:
    return tuple(int(x) for x in text.split("."))


def problems(runbook: str, dockerfile: str, pyproject: str, ci: str) -> list:
    """Everything that does not hold the four files to one JAX."""
    out = []
    installs = re.findall(r'\$PIP "jax\[cuda12\]==([0-9.]+)"', runbook)
    if len(installs) != 1:
        return [f"section 2a installs jax[cuda12]==<version> once, found {installs}"]
    pinned = installs[0]
    if 'PIP="python3 -m pip install -q --break-system-packages"' not in runbook:
        out.append("section 2a's PIP is not the system python3's pip with --break-system-packages")
    first = runbook.split("### 2a.")[1].split("```sh")[1].split("```")[0].strip().splitlines()
    if not (len(first) > 1 and first[1].startswith('python3 -c "import sys; assert '
                                                    'sys.version_info >= (3, 12)')):
        out.append("section 2a does not check the interpreter first")
    if "python3 --version" not in runbook.split("### 2a.")[1].split("session.txt")[0]:
        out.append("session.txt does not record python3 --version")
    built = re.findall(r"--build-arg JAX_SPEC='jax\[cuda12\]==([0-9.]+)'", runbook)
    if built != [pinned]:
        out.append(f"section 1 builds the image with JAX {built}, section 2a installs {pinned}")
    for route in (f'container_image="{IMAGE}:<commit>"', f"docker:{IMAGE}:<commit>"):
        if route not in " ".join(runbook.split()):
            out.append(f"section 1 does not launch on the session's own tag: {route}")
    if f"-t {IMAGE}:<commit>" not in runbook or f"docker push {IMAGE}:<commit>" not in runbook:
        out.append("section 1 does not build and push the session's tag")
    matrix = re.findall(r'jax-version: \[([^\]]*)\]', ci)
    lanes = {v.strip().strip('"') for row in matrix for v in row.split(",")}
    if pinned not in lanes:
        out.append(f"CI's test matrix runs jax {sorted(lanes)}, not the runbook's {pinned}")
    arg = re.findall(r'^ARG JAX_SPEC="jax\[cuda12\]>=([0-9.]+),<([0-9.]+)"$', dockerfile, re.M)
    if len(arg) != 1:
        return out + [f"the Dockerfile's JAX_SPEC default is not one range: {arg}"]
    low, high = arg[0]
    if not _version(low) <= _version(pinned) < _version(high):
        out.append(f"the runbook's {pinned} is outside the Dockerfile's default >={low},<{high}")
    if '"${JAX_SPEC}"' not in dockerfile or re.search(r'pip3 install[^\n]*\n\s*"jax\[', dockerfile):
        out.append("the Dockerfile installs a JAX of its own, not JAX_SPEC")
    extra = re.findall(r'^cuda12 = \["jax\[cuda12\]>=([0-9.]+),<([0-9.]+)"\]$', pyproject, re.M)
    if extra != [(low, high)]:
        out.append(f"the Dockerfile's default >={low},<{high} is not the cuda12 extra's {extra}")
    return out


def _texts() -> tuple:
    return tuple(p.read_text(encoding="utf-8") for p in (RUNBOOK, DOCKERFILE, PYPROJECT, CI))


def test_the_runbook_the_image_and_ci_name_one_jax():
    assert problems(*_texts()) == []


def test_the_check_fails_on_each_way_the_four_files_can_part():
    """The guard can fail: a range in section 2a, a build argument of
    another version, a version CI does not run, a Dockerfile that installs
    its own JAX or whose default leaves the extra's range, the launch on
    ``:latest``, no interpreter check."""
    runbook, dockerfile, pyproject, ci = _texts()
    pinned = re.findall(r'\$PIP "jax\[cuda12\]==([0-9.]+)"', runbook)[0]

    def said(**changed):
        texts = {"runbook": runbook, "dockerfile": dockerfile, "pyproject": pyproject, "ci": ci}
        texts.update(changed)
        found = problems(**texts)
        assert found, f"the edit was not noticed: {sorted(changed)}"
        return " / ".join(found)

    assert "installs jax[cuda12]==<version> once" in said(runbook=runbook.replace(
        f'$PIP "jax[cuda12]=={pinned}"', '$PIP "jax[cuda12]>=0.10,<0.13"'))
    assert "builds the image with JAX" in said(runbook=runbook.replace(
        f"JAX_SPEC='jax[cuda12]=={pinned}'", "JAX_SPEC='jax[cuda12]==0.11.0'"))
    assert "not the runbook's" in said(ci=ci.replace(f'"{pinned}"', '"0.11.1"'))
    assert "installs a JAX of its own" in said(dockerfile=dockerfile.replace(
        '"${JAX_SPEC}"', '"jax[cuda12]>=0.10,<0.13"'))
    assert "not the cuda12 extra's" in said(dockerfile=dockerfile.replace(
        'ARG JAX_SPEC="jax[cuda12]>=0.10,<0.13"', 'ARG JAX_SPEC="jax[cuda12]>=0.10,<0.14"'))
    assert "outside the Dockerfile's default" in said(dockerfile=dockerfile.replace(
        'ARG JAX_SPEC="jax[cuda12]>=0.10,<0.13"', 'ARG JAX_SPEC="jax[cuda12]>=0.4,<0.6"'))
    assert "does not launch on the session's own tag" in said(runbook=runbook.replace(
        f'container_image="{IMAGE}:<commit>"', f'container_image="{IMAGE}:latest"'))
    assert "does not check the interpreter first" in said(runbook=runbook.replace(
        'python3 -c "import sys; assert sys.version_info >= (3, 12), sys.version"', "true"))
    assert "python3 --version" in said(runbook=runbook.replace(
        "date -u; python3 --version;", "date -u;"))
