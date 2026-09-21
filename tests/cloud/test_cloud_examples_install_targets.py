"""The cloud examples' remote install commands are installable where they run.

An example that installs the ``cuda12`` pin under an interpreter below the
floor -- ``python3.10`` is the system ``python3`` of ``runpod/base`` -- fails
at the install step, on the remote machine, minutes in.  The floor is the
stricter of two, and both are read from ``pyproject.toml`` rather than
written down here:

* what the lowest ``jax`` the pin admits needs (``jax>=0.7``: 3.11;
  ``jax>=0.11``: 3.12), and
* what MADDENING's own ``requires-python`` needs.

Deriving it is the point.  While the bound was the jax one alone, an example
naming ``python3.11`` passed this test and still could not install MADDENING,
and raising ``requires-python`` moved nothing here.

The pin itself must be the ``pyproject`` range everywhere in the tree (no
stale ``>=0.4,<0.6`` / ``cuda11`` remnants).
"""

from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

import maddening

_ROOT = Path(maddening.__file__).resolve().parents[2]
_EXAMPLES = Path(maddening.__file__).resolve().parent / "examples" / "cloud"
_PYPROJECT = _ROOT / "pyproject.toml"


def _cuda12_pin() -> str:
    with open(_PYPROJECT, "rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]
    (pin,) = extras["cuda12"]
    return pin                                   # e.g. jax[cuda12]>=0.10,<0.13


def _jax_requires_python_minor(pin: str) -> int:
    """Minimum Python minor for the *lowest* jax the pin admits."""
    low = re.search(r">=(\d+)\.(\d+)", pin)
    assert low, pin
    major, minor = int(low.group(1)), int(low.group(2))
    # jax 0.11 and later require Python >= 3.12; 0.7 - 0.10, >= 3.11.
    if (major, minor) >= (0, 11):
        return 12
    return 11 if (major, minor) >= (0, 7) else 10


def _maddening_requires_python_minor() -> int:
    """MADDENING's own floor, read from ``requires-python``."""
    with open(_PYPROJECT, "rb") as f:
        spec = tomllib.load(f)["project"]["requires-python"]
    m = re.search(r">=\s*3\.(\d+)", spec)
    assert m, spec
    return int(m.group(1))


def _floor_minor(pin: str) -> int:
    """The interpreter an example must name: the stricter of the two."""
    return max(_jax_requires_python_minor(pin), _maddening_requires_python_minor())


def _examples_installing_jax() -> list[Path]:
    return sorted(p for p in _EXAMPLES.rglob("*.py") if "jax[cuda12]" in p.read_text(encoding="utf-8"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"cloud_example_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)              # the examples only define things at import
    return module


def _install_commands(module) -> list[str]:
    return [v for k, v in vars(module).items()
            if isinstance(v, str) and "INSTALL" in k and "jax[cuda12]" in v]


@pytest.mark.parametrize("path", _examples_installing_jax(), ids=lambda p: p.name)
def test_example_install_commands_target_python_that_jax_supports(path):
    pin = _cuda12_pin()
    min_minor = _floor_minor(pin)
    module = _load(path)
    commands = _install_commands(module)
    assert commands, f"{path.name} mentions jax[cuda12] outside an *INSTALL* command"
    interpreter = getattr(module, "PYTHON", None)
    for cmd in commands:
        assert f'"{pin}"' in cmd, f"{path.name}: pin differs from pyproject ({pin!r})"
        # Any named interpreter below the floor, not just ``python3.10``:
        # the literal spelling stopped covering the floor the moment the
        # floor moved past it.
        named = [int(v) for v in re.findall(r"python3\.(\d+)", cmd)]
        assert all(v >= min_minor for v in named), (
            f"{path.name}: installs under python3.{min(named)}, below the "
            f"3.{min_minor} floor")
        # bare ``python3``/``pip3`` is the 3.10 interpreter on runpod/base
        assert not re.search(r"(?<![\w.])python3 -m pip", cmd), f"{path.name}: bare python3 -m pip"
        assert not re.search(r"(?<![\w.])pip3 install", cmd), f"{path.name}: bare pip3"
        if interpreter is not None:
            assert cmd.lstrip().startswith(interpreter), (path.name, cmd[:60])
    if interpreter is not None:
        m = re.fullmatch(r"python3\.(\d+)", interpreter)
        assert m and int(m.group(1)) >= min_minor, (path.name, interpreter, f"needs >= 3.{min_minor}")
        # the interpreter that installs is the one that runs the remote scripts
        src = path.read_text(encoding="utf-8")
        assert not re.search(r"""ssh_run(?:_background)?\(\s*["']python3 """, src), (
            f"{path.name}: a remote command still runs under bare python3")


_STALE = re.compile(r">=0\.4,<0\.6|jax==0\.4|cuda11|JAX >=0\.4|jax(?:lib)?>=0\.4\b")
_SCAN = ("src", "docker", "docs", "benchmarks/multigpu", "pyproject.toml", "README.md")


#: Historical records quote old pins on purpose — a release note explaining
#: that a stale pin was corrected has to name the pin it corrected.  Scanning
#: them turns "we fixed this" into a failure, so they are excluded by path
#: rather than by trying to tell prose from a dependency declaration.
_HISTORY = ("docs/release_notes/", "CHANGELOG.md")


def test_no_stale_jax_pins_in_tree():
    offenders = []
    for top in _SCAN:
        root = _ROOT / top
        files = [root] if root.is_file() else [p for p in root.rglob("*")
                                               if p.is_file() and p.suffix in
                                               (".py", ".md", ".toml", ".txt", ".yml", ".yaml", ".cfg", "")
                                               and "__pycache__" not in p.parts]
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = path.relative_to(_ROOT).as_posix()
            if any(rel.startswith(h) for h in _HISTORY):
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if _STALE.search(line):
                    offenders.append(f"{path.relative_to(_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "\n".join(offenders)
