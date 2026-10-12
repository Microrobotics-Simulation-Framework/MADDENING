"""Every test file ``pyproject.toml`` names exists.

Its comments send a reader to the test that proves a setting (that the
wheel ships ``py.typed``, that the build backend is the declared one); a
test renamed or removed leaves the pointer naming nothing.  Until 0.4.0's
release review the ``py.typed`` comment named ``tests/test_py_typed_marker.py``,
which never existed: the test is ``tests/test_packaging_py_typed.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: A test file's path as a comment or a setting writes it (a node id's
#: ``::test`` part, a trailing full stop and backticks are not part of it).
_TEST_FILE = re.compile(r"tests/[A-Za-z0-9_./-]+\.py")


def _named(text: str) -> list[str]:
    return sorted(set(_TEST_FILE.findall(text)))


def test_every_test_file_pyproject_names_exists():
    named = _named((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert named, "pyproject.toml names no test file: the pattern no longer reads its comments"
    missing = [path for path in named if not (REPO_ROOT / path).is_file()]
    assert not missing, f"pyproject.toml names test files that do not exist: {missing}"


def test_the_pattern_reads_a_path_in_backticks_a_node_id_and_a_sentence():
    text = ("# `tests/test_a.py` builds the wheel; see tests/sub/test_b.py.\n"
            "# tests/test_c.py::test_d asserts it")
    assert _named(text) == ["tests/sub/test_b.py", "tests/test_a.py", "tests/test_c.py"]
