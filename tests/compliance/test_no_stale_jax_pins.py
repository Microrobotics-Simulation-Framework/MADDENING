"""The ``jax`` pin is the ``pyproject`` range everywhere in the tree.

No stale ``>=0.4,<0.6`` / ``jax==0.4`` / ``cuda11`` remnants in the
package, the Docker files, the docs, the multi-GPU benchmarks or the
README.  ``tests/cloud/test_cloud_examples_install_targets.py`` checks
that the cloud examples' install commands carry the ``pyproject`` pin and
an interpreter that can install it; this is the rest of the tree.

It lives under ``tests/compliance`` because it reads ``docs/`` and
``README.md``: a pull request that touches only documentation runs the
compliance job alone (``.github/workflows/ci.yml``, the ``changes`` job),
so a stale pin added to a page must meet this test there.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

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
