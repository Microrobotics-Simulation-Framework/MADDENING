"""The printed output shown in ``docs/user_guide/inspection.md`` is true.

The docs-snippet gate (``tests/compliance/test_docs_snippets.py``) runs
every Python block of the page, but does not read what they print.  Here
the page's blocks run in order, in one namespace, and the stdout of each
block that is followed by an ``<!-- output -->`` marker and a ``text``
fence must be exactly that fence's content.  A block whose snippet marker
``requires:`` a module that is not installed is skipped with its output,
as the snippet gate skips it; CI installs every one of them (the
terminal diagram's ``termaid`` comes with the ``ci`` extra).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import re
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

PAGE = Path(__file__).resolve().parents[2] / "docs" / "user_guide" / "inspection.md"
_FENCE = re.compile(r"^```(\w*)\n(.*?)^```\n", re.S | re.M)


_REQUIRES = re.compile(r"^<!--\s*snippet:.*\brequires:\s*([\w.]+(?:\s+[\w.]+)*)")
_SKIPPED = object()


def _blocks(text: str) -> list[tuple[str, str, bool, tuple[str, ...]]]:
    """``[(lang, body, marked, requires)]``: ``marked`` when the fence
    follows an ``<!-- output -->`` line, ``requires`` the modules a
    preceding snippet marker names."""
    out = []
    for m in _FENCE.finditer(text):
        before = text[:m.start()].rstrip("\n").rsplit("\n", 1)[-1].strip()
        req = _REQUIRES.match(before)
        out.append((m.group(1), m.group(2), before == "<!-- output -->",
                    tuple(req.group(1).split()) if req else ()))
    return out


def test_every_marked_output_block_is_what_the_code_before_it_prints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for var in ("FORCE_COLOR", "TTY_COMPATIBLE"):    # rich would colour the redirected stdout
        monkeypatch.delenv(var, raising=False)
    blocks = _blocks(PAGE.read_text(encoding="utf-8"))
    namespace: dict = {"__name__": "__inspection_doc__"}
    last_stdout: object = None
    compared = skipped = 0
    for lang, body, marked, requires in blocks:
        if lang == "python":
            if any(importlib.util.find_spec(mod) is None for mod in requires):
                last_stdout = _SKIPPED
                continue
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(body, str(PAGE), "exec"), namespace)  # noqa: S102
            last_stdout = buf.getvalue()
        elif marked:
            assert lang == "text", "an <!-- output --> marker must precede a text fence"
            assert last_stdout is not None, "an output block before any code"
            if last_stdout is _SKIPPED:
                skipped += 1
            else:
                assert isinstance(last_stdout, str)
                assert last_stdout.rstrip("\n") == body.rstrip("\n"), (
                    f"docs output differs from what the code prints:\n{last_stdout}")
                compared += 1
            last_stdout = None
    # print_graph, Mermaid, state, parameters, memory, and the terminal
    # diagram (requires termaid)
    assert compared + skipped == 6
    assert skipped == (importlib.util.find_spec("termaid") is None)
