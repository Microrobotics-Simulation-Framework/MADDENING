"""The printed output shown in ``docs/user_guide/inspection.md`` is true.

The docs-snippet gate (``tests/compliance/test_docs_snippets.py``) runs
every Python block of the page, but does not read what they print.  Here
the page's blocks run in order, in one namespace, and the stdout of each
block that is followed by an ``<!-- output -->`` marker and a ``text``
fence must be exactly that fence's content.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

PAGE = Path(__file__).resolve().parents[2] / "docs" / "user_guide" / "inspection.md"
_FENCE = re.compile(r"^```(\w*)\n(.*?)^```\n", re.S | re.M)


def _blocks(text: str) -> list[tuple[str, str, bool]]:
    """``[(lang, body, marked)]``: ``marked`` when the fence follows an
    ``<!-- output -->`` line."""
    out = []
    for m in _FENCE.finditer(text):
        before = text[:m.start()].rstrip("\n").rsplit("\n", 1)[-1].strip()
        out.append((m.group(1), m.group(2), before == "<!-- output -->"))
    return out


def test_every_marked_output_block_is_what_the_code_before_it_prints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    blocks = _blocks(PAGE.read_text(encoding="utf-8"))
    namespace: dict = {"__name__": "__inspection_doc__"}
    last_stdout = None
    compared = 0
    for lang, body, marked in blocks:
        if lang == "python":
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(body, str(PAGE), "exec"), namespace)  # noqa: S102
            last_stdout = buf.getvalue()
        elif marked:
            assert lang == "text", "an <!-- output --> marker must precede a text fence"
            assert last_stdout is not None, "an output block before any code"
            assert last_stdout.rstrip("\n") == body.rstrip("\n"), (
                f"docs output differs from what the code prints:\n{last_stdout}")
            compared += 1
            last_stdout = None
    # print_graph, Mermaid, state, parameters, memory
    assert compared == 5
