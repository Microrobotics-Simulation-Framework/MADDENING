"""Every Python code block in the documentation runs, or says why it can't.

A guide snippet is an API claim, like a ``>>>`` example
(``scripts/check_doctests.py``), but nothing executed them.  Three
release-record audits in a row each found seven to nine that did not run:
imports of modules that do not exist, keyword arguments that were renamed,
counts that went stale.  This module runs them.

What is collected
-----------------
Every fenced block whose info string is ``python`` / ``py`` / ``python3``,
or a MyST ``{code-block} python`` directive, in ``docs/**/*.md``,
``README.md``, ``CONTRIBUTING.md`` and ``SECURITY.md``.  Blocks nested in
other MyST directives (``{note}``, ``{warning}``) are found too.

Markers
-------
An HTML comment on the line immediately before the opening fence, at the
same indentation (GitHub and MyST both render it as nothing)::

    <!-- snippet: no-run, reason: fragment: a method body, not a module -->
    <!-- snippet: continues -->
    <!-- snippet: requires: pxr -->
    <!-- snippet: continues, no-run, reason: pseudo-code: ... -->

``no-run`` needs a ``reason:`` whose first word, up to a colon, is a
category from :data:`NO_RUN_CATEGORIES`, then an explanation.
``continues`` runs the block after the file's previous Python block, as
one program in one process (and after whatever that block continues);
a ``no-run`` block may carry it too, and is then skipped while the chain
runs on past it.  The first block of a chain must run.  ``requires:``
names importable modules; the snippet is skipped (with that reason)
where one is missing and runs in the job that installs it.  Anything
else that looks like a marker is an error: an orphaned marker, an
unknown item, a reason on a snippet that runs.  The full convention is
in ``docs/developer_guide/documentation_standards.md``.

The sandbox
-----------
Each runnable snippet (with whatever it continues) runs once, in a fresh
``python`` subprocess, with a timeout; cwd a fresh temporary directory;
``HOME`` an empty one; ``JAX_PLATFORMS=cpu``; ``MPLBACKEND=Agg``; and an
environment built from an allowlist, so no cloud credential or token in
the parent's environment reaches it.  An audit hook refuses any socket
connection or name lookup that is not loopback -- through Python's
``socket`` module, which is what HTTP clients use; a C extension's own
sockets (libzmq) are not seen, and there the static launch-path refusal,
the empty ``HOME`` and the missing credentials are what hold.  The code is compiled with
the Markdown file as its filename and its own line numbers, so a traceback
points at the line in the doc.  Snippets run in parallel, one process per
CPU of the test's affinity mask, each pinned to its own CPU.

Cloud launch paths are never run
--------------------------------
No snippet that names a launch path (:data:`LAUNCH_PATTERNS`) may run: it
must be marked ``no-run`` with the ``cloud`` category, and the runner
refuses it again, statically, before starting a process.  Its
``maddening`` imports are checked by reading the source, never by
importing anything.

``no-run`` is not "unchecked"
-----------------------------
The ``maddening`` imports of every ``no-run`` snippet are still resolved,
so an illustrative fragment cannot name a module or symbol that does not
exist; and every keyword argument it passes to a callable it imported
from ``maddening`` is checked against that callable's signature (unless
the callable takes ``**kwargs``).  Only the import statements run, never
the snippet.
"""

from __future__ import annotations

import ast
import concurrent.futures
import dataclasses
import importlib.util
import json
import os
import queue
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"

#: The top-level Markdown files collected beside ``docs/**/*.md``.
TOP_LEVEL_DOCS = ("README.md", "CONTRIBUTING.md", "SECURITY.md")

PYTHON_LANGS = frozenset({"python", "py", "python3"})
#: MyST directives whose body is code; a Python one is collected.
CODE_DIRECTIVES = frozenset({"code-block", "code", "sourcecode", "code-cell"})

#: The categories a ``no-run`` reason may start with, and what each means.
NO_RUN_CATEGORIES = {
    "fragment": "an intentionally partial excerpt: a method body, a dict entry, "
                "or calls on objects the surrounding prose builds",
    "pseudo-code": "placeholders (``...``, ``XXX``, ``bounds={...}``) stand for the "
                   "reader's own code",
    "legacy": "shows a removed or pre-migration API, for comparison",
    "cloud": "names a cloud launch path; never run (see LAUNCH_PATTERNS)",
    "network": "needs network access or a remote service",
    "gpu": "needs a GPU or several devices",
    "external": "needs a file, package, tool or configuration outside this repository",
}

#: Patterns that mark a snippet as able to reach a cloud launch path.  A
#: snippet matching any of them must be ``no-run`` with the ``cloud``
#: category, and is never executed or imported from.  Fail closed: a false
#: positive costs a marker, a false negative could provision a machine.
LAUNCH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("/cloud/launch", re.compile(r"/cloud/launch")),
    ("launch_vm", re.compile(r"\blaunch_vm\b")),
    ("CloudSession(", re.compile(r"\bCloudSession\s*\(")),
    ("CloudLauncher(", re.compile(r"\bCloudLauncher\s*\(")),
    (".launch(", re.compile(r"\.launch\s*\(")),
    ("sky.", re.compile(r"\bsky\.")),
    ("import sky", re.compile(r"^\s*(?:import|from)\s+sky\b", re.M)),
    ("the sky CLI", re.compile(r"\bsky\s+(?:launch|exec|start|jobs|serve|up)\b")),
    ("RUNPOD", re.compile(r"RUNPOD")),
    ("the runpod SDK", re.compile(r"\brunpod\s*\.|^\s*(?:import|from)\s+runpod\b", re.M)),
    ("a cloud provider SDK",
     re.compile(r"\b(?:boto3|botocore|googleapiclient|google\.cloud|lambda_cloud)\b")),
)

#: Environment variables a snippet's process may inherit.  Everything else
#: -- cloud credentials, API tokens, compilation-cache paths -- is dropped.
ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "SYSTEMROOT")

#: Seconds one snippet (with what it continues) may run.
SNIPPET_TIMEOUT_S = 120.0

#: How many snippets run.  Held *equal* to the count, not as a floor: a
#: scan that silently collects fewer blocks, or a snippet quietly demoted to
#: ``no-run``, must show up here.  Change it in the commit that adds or
#: demotes a snippet; the failure message prints the new count.
EXPECTED_RUNNABLE = 38

_FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_MARKER = re.compile(r"^(?P<indent>[ \t]*)<!--\s*snippet:(?P<body>.*?)-->[ \t]*$")
#: A line that *starts* with an HTML comment naming a snippet.  Anchored, so
#: prose that quotes a marker in backticks is not mistaken for one.
_MARKER_LIKE = re.compile(r"^[ \t]*<!--\s*snippets?\b", re.I)
_DIRECTIVE_OPTION = re.compile(r"^\s*:[\w-]+:")
_MODULE_NAME = re.compile(r"^[A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)*$")


# ── Collection ──────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Marker:
    no_run: str | None = None          # the reason, when no-run
    continues: bool = False
    requires: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Snippet:
    path: Path
    fence_line: int                    # 1-based line of the opening fence
    code: str                          # dedented; directive options blanked
    marker: Marker

    @property
    def where(self) -> str:
        try:
            rel = self.path.relative_to(REPO_ROOT)
        except ValueError:
            rel = self.path
        return f"{rel.as_posix()}:{self.fence_line}"

    @property
    def runs(self) -> bool:
        return self.marker.no_run is None

    @property
    def category(self) -> str | None:
        if self.marker.no_run is None:
            return None
        return self.marker.no_run.split(":", 1)[0].strip()


def parse_marker(body: str) -> tuple[Marker, list[str]]:
    """Parse the text between ``<!-- snippet:`` and ``-->``."""
    errors: list[str] = []
    if "--" in body:
        errors.append("'--' inside the marker: an HTML comment may not contain it")
    reason = None
    m = re.search(r"\breason:", body)
    if m:
        reason = body[m.end():].strip()
        body = body[:m.start()].rstrip().rstrip(",")
    no_run = continues = False
    requires: list[str] = []
    seen: set[str] = set()
    for item in (x.strip() for x in body.split(",")):
        if not item:
            continue
        key = item.split(":", 1)[0].strip()
        if key in seen:
            errors.append(f"{key!r} given twice")
        seen.add(key)
        if item == "no-run":
            no_run = True
        elif item == "continues":
            continues = True
        elif key == "requires" and ":" in item:
            mods = item.split(":", 1)[1].split()
            if not mods:
                errors.append("'requires:' names no module")
            for mod in mods:
                if not _MODULE_NAME.match(mod):
                    errors.append(f"'requires:' entry {mod!r} is not a module name")
            requires.extend(mods)
        else:
            errors.append(f"unknown marker item {item!r} (expected no-run, continues, "
                          "requires: <module>..., reason: <category>: <why>)")
    if reason is not None and not no_run:
        errors.append("a 'reason:' on a snippet that is not marked no-run")
    if no_run:
        if not reason:
            errors.append("'no-run' without a 'reason:'")
        else:
            category, sep, why = reason.partition(":")
            if not sep or category.strip() not in NO_RUN_CATEGORIES:
                errors.append(
                    f"the reason {reason!r} does not start with a category: write "
                    f"'reason: <category>: <why>', category one of {sorted(NO_RUN_CATEGORIES)}")
            elif not re.search(r"[A-Za-z]{3}", why):
                errors.append(f"the reason {reason!r} names a category but says nothing")
        if requires:
            errors.append("'requires:' on a no-run snippet means nothing")
    elif reason is None and not requires and not continues and not errors:
        errors.append("an empty snippet marker")
    return Marker(reason if no_run else None, continues, tuple(requires)), errors


def _closing_fence(lines: list[str], start: int, fence: str) -> int:
    """Index of the line closing a fence opened at ``start - 1`` (or len)."""
    close = re.compile(r"^[ \t]*" + re.escape(fence[0]) + "{" + str(len(fence)) + r",}[ \t]*$")
    for j in range(start, len(lines)):
        if close.match(lines[j]):
            return j
    return len(lines)


def _python_body(info: str) -> tuple[bool, bool, bool]:
    """``(is_python, is_code_directive, is_other_directive)`` for an info string."""
    info = info.strip()
    if info.startswith("{") and "}" in info:
        name = info[1:info.index("}")].strip()
        rest = info[info.index("}") + 1:].split()
        if name in CODE_DIRECTIVES:
            return bool(rest) and rest[0] in PYTHON_LANGS, True, False
        return False, False, True
    words = info.split()
    return bool(words) and words[0] in PYTHON_LANGS, False, False


def parse_markdown(text: str, path: Path, line_offset: int = 0
                   ) -> tuple[list[Snippet], list[str]]:
    """Every Python block in ``text`` and every marker error, in order."""
    lines = text.splitlines()
    snippets: list[Snippet] = []
    errors: list[str] = []
    pending: tuple[int, str, Marker] | None = None   # (line index, indent, marker)

    def where(i: int) -> str:
        try:
            rel = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel = str(path)
        return f"{rel}:{i + 1 + line_offset}"

    i = 0
    while i < len(lines):
        line = lines[i]
        fence = _FENCE.match(line)
        if fence and not (fence["fence"][0] == "`" and "`" in fence["info"]):
            end = _closing_fence(lines, i + 1, fence["fence"])
            is_py, code_directive, other_directive = _python_body(fence["info"])
            body = lines[i + 1:end]
            if pending is not None:
                p_line, p_indent, p_marker = pending
                if not is_py:
                    errors.append(f"{where(p_line)}: a snippet marker not followed by a "
                                  "Python block")
                elif p_line != i - 1:
                    errors.append(f"{where(p_line)}: a snippet marker must be on the line "
                                  "immediately before the fence")
                elif p_indent != fence["indent"]:
                    errors.append(f"{where(p_line)}: the marker's indentation differs from "
                                  "the fence's (it would move the block out of its list item)")
            if is_py:
                marker = pending[2] if pending is not None and pending[0] == i - 1 else Marker()
                indent = fence["indent"]
                code_lines = [ln[len(indent):] if ln.startswith(indent) else ln.lstrip()
                              for ln in body]
                if code_directive:
                    k = 0
                    if code_lines and code_lines[0].strip() == "---":
                        k = 1
                        while k < len(code_lines) and code_lines[k].strip() != "---":
                            k += 1
                        k = min(k + 1, len(code_lines))
                        code_lines[:k] = [""] * k
                    else:
                        while k < len(code_lines) and _DIRECTIVE_OPTION.match(code_lines[k]):
                            code_lines[k] = ""
                            k += 1
                code = textwrap.dedent("\n".join(code_lines))
                snippets.append(Snippet(path, i + 1 + line_offset, code, marker))
            elif other_directive:
                inner, inner_errors = parse_markdown(
                    "\n".join(body), path, line_offset + i + 1)
                snippets.extend(inner)
                errors.extend(inner_errors)
            pending = None
            i = end + 1
            continue
        if _MARKER_LIKE.search(line):
            if pending is not None:
                errors.append(f"{where(pending[0])}: a snippet marker not followed by a "
                              "Python block")
            m = _MARKER.match(line)
            if not m:
                errors.append(f"{where(i)}: malformed snippet marker {line.strip()!r} "
                              "(expected '<!-- snippet: ... -->' alone on its line)")
                pending = None
            else:
                marker, marker_errors = parse_marker(m["body"])
                errors.extend(f"{where(i)}: {e}" for e in marker_errors)
                pending = (i, m["indent"], marker)
            i += 1
            continue
        if pending is not None and line.strip():
            errors.append(f"{where(pending[0])}: a snippet marker not followed by a "
                          "Python block")
            pending = None
        i += 1
    if pending is not None:
        errors.append(f"{where(pending[0])}: a snippet marker not followed by a Python block")
    return snippets, errors


def doc_files(root: Path = REPO_ROOT) -> list[Path]:
    files = sorted((root / "docs").rglob("*.md"))
    return files + [root / name for name in TOP_LEVEL_DOCS if (root / name).exists()]


def collect(root: Path = REPO_ROOT) -> tuple[list[Snippet], list[str]]:
    snippets: list[Snippet] = []
    errors: list[str] = []
    for path in doc_files(root):
        found, errs = parse_markdown(path.read_text(encoding="utf-8"), path)
        snippets.extend(found)
        errors.extend(errs)
    return snippets, errors


def launch_hits(code: str) -> list[str]:
    """The launch-path patterns ``code`` matches (empty if none)."""
    hits = [name for name, pattern in LAUNCH_PATTERNS if pattern.search(code)]
    if re.search(r"\brun_pod\b", code) and "--dry-run" not in code:
        hits.append("run_pod without --dry-run")
    return hits


def chains(snippets: list[Snippet]) -> list[list[Snippet]]:
    """Every block, grouped into ``continues`` chains (no-run blocks included).

    A block marked ``continues`` joins the chain of the file's previous
    Python block; any other block starts a chain of its own.
    """
    out: list[list[Snippet]] = []
    chain_of_last: dict[Path, list[Snippet]] = {}
    for s in snippets:
        chain = chain_of_last.get(s.path)
        if s.marker.continues and chain is not None:
            chain.append(s)
        else:
            chain = [s]
            out.append(chain)
        chain_of_last[s.path] = chain
    return out


def chain_errors(snippets: list[Snippet]) -> list[str]:
    """Errors in ``continues`` chains and in launch-path classification."""
    errors: list[str] = []
    seen_files: set[Path] = set()
    for s in snippets:
        if s.marker.continues and s.path not in seen_files:
            errors.append(f"{s.where}: 'continues' on the first Python block in its file")
        seen_files.add(s.path)
        hits = launch_hits(s.code)
        if hits and s.category != "cloud":
            errors.append(f"{s.where}: names a cloud launch path ({', '.join(hits)}) and is "
                          "not marked '<!-- snippet: no-run, reason: cloud: ... -->'")
    for chain in chains(snippets):
        if not chain[0].runs:
            for s in chain[1:]:
                if s.runs:
                    errors.append(f"{s.where}: runs, but its chain starts at {chain[0].where}, "
                                  "which is no-run")
    return errors


@dataclasses.dataclass(frozen=True)
class Unit:
    """Snippets that run together: one block plus the blocks continuing it."""
    blocks: tuple[Snippet, ...]

    @property
    def where(self) -> str:
        head = self.blocks[0].where
        return head + "".join(f"+{b.fence_line}" for b in self.blocks[1:])

    @property
    def requires(self) -> tuple[str, ...]:
        return tuple(sorted({m for b in self.blocks for m in b.marker.requires}))


def units(snippets: list[Snippet]) -> list[Unit]:
    """The runnable blocks of every chain that starts with a runnable block."""
    return [Unit(tuple(b for b in chain if b.runs))
            for chain in chains(snippets) if chain[0].runs]


# ── The sandbox ─────────────────────────────────────────────────────────


#: Runs in the snippet's process before any snippet code.  Pins itself to
#: one CPU (before JAX sizes its thread pools from the affinity mask),
#: refuses non-loopback network access, then executes each block in one
#: shared namespace, compiled under the doc's own path and line numbers.
_DRIVER = r'''
import json, os, sys

_cpu = os.environ.pop("MADDENING_SNIPPET_CPU", "")
if _cpu and hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, {int(_cpu)})

def _is_local(host):
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if host is None:
        return True
    host = str(host)
    return host == "localhost" or host.startswith(("127.", "::1", "0.0.0.0")) or host == ""


def _refuse_remote(event, args):
    if event == "socket.connect":
        address = args[1]
        if isinstance(address, tuple) and address and not _is_local(address[0]):
            raise PermissionError(f"docs snippet sandbox: no network ({address[0]!r})")
    elif event == "socket.getaddrinfo":
        if not _is_local(args[0]):
            raise PermissionError(f"docs snippet sandbox: no name lookup ({args[0]!r})")


sys.addaudithook(_refuse_remote)

with open(sys.argv[1], encoding="utf-8") as fh:
    _payload = json.load(fh)
del fh
_ns = {"__name__": "__main__", "__builtins__": __builtins__}
for _block in _payload["blocks"]:
    _code = compile("\n" * (_block["first_line"] - 1) + _block["code"],
                    _block["path"], "exec")
    exec(_code, _ns)
'''

#: Resolves ``maddening`` import statements, and the keyword arguments a
#: snippet passes to what they import, without running any snippet: prints
#: one JSON object per failure.
_IMPORT_CHECKER = r'''
import inspect, json, sys
import maddening
print(json.dumps({"maddening": maddening.__file__}))
with open(sys.argv[1], encoding="utf-8") as fh:
    items = json.load(fh)
for item in items:
    ns = {"__name__": "__snippet_imports__"}
    try:
        exec(compile(item["statement"], item["where"], "exec"), ns)
    except BaseException as exc:
        print(json.dumps({"where": item["where"], "what": item["statement"],
                          "error": f"{type(exc).__name__}: {exc}"}))
        continue
    for call in item["calls"]:
        head, *attrs = call["name"].split(".")
        obj, missing = ns.get(head), None
        for attr in attrs:
            if not hasattr(obj, attr):
                missing = attr
                break
            obj = getattr(obj, attr)
        if missing is not None:
            print(json.dumps({"where": call["where"], "what": call["name"] + "(...)",
                              "error": f"{head} has no attribute {missing!r}"}))
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        params = sig.parameters
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            continue
        for kw in call["keywords"]:
            p = params.get(kw)
            if p is None or p.kind is p.POSITIONAL_ONLY:
                print(json.dumps({"where": call["where"], "what": call["name"] + "(...)",
                                  "error": f"no keyword argument {kw!r}; the signature "
                                           f"is {call['name']}{sig}"}))
'''


def sandbox_env(parent: dict[str, str], home: Path, tmp: Path) -> dict[str, str]:
    """The environment a snippet runs under, built from an allowlist."""
    env = {k: parent[k] for k in ENV_ALLOWLIST if k in parent}
    env.update({
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "JAX_PLATFORMS": "cpu",
        "MPLBACKEND": "Agg",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(SRC),
    })
    return env


@dataclasses.dataclass
class Result:
    unit: Unit
    returncode: int | None             # None: timed out / refused
    output: str
    seconds: float
    refused: str | None = None


def _python_flags() -> list[str]:
    # Deprecated API used in a snippet's own code (warnings attributed to
    # the snippet's namespace, ``__main__``) is a stale doc.
    return ["-W", "error::DeprecationWarning:__main__",
            "-W", "error::FutureWarning:__main__"]


def run_unit(unit: Unit, workdir: Path, cpu: int | None = None,
             timeout: float = SNIPPET_TIMEOUT_S) -> Result:
    """Run one unit in a fresh, sandboxed process."""
    for block in unit.blocks:
        hits = launch_hits(block.code)
        if hits:
            # Defence in depth: collection already refuses these.  Never start
            # a process for one, whatever its marker says.
            return Result(unit, None, "", 0.0,
                          refused=f"{block.where} names a launch path: {', '.join(hits)}")
    home, cwd, tmp = workdir / "home", workdir / "cwd", workdir / "tmp"
    for d in (home, cwd, tmp):
        d.mkdir(parents=True)
    driver = workdir / "driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    payload = workdir / "payload.json"
    payload.write_text(json.dumps({"blocks": [
        {"path": str(b.path), "first_line": b.fence_line + 1, "code": b.code}
        for b in unit.blocks]}), encoding="utf-8")
    env = sandbox_env(dict(os.environ), home, tmp)
    if cpu is not None:
        env["MADDENING_SNIPPET_CPU"] = str(cpu)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, *_python_flags(), str(driver), str(payload)],
            cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True,
            text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # On a timeout the captured output can be bytes even under text=True.
        out = "".join(x.decode("utf-8", "replace") if isinstance(x, bytes) else (x or "")
                      for x in (exc.stdout, exc.stderr))
        return Result(unit, None, out + f"\n[timed out after {timeout:.0f} s]",
                      time.perf_counter() - t0)
    return Result(unit, proc.returncode, proc.stdout + proc.stderr,
                  time.perf_counter() - t0)


def _cpus() -> list[int | None]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))[:8]
    return [None] * min(os.cpu_count() or 1, 8)


def run_units(todo: list[Unit], root: Path) -> list[Result]:
    """Run every unit, in parallel, one process per CPU of our affinity mask."""
    free: queue.Queue[int | None] = queue.Queue()
    cpus = _cpus()
    for c in cpus:
        free.put(c)

    def one(index_unit: tuple[int, Unit]) -> Result:
        index, unit = index_unit
        cpu = free.get()
        try:
            return run_unit(unit, root / f"u{index:03d}", cpu)
        finally:
            free.put(cpu)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        return list(pool.map(one, enumerate(todo)))


def _missing(modules: tuple[str, ...]) -> list[str]:
    return [m for m in modules if importlib.util.find_spec(m.split(".")[0]) is None
            or importlib.util.find_spec(m) is None]


def _relativise(text: str) -> str:
    return text.replace(str(REPO_ROOT) + os.sep, "")


# ── Static import resolution ────────────────────────────────────────────


def statements(code: str) -> list[tuple[int, ast.Module]]:
    """``code`` parsed as ``(line offset, module)`` chunks, tolerating fragments.

    A block that is valid Python is one chunk.  A fragment that is not (a
    dict entry, a method body, a bare ``return``) is cut into the shortest
    runs of lines that parse, after dedenting; a line that starts no such
    run within 30 lines is skipped.
    """
    try:
        return [(0, ast.parse(code))]
    except SyntaxError:
        pass
    out: list[tuple[int, ast.Module]] = []
    lines = code.splitlines()
    i = 0
    while i < len(lines):
        for j in range(i, min(i + 30, len(lines))):
            try:
                module = ast.parse(textwrap.dedent("\n".join(lines[i:j + 1])))
            except SyntaxError:
                continue
            out.append((i, module))
            i = j + 1
            break
        else:
            i += 1
    return out


def _nodes(code: str):
    """``(line offset, node)`` for every AST node of every chunk of ``code``."""
    for start, module in statements(code):
        for node in ast.walk(module):
            if hasattr(node, "lineno"):
                yield start + node.lineno - 1, node


def maddening_imports(code: str) -> list[tuple[int, ast.Import | ast.ImportFrom]]:
    """``(line offset, statement)`` for every ``maddening`` import in ``code``."""
    found = []
    for offset, node in _nodes(code):
        if isinstance(node, ast.ImportFrom) and node.level == 0 \
                and (node.module or "").split(".")[0] == "maddening":
            found.append((offset, node))
        elif isinstance(node, ast.Import) \
                and any(a.name.split(".")[0] == "maddening" for a in node.names):
            found.append((offset, node))
    return found


def keyword_calls(code: str, names: set[str]) -> list[tuple[int, str, list[str]]]:
    """``(line offset, name, keywords)`` for each call of one of ``names``."""
    out = []
    for offset, node in _nodes(code):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in names:
            keywords = [k.arg for k in node.keywords if k.arg is not None]
            if keywords:
                out.append((offset, node.func.id, keywords))
    return out


#: The docs' universal name for a :class:`~maddening.GraphManager`.
GRAPH_NAME = "gm"


def graph_method_calls(code: str) -> list[tuple[int, str, list[str]]]:
    """``(line offset, method, keywords)`` for each ``gm.<method>(...)`` call.

    Every guide spells a ``GraphManager`` ``gm``, so a fragment's calls on
    it are checked against the class: the method must exist and take the
    keywords passed.
    """
    out = []
    for offset, node in _nodes(code):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == GRAPH_NAME:
            out.append((offset, node.func.attr,
                        [k.arg for k in node.keywords if k.arg is not None]))
    return out


def _module_file(module: str) -> Path | None:
    base = SRC.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def statically_resolves(module: str, name: str | None) -> str | None:
    """Check ``from module import name`` by reading source; an error or None."""
    path = _module_file(module)
    if path is None:
        return f"no module {module!r} under src/"
    if name is None:
        return None
    if _module_file(f"{module}.{name}") is not None:
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bound: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            bound.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            bound.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
    if name in bound:
        return None
    return f"{module!r} binds no {name!r} at module level (checked by reading the source)"


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tree():
    snippets, errors = collect()
    return snippets, errors


#: Synthetic snippets run beside the real ones, in the same pool, to prove
#: in every run that the runner can fail and that the sandbox holds.
#: ``name: (code, "fail" or "pass", text the output must contain)``.  A
#: failing canary must also name its doc line: it is placed at line 10 of
#: ``docs/canary-<name>.md``, and its fault is on its last line of code.
_CANARIES = {
    "bad-import": ("from maddening.core.no_such_module import nothing\n", "fail",
                   "ModuleNotFoundError"),
    "wrong-keyword": ("from maddening import GraphManager\n"
                      "GraphManager().add_edge('a', 'b', 'x', 'y', no_such_keyword=1)\n",
                      "fail", "no_such_keyword"),
    "deprecated-call": ("import warnings\n"
                        "warnings.warn('canary: a deprecated API', DeprecationWarning)\n",
                        "fail", "DeprecationWarning"),
    "sandbox": ("import os, socket\n"
                "assert os.listdir(os.environ['HOME']) == [], 'HOME is not empty'\n"
                "assert os.getcwd() != os.environ['HOME']\n"
                "assert os.environ['JAX_PLATFORMS'] == 'cpu'\n"
                # 'RUN' 'POD' split: the literal would trip LAUNCH_PATTERNS,
                # which refuses this canary before it starts (as it should).
                "leaked = [k for k in os.environ if any(s in k.upper() for s in\n"
                "          ('RUN' 'POD', 'AWS', 'GOOGLE', 'AZURE', 'LAMBDA', 'SKY', 'TOKEN',\n"
                "           'SECRET', 'KEY', 'CREDENTIAL'))]\n"
                "assert not leaked, leaked\n"
                "try:\n"
                "    socket.create_connection(('192.0.2.1', 9), timeout=1)\n"
                "except PermissionError:\n"
                "    pass\n"
                "else:\n"
                "    raise AssertionError('a non-loopback connection was allowed')\n"
                "import maddening, pathlib\n"
                "print('MADDENING_FILE=' + maddening.__file__)\n",
                "pass", "MADDENING_FILE="),
}


def _canary_unit(name: str, code: str) -> Unit:
    return Unit((Snippet(REPO_ROOT / "docs" / f"canary-{name}.md", 10, code, Marker()),))


@pytest.fixture(scope="module")
def snippet_results(tree, tmp_path_factory):
    """Run every runnable snippet once, in parallel; the canaries alongside."""
    snippets, _ = tree
    todo, skipped = [], {}
    for unit in units(snippets):
        missing = _missing(unit.requires)
        if missing:
            skipped[unit.where] = missing
        else:
            todo.append(unit)
    canaries = [_canary_unit(n, c) for n, (c, _, _) in _CANARIES.items()]
    root = tmp_path_factory.mktemp("docs_snippets")
    t0 = time.perf_counter()
    results = run_units(todo + canaries, root)
    wall = time.perf_counter() - t0
    return {
        "real": results[:len(todo)],
        "canaries": dict(zip(_CANARIES, results[len(todo):])),
        "skipped": skipped,
        "wall": wall,
        "cpus": len(_cpus()),
    }


# ── The tests ───────────────────────────────────────────────────────────


def test_the_scan_finds_the_documentation(tree):
    """A scan that collects nothing passes every other test here."""
    snippets, _ = tree
    files = {s.path for s in snippets}
    assert REPO_ROOT / "README.md" in files
    assert REPO_ROOT / "docs" / "user_guide" / "quickstart.md" in files
    runnable = sum(s.runs for s in snippets)
    assert runnable == EXPECTED_RUNNABLE, (
        f"{runnable} documentation snippets run; EXPECTED_RUNNABLE says "
        f"{EXPECTED_RUNNABLE}.  If you added or removed a runnable snippet, or "
        "marked one no-run, change EXPECTED_RUNNABLE in this file to match.")


def test_every_snippet_marker_is_well_formed(tree):
    """Orphaned, malformed or reasonless markers, and broken ``continues``."""
    snippets, errors = tree
    errors = errors + chain_errors(snippets)
    assert not errors, "\n".join(errors)


def test_no_snippet_that_names_a_launch_path_is_run(tree):
    """Every snippet matching LAUNCH_PATTERNS is no-run, category ``cloud``."""
    snippets, _ = tree
    bad = [f"{s.where}: {', '.join(launch_hits(s.code))}"
           for s in snippets if launch_hits(s.code) and s.category != "cloud"]
    assert not bad, ("snippets that name a cloud launch path must be marked "
                     "'<!-- snippet: no-run, reason: cloud: ... -->':\n" + "\n".join(bad))
    for unit in units(snippets):
        assert not any(launch_hits(b.code) for b in unit.blocks), unit.where


def test_every_runnable_snippet_runs(snippet_results):
    """Each runnable snippet exits 0 in a fresh sandboxed process."""
    failures = []
    for r in snippet_results["real"]:
        if r.refused or r.returncode != 0:
            tail = "\n".join(_relativise(r.output).strip().splitlines()[-25:])
            failures.append(f"--- {r.unit.where} ({r.refused or f'exit {r.returncode}'}, "
                            f"{r.seconds:.1f} s)\n{tail}")
    slowest = sorted(snippet_results["real"], key=lambda r: -r.seconds)[:3]
    summary = (f"{len(snippet_results['real'])} units in {snippet_results['wall']:.1f} s on "
               f"{snippet_results['cpus']} CPUs; slowest: "
               + ", ".join(f"{r.unit.where} {r.seconds:.1f} s" for r in slowest))
    assert not failures, summary + "\n\n" + "\n\n".join(failures)
    if snippet_results["skipped"]:
        pytest.skip("ran every snippet except, for a missing optional dependency: "
                    + "; ".join(f"{w} needs {', '.join(m)}"
                                for w, m in snippet_results["skipped"].items()))


def test_the_runner_fails_a_snippet_that_is_wrong(snippet_results):
    """A bad import, a wrong keyword and a deprecated call fail, in this very run.

    And the traceback names the doc and the line the fault is on.
    """
    for name, (code, expect, needle) in _CANARIES.items():
        r = snippet_results["canaries"][name]
        if expect == "fail":
            assert r.returncode not in (0, None), f"canary {name!r} passed:\n{r.output}"
            assert needle in r.output, f"canary {name!r} failed for another reason:\n{r.output}"
            line = 10 + len(code.splitlines())
            assert f'canary-{name}.md", line {line}' in r.output, r.output


def test_a_snippet_that_hangs_is_stopped(tmp_path):
    (unit,) = units(_parse("```python\nimport time\ntime.sleep(30)\n```\n")[0])
    result = run_unit(unit, tmp_path / "u", timeout=1.0)
    assert result.returncode is None and "timed out" in result.output


def test_the_sandbox_holds_and_runs_this_tree(snippet_results):
    """Empty HOME, no credentials, no network -- and this checkout's maddening."""
    r = snippet_results["canaries"]["sandbox"]
    assert r.returncode == 0, r.output
    line = next(ln for ln in r.output.splitlines() if ln.startswith("MADDENING_FILE="))
    assert Path(line.split("=", 1)[1]).resolve().is_relative_to(SRC.resolve()), line


def test_the_environment_is_built_from_an_allowlist(tmp_path):
    parent = {"PATH": "/bin", "RUNPOD_API_KEY": "x", "AWS_SECRET_ACCESS_KEY": "x",
              "GOOGLE_APPLICATION_CREDENTIALS": "x", "MADDENING_API_TOKEN": "x",
              "SKYPILOT_DEBUG": "1", "HOME": "/home/someone", "JAX_PLATFORMS": "cuda"}
    env = sandbox_env(parent, tmp_path / "h", tmp_path / "t")
    assert env["PATH"] == "/bin"
    assert env["HOME"] == str(tmp_path / "h")
    assert env["JAX_PLATFORMS"] == "cpu"
    for leaked in ("RUNPOD_API_KEY", "AWS_SECRET_ACCESS_KEY",
                   "GOOGLE_APPLICATION_CREDENTIALS", "MADDENING_API_TOKEN", "SKYPILOT_DEBUG"):
        assert leaked not in env


def test_no_run_snippets_import_only_what_exists(tree, tmp_path):
    """An illustrative snippet still may not import a name that does not exist.

    Imports from a ``cloud`` snippet are resolved by reading the source:
    nothing in such a snippet is ever executed, its imports included.
    """
    snippets, _ = tree
    items: list[dict] = []
    failures: list[str] = []
    for s in snippets:
        if s.runs:
            continue
        path = s.where.rsplit(":", 1)[0]
        for offset, node in maddening_imports(s.code):
            where = f"{path}:{s.fence_line + 1 + offset}"
            if s.category == "cloud":
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        err = statically_resolves(node.module or "", alias.name)
                        if err:
                            failures.append(f"{where}: {err}")
                else:
                    for alias in node.names:
                        err = statically_resolves(alias.name, None)
                        if err:
                            failures.append(f"{where}: {err}")
                continue
            bound = ({a.asname or a.name for a in node.names}
                     if isinstance(node, ast.ImportFrom) else set())
            items.append({"where": where, "statement": ast.unparse(node), "calls": [
                {"where": f"{path}:{s.fence_line + 1 + off}", "name": name, "keywords": kws}
                for off, name, kws in keyword_calls(s.code, bound)]})
        if s.category == "cloud":
            continue
        graph_calls = graph_method_calls(s.code)
        if graph_calls:
            items.append({"where": s.where, "statement": "from maddening import GraphManager",
                          "calls": [{"where": f"{path}:{s.fence_line + 1 + off}",
                                     "name": f"GraphManager.{method}", "keywords": kws}
                                    for off, method, kws in graph_calls]})
    assert items, "no no-run snippet imports anything from maddening: is the scan broken?"
    assert any(i["calls"] for i in items), "no keyword argument found to check"
    checker = tmp_path / "check_imports.py"
    checker.write_text(_IMPORT_CHECKER, encoding="utf-8")
    statements = tmp_path / "statements.json"
    statements.write_text(json.dumps(items), encoding="utf-8")
    home, tmp = tmp_path / "home", tmp_path / "tmp"
    home.mkdir()
    tmp.mkdir()
    proc = subprocess.run([sys.executable, str(checker), str(statements)],
                          cwd=tmp, env=sandbox_env(dict(os.environ), home, tmp),
                          stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          timeout=SNIPPET_TIMEOUT_S)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.startswith("{")]
    origin = records[0]["maddening"]
    assert Path(origin).resolve().is_relative_to(SRC.resolve()), origin
    failures += [f"{r['where']}: {r['what']}  ->  {r['error']}" for r in records[1:]]
    assert not failures, "\n".join(failures)


# ── The machinery, on synthetic Markdown ────────────────────────────────


def _parse(md: str):
    snippets, errors = parse_markdown(textwrap.dedent(md), REPO_ROOT / "docs" / "x.md")
    return snippets, errors + chain_errors(snippets)


def test_a_marker_needs_a_categorised_reason():
    for marker in ("<!-- snippet: no-run -->",
                   "<!-- snippet: no-run, reason: -->",
                   "<!-- snippet: no-run, reason: because -->",
                   "<!-- snippet: no-run, reason: fragment: -->",
                   "<!-- snippet: reason: fragment: a method -->",
                   "<!-- snippet: no-run, requires: pxr, reason: fragment: x y z -->",
                   "<!-- snippet: no-rnu, reason: fragment: a method body -->",
                   "<!-- snippet no-run -->"):
        _, errors = _parse(f"{marker}\n```python\nx = 1\n```\n")
        assert errors, marker
    snippets, errors = _parse("<!-- snippet: no-run, reason: fragment: a method body -->\n"
                              "```python\nx = 1\n```\n")
    assert not errors and snippets[0].category == "fragment"


def test_a_marker_must_sit_directly_on_its_fence():
    for md in ("<!-- snippet: continues -->\n\n```python\nx = 1\n```\n",
               "<!-- snippet: no-run, reason: fragment: abc -->\ntext\n```python\nx\n```\n",
               "<!-- snippet: no-run, reason: fragment: abc -->\n```bash\nls\n```\n",
               "- item\n\n  <!-- snippet: no-run, reason: fragment: abc -->\n   ```python\n"
               "   x\n   ```\n",
               "<!-- snippet: no-run, reason: fragment: abc -->\n"):
        _, errors = _parse(md)
        assert errors, md


def test_continues_joins_blocks_and_refuses_a_broken_chain():
    snippets, errors = _parse("```python\nx = 1\n```\n\n<!-- snippet: continues -->\n"
                              "```python\nassert x == 1\n```\n")
    assert not errors
    (unit,) = units(snippets)
    assert [b.fence_line for b in unit.blocks] == [1, 6]
    # A no-run block carrying `continues` is skipped; the chain runs on past it.
    snippets, errors = _parse(
        "```python\nx = 1\n```\n"
        "<!-- snippet: continues, no-run, reason: fragment: a sketch -->\n```python\nx(\n```\n"
        "<!-- snippet: continues -->\n```python\nassert x == 1\n```\n")
    assert not errors
    (unit,) = units(snippets)
    assert [b.fence_line for b in unit.blocks] == [1, 9]
    _, errors = _parse("<!-- snippet: continues -->\n```python\nx\n```\n")
    assert any("first Python block" in e for e in errors)
    _, errors = _parse("<!-- snippet: no-run, reason: fragment: abc -->\n```python\nx\n```\n"
                       "<!-- snippet: continues -->\n```python\nx\n```\n")
    assert any("which is no-run" in e for e in errors)
    # Without `continues` on it, a no-run block in between starts a new chain.
    _, errors = _parse("```python\nx = 1\n```\n"
                       "<!-- snippet: no-run, reason: fragment: abc -->\n```python\nx(\n```\n"
                       "<!-- snippet: continues -->\n```python\nassert x == 1\n```\n")
    assert any("which is no-run" in e for e in errors)


def test_a_launch_snippet_without_the_marker_is_refused_and_never_started(tmp_path,
                                                                         monkeypatch):
    """Refused by collection *and* by the runner, before any process starts.

    The runner's refusal is tested with process creation stubbed to raise,
    and every sample opens with ``raise SystemExit``, so that even with the
    refusal broken (a mutation run) nothing here can reach a launch path.
    """
    def no_process(*args, **kwargs):
        raise AssertionError(f"a launch-path snippet reached subprocess.run: {args!r}")

    monkeypatch.setattr(subprocess, "run", no_process)
    for i, code in enumerate((
                 "from maddening.cloud.session import CloudSession\nCloudSession().launch(c)",
                 "requests.post('http://h/cloud/launch')", "launch_vm(cfg)",
                 "import sky\n", "sky.launch(task)", "os.environ['RUNPOD_API_KEY']",
                 "subprocess.run(['python', 'run_pod.py'])", "launcher.launch('j.yaml')")):
        code = "raise SystemExit('never run')\n" + code
        snippets, errors = _parse(f"```python\n{code}\n```\n")
        assert any("launch path" in e for e in errors), code
        workdir = tmp_path / f"u{i}"
        result = run_unit(units(snippets)[0], workdir)
        assert result.refused and result.returncode is None and not workdir.exists(), code
        # Marked no-run with another category: still an error.
        _, errors = _parse(f"<!-- snippet: no-run, reason: fragment: abc -->\n"
                           f"```python\n{code}\n```\n")
        assert any("launch path" in e for e in errors), code
    _, errors = _parse("```python\nsubprocess.run(['python', 'run_pod.py', '--dry-run'])\n```\n")
    assert not errors


def test_python_blocks_inside_directives_and_code_block_options_are_found():
    md = ("````{note}\nText.\n```python\nx = 1\n```\n````\n\n"
          "```{code-block} python\n:caption: A caption\nimport os\n```\n\n"
          "```{code-block} console\n$ ls\n```\n```bash\nls\n```\n")
    snippets, errors = _parse(md)
    assert not errors
    assert [(s.fence_line, s.code.strip()) for s in snippets] == [(3, "x = 1"),
                                                                 (8, "import os")]
    # The caption is blanked, not removed, so line numbers still match the doc.
    assert snippets[1].code.splitlines()[1] == "import os"


def test_imports_are_found_in_fragments_that_do_not_parse():
    code = ('"force": Spec(shape=(2,))\n'
            "from maddening.core.node import (\n    BoundaryInputSpec,\n)\n"
            "    def f(self): ...\n")
    found = maddening_imports(code)
    assert [(off, ast.unparse(n)) for off, n in found] == [
        (1, "from maddening.core.node import BoundaryInputSpec")]
    assert statically_resolves("maddening.core.node", "BoundaryInputSpec") is None
    assert statically_resolves("maddening.core.node", "NoSuchThing")
    assert statically_resolves("maddening.core.no_such_module", None)
