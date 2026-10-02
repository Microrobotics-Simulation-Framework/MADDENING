#!/usr/bin/env python
"""Every small absolute constant in the numerical core says what units it is in.

Four 0.4.0 defects were one mistake: an absolute number inside a computation
that is otherwise relative.  The implicit-function solve's ``atol=1e-8``
returned gradients of exactly zero for a group written in small units; IQN's
``max(residual_norm, 1e-12)`` floor and ``fit_lm``'s ``1e-12`` Marquardt floor
held a small-unit problem to a different criterion from the same problem at
scale one; and the accelerators' steps flushed to zero at small magnitudes.
Each looked harmless where it was written, because the author pictured a
state of order one.

This gate makes every such constant say what it is relative to.  It scans
``src/maddening/core/``, ``src/maddening/sysid.py`` and
``src/maddening/cloud/multigpu/`` for

* a **small float literal**: a nonzero literal with ``|v| <= 1e-3``, or one
  written with a decimal exponent of ``-4`` or below (``15e-4``); and
* a **dtype constant used additively**: ``finfo(...).tiny`` / ``.eps`` /
  ``.epsneg`` / ``.smallest_normal`` / ``.smallest_subnormal`` -- directly, or
  through a name bound to one, or scaled by a product or quotient -- as an
  operand of ``+`` or ``-`` or as an argument of a floor (``max``,
  ``maximum``, ``fmax``, ``clip``).  ``x + eps`` and ``max(x, tiny)`` are
  the IQN mistake with a dtype's constant in place of a literal: they are
  absolute in ``x``'s units.  ``x * eps`` (a relative resolution) and a
  comparison against ``tiny`` (a statement about the dtype's range) are not
  flagged.

and requires each hit to be justified in one of two ways:

* an inline comment ``# units: <why>`` on the hit's line, or in the block of
  comment-only lines directly above it; or
* an entry in ``scripts/numeric_constants_allowlist.txt``::

      <path> <qualname> <token>[*<count>]  # <units justification>

  where ``<qualname>`` is the enclosing ``Class.function`` (``<module>`` at
  top level; a default argument belongs to its function), ``<token>`` is the
  literal exactly as written (``1e-12``) or, for a dtype constant,
  ``finfo.<attr>``, and ``<count>`` (default 1) is how many such hits that
  scope holds.

A justification says what the number is relative to: "dimensionless, a
fraction of the largest singular value", "seconds, the caller's own dt
floor", "bytes".  One that cannot is a units-dependent constant, and the fix
is to make it relative, not to allowlist it.

The gate fails closed:

* a hit with neither justification fails;
* a **stale** allowlist entry -- one that matches no hit, or whose count is
  not the number of hits it names -- fails, so a removed constant takes its
  entry with it and a second copy of an allowlisted literal is not silently
  absorbed by the first one's entry;
* an allowlist line that does not parse, repeats a key, or carries no
  justification fails; and
* an **empty scan scope** fails: a scope path that does not exist, a scope
  directory with no Python file, or a scan that read nothing at all.  A gate
  whose scope has narrowed to nothing reports ``OK`` forever.

What the scan does not see: a constant computed at runtime (``10 ** -8``,
``float("1e-8")``), one imported from outside the scope, and a dtype constant
reached through an attribute chain it cannot resolve (``self.info.eps``).
Review has to catch those.

Usage:
    python scripts/check_numeric_constants.py [--root ROOT] [--allowlist PATH]
        [--scope PATH ...] [--list]

``--list`` prints every hit with how it is justified.

Exit codes:
    0 -- every hit is justified and the allowlist has no stale entry
    1 -- an unjustified hit, a stale or malformed allowlist entry, or an
         empty scan scope
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from collections import Counter
from pathlib import Path
from typing import NamedTuple

#: Scanned relative to the repository root.  The numerical core, the
#: system-identification solvers, and the sharded solvers.
SCOPE = (
    "src/maddening/core",
    "src/maddening/sysid.py",
    "src/maddening/cloud/multigpu",
)
ALLOWLIST = "scripts/numeric_constants_allowlist.txt"

#: A nonzero float literal at or below this magnitude is a hit.
SMALL = 1e-3
#: A literal written with a decimal exponent at or below this is a hit.
EXPONENT = -4
FINFO_ATTRS = frozenset({"tiny", "eps", "epsneg", "smallest_normal", "smallest_subnormal"})
#: Calls whose arguments are a floor (or a ceiling) on one another.
FLOOR_CALLS = frozenset({"max", "maximum", "fmax", "clip"})
#: A justification shorter than this cannot say what it is relative to.
MIN_JUSTIFICATION = 12

_UNITS = re.compile(r"#\s*units:\s*(\S.*)$")
_EXP = re.compile(r"[eE]([+-]?\d+)")
_ENTRY = re.compile(
    r"^(?P<path>\S+)\s+(?P<qual>\S+)\s+(?P<token>[^\s*#]+)(?:\*(?P<count>\d+))?"
    r"\s*#\s*(?P<why>.*)$"
)


class Hit(NamedTuple):
    path: str
    line: int
    qualname: str
    token: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.path, self.qualname, self.token)


def _is_small_literal(node: ast.AST, source: str) -> str | None:
    """The literal's source text if it is a small float literal, else None."""
    if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
        return None
    if not isinstance(node.value, float):
        return None
    text = ast.get_source_segment(source, node) or repr(node.value)
    v = node.value
    if v == 0.0:
        return None
    small = abs(v) <= SMALL
    m = _EXP.search(text)
    low_exponent = m is not None and int(m.group(1)) <= EXPONENT
    return text if (small or low_exponent) else None


def _is_number(node: ast.AST) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_number(node.operand)
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
        and not isinstance(node.value, bool)


def _scope_nodes(body):
    """Every node of a scope's body, not descending into nested scopes."""
    todo = list(body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        yield node
        todo.extend(ast.iter_child_nodes(node))


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class _Scanner(ast.NodeVisitor):
    """Walks one module, tracking the enclosing qualname and finfo-bound names."""

    def __init__(self, path: str, source: str):
        self.path = path
        self.source = source
        self.stack: list[str] = []
        self.hits: list[Hit] = []
        # Per scope (qualname tuple): names bound to a finfo object, and
        # names bound to a dtype constant derived from one.
        self.finfo_objs: dict[tuple[str, ...], set[str]] = {}
        self.finfo_vals: dict[tuple[str, ...], set[str]] = {}
        self.finfo_attr: dict[tuple[str, ...], dict[str, str]] = {}
        self._flagged: set[int] = set()

    # -- scope bookkeeping -------------------------------------------------
    def _qual(self) -> str:
        return ".".join(self.stack) if self.stack else "<module>"

    def _scopes(self):
        for i in range(len(self.stack), -1, -1):
            yield tuple(self.stack[:i])

    def _bound(self, table, name: str) -> bool:
        return any(name in table.get(s, ()) for s in self._scopes())

    def _bind_scope(self, body) -> None:
        """Bind the finfo names assigned anywhere in this scope (to a fixpoint)."""
        scope = tuple(self.stack)
        objs = self.finfo_objs.setdefault(scope, set())
        vals = self.finfo_vals.setdefault(scope, set())
        assigns = []
        for node in _scope_nodes(body):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        assigns.append((t.id, node.value))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                    and node.value is not None:
                assigns.append((node.target.id, node.value))
        changed = True
        while changed:
            changed = False
            for name, value in assigns:
                if self._is_finfo_obj(value) and name not in objs:
                    objs.add(name)
                    changed = True
                elif self._is_finfo_val(value) and name not in vals:
                    vals.add(name)
                    self.finfo_attr.setdefault(scope, {})[name] = self._finfo_attr(value)
                    changed = True

    def _is_finfo_obj(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Call):
            return _call_name(node.func) == "finfo"
        if isinstance(node, ast.Name):
            return self._bound(self.finfo_objs, node.id)
        return False

    def _is_finfo_val(self, node: ast.AST) -> bool:
        """A dtype constant: ``finfo(d).tiny``, a name bound to one, or one scaled."""
        if isinstance(node, ast.Attribute) and node.attr in FINFO_ATTRS:
            # Any ``x.tiny`` / ``x.eps``: the object is usually a ``finfo``
            # passed in or stored (``fi.tiny``, ``self.eps``), which the scan
            # cannot resolve, and failing closed costs one justification.
            return True
        if isinstance(node, ast.Name):
            return self._bound(self.finfo_vals, node.id)
        if isinstance(node, ast.Call) and node.args and _call_name(node.func) in (
                "float", "asarray", "array", "float32", "float64", "float16"):
            return self._is_finfo_val(node.args[0])
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
            # A dtype constant scaled by a number (``100 * eps``, ``tiny /
            # eps``) is still one; scaled by a quantity (``eps * abs(x)``)
            # it is a resolution in that quantity's units, and relative.
            left, right = self._is_finfo_val(node.left), self._is_finfo_val(node.right)
            return (left and (right or _is_number(node.right))) or (
                right and _is_number(node.left))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return self._is_finfo_val(node.operand)
        return False

    def _finfo_attr(self, node: ast.AST) -> str:
        """Which dtype constant ``node`` derives from (resolving bound names)."""
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr in FINFO_ATTRS:
                return sub.attr
            if isinstance(sub, ast.Name):
                for s in self._scopes():
                    attr = self.finfo_attr.get(s, {}).get(sub.id)
                    if attr is not None:
                        return attr
        return "?"

    def _flag_finfo(self, node: ast.AST) -> None:
        if id(node) in self._flagged or not self._is_finfo_val(node):
            return
        self._flagged.add(id(node))
        self.hits.append(Hit(self.path, node.lineno, self._qual(),
                             f"finfo.{self._finfo_attr(node)}"))

    # -- visitors -----------------------------------------------------------
    def visit_Module(self, node: ast.Module) -> None:
        self._bind_scope(node.body)
        self.generic_visit(node)

    def _visit_scope(self, node, body) -> None:
        self.stack.append(node.name)
        self._bind_scope(body)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for dec in node.decorator_list:
            self.visit(dec)
        self.stack.append(node.name)
        self._bind_scope(node.body)
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        for stmt in node.body:
            self.visit(stmt)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for dec in node.decorator_list:
            self.visit(dec)
        for base in node.bases:
            self.visit(base)
        self.stack.append(node.name)
        self._bind_scope(node.body)
        for stmt in node.body:
            self.visit(stmt)
        self.stack.pop()

    def visit_Constant(self, node: ast.Constant) -> None:
        text = _is_small_literal(node, self.source)
        if text is not None:
            self.hits.append(Hit(self.path, node.lineno, self._qual(), text))

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Add, ast.Sub)):
            self._flag_finfo(node.left)
            self._flag_finfo(node.right)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.op, (ast.Add, ast.Sub)):
            self._flag_finfo(node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _call_name(node.func) in FLOOR_CALLS:
            for arg in list(node.args) + [k.value for k in node.keywords]:
                self._flag_finfo(arg)
        self.generic_visit(node)


def _comments(source: str) -> dict[int, str]:
    out: dict[int, str] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                out[tok.start[0]] = tok.string
    except (tokenize.TokenError, IndentationError):
        pass
    return out


def _inline_units(line: int, lines: list[str], comments: dict[int, str]) -> bool:
    """A ``# units: <why>`` on ``line`` or in the comment block directly above it."""
    c = comments.get(line)
    if c is not None and _UNITS.search(c):
        return True
    i = line - 1
    while i >= 1 and lines[i - 1].lstrip().startswith("#"):
        if _UNITS.search(lines[i - 1].strip()):
            return True
        i -= 1
    return False


def scan_file(path: Path, rel: str) -> tuple[list[Hit], list[Hit]]:
    """``(justified inline, needing the allowlist)`` hits of one file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    scanner = _Scanner(rel, source)
    scanner.visit(tree)
    comments = _comments(source)
    lines = source.splitlines()
    inline, listed = [], []
    for hit in sorted(scanner.hits, key=lambda h: (h.line, h.token)):
        (inline if _inline_units(hit.line, lines, comments) else listed).append(hit)
    return inline, listed


def scope_files(root: Path, scope) -> tuple[list[tuple[Path, str]], list[str]]:
    """Every Python file the scope names, and the errors of an empty scope."""
    files: list[tuple[Path, str]] = []
    errors: list[str] = []
    for entry in scope:
        p = root / entry
        if p.is_file() and p.suffix == ".py":
            files.append((p, entry))
        elif p.is_dir():
            found = sorted(p.rglob("*.py"))
            if not found:
                errors.append(f"scope {entry!r} holds no Python file: nothing would be checked")
            files.extend((f, f.relative_to(root).as_posix()) for f in found)
        else:
            errors.append(f"scope {entry!r} does not exist under {root}: "
                          "nothing would be checked there")
    if not files and not errors:
        errors.append("the scan scope is empty: nothing would be checked")
    return files, errors


def parse_allowlist(text: str) -> tuple[dict[tuple[str, str, str], tuple[int, str, int]], list[str]]:
    """``{key: (count, justification, line)}`` and the parse errors."""
    entries: dict[tuple[str, str, str], tuple[int, str, int]] = {}
    errors: list[str] = []
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENTRY.match(line)
        if m is None:
            errors.append(f"allowlist line {n} does not parse "
                          f"(want '<path> <qualname> <token>[*N]  # <why>'): {raw!r}")
            continue
        why = m.group("why").strip()
        if len(why) < MIN_JUSTIFICATION:
            errors.append(f"allowlist line {n}: the justification {why!r} cannot say what "
                          "the constant is relative to")
            continue
        key = (m.group("path"), m.group("qual"), m.group("token"))
        if key in entries:
            errors.append(f"allowlist line {n} repeats line {entries[key][2]}: {' '.join(key)}")
            continue
        count = int(m.group("count") or 1)
        if count < 1:
            errors.append(f"allowlist line {n}: count must be at least 1")
            continue
        entries[key] = (count, why, n)
    return entries, errors


def check(root: Path, allowlist: Path, scope=SCOPE) -> tuple[list[str], list[Hit], list[Hit], dict]:
    """``(errors, inline hits, allowlisted hits, allowlist entries)``."""
    files, errors = scope_files(root, scope)
    inline: list[Hit] = []
    listed: list[Hit] = []
    for path, rel in files:
        try:
            a, b = scan_file(path, rel)
        except SyntaxError as e:
            errors.append(f"{rel}: does not parse: {e}")
            continue
        inline.extend(a)
        listed.extend(b)
    try:
        text = allowlist.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = ""
        if listed:
            errors.append(f"allowlist {allowlist} does not exist")
    entries, parse_errors = parse_allowlist(text)
    errors.extend(parse_errors)
    have = Counter(h.key for h in listed)
    first_line = {}
    for h in listed:
        first_line.setdefault(h.key, h.line)
    for key, n in sorted(have.items()):
        path, qual, token = key
        if key not in entries:
            lines = sorted(h.line for h in listed if h.key == key)
            errors.append(
                f"{path}:{','.join(map(str, lines))}: {token} in {qual} has no units "
                f"justification: add '# units: <what it is relative to>' or an allowlist "
                f"entry '{path} {qual} {token}{'*' + str(n) if n > 1 else ''}  # <why>' "
                "-- or make it relative")
        elif entries[key][0] != n:
            errors.append(
                f"allowlist line {entries[key][2]}: {' '.join(key)} expects "
                f"{entries[key][0]} hit(s) and the scope holds {n}: a new occurrence needs "
                "its own justification, a removed one takes its count with it")
    for key, (count, _why, n) in sorted(entries.items(), key=lambda kv: kv[1][2]):
        if key not in have:
            errors.append(f"allowlist line {n} is stale: no {key[2]} in {key[1]} "
                          f"of {key[0]}")
    return errors, inline, listed, entries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--allowlist", default=None)
    parser.add_argument("--scope", action="append", default=None,
                        help="a scope path relative to the root (repeatable); "
                             "default: the numerical core")
    parser.add_argument("--list", action="store_true",
                        help="print every hit and how it is justified")
    args = parser.parse_args(argv)
    root = Path(args.root)
    allowlist = Path(args.allowlist) if args.allowlist else root / ALLOWLIST
    scope = tuple(args.scope) if args.scope else SCOPE
    errors, inline, listed, entries = check(root, allowlist, scope)
    if args.list:
        for h in sorted(inline + listed, key=lambda h: (h.path, h.line)):
            how = "inline" if h in inline else (
                "allowlist: " + entries[h.key][1] if h.key in entries else "UNJUSTIFIED")
            print(f"{h.path}:{h.line}: {h.qualname}: {h.token}: {how}")
    for e in errors:
        print(f"ERROR: {e}")
    n_files = len(scope_files(root, scope)[0])
    if errors:
        print(f"FAILED: {len(errors)} problem(s); {len(inline) + len(listed)} hit(s) "
              f"in {n_files} file(s)")
        return 1
    print(f"OK: {len(inline) + len(listed)} small or dtype constant(s) in {n_files} file(s), "
          f"each justified ({len(inline)} inline, {len(listed)} by "
          f"{len(entries)} allowlist entr{'y' if len(entries) == 1 else 'ies'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
