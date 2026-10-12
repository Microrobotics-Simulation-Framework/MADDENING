#!/usr/bin/env python3
"""Report glossary terms that a documentation page uses without a tooltip.

A page links a glossary term with the MyST role ``{term}`Name```, or
``{term}`shown text <Name>``` where the text differs, and the docs site
pops the term's definition up on hover.  This script lists, for every
Markdown page under ``docs/``, each glossary term whose *first* occurrence
on the page is plain prose rather than such a role.  It prints the page,
the line and the term.

It is a report, not a gate: it always exits 0.  A word can be a glossary
term in one sentence and an everyday word in the next ("node" of a mesh,
"edge" of a domain), so every line it prints is a candidate to read, not
a defect.

What is read as prose
    Paragraphs, list items, table cells, and the bodies of prose
    directives (``note``, ``warning``, ``versionadded``, ...).

What is not read
    YAML front matter; fenced code and code-like directives
    (``code-block``, ``mermaid``, ``toctree``, ``glossary``, ``math``,
    ...); four-space indented code outside a list; ``$$`` math blocks;
    headings, ATX and setext, and a directive's argument and option
    lines; MyST comments (``%``) and targets (``(label)=``); HTML
    comments; link reference definitions.  Inside prose: inline code,
    the content of every MyST role (``{term}`` included), inline math,
    links and images (text and target), autolinks and bare URLs, HTML
    tags, citations (``[@Key]``) and footnote references.

How a term is matched
    Case-insensitively, as whole words, with an optional plural ``s`` or
    ``es``.  The parts of a term may be joined by any whitespace (a line
    break included), a hyphen or an en dash: ``sub-cycling`` matches
    ``sub cycling``, and ``Cramér-Rao`` matches ``Cramér–Rao``.  A term written inside a longer term (``UMR``
    in ``Helical UMR``) counts as the longer term only.  A role counts for
    the term it names, compared case-insensitively, as Sphinx compares it.

The glossary pages themselves and anything under ``release_notes/`` are
not scanned.

Usage::

    python scripts/report_glossary_coverage.py                # docs/, docs/glossary.md
    python scripts/report_glossary_coverage.py --context      # with each source line
    python scripts/report_glossary_coverage.py --by-term      # pages per term
    python scripts/report_glossary_coverage.py --glossary docs/glossary.md \\
        --glossary draft_terms.md                             # extra terms, same markup
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directives whose body is not prose.  Any other directive's body is read
#: as prose; its argument (the rest of the opening line) and its leading
#: ``:option:`` lines are not.  ``auto*`` directives count as code too.
CODE_DIRECTIVES = frozenset({
    "bibliography", "code", "code-block", "code-cell", "contents", "csv-table",
    "eval-rst", "glossary", "graphviz", "image", "include", "literalinclude",
    "math", "mermaid", "mermaid-tips", "raw", "sourcecode", "toctree",
})

FENCE = re.compile(r"^(?P<indent>\s*)(?P<fence>`{3,}|~{3,}|:{3,})(?P<info>.*)$")
DIRECTIVE = re.compile(r"^\{(?P<name>[^}\s]+)\}")
OPTION = re.compile(r"^\s*:[\w-]+:")
ATX_HEADING = re.compile(r"^\s{0,3}#{1,6}(?:\s|$)")
SETEXT_UNDERLINE = re.compile(r"^\s{0,3}(?:=+|-+)\s*$")
THEMATIC_BREAK = re.compile(r"^\s{0,3}(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})$")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
MYST_COMMENT = re.compile(r"^\s*%")
MYST_TARGET = re.compile(r"^\s*\([^()\s]+\)=\s*$")
BLOCK_BREAK = re.compile(r"^\s*\+\+\+")
LINK_DEFINITION = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*\S")

#: Inline constructs whose text is not prose, tried left to right as one
#: alternation, so whichever starts first wins: a role example written
#: inside a code span stays code, and a role's backticks are not a span.
INLINE = re.compile(
    r"""
      (?P<comment><!--.*?-->)
    | (?P<role>\{(?P<rname>[A-Za-z][\w:.+-]*)\}(?<!`)(?P<rtick>`+)(?!`)
               (?P<rbody>.+?)(?<!`)(?P=rtick)(?!`))
    | (?P<code>(?<!`)(?P<tick>`+)(?!`).+?(?<!`)(?P=tick)(?!`))
    | (?P<dmath>\$\$.+?\$\$)
    | (?P<math>(?<![\\$\w])\$(?![\s$]).+?(?<![\s\\])\$(?![\d$]))
    | (?P<link>!?\[(?:[^\[\]]|\[[^\[\]]*\])*\]
               (?:\((?:[^()]|\([^()]*\))*\)|\[[^\[\]]*\]))
    | (?P<cite>\[(?:[^\[\]]*@|\^)[^\[\]]*\])
    | (?P<url><(?:https?|ftp|mailto):[^>\s]+>|\b(?:https?://|www\.)[^\s<>]+)
    | (?P<tag></?[A-Za-z][\w:-]*(?:\s[^<>]*)?/?>)
    | (?P<subst>\{\{.*?\}\})
    """,
    re.S | re.X,
)

ROLE_TARGET = re.compile(r"^(?P<text>.*?)\s*<(?P<target>[^<>]+)>$", re.S)


@dataclass(frozen=True)
class Candidate:
    """A glossary term whose first occurrence on a page has no role."""

    page: str
    line: int
    column: int
    term: str
    text: str


@dataclass
class PageReport:
    """What one page uses: candidates, covered terms, unknown role targets."""

    page: str
    path: Path
    candidates: list
    used: set
    unknown_roles: list


# ---------------------------------------------------------------------------
# The glossary
# ---------------------------------------------------------------------------

def read_terms(path: Path) -> list:
    """The terms a MyST ``{glossary}`` block in ``path`` defines, in order.

    A term is a line at the block's own indentation that is neither blank
    nor an option (``:sorted:``); its definition is the indented lines
    after it.  Consecutive term lines share one definition, as in Sphinx.
    A ``term : classifier`` line yields ``term``.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    terms = []
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        directive = DIRECTIVE.match(m.group("info").strip()) if m else None
        if not (directive and directive.group("name") == "glossary"):
            i += 1
            continue
        indent, fence = len(m.group("indent")), m.group("fence")
        i += 1
        while i < len(lines):
            line = lines[i]
            closing = FENCE.match(line)
            if (closing and not closing.group("info").strip()
                    and closing.group("fence")[0] == fence[0]
                    and len(closing.group("fence")) >= len(fence)):
                break
            body = line[indent:] if len(line) >= indent else line.lstrip()
            if body.strip() and not body[0].isspace() and not body.startswith(":"):
                terms.append(re.sub(r"\s+:\s.*$", "", body.strip()))
            i += 1
        i += 1
    return terms


def _key(name: str) -> str:
    return " ".join(name.split()).casefold()


#: Between the parts of a term: whitespace, a hyphen or an en dash, so
#: "sub-cycling" matches "sub cycling" and "Cramér-Rao" "Cramér–Rao".
_JOINER = "[\\s\\-\u2010\u2011\u2013]+"


def term_pattern(term: str) -> str:
    """A regex for ``term``: whole words, any joiner between parts, plurals."""
    parts = [p for p in re.split(_JOINER, term) if p]
    escaped = [re.escape(p) for p in parts]
    last = parts[-1]
    # A term written in the plural ("Boundary inputs") also matches its
    # singular: drop the final "s" and let the plural suffix put it back.
    if last[-1:] == "s" and last[-2:-1].islower() and not last.endswith("ss"):
        escaped[-1] = re.escape(last[:-1])
    return r"(?<!\w)" + _JOINER.join(escaped) + r"(?:s|es)?(?!\w)"


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------

def prose_lines(lines: list) -> list:
    """``lines`` with every line that is not prose replaced by ``None``.

    Prose lines keep their text, so line numbers are unchanged.  The
    rules are listed in the module docstring.
    """
    n = len(lines)
    out = [None] * n
    i = 0
    if n and lines[0].strip() == "---":                     # YAML front matter
        for j in range(1, n):
            if lines[j].strip() in ("---", "..."):
                i = j + 1
                break
    stack = []           # open prose directives: (fence char, fence length)
    in_list = False
    prev_blank = True
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            prev_blank = True
            i += 1
            continue

        fence = FENCE.match(line)
        if fence and not (fence.group("fence")[0] == ":"
                          and fence.group("info").strip()[:1] not in ("", "{")):
            char, length = fence.group("fence")[0], len(fence.group("fence"))
            info = fence.group("info").strip()
            if (not info and stack and char == stack[-1][0]
                    and length >= stack[-1][1]):
                stack.pop()                             # a prose directive ends
                i, prev_blank = i + 1, True
                continue
            if char == "`" and "`" in info:
                fence = None                            # an inline code span, not a fence
            elif char == ":" and not info:
                i, prev_blank = i + 1, True             # a stray closing colon fence
                continue
            else:
                directive = DIRECTIVE.match(info)
                name = directive.group("name") if directive else None
                if (name is not None and name not in CODE_DIRECTIVES
                        and not name.startswith("auto")):
                    stack.append((char, length))        # prose directive: read its body
                    i += 1
                    while i < n and OPTION.match(lines[i]):
                        i += 1
                    prev_blank = True
                    continue
                i += 1                                  # code: skip to its closing fence
                while i < n:
                    closing = FENCE.match(lines[i])
                    if (closing and not closing.group("info").strip()
                            and closing.group("fence")[0] == char
                            and len(closing.group("fence")) >= length):
                        break
                    i += 1
                i, prev_blank = i + 1, True
                continue

        expanded = line.expandtabs(4)
        indent = len(expanded) - len(expanded.lstrip())
        if LIST_ITEM.match(line):
            in_list = True
        elif prev_blank and indent == 0:
            in_list = False

        if prev_blank and indent >= 4 and not in_list:   # indented code block
            while i < n and (not lines[i].strip() or len(lines[i].expandtabs(4))
                             - len(lines[i].expandtabs(4).lstrip()) >= 4):
                i += 1
            prev_blank = True
            continue

        if stripped.startswith("$$"):                    # display math
            j = i
            if not (len(stripped) > 2 and stripped.endswith("$$")):
                j = i + 1
                while j < n and not lines[j].rstrip().endswith("$$"):
                    j += 1
            i, prev_blank = j + 1, True
            continue

        if "<!--" in line and "-->" not in line[line.index("<!--"):]:
            head = line[:line.index("<!--")]
            if head.strip():
                out[i] = head
            j = i + 1
            while j < n and "-->" not in lines[j]:
                j += 1
            if j < n:
                tail = lines[j][lines[j].index("-->") + 3:]
                if tail.strip():
                    out[j] = " " * (len(lines[j]) - len(tail)) + tail
            i, prev_blank = j + 1, False
            continue

        if (ATX_HEADING.match(line) or MYST_COMMENT.match(line)
                or MYST_TARGET.match(line) or BLOCK_BREAK.match(line)
                or LINK_DEFINITION.match(line) or THEMATIC_BREAK.match(line)):
            i, prev_blank = i + 1, True
            continue

        if (i + 1 < n and SETEXT_UNDERLINE.match(lines[i + 1])
                and not LIST_ITEM.match(line)):
            i, prev_blank = i + 2, True                 # setext heading + underline
            continue

        out[i] = line
        prev_blank = False
        i += 1
    return out


def _paragraphs(kept: list):
    """(first line number, joined text) for each run of adjacent prose lines."""
    start, buf = None, []
    for number, text in enumerate(kept, start=1):
        if text is None or not text.strip():
            if buf:
                yield start, "\n".join(buf)
            start, buf = None, []
            continue
        if start is None:
            start = number
        buf.append(text)
    if buf:
        yield start, "\n".join(buf)


def _column(text: str, offset: int) -> int:
    """The 0-based column of ``offset`` in the line of ``text`` holding it."""
    return offset - (text.rfind("\n", 0, offset) + 1)


def _blank(text: str, start: int, end: int) -> str:
    """``text`` with ``[start, end)`` blanked, keeping line breaks."""
    return text[:start] + re.sub(r"[^\n]", " ", text[start:end]) + text[end:]


def scan_page(path: Path, terms: list, label: str = None) -> PageReport:
    """Candidates on one page, for the given glossary ``terms``."""
    label = label or str(path)
    keys = {_key(t): t for t in terms}
    patterns = [(t, re.compile(term_pattern(t), re.I)) for t in terms]
    lines = path.read_text(encoding="utf-8").splitlines()

    events = []          # (line, column, term, kind, matched text)
    unknown = []
    for first, text in _paragraphs(prose_lines(lines)):
        masked = text
        for m in INLINE.finditer(text):
            masked = _blank(masked, m.start(), m.end())
            if m.group("role") and m.group("rname") in ("term", "std:term"):
                body = " ".join(m.group("rbody").split())
                target = ROLE_TARGET.match(body)
                name = target.group("target") if target else body
                line = first + text.count("\n", 0, m.start())
                if _key(name) in keys:
                    events.append((line, _column(text, m.start()), keys[_key(name)],
                                   "role", body))
                else:
                    unknown.append((line, name))
        found = []
        for term, pattern in patterns:
            for m in pattern.finditer(masked):
                found.append((m.start(), m.end(), term, m.group(0)))
        found.sort(key=lambda f: (f[0], f[0] - f[1]))
        covered_to = -1
        for start, end, term, matched in found:
            if start < covered_to:
                continue                     # inside a longer term's match
            covered_to = end
            line = first + text.count("\n", 0, start)
            events.append((line, _column(text, start), term, "bare",
                           " ".join(matched.split())))

    events.sort(key=lambda e: (e[0], e[1]))
    first_seen = {}
    for event in events:
        first_seen.setdefault(event[2], event)
    candidates = [
        Candidate(label, line, column, term, matched)
        for (line, column, term, kind, matched) in first_seen.values()
        if kind == "bare"
    ]
    candidates.sort(key=lambda c: (c.line, c.term))
    return PageReport(label, path, candidates, set(first_seen), unknown)


def pages_under(paths: list, exclude: set) -> list:
    """Markdown pages to scan: files given, and ``*.md`` under directories given."""
    pages = []
    for p in paths:
        if p.is_dir():
            pages.extend(sorted(p.rglob("*.md")))
        elif p.suffix == ".md":
            pages.append(p)
    return [p for p in pages
            if p.resolve() not in exclude and "release_notes" not in p.parts]


def _excerpt(line: str, column: int, length: int, width: int = 60) -> str:
    """About ``width`` characters of ``line`` either side of a match."""
    start, end = max(0, column - width), min(len(line), column + length + width)
    return (("..." if start else "") + line[start:end].strip()
            + ("..." if end < len(line) else ""))


def _label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", type=Path,
                        help="pages or directories to scan (default: docs/)")
    parser.add_argument("--glossary", action="append", type=Path,
                        help="a file holding a MyST {glossary} block; repeatable "
                             "(default: docs/glossary.md)")
    parser.add_argument("--context", action="store_true",
                        help="print each candidate's source line under it")
    parser.add_argument("--by-term", action="store_true",
                        help="print, per term, the pages that name it in prose")
    args = parser.parse_args(argv)

    glossaries = args.glossary or [REPO_ROOT / "docs" / "glossary.md"]
    terms = []
    for g in glossaries:
        try:
            read = read_terms(g)
        except OSError as exc:
            print(f"WARNING: cannot read glossary {g}: {exc}", file=sys.stderr)
            continue
        if not read:
            print(f"WARNING: no glossary terms found in {g}", file=sys.stderr)
        for term in read:
            if _key(term) not in {_key(t) for t in terms}:
                terms.append(term)

    exclude = {g.resolve() for g in glossaries}
    pages = pages_under(args.paths or [REPO_ROOT / "docs"], exclude)
    reports = []
    for page in pages:
        try:
            reports.append(scan_page(page, terms, _label(page)))
        except (OSError, UnicodeDecodeError) as exc:
            print(f"WARNING: cannot read {page}: {exc}", file=sys.stderr)

    total = 0
    for report in reports:
        source = None
        for c in report.candidates:
            total += 1
            shown = "" if c.text.casefold() == c.term.casefold() else f' (as "{c.text}")'
            print(f"{c.page}:{c.line}: {c.term}{shown}")
            if args.context:
                if source is None:
                    source = report.path.read_text(encoding="utf-8").splitlines()
                if 0 < c.line <= len(source):
                    print(f"    | {_excerpt(source[c.line - 1], c.column, len(c.text))}")
        for line, name in report.unknown_roles:
            print(f"{report.page}:{line}: {{term}} role names no glossary term: {name!r}")

    if args.by_term:
        print()
        for term in sorted(terms, key=str.casefold):
            named = [r.page for r in reports if term in r.used]
            bare = [r.page for r in reports if any(c.term == term for c in r.candidates)]
            print(f"{term}: named on {len(named)} page(s), "
                  f"{len(bare)} without a role at the first")
            for page in named:
                print(f"    {'-' if page in bare else '+'} {page}")

    with_candidates = sum(1 for r in reports if r.candidates)
    print(f"\n{total} candidate(s) on {with_candidates} of {len(reports)} page(s), "
          f"{len(terms)} glossary term(s) from {len(glossaries)} file(s). "
          "This is a report; it does not fail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
