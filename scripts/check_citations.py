#!/usr/bin/env python3
"""CI bridge: verify bibliography citations across the documentation.

Parses docs/bibliography.bib for the keys BibTeX would actually define and
scans Markdown under docs/ for Pandoc-style ``[@Key]`` citations, verifying
that every cited key exists.

What the naive version of this got wrong, and why each matters:

* A ``%``-commented entry (``% @book{Crank1975,``) used to count as
  defined.  BibTeX ignores it and Sphinx renders a broken reference --
  precisely the case this gate exists to catch.
* Duplicate keys collapsed into a set, so a bibliography defining the same
  key twice (with different contents) looked fine.
* The key pattern was ``\\w+``, so a hyphenated key such as ``Van-Leer1979``
  was neither defined nor cited correctly.
* The scan covered docs/algorithm_guide/ only: 5 of 41 documentation files.
* The key pattern also demanded a leading letter, but Pandoc keys may
  begin with a digit or ``_`` (``[@1975Crank]``); a citation bracket may
  wrap onto the next line (``[see\n@Key, p. 3]``); and an entry inside
  ``@comment{...}`` is not an entry.  All three passed with an undefined
  key (audit_040_p4_1, C8-C10).

Unused bibliography entries are reported as warnings (non-blocking).

Usage:
    python scripts/check_citations.py [docs/]

Exits 0 if all citations resolve, nonzero if any are dangling or the
bibliography is malformed.
"""

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

DEFAULT_SCAN_DIR = os.path.join("docs")
DEFAULT_BIB = os.path.join("docs", "bibliography.bib")

# Citations that are deliberately dangling because the surrounding text is
# teaching the syntax rather than citing anything.  Keyed by (path relative
# to the repository root, key) so a genuinely broken citation in the same
# file is still caught, with a one-line reason each so a deliberate
# exemption stays distinguishable from an accumulated one.
#
# This was the one allowlist in the tree with no reason, no cap and no
# staleness check -- compare ``_ALLOWED_UNRESOLVABLE`` in
# check_transforms.py and ``_ALLOWED_UNSTABLE`` in check_heat_stability.py,
# both of which have all three.  ``tests/compliance/test_gate_scripts.py::
# TestCitationTemplateAllowlist`` now supplies them.
#
# Add an entry only where the prose is *demonstrating* citation syntax --
# never to quiet a citation whose key should have been added to the
# bibliography.
_TEMPLATE_CITATIONS = {
    (os.path.join("docs", "developer_guide", "documentation_standards.md"),
     "Key"):
        "the \"Cite as [@Key]\" example in the documentation standards",
    (os.path.join("docs", "developer_guide", "node_authoring.md"), "Key"):
        "the \"Cite as [@Key]\" example in the node authoring guide",
}

# A ceiling, not a target.  Two authoring guides teach the syntax; a third
# is plausible, a tenth means dangling citations are being allowlisted
# rather than fixed.
_MAX_TEMPLATE_CITATIONS = 5

# BibTeX entry types that declare no citable key.
_NON_ENTRY_TYPES = {"comment", "string", "preamble"}

# A key may contain hyphens, colons, dots and slashes -- anything but
# whitespace, a comma or a brace.
_BIB_ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,\s{}]+)\s*,")
# Pandoc's citation key: it "must begin with a letter, digit, or _, and may
# contain alphanumerics, _, and internal punctuation characters
# (:.#$%&-+?<>~/)", or be any text in braces, ``@{...}``.  The ``@`` starts
# a citation only at the start of the bracket, after whitespace or ``;``,
# or after ``-`` (suppress-author) -- so ``me@example.org`` is not one.
# A leading letter used to be required, so ``[@1975Crank]`` was never read
# and could cite nothing without failing (audit_040_p4_1, C8).
_KEY_PUNCTUATION = ":.#$%&-+?<>~/"
_CITE_KEY = re.compile(
    r"(?:(?<=[\s;\[-])|^)@(?:\{([^{}]*)\}|([A-Za-z0-9_][A-Za-z0-9_"
    + re.escape(_KEY_PUNCTUATION) + r"]*))")
# A bracket holding an ``@``, which may wrap across lines but not across a
# paragraph break (a blank line) -- Pandoc reads ``[see\n@Key, p. 3]`` as
# one citation, and a line-by-line scan never saw it (audit_040_p4_1, C9).
_NOT_A_BRACKET_END = r"(?:[^\]\n]|\n(?![ \t]*\n))"
_CITE_BRACKET = re.compile(
    r"\[(" + _NOT_A_BRACKET_END + r"*?@" + _NOT_A_BRACKET_END + r"+)\]")
# ``@comment{...}`` / ``@comment(...)``: its contents are not entries.  An
# entry wrapped in one still counted as defined (audit_040_p4_1, C10).
_BIB_COMMENT = re.compile(r"@comment\s*([{(])", re.IGNORECASE)


def _blank_bib_comments(text: str) -> tuple[str, list[str]]:
    """``text`` with every comment blanked out, and any malformed one.

    Lines whose first non-whitespace character is ``%`` go, and so does the
    whole body of each ``@comment{...}`` (braces balanced) or
    ``@comment(...)``.  Blanked, not deleted: every character but a newline
    becomes a space, so line numbers still point at the source.  An
    unterminated ``@comment`` blanks to the end of the file and is
    reported, since it swallows every entry after it.
    """
    text = "".join(
        ("\n" if line.endswith("\n") else "") if line.lstrip().startswith("%")
        else line
        for line in text.splitlines(keepends=True))
    chars = list(text)
    problems = []
    pos = 0
    while True:
        match = _BIB_COMMENT.search(text, pos)
        if match is None:
            break
        opener = match.group(1)
        closer = "}" if opener == "{" else ")"
        depth, end = 0, len(text)
        for i in range(match.end() - 1, len(text)):
            if text[i] == opener:
                depth += 1
            elif text[i] == closer:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if depth:
            problems.append(
                f"unterminated @comment on line "
                f"{text.count(chr(10), 0, match.start()) + 1}: it hides every "
                f"entry after it")
        for i in range(match.start(), end):
            if chars[i] != "\n":
                chars[i] = " "
        pos = end
    return "".join(chars), problems


def parse_bib_entries(bib_path: str) -> list[tuple[int, str]]:
    """Return ``(line_number, key)`` for every entry BibTeX would define.

    Lines whose first non-whitespace character is ``%`` are comments and are
    dropped before matching, because BibTeX ignores them -- an entry left
    commented out is exactly how a live citation goes dangling.  So is an
    entry inside ``@comment{...}`` (:func:`_blank_bib_comments`).
    """
    with open(bib_path) as f:
        text, _problems = _blank_bib_comments(f.read())
    entries = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for match in _BIB_ENTRY.finditer(line):
            if match.group(1).lower() in _NON_ENTRY_TYPES:
                continue
            entries.append((lineno, match.group(2)))
    return entries


def bib_problems(bib_path: str) -> list[str]:
    """Malformed comments in the bibliography (see :func:`_blank_bib_comments`)."""
    with open(bib_path) as f:
        return _blank_bib_comments(f.read())[1]


def parse_bib_keys(bib_path: str) -> set[str]:
    """The set of keys the bibliography defines."""
    return {key for _lineno, key in parse_bib_entries(bib_path)}


def find_duplicate_keys(entries: list[tuple[int, str]]) -> list[str]:
    """Describe every key the bibliography defines more than once."""
    seen: dict[str, list[int]] = {}
    for lineno, key in entries:
        seen.setdefault(key, []).append(lineno)
    return [
        f"duplicate bibliography key '{key}' defined on lines "
        f"{', '.join(str(n) for n in linenos)}"
        for key, linenos in sorted(seen.items())
        if len(linenos) > 1
    ]


def extract_citations(md_path: str) -> list[tuple[int, str]]:
    """Extract all ``[@Key]`` citations from a Markdown file.

    Returns ``(line_number, key)`` pairs, the line being the key's own.
    Handles single citations ``[@Key]``, multiple citations ``[@Key1;
    @Key2]``, and a bracket that wraps onto the next line.
    """
    with open(md_path) as f:
        text = f.read()
    citations = []
    for bracket in _CITE_BRACKET.finditer(text):
        content = bracket.group(1)
        for key_match in _CITE_KEY.finditer(content):
            braced, bare = key_match.groups()
            # Trailing punctuation belongs to the prose, not the key.
            key = braced if braced is not None else bare.rstrip(_KEY_PUNCTUATION)
            if key:
                offset = bracket.start(1) + key_match.start()
                citations.append((text.count("\n", 0, offset) + 1, key))
    return citations


def scan_directory(scan_dir: str) -> list[tuple[str, int, str]]:
    """Recursively find all ``[@Key]`` citations in .md files under scan_dir.

    Returns ``(filepath, line_number, key)`` triples.
    """
    results = []
    for root, _dirs, files in os.walk(scan_dir):
        for fname in sorted(files):
            if not fname.endswith(".md") or fname.startswith("_"):
                continue
            fpath = os.path.join(root, fname)
            for lineno, key in extract_citations(fpath):
                results.append((fpath, lineno, key))
    return results


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    scan_dir = argv[0] if argv else os.path.join(_REPO_ROOT, DEFAULT_SCAN_DIR)
    bib_path = os.environ.get(
        "BIB_PATH", os.path.join(_REPO_ROOT, DEFAULT_BIB)
    )

    if not os.path.isfile(bib_path):
        print(f"Bibliography not found: {bib_path}", file=sys.stderr)
        return 1

    if not os.path.isdir(scan_dir):
        print(f"Documentation directory not found: {scan_dir}", file=sys.stderr)
        return 1

    entries = parse_bib_entries(bib_path)
    bib_keys = {key for _lineno, key in entries}

    errors = list(find_duplicate_keys(entries)) + bib_problems(bib_path)
    if not bib_keys:
        errors.append(f"no BibTeX entries found in {bib_path}")

    citations = scan_directory(scan_dir)
    if not citations:
        errors.append(
            f"no [@Key] citations found under {scan_dir}: a gate that "
            f"verifies nothing cannot fail, so check the scan scope"
        )

    # Check for dangling citations (cited but not in bib)
    cited_keys = set()
    verified = 0
    declined: list[str] = []
    for fpath, lineno, key in citations:
        try:
            relpath = os.path.relpath(fpath, _REPO_ROOT)
        except ValueError:  # pragma: no cover - different drive
            relpath = fpath
        if (relpath, key) in _TEMPLATE_CITATIONS:
            # Skipped before the existence check -- so it must not be
            # counted as verified afterwards.  It was: the gate reported
            # "50 citation(s) verified" having checked 45
            # (audit_040_r2/gates, finding G1).
            declined.append(f"{relpath}:{lineno}: [@{key}] is a syntax "
                            f"example and was NOT checked")
            continue
        cited_keys.add(key)
        verified += 1
        if key not in bib_keys:
            errors.append(f"{relpath}:{lineno}: [@{key}] not found in {bib_path}")

    # Report unused bib entries (warning, non-blocking)
    unused = bib_keys - cited_keys
    for key in sorted(unused):
        print(f"WARNING: {bib_path} entry '{key}' is not cited by any document")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(
            f"\n{len(errors)} citation problem(s) found "
            f"({len(cited_keys)} unique keys cited, {len(bib_keys)} bib entries)",
            file=sys.stderr,
        )
        return 1

    for note in declined:
        print(f"NOTE: {note}")

    if verified == 0:
        print(
            f"FAIL: all {len(citations)} citation(s) under {scan_dir} are "
            f"allowlisted syntax examples; none was checked.\n"
            "A gate that verifies nothing cannot fail; fix the scope or the "
            "allowlist rather than trusting the OK.",
            file=sys.stderr,
        )
        return 1

    # Verified and declined, separately.  One headline number that folds in
    # the references the gate declined to check is the whole defect.
    note = f", {len(declined)} not checked" if declined else ""
    print(
        f"OK: {verified} citation(s) verified{note} "
        f"({len(cited_keys)} unique keys, {len(bib_keys)} bib entries)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
