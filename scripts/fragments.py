#!/usr/bin/env python3
"""Changelog and registry fragments: write them, assemble them, relabel after a merge.

A pull request never edits a shared record file (``CHANGELOG.md``, the release
notes, ``docs/validation/known_anomalies.yaml``, ``_HIGHEST_ANOMALY_ID``, the
SOUP table, the development record's index).  It adds its own files instead:

    docs/fragments/anomalies/<stem>.yaml   new registry entries, each labelled
                                           MADD-ANO-NEW-<stem>-<n>
    docs/fragments/changes/<stem>.yaml     changelog bullets and release-notes text

``<stem>`` is the branch name, lower-cased, with ``/``, ``_`` and ``.`` turned
into ``-``.  The developer guide (``docs/developer_guide/
documentation_standards.md``, "Recording a change") has the whole procedure and
``docs/fragments/README.txt`` the short form.

    fragments.py new [--anomaly | --changes] [--branch NAME]
        write this branch's fragment from a template
    fragments.py assemble --registry
        number this line's anomalies (on the numbering line only), at a quiet point
    fragments.py assemble --release X.Y.Z --date YYYY-MM-DD
        before a release, on every line: the registry as above, plus the
        changelog section and the release notes
    fragments.py relabel
        rewrite every label the registry records to its number, in every
        tracked file; run after each forward merge between release lines
    fragments.py from-branch --base REF
        move the registry entries a branch opened before fragments existed
        into a fragment, with labels

Assembly is always its own pull request, made in a full clone.  It refuses to
start on a dirty tree, a shallow clone or a red ``scripts/check_fragments.py``,
and checks its own output (every entry in comes out exactly once, no assigned
label survives, every assembled entry records its label); on a failure it
restores every path it touched and exits 1.  This module also holds the
definitions the gates share: ``scripts/check_fragments.py``,
``scripts/check_anomalies.py`` (the merged view) and ``scripts/anom.py``.

Standard library only at import; PyYAML is imported where a fragment is
parsed.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import textwrap
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The open release lines, oldest first.  Identical on every line, so a
#: forward merge never conflicts on it.  The first one numbers anomalies; the
#: others keep their labels until it hands numbering over (a one-line commit
#: here, in 0.5.0's release-assembly pull request).
RELEASE_LINES = ("0.4", "0.5")

#: The first anomaly id the assembler can assign.  Every entry from here on
#: carries the ``# label:`` comment of the fragment it came from.
FRAGMENTS_FROM = 297

FRAGMENTS = "docs/fragments"
ANOMALIES = "docs/fragments/anomalies"
CHANGES = "docs/fragments/changes"
README = "docs/fragments/README.txt"
REGISTRY = "docs/validation/known_anomalies.yaml"
HW_FILE = "tests/compliance/test_soup_evidence.py"
CHANGELOG = "CHANGELOG.md"
NOTES_DIR = "docs/release_notes"
NOTES_INDEX = "docs/release_notes/index.md"
PYPROJECT = "pyproject.toml"
SOUP_GENERATOR = "scripts/generate_soup_tables.py"

STEM = r"[a-z0-9]+(?:-[a-z0-9]+)*"
STEM_RE = re.compile(STEM)

#: ``<n>`` of a label: 1, 2, ... with no leading zero, so that one number has
#: one spelling (``-01`` and ``-1`` would otherwise be two labels the
#: registry records as one).
N = r"[1-9][0-9]*"

#: The one spelling of a provisional label.
LABEL_RE = re.compile(
    rf"(?<![A-Za-z0-9_])MADD-ANO-NEW-({STEM})-({N})(?![A-Za-z0-9_-])")

#: Anything that sets out to be a label: ``MADD``, ``ANO`` and ``NEW`` in any
#: case, joined by ``-``, ``_``, a space or nothing, then a separator and a
#: letter or digit.  Each must be :data:`LABEL_RE` at the same place, or it is
#: a variant spelling.  ``MADD-ANO-NEW-<stem>-<n>`` (a placeholder) is not an
#: attempt, nor is a regular expression.
ATTEMPT_RE = re.compile(
    r"(?<![A-Za-z0-9_])MADD[-_ ]?ANO[-_ ]?NEW[-_ ](?=[A-Za-z0-9])", re.I)

#: The comment the assembler writes right after an assembled entry's id line.
LABEL_COMMENT_RE = re.compile(rf"^    # label: NEW-({STEM})-({N})[ \t]*$", re.M)
#: Any comment line that claims to record a label, whatever its layout.
ANY_LABEL_COMMENT_RE = re.compile(r"^[ \t]*#[ \t]*label[ \t]*:", re.M | re.I)
#: A canonical registry id line, as ``check_anomalies.py`` requires it.
NUMBERED_ID_LINE_RE = re.compile(r'^  - anomaly_id: "MADD-ANO-(\d+)"[ \t]*$', re.M)
#: A fragment entry's id line: the registry's layout, with the label as id.
FRAGMENT_ID_LINE_RE = re.compile(r'^  - anomaly_id: "([^"\\\s]+)"[ \t]*$', re.M)

#: Fragment-only keys of an anomaly entry; neither reaches the registry.
EXTRA_KEYS = ("release_note", "record_index")

#: The changelog's sections, in the order a release section lists them.
CHANGELOG_SECTIONS = ("Added", "Changed", "Deprecated", "Removed", "Fixed",
                      "Security", "Verification", "Known Anomalies")

#: The release notes' headings, in the order a notes file has them.
NOTES_HEADINGS = ("Summary", "What is new", "What changes for existing code",
                  "Security", "Fixed", "Known anomalies", "API stability",
                  "Upgrade checklist", "For contributors")

#: ``resolution_status`` values whose defect no user can reach; the same set
#: as ``check_anomalies.UNREACHABLE_STATUSES`` (a test holds them equal).
UNREACHABLE = frozenset({"resolved", "duplicate"})

ANOMALIES_KEYS = ("line", "anomalies")
CHANGES_KEYS = ("line", "changelog", "release_notes")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def stem_of(branch: str) -> str:
    """The fragment stem of a branch name, or ``ValueError``."""
    stem = re.sub(r"[/_.]", "-", branch.strip().lower())
    if not STEM_RE.fullmatch(stem):
        raise ValueError(
            f"branch {branch!r} gives the stem {stem!r}, which is not "
            f"{STEM} (letters, digits and single dashes); pass --branch")
    return stem


def label(stem: str, n: int) -> str:
    return f"MADD-ANO-NEW-{stem}-{n}"


def numbered(n: int) -> str:
    return f"MADD-ANO-{n:03d}"


def load_yaml(text: str):
    """``yaml.safe_load``: the loader every reader of these files uses.

    Never libyaml's: it reads some text the pure-Python loader refuses (a tab
    after a colon, a block scalar's header followed by a comment without a
    space), so a fragment the gates read with it could fail its assembly,
    whose SOUP generator and validator use ``yaml.safe_load``.
    """
    import yaml

    return yaml.safe_load(text)


def git(root, *args, check=True, binary=False):
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                          text=not binary)
    if check and done.returncode:
        err = done.stderr if not binary else done.stderr.decode(errors="replace")
        raise RuntimeError(f"git {' '.join(args)}: {err.strip()}")
    return done.stdout


def tree_version(root) -> str:
    """The tree's version, from ``pyproject.toml``."""
    data = tomllib.loads((Path(root) / PYPROJECT).read_text(encoding="utf-8"))
    return str(data["project"]["version"])


_VERSION_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?(.*)")


def version_parts(version: str):
    """``(major, minor, patch, suffix)`` of a version string."""
    m = _VERSION_RE.fullmatch(version.strip())
    if not m:
        raise ValueError(f"cannot read the version {version!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 0), m.group(4)


def line_of(version: str) -> str:
    major, minor, _, _ = version_parts(version)
    return f"{major}.{minor}"


def base_of(version: str) -> str:
    major, minor, patch, _ = version_parts(version)
    return f"{major}.{minor}.{patch}"


def is_dev(version: str) -> bool:
    """A development version: the release gate is off.  ``rc`` is not one."""
    return re.search(r"\.dev\d*$", version.strip()) is not None


def line_key(line: str):
    major, minor = line.split(".")
    return int(major), int(minor)


# ---------------------------------------------------------------------------
# Reading fragments and the registry
# ---------------------------------------------------------------------------

@dataclass
class Fragment:
    """One fragment file, parsed (``data`` is ``None`` when it does not)."""

    rel: str
    kind: str                    # "anomalies" or "changes"
    stem: str
    text: str
    data: object = None
    error: str | None = None

    @property
    def line(self):
        return self.data.get("line") if isinstance(self.data, dict) else None

    @property
    def entries(self) -> list:
        if not isinstance(self.data, dict):
            return []
        found = self.data.get("anomalies")
        return [a for a in found if isinstance(a, dict)] if isinstance(found, list) else []


def fragment_paths(root) -> list[Path]:
    """Every file under ``docs/fragments/``, README included."""
    base = Path(root) / FRAGMENTS
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*") if p.is_file())


def read_fragments(root, kinds=("anomalies", "changes")) -> list[Fragment]:
    """The fragment files of ``kinds`` named ``<stem>.yaml``, parsed."""
    found = []
    for kind in kinds:
        folder = Path(root) / FRAGMENTS / kind
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.yaml")):
            rel = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8")
            frag = Fragment(rel, kind, path.stem, text)
            try:
                frag.data = load_yaml(text)
            except Exception as exc:  # yaml.YAMLError and anything it wraps
                frag.error = f"{rel} does not parse as YAML: {exc}"
            found.append(frag)
    return found


def strip_extra(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if k not in EXTRA_KEYS}


def recorded_labels(registry_text: str) -> dict[str, list[str]]:
    """``{label: [ids it is recorded under]}`` from the registry's comments.

    A ``# label:`` comment belongs to the canonical id line right above it.
    """
    found: dict[str, list[str]] = {}
    current = None
    for line in registry_text.splitlines():
        m = NUMBERED_ID_LINE_RE.match(line)
        if m:
            current = numbered(int(m.group(1)))
            continue
        c = LABEL_COMMENT_RE.match(line)
        if c:
            found.setdefault(label(c.group(1), int(c.group(2))), []).append(
                current or "<no entry>")
    return found


def merged_statuses(root) -> dict[str, str]:
    """``{id or label: resolution_status}``: the registry and every fragment."""
    data = load_yaml((Path(root) / REGISTRY).read_text(encoding="utf-8")) or {}
    statuses = {str(a.get("anomaly_id")): str(a.get("resolution_status", ""))
                for a in data.get("anomalies") or [] if isinstance(a, dict)}
    for frag in read_fragments(root, ("anomalies",)):
        for a in frag.entries:
            statuses.setdefault(str(a.get("anomaly_id")),
                                str(a.get("resolution_status", "")))
    return statuses


def registry_highest(registry_text: str) -> int:
    ids = [int(m.group(1)) for m in NUMBERED_ID_LINE_RE.finditer(registry_text)]
    return max(ids, default=0)


_HW_RE = re.compile(r"^(_HIGHEST_ANOMALY_ID\s*=\s*)(\d+)", re.M)
_RETIRED_RE = re.compile(r"^_RETIRED_ANOMALY_IDS[^=]*=\s*\{(.*?)\}", re.M | re.S)


def high_water(root) -> int:
    m = _HW_RE.search((Path(root) / HW_FILE).read_text(encoding="utf-8"))
    if not m:
        raise RuntimeError(f"no _HIGHEST_ANOMALY_ID in {HW_FILE}")
    return int(m.group(2))


def retired_numbers(root) -> set[int]:
    m = _RETIRED_RE.search((Path(root) / HW_FILE).read_text(encoding="utf-8"))
    if not m:
        return set()
    return {int(x) for x in re.findall(r"MADD-ANO-0*(\d+)", m.group(1))}


def tracked_label_hits(root) -> list[tuple[str, int, str]]:
    """``(path, line, text)`` for every attempted label in a tracked file."""
    # -z: a path is followed by NUL, never quoted or split on a space or a
    # colon.
    out = git(root, "grep", "-n", "-z", "-I", "-i", "-E",
              r"MADD[-_ ]?ANO[-_ ]?NEW[-_ ][A-Za-z0-9]", check=False)
    hits = []
    for row in out.splitlines():
        parts = row.split("\0", 2)
        if len(parts) == 3 and parts[1].isdigit():
            hits.append((parts[0], int(parts[1]), parts[2]))
    return hits


def git_paths(root, *args) -> list[str]:
    """The paths a ``git`` command lists with ``-z``, split on NUL.

    A path with a space, a colon or a non-ASCII character comes back whole
    and unquoted, which a split on whitespace or lines does not give.
    """
    return [p for p in git(root, *args, check=False).split("\0") if p]


def files_naming(root, needle: str) -> list[str]:
    """Every tracked text file that holds ``needle``, as a fixed string."""
    return git_paths(root, "grep", "-l", "-z", "-I", "-F", needle)


def next_label_number(registry_text: str, stem: str) -> int:
    """The first ``<n>`` a fragment of ``stem`` may use: one past every
    label of that stem the registry records (a stem reused after its
    fragment was assembled starts after the old labels)."""
    used = [int(m.group(2)) for m in map(LABEL_RE.fullmatch, recorded_labels(registry_text))
            if m and m.group(1) == stem]
    return max(used, default=0) + 1


# ---------------------------------------------------------------------------
# Text surgery on a fragment and on the record files
# ---------------------------------------------------------------------------

def entry_blocks(text: str) -> list[tuple[str, list[str]]]:
    """``(id, lines)`` for each entry of an anomalies fragment, from its text.

    An entry runs from its ``  - anomaly_id:`` line to the next one, or to
    the next top-level line, or to the end; trailing blank lines are dropped.
    """
    lines = text.splitlines()
    starts = [i for i, ln in enumerate(lines) if FRAGMENT_ID_LINE_RE.match(ln)]
    blocks = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(lines)
        for j in range(start + 1, end):
            if lines[j] and not lines[j][0].isspace() and not lines[j].startswith("#"):
                end = j
                break
        block = lines[start:end]
        while block and not block[-1].strip():
            block.pop()
        blocks.append((FRAGMENT_ID_LINE_RE.match(lines[start]).group(1), block))
    return blocks


def drop_extra_keys(block: list[str]) -> list[str]:
    """``block`` without its ``release_note`` / ``record_index`` keys."""
    out, skipping = [], False
    key = re.compile(rf"^    ({'|'.join(EXTRA_KEYS)})\s*:")
    for line in block:
        if key.match(line):
            skipping = True
            continue
        if skipping:
            if not line.strip() or (len(line) - len(line.lstrip(" ")) > 4):
                continue
            skipping = False
        out.append(line)
    while out and not out[-1].strip():
        out.pop()
    return out


def rewrite_labels(text: str, mapping: dict[str, str]) -> tuple[str, int]:
    """Every label in ``mapping`` replaced, as an exact token, by its id."""
    count = 0

    def sub(m):
        nonlocal count
        new = mapping.get(m.group(0))
        if new is None:
            return m.group(0)
        count += 1
        return new

    return LABEL_RE.sub(sub, text), count


def section_span(text: str, heading: str):
    """``(start, end)`` of the body of a ``## heading`` section, or ``None``."""
    m = re.search(rf"^## {re.escape(heading)}[ \t]*\n", text, re.M)
    if not m:
        return None
    nxt = re.search(r"^## ", text[m.end():], re.M)
    return m.end(), (m.end() + nxt.start()) if nxt else len(text)


def append_to_section(text: str, heading: str, block: str,
                      order=NOTES_HEADINGS) -> str:
    """``block`` appended to the ``## heading`` section, which is created in
    ``order``'s place if the file lacks it."""
    block = block.strip("\n")
    span = section_span(text, heading)
    if span is not None:
        start, end = span
        body = text[start:end].rstrip("\n")
        sep = "\n\n" if body.strip() else "\n"
        rest = text[end:]
        return text[:start] + body + sep + block + "\n" + ("\n" + rest if rest else "")
    later = [h for h in order[order.index(heading) + 1:] if section_span(text, h)] \
        if heading in order else []
    new = f"## {heading}\n\n{block}\n"
    if later:
        at = re.search(rf"^## {re.escape(later[0])}[ \t]*\n", text, re.M).start()
        return text[:at] + new + "\n" + text[at:]
    return text.rstrip("\n") + "\n\n" + new


def notes_template(version: str) -> str:
    return (f"---\norphan: false\n---\n\n# What's new in v{version}\n\n"
            f"```{{versionadded}} v{version}\n```\n")


def add_to_toctree(index_text: str, version: str) -> str:
    """``index.md`` with ``v<version>`` in its toctree, newest first."""
    entry = f"v{version}"
    m = re.search(r"(```\{toctree\}\n(?::[^\n]*\n)*\n)(.*?)(```)", index_text, re.S)
    if not m:
        raise RuntimeError(f"{NOTES_INDEX} has no toctree")
    names = [x.strip() for x in m.group(2).splitlines() if x.strip()]
    if entry in names:
        return index_text

    def key(name):
        try:
            return version_parts(name.lstrip("v"))[:3]
        except ValueError:
            return (-1, -1, -1)

    names.append(entry)
    names.sort(key=key, reverse=True)
    return index_text[:m.start(2)] + "".join(n + "\n" for n in names) + index_text[m.end(2):]


# ---------------------------------------------------------------------------
# new
# ---------------------------------------------------------------------------

_ANOMALIES_TEMPLATE = """\
# New registry entries of this branch.  The layout is the registry's own
# (docs/validation/known_anomalies.yaml), with two differences:
#   - anomaly_id is this file's label: MADD-ANO-NEW-<this file's name>-<n>,
#     n = 1, 2, ...  scripts/fragments.py assemble gives each its number later;
#     until then, name the entry by its label in tests and other files;
#   - release_note: the release notes' Known-anomalies text (what a user sees
#     and what to do), required unless the entry is resolved or duplicate;
#     record_index: optional, the development record's index line.
# Check it with scripts/check_anomalies.py and scripts/check_fragments.py.
line: "{line}"
anomalies:
  - anomaly_id: "{label}"
    title: ""
    description: >
      ...
    severity: "minor"
    safety_relevance: "not_safety_relevant"
    safety_relevance_rationale: >
      ...
    affected_components:
      - "maddening.some.module.Symbol"
    affected_versions: ">={first}"
    workaround: >
      ...
    resolution_status: "open"
    verification:
      - "tests/path/test_file.py::test_name"
    github_issue: null
    release_note: >
      ...
"""

_CHANGES_TEMPLATE = """\
# This branch's changelog bullets and release-notes text.  Never edit
# CHANGELOG.md or docs/release_notes/ in a pull request: the release assembly
# (scripts/fragments.py assemble --release) writes them from this file.
#   changelog: section -> bullets ({sections})
#   release_notes: heading -> Markdown ({headings})
# Delete what you do not use; keep at least one of the two.
line: "{line}"
changelog:
  Fixed:
    - "..."
release_notes:
  Fixed: |
    - ...
"""


def cmd_new(args, root: Path) -> int:
    branch = args.branch or git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    try:
        stem = stem_of(branch)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    kinds = (["anomalies"] if args.anomaly else []) + (["changes"] if args.changes else [])
    kinds = kinds or ["changes"]
    version = tree_version(root)
    written = []
    for kind in kinds:
        path = root / FRAGMENTS / kind / f"{stem}.yaml"
        if path.exists():
            print(f"{path.relative_to(root)} exists: a branch of this name already "
                  f"has a fragment on this tree.  Use another branch name, or add to "
                  f"that file if it is yours.", file=sys.stderr)
            return 2
        if kind == "anomalies":
            text = (root / REGISTRY).read_text(encoding="utf-8")
            body = _ANOMALIES_TEMPLATE.format(
                line=line_of(version), label=label(stem, next_label_number(text, stem)),
                first=f"{base_of(version)}.dev0")
        else:
            body = _CHANGES_TEMPLATE.format(
                line=line_of(version), sections=", ".join(CHANGELOG_SECTIONS),
                headings=", ".join(NOTES_HEADINGS))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        written.append(path.relative_to(root).as_posix())
    print("wrote " + ", ".join(written))
    return 0


# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------

class AssemblyError(RuntimeError):
    pass


def merge_order(root, rels: list[str]) -> dict[str, int]:
    """Where each fragment entered this line's first-parent history."""
    # --no-renames: a fragment renamed after it merged is an addition under
    # its new name, not hidden as a rename (and the gate's history ratchet
    # refuses the old name's disappearance).
    out = git(root, "log", "--first-parent", "--reverse", "--diff-filter=A",
              "--no-renames", "--name-only", "--format=%x00%H", "--", FRAGMENTS)
    order: dict[str, int] = {}
    for n, chunk in enumerate(out.split("\x00")):
        for path in chunk.splitlines()[1:]:
            if path.strip():
                order[path.strip()] = n           # the last addition wins
    missing = [r for r in rels if r not in order]
    if missing:
        raise AssemblyError(f"not in this line's first-parent history: {missing}; "
                            f"commit them first")
    return order


def _labels_named(text: str) -> set[str]:
    return {m.group(0) for m in LABEL_RE.finditer(text)}


def _wrap(text: str, **kw) -> str:
    return textwrap.fill(" ".join(str(text).split()), 79, **kw)


def _index_line(n: int, entry: dict) -> tuple[str, bool]:
    """The record's index text for an assembled entry, and whether it is reachable."""
    status = str(entry.get("resolution_status") or "open")
    title = " ".join(str(entry.get("title", "")).split()).rstrip(".")
    note = entry.get("record_index")
    reachable = status not in UNREACHABLE
    if note:
        return " ".join(str(note).split()), reachable
    if reachable:
        extra = entry.get("release_note")
        return _wrap(f"`{numbered(n)}` — *{title}* — **{status}**."
                     + (f"  {extra}" if extra else "")), True
    return _wrap(f"`{n:03d}` ({status}): {title}."), False


def _write_index(record: str, rel: str, lines: list[tuple[str, bool]]) -> str:
    span = section_span(record, "Index of the anomaly registry")
    marker = "**Resolved.**"
    if span is None or marker not in record[span[0]:span[1]]:
        raise AssemblyError(f"{rel}: no '## Index of the anomaly registry' section "
                            f"with a {marker} paragraph")
    start, end = span
    body = record[start:end]
    at = start + body.index(marker)
    tail = start + len(body.rstrip("\n"))
    live = "".join(text + "\n\n" for text, reach in lines if reach)
    closed = "".join("\n" + text for text, reach in lines if not reach)
    return record[:at] + live + record[at:tail] + closed + record[tail:]


def canonical_bullet(text) -> str:
    """A bullet's text as a release section carries it: verbatim, but for
    the whitespace around it and the spaces on a line holding nothing else."""
    return "\n".join(ln if ln.strip() else "" for ln in str(text).strip().split("\n"))


def render_bullet(text) -> str:
    """A changelog bullet as a release section holds it.

    ``- `` before the first line, and two spaces before every later line that
    is not blank, which keeps a multi-line bullet, a nested list and a code
    fence inside its list item.  The text is otherwise carried verbatim
    (:func:`canonical_bullet`): a leading ``- `` of its own, inner spacing and
    code stay as written.
    """
    lines = canonical_bullet(text).split("\n")
    return "\n".join(["- " + lines[0]] + [("  " + ln) if ln else "" for ln in lines[1:]])


def unrender_bullet(block: str) -> str:
    """The text :func:`render_bullet` wrote ``block`` from, as
    :func:`canonical_bullet` gives it.  The assembler's own check that no
    bullet lost anything on the way."""
    lines = block.split("\n")
    first = lines[0][2:] if lines[0].startswith("- ") else lines[0]
    return "\n".join([first] + [ln[2:] if ln.startswith("  ") else ln for ln in lines[1:]])


def _section_parts(section: str):
    """``(heading line, preamble, [(sub-heading, body)])`` of a release section."""
    heading, _, rest = section.partition("\n")
    parts = re.split(r"^### (.+?)[ \t]*$", rest, flags=re.M)
    return heading, parts[0], [(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]


def _emit_section(heading: str, preamble: str, subs) -> str:
    out = heading + "\n"
    if preamble.strip():
        out += "\n" + preamble.strip("\n") + "\n"
    for name, body in subs:
        out += f"\n### {name}\n\n" + body.strip("\n") + "\n"
    return out + "\n"


def release_section(existing: str | None, version: str, date: str,
                    bullets: dict[str, list[str]], count: str) -> str:
    """The ``## [version] - date`` section, with ``bullets`` added to
    ``existing`` (the section an earlier release assembly wrote) or to an
    empty one.

    Bullets go in the changelog's section order, each after the ones already
    there; the count of reachable entries replaces the earlier count.  A
    section with nothing new comes back unchanged, byte for byte.
    """
    heading = f"## [{version}] - {date}"
    preamble, subs = "", []
    if existing is not None:
        old_heading, preamble, subs = _section_parts(existing)
    bodies = {name: body.strip("\n") for name, body in subs}
    order = [name for name, _ in subs]
    count_re = re.compile(rf"^- \d+ registry entries are reachable in "
                          rf"{re.escape(version)}: .*$", re.M)
    known = bodies.get("Known Anomalies", "")
    m = count_re.search(known)
    counted = render_bullet(count)
    if existing is not None and old_heading == heading and m and m.group(0) == counted \
            and not any(bullets.get(sec) for sec in CHANGELOG_SECTIONS):
        return existing
    if m:
        known = known[:m.start()] + counted + known[m.end():]
    else:
        known = counted + ("\n" + known if known else "")
    bodies["Known Anomalies"] = known
    for sec in CHANGELOG_SECTIONS:
        items = [render_bullet(item) for item in bullets.get(sec, [])]
        if items:
            bodies[sec] = "\n".join(([bodies[sec]] if bodies.get(sec) else []) + items)
    names = ([sec for sec in CHANGELOG_SECTIONS if bodies.get(sec)]
             + [name for name in order if name not in CHANGELOG_SECTIONS])
    return _emit_section(heading, preamble, [(name, bodies[name]) for name in names])


def version_section_span(changelog: str, version: str):
    """``[(start, end)]`` of each ``## [version]`` section of a changelog."""
    spans = []
    for m in re.finditer(rf"^## \[{re.escape(version)}\][^\n]*(?:\n|\Z)", changelog, re.M):
        nxt = re.search(r"^## ", changelog[m.end():], re.M)
        spans.append((m.start(), m.end() + nxt.start() if nxt else len(changelog)))
    return spans


def reachable_count_line(registry: dict, version: str) -> str:
    """The changelog's count of reachable entries, as the test reads it."""
    entries = [a for a in registry.get("anomalies") or [] if isinstance(a, dict)
               and a.get("resolution_status") not in UNREACHABLE]
    n_open = sum(a.get("resolution_status") == "open" for a in entries)
    n_part = sum(a.get("resolution_status") == "partially_resolved" for a in entries)
    other = len(entries) - n_open - n_part
    return (f"{len(entries)} registry entries are reachable in {version}: "
            f"{n_open} open and {n_part} partly resolved"
            + (f", and {other} with another status" if other else "")
            + ".  The release notes' \"Known anomalies\" section lists each; the "
            "registry is `docs/validation/known_anomalies.yaml`.")


class _Snapshot:
    """Every path the assembler touches, as it was, for a restore."""

    def __init__(self, root: Path):
        self.root, self.saved = root, {}

    def keep(self, rel: str):
        if rel not in self.saved:
            path = self.root / rel
            self.saved[rel] = path.read_bytes() if path.exists() else None

    def write(self, rel: str, text: str):
        self.keep(rel)
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def delete(self, rel: str):
        self.keep(rel)
        (self.root / rel).unlink()

    def restore(self):
        for rel, data in self.saved.items():
            path = self.root / rel
            if data is None:
                if path.exists():
                    path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)


def _preflight(root: Path, release: str | None) -> tuple[str, str]:
    version = tree_version(root)
    line = line_of(version)
    if git(root, "rev-parse", "--is-shallow-repository").strip() != "false":
        raise AssemblyError("this is a shallow clone: merge order and the ratchet "
                            "need the whole history (git fetch --unshallow)")
    dirty = git(root, "status", "--porcelain", "--untracked-files=no").strip()
    if dirty:
        raise AssemblyError(f"the tree has uncommitted changes:\n{dirty}")
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "--",
                    FRAGMENTS).strip()
    if untracked:
        raise AssemblyError(f"untracked fragments (commit or remove them):\n{untracked}")
    if line not in RELEASE_LINES:
        raise AssemblyError(f"this tree is on line {line}, which is not in "
                            f"RELEASE_LINES {RELEASE_LINES}")
    if release is not None and base_of(version) != release:
        raise AssemblyError(f"--release {release} is not this tree's release "
                            f"({version} builds towards {base_of(version)})")
    if release is not None and git(root, "rev-parse", "-q", "--verify",
                                   f"refs/tags/v{release}^{{commit}}", check=False).strip():
        raise AssemblyError(f"v{release} is tagged: {release} is out, and a fragment that "
                            f"arrives now belongs to the next version.  Bump the tree to "
                            f"the next .dev0 version; the fragment waits for that "
                            f"release's assembly")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_fragments                                   # noqa: PLC0415

    # The release gate is left out: at a release version it is red until this
    # assembly has run.  _verify holds it on the result.
    problems = check_fragments.findings(root, release_gate=False)
    if problems:
        raise AssemblyError("check_fragments.py is red; fix it first:\n  "
                            + "\n  ".join(problems))
    return version, line


def assemble(root: Path, *, release: str | None = None, date: str | None = None,
             soup=True, out=print) -> dict[str, str]:
    """One assembly; returns ``{label: id}``.  Raises :class:`AssemblyError`."""
    version, line = _preflight(root, release)
    numbering = line == RELEASE_LINES[0]
    if release is None and not numbering:
        raise AssemblyError(f"--registry runs on the numbering line only "
                            f"({RELEASE_LINES[0]}); this tree is on {line}")
    frags = [f for f in read_fragments(root) if f.line == line]
    if release is None:
        frags = [f for f in frags if f.kind == "anomalies"]
    order = merge_order(root, [f.rel for f in frags])
    frags.sort(key=lambda f: (order[f.rel], f.rel))

    reg_text = (root / REGISTRY).read_text(encoding="utf-8")
    numbered_ids = {int(m.group(1)) for m in NUMBERED_ID_LINE_RE.finditer(reg_text)}
    hw = high_water(root)
    highest = max(numbered_ids | retired_numbers(root), default=0)
    if highest != hw:
        raise AssemblyError(f"_HIGHEST_ANOMALY_ID is {hw} but the registry's highest "
                            f"id (retired ones included) is {highest}")

    # Deferral: a fragment naming a label neither assigned in this run nor
    # already numbered waits, and so does anything naming one of its labels.
    defined = {str(a.get("anomaly_id")): f for f in frags for a in f.entries}
    deferred: dict[str, str] = {}
    if not numbering:
        for f in frags:
            if f.kind == "anomalies":
                deferred[f.rel] = "anomalies are numbered on the numbering line only"
    changed = True
    while changed:
        changed = False
        live = {lab for lab, f in defined.items() if f.rel not in deferred}
        for f in frags:
            if f.rel in deferred:
                continue
            missing = sorted(_labels_named(f.text) - live)
            if missing:
                deferred[f.rel] = f"names {missing}, not assigned in this run"
                changed = True
    for rel, why in deferred.items():
        out(f"deferred: {rel}: {why}")
    take = [f for f in frags if f.rel not in deferred]

    mapping: dict[str, str] = {}
    appended: list[tuple[int, dict, list[str], str]] = []
    n = hw
    for f in take:
        if f.kind != "anomalies":
            continue
        blocks = dict(entry_blocks(f.text))
        for entry in f.entries:
            lab = str(entry["anomaly_id"])
            n += 1
            mapping[lab] = numbered(n)
            appended.append((n, entry, blocks[lab], f.rel))
    k = len(appended)

    snap = _Snapshot(root)
    try:
        _apply(root, snap, version, line, release, date, take, mapping, appended,
               reg_text, hw, k, soup, out)
    except Exception:
        snap.restore()
        raise
    return mapping


def _apply(root, snap, version, line, release, date, take, mapping, appended,
           reg_text, hw, k, soup, out):
    rel_frags = [f.rel for f in take]
    # 3. the registry: append as text, record the label, move the mark by k
    if appended:
        pieces = []
        for n, entry, block, _ in appended:
            body = drop_extra_keys(block)
            lab = LABEL_RE.fullmatch(str(entry["anomaly_id"]))
            pieces.append("\n".join([f'  - anomaly_id: "{numbered(n)}"',
                                     f"    # label: NEW-{lab.group(1)}-{lab.group(2)}"]
                                    + body[1:]))
        snap.write(REGISTRY, reg_text.rstrip("\n") + "\n\n" + "\n\n".join(pieces) + "\n")
        hw_text = (root / HW_FILE).read_text(encoding="utf-8")
        snap.write(HW_FILE, _HW_RE.sub(lambda m: m.group(1) + str(hw + k), hw_text, count=1))

    # 4. the record's index, and the notes' Known anomalies
    base = base_of(version)
    notes_rel = f"{NOTES_DIR}/v{base}.md"
    if appended:
        record_rel = f"docs/validation/development_record_v{line}.0.md"
        if not (root / record_rel).exists():
            raise AssemblyError(f"{record_rel} does not exist: the line's record holds "
                                f"the registry's index; create it with that section")
        record = (root / record_rel).read_text(encoding="utf-8")
        lines = [_index_line(n, e) for n, e, _, _ in appended]
        snap.write(record_rel, _write_index(record, record_rel, lines))
        notes = [f"- `{numbered(n)}`: " + " ".join(str(e['release_note']).split())
                 for n, e, _, _ in appended if e.get("release_note")]
        if notes:
            _notes_append(root, snap, base, "Known anomalies",
                          "\n".join(_wrap(x, subsequent_indent="  ") for x in notes))

    # 5. release only: the changelog section and the release notes.  A
    #    second run (a fragment merged after the first, before the tag) adds
    #    to the section the first wrote; with nothing new it changes nothing.
    if release is not None:
        bullets: dict[str, list[str]] = {}
        blocks: list[tuple[str, str]] = []
        for f in take:
            if f.kind != "changes":
                continue
            for sec, items in (f.data.get("changelog") or {}).items():
                bullets.setdefault(sec, []).extend(items or [])
            for heading, block in (f.data.get("release_notes") or {}).items():
                blocks.append((heading, str(block)))
        reg_now = load_yaml((root / REGISTRY).read_text(encoding="utf-8"))
        # the count is of the registry after this run's numbering
        changelog = (root / CHANGELOG).read_text(encoding="utf-8")
        m = re.search(r"^## \[Unreleased\][ \t]*\n", changelog, re.M)
        nxt = re.search(r"^## \[", changelog[m.end():], re.M) if m else None
        if not m or not nxt:
            raise AssemblyError(f"{CHANGELOG}: no '## [Unreleased]' followed by a release heading")
        if changelog[m.end():m.end() + nxt.start()].strip():
            raise AssemblyError(f"{CHANGELOG}: '## [Unreleased]' is not empty")
        at = m.end() + nxt.start()
        count = reachable_count_line(reg_now, release)
        spans = version_section_span(changelog, release)
        if len(spans) > 1:
            raise AssemblyError(f"{CHANGELOG} has {len(spans)} '## [{release}]' sections")
        if spans and spans[0][0] != at:
            raise AssemblyError(f"{CHANGELOG}: '## [{release}]' is not the newest section")
        if spans:
            s0, e0 = spans[0]
            changelog = changelog[:s0] + release_section(
                changelog[s0:e0], release, date, bullets, count) + changelog[e0:]
        else:
            changelog = changelog[:at] + release_section(
                None, release, date, bullets, count) + changelog[at:]
        snap.write(CHANGELOG, changelog)
        _notes_append(root, snap, release, None, None)          # the file exists
        for heading, block in blocks:
            _notes_append(root, snap, release, heading, block)
        if version_parts(release)[2] == 0:
            _name_every_reachable_entry(root, snap, release, reg_now)

    # 6. rewrite every assigned label in every tracked text file, and in the
    #    files this run created (a new notes file is not tracked yet)
    if mapping:
        hits = files_naming(root, "MADD-ANO-NEW-")
        created = [rel for rel, data in snap.saved.items() if data is None]
        for rel in sorted(set(hits) | set(created)):
            path = root / rel
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            new, count = rewrite_labels(text, mapping)
            if count:
                snap.write(rel, new)

    # 7. delete what was assembled; regenerate the SOUP table
    for rel in rel_frags:
        snap.delete(rel)
    if soup and mapping:
        snap.keep("docs/validation/soup_package.md")
        env = dict(os.environ, JAX_PLATFORMS="cpu")
        done = subprocess.run([sys.executable, SOUP_GENERATOR], cwd=root,
                              capture_output=True, text=True, env=env)
        if done.returncode:
            raise AssemblyError(f"{SOUP_GENERATOR} failed:\n{done.stdout}{done.stderr}")

    # 8. verify the invariants on the files as written
    _verify(root, take, mapping, appended, hw, k, release, out, snap)


def _notes_append(root, snap, version, heading, block):
    rel = f"{NOTES_DIR}/v{version}.md"
    path = root / rel
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = notes_template(version)
        index = (root / NOTES_INDEX).read_text(encoding="utf-8")
        snap.write(NOTES_INDEX, add_to_toctree(index, version))
    if heading is not None:
        if heading not in NOTES_HEADINGS:
            raise AssemblyError(f"unknown release-notes heading {heading!r}")
        text = append_to_section(text, heading, block)
    snap.write(rel, text)


_STATUS_WORDS = {"partially_resolved": "partly resolved"}


def _name_every_reachable_entry(root, snap, release, registry):
    """A ``.0`` release's notes name every reachable entry, as 0.4.0's did.

    The notes of a line are read from its ``.0`` file on (the release-notes
    test), so the ``.0`` notes must list every entry a user can reach in it,
    not only those this assembly numbered.  Each entry the Known anomalies
    section does not name yet gets a line: its id, its status unless open,
    and its title.  The registry has the full description.
    """
    rel = f"{NOTES_DIR}/v{release}.md"
    text = (root / rel).read_text(encoding="utf-8")
    span = section_span(text, "Known anomalies")
    named = set(re.findall(r"MADD-ANO-\d{3,}", text[span[0]:span[1]])) if span else set()
    lines = []
    for a in registry.get("anomalies") or []:
        if not isinstance(a, dict) or a.get("resolution_status") in UNREACHABLE:
            continue
        aid = str(a.get("anomaly_id"))
        if aid in named:
            continue
        status = str(a.get("resolution_status") or "open")
        shown = "" if status == "open" else \
            f" ({_STATUS_WORDS.get(status, status.replace('_', ' '))})"
        title = " ".join(str(a.get("title", "")).split()).rstrip(".")
        lines.append(_wrap(f"- `{aid}`{shown}: {title}.", subsequent_indent="  "))
    if not lines:
        return
    intro = ("" if span else
             f"These are the registry entries a user can reach in {release}. The "
             f"registry, `docs/validation/known_anomalies.yaml`, has the full description "
             f"of each, and the development record indexes every entry.\n\n")
    snap.write(rel, append_to_section(text, "Known anomalies", intro + "\n".join(lines)))


def _verify(root, take, mapping, appended, hw, k, release, out, snap=None):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_fragments                                    # noqa: PLC0415

    problems = []
    reg_text = (root / REGISTRY).read_text(encoding="utf-8")
    registry = load_yaml(reg_text)
    entries = registry["anomalies"]
    got = entries[len(entries) - k:] if k else []
    want = []
    for _, entry, _, _ in appended:
        clean = strip_extra(entry)
        rewritten = load_yaml(rewrite_labels(_dump(clean), mapping)[0])
        want.append(rewritten)
    if got != want:
        problems.append("the appended entries are not the fragment entries "
                        "(extra keys dropped, labels rewritten)")
    if high_water(root) != hw + k:
        problems.append(f"_HIGHEST_ANOMALY_ID is {high_water(root)}, not {hw + k}")
    ids = [str(a.get("anomaly_id")) for a in entries]
    if sorted(set(ids)) != sorted(ids):
        problems.append("the registry holds an id twice")
    recorded = recorded_labels(reg_text)
    for lab, aid in mapping.items():
        if recorded.get(lab) != [aid]:
            problems.append(f"{lab} is recorded as {recorded.get(lab)}, not [{aid}]")
    survivors = [h for h in tracked_label_hits(root)
                 if any(lab in h[2] for lab in mapping)]
    survivors += [(rel, 0, "") for rel in _untracked_with(root, mapping)]
    for path, line_no, _ in survivors:
        problems.append(f"an assigned label survives in {path}:{line_no}")
    for f in take:
        if (root / f.rel).exists():
            problems.append(f"{f.rel} was assembled and still exists")
    if release is not None:
        problems += _verify_release_text(root, take, mapping, release, snap)
    hist = check_fragments.history(root)
    problems += check_fragments.label_record_findings(
        reg_text, None if hist.error else hist.held)
    version = tree_version(root)
    if not is_dev(version):
        # Invariant 3 at a release version: no label at all, nor any fragment.
        problems += [f"release gate: {p}" for p in check_fragments.release_findings(
            root, version, set(), leftover_why=(
                f"this release assembly deferred it, and at release version {version} "
                f"a fragment never assembled blocks the release"))]
    if problems:
        raise AssemblyError("the assembly's own checks failed:\n  " + "\n  ".join(problems))


def _before(root, snap, rel) -> str:
    """``rel``'s text before this run touched it ('' if it did not exist)."""
    if snap is not None and rel in snap.saved:
        data = snap.saved[rel]
        return "" if data is None else data.decode("utf-8")
    path = Path(root) / rel
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _verify_release_text(root, take, mapping, release, snap) -> list[str]:
    """Invariant 5.1 for the release text: each changelog bullet and notes
    block of the assembled fragments is added exactly as written, as many
    times as the fragments give it, to the one ``## [release]`` section and
    to ``v<release>.md``; a ``.0`` notes file names every reachable entry."""
    problems = []
    changelog = (root / CHANGELOG).read_text(encoding="utf-8")
    spans = version_section_span(changelog, release)
    if len(spans) != 1:
        problems.append(f"{CHANGELOG} has {len(spans)} '## [{release}]' sections, not one")
    section = changelog[spans[0][0]:spans[0][1]] if spans else ""
    before = _before(root, snap, CHANGELOG)
    old = version_section_span(before, release)
    old_section = before[old[0][0]:old[0][1]] if old else ""
    wanted: dict[str, int] = {}
    blocks: dict[str, int] = {}
    for f in take:
        if f.kind != "changes":
            continue
        for items in (f.data.get("changelog") or {}).values():
            for item in items or []:
                text = rewrite_labels(str(item), mapping)[0]
                bullet = render_bullet(text)
                if unrender_bullet(bullet) != canonical_bullet(text):
                    problems.append(f"a changelog bullet from {f.rel} does not survive its "
                                    f"rendering: {' '.join(text.split())[:60]}")
                wanted[bullet] = wanted.get(bullet, 0) + 1
        for block in (f.data.get("release_notes") or {}).values():
            text = rewrite_labels(str(block), mapping)[0].strip()
            blocks[text] = blocks.get(text, 0) + 1
    for bullet, times in wanted.items():
        added = section.count(bullet) - old_section.count(bullet)
        if added != times:
            problems.append(f"a changelog bullet was added {added} time(s) to the "
                            f"'## [{release}]' section, not {times}: {bullet[:60]}")
    notes_rel = f"{NOTES_DIR}/v{release}.md"
    notes = (root / notes_rel).read_text(encoding="utf-8")
    old_notes = _before(root, snap, notes_rel)
    for text, times in blocks.items():
        if notes.count(text) - old_notes.count(text) < times:
            problems.append(f"a release-notes block is missing from {notes_rel}: "
                            f"{' '.join(text.split())[:60]}")
    if version_parts(release)[2] == 0:
        span = section_span(notes, "Known anomalies")
        named = set(re.findall(r"MADD-ANO-\d{3,}", notes[span[0]:span[1]])) if span else set()
        registry = load_yaml((root / REGISTRY).read_text(encoding="utf-8"))
        for a in registry.get("anomalies") or []:
            if (isinstance(a, dict) and a.get("resolution_status") not in UNREACHABLE
                    and str(a.get("anomaly_id")) not in named):
                problems.append(f"{notes_rel} does not name {a.get('anomaly_id')}, which "
                                f"is reachable in {release}")
    return problems


def _untracked_with(root, mapping):
    """New files the assembly wrote (a notes file) that still hold a label."""
    found = []
    for rel in git_paths(root, "ls-files", "-z", "--others", "--exclude-standard"):
        path = Path(root) / rel
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(lab in text for lab in mapping):
            found.append(rel)
    return found


def _dump(data) -> str:
    import yaml

    dumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
    return yaml.dump(data, Dumper=dumper, sort_keys=False, allow_unicode=True,
                     width=10 ** 6)


def cmd_assemble(args, root: Path) -> int:
    if (args.release is None) == (not args.registry):
        print("give --registry, or --release X.Y.Z --date YYYY-MM-DD", file=sys.stderr)
        return 2
    if args.release and not (args.date and re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.date)):
        print("--release needs --date YYYY-MM-DD", file=sys.stderr)
        return 2
    try:
        mapping = assemble(root, release=args.release, date=args.date,
                           soup=not args.no_soup)
    except AssemblyError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    if mapping:
        print("\n| label | id |\n|---|---|")
        for lab, aid in mapping.items():
            print(f"| `{lab}` | `{aid}` |")
    else:
        print("nothing to number")
    print("\nnext: run the gates, then commit the result as its own pull request "
          "(the table above goes in its body)")
    return 0


# ---------------------------------------------------------------------------
# relabel
# ---------------------------------------------------------------------------

def relabel(root: Path) -> list[tuple[str, int]]:
    """Every recorded label rewritten to its id in every tracked file."""
    reg_text = (root / REGISTRY).read_text(encoding="utf-8")
    recorded = recorded_labels(reg_text)
    mapping = {lab: ids[0] for lab, ids in recorded.items() if len(ids) == 1}
    changed = []
    if not mapping:
        return changed
    for rel in sorted(set(files_naming(root, "MADD-ANO-NEW-"))):
        path = root / rel
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if rel.startswith(ANOMALIES + "/"):
            # A fragment's own id lines are left alone: a fragment whose label is
            # recorded was resurrected by a merge, and check_fragments.py says so.
            parts = [ln if FRAGMENT_ID_LINE_RE.match(ln) else rewrite_labels(ln, mapping)[0]
                     for ln in text.split("\n")]
            new = "\n".join(parts)
            count = int(new != text)
        else:
            new, count = rewrite_labels(text, mapping)
        if count:
            path.write_text(new, encoding="utf-8")
            changed.append((rel, count))
    return changed


def cmd_relabel(args, root: Path) -> int:
    changed = relabel(root)
    for rel, count in changed:
        print(f"rewrote {count} label(s) in {rel}")
    if not changed:
        print("nothing to relabel")
    return 0


# ---------------------------------------------------------------------------
# from-branch
# ---------------------------------------------------------------------------

def _registry_blocks(text: str) -> list[tuple[int, int, int]]:
    """``(id, start line, end line)`` of each canonical entry, ends exclusive."""
    lines = text.splitlines()
    starts = [(i, int(m.group(1))) for i, ln in enumerate(lines)
              if (m := NUMBERED_ID_LINE_RE.match(ln))]
    out = []
    for k, (i, n) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        out.append((n, i, end))
    return out


def from_branch(root: Path, base: str, branch: str | None = None) -> dict[str, str]:
    """Move this branch's own registry entries into a fragment; ``{old id: label}``.

    The branch's own entries are those whose id the registry at the merge
    base of the branch and ``base`` does not hold: an entry the branch only
    edited keeps its number.  Run it before the branch merges ``base`` (as
    ``renumber`` was): after that merge, an id both sides used is in the
    registry twice and cannot be told apart, and this refuses.
    """
    branch = branch or git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    stem = stem_of(branch)
    target = root / ANOMALIES / f"{stem}.yaml"
    if target.exists():
        raise AssemblyError(f"{target.relative_to(root)} exists already")
    fork = git(root, "merge-base", "HEAD", base).strip()
    text = (root / REGISTRY).read_text(encoding="utf-8")
    fork_text = git(root, "show", f"{fork}:{REGISTRY}")
    fork_ids = {n for n, _, _ in _registry_blocks(fork_text)}
    lines = text.splitlines()
    seen: dict[int, int] = {}
    for n, _, _ in _registry_blocks(text):
        seen[n] = seen.get(n, 0) + 1
    twice = sorted(n for n, c in seen.items() if c > 1)
    if twice:
        raise AssemblyError(f"{[numbered(n) for n in twice]} are in the registry twice: "
                            f"the base is merged already, so the branch's entries cannot "
                            f"be told apart; run from-branch before merging {base}")
    own = [(n, s0, e) for n, s0, e in _registry_blocks(text) if n not in fork_ids]
    if not own:
        return {}
    # As ``new`` does: a stem whose earlier fragment was assembled starts
    # after the labels the registry records for it.
    first = next_label_number(text, stem)
    mapping = {numbered(n): label(stem, k) for k, (n, _, _) in enumerate(own, first)}
    blocks = []
    keep = [True] * len(lines)
    for n, s, e in own:
        block = lines[s:e]
        while block and not block[-1].strip():
            block.pop()
        block = [ln for ln in block if not LABEL_COMMENT_RE.match(ln)]
        block[0] = f'  - anomaly_id: "{mapping[numbered(n)]}"'
        blocks.append("\n".join(block))
        for i in range(s, e):
            keep[i] = False
    remaining = "\n".join(ln for ln, k in zip(lines, keep) if k).rstrip("\n") + "\n"
    version = tree_version(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f'line: "{line_of(version)}"\nanomalies:\n'
                      + "\n\n".join(blocks) + "\n", encoding="utf-8")
    (root / REGISTRY).write_text(remaining, encoding="utf-8")
    full = re.compile(r"(?<![A-Za-z0-9_])MADD-ANO-(\d{3,})(?![0-9])")
    for rel in sorted(set(git_paths(root, "grep", "-l", "-z", "-I", "-E", "MADD-ANO-[0-9]"))):
        path = root / rel
        if not path.exists() or rel == target.relative_to(root).as_posix():
            continue
        old = path.read_text(encoding="utf-8")
        new = full.sub(lambda m: mapping.get(numbered(int(m.group(1))), m.group(0)), old)
        if new != old:
            path.write_text(new, encoding="utf-8")
    # labels inside the moved entries themselves
    frag = target.read_text(encoding="utf-8")
    target.write_text(full.sub(lambda m: mapping.get(numbered(int(m.group(1))),
                                                     m.group(0)), frag), encoding="utf-8")
    hw_text = (root / HW_FILE).read_text(encoding="utf-8")
    base_hw = _HW_RE.search(git(root, "show", f"{fork}:{HW_FILE}"))
    if base_hw:
        (root / HW_FILE).write_text(
            _HW_RE.sub(lambda m: m.group(1) + base_hw.group(2), hw_text, count=1),
            encoding="utf-8")
    return mapping


def cmd_from_branch(args, root: Path) -> int:
    try:
        mapping = from_branch(root, args.base, args.branch)
    except (AssemblyError, ValueError, RuntimeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if not mapping:
        print(f"this branch adds no registry entry that {args.base} lacks")
        return 0
    for old, new in mapping.items():
        print(f"{old} -> {new}")
    print("\nnext: give each reachable entry a release_note in the fragment; move any "
          "text this branch added to CHANGELOG.md, the release notes or the "
          "development record into a changes fragment (fragments.py new --changes); "
          "regenerate the SOUP table (scripts/generate_soup_tables.py); then run "
          "scripts/check_fragments.py and scripts/check_anomalies.py")
    return 0


# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO_ROOT,
                    help="the repository to work on (default: the one holding this script)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="write this branch's fragment from a template")
    n.add_argument("--anomaly", action="store_true", help="an anomalies fragment")
    n.add_argument("--changes", action="store_true", help="a changes fragment (the default)")
    n.add_argument("--branch", help="the branch name (default: the checked-out one)")
    a = sub.add_parser("assemble", help="number anomalies; at a release, write the "
                       "changelog and the notes")
    a.add_argument("--registry", action="store_true", help="a quiet-point assembly "
                   "of the anomalies (numbering line only)")
    a.add_argument("--release", metavar="X.Y.Z", help="a release assembly, everything")
    a.add_argument("--date", metavar="YYYY-MM-DD", help="the release date (with --release)")
    a.add_argument("--no-soup", action="store_true", help=argparse.SUPPRESS)
    sub.add_parser("relabel", help="rewrite every recorded label to its id")
    f = sub.add_parser("from-branch", help="move a branch's own registry entries "
                       "into a fragment")
    f.add_argument("--base", required=True, metavar="REF", help="the branch's base")
    f.add_argument("--branch", help="the branch name (default: the checked-out one)")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    return {"new": cmd_new, "assemble": cmd_assemble, "relabel": cmd_relabel,
            "from-branch": cmd_from_branch}[args.cmd](args, root)


if __name__ == "__main__":
    sys.exit(main())
