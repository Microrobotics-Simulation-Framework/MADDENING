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

#: The one spelling of a provisional label.
LABEL_RE = re.compile(
    rf"(?<![A-Za-z0-9_])MADD-ANO-NEW-({STEM})-([0-9]+)(?![A-Za-z0-9_-])")

#: Anything that sets out to be a label: ``MADD``, ``ANO`` and ``NEW`` in any
#: case, joined by ``-``, ``_``, a space or nothing, then a separator and a
#: letter or digit.  Each must be :data:`LABEL_RE` at the same place, or it is
#: a variant spelling.  ``MADD-ANO-NEW-<stem>-<n>`` (a placeholder) is not an
#: attempt, nor is a regular expression.
ATTEMPT_RE = re.compile(
    r"(?<![A-Za-z0-9_])MADD[-_ ]?ANO[-_ ]?NEW[-_ ](?=[A-Za-z0-9])", re.I)

#: The comment the assembler writes right after an assembled entry's id line.
LABEL_COMMENT_RE = re.compile(rf"^    # label: NEW-({STEM})-([0-9]+)[ \t]*$", re.M)
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
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    return yaml.load(text, Loader=loader)


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
    out = git(root, "grep", "-n", "-I", "-i", "-E",
              r"MADD[-_ ]?ANO[-_ ]?NEW[-_ ][A-Za-z0-9]", check=False)
    hits = []
    for row in out.splitlines():
        path, _, rest = row.partition(":")
        number, _, text = rest.partition(":")
        if number.isdigit():
            hits.append((path, int(number), text))
    return hits


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
            used = [int(m.group(2)) for m in map(LABEL_RE.fullmatch, recorded_labels(text))
                    if m and m.group(1) == stem]
            body = _ANOMALIES_TEMPLATE.format(
                line=line_of(version), label=label(stem, max(used, default=0) + 1),
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
    out = git(root, "log", "--first-parent", "--reverse", "--diff-filter=A",
              "--name-only", "--format=%x00%H", "--", FRAGMENTS)
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


def _changelog_section(version: str, date: str, bullets: dict[str, list[str]],
                       count: str) -> str:
    out = [f"## [{version}] - {date}\n"]
    for sec in CHANGELOG_SECTIONS:
        items = list(bullets.get(sec, []))
        if sec == "Known Anomalies":
            items = [count] + items
        if not items:
            continue
        out.append(f"\n### {sec}\n\n")
        for item in items:
            item = " ".join(str(item).split())
            out.append(f"{item if item.startswith('- ') else '- ' + item}\n")
    return "".join(out) + "\n"


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

    # 5. release only: the changelog section and the release notes
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
        section = _changelog_section(release, date, bullets,
                                     reachable_count_line(reg_now, release))
        snap.write(CHANGELOG, changelog[:at] + section + changelog[at:])
        _notes_append(root, snap, release, None, None)          # the file exists
        for heading, block in blocks:
            _notes_append(root, snap, release, heading, block)

    # 6. rewrite every assigned label in every tracked text file, and in the
    #    files this run created (a new notes file is not tracked yet)
    if mapping:
        hits = git(root, "grep", "-l", "-I", "-F", "MADD-ANO-NEW-", check=False).split()
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
    _verify(root, take, mapping, appended, hw, k, release, out)


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


def _verify(root, take, mapping, appended, hw, k, release, out):
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
        changelog = (root / CHANGELOG).read_text(encoding="utf-8")
        head = re.search(rf"^## \[{re.escape(release)}\] - ", changelog, re.M)
        section = ""
        if head:
            nxt = re.search(r"^## \[", changelog[head.end():], re.M)
            section = changelog[head.start():head.end() + (nxt.start() if nxt else len(changelog))]
        else:
            problems.append(f"no '## [{release}]' section was written")
        wanted: dict[str, int] = {}
        for f in take:
            if f.kind != "changes":
                continue
            for items in (f.data.get("changelog") or {}).values():
                for item in items or []:
                    text = " ".join(rewrite_labels(str(item), mapping)[0].split())
                    wanted[text] = wanted.get(text, 0) + 1
        for text, times in wanted.items():
            if section.count(text) != times:
                problems.append(f"a changelog bullet appears {section.count(text)} "
                                f"time(s) in the new section, not {times}: {text[:60]}")
        notes = (root / NOTES_DIR / f"v{release}.md").read_text(encoding="utf-8")
        for f in take:
            if f.kind != "changes":
                continue
            for block in (f.data.get("release_notes") or {}).values():
                text = rewrite_labels(str(block), mapping)[0].strip()
                if notes.count(text) < 1:
                    problems.append(f"a release-notes block from {f.rel} is missing")
    problems += check_fragments.label_record_findings(reg_text)
    version = tree_version(root)
    if not is_dev(version):
        # Invariant 3 at a release version: no label at all, nor any fragment.
        problems += [f"release gate: {p}" for p in check_fragments.release_findings(
            root, version, set())]
    if problems:
        raise AssemblyError("the assembly's own checks failed:\n  " + "\n  ".join(problems))


def _untracked_with(root, mapping):
    """New files the assembly wrote (a notes file) that still hold a label."""
    found = []
    for rel in git(root, "ls-files", "--others", "--exclude-standard").split():
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
    for rel in sorted(set(git(root, "grep", "-l", "-I", "-F", "MADD-ANO-NEW-",
                              check=False).split())):
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
    mapping = {numbered(n): label(stem, k) for k, (n, _, _) in enumerate(own, 1)}
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
    for rel in sorted(set(git(root, "grep", "-l", "-I", "-E", "MADD-ANO-[0-9]",
                              check=False).split())):
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
