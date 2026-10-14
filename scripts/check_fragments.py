#!/usr/bin/env python3
"""CI gate: the changelog and registry fragments, and every provisional label.

Usage:
    python scripts/check_fragments.py [--base REF] [--root DIR]

A pull request records its change in ``docs/fragments/`` (``scripts/
fragments.py`` says how) and never edits a shared record file.  This gate holds
the fragments to the scheme, on every push:

* **Layout.**  Every file under ``docs/fragments/`` is ``README.txt``,
  ``anomalies/<stem>.yaml`` or ``changes/<stem>.yaml``, ``<stem>`` being
  ``[a-z0-9]+(-[a-z0-9]+)*``.
* **Format.**  A fragment parses; it carries ``line:``, a release line in
  ``RELEASE_LINES`` no newer than the tree's, and no unknown top-level key.  A
  changes file maps the changelog's sections to bullets and the notes'
  headings to text, and names no other section or heading.  An anomalies
  file lists entries whose ids are labels.
* **Labels.**  ``MADD-ANO-NEW-<stem>-<n>``: the stem is the file's own, ``<n>``
  is unique in the file, no label is one the registry already records (a
  fragment resurrected by a stale merge, or a stem reused after assembly), an
  entry's id line is the registry's ``  - anomaly_id: "..."``, a reachable
  entry has a ``release_note``, and a number in a fragment or a label in the
  registry is refused.
* **Every label in the tree is live.**  Each ``MADD-ANO-NEW-`` spelled in a
  tracked file names a label a fragment defines, in the one spelling; one the
  registry records has been assembled and must be rewritten
  (``fragments.py relabel``).  A label in a file name is refused.
* **Every assembled entry records its label** (the note's invariant 5.4):
  each entry numbered from ``FRAGMENTS_FROM`` on carries exactly one ``# label:``
  comment, right after its id line; no older entry carries one; no label is
  recorded twice.  So a hand-numbered new entry fails.
* **The release gate.**  At a version without ``.dev`` (``rc`` included): no
  file under ``docs/fragments/`` but the README, no label anywhere, an empty
  Unreleased section (``## [Unreleased]``, ``## Unreleased``, in any case),
  and a ``docs/release_notes/vX.Y.Z.md``.  At any version, ``CHANGELOG.md``
  has one ``## [X.Y.Z]`` section per version.
* **The ratchet, over the line's whole history.**  Every label ever added
  under ``docs/fragments/anomalies/`` in the line's first-parent history is
  still in a fragment here, or recorded by the registry; every changes
  fragment ever added there is still here, or each of its changelog bullets
  is in ``CHANGELOG.md`` and each release-notes block in a notes file (as the
  release assembly writes them).  Deleting a merged fragment turned nothing
  red otherwise: the SOUP table never listed it and the high-water mark never
  counted it.  A wrong entry is closed (``duplicate``, ``resolved``), never
  deleted.  The line is ``HEAD`` (a push), or ``--base REF`` (a pull request:
  its base, so that the branch's own unmerged history is not read).  How many
  commits a push holds does not matter: every commit of the line is read.
* **A recorded label has a history.**  A registry ``# label:`` comment names
  a label some fragment in this tree's history held (any parent), so a
  hand-written comment does not pass as an assembled one.

Entry fields themselves (the schema, references, evidence, ranges) are
``scripts/check_anomalies.py``'s: it reads the registry and the fragments as
one merged view.  Exit 0 when nothing is found, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fragments as F  # noqa: E402

_NUMBERED_ID = re.compile(r"MADD-ANO-\d+")


def layout_findings(root: Path) -> list[str]:
    out = []
    for path in F.fragment_paths(root):
        rel = path.relative_to(root).as_posix()
        if rel == F.README:
            continue
        parent, name = os.path.split(rel)
        ok = (parent in (F.ANOMALIES, F.CHANGES) and name.endswith(".yaml")
              and F.STEM_RE.fullmatch(name[:-len(".yaml")]))
        if not ok:
            out.append(f"{rel}: not a fragment.  docs/fragments/ holds README.txt, "
                       f"anomalies/<stem>.yaml and changes/<stem>.yaml only, <stem> "
                       f"being the branch name lower-cased with / _ . as - "
                       f"({F.STEM})")
    return out


def _line_findings(frag: F.Fragment, tree_line: str) -> list[str]:
    line = frag.line
    if line is None:
        return [f"{frag.rel}: no 'line:' (the release line it was written on, as a "
                f"quoted string such as \"{tree_line}\")"]
    if not isinstance(line, str) or line not in F.RELEASE_LINES:
        return [f"{frag.rel}: line {line!r} is not a release line in RELEASE_LINES "
                f"{F.RELEASE_LINES} (quote it: \"0.4\")"]
    if F.line_key(line) > F.line_key(tree_line):
        return [f"{frag.rel}: line {line!r} is newer than this tree's line "
                f"{tree_line}"]
    return []


def _changes_findings(frag: F.Fragment) -> list[str]:
    out, data = [], frag.data
    if "changelog" not in data and "release_notes" not in data:
        out.append(f"{frag.rel}: neither 'changelog:' nor 'release_notes:'")
    log = data.get("changelog")
    if "changelog" in data:
        if not isinstance(log, dict) or not log:
            out.append(f"{frag.rel}: 'changelog:' must map sections to bullets")
        else:
            for sec, items in log.items():
                if sec not in F.CHANGELOG_SECTIONS:
                    out.append(f"{frag.rel}: changelog section {sec!r} is not one of "
                               f"{list(F.CHANGELOG_SECTIONS)}")
                if (not isinstance(items, list) or not items or not all(
                        isinstance(x, str) and x.strip() for x in items)):
                    out.append(f"{frag.rel}: changelog section {sec!r} must be a list "
                               f"of non-empty bullets")
    notes = data.get("release_notes")
    if "release_notes" in data:
        if not isinstance(notes, dict) or not notes:
            out.append(f"{frag.rel}: 'release_notes:' must map headings to text")
        else:
            for heading, text in notes.items():
                if heading not in F.NOTES_HEADINGS:
                    out.append(f"{frag.rel}: release-notes heading {heading!r} is not "
                               f"one of {list(F.NOTES_HEADINGS)}")
                if not isinstance(text, str) or not text.strip():
                    out.append(f"{frag.rel}: release-notes heading {heading!r} has no text")
    return out


def _anomalies_findings(frag: F.Fragment, recorded: dict) -> list[str]:
    out = []
    raw = frag.data.get("anomalies")
    if not isinstance(raw, list) or not raw:
        return [f"{frag.rel}: 'anomalies:' must be a non-empty list of entries"]
    seen: dict[int, str] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            out.append(f"{frag.rel}: an entry is not a mapping: {entry!r}")
            continue
        aid = str(entry.get("anomaly_id", ""))
        m = F.LABEL_RE.fullmatch(aid)
        if not m:
            if _NUMBERED_ID.fullmatch(aid):
                out.append(f"{frag.rel}: {aid} is a number.  A fragment entry carries "
                           f"its label, MADD-ANO-NEW-{frag.stem}-<n>; only "
                           f"scripts/fragments.py assemble assigns numbers")
            else:
                out.append(f"{frag.rel}: anomaly_id {aid!r} is not a label "
                           f"MADD-ANO-NEW-{frag.stem}-<n>")
            continue
        stem, n = m.group(1), int(m.group(2))
        if stem != frag.stem:
            out.append(f"{frag.rel}: {aid} carries the stem {stem!r}, not this file's "
                       f"{frag.stem!r}; every label in a file carries its name")
        if n in seen:
            out.append(f"{frag.rel}: {aid} is given twice; <n> is unique in a file")
        seen[n] = aid
        if aid in recorded:
            out.append(f"{frag.rel}: {aid} is already assembled as "
                       f"{', '.join(recorded[aid])}.  A merge resurrected this "
                       f"fragment (take the deletion, and re-apply any edit to the "
                       f"numbered entry), or the stem was reused (start <n> after "
                       f"the recorded ones)")
        for key in F.EXTRA_KEYS:
            if key in entry and not (isinstance(entry[key], str) and entry[key].strip()):
                out.append(f"{frag.rel}: {aid}: {key} must be non-empty text")
        if (entry.get("resolution_status") not in F.UNREACHABLE
                and not entry.get("release_note")):
            out.append(f"{frag.rel}: {aid} is reachable "
                       f"({entry.get('resolution_status')!r}) and has no release_note "
                       f"(the release notes' Known-anomalies text)")
    spelled = [aid for aid, _ in F.entry_blocks(frag.text)]
    parsed = [str(e.get("anomaly_id")) for e in raw if isinstance(e, dict)]
    if spelled != parsed:
        out.append(f"{frag.rel}: the entries' id lines spell {spelled} but the file "
                   f"parses to {parsed}; write each as '  - anomaly_id: \"MADD-ANO-NEW-"
                   f"{frag.stem}-<n>\"', the registry's own layout")
    if F.ANY_LABEL_COMMENT_RE.search(frag.text):
        out.append(f"{frag.rel}: a '# label:' comment; only the registry records labels")
    return out


def format_findings(root: Path, tree_line: str, recorded: dict) -> tuple[list[str], set]:
    """``(findings, live labels)`` over every fragment."""
    out, live = [], set()
    for frag in F.read_fragments(root):
        if frag.error:
            out.append(frag.error)
            continue
        if not isinstance(frag.data, dict):
            out.append(f"{frag.rel}: not a YAML mapping")
            continue
        allowed = F.ANOMALIES_KEYS if frag.kind == "anomalies" else F.CHANGES_KEYS
        for key in sorted(set(frag.data) - set(allowed), key=str):
            out.append(f"{frag.rel}: unknown key {key!r} (allowed: {', '.join(allowed)})")
        out += _line_findings(frag, tree_line)
        if frag.kind == "changes":
            out += _changes_findings(frag)
        else:
            out += _anomalies_findings(frag, recorded)
            live |= {str(e.get("anomaly_id")) for e in frag.entries
                     if F.LABEL_RE.fullmatch(str(e.get("anomaly_id")))}
    return out, live


def label_findings(root: Path, live: set, recorded: dict, registry_text: str) -> list[str]:
    """Every attempted label in a tracked file is a live one, spelled one way."""
    out = []
    for path, line_no, text in F.tracked_label_hits(root):
        for m in F.ATTEMPT_RE.finditer(text):
            exact = F.LABEL_RE.match(text, m.start())
            where = f"{path}:{line_no}"
            if exact is None:
                shown = text[m.start():m.start() + 40].split()[0]
                out.append(f"{where}: {shown!r} is not the label spelling "
                           f"MADD-ANO-NEW-<stem>-<n> (lower-case stem, dashes)")
                continue
            lab = exact.group(0)
            if lab in live:
                continue
            if lab in recorded:
                out.append(f"{where}: {lab} was assembled as "
                           f"{', '.join(recorded[lab])}; run scripts/fragments.py relabel")
            else:
                out.append(f"{where}: {lab} names no label any fragment defines")
    for name in F.git(root, "ls-files").splitlines():
        if re.search(r"MADD[-_ ]?ANO[-_ ]?NEW", name, re.I):
            out.append(f"{name}: a label in a file name; name files by what they hold")
    for m in re.finditer(r'^\s*-?\s*anomaly_id:\s*["\']?([^"\'\s]+)', registry_text, re.M):
        if F.ATTEMPT_RE.match(m.group(1)) or F.LABEL_RE.fullmatch(m.group(1)):
            out.append(f"{F.REGISTRY}: {m.group(1)} is a label in the registry; an "
                       f"entry reaches the registry numbered, by scripts/fragments.py "
                       f"assemble")
    return out


def label_record_findings(registry_text: str, held=None) -> list[str]:
    """Invariant 5.4: each entry from FRAGMENTS_FROM on records exactly one label.

    ``held`` is every label a fragment in the tree's history held
    (:func:`history`); when given, a recorded label outside it is refused: the
    comment was written by hand, not by the assembler.
    """
    out = []
    lines = registry_text.splitlines()
    starts = [(i, int(m.group(1))) for i, ln in enumerate(lines)
              if (m := F.NUMBERED_ID_LINE_RE.match(ln))]
    for k, (i, n) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        block = lines[i + 1:end]
        strict = [j for j, ln in enumerate(block) if F.LABEL_COMMENT_RE.match(ln)]
        loose = [j for j, ln in enumerate(block) if F.ANY_LABEL_COMMENT_RE.match(ln)]
        aid = F.numbered(n)
        if len(loose) != len(strict):
            out.append(f"{F.REGISTRY}: {aid} has a label comment not written "
                       f"'    # label: NEW-<stem>-<n>'")
        if n >= F.FRAGMENTS_FROM:
            if len(strict) != 1 or strict[0] != 0:
                out.append(f"{F.REGISTRY}: {aid} is numbered from {F.FRAGMENTS_FROM} on "
                           f"and carries {len(strict)} '# label:' comment(s) "
                           f"{'' if not strict else 'not right after its id line'}; "
                           f"every such entry comes from a fragment, numbered by "
                           f"scripts/fragments.py assemble, which records exactly "
                           f"one.  Do not number an entry by hand")
        elif strict:
            out.append(f"{F.REGISTRY}: {aid} predates fragments (below "
                       f"{F.FRAGMENTS_FROM}) and carries a '# label:' comment")
    for lab, ids in F.recorded_labels(registry_text).items():
        if len(ids) > 1:
            out.append(f"{F.REGISTRY}: {lab} is recorded {len(ids)} times ({ids})")
        if held is not None and lab not in held:
            out.append(f"{F.REGISTRY}: {', '.join(ids)} records {lab}, which no fragment "
                       f"in this tree's history ever held.  The '# label:' comment is "
                       f"written by scripts/fragments.py assemble when it numbers a "
                       f"fragment's entry; do not number an entry or write the comment "
                       f"by hand")
    return out


#: Every spelling of the Unreleased heading the release gate reads: with or
#: without brackets, in any case.
UNRELEASED_RE = re.compile(r"^##[ \t]+\[?unreleased\]?[ \t]*$", re.M | re.I)
#: A version's changelog heading, ``## [X.Y.Z]`` with or without a date.
VERSION_HEADING_RE = re.compile(r"^## \[(\d[^\]\s]*)\]", re.M)


def is_tagged(root: Path, version: str) -> bool:
    """Is ``v<version>`` a tag here (the release is out)?"""
    return bool(F.git(root, "rev-parse", "-q", "--verify", f"refs/tags/v{version}^{{commit}}",
                      check=False).strip())


def changelog_findings(root: Path) -> list[str]:
    """One ``## [X.Y.Z]`` section per version, at any version of the tree."""
    changelog = (root / F.CHANGELOG).read_text(encoding="utf-8")
    seen: dict[str, int] = {}
    for m in VERSION_HEADING_RE.finditer(changelog):
        if m.group(1).lower() != "unreleased":
            seen[m.group(1)] = seen.get(m.group(1), 0) + 1
    return [f"{F.CHANGELOG} has {n} '## [{v}]' sections; a version has one (a second "
            f"release assembly adds to the first)" for v, n in seen.items() if n > 1]


def release_findings(root: Path, version: str, live: set, leftover_why: str | None = None) -> list[str]:
    """The release gate, at a version without ``.dev``.  ``leftover_why`` replaces
    the reason given for a fragment still here (the assembler says it deferred it)."""
    if F.is_dev(version):
        return []
    out = []
    base = F.base_of(version)
    changelog = (root / F.CHANGELOG).read_text(encoding="utf-8")
    leftovers = [p.relative_to(root).as_posix() for p in F.fragment_paths(root)
                 if p.relative_to(root).as_posix() != F.README]
    if leftover_why is not None:
        why = leftover_why
    elif leftovers and is_tagged(root, base):
        why = (f"v{base} is tagged, so {base} is out and this fragment belongs to the "
               f"next version: bump the tree to the next .dev0 version, and the fragment "
               f"waits for that release's assembly")
    elif leftovers and re.search(rf"^## \[{re.escape(base)}\]", changelog, re.M):
        why = (f"{version} is a release version and this fragment arrived after its "
               f"release assembly: run scripts/fragments.py assemble --release {base} "
               f"again, which adds it to the existing '## [{base}]' section")
    else:
        why = (f"{version} is a release version and this fragment was never assembled "
               f"(scripts/fragments.py assemble --release {base})")
    out += [f"{rel}: {why}" for rel in leftovers]
    for path, line_no, _ in F.tracked_label_hits(root):
        out.append(f"{path}:{line_no}: a provisional label at release version {version}")
    for m in UNRELEASED_RE.finditer(changelog):
        nxt = re.search(r"^## ", changelog[m.end():], re.M)
        body = changelog[m.end():m.end() + nxt.start()] if nxt else changelog[m.end():]
        if body.strip():
            out.append(f"{F.CHANGELOG}: '{m.group(0).strip()}' is not empty at release "
                       f"version {version}; the release assembly writes the section")
    notes = root / F.NOTES_DIR / f"v{base}.md"
    if not notes.is_file():
        out.append(f"{notes.relative_to(root).as_posix()} is missing at release "
                   f"version {version}")
    return out


# ---------------------------------------------------------------------------
# The history: what the line ever merged
# ---------------------------------------------------------------------------

_ADDED_ID_LINE = re.compile(r'^\+[ \t]*-[ \t]*anomaly_id:[ \t]*["\']?(MADD-ANO-NEW-[A-Za-z0-9-]+)')


class History:
    """What a line's history added under ``docs/fragments/``.

    ``labels``: every label an anomalies fragment of the line ever held, read
    from the first-parent history (each merge as its diff against the line,
    so a branch's own commits before it merged are never read), with the
    commit that brought it.  ``held``: every label any fragment held in any
    commit reachable from ``HEAD``, every parent (``# label:`` comments must
    name one).  ``changes``: every changes fragment the line ever added, with
    the commit that added it.  ``error``: why the history could not be read
    (the ratchet then fails closed).
    """

    def __init__(self):
        self.labels: dict[str, str] = {}
        self.held: set[str] = set()
        self.changes: dict[str, str] = {}
        self.error: str | None = None


def _added_labels(log: str) -> dict[str, str]:
    """``{label: commit}`` for every label an id line adds in ``log``
    (``git log -p --format=%x00%H``), the oldest commit first."""
    found: dict[str, str] = {}
    for chunk in log.split("\0")[1:]:
        sha, _, patch = chunk.partition("\n")
        for line in patch.splitlines():
            if line.startswith("+++"):
                continue
            m = _ADDED_ID_LINE.match(line)
            if m and F.LABEL_RE.fullmatch(m.group(1)):
                found.setdefault(m.group(1), sha.strip())
    return found


def history(root: Path, line: str = "HEAD") -> History:
    """The :class:`History` of ``line`` (a ref), and of ``HEAD`` for ``held``."""
    h = History()
    try:
        if F.git(root, "rev-parse", "--is-shallow-repository").strip() != "false":
            h.error = ("this is a shallow clone, and the ratchet reads the whole history "
                       "(CI: fetch-depth 0; here: git fetch --unshallow)")
            return h
        first_parent = ("log", "--first-parent", "--diff-merges=first-parent", "--no-renames",
                        "--reverse", "--format=%x00%H")
        h.labels = _added_labels(F.git(root, *first_parent, "-p", line, "--", F.ANOMALIES))
        h.held = set(_added_labels(F.git(root, "log", "-m", "--no-renames", "-p",
                                         "--format=%x00%H", "HEAD", "--", F.ANOMALIES)))
        log = F.git(root, *first_parent, "--diff-filter=A", "--name-only", line, "--",
                    F.CHANGES)
        for chunk in log.split("\0")[1:]:
            sha, _, names = chunk.partition("\n")
            for rel in names.splitlines():
                if rel.strip().endswith(".yaml"):
                    h.changes.setdefault(rel.strip(), sha.strip())
    except RuntimeError as exc:
        h.error = f"the history of {line} cannot be read: {exc}"
    return h


def _last_text(root: Path, line: str, rel: str) -> str | None:
    """``rel`` as the line last held it: at ``line``, or just before the
    first-parent commit that deleted it."""
    here = F.git(root, "show", f"{line}:{rel}", check=False)
    if here:
        return here
    gone = F.git(root, "log", "--first-parent", "--diff-merges=first-parent", "--no-renames",
                 "--diff-filter=D", "--no-patch", "--format=%H", "-1", line, "--", rel,
                 check=False).strip()
    if not gone:
        return None
    return F.git(root, "show", f"{gone}^1:{rel}", check=False) or None


def _not_merged_hint(root: Path, sha: str) -> str:
    """A hint when ``sha`` (from the base's history) is not in this branch yet."""
    done = subprocess.run(["git", "-C", str(root), "merge-base", "--is-ancestor", sha,
                           "HEAD"], capture_output=True)
    return "" if done.returncode == 0 else (
        "  (This branch does not hold that commit yet: merge the base in.)")


def ratchet_findings(root: Path, hist: History, live: set, recorded: dict,
                     line: str = "HEAD", own_branch_hint: bool = False) -> list[str]:
    """Nothing the line ever merged under ``docs/fragments/`` was dropped."""
    out = _ratchet_findings(root, hist, live, recorded, line)
    if out and own_branch_hint:
        out.append("(Reading HEAD's history: on a branch of your own, a fragment you added "
                   "and dropped before merging shows above.  Run with --base <the branch "
                   "you merge into>, which reads that line's history only, as CI does for "
                   "a pull request.)")
    return out


def _ratchet_findings(root, hist, live, recorded, line):
    if hist.error:
        return [f"the ratchet cannot check that no merged fragment was deleted: "
                f"{hist.error}"]
    out = []
    for lab, sha in sorted(hist.labels.items()):
        if lab not in live and lab not in recorded:
            out.append(f"{lab} came into the line in {sha[:10]} and is in no fragment "
                       f"here, nor recorded by the registry: a merged fragment, or an "
                       f"entry of one, was deleted or renamed.  A wrong entry is closed "
                       f"(resolution_status duplicate or resolved), never deleted"
                       + _not_merged_hint(root, sha))
    changelog = (root / F.CHANGELOG).read_text(encoding="utf-8")
    notes = "\n".join(p.read_text(encoding="utf-8")
                      for p in sorted((root / F.NOTES_DIR).glob("v*.md")))
    mapping = {lab: ids[0] for lab, ids in recorded.items() if len(ids) == 1}
    for rel, sha in sorted(hist.changes.items()):
        if (root / rel).is_file():
            continue
        text = _last_text(root, line, rel)
        try:
            data = F.load_yaml(text) if text else None
        except Exception:   # yaml.YAMLError: a fragment the gate refused, merged anyway
            data = None
        if not isinstance(data, dict):
            out.append(f"{rel} came into the line in {sha[:10]} and is gone, and its last "
                       f"text cannot be read to show it was assembled")
            continue
        for sec, items in (data.get("changelog") or {}).items():
            for item in items if isinstance(items, list) else []:
                bullet = F.render_bullet(F.rewrite_labels(str(item), mapping)[0])
                if bullet not in changelog:
                    out.append(f"{rel} came into the line in {sha[:10]} and is gone, but "
                               f"its {sec} bullet is not in {F.CHANGELOG}: "
                               f"{' '.join(str(item).split())[:70]!r}.  A merged changes "
                               f"fragment leaves only by a release assembly"
                               + _not_merged_hint(root, sha))
        for heading, block in (data.get("release_notes") or {}).items():
            text = F.rewrite_labels(str(block), mapping)[0].strip()
            if text and text not in notes:
                out.append(f"{rel} came into the line in {sha[:10]} and is gone, but its "
                           f"{heading!r} release-notes text is in no notes file: "
                           f"{' '.join(text.split())[:70]!r}.  A merged changes fragment "
                           f"leaves only by a release assembly"
                           + _not_merged_hint(root, sha))
    return out


def findings(root, base: str | None = None, release_gate: bool = True) -> list[str]:
    """Every finding of the gate; ``release_gate=False`` leaves out the release
    gate alone (the assembler's preflight: a release assembly is what clears
    it).  ``base`` is the line whose history the ratchet reads (default
    ``HEAD``)."""
    root = Path(root)
    version = F.tree_version(root)
    tree_line = F.line_of(version)
    registry_text = (root / F.REGISTRY).read_text(encoding="utf-8")
    recorded = F.recorded_labels(registry_text)
    line = base or "HEAD"
    hist = history(root, line)
    out = layout_findings(root)
    fmt, live = format_findings(root, tree_line, recorded)
    out += fmt
    out += label_findings(root, live, recorded, registry_text)
    out += label_record_findings(registry_text, None if hist.error else hist.held)
    out += changelog_findings(root)
    if release_gate:
        out += release_findings(root, version, live)
    out += ratchet_findings(root, hist, live, recorded, line, own_branch_hint=base is None)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", metavar="REF",
                    help="the line whose history the ratchet reads: a pull request's "
                         "base (default: HEAD, the checked-out line)")
    ap.add_argument("--root", type=Path, default=F.REPO_ROOT,
                    help="the repository to check (default: the one holding this script)")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    problems = findings(root, args.base)
    for p in problems:
        print(f"ERROR: {p}", file=sys.stderr)
    if problems:
        return 1
    n = len([p for p in F.fragment_paths(root) if p.relative_to(root).as_posix() != F.README])
    print(f"OK: {n} fragment(s); every label live and spelled one way; every assembled "
          f"entry records its label, and every recorded label was a fragment's; of what "
          f"fragments brought into the history of {args.base or 'HEAD'}, nothing was "
          f"dropped (each label live or recorded, each changelog bullet and notes text "
          f"live or assembled)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
