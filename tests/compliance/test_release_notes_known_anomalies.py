"""The release notes list the reachable anomalies; the development record indexes them all.

A ``.0`` release's notes say that their "Known anomalies" section lists every
entry of ``known_anomalies.yaml`` whose defect a user can reach.  The claim
was false twice in 0.4.0: first as a hand-counted total that was stale by the
time it was read, then as a list that left out MADD-ANO-035 and never
mentioned MADD-ANO-050 anywhere in the document.  Nothing checked it.

This does, for a whole release line rather than one file, so that a patch
release never has to edit the released ``.0`` notes.  The reachable set is
read with ``scripts/check_anomalies.py``'s own ``UNREACHABLE_STATUSES`` -- the
rule the registry gate and the SOUP headline already share -- over the merged
view: the registry's entries and every entry still in a fragment
(``docs/fragments/anomalies/``, ``scripts/fragments.py``).

The notes files of the line are ``docs/release_notes/vX.Y.*.md`` for the
tree's line X.Y; on a development tree whose line has no ``.0`` notes yet
(the next minor's cycle), the newest line that has them.  Of their "Known
anomalies" sections:

* every reachable registry entry is named by its full ID in one of them, or
  by a fragment (a changes fragment's "Known anomalies" notes text, or an
  anomalies fragment's ``release_note``);
* every reachable fragment entry carries its ``release_note``, which the
  assembler writes into the notes when it numbers the entry;
* an entry the ``.0`` notes name that is no longer reachable is named again
  by a later notes file of the line, or by a fragment, so a resolved defect
  is not left listed as one a user can still meet;
* no section names an ID the registry lacks.

The index of the whole registry lives in the development records,
``docs/validation/development_record_v*.md``, under "Index of the anomaly
registry".  Across them:

* every registry entry is named, by its full ID or by a closing
  "**Resolved.**" paragraph's three-digit shorthand;
* no "**Resolved.**" paragraph names a reachable entry, so a live defect
  cannot be filed among the closed ones.

At a release version (no ``.dev``), that version's changelog section gives
the reachable entries as a count, which the release assembly writes; the
count is held to the registry here, so it cannot go stale.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTES_DIR = REPO_ROOT / "docs" / "release_notes"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
VALIDATION = REPO_ROOT / "docs" / "validation"
REGISTRY = VALIDATION / "known_anomalies.yaml"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import fragments  # noqa: E402
from check_anomalies import UNREACHABLE_STATUSES  # noqa: E402

NOTES_HEADING = "## Known anomalies"
INDEX_HEADING = "## Index of the anomaly registry"
RESOLVED_MARKER = "**Resolved.**"
_FULL_ID = re.compile(r"MADD-ANO-(\d{3})")
#: The closing paragraph's shorthand: "`MADD-ANO-006`, `019`, `020` ...".
_BARE_ID = re.compile(r"`(\d{3})`")
_NOTES_FILE = re.compile(r"v(\d+)\.(\d+)(?:\.(\d+))?\.md")


def section(text: str, heading: str) -> str:
    """The text of the ``heading`` section, up to the next ``## `` or the end."""
    match = re.search(rf"^{re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    if match is None:
        raise AssertionError(f"no {heading!r} section")
    return match.group(1)


def _section_or_empty(text: str, heading: str) -> str:
    try:
        return section(text, heading)
    except AssertionError:
        return ""


def _full_ids(text: str) -> set[str]:
    return {f"MADD-ANO-{n}" for n in _FULL_ID.findall(text)}


def _any_ids(text: str) -> set[str]:
    return _full_ids(text) | {f"MADD-ANO-{n}" for n in _BARE_ID.findall(text)}


def _reachable(entries: list[dict]) -> dict:
    return {a["anomaly_id"]: a for a in entries
            if a.get("resolution_status") not in UNREACHABLE_STATUSES}


def line_notes(names: list[str], version: str) -> list[str]:
    """The notes files of ``version``'s line, oldest first, from ``names``.

    A development tree whose line has no ``.0`` notes yet reads the newest
    line that has them.
    """
    by_line: dict[tuple, list] = {}
    for name in names:
        m = _NOTES_FILE.fullmatch(name)
        if m:
            key = (int(m.group(1)), int(m.group(2)))
            by_line.setdefault(key, []).append((int(m.group(3) or 0), name))
    released = sorted(k for k, files in by_line.items() if any(p == 0 for p, _ in files))
    here = tuple(int(x) for x in fragments.line_of(version).split("."))
    if here not in released:
        older = [k for k in released if k < here]
        if not older:
            return []
        here = older[-1]
    return [name for _, name in sorted(by_line[here])]


def fragment_view(root: Path) -> tuple[list[dict], set[str]]:
    """``(entries, named)``: every fragment entry, and every numbered ID a
    fragment names in Known-anomalies text."""
    entries, named = [], set()
    for frag in fragments.read_fragments(root):
        if not isinstance(frag.data, dict):
            continue
        if frag.kind == "anomalies":
            entries += frag.entries
            named |= {aid for a in frag.entries
                      for aid in _full_ids(str(a.get("release_note", "")))}
        else:
            notes = frag.data.get("release_notes") or {}
            named |= _full_ids(str(notes.get("Known anomalies", "")))
    return entries, named


def notes_findings(notes: list[tuple[str, str]], registry: dict,
                   frag_entries=(), frag_named=frozenset()) -> list[str]:
    """Every way the line's notes fail their requirements; empty if none.

    ``notes`` is ``[(name, text)]`` for the notes files of the line, oldest
    (the ``.0`` notes) first.
    """
    if not notes:
        return ["no notes file of this line or an older one has a .0 release"]
    entries = {a["anomaly_id"]: a for a in registry["anomalies"]}
    reachable = _reachable(registry["anomalies"])
    named = [(name, _full_ids(_section_or_empty(text, NOTES_HEADING)))
             for name, text in notes]
    everywhere = set().union(*(ids for _, ids in named)) | set(frag_named)
    later = set().union(*(ids for _, ids in named[1:])) | set(frag_named)
    findings = []
    for aid in sorted(set(reachable) - everywhere):
        a = entries[aid]
        findings.append(
            f"{aid} is reachable ({a.get('resolution_status')}) and no notes file of "
            f"the line names it: {a.get('title', '')[:70]}")
    first_name, first = named[0]
    for aid in sorted(first - set(reachable) - later):
        if aid in entries:
            findings.append(
                f"{aid} is named by {first_name} and is no longer reachable "
                f"({entries[aid].get('resolution_status')}), and no later notes file "
                f"of the line names it again")
    for name, ids in named:
        for aid in sorted(ids - set(entries)):
            findings.append(f"{name} names {aid}, which is not in the registry")
    for a in frag_entries:
        if (a.get("resolution_status") not in UNREACHABLE_STATUSES
                and not str(a.get("release_note") or "").strip()):
            findings.append(f"{a.get('anomaly_id')} is reachable and its fragment "
                            f"gives no release_note")
    return findings


def index_findings(records: list[str], registry: dict) -> list[str]:
    """Every way the records' indexes fail their two requirements; empty if none."""
    indexes = []
    for record in records:
        try:
            indexes.append(section(record, INDEX_HEADING))
        except AssertionError:
            continue
    if not indexes:
        return [f"no development record has an {INDEX_HEADING!r} section"]
    findings = []
    entries = {a["anomaly_id"]: a for a in registry["anomalies"]}
    reachable = _reachable(registry["anomalies"])
    named = set()
    for index in indexes:
        named |= _any_ids(index)
        if RESOLVED_MARKER not in index:
            findings.append(f"an index has no {RESOLVED_MARKER!r} paragraph")
            continue
        _, resolved_paragraph = index.split(RESOLVED_MARKER, 1)
        for aid in sorted(set(reachable) & _any_ids(resolved_paragraph)):
            findings.append(
                f"{aid} is reachable ({entries[aid].get('resolution_status')}) "
                f"and the {RESOLVED_MARKER!r} paragraph lists it")
    for aid in sorted(set(entries) - named):
        findings.append(f"{aid} is in the registry and not named in the index")
    return findings


def changelog_count_findings(changelog: str, registry: dict) -> list[str]:
    """At a release version, whether its changelog section's count of
    reachable entries is the registry's.  Not checked on a ``.dev`` tree:
    the release assembly writes the count."""
    version = str(registry.get("maddening_version", ""))
    if fragments.is_dev(version):
        return []
    head = re.search(rf"^## \[{re.escape(version)}\] - ", changelog, re.M)
    if head is None:
        return [f"the changelog has no [{version}] section"]
    nxt = re.search(r"^## \[", changelog[head.end():], re.M)
    release = changelog[head.start():head.end() + (nxt.start() if nxt else len(changelog))]
    count = re.compile(rf"(\d+) registry entries are reachable in {re.escape(version)}: "
                       r"(\d+) open and (\d+) partly resolved"
                       r"(?:, and (\d+) with another status)?")
    match = count.search(" ".join(release.split()))
    if match is None:
        return [f"the changelog's [{version}] section gives no count of reachable entries"]
    reachable = _reachable(registry["anomalies"]).values()
    n_open = sum(a.get("resolution_status") == "open" for a in reachable)
    n_part = sum(a.get("resolution_status") == "partially_resolved" for a in reachable)
    said = tuple(int(x or 0) for x in match.groups())
    true = (len(reachable), n_open, n_part, len(reachable) - n_open - n_part)
    return [] if said == true else [f"the changelog says {said} (reachable, open, partly "
                                    f"resolved, other); the registry has {true}"]


def _notes_of(version: str) -> list[tuple[str, str]]:
    names = sorted(p.name for p in NOTES_DIR.glob("v*.md"))
    return [(n, (NOTES_DIR / n).read_text()) for n in line_notes(names, version)]


@pytest.fixture(scope="module")
def registry() -> dict:
    return yaml.load(REGISTRY.read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))


@pytest.fixture(scope="module")
def notes(registry) -> list[tuple[str, str]]:
    return _notes_of(str(registry["maddening_version"]))


@pytest.fixture(scope="module")
def records() -> list[str]:
    return [p.read_text() for p in sorted(VALIDATION.glob("development_record_v*.md"))]


@pytest.fixture(scope="module")
def frag() -> tuple[list[dict], set[str]]:
    return fragment_view(REPO_ROOT)


def test_the_notes_name_every_reachable_entry(notes, registry, frag):
    assert notes_findings(notes, registry, *frag) == []


def test_the_record_indexes_every_registry_entry(records, registry):
    assert index_findings(records, registry) == []


def test_the_reachable_set_is_not_empty(registry):
    """A check over an empty set passes vacuously; this line has open entries."""
    assert _reachable(registry["anomalies"])


def test_the_changelog_counts_the_reachable_entries(registry):
    assert changelog_count_findings(CHANGELOG.read_text(), registry) == []


# The checks have to be able to fail.  Each case below breaks one requirement
# on a copy of the real inputs and expects the finding that names it.


def _entry(aid="MADD-ANO-999", status="open", **more):
    return {"anomaly_id": aid, "title": "a synthetic entry",
            "resolution_status": status, **more}


def _grown(registry, *entries):
    return {**registry, "anomalies": [*registry["anomalies"], *entries]}


def _edit_first(notes, change):
    (name, text), *rest = notes
    body = section(text, NOTES_HEADING)
    return [(name, text.replace(body, change(body)))] + rest


@pytest.mark.parametrize("status", ["open", "partially_resolved", "wont_fix",
                                    "a status nobody enumerated"])
def test_a_reachable_entry_the_notes_omit_is_found(notes, registry, status):
    findings = notes_findings(notes, _grown(registry, _entry(status=status)))
    assert any(f.startswith("MADD-ANO-999 is reachable") for f in findings), findings


def _edit_every(notes, change):
    """Every notes file of the line with ``change`` applied to its Known
    anomalies section (a file without one is left as it is)."""
    edited = []
    for name, text in notes:
        body = _section_or_empty(text, NOTES_HEADING)
        edited.append((name, text.replace(body, change(body)) if body else text))
    return edited


def _deleting_an_entry_from_the_notes_is_found(notes, registry):
    """The newest reachable entry, deleted from every notes file of the line
    that names it, is reported.  (Deleting it from the ``.0`` notes alone
    proved nothing once a later notes file named it too.)"""
    victim = sorted(_reachable(registry["anomalies"]))[-1]
    edited = _edit_every(notes, lambda body: body.replace(victim, "MADD-ANO-XXX"))
    findings = notes_findings(edited, registry)
    assert any(f.startswith(f"{victim} is reachable") for f in findings), findings


def test_a_reachable_entry_deleted_from_the_notes_is_found(notes, registry):
    _deleting_an_entry_from_the_notes_is_found(notes, registry)


def test_it_is_found_after_an_assembly_names_a_new_entry_in_a_later_file(notes, registry):
    """A quiet-point assembly numbers the next entry and names it in the
    patch release's notes (``vX.Y.Z.md``) only, never in the ``.0`` notes."""
    newest = max(int(a["anomaly_id"][-3:]) for a in registry["anomalies"])
    new = f"MADD-ANO-{newest + 1:03d}"
    assembled = [*notes, ("v9.9.9.md", f"# x\n\n## Known anomalies\n\n- `{new}`: new.\n")]
    grown = _grown(registry, _entry(new))
    assert notes_findings(assembled, grown) == []
    _deleting_an_entry_from_the_notes_is_found(assembled, grown)


def test_only_a_changes_fragments_known_anomalies_text_covers_an_entry(tmp_path):
    """A changes fragment names an entry for the notes under its 'Known
    anomalies' heading; the same id under another heading covers nothing."""
    folder = tmp_path / "docs" / "fragments" / "changes"
    folder.mkdir(parents=True)
    (folder / "fix-x.yaml").write_text(
        'line: "0.4"\nrelease_notes:\n  Fixed: "MADD-ANO-998 is fixed."\n'
        '  Known anomalies: "MADD-ANO-999 remains."\n')
    _, named = fragment_view(tmp_path)
    assert named == {"MADD-ANO-999"}


def test_a_later_notes_file_covers_a_new_reachable_entry(notes, registry):
    later = ("v9.9.9.md", "# x\n\n## Known anomalies\n\n- `MADD-ANO-999`: new.\n")
    assert notes_findings([*notes, later], _grown(registry, _entry())) == []


def test_a_fragment_covers_a_reachable_entry_by_its_release_note(notes, registry):
    """A fragment entry is covered by its release_note; a numbered entry by
    a changes fragment's Known-anomalies text, until the assembler writes it."""
    label = "MADD-ANO-" "NEW-fix-x-1"
    assert notes_findings(notes, registry, [_entry(label, release_note="Do this.")]) == []
    assert notes_findings(notes, registry, [_entry(label)]) == [
        f"{label} is reachable and its fragment gives no release_note"]
    assert notes_findings(notes, _grown(registry, _entry()), (), {"MADD-ANO-999"}) == []


@pytest.mark.parametrize("status", sorted(UNREACHABLE_STATUSES))
def test_an_entry_the_first_notes_name_that_closed_must_be_named_again(
        notes, registry, status):
    grown = _grown(registry, _entry(status=status))
    listed = _edit_first(notes, lambda body: body + "\n- `MADD-ANO-999`: a synthetic entry.\n")
    first = listed[0][0]
    assert notes_findings(listed, grown) == [
        f"MADD-ANO-999 is named by {first} and is no longer reachable ({status}), and "
        f"no later notes file of the line names it again"]
    later = ("v9.9.9.md", "## Known anomalies\n\n- `MADD-ANO-999` is resolved.\n")
    assert notes_findings([*listed, later], grown) == []
    assert notes_findings(listed, grown, (), {"MADD-ANO-999"}) == []


def test_an_entry_the_registry_lacks_listed_in_the_notes_is_found(notes, registry):
    listed = _edit_first(notes, lambda body: body + "\n- `MADD-ANO-998`: no such entry.\n")
    assert notes_findings(listed, registry) == [
        f"{listed[0][0]} names MADD-ANO-998, which is not in the registry"]


def test_the_line_is_the_trees_or_the_newest_released_one():
    names = ["v0.3.1.md", "v0.4.0.md", "v0.4.1.md", "v0.4.2.md", "v0.2.md", "index.md"]
    assert line_notes(names, "0.4.2.dev0") == ["v0.4.0.md", "v0.4.1.md", "v0.4.2.md"]
    assert line_notes(names, "0.5.0.dev0") == ["v0.4.0.md", "v0.4.1.md", "v0.4.2.md"]
    assert line_notes(names + ["v0.5.0.md"], "0.5.0.dev0") == ["v0.5.0.md"]
    assert line_notes(names + ["v0.5.0.md"], "0.5.0") == ["v0.5.0.md"]
    assert line_notes(["v0.4.1.md"], "0.4.2.dev0") == []
    assert notes_findings([], {"anomalies": []}) == [
        "no notes file of this line or an older one has a .0 release"]


@pytest.mark.parametrize("status", ["open", *sorted(UNREACHABLE_STATUSES)])
def test_an_entry_the_index_omits_is_found(records, registry, status):
    assert index_findings(records, _grown(registry, _entry(status=status))) == [
        "MADD-ANO-999 is in the registry and not named in the index"]


def test_an_index_in_a_later_record_counts(records, registry):
    later = (f"{INDEX_HEADING}\n\n`MADD-ANO-999` — *x* — **open**.\n\n"
             f"{RESOLVED_MARKER}  `998` (resolved): y.\n")
    grown = _grown(registry, _entry(), _entry("MADD-ANO-998", "resolved"))
    assert index_findings([*records, later], grown) == []


def test_a_reachable_entry_filed_among_the_resolved_is_found(records, registry):
    victim = sorted(_reachable(registry["anomalies"]))[0]
    moved = []
    for record in records:
        if INDEX_HEADING in record:
            index = section(record, INDEX_HEADING)
            record = record.replace(index, index.replace(
                RESOLVED_MARKER, f"{RESOLVED_MARKER}  `{victim[-3:]}`,", 1))
        moved.append(record)
    findings = index_findings(moved, registry)
    assert any(f.startswith(f"{victim} is reachable") and "paragraph lists it" in f
               for f in findings), findings


def test_an_index_without_its_resolved_paragraph_is_refused(records, registry):
    stripped = [r.replace(RESOLVED_MARKER, "**Closed.**") for r in records]
    assert f"an index has no {RESOLVED_MARKER!r} paragraph" in index_findings(stripped, registry)
    assert index_findings(["# no index here\n"], registry) == [
        f"no development record has an {INDEX_HEADING!r} section"]


def test_a_section_at_the_end_of_a_file_is_read_to_the_end():
    text = "# T\n\n## Other\n\nx\n\n## Known anomalies\n\n- `MADD-ANO-001`\n"
    assert "MADD-ANO-001" in section(text, NOTES_HEADING)
    with pytest.raises(AssertionError):
        section(text, INDEX_HEADING)


def _at(registry, version):
    return {**registry, "maddening_version": version}


def test_the_count_is_checked_at_a_release_version_only(registry):
    true = fragments.reachable_count_line(_at(registry, "9.9.9"), "9.9.9")
    good = f"# C\n\n## [Unreleased]\n\n## [9.9.9] - 2026-12-01\n\n### Known Anomalies\n\n- {true}\n"
    assert changelog_count_findings(good, _at(registry, "9.9.9")) == []
    assert changelog_count_findings("# nothing\n", _at(registry, "9.9.9.dev0")) == []
    assert changelog_count_findings("# nothing\n", _at(registry, "9.9.9")) == [
        "the changelog has no [9.9.9] section"]


def test_a_stale_changelog_count_is_found(registry):
    true = fragments.reachable_count_line(_at(registry, "9.9.9"), "9.9.9")
    text = f"## [9.9.9] - 2026-12-01\n\n- {true}\n"
    grown = _at(_grown(registry, _entry(status="open")), "9.9.9")
    findings = changelog_count_findings(text, grown)
    assert len(findings) == 1 and findings[0].startswith("the changelog says"), findings
    other = _at(_grown(registry, _entry(status="wont_fix")), "9.9.9")
    assert changelog_count_findings(
        f"## [9.9.9] - 2026-12-01\n\n- {fragments.reachable_count_line(other, '9.9.9')}\n",
        other) == []


def test_a_changelog_without_the_count_is_found(registry):
    assert changelog_count_findings("## [9.9.9] - 2026-12-01\n\n- nothing\n",
                                    _at(registry, "9.9.9")) == [
        "the changelog's [9.9.9] section gives no count of reachable entries"]


def test_the_count_line_the_assembler_writes_is_the_one_read_here(registry):
    assert fragments.UNREACHABLE == UNREACHABLE_STATUSES
    line = fragments.reachable_count_line(_at(registry, "9.9.9"), "9.9.9")
    assert changelog_count_findings(f"## [9.9.9] - 2026-12-01\n{line}\n",
                                    _at(registry, "9.9.9")) == []
