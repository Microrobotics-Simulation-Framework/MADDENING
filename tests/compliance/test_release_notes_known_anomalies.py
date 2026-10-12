"""The release notes list the reachable anomalies; the development record indexes them all.

``docs/release_notes/v0.4.0.md`` says that its "Known anomalies" section
lists every entry of ``known_anomalies.yaml`` whose defect a user can reach in
0.4.0.  The claim was false twice: first as a hand-counted total that was stale
by the time it was read, then as a list that left out MADD-ANO-035 and never
mentioned MADD-ANO-050 anywhere in the document.  Nothing checked it.

This does.  The reachable set is read with ``scripts/check_anomalies.py``'s
own ``UNREACHABLE_STATUSES`` -- the rule the registry gate and the SOUP
headline already share -- so the notes cannot come to disagree with them
about which entries are live.

Of the notes' "Known anomalies" section two things are required:

* every reachable entry is named by its full ID;
* no other entry is named there, so a resolved defect is not listed as one
  a user can still meet.

The index of the whole registry lives in the development record,
``docs/validation/development_record_v0.4.0.md``, under "Index of the anomaly
registry".  Of that section two things are required:

* every registry entry is named, by its full ID or by the closing
  "**Resolved.**" paragraph's three-digit shorthand;
* that paragraph names no reachable entry, so a live defect cannot be filed
  among the closed ones.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTES = REPO_ROOT / "docs" / "release_notes" / "v0.4.0.md"
RECORD = REPO_ROOT / "docs" / "validation" / "development_record_v0.4.0.md"
REGISTRY = REPO_ROOT / "docs" / "validation" / "known_anomalies.yaml"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_anomalies import UNREACHABLE_STATUSES  # noqa: E402

NOTES_HEADING = "## Known anomalies"
INDEX_HEADING = "## Index of the anomaly registry"
RESOLVED_MARKER = "**Resolved.**"
_FULL_ID = re.compile(r"MADD-ANO-(\d{3})")
#: The closing paragraph's shorthand: "`MADD-ANO-006`, `019`, `020` ...".
_BARE_ID = re.compile(r"`(\d{3})`")


def section(text: str, heading: str) -> str:
    """The text of the ``heading`` section, up to the next ``## `` or the end."""
    match = re.search(rf"^{re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    if match is None:
        raise AssertionError(f"no {heading!r} section")
    return match.group(1)


def _full_ids(text: str) -> set[str]:
    return {f"MADD-ANO-{n}" for n in _FULL_ID.findall(text)}


def _any_ids(text: str) -> set[str]:
    return _full_ids(text) | {f"MADD-ANO-{n}" for n in _BARE_ID.findall(text)}


def _reachable(registry: dict) -> dict:
    return {a["anomaly_id"]: a for a in registry["anomalies"]
            if a.get("resolution_status") not in UNREACHABLE_STATUSES}


def notes_findings(notes: str, registry: dict) -> list[str]:
    """Every way the notes' section fails its two requirements; empty if none."""
    body = section(notes, NOTES_HEADING)
    entries = {a["anomaly_id"]: a for a in registry["anomalies"]}
    reachable = _reachable(registry)
    named = _full_ids(body)
    findings = []
    for aid in sorted(set(reachable) - named):
        a = entries[aid]
        findings.append(
            f"{aid} is reachable ({a.get('resolution_status')}) and the notes' "
            f"section does not name it: {a.get('title', '')[:70]}")
    for aid in sorted(named - set(reachable)):
        status = entries[aid].get("resolution_status") if aid in entries else "not in the registry"
        findings.append(f"{aid} is not reachable ({status}) and the notes' section names it")
    return findings


def index_findings(record: str, registry: dict) -> list[str]:
    """Every way the record's index fails its two requirements; empty if none."""
    index = section(record, INDEX_HEADING)
    if RESOLVED_MARKER not in index:
        return [f"the index has no {RESOLVED_MARKER!r} paragraph"]
    _, resolved_paragraph = index.split(RESOLVED_MARKER, 1)
    entries = {a["anomaly_id"]: a for a in registry["anomalies"]}
    reachable = _reachable(registry)
    findings = []
    for aid in sorted(set(reachable) & _any_ids(resolved_paragraph)):
        findings.append(
            f"{aid} is reachable ({entries[aid].get('resolution_status')}) "
            f"and the {RESOLVED_MARKER!r} paragraph lists it")
    for aid in sorted(set(entries) - _any_ids(index)):
        findings.append(f"{aid} is in the registry and not named in the index")
    return findings


@pytest.fixture(scope="module")
def notes() -> str:
    return NOTES.read_text()


@pytest.fixture(scope="module")
def record() -> str:
    return RECORD.read_text()


@pytest.fixture(scope="module")
def registry() -> dict:
    return yaml.safe_load(REGISTRY.read_text())


def test_the_notes_name_every_reachable_entry_and_no_other(notes, registry):
    assert notes_findings(notes, registry) == []


def test_the_record_indexes_every_registry_entry(record, registry):
    assert index_findings(record, registry) == []


def test_the_reachable_set_is_not_empty(registry):
    """A check over an empty set passes vacuously; 0.4.0 has open entries."""
    assert _reachable(registry)


# The checks have to be able to fail.  Each case below breaks one requirement
# on a copy of the real inputs and expects the finding that names it.


def _entry(aid="MADD-ANO-999", status="open"):
    return {"anomaly_id": aid, "title": "a synthetic entry",
            "resolution_status": status}


@pytest.mark.parametrize("status", ["open", "partially_resolved", "wont_fix",
                                    "a status nobody enumerated"])
def test_a_reachable_entry_the_notes_omit_is_found(notes, registry, status):
    grown = {**registry, "anomalies": [*registry["anomalies"], _entry(status=status)]}
    findings = notes_findings(notes, grown)
    assert any(f.startswith("MADD-ANO-999 is reachable") for f in findings), findings


def test_a_reachable_entry_deleted_from_the_notes_is_found(notes, registry):
    victim = sorted(_reachable(registry))[-1]
    body = section(notes, NOTES_HEADING)
    findings = notes_findings(notes.replace(body, body.replace(victim, "MADD-ANO-XXX")), registry)
    assert any(f.startswith(f"{victim} is reachable") for f in findings), findings


@pytest.mark.parametrize("status", sorted(UNREACHABLE_STATUSES))
def test_an_unreachable_entry_listed_in_the_notes_is_found(notes, registry, status):
    grown = {**registry, "anomalies": [*registry["anomalies"], _entry(status=status)]}
    body = section(notes, NOTES_HEADING)
    listed = notes.replace(body, body + "\n- `MADD-ANO-999`: a synthetic entry.\n")
    assert notes_findings(listed, grown) == [
        f"MADD-ANO-999 is not reachable ({status}) and the notes' section names it"]


def test_an_entry_the_registry_lacks_listed_in_the_notes_is_found(notes, registry):
    body = section(notes, NOTES_HEADING)
    listed = notes.replace(body, body + "\n- `MADD-ANO-998`: no such entry.\n")
    assert notes_findings(listed, registry) == [
        "MADD-ANO-998 is not reachable (not in the registry) and the notes' section names it"]


@pytest.mark.parametrize("status", ["open", *sorted(UNREACHABLE_STATUSES)])
def test_an_entry_the_index_omits_is_found(record, registry, status):
    grown = {**registry, "anomalies": [*registry["anomalies"], _entry(status=status)]}
    assert index_findings(record, grown) == [
        "MADD-ANO-999 is in the registry and not named in the index"]


def test_a_reachable_entry_filed_among_the_resolved_is_found(record, registry):
    victim = sorted(_reachable(registry))[0]
    index = section(record, INDEX_HEADING)
    moved = record.replace(index, index.replace(RESOLVED_MARKER, f"{RESOLVED_MARKER}  `{victim[-3:]}`,", 1))
    findings = index_findings(moved, registry)
    assert any(f.startswith(f"{victim} is reachable") and "paragraph lists it" in f
               for f in findings), findings


def test_an_index_without_its_resolved_paragraph_is_refused(record, registry):
    index = section(record, INDEX_HEADING)
    stripped = record.replace(index, index.replace(RESOLVED_MARKER, "**Closed.**"))
    assert index_findings(stripped, registry) == [
        f"the index has no {RESOLVED_MARKER!r} paragraph"]


def test_a_section_at_the_end_of_a_file_is_read_to_the_end():
    text = "# T\n\n## Other\n\nx\n\n## Known anomalies\n\n- `MADD-ANO-001`\n"
    assert "MADD-ANO-001" in section(text, NOTES_HEADING)
    with pytest.raises(AssertionError):
        section(text, INDEX_HEADING)
