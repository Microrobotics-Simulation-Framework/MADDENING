"""The release notes' "Known anomalies" section names every reachable entry.

``docs/release_notes/v0.4.0.md`` says that every entry of
``known_anomalies.yaml`` whose defect is reachable in 0.4.0 is listed in its
"Known anomalies" section.  The claim was false twice: first as a
hand-counted total that was stale by the time it was read, then as a list
that left out MADD-ANO-035 and never mentioned MADD-ANO-050 anywhere in the
document.  Nothing checked it.

This does.  The reachable set is read with ``scripts/check_anomalies.py``'s
own ``UNREACHABLE_STATUSES`` -- the rule the registry gate and the SOUP
headline already share -- so the notes cannot come to disagree with them
about which entries are live.  Three things are required of the section:

* every reachable entry is named by its full ID outside the closing
  "**Resolved.**" paragraph;
* that paragraph names no reachable entry, so a live defect cannot be filed
  among the closed ones;
* every registry entry is named somewhere in the section, full or by the
  paragraph's three-digit shorthand, so the section is an index of the
  whole registry.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTES = REPO_ROOT / "docs" / "release_notes" / "v0.4.0.md"
REGISTRY = REPO_ROOT / "docs" / "validation" / "known_anomalies.yaml"

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from check_anomalies import UNREACHABLE_STATUSES  # noqa: E402

SECTION_HEADING = "## Known anomalies"
RESOLVED_MARKER = "**Resolved.**"
_FULL_ID = re.compile(r"MADD-ANO-(\d{3})")
#: The closing paragraph's shorthand: "`MADD-ANO-006`, `019`, `020` ...".
_BARE_ID = re.compile(r"`(\d{3})`")


def known_anomalies_section(notes: str) -> str:
    """The text of the "## Known anomalies" section, up to the next ``## ``."""
    match = re.search(rf"^{re.escape(SECTION_HEADING)}\n(.*?)(?=^## )",
                      notes, re.S | re.M)
    if match is None:
        raise AssertionError(f"no {SECTION_HEADING!r} section in the notes")
    return match.group(1)


def _full_ids(text: str) -> set[str]:
    return {f"MADD-ANO-{n}" for n in _FULL_ID.findall(text)}


def _any_ids(text: str) -> set[str]:
    return _full_ids(text) | {f"MADD-ANO-{n}" for n in _BARE_ID.findall(text)}


def section_findings(notes: str, registry: dict) -> list[str]:
    """Every way the section fails the three requirements; empty if none."""
    section = known_anomalies_section(notes)
    if RESOLVED_MARKER not in section:
        return [f"the section has no {RESOLVED_MARKER!r} paragraph"]
    body, resolved_paragraph = section.split(RESOLVED_MARKER, 1)
    entries = {a["anomaly_id"]: a for a in registry["anomalies"]}
    reachable = {aid for aid, a in entries.items()
                 if a.get("resolution_status") not in UNREACHABLE_STATUSES}

    findings = []
    for aid in sorted(reachable - _full_ids(body)):
        a = entries[aid]
        findings.append(
            f"{aid} is reachable ({a.get('resolution_status')}) and the "
            f"section does not name it outside the resolved paragraph: "
            f"{a.get('title', '')[:70]}")
    for aid in sorted(reachable & _any_ids(resolved_paragraph)):
        findings.append(
            f"{aid} is reachable ({entries[aid].get('resolution_status')}) "
            f"and the {RESOLVED_MARKER!r} paragraph lists it")
    for aid in sorted(set(entries) - _any_ids(section)):
        findings.append(f"{aid} is in the registry and not named in the section")
    return findings


@pytest.fixture(scope="module")
def notes() -> str:
    return NOTES.read_text()


@pytest.fixture(scope="module")
def registry() -> dict:
    return yaml.safe_load(REGISTRY.read_text())


def test_the_known_anomalies_section_names_every_reachable_entry(notes, registry):
    assert section_findings(notes, registry) == []


def test_the_reachable_set_is_not_empty(registry):
    """A check over an empty set passes vacuously; 0.4.0 has open entries."""
    reachable = [a for a in registry["anomalies"]
                 if a.get("resolution_status") not in UNREACHABLE_STATUSES]
    assert reachable


# The check has to be able to fail.  Each case below breaks one requirement
# on a copy of the real inputs and expects the finding that names it.


def _open_entry(aid="MADD-ANO-999", status="open"):
    return {"anomaly_id": aid, "title": "a synthetic entry",
            "resolution_status": status}


@pytest.mark.parametrize("status", ["open", "partially_resolved", "wont_fix",
                                    "a status nobody enumerated"])
def test_a_reachable_entry_the_section_omits_is_found(notes, registry, status):
    grown = {**registry,
             "anomalies": [*registry["anomalies"], _open_entry(status=status)]}
    findings = section_findings(notes, grown)
    assert any(f.startswith("MADD-ANO-999 is reachable") for f in findings), findings


@pytest.mark.parametrize("status", sorted(UNREACHABLE_STATUSES))
def test_an_unreachable_entry_need_not_be_described(notes, registry, status):
    """Only the index requirement applies to a resolved or duplicate entry."""
    grown = {**registry,
             "anomalies": [*registry["anomalies"], _open_entry(status=status)]}
    assert section_findings(notes, grown) == [
        "MADD-ANO-999 is in the registry and not named in the section"]


def test_a_reachable_entry_deleted_from_the_section_is_found(notes, registry):
    reachable = sorted(a["anomaly_id"] for a in registry["anomalies"]
                       if a.get("resolution_status") not in UNREACHABLE_STATUSES)
    victim = reachable[-1]
    section = known_anomalies_section(notes)
    stripped = section.replace(victim, "MADD-ANO-XXX")
    findings = section_findings(notes.replace(section, stripped), registry)
    assert any(f.startswith(f"{victim} is reachable") for f in findings), findings


def test_a_reachable_entry_filed_among_the_resolved_is_found(notes, registry):
    reachable = sorted(a["anomaly_id"] for a in registry["anomalies"]
                       if a.get("resolution_status") not in UNREACHABLE_STATUSES)
    victim = reachable[0]
    moved = notes.replace(RESOLVED_MARKER, f"{RESOLVED_MARKER}  `{victim[-3:]}`,", 1)
    findings = section_findings(moved, registry)
    assert any(f.startswith(f"{victim} is reachable") and "paragraph lists it" in f
               for f in findings), findings


def test_a_section_without_its_resolved_paragraph_is_refused(notes, registry):
    findings = section_findings(notes.replace(RESOLVED_MARKER, "**Closed.**"),
                                registry)
    assert findings == [f"the section has no {RESOLVED_MARKER!r} paragraph"]
