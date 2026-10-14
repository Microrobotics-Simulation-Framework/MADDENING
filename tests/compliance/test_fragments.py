"""Changelog and registry fragments: the format, the merged view, the assembler,
the ratchet, the release gate and the forward merge between two release lines.

Every case runs on a throwaway git repository built by :func:`_repo`, shaped
like this one: ``pyproject.toml``, ``CHANGELOG.md``, the registry with three
entries, the high-water mark, a development record with its index, and the
release notes.  The gates and the assembler are run in-process.

No literal label appears in this file: ``scripts/check_fragments.py`` scans
every tracked file and refuses a label no fragment defines, so labels are
built by :func:`L` from two pieces.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_fragments  # noqa: E402
import fragments  # noqa: E402

_NEW = "MADD-ANO-" "NEW-"


def L(stem: str, n: int) -> str:
    """A label, built so that this file never spells one."""
    return f"{_NEW}{stem}-{n}"


@pytest.fixture(autouse=True)
def small_registry(monkeypatch):
    """The repositories here hold three entries, so the first id the
    assembler can assign is 4 (297 in the shipped registry)."""
    monkeypatch.setattr(fragments, "FRAGMENTS_FROM", 4)


def test_the_first_assignable_id_follows_the_shipped_registry(monkeypatch):
    """FRAGMENTS_FROM is the shipped high-water mark plus one, as it was when
    fragments arrived; every entry numbered from it on records its label,
    which scripts/check_fragments.py holds on the real tree."""
    import re

    source = (REPO_ROOT / "scripts" / "fragments.py").read_text()
    assert re.search(r"^FRAGMENTS_FROM = (\d+)$", source, re.M).group(1) == "297"
    monkeypatch.setattr(fragments, "FRAGMENTS_FROM", 297)
    text = (REPO_ROOT / fragments.REGISTRY).read_text()
    assert check_fragments.label_record_findings(text) == []


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location(
        "_gate_check_anomalies_fragments", REPO_ROOT / "scripts" / "check_anomalies.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# A small repository
# ---------------------------------------------------------------------------

def _git(root, *args):
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    assert done.returncode == 0, f"git {args}: {done.stderr}"
    return done.stdout


def _write(root, rel, text):
    path = Path(root) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _commit(root, message="change"):
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)


def entry(aid, status="open", *, versions=None, note="Avoid it; see the guide.",
          refs=("tests/test_probe.py::test_runs",), title=None, extra=""):
    """One entry in the registry's layout (a label or a number as its id)."""
    closed = status in ("resolved",)
    versions = versions or (">=0.4.0.dev0, <0.4.2" if closed else ">=0.4.0.dev0")
    lines = [f'  - anomaly_id: "{aid}"',
             f'    title: "{title or "Entry " + aid[-12:]}"',
             '    description: "Seeded."',
             '    severity: "minor"',
             '    safety_relevance: "not_safety_relevant"',
             '    safety_relevance_rationale: "Seeded."',
             '    affected_components:',
             '      - "maddening.nodes.heat.HeatNode"',
             f'    affected_versions: "{versions}"',
             f'    resolution_status: "{status}"']
    if closed:
        lines.append('    resolution_version: "0.4.2"')
    if refs:
        lines.append("    verification:")
        lines += [f'      - "{r}"' for r in refs]
    if note and aid.startswith(_NEW):
        lines.append(f'    release_note: "{note}"')
    if extra:
        lines.append(extra)
    return "\n".join(lines) + "\n"


def anomalies_fragment(stem, *entries_text, line="0.4"):
    return f'line: "{line}"\nanomalies:\n' + "\n".join(entries_text)


_HEADER = ('schema_version: "1.0"\nmaddening_version: "{version}"\n'
           'generated_date: "2026-10-14"\n\nanomalies:\n')

_RECORD = """# Development record

## Index of the anomaly registry

Every entry is named below.

`MADD-ANO-002` — *two* — **open**.

`MADD-ANO-003` — *three* — **open**.

**Resolved.**  `001`.

## Afterwards

Nothing.
"""

_INDEX = """# Release notes

```{toctree}
:maxdepth: 1

v0.4.1
v0.4.0
```
"""

_CHANGELOG = """# Changelog

## [Unreleased]

## [0.4.1] - 2026-10-13

### Fixed

- One fix.

## [0.4.0] - 2026-10-13

### Added

- Things.
"""


def _repo(tmp_path, version="0.4.2.dev0"):
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@example.org")
    _git(root, "config", "user.name", "T")
    _write(root, "pyproject.toml", f'[project]\nname = "t"\nversion = "{version}"\n')
    _write(root, "CHANGELOG.md", _CHANGELOG)
    _write(root, fragments.REGISTRY, _HEADER.format(version=version)
           + entry("MADD-ANO-001", "resolved", versions=">=0.1.0, <0.4.0",
                   title="one").replace('"0.4.2"', '"0.4.0"')
           + "\n" + entry("MADD-ANO-002", title="two") + "\n"
           + entry("MADD-ANO-003", title="three"))
    _write(root, fragments.HW_FILE,
           "_HIGHEST_ANOMALY_ID = 3\n_RETIRED_ANOMALY_IDS: dict = {}\n")
    _write(root, "tests/test_probe.py", "def test_runs():\n    assert True\n")
    _write(root, "docs/validation/development_record_v0.4.0.md", _RECORD)
    _write(root, "docs/release_notes/index.md", _INDEX)
    _write(root, "docs/release_notes/v0.4.0.md",
           "# v0.4.0\n\n## Known anomalies\n\n- `MADD-ANO-002`: two.\n"
           "- `MADD-ANO-003`: three.\n")
    _write(root, "docs/release_notes/v0.4.1.md", "# v0.4.1\n\n## Fixed\n\n- One fix.\n")
    _write(root, fragments.README, "Fragments.\n")
    _commit(root, "base")
    return root


def _merge_branch(root, branch, files, message=None):
    """Add ``files`` on ``branch`` and merge it into the current branch, as a PR."""
    here = _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    _git(root, "checkout", "-q", "-b", branch)
    for rel, text in files.items():
        _write(root, rel, text)
    _commit(root, message or f"work on {branch}")
    _git(root, "checkout", "-q", here)
    _git(root, "merge", "-q", "--no-ff", "-m", f"Merge {branch}", branch)


def _findings(root, base=None):
    _git(root, "add", "-A")
    return check_fragments.findings(root, base)


def _gate(gate, root, capsys, *extra):
    status = gate.main([str(root / fragments.REGISTRY), "--repo-root", str(root),
                        "--prefix", "MADD-ANO-", *extra])
    out = capsys.readouterr()
    return status, out.out, out.err


def _anomaly_file(stem):
    return f"{fragments.ANOMALIES}/{stem}.yaml"


def _changes_file(stem):
    return f"{fragments.CHANGES}/{stem}.yaml"


def _tree(root):
    """Every tracked and untracked file's bytes, for byte-identity checks."""
    files = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and ".git" not in path.relative_to(root).parts:
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------

class TestFormat:
    def test_a_good_pair_of_fragments_passes(self, tmp_path, gate, capsys):
        root = _repo(tmp_path)
        _write(root, _anomaly_file("fix-heat"), anomalies_fragment(
            "fix-heat", entry(L("fix-heat", 1)), entry(L("fix-heat", 2), "resolved")))
        _write(root, _changes_file("fix-heat"),
               'line: "0.4"\nchangelog:\n  Fixed:\n    - "A fix."\n'
               'release_notes:\n  Known anomalies: |\n    - Named.\n')
        _write(root, "tests/test_heat_pin.py",
               f'# pins {L("fix-heat", 1)}\ndef test_x():\n    pass\n')
        assert _findings(root) == []
        status, out, err = _gate(gate, root, capsys)
        assert status == 0, err
        assert "3 anomal(ies) and 2 in fragments" in out

    @pytest.mark.parametrize("name", ["Fix_Heat.yaml", "fix--heat.yaml", "fix-heat.yml",
                                      "-fix.yaml", "nested/fix.yaml", "fix-heat.md"])
    def test_a_bad_file_name_is_refused(self, tmp_path, name):
        root = _repo(tmp_path)
        _write(root, f"{fragments.CHANGES}/{name}",
               'line: "0.4"\nchangelog:\n  Fixed:\n    - "x"\n')
        found = _findings(root)
        assert any("not a fragment" in f for f in found), found

    def test_a_label_carrying_another_stem_is_refused(self, tmp_path):
        root = _repo(tmp_path)
        _write(root, _anomaly_file("fix-heat"),
               anomalies_fragment("fix-heat", entry(L("fix-other", 1))))
        found = _findings(root)
        assert any("not this file's 'fix-heat'" in f for f in found), found

    def test_a_label_given_twice_is_refused(self, tmp_path, gate, capsys):
        root = _repo(tmp_path)
        _write(root, _anomaly_file("fix-heat"), anomalies_fragment(
            "fix-heat", entry(L("fix-heat", 1)), entry(L("fix-heat", 1), title="again")))
        assert any("is given twice" in f for f in _findings(root))
        status, _, err = _gate(gate, root, capsys)
        assert status == 1 and "Duplicate anomaly_id" in err

    def test_a_number_in_a_fragment_is_refused(self, tmp_path, gate, capsys):
        root = _repo(tmp_path)
        _write(root, _anomaly_file("fix-heat"),
               anomalies_fragment("fix-heat", entry("MADD-ANO-004")))
        assert any("MADD-ANO-004 is a number" in f for f in _findings(root))
        status, _, err = _gate(gate, root, capsys)
        assert status == 1 and "'MADD-ANO-004' is not a label" in err

    def test_a_label_in_the_registry_is_refused(self, tmp_path, gate, capsys):
        root = _repo(tmp_path)
        reg = root / fragments.REGISTRY
        reg.write_text(reg.read_text() + "\n" + entry(L("fix-heat", 1)).replace(
            '    release_note: "Avoid it; see the guide."\n', ""))
        assert any("is a label in the registry" in f for f in _findings(root))
        status, _, err = _gate(gate, root, capsys)
        assert status == 1 and "is a provisional label in the registry" in err

    def test_an_unknown_key_is_refused(self, tmp_path, gate, capsys):
        root = _repo(tmp_path)
        _write(root, _anomaly_file("fix-heat"), anomalies_fragment(
            "fix-heat", entry(L("fix-heat", 1), extra='    workaroud: "typo"')) + "notes: x\n")
        assert any("unknown key 'notes'" in f for f in _findings(root))
        status, _, err = _gate(gate, root, capsys)
        assert status == 1
        assert f"{_anomaly_file('fix-heat')}: {L('fix-heat', 1)}: unknown field 'workaroud'" in err

    @pytest.mark.parametrize("body, needle", [
        ('changelog:\n  Fixes:\n    - "x"\n', "changelog section 'Fixes'"),
        ('release_notes:\n  Highlights: "x"\n', "release-notes heading 'Highlights'"),
        ('changelog:\n  Fixed: []\n', "must be a list of non-empty bullets"),
        ('other: 1\n', "neither 'changelog:' nor 'release_notes:'"),
    ])
    def test_an_unknown_section_or_heading_is_refused(self, tmp_path, body, needle):
        root = _repo(tmp_path)
        _write(root, _changes_file("fix-heat"), 'line: "0.4"\n' + body)
        found = _findings(root)
        assert any(needle in f for f in found), found

    @pytest.mark.parametrize("line, needle", [
        (None, "no 'line:'"), ('"0.5"', "newer than this tree's line 0.4"),
        ('"0.3"', "not a release line"), ("0.4", "not a release line"),
    ])
    def test_a_missing_or_newer_line_is_refused(self, tmp_path, line, needle):
        root = _repo(tmp_path)
        head = f"line: {line}\n" if line else ""
        _write(root, _changes_file("fix-heat"), head + 'changelog:\n  Fixed:\n    - "x"\n')
        found = _findings(root)
        assert any(needle in f for f in found), found

    # Each spelling in two pieces, so that this file holds none of them.
    @pytest.mark.parametrize("spelling", [
        "madd-ano-" "new-fix-heat-1", "MADD_ANO_" "NEW_fix_heat_1", "MADD-ANO-" "NEW-Fix-heat-1",
        "MADD-ANO-" "NEW_fix-heat-1", "MADD" "ANONEW-fix-heat-1"])
    def test_a_variant_spelling_is_refused(self, tmp_path, spelling):
        root = _repo(tmp_path)
        _write(root, _anomaly_file("fix-heat"),
               anomalies_fragment("fix-heat", entry(L("fix-heat", 1))))
        _write(root, "tests/test_pin.py", f"# {spelling}\n")
        found = _findings(root)
        assert any("is not the label spelling" in f for f in found), found

    def test_a_label_in_a_file_name_is_refused(self, tmp_path):
        root = _repo(tmp_path)
        _write(root, _anomaly_file("fix-heat"),
               anomalies_fragment("fix-heat", entry(L("fix-heat", 1))))
        _write(root, f"tests/test_{L('fix-heat', 1)}.py", "x = 1\n")
        found = _findings(root)
        assert any("a label in a file name" in f for f in found), found

    def test_a_reachable_entry_needs_a_release_note(self, tmp_path):
        root = _repo(tmp_path)
        _write(root, _anomaly_file("fix-heat"), anomalies_fragment(
            "fix-heat", entry(L("fix-heat", 1), note=None), entry(L("fix-heat", 2), "resolved",
                                                                 note=None)))
        found = _findings(root)
        assert any(f"{L('fix-heat', 1)} is reachable" in f for f in found), found
        assert not any(L("fix-heat", 2) in f for f in found), found

    def test_a_label_no_fragment_defines_is_refused(self, tmp_path):
        root = _repo(tmp_path)
        _write(root, "tests/test_pin.py", f"# {L('fix-heat', 7)}\n")
        found = _findings(root)
        assert any("names no label any fragment defines" in f for f in found), found

    def test_an_id_line_in_another_spelling_is_refused(self, tmp_path):
        root = _repo(tmp_path)
        text = anomalies_fragment("fix-heat", entry(L("fix-heat", 1))).replace(
            f'  - anomaly_id: "{L("fix-heat", 1)}"', f"  - anomaly_id: '{L('fix-heat', 1)}'")
        _write(root, _anomaly_file("fix-heat"), text)
        assert any("id lines spell" in f for f in _findings(root))


# ---------------------------------------------------------------------------
# The merged view
# ---------------------------------------------------------------------------

class TestTheMergedView:
    def test_a_dead_reference_in_a_fragment_fails_the_registry_gate(self, tmp_path, gate,
                                                                    capsys):
        root = _repo(tmp_path)
        _write(root, _anomaly_file("fix-heat"), anomalies_fragment("fix-heat", entry(
            L("fix-heat", 1), refs=("tests/test_probe.py::test_gone",))))
        status, _, err = _gate(gate, root, capsys)
        assert status == 1
        assert (f"{_anomaly_file('fix-heat')}: {L('fix-heat', 1)}: verification entry "
                f"'tests/test_probe.py::test_gone'") in err

    def test_a_resolution_without_evidence_fails(self, tmp_path, gate, capsys):
        root = _repo(tmp_path)
        _write(root, _anomaly_file("fix-heat"), anomalies_fragment(
            "fix-heat", entry(L("fix-heat", 1), "resolved", refs=())))
        status, _, err = _gate(gate, root, capsys)
        assert status == 1
        assert f"{L('fix-heat', 1)}: resolution_status is 'resolved' but the verification" in err

    def test_a_bad_range_fails(self, tmp_path, gate, capsys):
        root = _repo(tmp_path)
        _write(root, _anomaly_file("fix-heat"), anomalies_fragment(
            "fix-heat", entry(L("fix-heat", 1), versions=">=0.1.5")))
        status, _, err = _gate(gate, root, capsys)
        assert status == 1 and f"{L('fix-heat', 1)}: affected_versions FIRST 0.1.5" in err

    def test_labels_are_not_held_to_contiguous_numbers(self, tmp_path, gate, capsys):
        root = _repo(tmp_path)
        _write(root, _anomaly_file("fix-heat"), anomalies_fragment(
            "fix-heat", entry(L("fix-heat", 3)), entry(L("fix-heat", 9))))
        assert _gate(gate, root, capsys)[0] == 0

    def test_a_fragment_entry_at_a_release_version_fails(self, tmp_path, gate, capsys):
        root = _repo(tmp_path, version="0.4.2")
        _write(root, _anomaly_file("fix-heat"),
               anomalies_fragment("fix-heat", entry(L("fix-heat", 1))))
        status, _, err = _gate(gate, root, capsys)
        assert status == 1 and "still a fragment entry at release version 0.4.2" in err

    def test_a_registry_elsewhere_reads_no_fragments(self, tmp_path, gate, capsys):
        """A copy of a registry outside its tree is checked alone, unless asked."""
        root = _repo(tmp_path)
        _write(root, _anomaly_file("fix-heat"), anomalies_fragment("fix-heat", entry(
            L("fix-heat", 1), refs=("tests/test_probe.py::test_gone",))))
        copy = tmp_path / "known_anomalies.yaml"
        copy.write_text((root / fragments.REGISTRY).read_text())
        assert gate.main([str(copy), "--repo-root", str(root)]) == 0
        assert gate.main([str(copy), "--repo-root", str(root),
                          "--fragments-root", str(root)]) == 1
        capsys.readouterr()


class TestTheCrossLineAllowance:
    RELEASED = ("0.1.0", "0.2.0", "0.3.0", "0.3.1", "0.4.0", "0.4.1")

    def _errors(self, gate, version, **fields):
        a = {"anomaly_id": "MADD-ANO-001", "resolution_status": "open",
             "affected_versions": ">=0.4.0"}
        a.update(fields)
        return gate.version_range_errors(
            {"maddening_version": version, "anomalies": [a]}, released=self.RELEASED)

    def test_it_admits_the_next_patch_of_an_older_open_line(self, gate):
        assert self._errors(gate, "0.5.0.dev0", affected_versions=">=0.4.2.dev0") == []
        assert self._errors(gate, "0.5.0.dev0", resolution_status="resolved",
                            resolution_version="0.4.2",
                            affected_versions=">=0.4.0, <0.4.2") == []

    def test_it_refuses_a_line_that_is_not_open(self, gate):
        found = self._errors(gate, "0.5.0.dev0", resolution_status="resolved",
                             resolution_version="0.3.2", affected_versions=">=0.3.0, <0.3.2")
        assert any("0.3.2 is not a version MADDENING released" in e for e in found), found

    def test_it_refuses_a_patch_beyond_the_next(self, gate):
        found = self._errors(gate, "0.5.0.dev0", affected_versions=">=0.4.3.dev0")
        assert any("FIRST 0.4.3.dev0" in e for e in found), found

    def test_it_is_off_at_a_release_version(self, gate):
        found = self._errors(gate, "0.5.0", affected_versions=">=0.4.2.dev0")
        assert any("FIRST 0.4.2.dev0" in e for e in found), found

    def test_it_follows_release_lines(self, gate):
        from packaging.version import Version

        released = frozenset(Version(v) for v in self.RELEASED)
        assert gate.cross_line_versions(Version("0.5.0.dev0"), released, Version) == {
            Version("0.4.2")}
        assert gate.cross_line_versions(Version("0.4.2.dev0"), released, Version) == set()
        assert gate.cross_line_versions(Version("0.6.0.dev0"), released, Version,
                                        lines=("0.4", "0.5", "0.6")) == {Version("0.4.2")}


# ---------------------------------------------------------------------------
# The assembler
# ---------------------------------------------------------------------------

def _two_fragments(root):
    """Two PRs merged in the order b then a; a's entry names b's label, and a
    test, a claims inventory and a changes fragment name a's."""
    _merge_branch(root, "fix/b", {
        _anomaly_file("fix-b"): anomalies_fragment(
            "fix-b", entry(L("fix-b", 1), title="b one"),
            entry(L("fix-b", 2), "resolved", title="b two", note=None))})
    _merge_branch(root, "fix/a", {
        _anomaly_file("fix-a"): anomalies_fragment("fix-a", entry(
            L("fix-a", 1), title="a one",
            note=f"Like {L('fix-b', 1)}: avoid it.")),
        "tests/test_a.py": f'"""Pins {L("fix-a", 1)}."""\ndef test_a():\n    pass\n',
        "docs/validation/x_claims.yaml": f"rows:\n  - id: X-001\n    sources: [{L('fix-a', 1)}]\n",
        _changes_file("fix-a"): ('line: "0.4"\nchangelog:\n  Known Anomalies:\n'
                                 f'    - "{L("fix-a", 1)}: avoid it."\n'),
    })


class TestTheAssembler:
    def test_it_numbers_in_merge_order_and_moves_the_mark_by_k(self, tmp_path):
        root = _repo(tmp_path)
        _two_fragments(root)
        mapping = fragments.assemble(root, soup=False, out=lambda *_: None)
        assert mapping == {L("fix-b", 1): "MADD-ANO-004", L("fix-b", 2): "MADD-ANO-005",
                           L("fix-a", 1): "MADD-ANO-006"}
        reg = (root / fragments.REGISTRY).read_text()
        ids = [a["anomaly_id"] for a in yaml.safe_load(reg)["anomalies"]]
        assert ids == [f"MADD-ANO-{n:03d}" for n in range(1, 7)]
        assert fragments.high_water(root) == 6
        assert fragments.recorded_labels(reg) == {lab: [aid] for lab, aid in mapping.items()}
        assert '  - anomaly_id: "MADD-ANO-004"\n    # label: NEW-fix-b-1\n' in reg
        assert "release_note" not in reg
        assert not (root / _anomaly_file("fix-a")).exists()
        assert not (root / _anomaly_file("fix-b")).exists()

    def test_it_rewrites_every_label_and_writes_the_index_and_the_notes(self, tmp_path):
        root = _repo(tmp_path)
        _two_fragments(root)
        fragments.assemble(root, soup=False, out=lambda *_: None)
        assert "MADD-ANO-006" in (root / "tests/test_a.py").read_text()
        assert "MADD-ANO-006" in (root / "docs/validation/x_claims.yaml").read_text()
        changes = (root / _changes_file("fix-a")).read_text()
        assert "MADD-ANO-006: avoid it." in changes and _NEW not in changes
        reg = (root / fragments.REGISTRY).read_text()
        assert _NEW not in reg
        notes = (root / "docs/release_notes/v0.4.2.md").read_text()
        assert "- `MADD-ANO-004`: Avoid it; see the guide." in notes
        assert "- `MADD-ANO-006`: Like MADD-ANO-004: avoid it." in notes
        assert "MADD-ANO-005" not in notes            # resolved, no release_note
        assert "v0.4.2\nv0.4.1\nv0.4.0" in (root / fragments.NOTES_INDEX).read_text()
        record = (root / "docs/validation/development_record_v0.4.0.md").read_text()
        live, closed = record.split("**Resolved.**")
        assert "`MADD-ANO-004` — *b one* — **open**." in live
        assert "`005` (resolved): b two." in closed.split("## Afterwards")[0]
        assert check_fragments.findings(root) == []

    def test_the_merged_tree_passes_the_registry_gate(self, tmp_path, gate, capsys):
        root = _repo(tmp_path)
        _two_fragments(root)
        assert _gate(gate, root, capsys)[0] == 0
        fragments.assemble(root, soup=False, out=lambda *_: None)
        status, out, err = _gate(gate, root, capsys)
        assert status == 0, err
        assert "6 anomal(ies) and 0 in fragments" in out

    def test_a_second_run_changes_nothing(self, tmp_path):
        root = _repo(tmp_path)
        _two_fragments(root)
        fragments.assemble(root, soup=False, out=lambda *_: None)
        _commit(root, "assembly")
        before = _tree(root)
        assert fragments.assemble(root, soup=False, out=lambda *_: None) == {}
        assert _tree(root) == before

    def test_a_resurrected_fragment_is_refused(self, tmp_path):
        root = _repo(tmp_path)
        _two_fragments(root)
        old = (root / _anomaly_file("fix-b")).read_text()
        fragments.assemble(root, soup=False, out=lambda *_: None)
        _commit(root, "assembly")
        _write(root, _anomaly_file("fix-b"), old)
        _commit(root, "a stale merge brings it back")
        found = check_fragments.findings(root)
        assert any(f"{L('fix-b', 1)} is already assembled as MADD-ANO-004" in f
                   for f in found), found
        with pytest.raises(fragments.AssemblyError, match="check_fragments.py is red"):
            fragments.assemble(root, soup=False, out=lambda *_: None)

    @pytest.mark.parametrize("break_it, needle", [
        ("dirty", "uncommitted changes"), ("untracked", "untracked fragments"),
        ("red", "check_fragments.py is red"), ("mark", "_HIGHEST_ANOMALY_ID is 9"),
    ])
    def test_it_refuses_to_start(self, tmp_path, break_it, needle):
        root = _repo(tmp_path)
        _two_fragments(root)
        if break_it == "dirty":
            (root / "CHANGELOG.md").write_text("edited\n")
        elif break_it == "untracked":
            _write(root, _changes_file("fix-c"), 'line: "0.4"\nchangelog:\n  Fixed:\n    - "x"\n')
        elif break_it == "red":
            _write(root, "tests/test_pin.py", f"# {L('nobody', 1)}\n")
            _commit(root)
        else:
            hw = root / fragments.HW_FILE
            hw.write_text(hw.read_text().replace("= 3", "= 9"))
            _commit(root)
        before = _tree(root)
        with pytest.raises(fragments.AssemblyError, match=needle):
            fragments.assemble(root, soup=False, out=lambda *_: None)
        assert _tree(root) == before

    def test_off_the_numbering_line_it_defers_anomalies_and_leaves_other_lines(self, tmp_path):
        root = _repo(tmp_path, version="0.5.0.dev0")
        _merge_branch(root, "fix/old", {_anomaly_file("fix-old"): anomalies_fragment(
            "fix-old", entry(L("fix-old", 1)), line="0.4")})
        _merge_branch(root, "feat/new", {
            _anomaly_file("feat-new"): anomalies_fragment(
                "feat-new", entry(L("feat-new", 1)), line="0.5"),
            _changes_file("feat-new"): ('line: "0.5"\nchangelog:\n  Known Anomalies:\n'
                                        f'    - "{L("feat-new", 1)}: new."\n'),
            _changes_file("feat-other"): 'line: "0.5"\nchangelog:\n  Added:\n    - "A thing."\n',
        })
        with pytest.raises(fragments.AssemblyError, match="numbering line only"):
            fragments.assemble(root, soup=False, out=lambda *_: None)
        # At the release version itself the leftovers fail the assembler's own
        # release gate, and nothing is written: 0.5.0 waits for numbering.
        root.joinpath("pyproject.toml").write_text('[project]\nname = "t"\nversion = "0.5.0"\n')
        _commit(root, "version 0.5.0")
        before = _tree(root)
        with pytest.raises(fragments.AssemblyError, match="release gate: .*never assembled"):
            fragments.assemble(root, release="0.5.0", date="2026-12-01", soup=False,
                               out=lambda *_: None)
        assert _tree(root) == before
        # Before the bump, the same run writes what it can and defers the rest.
        root.joinpath("pyproject.toml").write_text(
            '[project]\nname = "t"\nversion = "0.5.0.dev0"\n')
        _commit(root, "back to 0.5.0.dev0")
        said = []
        mapping = fragments.assemble(root, release="0.5.0", date="2026-12-01", soup=False,
                                     out=said.append)
        assert mapping == {}
        assert (root / _anomaly_file("fix-old")).exists()            # another line's
        assert (root / _anomaly_file("feat-new")).exists()           # deferred
        assert (root / _changes_file("feat-new")).exists()           # names a deferred label
        assert not (root / _changes_file("feat-other")).exists()     # assembled
        assert any("feat-new.yaml: anomalies are numbered on the numbering line only" in s
                   for s in said), said
        changelog = (root / "CHANGELOG.md").read_text()
        assert "## [0.5.0] - 2026-12-01\n\n### Added\n\n- A thing.\n" in changelog
        assert _NEW not in changelog

    def test_a_release_writes_the_section_the_count_the_notes_and_the_toctree(self, tmp_path):
        root = _repo(tmp_path)
        _two_fragments(root)
        _merge_branch(root, "docs/c", {_changes_file("docs-c"): (
            'line: "0.4"\nchangelog:\n  Fixed:\n    - "**A fix.** Do nothing."\n'
            '  Added:\n    - "A thing."\nrelease_notes:\n  Fixed: |\n    - The long text.\n'
            '  For contributors: "Tooling."\n')})
        root.joinpath("pyproject.toml").write_text('[project]\nname = "t"\nversion = "0.4.2"\n')
        _commit(root, "version 0.4.2")
        fragments.assemble(root, release="0.4.2", date="2026-11-01", soup=False,
                           out=lambda *_: None)
        changelog = (root / "CHANGELOG.md").read_text()
        head = changelog.index("## [0.4.2] - 2026-11-01")
        assert changelog.index("## [Unreleased]") < head < changelog.index("## [0.4.1]")
        section = changelog[head:changelog.index("## [0.4.1]")]
        assert section.index("### Added") < section.index("### Fixed") < section.index(
            "### Known Anomalies")
        assert ("- 4 registry entries are reachable in 0.4.2: 4 open and 0 partly resolved."
                in section)
        assert "- MADD-ANO-006: avoid it." in section
        assert changelog.split("## [Unreleased]")[1].split("## [")[0].strip() == ""
        notes = (root / "docs/release_notes/v0.4.2.md").read_text()
        assert notes.startswith("---\norphan: false\n---\n\n# What's new in v0.4.2")
        assert notes.index("## Fixed") < notes.index("## Known anomalies") < notes.index(
            "## For contributors")
        assert "- The long text." in notes and "Tooling." in notes
        assert list((root / fragments.FRAGMENTS).rglob("*.yaml")) == []
        assert check_fragments.findings(root) == []                   # the release gate

    def test_a_failed_check_restores_every_path(self, tmp_path, monkeypatch):
        """The assembler checks its own output; a label left behind is caught,
        and the tree is put back as it was."""
        root = _repo(tmp_path)
        _two_fragments(root)
        before = _tree(root)
        monkeypatch.setattr(fragments, "rewrite_labels", lambda text, mapping: (text, 0))
        with pytest.raises(fragments.AssemblyError, match="an assigned label survives"):
            fragments.assemble(root, soup=False, out=lambda *_: None)
        assert _tree(root) == before

    def test_an_entry_that_does_not_come_out_whole_is_caught(self, tmp_path, monkeypatch):
        root = _repo(tmp_path)
        _two_fragments(root)
        before = _tree(root)
        monkeypatch.setattr(fragments, "drop_extra_keys", lambda block: block[:-1])
        with pytest.raises(fragments.AssemblyError, match="appended entries are not"):
            fragments.assemble(root, soup=False, out=lambda *_: None)
        assert _tree(root) == before

    def test_relabel_rewrites_what_a_merge_brought_in(self, tmp_path):
        root = _repo(tmp_path)
        _two_fragments(root)
        fragments.assemble(root, soup=False, out=lambda *_: None)
        _commit(root, "assembly")
        _write(root, "tests/test_late.py", f"# {L('fix-a', 1)} and {L('fix-b', 2)}\n")
        _commit(root, "a branch that still says the label")
        assert any("run scripts/fragments.py relabel" in f
                   for f in check_fragments.findings(root))
        assert fragments.relabel(root) == [("tests/test_late.py", 2)]
        assert (root / "tests/test_late.py").read_text() == "# MADD-ANO-006 and MADD-ANO-005\n"
        assert fragments.relabel(root) == []
        assert _findings(root) == []


# ---------------------------------------------------------------------------
# The ratchet, the release gate
# ---------------------------------------------------------------------------

class TestTheRatchet:
    def _merged(self, tmp_path):
        root = _repo(tmp_path)
        _merge_branch(root, "fix/b", {_anomaly_file("fix-b"): anomalies_fragment(
            "fix-b", entry(L("fix-b", 1)), entry(L("fix-b", 2)))})
        return root, _git(root, "rev-parse", "HEAD").strip()

    def test_deleting_a_merged_fragment_fails(self, tmp_path):
        root, base = self._merged(tmp_path)
        (root / _anomaly_file("fix-b")).unlink()
        _commit(root, "drop it")
        for line in (base, None):                     # the base's history, or HEAD's
            found = check_fragments.findings(root, line)
            assert any(f"{L('fix-b', 1)} came into the line in {base[:10]}" in f
                       for f in found), found

    def test_deleting_one_entry_of_a_merged_fragment_fails(self, tmp_path):
        root, base = self._merged(tmp_path)
        _write(root, _anomaly_file("fix-b"),
               anomalies_fragment("fix-b", entry(L("fix-b", 1))))
        _commit(root, "drop one")
        found = check_fragments.findings(root, base)
        assert found and all(L("fix-b", 2) in f for f in found), found

    def test_closing_or_assembling_it_passes(self, tmp_path):
        root, base = self._merged(tmp_path)
        _write(root, _anomaly_file("fix-b"), anomalies_fragment(
            "fix-b", entry(L("fix-b", 1), "duplicate", note=None), entry(L("fix-b", 2))))
        _commit(root, "close one")
        assert check_fragments.findings(root, base) == []
        fragments.assemble(root, soup=False, out=lambda *_: None)
        _commit(root, "assembly")
        assert check_fragments.findings(root, base) == []

    def test_an_unreadable_base_fails_closed(self, tmp_path):
        root, _ = self._merged(tmp_path)
        found = check_fragments.findings(root, "no-such-ref")
        assert any("the ratchet cannot check" in f and "no-such-ref" in f
                   for f in found), found


class TestTheReleaseGate:
    @pytest.mark.parametrize("leftover", ["fragment", "label", "unreleased", "notes"])
    def test_a_leftover_fails_at_a_release_version_and_passes_at_dev(self, tmp_path,
                                                                      leftover):
        for version, red in (("0.4.2", True), ("0.4.2.dev0", False)):
            root = _repo(tmp_path / version, version=version)
            _write(root, "docs/release_notes/v0.4.2.md", "# v0.4.2\n")
            if leftover == "fragment":
                _write(root, _changes_file("fix-c"),
                       'line: "0.4"\nchangelog:\n  Fixed:\n    - "x"\n')
            elif leftover == "label":
                _write(root, _anomaly_file("fix-c"),
                       anomalies_fragment("fix-c", entry(L("fix-c", 1))))
            elif leftover == "unreleased":
                ch = root / "CHANGELOG.md"
                ch.write_text(ch.read_text().replace(
                    "## [Unreleased]\n", "## [Unreleased]\n\n- by hand\n"))
            else:
                (root / "docs/release_notes/v0.4.2.md").unlink()
            found = _findings(root)
            needle = {"fragment": "never assembled", "label": "a provisional label at release",
                      "unreleased": "'## [Unreleased]' is not empty",
                      "notes": "v0.4.2.md is missing"}[leftover]
            assert any(needle in f for f in found) is red, (version, found)
            if not red:
                assert found == [], found

    def test_the_readme_alone_passes_at_a_release_version(self, tmp_path):
        root = _repo(tmp_path, version="0.4.2")
        _write(root, "docs/release_notes/v0.4.2.md", "# v0.4.2\n")
        assert _findings(root) == []


class TestEveryAssembledEntryRecordsItsLabel:
    def test_a_hand_numbered_entry_fails(self, tmp_path):
        root = _repo(tmp_path)
        reg = root / fragments.REGISTRY
        reg.write_text(reg.read_text() + "\n" + entry("MADD-ANO-004", title="by hand"))
        hw = root / fragments.HW_FILE
        hw.write_text(hw.read_text().replace("= 3", "= 4"))
        found = _findings(root)
        assert any("MADD-ANO-004 is numbered from 4 on and carries 0" in f
                   for f in found), found

    def test_a_label_recorded_twice_or_on_an_old_entry_fails(self, tmp_path):
        root = _repo(tmp_path)
        reg = root / fragments.REGISTRY
        text = reg.read_text().replace(
            '  - anomaly_id: "MADD-ANO-002"\n',
            '  - anomaly_id: "MADD-ANO-002"\n    # label: NEW-fix-b-1\n').replace(
            '  - anomaly_id: "MADD-ANO-003"\n',
            '  - anomaly_id: "MADD-ANO-003"\n    # label: NEW-fix-b-1\n')
        reg.write_text(text)
        found = _findings(root)
        assert any("predates fragments" in f for f in found), found
        assert any("is recorded 2 times" in f for f in found), found

    def test_a_label_comment_in_another_layout_fails(self, tmp_path):
        found = check_fragments.label_record_findings(
            '  - anomaly_id: "MADD-ANO-004"\n    #label:NEW-fix-b-1\n')
        assert any("not written" in f for f in found), found


# ---------------------------------------------------------------------------
# Seeded faults on an assembled tree: each must turn something red
# ---------------------------------------------------------------------------

class TestSeededFaults:
    @pytest.fixture
    def assembled(self, tmp_path):
        root = _repo(tmp_path)
        _two_fragments(root)
        base = _git(root, "rev-parse", "HEAD").strip()
        fragments.assemble(root, soup=False, out=lambda *_: None)
        _commit(root, "assembly")
        return root, base

    def test_a_dropped_fragment(self, tmp_path):
        root = _repo(tmp_path)
        _two_fragments(root)
        base = _git(root, "rev-parse", "HEAD").strip()
        (root / _anomaly_file("fix-b")).unlink()
        _commit(root, "drop")
        assert any("a merged fragment, or an entry of one, was deleted" in f
                   for f in check_fragments.findings(root, base))

    def test_a_duplicated_entry(self, assembled, gate, capsys):
        root, _ = assembled
        reg = root / fragments.REGISTRY
        text = reg.read_text()
        block = text[text.index('  - anomaly_id: "MADD-ANO-006"'):]
        reg.write_text(text.rstrip("\n") + "\n\n" + block)
        assert any("recorded 2 times" in f for f in _findings(root))
        status, _, err = _gate(gate, root, capsys)
        assert status == 1 and "Duplicate anomaly_id: MADD-ANO-006" in err

    def test_a_label_left_behind(self, assembled):
        root, _ = assembled
        test = root / "tests/test_a.py"
        test.write_text(test.read_text().replace("MADD-ANO-006", L("fix-a", 1)))
        assert any("was assembled as MADD-ANO-006; run scripts/fragments.py relabel" in f
                   for f in _findings(root))

    def test_a_skipped_high_water_bump(self, assembled):
        root, _ = assembled
        hw = root / fragments.HW_FILE
        hw.write_text(hw.read_text().replace("= 6", "= 3"))
        _commit(root, "skip the bump")
        ids = {a["anomaly_id"] for a in yaml.safe_load(
            (root / fragments.REGISTRY).read_text())["anomalies"]}
        # tests/compliance/test_soup_evidence.py's rule: the ids are 1..HW
        assert ids != {f"MADD-ANO-{n:03d}" for n in range(1, fragments.high_water(root) + 1)}
        with pytest.raises(fragments.AssemblyError, match="_HIGHEST_ANOMALY_ID is 3"):
            fragments.assemble(root, soup=False, out=lambda *_: None)

    def test_a_second_run_over_a_resurrected_fragment(self, assembled):
        root, base = assembled
        _git(root, "checkout", "-q", base, "--", _anomaly_file("fix-a"))
        _commit(root, "the fragment comes back")
        assert any("already assembled as MADD-ANO-006" in f
                   for f in check_fragments.findings(root))
        with pytest.raises(fragments.AssemblyError, match="check_fragments.py is red"):
            fragments.assemble(root, soup=False, out=lambda *_: None)

    def test_a_fragment_left_at_a_release_version(self, tmp_path):
        root = _repo(tmp_path, version="0.4.2")
        _write(root, "docs/release_notes/v0.4.2.md", "# v0.4.2\n")
        _write(root, _changes_file("late"), 'line: "0.4"\nchangelog:\n  Fixed:\n    - "x"\n')
        assert any("never assembled" in f for f in _findings(root))


# ---------------------------------------------------------------------------
# Two release lines
# ---------------------------------------------------------------------------

def _conflicted(root):
    return sorted(_git(root, "diff", "--name-only", "--diff-filter=U").split())


def _two_lines(tmp_path):
    """A 0.4 line and a 0.5 line forked from it, each with work in flight.

    The 0.4 line holds a fragment that already reached 0.5 by a forward
    merge; 0.5 has its own fragment, which names the 0.4 label, and resolved
    MADD-ANO-002 in place.
    """
    root = _repo(tmp_path)
    _merge_branch(root, "fix/old", {_anomaly_file("fix-old"): anomalies_fragment(
        "fix-old", entry(L("fix-old", 1), title="old one"))})
    _git(root, "branch", "line-05")
    _git(root, "checkout", "-q", "line-05")
    (root / "pyproject.toml").write_text('[project]\nname = "t"\nversion = "0.5.0.dev0"\n')
    reg = root / fragments.REGISTRY
    reg.write_text(reg.read_text().replace('maddening_version: "0.4.2.dev0"',
                                           'maddening_version: "0.5.0.dev0"'))
    _commit(root, "version 0.5.0.dev0")
    text = reg.read_text()
    two = text[text.index('  - anomaly_id: "MADD-ANO-002"'):text.index(
        '  - anomaly_id: "MADD-ANO-003"')]
    reg.write_text(text.replace(two, two.replace(
        'resolution_status: "open"',
        'resolution_status: "resolved"\n    resolution_version: "0.5.0"').replace(
        '">=0.4.0.dev0"', '">=0.4.0.dev0, <0.5.0"')))
    _merge_branch(root, "feat/new", {
        _anomaly_file("feat-new"): anomalies_fragment("feat-new", entry(
            L("feat-new", 1), title="new one", note=f"Seen with {L('fix-old', 1)}."),
            line="0.5"),
        _changes_file("feat-new"): ('line: "0.5"\nrelease_notes:\n  Known anomalies: |\n'
                                    '    - `MADD-ANO-002` is resolved.\n'),
    }, message="resolve 002, add a 0.5 entry")
    _git(root, "checkout", "-q", "main")
    return root


class TestTwoLines:
    def test_a_quiet_point_assembly_merges_forward_with_no_conflict(self, tmp_path, gate,
                                                                    capsys):
        root = _two_lines(tmp_path)
        assert fragments.assemble(root, soup=False, out=lambda *_: None) == {
            L("fix-old", 1): "MADD-ANO-004"}
        _commit(root, "assemble the 0.4 line")
        _git(root, "checkout", "-q", "line-05")
        done = subprocess.run(["git", "-C", str(root), "merge", "-q", "--no-ff", "-m",
                               "forward merge", "main"], capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr
        assert any("run scripts/fragments.py relabel" in f
                   for f in check_fragments.findings(root))
        assert fragments.relabel(root) == [(_anomaly_file("feat-new"), 1)]
        _commit(root, "relabel")
        assert check_fragments.findings(root, "HEAD^1") == []
        assert _gate(gate, root, capsys)[0] == 0
        _git(root, "checkout", "-q", "main")
        assert check_fragments.findings(root) == []
        assert _gate(gate, root, capsys)[0] == 0

    def test_a_release_assembly_conflicts_on_the_version_header_alone(self, tmp_path, gate,
                                                                      capsys):
        root = _two_lines(tmp_path)
        (root / "pyproject.toml").write_text('[project]\nname = "t"\nversion = "0.4.2"\n')
        reg = root / fragments.REGISTRY
        reg.write_text(reg.read_text().replace('"0.4.2.dev0"', '"0.4.2"', 1))
        _commit(root, "version 0.4.2")
        assert fragments.assemble(root, release="0.4.2", date="2026-11-01", soup=False,
                                  out=lambda *_: None) == {L("fix-old", 1): "MADD-ANO-004"}
        _commit(root, "assemble 0.4.2")
        assert check_fragments.findings(root) == []                   # the release gate
        assert _gate(gate, root, capsys)[0] == 0
        _git(root, "checkout", "-q", "line-05")
        subprocess.run(["git", "-C", str(root), "merge", "-q", "--no-ff", "-m",
                        "forward merge", "main"], capture_output=True, text=True)
        assert _conflicted(root) == [fragments.REGISTRY, "pyproject.toml"]
        hunks = (root / fragments.REGISTRY).read_text().split("<<<<<<<")[1:]
        assert len(hunks) == 1 and "maddening_version" in hunks[0], hunks
        # The mechanical resolution: keep 0.5's version.
        for rel in (fragments.REGISTRY, "pyproject.toml"):
            path = root / rel
            text = path.read_text()
            ours = text.split("<<<<<<< HEAD\n")[1].split("=======\n")[0]
            start = text.index("<<<<<<< HEAD\n")
            end = text.index("\n", text.index(">>>>>>> ")) + 1
            path.write_text(text[:start] + ours + text[end:])
        _commit(root, "forward merge 0.4.2")
        assert fragments.relabel(root) == [(_anomaly_file("feat-new"), 1)]
        _commit(root, "relabel")
        assert check_fragments.findings(root, "HEAD^1") == []
        status, _, err = _gate(gate, root, capsys)
        assert status == 0, err


# ---------------------------------------------------------------------------
# CI runs the gate with the ratchet
# ---------------------------------------------------------------------------

def test_the_compliance_job_runs_the_ratchet_against_the_right_base(tmp_path):
    """The step's script, run under bash as GitHub runs it, with a stand-in
    python that records what it was asked to do."""
    workflow = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text())
    steps = workflow["jobs"]["compliance"]["steps"]
    assert steps[0]["uses"].startswith("actions/checkout@")
    assert steps[0]["with"]["fetch-depth"] == 0, "the ratchet reads the base's history"
    ratchet = [s for s in steps if "check_fragments.py" in s.get("run", "")]
    assert len(ratchet) == 1, "the compliance job runs scripts/check_fragments.py once"
    step = ratchet[0]
    assert "continue-on-error" not in step and "if" not in step
    assert step["env"] == {"EVENT": "${{ github.event_name }}",
                           "BASE_REF": "${{ github.base_ref }}"}
    fake = tmp_path / "bin" / "python"
    fake.parent.mkdir()
    fake.write_text('#!/bin/sh\necho "$@" >> "$OUT"\n')
    fake.chmod(0o755)
    for event, base, want in (
            ("pull_request", "release/0.4.2", "scripts/check_fragments.py --base origin/release/0.4.2"),
            ("push", "", "scripts/check_fragments.py")):
        out = tmp_path / f"{event}.txt"
        env = {"PATH": f"{fake.parent}:{os.environ['PATH']}", "OUT": str(out),
               "EVENT": event, "BASE_REF": base}
        subprocess.run(["bash", "-e", "-c", step["run"]], env=env, check=True, cwd=tmp_path)
        assert out.read_text().strip() == want
    names = [s.get("name", "") for s in steps]
    assert names.index(step["name"]) < names.index("Run compliance tests")


# ---------------------------------------------------------------------------
# new and from-branch
# ---------------------------------------------------------------------------

class TestNew:
    def test_it_writes_the_branchs_fragments_from_templates(self, tmp_path):
        root = _repo(tmp_path)
        _git(root, "checkout", "-q", "-b", "fix/Heat_CFL.check")
        assert fragments.main(["--root", str(root), "new", "--anomaly", "--changes"]) == 0
        anomalies = (root / _anomaly_file("fix-heat-cfl-check")).read_text()
        assert f'  - anomaly_id: "{L("fix-heat-cfl-check", 1)}"' in anomalies
        assert 'line: "0.4"' in anomalies
        assert (root / _changes_file("fix-heat-cfl-check")).exists()
        assert _findings(root) == []
        assert fragments.main(["--root", str(root), "new", "--changes"]) == 2

    def test_a_stem_reused_after_assembly_starts_after_the_recorded_labels(self, tmp_path):
        root = _repo(tmp_path)
        _two_fragments(root)
        fragments.assemble(root, soup=False, out=lambda *_: None)
        _commit(root, "assembly")
        assert fragments.main(["--root", str(root), "new", "--anomaly",
                               "--branch", "fix/b"]) == 0
        text = (root / _anomaly_file("fix-b")).read_text()
        assert f'"{L("fix-b", 3)}"' in text

    def test_a_branch_name_that_gives_no_stem_is_refused(self, tmp_path, capsys):
        root = _repo(tmp_path)
        assert fragments.main(["--root", str(root), "new", "--branch", "fix/+plus"]) == 2
        assert "not [a-z0-9]" in capsys.readouterr().err


class TestFromBranch:
    def test_it_moves_the_branchs_own_entries_into_a_fragment(self, tmp_path, gate, capsys):
        root = _repo(tmp_path)
        _git(root, "checkout", "-q", "-b", "fix/old-style")
        reg = root / fragments.REGISTRY
        text = reg.read_text().replace('title: "two"', 'title: "two, edited here"')
        reg.write_text(text + "\n" + entry("MADD-ANO-004", title="mine four") + "\n"
                       + entry("MADD-ANO-005", "resolved", title="mine five"))
        hw = root / fragments.HW_FILE
        hw.write_text(hw.read_text().replace("= 3", "= 5"))
        _write(root, "tests/test_mine.py", '"""Pins MADD-ANO-004 and MADD-ANO-005, not MADD-ANO-002."""\n')
        _commit(root, "the old way")
        mapping = fragments.from_branch(root, "main")
        assert mapping == {"MADD-ANO-004": L("fix-old-style", 1),
                           "MADD-ANO-005": L("fix-old-style", 2)}
        assert fragments.high_water(root) == 3
        ids = [a["anomaly_id"] for a in yaml.safe_load(reg.read_text())["anomalies"]]
        assert ids == ["MADD-ANO-001", "MADD-ANO-002", "MADD-ANO-003"]
        assert 'title: "two, edited here"' in reg.read_text()     # the base's entry stays
        assert (root / "tests/test_mine.py").read_text() == (
            f'"""Pins {L("fix-old-style", 1)} and {L("fix-old-style", 2)}, not MADD-ANO-002."""\n')
        frag = root / _anomaly_file("fix-old-style")
        found = _findings(root)
        assert found == [f"{_anomaly_file('fix-old-style')}: {L('fix-old-style', 1)} is "
                         f"reachable ('open') and has no release_note (the release notes' "
                         f"Known-anomalies text)"], found
        frag.write_text(frag.read_text().replace(
            "    resolution_status: \"open\"\n",
            "    resolution_status: \"open\"\n    release_note: \"Avoid it.\"\n", 1))
        assert _findings(root) == []
        assert _gate(gate, root, capsys)[0] == 0
        with pytest.raises(fragments.AssemblyError, match="exists already"):
            fragments.from_branch(root, "main")

    def test_an_id_the_base_also_used_meanwhile_is_still_the_branchs(self, tmp_path):
        root = _repo(tmp_path)
        _git(root, "checkout", "-q", "-b", "fix/mine")
        reg = root / fragments.REGISTRY
        reg.write_text(reg.read_text() + "\n" + entry("MADD-ANO-004", title="mine"))
        _commit(root, "mine is 004")
        _git(root, "checkout", "-q", "main")
        reg.write_text(reg.read_text() + "\n" + entry("MADD-ANO-004", title="theirs"))
        hw = root / fragments.HW_FILE
        hw.write_text(hw.read_text().replace("= 3", "= 4"))
        _commit(root, "theirs is 004 too")
        _git(root, "checkout", "-q", "fix/mine")
        assert fragments.from_branch(root, "main") == {"MADD-ANO-004": L("fix-mine", 1)}
        _commit(root, "moved")
        _git(root, "merge", "-q", "--no-edit", "main")
        ids = [a["anomaly_id"] for a in yaml.safe_load(reg.read_text())["anomalies"]]
        assert ids == ["MADD-ANO-001", "MADD-ANO-002", "MADD-ANO-003", "MADD-ANO-004"]
        assert "theirs" in reg.read_text() and "mine" not in reg.read_text()

    def test_after_merging_the_base_it_refuses(self, tmp_path):
        root = _repo(tmp_path)
        _git(root, "checkout", "-q", "-b", "fix/late")
        reg = root / fragments.REGISTRY
        reg.write_text(reg.read_text() + "\n" + entry("MADD-ANO-004", title="mine"))
        _commit(root, "mine is 004")
        reg.write_text(reg.read_text() + "\n" + entry("MADD-ANO-004", title="theirs"))
        _commit(root, "as a merge of the base would leave it")
        with pytest.raises(fragments.AssemblyError, match="in the registry twice"):
            fragments.from_branch(root, "main")


# ---------------------------------------------------------------------------
# The fix round after the pre-merge audit (its finding ids in the names)
# ---------------------------------------------------------------------------

def _version(root, version):
    root.joinpath("pyproject.toml").write_text(f'[project]\nname = "t"\nversion = "{version}"\n')
    _commit(root, f"version {version}")


def _release(root, version="0.4.2", date="2026-11-01"):
    return fragments.assemble(root, release=version, date=date, soup=False,
                              out=lambda *_: None)


def _release_section(root, version="0.4.2"):
    text = (root / "CHANGELOG.md").read_text()
    spans = fragments.version_section_span(text, version)
    assert len(spans) == 1, spans
    return text[spans[0][0]:spans[0][1]]


class TestTheReleaseAssemblyRunsTwice:
    """F1: a second ``assemble --release X.Y.Z`` adds to the section the first
    wrote, or changes nothing; after the tag it refuses."""

    def test_a_second_run_with_nothing_new_is_byte_identical(self, tmp_path):
        root = _repo(tmp_path)
        _two_fragments(root)
        _version(root, "0.4.2")
        _release(root)
        _commit(root, "release assembly")
        before = _tree(root)
        assert _release(root) == {}
        assert _tree(root) == before
        assert (root / "CHANGELOG.md").read_text().count("## [0.4.2]") == 1

    def test_a_late_fragment_joins_the_existing_section(self, tmp_path, gate, capsys):
        root = _repo(tmp_path)
        _two_fragments(root)
        _version(root, "0.4.2")
        _release(root)
        _commit(root, "release assembly")
        _merge_branch(root, "fix/late", {
            _changes_file("fix-late"): 'line: "0.4"\nchangelog:\n  Fixed:\n    - "A late fix."\n',
            _anomaly_file("fix-late"): anomalies_fragment("fix-late", entry(L("fix-late", 1))),
        })
        found = _findings(root)
        assert any("arrived after its release assembly: run scripts/fragments.py "
                   "assemble --release 0.4.2 again" in f for f in found), found
        assert _release(root) == {L("fix-late", 1): "MADD-ANO-007"}
        _commit(root, "fold the late fragment in")
        section = _release_section(root)
        assert section.index("### Fixed\n\n- A late fix.\n") < section.index("### Known Anomalies")
        # the count is the registry's after the late entry, written once
        assert section.count("registry entries are reachable in 0.4.2") == 1
        assert "- 5 registry entries are reachable in 0.4.2: 5 open" in section
        assert "- MADD-ANO-006: avoid it." in section
        assert check_fragments.findings(root) == []
        assert _gate(gate, root, capsys)[0] == 0

    def test_after_the_tag_a_fragment_belongs_to_the_next_version(self, tmp_path):
        root = _repo(tmp_path)
        _two_fragments(root)
        _version(root, "0.4.2")
        _release(root)
        _commit(root, "release assembly")
        _git(root, "tag", "v0.4.2")
        _merge_branch(root, "fix/late", {
            _changes_file("fix-late"): 'line: "0.4"\nchangelog:\n  Fixed:\n    - "Late."\n'})
        found = _findings(root)
        assert any("v0.4.2 is tagged, so 0.4.2 is out and this fragment belongs to the "
                   "next version" in f for f in found), found
        before = _tree(root)
        with pytest.raises(fragments.AssemblyError, match="v0.4.2 is tagged.*belongs to the "
                                                           "next version"):
            _release(root)
        assert _tree(root) == before

    def test_two_sections_for_one_version_are_refused_at_any_version(self, tmp_path):
        root = _repo(tmp_path)
        changelog = root / "CHANGELOG.md"
        changelog.write_text(changelog.read_text().replace(
            "## [0.4.1] - 2026-10-13\n", "## [0.4.1] - 2026-10-14\n\n- again\n\n"
                                         "## [0.4.1] - 2026-10-13\n", 1))
        found = _findings(root)
        assert any("has 2 '## [0.4.1]' sections" in f for f in found), found

    def test_the_assembler_refuses_a_changelog_with_two_sections(self, tmp_path):
        root = _repo(tmp_path)
        _two_fragments(root)
        _version(root, "0.4.2")
        changelog = root / "CHANGELOG.md"
        changelog.write_text(changelog.read_text().replace(
            "## [Unreleased]\n", "## [Unreleased]\n\n## [0.4.2] - 2026-11-01\n\n"
                                 "## [0.4.2] - 2026-11-01\n", 1))
        _commit(root, "a doubled section")
        with pytest.raises(fragments.AssemblyError):
            _release(root)


class TestTheRatchetReadsTheWholeHistory:
    """F2: a label the line ever merged is live or recorded, however the
    commits that dropped it were pushed."""

    def test_a_two_commit_push_cannot_drop_a_merged_fragment(self, tmp_path):
        """The audit's r02: commit A drops the fragment, commit B is unrelated,
        and both are pushed together."""
        root = _repo(tmp_path)
        _merge_branch(root, "fix/b", {_anomaly_file("fix-b"): anomalies_fragment(
            "fix-b", entry(L("fix-b", 1)))})
        (root / _anomaly_file("fix-b")).unlink()
        _commit(root, "A: drop it")
        _write(root, "README.md", "x\n")
        _commit(root, "B: unrelated")
        for line in (None, "HEAD", "HEAD^1"):
            found = check_fragments.findings(root, line)
            assert any(f"{L('fix-b', 1)} came into the line" in f for f in found), (line, found)

    def test_a_forward_merge_pushed_with_its_relabel_cannot_drop_a_fragment(self, tmp_path):
        """The audit's r02c: the merge's resolution loses the 0.5 line's own
        fragment, and the relabel commit follows it in the same push."""
        root = _two_lines(tmp_path)
        fragments.assemble(root, soup=False, out=lambda *_: None)
        _commit(root, "assemble the 0.4 line")
        _git(root, "checkout", "-q", "line-05")
        _write(root, "tests/test_x.py", f"# pins {L('fix-old', 1)}\n")
        _commit(root, "0.5 pins the 0.4 label")
        subprocess.run(["git", "-C", str(root), "merge", "-q", "--no-ff", "--no-commit", "main"],
                       capture_output=True)
        _git(root, "rm", "-q", _anomaly_file("feat-new"))
        _git(root, "commit", "-q", "-m", "forward merge")
        fragments.relabel(root)
        _commit(root, "relabel")
        found = check_fragments.findings(root)
        assert any(f"{L('feat-new', 1)} came into the line" in f for f in found), found

    def test_a_branchs_own_unmerged_history_is_not_read_after_it_merges(self, tmp_path):
        """A label a branch added and dropped before it merged never reached
        the line: the merge's diff against the line is what the line saw."""
        root = _repo(tmp_path)
        _git(root, "checkout", "-q", "-b", "fix/w")
        _write(root, _anomaly_file("fix-w"), anomalies_fragment(
            "fix-w", entry(L("fix-w", 1)), entry(L("fix-w", 2))))
        _commit(root, "two entries")
        _write(root, _anomaly_file("fix-w"), anomalies_fragment("fix-w", entry(L("fix-w", 1))))
        _commit(root, "one was a mistake")
        _git(root, "checkout", "-q", "main")
        _git(root, "merge", "-q", "--no-ff", "-m", "Merge fix/w", "fix/w")
        assert check_fragments.findings(root) == []
        # On the branch itself, HEAD's history holds the dropped label; the
        # base's history does not, which is what --base reads.
        _git(root, "checkout", "-q", "fix/w")
        assert any(L("fix-w", 2) in f for f in check_fragments.findings(root))
        assert check_fragments.findings(root, "main") == []

    def test_a_shallow_clone_fails_closed(self, tmp_path):
        root = _repo(tmp_path)
        _merge_branch(root, "fix/b", {_anomaly_file("fix-b"): anomalies_fragment(
            "fix-b", entry(L("fix-b", 1)))})
        shallow = tmp_path / "shallow"
        subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{root}", str(shallow)],
                       check=True, capture_output=True)
        found = check_fragments.findings(shallow)
        assert any("the ratchet cannot check" in f and "shallow clone" in f
                   for f in found), found
        with pytest.raises(fragments.AssemblyError, match="shallow clone"):
            fragments.assemble(shallow, soup=False, out=lambda *_: None)


class TestChangesFragmentsJoinTheRatchet:
    """F3: a merged changes fragment leaves only by a release assembly."""

    @pytest.mark.parametrize("body", [
        'changelog:\n  Fixed:\n    - "A user-visible fix."\n',
        'release_notes:\n  Fixed: "Only notes text."\n',
    ])
    def test_deleting_one_unassembled_fails(self, tmp_path, body):
        root = _repo(tmp_path)
        _merge_branch(root, "fix/c", {_changes_file("fix-c"): 'line: "0.4"\n' + body})
        (root / _changes_file("fix-c")).unlink()
        _commit(root, "drop the changes fragment")
        found = check_fragments.findings(root)
        assert any(f"{_changes_file('fix-c')} came into the line" in f
                   and "A merged changes fragment leaves only by a release assembly" in f
                   for f in found), found

    def test_one_the_release_assembly_wrote_passes(self, tmp_path):
        root = _repo(tmp_path)
        _merge_branch(root, "fix/c", {_changes_file("fix-c"): (
            'line: "0.4"\nchangelog:\n  Fixed:\n    - |\n      Two lines\n      of text.\n'
            'release_notes:\n  Fixed: "Notes text."\n')})
        _version(root, "0.4.2")
        _release(root)
        _commit(root, "release assembly")
        assert check_fragments.findings(root) == []
        # ... and the bullet is what the check compares: take it out of the
        # changelog and the deleted fragment is reported.
        changelog = root / "CHANGELOG.md"
        changelog.write_text(changelog.read_text().replace("- Two lines\n  of text.\n", ""))
        found = _findings(root)
        assert any("its Fixed bullet is not in CHANGELOG.md" in f for f in found), found


class TestARecordedLabelHasAHistory:
    """F4: a '# label:' comment names a label some fragment held."""

    @pytest.mark.parametrize("version", ["0.4.2.dev0", "0.5.0.dev0"])
    def test_a_hand_written_comment_fails_on_either_line(self, tmp_path, version):
        root = _repo(tmp_path, version=version)
        reg = root / fragments.REGISTRY
        reg.write_text(reg.read_text() + "\n" + entry("MADD-ANO-004", title="by hand").replace(
            '  - anomaly_id: "MADD-ANO-004"\n',
            '  - anomaly_id: "MADD-ANO-004"\n    # label: NEW-never-a-fragment-1\n'))
        hw = root / fragments.HW_FILE
        hw.write_text(hw.read_text().replace("= 3", "= 4"))
        _commit(root, "number an entry by hand")
        found = check_fragments.findings(root)
        assert any(f"records {L('never-a-fragment', 1)}, which no fragment in this "
                   "tree's history ever held" in f for f in found), found

    def test_a_label_another_line_assembled_has_its_history_through_the_merge(self, tmp_path):
        """A 0.4 fragment added and assembled between two forward merges never
        appears in the 0.5 line's first-parent history; its label was held in
        a commit the merge reaches."""
        root = _two_lines(tmp_path)
        _merge_branch(root, "fix/mid", {_anomaly_file("fix-mid"): anomalies_fragment(
            "fix-mid", entry(L("fix-mid", 1), title="mid one"))})
        fragments.assemble(root, soup=False, out=lambda *_: None)
        _commit(root, "assemble the 0.4 line")
        _git(root, "checkout", "-q", "line-05")
        done = subprocess.run(["git", "-C", str(root), "merge", "-q", "--no-ff", "-m",
                               "forward merge", "main"], capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr
        fragments.relabel(root)
        _commit(root, "relabel")
        assert check_fragments.findings(root) == []


def test_a_fragment_only_libyaml_reads_is_refused_by_both_gates(tmp_path, gate, capsys):
    """F5: fragments are read with yaml.safe_load, as the assembler's SOUP
    generator reads the registry they become."""
    root = _repo(tmp_path)
    _write(root, _anomaly_file("fix-heat"), anomalies_fragment(
        "fix-heat", entry(L("fix-heat", 1))).replace('severity: "minor"', 'severity:\t"minor"'))
    found = _findings(root)
    assert any("does not parse as YAML" in f for f in found), found
    status, _, err = _gate(gate, root, capsys)
    assert status == 1 and "does not parse as YAML" in err


def test_a_dot0_release_assembly_names_every_reachable_entry(tmp_path, monkeypatch):
    """F7: from its .0 notes on, a line's notes are read alone, so the .0
    release assembly names every reachable entry there, not only those it
    numbered (as 0.4.0's notes did)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_release_notes_rules", REPO_ROOT / "tests" / "compliance"
        / "test_release_notes_known_anomalies.py")
    rules = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rules)
    root = _repo(tmp_path, version="0.5.0.dev0")
    monkeypatch.setattr(fragments, "RELEASE_LINES", ("0.5",))      # the hand-over commit
    _write(root, "docs/validation/development_record_v0.5.0.md", _RECORD)
    _commit(root, "the 0.5 line's record")
    _merge_branch(root, "feat/x", {
        _changes_file("feat-x"): 'line: "0.5"\nchangelog:\n  Added:\n    - "A thing."\n',
        _anomaly_file("feat-x"): anomalies_fragment("feat-x", entry(L("feat-x", 1)), line="0.5"),
    })
    _release(root, "0.5.0", "2026-12-01")
    notes = (root / "docs/release_notes/v0.5.0.md").read_text()
    known = rules.section(notes, rules.NOTES_HEADING)
    assert "- `MADD-ANO-002`: two." in known and "- `MADD-ANO-003`: three." in known
    assert "- `MADD-ANO-004`: Avoid it; see the guide." in known
    assert "MADD-ANO-001" not in known                      # resolved
    reg = yaml.safe_load((root / fragments.REGISTRY).read_text())
    names = sorted(p.name for p in (root / "docs/release_notes").glob("v*.md"))
    line_notes = [(n, (root / "docs/release_notes" / n).read_text())
                  for n in rules.line_notes(names, "0.5.0")]
    assert [n for n, _ in line_notes] == ["v0.5.0.md"]
    assert rules.notes_findings(line_notes, reg, *rules.fragment_view(root)) == []


class TestBulletsAreCarriedVerbatim:
    """F8: the release section holds each bullet as the fragment wrote it."""

    CASES = {
        "multi-line text": ("    - |\n      First line,\n      second line.\n",
                            "- First line,\n  second line.\n"),
        "a nested list": ("    - |\n      A fix with a list:\n\n      - first point\n"
                          "      - second point\n",
                          "- A fix with a list:\n\n  - first point\n  - second point\n"),
        "a code fence": ("    - |\n      Para one.\n\n      ```python\n      x  =  1\n      ```\n",
                         "- Para one.\n\n  ```python\n  x  =  1\n  ```\n"),
        "a leading dash": ('    - "- starts with a dash"\n', "- - starts with a dash\n"),
        "two spaces in code": ('    - "`a  b` keeps two spaces"\n', "- `a  b` keeps two spaces\n"),
    }

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_through_a_release_assembly(self, tmp_path, case):
        given, written = self.CASES[case]
        root = _repo(tmp_path)
        _merge_branch(root, "fix/t", {_changes_file("fix-t"):
                                      'line: "0.4"\nchangelog:\n  Fixed:\n' + given})
        _version(root, "0.4.2")
        _release(root)
        section = _release_section(root)
        assert "### Fixed\n\n" + written in section, section
        _commit(root, "release assembly")
        assert check_fragments.findings(root) == []

    def test_rendering_loses_nothing(self):
        for text in ("one", "a\n\n  - b\n  - c", "x  y", "- dash", "```\ncode  here\n```"):
            assert fragments.unrender_bullet(fragments.render_bullet(text)) == text

    def test_a_lossy_rendering_is_caught_and_the_tree_restored(self, tmp_path, monkeypatch):
        root = _repo(tmp_path)
        _merge_branch(root, "fix/t", {_changes_file("fix-t"):
                                      'line: "0.4"\nchangelog:\n  Fixed:\n    - "`a  b`"\n'})
        _version(root, "0.4.2")
        before = _tree(root)
        render = fragments.render_bullet
        monkeypatch.setattr(fragments, "render_bullet",
                            lambda text: render(" ".join(str(text).split())))
        with pytest.raises(fragments.AssemblyError, match="does not survive its rendering"):
            _release(root)
        assert _tree(root) == before

    def test_a_bullet_the_section_loses_is_caught(self, tmp_path, monkeypatch):
        """_verify's count of each bullet in the section (audit mutant F1)."""
        root = _repo(tmp_path)
        _merge_branch(root, "fix/t", {_changes_file("fix-t"):
                                      'line: "0.4"\nchangelog:\n  Fixed:\n    - "Kept."\n'
                                      '    - "Lost."\n'})
        _version(root, "0.4.2")
        before = _tree(root)
        section = fragments.release_section
        monkeypatch.setattr(fragments, "release_section", lambda existing, v, d, bullets, c: section(
            existing, v, d, {k: [b for b in items if b != "Lost."] for k, items in bullets.items()},
            c))
        with pytest.raises(fragments.AssemblyError, match="added 0 time"):
            _release(root)
        assert _tree(root) == before

    def test_a_notes_block_the_notes_lose_is_caught(self, tmp_path, monkeypatch):
        """_verify's check of each release-notes block (audit mutant F2)."""
        root = _repo(tmp_path)
        _merge_branch(root, "fix/t", {_changes_file("fix-t"):
                                      'line: "0.4"\nrelease_notes:\n  Fixed: "The text."\n'})
        _version(root, "0.4.2")
        before = _tree(root)
        append = fragments._notes_append
        monkeypatch.setattr(fragments, "_notes_append", lambda root_, snap, version, heading, block:
                            append(root_, snap, version, None, None))
        with pytest.raises(fragments.AssemblyError, match="release-notes block is missing"):
            _release(root)
        assert _tree(root) == before


class TestTheGatesRefuseWhatTheAssemblerWould:
    """F9: each input the assembler refused passed both gates; now the gates
    refuse it, or the assembler takes it."""

    def test_a_label_with_a_leading_zero(self, tmp_path, gate, capsys):
        root = _repo(tmp_path)
        lab = _NEW + "fix-z-01"
        _write(root, _anomaly_file("fix-z"), anomalies_fragment("fix-z", entry(lab)))
        found = _findings(root)
        assert any(f"anomaly_id '{lab}' is not a label" in f for f in found), found
        assert _gate(gate, root, capsys)[0] == 1
        assert fragments.LABEL_RE.fullmatch(lab) is None
        assert fragments.LABEL_COMMENT_RE.match("    # label: NEW-fix-z-01") is None

    @pytest.mark.parametrize("kind", ["anomalies", "changes"])
    def test_a_fragment_renamed_after_it_merged(self, tmp_path, kind):
        root = _repo(tmp_path)
        if kind == "anomalies":
            old, new = _anomaly_file("fix-a"), _anomaly_file("fix-aa")
            _merge_branch(root, "fix/a", {old: anomalies_fragment("fix-a", entry(L("fix-a", 1)))})
            text = (root / old).read_text().replace("fix-a-1", "fix-aa-1")
        else:
            old, new = _changes_file("fix-a"), _changes_file("fix-b")
            _merge_branch(root, "fix/a", {old: 'line: "0.4"\nchangelog:\n  Fixed:\n    - "A fix."\n'})
            text = (root / old).read_text()
        _git(root, "mv", old, new)
        _write(root, new, text)
        _commit(root, "rename")
        found = check_fragments.findings(root)
        assert any("came into the line" in f for f in found), found
        # The assembler's merge order sees the renamed file as an addition.
        assert new in fragments.merge_order(root, [new])

    def test_a_label_in_a_path_with_a_space_or_an_accent(self, tmp_path):
        root = _repo(tmp_path)
        lab = L("fix-s", 1)
        _merge_branch(root, "fix/s", {_anomaly_file("fix-s"): anomalies_fragment("fix-s", entry(lab)),
                                      "docs/user guide/notes.md": f"See {lab}.\n",
                                      "docs/notes é.md": f"See {lab}.\n"})
        assert check_fragments.findings(root) == []
        assert fragments.assemble(root, soup=False, out=lambda *_: None) == {lab: "MADD-ANO-004"}
        assert (root / "docs/user guide/notes.md").read_text() == "See MADD-ANO-004.\n"
        assert (root / "docs/notes é.md").read_text() == "See MADD-ANO-004.\n"

    def test_relabel_reads_a_path_with_a_space(self, tmp_path):
        root = _repo(tmp_path)
        _two_fragments(root)
        fragments.assemble(root, soup=False, out=lambda *_: None)
        _commit(root, "assembly")
        _write(root, "docs/a b/late.md", f"# {L('fix-a', 1)}\n")
        _commit(root, "a merge brings the label back")
        assert fragments.relabel(root) == [("docs/a b/late.md", 1)]

    def test_from_branch_on_a_reused_stem_numbers_after_the_recorded_labels(self, tmp_path):
        root = _repo(tmp_path)
        _merge_branch(root, "fix/b", {_anomaly_file("fix-b"): anomalies_fragment(
            "fix-b", entry(L("fix-b", 1)))})
        fragments.assemble(root, soup=False, out=lambda *_: None)
        _commit(root, "assembly")
        _git(root, "branch", "-D", "fix/b")
        _git(root, "checkout", "-q", "-b", "fix/b")
        reg = root / fragments.REGISTRY
        reg.write_text(reg.read_text() + "\n" + entry("MADD-ANO-005", title="old style"))
        hw = root / fragments.HW_FILE
        hw.write_text(hw.read_text().replace("= 4", "= 5"))
        _commit(root, "an entry the old way")
        assert fragments.from_branch(root, "main") == {"MADD-ANO-005": L("fix-b", 2)}
        found = _findings(root)
        assert not any("already assembled" in f for f in found), found


@pytest.mark.parametrize("heading", ["## [Unreleased]", "## [unreleased]", "## Unreleased",
                                     "## UNRELEASED"])
def test_the_release_gate_reads_every_spelling_of_unreleased(tmp_path, heading):
    """N1."""
    root = _repo(tmp_path, version="0.4.2")
    _write(root, "docs/release_notes/v0.4.2.md", "# v0.4.2\n")
    changelog = root / "CHANGELOG.md"
    changelog.write_text(changelog.read_text().replace(
        "## [Unreleased]\n", f"{heading}\n\n- by hand\n"))
    found = _findings(root)
    assert any(f"'{heading}' is not empty at release version 0.4.2" in f for f in found), found


# ---------------------------------------------------------------------------
# Tests the audit's surviving mutants showed missing
# ---------------------------------------------------------------------------

def test_is_dev_is_the_dot_dev_suffix_alone():
    """Mutant F7: anything else is a release version, so the gate fails closed."""
    for version in ("0.4.2.dev0", "0.5.0.dev3", "1.0.0.dev"):
        assert fragments.is_dev(version), version
    for version in ("0.4.2", "0.4.2rc1", "0.4.2.dev0+local", "0.4.2dev0", "0.4.2.devel"):
        assert not fragments.is_dev(version), version


def test_release_must_be_the_trees_release(tmp_path):
    """Mutant F6."""
    root = _repo(tmp_path)
    _two_fragments(root)
    before = _tree(root)
    with pytest.raises(fragments.AssemblyError, match="--release 0.4.3 is not this tree's"):
        _release(root, "0.4.3")
    assert _tree(root) == before


def test_the_templates_multi_line_extra_keys_never_reach_the_registry(tmp_path):
    """Mutant F9: the template writes ``release_note: >`` over several lines."""
    root = _repo(tmp_path)
    text = (entry(L("fix-m", 1), note=None) + "    release_note: >\n      Line one of the note,\n"
            "      line two.\n    record_index: >\n      `MADD-ANO-004` — *m* —\n      **open**.\n")
    _merge_branch(root, "fix/m", {_anomaly_file("fix-m"): anomalies_fragment("fix-m", text)})
    fragments.assemble(root, soup=False, out=lambda *_: None)
    reg = (root / fragments.REGISTRY).read_text()
    assert "release_note" not in reg and "record_index" not in reg and "Line one" not in reg
    assert yaml.safe_load(reg)["anomalies"][-1]["anomaly_id"] == "MADD-ANO-004"


def test_a_stem_used_again_is_ordered_by_its_latest_addition(tmp_path):
    """Mutant F10: a fragment path added, assembled (deleted) and added again
    by a later branch of the same name is numbered where it merged last."""
    root = _repo(tmp_path)
    _merge_branch(root, "fix/b", {_anomaly_file("fix-b"): anomalies_fragment(
        "fix-b", entry(L("fix-b", 1)))})
    fragments.assemble(root, soup=False, out=lambda *_: None)
    _commit(root, "assembly")                                   # fix-b-1 -> 004
    _git(root, "branch", "-D", "fix/b")
    _merge_branch(root, "fix/c", {_anomaly_file("fix-c"): anomalies_fragment(
        "fix-c", entry(L("fix-c", 1)))})
    _merge_branch(root, "fix/b", {_anomaly_file("fix-b"): anomalies_fragment(
        "fix-b", entry(L("fix-b", 2)))})
    assert fragments.assemble(root, soup=False, out=lambda *_: None) == {
        L("fix-c", 1): "MADD-ANO-005", L("fix-b", 2): "MADD-ANO-006"}


def test_relabel_leaves_a_resurrected_fragments_id_lines_alone(tmp_path):
    """Mutant F11: the fragment then still says 'already assembled', not 'is a
    number', and its labels elsewhere in the file are rewritten."""
    root = _repo(tmp_path)
    _two_fragments(root)
    old = (root / _anomaly_file("fix-a")).read_text()
    fragments.assemble(root, soup=False, out=lambda *_: None)
    _commit(root, "assembly")
    _write(root, _anomaly_file("fix-a"), old)
    _commit(root, "a stale merge brings it back")
    fragments.relabel(root)
    text = (root / _anomaly_file("fix-a")).read_text()
    assert f'  - anomaly_id: "{L("fix-a", 1)}"' in text
    assert "Like MADD-ANO-004: avoid it." in text                # its release_note
    found = check_fragments.findings(root)
    assert any(f"{L('fix-a', 1)} is already assembled as MADD-ANO-006" in f for f in found), found


def test_a_label_runs_to_a_word_boundary(tmp_path):
    """Mutant F12: ``...-1x`` is not the label ``...-1`` followed by text."""
    assert fragments.LABEL_RE.search(L("fix-heat", 1) + "x") is None
    assert fragments.LABEL_RE.search(L("fix-heat", 1) + "_x") is None
    root = _repo(tmp_path)
    _write(root, _anomaly_file("fix-heat"), anomalies_fragment("fix-heat", entry(L("fix-heat", 1))))
    _write(root, "tests/test_pin.py", f"# {L('fix-heat', 1)}x\n")
    found = _findings(root)
    assert any("is not the label spelling" in f for f in found), found


def test_a_label_comment_inside_a_fragment_is_refused(tmp_path):
    """Mutant X1: only the registry records labels."""
    root = _repo(tmp_path)
    _write(root, _anomaly_file("fix-heat"), anomalies_fragment("fix-heat", entry(
        L("fix-heat", 1), extra="    # label: NEW-fix-heat-1")))
    found = _findings(root)
    assert any("a '# label:' comment; only the registry records labels" in f
               for f in found), found


def test_the_label_comment_sits_right_after_the_id_line():
    """Mutant X2: invariant 5.4's layout half."""
    text = ('  - anomaly_id: "MADD-ANO-004"\n    title: "x"\n    # label: NEW-fix-b-1\n')
    found = check_fragments.label_record_findings(text)
    assert any("not right after its id line" in f for f in found), found
    assert check_fragments.label_record_findings(
        '  - anomaly_id: "MADD-ANO-004"\n    # label: NEW-fix-b-1\n    title: "x"\n') == []


def test_the_cross_line_allowance_ignores_pre_release_headings(gate):
    """Mutant A5: an rc of 0.4.2 is not a 0.4 release, so the next patch is
    still 0.4.2."""
    from packaging.version import Version

    released = frozenset(Version(v) for v in ("0.4.0", "0.4.1", "0.4.2rc1"))
    assert gate.cross_line_versions(Version("0.5.0.dev0"), released, Version) == {
        Version("0.4.2")}
