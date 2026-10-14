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

    @pytest.mark.parametrize("spelling", [
        "madd-ano-new-fix-heat-1", "MADD_ANO_NEW_fix_heat_1", "MADD-ANO-NEW-Fix-heat-1",
        "MADD-ANO-NEW_fix-heat-1", "MADDANONEW-fix-heat-1"])
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
        found = check_fragments.findings(root, base)
        assert any(f"{L('fix-b', 1)} is in {base}'s fragments" in f for f in found), found
        assert check_fragments.findings(root) == []           # nothing else sees it

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
        assert any("the ratchet cannot read no-such-ref" in f for f in found), found


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
        assert any("a merged fragment was deleted" in f
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
