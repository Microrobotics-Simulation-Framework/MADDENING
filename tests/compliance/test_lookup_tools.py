"""The lookup tools in ``scripts/``: ``anom.py``, ``claim.py`` and ``outline.py``.

Each command runs in a subprocess, as a contributor runs it, on a small tree
built here: a registry, a fragment, a high-water mark, two claims
inventories and a Python file.  ``--root`` defaults to the repository holding
the script, and no tool carries an absolute path or a private name.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
TOOLS = ("anom.py", "claim.py", "outline.py", "fragments.py", "check_fragments.py")

#: A label, written in two pieces so that this file spells none.
_LABEL = "MADD-ANO-" "NEW-fix-heat-1"


def _run(tool, *args, cwd=None):
    done = subprocess.run([sys.executable, str(SCRIPTS / tool), *map(str, args)],
                          capture_output=True, text=True, cwd=cwd or REPO_ROOT, timeout=60)
    return done.returncode, done.stdout, done.stderr


def _entry(aid, status, title, versions, extra=""):
    return (f'  - anomaly_id: "{aid}"\n    title: "{title}"\n    description: >\n'
            f'      Seen in the heat solver.\n    severity: "minor"\n'
            f'    affected_components:\n      - "maddening.nodes.heat.HeatNode"\n'
            f'    affected_versions: "{versions}"\n    resolution_status: "{status}"\n{extra}')


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("tools")
    (root / "docs" / "validation").mkdir(parents=True)
    (root / "docs" / "validation" / "known_anomalies.yaml").write_text(
        'schema_version: "1.0"\nmaddening_version: "0.4.2.dev0"\nanomalies:\n'
        + _entry("MADD-ANO-001", "resolved", "Segfault on an old CUDA", ">=0.1.0, <0.4.0")
        + "\n" + _entry("MADD-ANO-002", "open", "Coupling drifts on units", ">=0.3.0")
        + "\n" + _entry("MADD-ANO-003", "open", "Introduced and fixed in the cycle", "none")
        + '    # label: NEW-old-fix-1\n')
    frag = root / "docs" / "fragments" / "anomalies" / "fix-heat.yaml"
    frag.parent.mkdir(parents=True)
    frag.write_text('line: "0.4"\nanomalies:\n' + _entry(
        _LABEL, "open", "Heat timestep lost after a reset", ">=0.4.2.dev0",
        '    release_note: "Set it again."\n'))
    (root / "tests" / "compliance").mkdir(parents=True)
    (root / "tests" / "compliance" / "test_soup_evidence.py").write_text(
        "_HIGHEST_ANOMALY_ID = 3\n")
    (root / "tests" / "test_heat.py").write_text(
        f'"""Pins {_LABEL} and MADD-ANO-002."""\n\n\ndef test_reset():\n    pass\n')
    (root / "CHANGELOG.md").write_text(
        "# C\n\n## [Unreleased]\n\n## [0.4.0] - 2026-10-13\n\n## [0.3.1] - 2026-06-22\n")
    for name, prefix in (("coupling", "CPL"), ("rest_runpod", "RST")):
        (root / "docs" / "validation" / f"{name}_claims.yaml").write_text(
            "claims:\n" + "".join(
                f'  - id: "{prefix}-{n:03d}"\n    status: "holds"\n'
                f'    claim: "{prefix} claim number {n} about a floor"\n' for n in (1, 2, 7)))
    (root / "pkg.py").write_text(
        "def a():\n    pass\n\n\nclass B:\n    def m(self):\n        x = 1\n"
        "        return x\n\n\ndef long_one():\n" + "    y = 1\n" * 10)
    for args in (("init", "-q"), ("add", "-A")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return root


class TestAnom:
    def test_list_shows_fragment_entries_by_label(self, tree):
        rc, out, _ = _run("anom.py", "--root", tree, "list")
        assert rc == 0
        rows = out.strip().splitlines()
        assert len(rows) == 4 and rows[0].split()[0] == "1"
        assert rows[-1].startswith(_LABEL)
        assert rows[-1].endswith("[fragment docs/fragments/anomalies/fix-heat.yaml]")

    @pytest.mark.parametrize("args, first_words", [
        (("--open",), ["2", "3", _LABEL]),
        (("--status", "resolved"), ["1"]),
        (("--grep", "units"), ["2"]),
        (("--fragments-only",), [_LABEL]),
        (("--ids", "2..3"), ["2", "3"]),
        (("--shipped",), ["1", "2"]),
        (("--never-shipped",), ["3", _LABEL]),
        (("--title", "heat reset"), [_LABEL]),
    ])
    def test_list_filters(self, tree, args, first_words):
        rc, out, _ = _run("anom.py", "--root", tree, "list", *args)
        assert rc == 0
        assert [row.split()[0] for row in out.strip().splitlines()] == first_words

    def test_similar_searches_the_fragments_too(self, tree):
        rc, out, _ = _run("anom.py", "--root", tree, "list", "--similar", "heat timestep reset")
        assert rc == 0
        assert out.splitlines()[0].split("] ", 1)[1].split()[0] == _LABEL

    def test_show_takes_a_number_an_id_or_a_label(self, tree):
        for spec in ("2", "MADD-ANO-002"):
            rc, out, _ = _run("anom.py", "--root", tree, "show", spec)
            assert rc == 0 and out.startswith("# docs/validation/known_anomalies.yaml line")
            assert 'anomaly_id: "MADD-ANO-002"' in out
        rc, out, _ = _run("anom.py", "--root", tree, "show", _LABEL)
        assert rc == 0 and out.startswith("# docs/fragments/anomalies/fix-heat.yaml line 3")
        assert "release_note" in out
        rc, out, _ = _run("anom.py", "--root", tree, "show", "99")
        assert rc == 1 and "MADD-ANO-099: not in the registry or a fragment" in out

    def test_next_says_numbers_come_from_the_assembler(self, tree):
        rc, out, _ = _run("anom.py", "--root", tree, "next")
        assert rc == 0
        assert ("highest id in the registry: 3; _HIGHEST_ANOMALY_ID: 3; in fragments: "
                "1 entr(ies) in 1 file(s)") in out
        assert "scripts/fragments.py assemble numbers it" in out

    def test_refs_of_a_label_and_of_an_assembled_one(self, tree):
        rc, out, _ = _run("anom.py", "--root", tree, "refs", _LABEL)
        assert rc == 0
        assert "tests/test_heat.py:1:" in out and "fix-heat.yaml" in out
        rc, out, _ = _run("anom.py", "--root", tree, "refs", "MADD-ANO-" "NEW-old-fix-1")
        assert rc == 0 and "was assembled as MADD-ANO-003" in out
        rc, out, _ = _run("anom.py", "--root", tree, "refs", "2")
        assert rc == 0 and "tests/test_heat.py:1:" in out
        rc, out, _ = _run("anom.py", "--root", tree, "refs", "3")
        assert "came from the fragment label" in out

    def test_count(self, tree):
        rc, out, _ = _run("anom.py", "--root", tree, "list", "--count")
        assert rc == 0 and out.startswith("4 entries (1 in fragments)")
        assert "shipped: 2, never shipped: 2" in out


class TestClaim:
    def test_inventories_are_discovered(self, tree):
        rc, out, _ = _run("claim.py", "--root", tree, "inventories")
        assert rc == 0 and [r.split()[0] for r in out.splitlines()] == ["coupling", "rest_runpod"]

    def test_show_list_and_next(self, tree):
        rc, out, _ = _run("claim.py", "--root", tree, "show", "RST-007", "CPL-002")
        assert rc == 0
        assert "# rest_runpod_claims.yaml line 8" in out and "RST claim number 7" in out
        assert "# coupling_claims.yaml line 5" in out
        rc, out, _ = _run("claim.py", "--root", tree, "list", "rest", "--grep", "number 2")
        assert rc == 0 and out.split()[:2] == ["RST-002", "holds"]
        rc, out, _ = _run("claim.py", "--root", tree, "next", "CPL")
        assert out.strip() == "CPL: 3 rows, highest 7, next free CPL-008"
        rc, _, err = _run("claim.py", "--root", tree, "list", "nothing")
        assert rc == 2 and "coupling" in err


class TestOutline:
    def test_line_ranges(self, tree):
        rc, out, _ = _run("outline.py", "--root", tree, "pkg.py")
        assert rc == 0
        rows = [r.split() for r in out.splitlines()[1:]]
        assert rows == [["1-2", "2", "def", "a"], ["5-8", "4", "class", "B"],
                        ["6-8", "3", "def", "m"], ["11-21", "11", "def", "long_one"]]
        rc, out, _ = _run("outline.py", "--root", tree, "pkg.py", "--min", "5")
        assert [r.split()[-1] for r in out.splitlines()[1:]] == ["long_one"]


def test_root_defaults_to_the_repository_holding_the_script(tmp_path):
    for tool, args in (("anom.py", ("next",)), ("claim.py", ("inventories",)),
                       ("outline.py", ("scripts/outline.py",))):
        default = _run(tool, *args, cwd=tmp_path)
        given = _run(tool, "--root", REPO_ROOT, *args, cwd=tmp_path)
        assert default[0] == 0 and default == given, tool


#: sha256 of the lower-cased private project names that stay out of this
#: repository (they are not written here either).
_PRIVATE = {"52216bda4f3d273f0f7f4984e19b3da29f34eeb7a2428406dd0819fc3ff8b5cc",
            "f4fc453cfc99e227100fd7a864dd26de2541e1f4143308e455ec18bc2851814b"}


@pytest.mark.parametrize("tool", TOOLS)
def test_no_tool_holds_an_absolute_path_or_a_private_name(tool):
    text = (SCRIPTS / tool).read_text()
    assert "/home/" not in text and "/Users/" not in text
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", text)}
    assert not {hashlib.sha256(w.encode()).hexdigest() for w in words} & _PRIVATE


def test_the_private_name_scan_can_find_a_name():
    word = "".join(chr(c) for c in (99, 97, 108, 109, 115))
    assert hashlib.sha256(word.encode()).hexdigest() in _PRIVATE


@pytest.mark.parametrize("tool", TOOLS)
def test_every_tool_says_how_to_use_it(tool):
    rc, out, _ = _run(tool, "--help")
    assert rc == 0 and "usage:" in out
