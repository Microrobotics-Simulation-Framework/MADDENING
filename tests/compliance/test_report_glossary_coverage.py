"""The glossary-coverage report finds an untipped term, and only in prose.

``scripts/report_glossary_coverage.py`` lists, for each documentation page,
every glossary term whose first occurrence on the page is not a ``{term}``
role.  It is a report, not a gate, so it exits 0 whatever it finds.  These
tests plant a glossary and a page, and require the one bare use in prose to
be reported and the uses inside a role, inside code and inside a heading
not to be.

Everything runs on hand-written files, in-process except one run as a
user would start it; nothing imports JAX.
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "report_glossary_coverage.py"

GLOSSARY = """\
# Glossary

Prose about the halo and the shard, which is not a term definition.

```{glossary}
:sorted:

Halo
   Cells copied in from a neighbouring shard.

Shard
   One device's slice of a field.

Spectral radius
   The largest modulus among a matrix's eigenvalues.

Coupling group
   Nodes iterated together within a step.

Coupling
   How mutually dependent nodes exchange data.
```
"""


@pytest.fixture(scope="module")
def report():
    spec = importlib.util.spec_from_file_location("_report_glossary_coverage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def glossary(tmp_path):
    path = tmp_path / "glossary.md"
    path.write_text(GLOSSARY, encoding="utf-8")
    return path


def _found(report, glossary, tmp_path, page_text):
    page = tmp_path / "page.md"
    page.write_text(page_text, encoding="utf-8")
    result = report.scan_page(page, report.read_terms(glossary))
    return [(c.term, c.line) for c in result.candidates], result


def test_the_glossary_block_yields_its_terms_and_nothing_else(report, glossary):
    assert report.read_terms(glossary) == [
        "Halo", "Shard", "Spectral radius", "Coupling group", "Coupling"]


def test_only_the_bare_use_in_prose_is_reported(report, glossary, tmp_path):
    """One term with its role, one without, one inside code, one in a heading."""
    found, _ = _found(report, glossary, tmp_path, """\
# The halo exchange

A {term}`coupling group <Coupling group>` iterates its members, and
each member's data lives on one shard.  Inline `spectral radius` is code.

```python
radius = 1  # the spectral radius of the halo
```
""")
    assert found == [("Shard", 4)]


def test_the_first_occurrence_decides_and_plurals_and_line_breaks_match(
    report, glossary, tmp_path
):
    """A role after a bare use does not cover it; ``shards`` is ``Shard``; a
    two-word term split over a line is reported where it starts; and the
    ``coupling`` inside ``coupling groups`` is the longer term only."""
    found, _ = _found(report, glossary, tmp_path, """\
Two Shards meet at a halo, and the {term}`Halo` is the overlap.
The spectral
radius of the coupling groups is small, and so is a {term}`shard`'s.
""")
    assert sorted(found) == [("Coupling group", 3), ("Halo", 1), ("Shard", 1),
                             ("Spectral radius", 2)]


def test_the_parts_of_a_term_may_be_joined_by_a_space_hyphen_dash_or_line_break(report):
    pattern = re.compile(report.term_pattern("Cramér-Rao bound"), re.I)
    for text in ("the Cramér–Rao bounds", "a cramér-rao\nbound", "Cramér Rao bound"):
        assert pattern.search(text), text
    for text in ("Cramér-Raobound", "xCramér-Rao bound", "Cramér-Rao boundary"):
        assert not pattern.search(text), text


def test_links_comments_code_directives_and_front_matter_are_not_prose(
    report, glossary, tmp_path
):
    """A prose directive's body is read; its argument and options are not;
    a code-like directive is not read at all."""
    found, result = _found(report, glossary, tmp_path, """\
---
orphan: halo
html_meta: {"description": "the halo"}
---
See [the halo guide](halo.md), <https://example.org/halo> and src/shard/halo.py.
% a halo in a MyST comment
<!-- a halo in an HTML comment -->
(halo-target)=

```{mermaid}
graph LR; halo --> spectral_radius
```

    an indented block about the halo

```{note} The halo
:class: halo
A shard, in a note, and {term}`no such term <Nonexistent>`.
```
""")
    assert found == [("Shard", 18)]
    assert result.unknown_roles == [(18, "Nonexistent")]


def test_the_report_exits_zero_when_it_finds_candidates(report, glossary, tmp_path, capsys):
    page = tmp_path / "docs" / "page.md"
    page.parent.mkdir()
    page.write_text("A halo.\n", encoding="utf-8")
    notes = tmp_path / "docs" / "release_notes" / "v9.md"
    notes.parent.mkdir()
    notes.write_text("A shard.\n", encoding="utf-8")

    assert report.main(["--glossary", str(glossary), str(tmp_path / "docs")]) == 0
    out = capsys.readouterr().out
    assert "page.md:1: Halo" in out
    assert "v9.md" not in out                                # release notes skipped
    assert "1 candidate(s) on 1 of 1 page(s)" in out

    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--glossary", str(glossary), str(page)],
        capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "page.md:1: Halo" in run.stdout


def test_an_empty_glossary_warns_and_still_exits_zero(report, tmp_path, capsys):
    empty = tmp_path / "empty.md"
    empty.write_text("# No glossary block here\n", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text("A halo.\n", encoding="utf-8")
    assert report.main(["--glossary", str(empty), str(page)]) == 0
    assert "no glossary terms found" in capsys.readouterr().err


def test_the_repository_glossary_and_pages_are_in_scope(report):
    """An empty scope would make the report print nothing and look clean."""
    glossary = REPO_ROOT / "docs" / "glossary.md"
    terms = report.read_terms(glossary)
    # 75 on the 0.4.0 tip; the glossary only grows.
    assert len(terms) >= 75 and {"Node", "Coupling", "SBOM"} <= set(terms)
    pages = report.pages_under([REPO_ROOT / "docs"], {glossary.resolve()})
    names = {p.relative_to(REPO_ROOT / "docs").as_posix() for p in pages}
    assert len(pages) >= 40 and "index.md" in names
    assert "glossary.md" not in names
    assert not any(n.startswith("release_notes/") for n in names)
