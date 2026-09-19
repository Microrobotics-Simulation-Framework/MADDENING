"""Property tests for the parsers behind the compliance gates.

The gates are only as good as the three parsers underneath them: the BibTeX
key parser, the ``[@Key]`` citation extractor, and the AST walk that finds
string edge-transform references.  A mutation audit found each of them
accepting a defect it was written to reject, so these properties pin the
parsers directly, over generated inputs, rather than over the one
repository state that happens to be checked in.
"""

import importlib.util
import os
import tempfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import yaml
from hypothesis import assume, given, strategies as st

from maddening.compliance._validate import (
    _VALID_RESOLUTION_STATUSES,
    _VALID_SAFETY_RELEVANCES,
    _VALID_SEVERITIES,
    validate_anomaly_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_parser_{name}", SCRIPTS / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CITATIONS = _load("check_citations")
TRANSFORMS = _load("check_transforms")

# Keys that survive a BibTeX round trip and the citation extractor's
# trailing-punctuation strip: start with a letter, end with alphanumeric.
bib_keys = st.from_regex(r"\A[A-Za-z][A-Za-z0-9-]{0,8}[A-Za-z0-9]\Z", fullmatch=True)
key_sets = st.lists(bib_keys, min_size=1, max_size=6, unique=True)


# ---------------------------------------------------------------------------
# Bibliography parsing
# ---------------------------------------------------------------------------

@given(live=key_sets, commented=key_sets)
def test_a_commented_entry_never_counts_as_a_defined_key(live, commented):
    """BibTeX ignores a ``%`` line, so the gate must ignore it too."""
    commented = [k for k in commented if k not in live]
    assume(commented)
    lines = []
    for key in live:
        lines.append("@book{%s,\n  title = {T},\n}\n" % key)
    for key in commented:
        lines.append("%% @book{%s,\n%%  title = {T},\n%%}\n" % key)

    with tempfile.TemporaryDirectory() as tmp:
        bib = os.path.join(tmp, "bibliography.bib")
        with open(bib, "w") as f:
            f.write("\n".join(lines))
        assert CITATIONS.parse_bib_keys(bib) == set(live)


@given(keys=key_sets, repeated=st.integers(min_value=2, max_value=4))
def test_a_key_defined_more_than_once_is_reported_as_duplicate(keys, repeated):
    doubled = keys[0]
    entries = list(keys) + [doubled] * (repeated - 1)
    with tempfile.TemporaryDirectory() as tmp:
        bib = os.path.join(tmp, "bibliography.bib")
        with open(bib, "w") as f:
            for key in entries:
                f.write("@book{%s,\n  title = {T},\n}\n\n" % key)
        parsed = CITATIONS.parse_bib_entries(bib)
        duplicates = CITATIONS.find_duplicate_keys(parsed)
    assert len(duplicates) == 1
    assert doubled in duplicates[0]


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

@given(keys=key_sets)
def test_every_key_in_a_multi_citation_bracket_is_extracted(keys):
    """``[@A; @B; @C]`` cites three works, not one."""
    line = "As shown [" + "; ".join(f"@{k}" for k in keys) + "] the result holds.\n"
    with tempfile.TemporaryDirectory() as tmp:
        md = os.path.join(tmp, "doc.md")
        with open(md, "w") as f:
            f.write(line)
        found = [key for _lineno, key in CITATIONS.extract_citations(md)]
    assert found == list(keys)


@given(defined=key_sets, cited=key_sets)
def test_the_gate_fails_exactly_when_a_cited_key_is_undefined(defined, cited):
    dangling = set(cited) - set(defined)
    with tempfile.TemporaryDirectory() as tmp:
        bib = os.path.join(tmp, "bibliography.bib")
        with open(bib, "w") as f:
            for key in defined:
                f.write("@book{%s,\n  title = {T},\n}\n\n" % key)
        docs = os.path.join(tmp, "docs")
        os.mkdir(docs)
        with open(os.path.join(docs, "doc.md"), "w") as f:
            for key in cited:
                f.write(f"See [@{key}].\n")

        os.environ["BIB_PATH"] = bib
        try:
            rc = CITATIONS.main([docs])
        finally:
            del os.environ["BIB_PATH"]

    assert rc == (1 if dangling else 0)


# ---------------------------------------------------------------------------
# Transform reference scanning
# ---------------------------------------------------------------------------

transform_names = st.from_regex(r"\A[a-z][a-z0-9_]{0,12}\Z", fullmatch=True)


@given(names=st.lists(transform_names, min_size=1, max_size=5, unique=True))
def test_an_edge_transform_string_is_found_wherever_the_call_is_written(names):
    """add_edge and EdgeSpec, attribute or bare call, all count."""
    source = "".join(
        f'gm.add_edge("a", "b", "x", "y", transform="{n}")\n'
        f'EdgeSpec(source="a", target="b", transform="{n}")\n'
        for n in names
    )
    with tempfile.TemporaryDirectory() as tmp:
        py = Path(tmp) / "graph.py"
        py.write_text(source)
        refs, _local = TRANSFORMS.scan_file(py)
    assert sorted({name for _lineno, name in refs}) == sorted(names)


@given(
    registered=st.lists(transform_names, min_size=1, max_size=4, unique=True),
    ghost=transform_names,
)
def test_a_file_registering_its_own_transforms_satisfies_its_own_references(
    registered, ghost
):
    assume(ghost not in registered)
    body = "".join(
        f'@register_transform("{n}")\ndef _f_{n}(x):\n    return x\n\n'
        for n in registered
    )
    body += "".join(
        f'gm.add_edge("a", "b", "x", "y", transform="{n}")\n' for n in registered
    )
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "ok.py").write_text(body)
        assert TRANSFORMS.main([tmp]) == 0

        (Path(tmp) / "bad.py").write_text(
            f'gm.add_edge("a", "b", "x", "y", transform="{ghost}")\n'
        )
        assert TRANSFORMS.main([tmp]) == 1


# ---------------------------------------------------------------------------
# Anomaly records
# ---------------------------------------------------------------------------

text = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=40
).map(str.strip).filter(bool)

anomaly_records = st.fixed_dictionaries({
    "anomaly_id": st.from_regex(r"\AMADD-ANO-[0-9]{3}\Z", fullmatch=True),
    "title": text,
    "description": text,
    "severity": st.sampled_from(sorted(_VALID_SEVERITIES)),
    "safety_relevance": st.sampled_from(sorted(_VALID_SAFETY_RELEVANCES)),
    "safety_relevance_rationale": text,
    "resolution_status": st.sampled_from(sorted(_VALID_RESOLUTION_STATUSES)),
})


def _validate(record, **kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "known_anomalies.yaml")
        with open(path, "w") as f:
            yaml.safe_dump({
                "schema_version": "1.0",
                "generated_date": "2026-03-12",
                "anomalies": [record],
            }, f)
        return validate_anomaly_registry(path, **kwargs)


@given(record=anomaly_records)
def test_a_well_formed_anomaly_record_validates_clean(record):
    assert _validate(record, prefix="MADD-ANO-") == []


@given(
    record=anomaly_records,
    dropped=st.sampled_from([
        "title", "description", "severity", "safety_relevance",
        "safety_relevance_rationale", "resolution_status",
    ]),
)
def test_dropping_any_required_field_is_reported(record, dropped):
    del record[dropped]
    errors = _validate(record)
    assert any(dropped in e for e in errors), errors


@given(record=anomaly_records, bogus=text)
def test_an_unrecognised_resolution_status_is_reported(record, bogus):
    assume(bogus not in _VALID_RESOLUTION_STATUSES)
    record["resolution_status"] = bogus
    errors = _validate(record)
    assert any("resolution_status" in e for e in errors), errors


@given(
    record=anomaly_records,
    typo=st.sampled_from(["workaroud", "resolution_stat", "affected_component", "notes"]),
)
def test_an_unknown_field_is_reported_rather_than_silently_dropped(record, typo):
    """``workaroud:`` used to drop the workaround with exit 0."""
    record[typo] = "something a reader would have relied on"
    errors = _validate(record)
    assert any(typo in e for e in errors), errors


@given(record=anomaly_records, missing=st.from_regex(r"\A[A-Z][a-z]{2,10}\Z", fullmatch=True))
def test_an_affected_component_that_does_not_import_is_reported(record, missing):
    record["affected_components"] = [f"maddening.nodes.heat.{missing}Node"]
    errors = _validate(record)
    assert any("affected_components" in e for e in errors), errors


@given(record=anomaly_records, name=st.from_regex(r"\A[a-z_]{3,12}\Z", fullmatch=True))
def test_a_verification_entry_naming_no_real_test_is_reported(record, name):
    record["resolution_status"] = "resolved"
    record["verification"] = [f"tests/compliance/test_validator.py::test_{name}_absent"]
    errors = _validate(record, repo_root=str(REPO_ROOT))
    assert any("verification" in e for e in errors), errors


def test_verification_entries_cannot_be_skipped_for_want_of_a_repo_root():
    """Silently not resolving is the failure mode being fixed, not a pass."""
    record = {
        "anomaly_id": "MADD-ANO-001",
        "title": "T", "description": "D", "severity": "minor",
        "safety_relevance": "not_safety_relevant",
        "safety_relevance_rationale": "R",
        "resolution_status": "resolved",
        "verification": ["tests/does_not_matter.py::test_x"],
    }
    errors = _validate(record, repo_root=None)
    assert any("repository root" in e for e in errors), errors
