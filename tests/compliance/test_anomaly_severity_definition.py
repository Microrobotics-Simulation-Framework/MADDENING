"""Every anomaly severity is defined, in one place, with its rule.

Until 0.4.0 the only written definition of a severity was a comment in
DOCUMENTATION_ARCHITECTURE.md ("minor: cosmetic or minor inconvenience"),
and registry entries whose results were silently wrong carried ``minor``
against it.  ``AnomalySeverity``'s docstring is now the definition and the
architecture document repeats it.  This pins that each level is defined
there, that the rule for silent wrong results is stated in both, and that a
level added to the enum cannot ship without a definition.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from maddening.core.compliance.anomaly import AnomalySeverity

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE = REPO_ROOT / "DOCUMENTATION_ARCHITECTURE.md"
RULE = "a silent wrong result is never minor"


def _flat(text: str) -> str:
    """Lower-cased, markup-free, whitespace-collapsed text."""
    return re.sub(r"\s+", " ", re.sub(r"[`*]", "", text)).lower()


def _defined_terms(doc: str) -> set[str]:
    """The NumPy-style definition-list terms of a class docstring."""
    return {line.strip() for line in doc.splitlines()
            if re.fullmatch(r"    [A-Z]+", line)}


@pytest.mark.parametrize("level", list(AnomalySeverity), ids=lambda m: m.name)
def test_every_severity_is_defined_in_the_enum_docstring(level):
    assert level.name in _defined_terms(AnomalySeverity.__doc__ or "")


def test_the_docstring_defines_no_level_the_enum_lacks():
    assert _defined_terms(AnomalySeverity.__doc__ or "") == {
        m.name for m in AnomalySeverity}


def test_the_rule_for_silent_wrong_results_is_stated_in_both_places():
    assert RULE in _flat(AnomalySeverity.__doc__ or "")
    assert RULE in _flat(ARCHITECTURE.read_text())


@pytest.mark.parametrize("level", list(AnomalySeverity), ids=lambda m: m.name)
def test_the_architecture_document_tabulates_every_severity(level):
    rows = re.findall(r"^\| `([a-z]+)` \|", ARCHITECTURE.read_text(), re.M)
    assert level.value in rows


def test_the_old_one_line_definition_of_minor_is_gone():
    """The schema comment the audit found contradicting the registry (the
    prose that tells its history may still quote it)."""
    assert not re.search(r'MINOR = "minor"\s*# Cosmetic or minor inconvenience',
                         ARCHITECTURE.read_text())
