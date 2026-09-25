"""A reference into an uninstalled optional subpackage is unverified, not stale.

`resolve_dotted_name` walks a dotted name's module prefixes from longest to
shortest.  When `maddening.usd.serialization` fails to import because the
`usd` extra is absent, `maddening` still imports, and the accumulated import
error used to be discarded at that point -- so the gate reported
`'maddening' has no attribute 'usd'` and failed a *legitimate* reference.
A submodule is not an attribute of its package until it is imported, so the
attribute walk was never going to succeed; the message described the symptom
and blamed the wrong thing.

That CI failure needed three things at once: the resolver, an anomaly naming
a `maddening.usd` symbol, and a job installing `.[ci]` without the USD
extra.  These tests need none of them -- they build a throwaway package with
the same import shapes, so the behaviour is pinned in every environment,
including one where `pxr` happens to be installed (which is why local runs
of the four gates could not see it).
"""

import importlib
import os
import sys
import textwrap

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import yaml

from maddening.compliance._validate import (
    resolve_dotted_name,
    validate_anomaly_registry,
)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    """A package whose subpackages reproduce every import shape that matters.

    * ``present`` imports cleanly.
    * ``guarded`` raises the friendly ``ImportError`` that
      ``maddening/usd/__init__.py`` raises -- no ``name`` attribute at all.
    * ``bare`` raises a plain ``ModuleNotFoundError`` for a third-party
      package, the other shape the same situation takes.
    * ``broken`` raises a non-import error, which must stay a hard failure.
    * ``stale`` imports a name its own package no longer has -- a stale
      internal rename, the shape audit_040_p4_1's A25 seeded.
    * ``missing_sub`` imports a first-party module that does not exist.
    * ``wrapped`` is an extra's guard around a *first-party* failure, as
      ``maddening.viz``'s lazy loader re-raises any ``ImportError``.
    * ``chained`` is the real ``maddening.usd`` shape: the guard re-raises
      ``from`` the third-party ``ModuleNotFoundError``.
    """
    root = tmp_path / "pkgroot"
    pkg = root / "fakelib"
    _write(pkg / "__init__.py", "")
    _write(pkg / "present" / "__init__.py", """
        def a_function():
            return 1

        A_CONSTANT = 2
    """)
    _write(pkg / "guarded" / "__init__.py", """
        raise ImportError(
            "fakelib.guarded requires 'fake-extra'. "
            "Install with:  pip install fakelib[guarded]"
        )
    """)
    _write(pkg / "guarded" / "inner.py", "def writes_a_stage(): pass\n")
    _write(pkg / "bare" / "__init__.py", "import definitely_not_installed_xyz\n")
    _write(pkg / "bare" / "inner.py", "def also_here(): pass\n")
    _write(pkg / "broken" / "__init__.py", "raise RuntimeError('this module is broken')\n")
    _write(pkg / "stale" / "__init__.py",
           "from fakelib.present import gone_by_rename\n\nTHING = 1\n")
    _write(pkg / "missing_sub" / "__init__.py", "import fakelib.no_such_module\n")
    _write(pkg / "wrapped" / "__init__.py", """
        try:
            from fakelib.present import gone_by_rename
        except ImportError as _exc:
            raise ImportError(
                "fakelib.wrapped requires 'fake-extra'. "
                "Install with:  pip install fakelib[wrapped]"
            ) from _exc
    """)
    _write(pkg / "chained" / "__init__.py", """
        try:
            import definitely_not_installed_xyz
        except ImportError as _exc:
            raise ImportError(
                "fakelib.chained requires 'fake-extra'. "
                "Install with:  pip install fakelib[chained]"
            ) from _exc
    """)
    _write(pkg / "chained" / "inner.py", "def writes_a_stage(): pass\n")

    monkeypatch.syspath_prepend(str(root))
    importlib.invalidate_caches()
    yield "fakelib"
    for name in [m for m in list(sys.modules) if m.split(".")[0] == "fakelib"]:
        del sys.modules[name]


class TestUnavailableIsNotStale:
    def test_a_guarded_subpackage_reports_unavailable_not_missing_attribute(
        self, fake_package
    ):
        """The exact shape of `maddening.usd`: ImportError, no `name`."""
        res = resolve_dotted_name("fakelib.guarded.inner.writes_a_stage")
        assert not res.ok
        assert res.unavailable == "fakelib.guarded"
        # The old message; it must not come back.
        assert "has no attribute" not in res.reason
        assert "could not be checked" in res.reason
        # The remedy the subpackage itself names is carried through.
        assert "pip install fakelib[guarded]" in res.reason

    def test_a_missing_third_party_package_reports_unavailable(self, fake_package):
        """The other shape: a bare ModuleNotFoundError naming the package."""
        res = resolve_dotted_name("fakelib.bare.inner.also_here")
        assert not res.ok
        assert res.unavailable == "definitely_not_installed_xyz"
        assert "has no attribute" not in res.reason

    def test_a_genuinely_stale_reference_is_still_stale(self, fake_package):
        """The fix must not turn real rot into a shrug."""
        res = resolve_dotted_name("fakelib.present.no_such_function")
        assert not res.ok
        assert res.unavailable is None
        assert "has no attribute 'no_such_function'" in res.reason

    def test_a_misspelled_subpackage_is_still_stale(self, fake_package):
        res = resolve_dotted_name("fakelib.no_such_sub.thing")
        assert not res.ok
        assert res.unavailable is None

    def test_a_module_that_raises_a_non_import_error_is_a_hard_failure(
        self, fake_package
    ):
        """A broken module is a defect, not an absent optional dependency."""
        res = resolve_dotted_name("fakelib.broken.anything")
        assert not res.ok
        assert res.unavailable is None
        assert "RuntimeError" in res.reason

    def test_a_resolvable_reference_is_unaffected(self, fake_package):
        assert resolve_dotted_name("fakelib.present.a_function").ok
        assert resolve_dotted_name(
            "fakelib.present.a_function", require_callable=True
        ).ok
        assert not resolve_dotted_name(
            "fakelib.present.A_CONSTANT", require_callable=True
        ).ok


class TestAFirstPartyBreakIsNotAMissingExtra:
    """A broken import of the package's *own* code is a defect.

    ``cannot import name 'x' from 'maddening.core.solver_utils'`` carries
    ``name='maddening.core.solver_utils'``; the resolver labelled it with
    its top-level package, ``maddening``, and reported every reference
    behind the module "not checked (missing optional extra)" -- so a stale
    internal rename left the anomaly gate at exit 0 (audit_040_p4_1, A25).
    """

    @pytest.mark.parametrize("qname, culprit", [
        ("fakelib.stale.THING", "gone_by_rename"),
        ("fakelib.missing_sub.anything", "fakelib.no_such_module"),
        ("fakelib.wrapped.anything", "gone_by_rename"),
    ])
    def test_a_first_party_import_failure_is_a_hard_failure(
        self, fake_package, qname, culprit
    ):
        res = resolve_dotted_name(qname)
        assert not res.ok
        assert res.unavailable is None, res
        assert "broken first-party import" in res.reason
        assert culprit in res.reason

    def test_an_extra_guard_chained_from_a_third_party_package_is_unavailable(
        self, fake_package
    ):
        """The control: the real ``maddening.usd`` shape is still a skip."""
        res = resolve_dotted_name("fakelib.chained.inner.writes_a_stage")
        assert not res.ok
        assert res.unavailable == "fakelib.chained"
        assert "pip install fakelib[chained]" in res.reason

    def test_the_anomaly_validator_reports_it_as_an_error_not_a_note(
        self, fake_package, tmp_path
    ):
        notes = []
        path = _registry(tmp_path, ["fakelib.stale.THING"])
        errors = validate_anomaly_registry(path, notes=notes)
        assert notes == []
        assert len(errors) == 1 and "broken first-party import" in errors[0]


def _registry(tmp_path, components):
    data = {
        "schema_version": "1.0",
        "generated_date": "2026-03-12",
        "anomalies": [{
            "anomaly_id": "MADD-ANO-001",
            "title": "T",
            "description": "D",
            "severity": "minor",
            "safety_relevance": "not_safety_relevant",
            "safety_relevance_rationale": "R",
            "resolution_status": "open",
            "affected_components": components,
        }],
    }
    path = tmp_path / "known_anomalies.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


class TestTheAnomalyGateSkipsRatherThanLies:
    def test_an_unavailable_component_is_a_note_not_an_error(
        self, fake_package, tmp_path
    ):
        notes = []
        path = _registry(tmp_path, ["fakelib.guarded.inner.writes_a_stage"])
        errors = validate_anomaly_registry(path, notes=notes)
        assert errors == []
        assert len(notes) == 1
        assert "was NOT checked" in notes[0]

    def test_a_stale_component_is_still_an_error(self, fake_package, tmp_path):
        notes = []
        path = _registry(tmp_path, ["fakelib.present.no_such_function"])
        errors = validate_anomaly_registry(path, notes=notes)
        assert len(errors) == 1
        assert notes == []

    def test_the_notes_list_is_optional(self, fake_package, tmp_path):
        """Downstream callers on the old signature must not break."""
        path = _registry(tmp_path, ["fakelib.guarded.inner.writes_a_stage"])
        assert validate_anomaly_registry(path) == []


class TestTheMappingGateSkipsRatherThanLies:
    def test_an_unavailable_mapping_row_is_skipped_and_not_counted(
        self, fake_package, tmp_path, monkeypatch
    ):
        import importlib.util
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location(
            "_gate_mapping_optional", repo_root / "scripts" / "check_impl_mapping.py"
        )
        gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gate)
        # The gate only reads `maddening.*` spans; point it at the fake
        # package by widening its pattern for this test alone.
        monkeypatch.setattr(
            gate, "_QNAME", __import__("re").compile(r"`(fakelib\.[^`]+)`")
        )

        guide = tmp_path / "guide.md"
        guide.write_text(
            "## Implementation Mapping\n\n"
            "| Equation Term | Implementation | Notes |\n"
            "|---------------|---------------|-------|\n"
            "| Stage write | `fakelib.guarded.inner.writes_a_stage` | |\n"
            "\n## Next\n"
        )
        checked, errors, _notes, skipped = gate.check_guide(str(guide), "guide.md")
        assert errors == []
        assert checked == 0, "an unchecked reference must not count as verified"
        assert len(skipped) == 1
        assert "was NOT checked" in skipped[0]


class TestTheRepositoryItselfIsFullyChecked:
    """The skip is a safety valve, not a place for coverage to leak away."""

    def test_the_shipped_registry_has_no_unverified_component(self):
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        registry = os.path.join(
            repo_root, "docs", "validation", "known_anomalies.yaml"
        )
        if not os.path.exists(registry):
            pytest.skip("known_anomalies.yaml not found")
        try:
            importlib.import_module("maddening.usd")
        except ImportError as exc:
            pytest.skip(
                "this environment cannot import every optional subpackage, so "
                f"it cannot prove the registry is fully checked: {exc}"
            )
        notes = []
        errors = validate_anomaly_registry(
            registry, prefix="MADD-ANO-", repo_root=repo_root, notes=notes
        )
        assert errors == []
        assert notes == [], (
            "a reference went unverified in a fully provisioned environment, "
            f"which means it is genuinely unresolvable: {notes}"
        )
