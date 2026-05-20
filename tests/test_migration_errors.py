"""Tests for v0.2 #1 follow-up: MigrationError pattern."""

from __future__ import annotations

import pytest

from maddening.warnings import (
    EdgeValidationWarning,
    MigrationError,
    UnitMismatchWarning,
)


class TestMigrationError:
    def test_minimal_construction(self):
        err = MigrationError("SomeAPI.foo")
        assert err.api_name == "SomeAPI.foo"
        assert err.affected_class is None
        assert err.replacement is None
        assert err.migration_guide is None
        assert "SomeAPI.foo was removed." in str(err)

    def test_full_construction(self):
        class _DummyClass:
            pass

        err = MigrationError(
            "SimulationNode.requires_halo",
            affected_class=_DummyClass,
            replacement="halo_width() -> dict[int, int]",
            migration_guide="https://example.com/migrate.html",
        )
        msg = str(err)
        assert "SimulationNode.requires_halo was removed" in msg
        assert "_DummyClass" in msg
        assert "halo_width() -> dict[int, int]" in msg
        assert "https://example.com/migrate.html" in msg

    def test_is_runtimeerror_subclass(self):
        # Callers that already catch RuntimeError see migration errors.
        assert issubclass(MigrationError, RuntimeError)

    def test_structured_fields_introspectable(self):
        # An auto-migration tool can read structured fields rather than
        # parsing the error message string.
        class _C:
            pass

        err = MigrationError(
            "Foo.bar",
            affected_class=_C,
            replacement="Foo.baz",
            migration_guide="https://example.com",
        )
        assert err.api_name == "Foo.bar"
        assert err.affected_class is _C
        assert err.replacement == "Foo.baz"
        assert err.migration_guide == "https://example.com"

    def test_raise_and_catch(self):
        with pytest.raises(MigrationError, match="Foo.bar"):
            raise MigrationError(
                "Foo.bar",
                replacement="Foo.baz",
            )


class TestUnitMismatchPermanentlyAdvisory:
    """v0.2 #4 follow-up: UnitMismatchWarning's docstring closes the
    'do units flip to errors' question.  This test pins the contract
    in code so future-you can't accidentally promote it."""

    def test_docstring_calls_out_permanently_advisory(self):
        assert "Permanently advisory" in UnitMismatchWarning.__doc__

    def test_inheritance_unchanged(self):
        # If unit mismatches stay warnings, they stay rooted at
        # EdgeValidationWarning, NOT at any future error hierarchy.
        assert issubclass(UnitMismatchWarning, EdgeValidationWarning)
        assert issubclass(UnitMismatchWarning, Warning)
        assert not issubclass(UnitMismatchWarning, Exception) or \
            issubclass(Warning, Exception)
