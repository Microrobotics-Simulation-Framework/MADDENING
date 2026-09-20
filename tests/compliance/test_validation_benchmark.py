"""Tests for the @verification_benchmark decorator and registry."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from maddening.core.compliance.validation import (
    verification_benchmark,
    BenchmarkType,
    get_benchmark_registry,
    _BENCHMARK_REGISTRY,
)


class TestVerificationBenchmark:
    def test_decorator_preserves_function(self):
        @verification_benchmark(
            benchmark_id="TEST-VER-001",
            description="Test benchmark",
            node_type="TestNode",
            benchmark_type=BenchmarkType.ANALYTICAL,
            acceptance_criteria="error < 1e-3",
        )
        def test_func():
            return 42

        assert test_func() == 42

    def test_decorator_registers_benchmark(self):
        @verification_benchmark(
            benchmark_id="TEST-VER-002",
            description="Another test",
            node_type="TestNode",
            benchmark_type=BenchmarkType.REGRESSION,
            acceptance_criteria="outputs match",
        )
        def test_func2():
            pass

        assert "TEST-VER-002" in _BENCHMARK_REGISTRY
        bm = _BENCHMARK_REGISTRY["TEST-VER-002"]
        assert bm.description == "Another test"
        assert bm.benchmark_type == BenchmarkType.REGRESSION

    def test_benchmark_has_function_name(self):
        @verification_benchmark(
            benchmark_id="TEST-VER-003",
            description="Test",
            node_type="TestNode",
            benchmark_type=BenchmarkType.ANALYTICAL,
            acceptance_criteria="test",
        )
        def my_test_function():
            pass

        bm = _BENCHMARK_REGISTRY["TEST-VER-003"]
        assert "my_test_function" in bm.test_function

    def test_get_benchmark_registry(self):
        reg = get_benchmark_registry()
        assert isinstance(reg, dict)
        # Should contain our test benchmarks
        assert "TEST-VER-001" in reg


class TestDuplicateBenchmarkIds:
    """A repeated ``benchmark_id`` must raise, not overwrite.

    Replays the audit mutation: retyping one decorator's benchmark id from
    MADD-VER-011 to MADD-VER-010 -- the shape of a copy-paste slip -- used to
    delete MADD-VER-011 from the registry with no error anywhere.  Regenerating the evidence tables then removed it from
    ``docs/validation/framework_verification.md`` and every compliance test
    stayed green, because nothing checks document -> registry.
    """

    def test_a_second_decorator_claiming_the_same_id_raises(self):
        @verification_benchmark(
            benchmark_id="TEST-VER-DUP",
            description="The first claimant",
            node_type="TestNode",
            benchmark_type=BenchmarkType.ANALYTICAL,
            acceptance_criteria="error < 1e-3",
        )
        def first_benchmark():
            pass

        with pytest.raises(ValueError, match="duplicate verification benchmark_id"):
            @verification_benchmark(
                benchmark_id="TEST-VER-DUP",
                description="The copy-paste slip",
                node_type="TestNode",
                benchmark_type=BenchmarkType.ANALYTICAL,
                acceptance_criteria="error < 1e-3",
            )
            def second_benchmark():
                pass

    def test_the_first_registration_survives_the_rejected_duplicate(self):
        """The point of raising: the original entry is still the evidence."""
        assert _BENCHMARK_REGISTRY["TEST-VER-DUP"].description == (
            "The first claimant"
        )

    def test_the_error_names_both_functions(self):
        @verification_benchmark(
            benchmark_id="TEST-VER-DUP-NAMED",
            description="First",
            node_type="TestNode",
            benchmark_type=BenchmarkType.ANALYTICAL,
            acceptance_criteria="x",
        )
        def the_incumbent():
            pass

        with pytest.raises(ValueError) as excinfo:
            @verification_benchmark(
                benchmark_id="TEST-VER-DUP-NAMED",
                description="Second",
                node_type="TestNode",
                benchmark_type=BenchmarkType.ANALYTICAL,
                acceptance_criteria="x",
            )
            def the_claimant():
                pass

        assert "the_incumbent" in str(excinfo.value)
        assert "the_claimant" in str(excinfo.value)

    def test_re_registering_the_identical_function_is_idempotent(self):
        """A module imported twice under two names must not be an error.

        ``generate_soup_tables.py`` imports the benchmark modules and pytest
        imports them again; only a *different* function claiming the same ID
        is a defect, so the guard keys on the qualified name rather than on
        the ID alone.
        """
        def make():
            @verification_benchmark(
                benchmark_id="TEST-VER-IDEMPOTENT",
                description="Same function, registered twice",
                node_type="TestNode",
                benchmark_type=BenchmarkType.ANALYTICAL,
                acceptance_criteria="x",
            )
            def same_function():
                pass
            return same_function

        make()
        make()  # must not raise
        assert "TEST-VER-IDEMPOTENT" in _BENCHMARK_REGISTRY


class TestBenchmarkType:
    def test_all_types(self):
        assert BenchmarkType.ANALYTICAL.value == "analytical"
        assert BenchmarkType.MANUFACTURED_SOLUTION.value == "manufactured_solution"
        assert BenchmarkType.CONVERGENCE_STUDY.value == "convergence_study"
        assert BenchmarkType.REGRESSION.value == "regression"
        assert BenchmarkType.CROSS_CODE.value == "cross_code"
