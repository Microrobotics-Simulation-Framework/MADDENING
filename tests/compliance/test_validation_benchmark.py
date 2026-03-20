"""Tests for the @verification_benchmark decorator and registry."""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

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


class TestBenchmarkType:
    def test_all_types(self):
        assert BenchmarkType.ANALYTICAL.value == "analytical"
        assert BenchmarkType.MANUFACTURED_SOLUTION.value == "manufactured_solution"
        assert BenchmarkType.CONVERGENCE_STUDY.value == "convergence_study"
        assert BenchmarkType.REGRESSION.value == "regression"
        assert BenchmarkType.CROSS_CODE.value == "cross_code"
