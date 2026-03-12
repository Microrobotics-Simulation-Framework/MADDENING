"""
Verification benchmark registry (Section 9.3).

Provides the ``@verification_benchmark`` decorator and ``ValidationBenchmark``
dataclass for registering and discovering verification benchmarks.

Pure Python — no JAX dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional
import functools


class BenchmarkType(Enum):
    """Classification of verification benchmark."""
    ANALYTICAL = "analytical"
    MANUFACTURED_SOLUTION = "manufactured_solution"
    CONVERGENCE_STUDY = "convergence_study"
    REGRESSION = "regression"
    CROSS_CODE = "cross_code"


@dataclass(frozen=True)
class ValidationBenchmark:
    """Metadata for a registered verification benchmark."""
    benchmark_id: str
    description: str
    node_type: str
    benchmark_type: BenchmarkType
    acceptance_criteria: str
    references: tuple[str, ...] = ()
    test_function: Optional[str] = None  # qualified name


# Global registry
_BENCHMARK_REGISTRY: dict[str, ValidationBenchmark] = {}


def verification_benchmark(
    benchmark_id: str,
    description: str,
    node_type: str,
    benchmark_type: BenchmarkType,
    acceptance_criteria: str,
    references: tuple[str, ...] = (),
) -> Callable:
    """Decorator that registers a test function as a verification benchmark.

    Usage::

        @verification_benchmark(
            benchmark_id="MADD-VER-001",
            description="Poiseuille flow analytical benchmark",
            node_type="LBMPipeNode",
            benchmark_type=BenchmarkType.ANALYTICAL,
            acceptance_criteria="L2 error < 1e-3",
        )
        def test_poiseuille_flow():
            ...
    """
    def decorator(func: Callable) -> Callable:
        qual_name = f"{func.__module__}.{func.__qualname__}"
        benchmark = ValidationBenchmark(
            benchmark_id=benchmark_id,
            description=description,
            node_type=node_type,
            benchmark_type=benchmark_type,
            acceptance_criteria=acceptance_criteria,
            references=references,
            test_function=qual_name,
        )
        _BENCHMARK_REGISTRY[benchmark_id] = benchmark

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper._benchmark = benchmark
        return wrapper
    return decorator


def get_benchmark_registry() -> dict[str, ValidationBenchmark]:
    """Return a copy of the current benchmark registry."""
    return dict(_BENCHMARK_REGISTRY)
