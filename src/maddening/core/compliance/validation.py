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
        existing = _BENCHMARK_REGISTRY.get(benchmark_id)
        if existing is not None and existing.test_function != qual_name:
            # A plain dict assignment here meant a copy-pasted benchmark_id
            # silently *deleted* the first benchmark.  Nothing downstream
            # noticed: every check runs registry -> document, so once the
            # registry had shrunk, regenerating the evidence tables made the
            # committed document agree with it again and the lost benchmark
            # was simply absent from the IEC 62304 verification index.
            # Import time is where this belongs -- the second decorator is
            # the defect, and it cannot execute without saying so.
            raise ValueError(
                f"duplicate verification benchmark_id {benchmark_id!r}: "
                f"already registered by {existing.test_function}, now claimed "
                f"by {qual_name}.  Benchmark IDs are IEC 62304 evidence "
                f"identifiers and must be unique; give the new benchmark the "
                f"next free ID rather than overwriting an existing one."
            )
        _BENCHMARK_REGISTRY[benchmark_id] = benchmark

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper._benchmark = benchmark  # type: ignore[attr-defined]
        return wrapper
    return decorator


def get_benchmark_registry() -> dict[str, ValidationBenchmark]:
    """Return a copy of the current benchmark registry."""
    return dict(_BENCHMARK_REGISTRY)
