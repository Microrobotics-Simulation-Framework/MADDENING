"""
Stability decorator and registry (Section 9.5).

Phase 0-3: ``@stability`` is an identity decorator that attaches a
``_stability_level`` attribute but does not populate a registry or
generate reports.

Phase 4: The decorator populates ``_STABILITY_REGISTRY`` and
``generate_stability_report()`` produces valid Markdown output.

Pure Python — no JAX dependency.
"""

from __future__ import annotations

from typing import Callable, Optional, TypeVar
import functools

from maddening.core.metadata import StabilityLevel

T = TypeVar("T")

# Global registry — populated by the decorator
_STABILITY_REGISTRY: dict[str, StabilityLevel] = {}


def stability(level: StabilityLevel) -> Callable[[T], T]:
    """Decorator marking a class or function with its API stability level.

    Usage::

        from maddening.core.metadata import StabilityLevel
        from maddening.core.stability import stability

        @stability(StabilityLevel.STABLE)
        class GraphManager:
            ...
    """
    def decorator(obj: T) -> T:
        obj._stability_level = level  # type: ignore[attr-defined]
        # Register with qualified name
        qual_name = getattr(obj, "__qualname__", getattr(obj, "__name__", str(obj)))
        module = getattr(obj, "__module__", "")
        full_name = f"{module}.{qual_name}" if module else qual_name
        _STABILITY_REGISTRY[full_name] = level
        return obj
    return decorator


def generate_stability_report() -> str:
    """Generate a Markdown stability report from the registry.

    Returns
    -------
    str
        Markdown-formatted stability report.
    """
    if not _STABILITY_REGISTRY:
        return "# Stability Report\n\nNo API surfaces registered.\n"

    lines = ["# Stability Report\n"]
    lines.append("| API Surface | Stability Level |")
    lines.append("|---|---|")

    for name, level in sorted(_STABILITY_REGISTRY.items()):
        lines.append(f"| `{name}` | {level.value} |")

    lines.append(f"\n*{len(_STABILITY_REGISTRY)} API surfaces registered.*\n")
    return "\n".join(lines)
