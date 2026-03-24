"""
Transform Registry -- named, serializable edge transforms.

Edge transforms are Python callables applied to values as they pass
along edges between nodes.  For USD serialization, transforms must
be registered with a unique name so they can be stored as a string
reference and resolved on deserialization.

Registration follows the same pattern as ``NodeMeta`` and
``@stability``: a decorator registers the transform in a global
registry, and validation catches unregistered transforms at
serialization time.

Usage
-----
Define and register a transform::

    from maddening.core.transforms import register_transform

    @register_transform("extract_right_boundary")
    def extract_right_boundary(T):
        '''Extract the rightmost cell of a temperature array.'''
        return T[-1]

Use in edges (either by name or direct reference)::

    gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                transform="extract_right_boundary")
    # OR:
    gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                transform=extract_right_boundary)

Registration is **optional for local/testing use** -- inline lambdas
keep working.  Registration is **required for USD serialization**.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional


# ------------------------------------------------------------------
# Registry singleton
# ------------------------------------------------------------------

_TRANSFORM_REGISTRY: dict[str, _RegisteredTransform] = {}


class _RegisteredTransform:
    """Internal container for a registered transform."""
    __slots__ = ("name", "fn", "doc")

    def __init__(self, name: str, fn: Callable, doc: str = ""):
        self.name = name
        self.fn = fn
        self.doc = doc


class UnregisteredTransformError(Exception):
    """Raised when an unregistered transform is used in a context
    that requires registration (e.g., USD serialization)."""
    pass


# ------------------------------------------------------------------
# Registration API
# ------------------------------------------------------------------

def register_transform(
    name: str,
    doc: str = "",
) -> Callable:
    """Decorator that registers a transform function by name.

    Parameters
    ----------
    name : str
        Unique registry name.  Used as the serialized identifier
        in USD and other persistence formats.
    doc : str, optional
        Human-readable description (stored as metadata).

    Returns
    -------
    Callable
        The original function, with ``_transform_name`` attribute set.

    Raises
    ------
    ValueError
        If *name* is already registered to a different function.
    """
    def decorator(fn: Callable) -> Callable:
        if name in _TRANSFORM_REGISTRY:
            existing = _TRANSFORM_REGISTRY[name]
            if existing.fn is not fn:
                raise ValueError(
                    f"Transform name '{name}' is already registered to "
                    f"{existing.fn.__qualname__}. Cannot re-register to "
                    f"{fn.__qualname__}."
                )
            return fn
        description = doc or (fn.__doc__ or "").strip().split("\n")[0]
        _TRANSFORM_REGISTRY[name] = _RegisteredTransform(
            name=name, fn=fn, doc=description
        )
        fn._transform_name = name
        return fn
    return decorator


def resolve_transform(name_or_callable):
    """Resolve a transform reference to a callable.

    Parameters
    ----------
    name_or_callable : str or callable or None
        - ``None``: returns ``None``
        - ``str``: looks up the name in the registry
        - callable with ``_transform_name``: returns as-is
        - callable without ``_transform_name``: returns as-is
          (unregistered -- allowed for local use)

    Returns
    -------
    Callable or None

    Raises
    ------
    KeyError
        If *name_or_callable* is a string not found in the registry.
    """
    if name_or_callable is None:
        return None
    if isinstance(name_or_callable, str):
        if name_or_callable not in _TRANSFORM_REGISTRY:
            raise KeyError(
                f"Transform '{name_or_callable}' not found in registry. "
                f"Available: {sorted(_TRANSFORM_REGISTRY.keys())}"
            )
        return _TRANSFORM_REGISTRY[name_or_callable].fn
    return name_or_callable


def get_transform_name(fn: Callable) -> Optional[str]:
    """Return the registered name of a transform, or None if unregistered."""
    if fn is None:
        return None
    if hasattr(fn, "_transform_name"):
        return fn._transform_name
    # Search by identity
    for name, entry in _TRANSFORM_REGISTRY.items():
        if entry.fn is fn:
            return name
    return None


def is_registered(fn: Callable) -> bool:
    """Check if a transform callable is registered."""
    return get_transform_name(fn) is not None


def list_transforms() -> dict[str, str]:
    """Return a dict of {name: description} for all registered transforms."""
    return {
        name: entry.doc
        for name, entry in sorted(_TRANSFORM_REGISTRY.items())
    }


def validate_all_registered(edges: list) -> list[str]:
    """Check that all edge transforms are registered.

    Parameters
    ----------
    edges : list of EdgeSpec
        Edges to validate.

    Returns
    -------
    list of str
        Warning messages for unregistered transforms.
        Empty list if all transforms are registered or None.
    """
    warnings = []
    for edge in edges:
        if edge.transform is not None and not is_registered(edge.transform):
            warnings.append(
                f"Edge {edge.source_node}.{edge.source_field} -> "
                f"{edge.target_node}.{edge.target_field} has an "
                f"unregistered transform ({edge.transform.__qualname__}). "
                f"Use @register_transform for USD serialization support."
            )
    return warnings


# ------------------------------------------------------------------
# Built-in transforms
# ------------------------------------------------------------------

@register_transform("extract_first", "Extract the first element of an array.")
def extract_first(arr):
    """Extract the first element of an array."""
    return arr[0]


@register_transform("extract_last", "Extract the last element of an array.")
def extract_last(arr):
    """Extract the last element of an array."""
    return arr[-1]


@register_transform(
    "extract_second", "Extract the second element (index 1) of an array."
)
def extract_second(arr):
    """Extract element at index 1."""
    return arr[1]


@register_transform(
    "extract_second_last",
    "Extract the second-to-last element (index -2) of an array.",
)
def extract_second_last(arr):
    """Extract element at index -2."""
    return arr[-2]


@register_transform("negate", "Negate the value.")
def negate(x):
    """Negate the value: -x."""
    return -x


def scale(factor: float) -> Callable:
    """Create a registered scaling transform.

    Parameters
    ----------
    factor : float
        Scale factor.

    Returns
    -------
    Callable
        A transform that multiplies by *factor*.

    Note
    ----
    Each unique factor creates a separate registered transform
    named ``"scale_{factor}"``.  For one-off use, prefer an
    inline lambda.
    """
    name = f"scale_{factor}"
    if name in _TRANSFORM_REGISTRY:
        return _TRANSFORM_REGISTRY[name].fn

    def _scale(x, _f=factor):
        return _f * x

    _scale.__qualname__ = f"scale({factor})"
    _scale.__doc__ = f"Multiply by {factor}."
    register_transform(name, f"Multiply by {factor}.")(_scale)
    return _scale


@register_transform("identity", "Pass through unchanged.")
def identity(x):
    """Identity transform (pass through unchanged)."""
    return x


# ------------------------------------------------------------------
# Re-export unit conversion factories for discoverability
# ------------------------------------------------------------------
# The canonical import path for unit transforms is:
#     from maddening.core.transforms import lbm_to_si_force
# The implementation lives in transforms_unit.py.

from maddening.core.transforms_unit import (  # noqa: E402, F401
    lbm_to_si_force,
    lbm_to_si_length,
    lbm_to_si_pressure,
    lbm_to_si_torque,
    lbm_to_si_velocity,
    si_to_lbm_force,
)
