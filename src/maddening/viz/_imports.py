"""Lazy import helpers for optional visualization dependencies.

Centralises the try/except + error message pattern so individual
modules can write ``pv = _import_pyvista()`` instead of duplicating
the guard everywhere.
"""


def _import_pyvista():
    """Import pyvista with a clear error if not installed."""
    try:
        import pyvista as pv
        return pv
    except ImportError as exc:
        raise ImportError(
            "This feature requires PyVista. "
            "Install with:  pip install maddening[viz3d]"
        ) from exc


def _import_pxr(*names):
    """Import pxr modules with a clear error if not installed.

    Usage::

        Usd, Sdf = _import_pxr("Usd", "Sdf")
    """
    try:
        import pxr
        return tuple(getattr(pxr, n) for n in names)
    except ImportError as exc:
        raise ImportError(
            "This feature requires usd-core. "
            "Install with:  pip install maddening[usd]"
        ) from exc
