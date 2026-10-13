"""Visualization backends for MADDENING.

Backends are imported lazily to avoid pulling in matplotlib / rich
when only one backend (or none) is needed.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # PEP 562: the names below are resolved lazily by the module
    # `__getattr__`, which a type checker cannot see through -- without
    # these re-exports it reports every one of them as missing from
    # `__all__`, and a downstream `from maddening.viz.backends import
    # X` is an error.
    # Nothing here runs at import time; the lazy table stays the only
    # runtime path.
    from maddening.viz.backends.matplotlib_renderer import (
        MatplotlibRenderer,
        MatplotlibSceneRenderer,
        MatplotlibTimeSeriesRenderer,
        run_matplotlib,
    )
    from maddening.viz.backends.pygfx_viewer import GPUHistoryViewer
    from maddening.viz.backends.selkies_renderer import SelkiesRenderer
    from maddening.viz.backends.terminal_renderer import TerminalRenderer


_INSTALL_HINTS = {
    "MatplotlibTimeSeriesRenderer": "viz",
    "MatplotlibSceneRenderer": "viz",
    "MatplotlibRenderer": "viz",
    "run_matplotlib": "viz",
    "TerminalRenderer": "terminal",
    "GPUHistoryViewer": "gpu-viz",
    "SelkiesRenderer": "streaming",
}


def __getattr__(name: str) -> Any:
    _lazy = {
        "MatplotlibTimeSeriesRenderer": "maddening.viz.backends.matplotlib_renderer",
        "MatplotlibSceneRenderer": "maddening.viz.backends.matplotlib_renderer",
        "MatplotlibRenderer": "maddening.viz.backends.matplotlib_renderer",
        "run_matplotlib": "maddening.viz.backends.matplotlib_renderer",
        "TerminalRenderer": "maddening.viz.backends.terminal_renderer",
        "GPUHistoryViewer": "maddening.viz.backends.pygfx_viewer",
        "SelkiesRenderer": "maddening.viz.backends.selkies_renderer",
    }
    if name in _lazy:
        try:
            import importlib
            mod = importlib.import_module(_lazy[name])
            return getattr(mod, name)
        except ImportError as exc:
            extra = _INSTALL_HINTS.get(name, "")
            raise ImportError(
                f"'{name}' requires additional dependencies. "
                f"Install with:  pip install maddening[{extra}]"
            ) from exc
    raise AttributeError(f"module 'maddening.viz.backends' has no attribute {name!r}")


__all__ = [
    "MatplotlibTimeSeriesRenderer",
    "MatplotlibSceneRenderer",
    "MatplotlibRenderer",
    "run_matplotlib",
    "TerminalRenderer",
    "GPUHistoryViewer",
    "SelkiesRenderer",
]
