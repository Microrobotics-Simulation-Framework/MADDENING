"""Visualization backends for MADDENING.

Backends are imported lazily to avoid pulling in matplotlib / rich
when only one backend (or none) is needed.
"""

_INSTALL_HINTS = {
    "MatplotlibTimeSeriesRenderer": "viz",
    "MatplotlibSceneRenderer": "viz",
    "MatplotlibRenderer": "viz",
    "run_matplotlib": "viz",
    "TerminalRenderer": "terminal",
    "GPUHistoryViewer": "gpu-viz",
    "SelkiesRenderer": "streaming",
}


def __getattr__(name: str):
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
