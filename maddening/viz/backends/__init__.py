"""Visualization backends for MADDENING.

Backends are imported lazily to avoid pulling in matplotlib / rich
when only one backend (or none) is needed.
"""


def __getattr__(name: str):
    if name in ("MatplotlibTimeSeriesRenderer", "MatplotlibSceneRenderer",
                "MatplotlibRenderer", "run_matplotlib"):
        from maddening.viz.backends import matplotlib_renderer as _mpl
        return getattr(_mpl, name)
    if name == "TerminalRenderer":
        from maddening.viz.backends.terminal_renderer import TerminalRenderer
        return TerminalRenderer
    if name == "GPUHistoryViewer":
        from maddening.viz.backends.pygfx_viewer import GPUHistoryViewer
        return GPUHistoryViewer
    raise AttributeError(f"module 'maddening.viz.backends' has no attribute {name!r}")


__all__ = [
    "MatplotlibTimeSeriesRenderer",
    "MatplotlibSceneRenderer",
    "MatplotlibRenderer",
    "run_matplotlib",
    "TerminalRenderer",
    "GPUHistoryViewer",
]
