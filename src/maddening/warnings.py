"""MADDENING-specific warning categories."""


class PerformanceWarning(UserWarning):
    """Issued when a suboptimal code path is taken (e.g. CPU fallback)."""

    pass
