"""MADDENING-specific warning categories."""


class PerformanceWarning(UserWarning):
    """Issued when a suboptimal code path is taken (e.g. CPU fallback)."""

    pass


# ---------------------------------------------------------------------------
# v0.2 #4: compile-time edge validation
# ---------------------------------------------------------------------------


class EdgeValidationWarning(UserWarning):
    """Base class for edge-validation warnings raised at ``compile()`` time.

    In v0.2 these are warnings; the v0.2.1 plan flips them to
    ``EdgeValidationError`` (a hard exception).  Catch this class to
    handle all edge-validation warnings uniformly, or catch one of the
    more specific subclasses below.
    """

    pass


class ShapeMismatchWarning(EdgeValidationWarning):
    """Edge brings a field whose runtime shape disagrees with the
    target node's :attr:`BoundaryInputSpec.shape`."""

    pass


class DtypeMismatchWarning(EdgeValidationWarning):
    """Edge brings a field whose dtype disagrees with the target node's
    :attr:`BoundaryInputSpec.dtype`."""

    pass


class UnitMismatchWarning(EdgeValidationWarning):
    """Edge declares units that don't match the target node's
    :attr:`BoundaryInputSpec.expected_units`."""

    pass
