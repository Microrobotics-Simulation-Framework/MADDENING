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
    :attr:`BoundaryInputSpec.expected_units`.

    Permanently advisory.  Unlike :class:`ShapeMismatchWarning` and
    :class:`DtypeMismatchWarning` (which flip to errors in v0.2.1),
    unit mismatches stay as warnings forever — units are
    documentation, not contract.  MADDENING does not second-guess
    physics decisions that are the user's domain.
    """

    pass


# ---------------------------------------------------------------------------
# v0.2 #1 follow-up: structured migration errors
# ---------------------------------------------------------------------------


class MigrationError(RuntimeError):
    """Raised when a removed API is still in use (typically across a
    minor-version cutover that removed a compat shim).

    Carries enough structured detail that callers can build
    auto-migration tooling against the error — no need to grep the
    string message.

    Attributes
    ----------
    api_name : str
        The removed symbol (e.g. ``"SimulationNode.requires_halo"``).
    affected_class : type | None
        Which subclass triggered the error, when available.
    replacement : str | None
        Short hint at the new API (e.g. ``"halo_width()"``).
    migration_guide : str | None
        Documentation URL the user can follow.

    Example
    -------
    Used as a placeholder in v0.2 for the v0.3 hard-removal of
    :attr:`SimulationNode.requires_halo`::

        raise MigrationError(
            api_name="SimulationNode.requires_halo",
            affected_class=cls,
            replacement="halo_width() -> dict[int, int]",
            migration_guide=(
                "https://microrobotica.org/maddening/developer_guide/"
                "halo_width_migration.html"
            ),
        )
    """

    def __init__(
        self,
        api_name: str,
        *,
        affected_class: type | None = None,
        replacement: str | None = None,
        migration_guide: str | None = None,
    ):
        self.api_name = api_name
        self.affected_class = affected_class
        self.replacement = replacement
        self.migration_guide = migration_guide
        parts = [f"{api_name} was removed."]
        if affected_class is not None:
            parts.append(f"Still used by {affected_class.__qualname__}.")
        if replacement is not None:
            parts.append(f"Use {replacement} instead.")
        if migration_guide is not None:
            parts.append(f"See {migration_guide}.")
        super().__init__(" ".join(parts))
