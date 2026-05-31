"""MADDENING-specific warning categories and edge-validation exceptions."""

import sys

# ExceptionGroup is a builtin on 3.11+; the ``exceptiongroup`` backport
# package supplies it on 3.10.  GraphManager.compile() raises
# ExceptionGroup so callers can ``except*`` over Shape/Dtype mismatches
# independently while still seeing every problem in a single pass.
if sys.version_info >= (3, 11):
    BaseExceptionGroup = BaseExceptionGroup  # noqa: F821 — 3.11+ builtin
    ExceptionGroup = ExceptionGroup  # noqa: F821 — 3.11+ builtin
else:  # pragma: no cover — 3.10 fallback
    from exceptiongroup import (  # type: ignore[import-not-found]
        BaseExceptionGroup,
        ExceptionGroup,
    )


class PerformanceWarning(UserWarning):
    """Issued when a suboptimal code path is taken (e.g. CPU fallback)."""

    pass


# ---------------------------------------------------------------------------
# v0.2.1: compile-time edge validation — Shape / Dtype mismatches are now
# hard errors (pre-announced in v0.2.0 release notes; see semver carve-out
# in CHANGELOG and docs/developer_guide/edge_validation_migration.md).
# Unit mismatches stay as warnings (UnitMismatchWarning below).
# ---------------------------------------------------------------------------


class EdgeValidationError(Exception):
    """Base class for edge-validation errors raised at ``compile()`` time.

    :meth:`maddening.core.graph_manager.GraphManager.compile` aggregates
    every detected problem and raises a single
    :class:`ExceptionGroup` whose ``.exceptions`` contains one
    :class:`ShapeMismatchError` or :class:`DtypeMismatchError` per
    problem.  Catch :class:`EdgeValidationError` (or the more specific
    subclasses) inside an ``except*`` to handle them uniformly::

        try:
            gm.compile()
        except* ShapeMismatchError as eg:
            for err in eg.exceptions:
                ...  # report or auto-fix shape mismatches
        except* DtypeMismatchError as eg:
            ...
    """


class ShapeMismatchError(EdgeValidationError):
    """Edge brings a field whose runtime shape disagrees with the
    target node's :attr:`BoundaryInputSpec.shape`."""


class DtypeMismatchError(EdgeValidationError):
    """Edge brings a field whose dtype disagrees with the target node's
    :attr:`BoundaryInputSpec.dtype`."""


# ---------------------------------------------------------------------------
# Permanently advisory warnings (units only — shape / dtype are errors).
# ---------------------------------------------------------------------------
#
# The v0.2.1 deprecation aliases EdgeValidationWarning,
# ShapeMismatchWarning, and DtypeMismatchWarning were removed in v0.3.0
# per the v0.2.x release notes.  Use EdgeValidationError /
# ShapeMismatchError / DtypeMismatchError above.


class UnitMismatchWarning(UserWarning):
    """Edge declares units that don't match the target node's
    :attr:`BoundaryInputSpec.expected_units`.

    **Permanently advisory.**  Unlike the (now-removed)
    ``ShapeMismatchWarning`` / ``DtypeMismatchWarning`` paths, unit
    mismatches stay as warnings forever — units are documentation,
    not contract.  MADDENING does not second-guess physics decisions
    that are the user's domain.
    """


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
