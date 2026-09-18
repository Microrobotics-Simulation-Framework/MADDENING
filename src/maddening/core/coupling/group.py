"""
Iterative coupling for algebraic loops (Gauss-Seidel fixed-point).

When nodes form a cycle, the default behaviour is *staggering*: back-edges
read previous-timestep values.  For strongly-coupled subsystems this can be
inaccurate or unstable.

A ``CouplingGroup`` wraps a set of cyclic nodes in a ``jax.lax.while_loop``
that iterates the group each timestep until the state change drops below a
tolerance (or ``max_iterations`` is reached), differentiating through the
converged fixed point via the implicit function theorem.  All edges within
the group become *forward* edges during iteration, giving Gauss-Seidel
convergence.

Supports multiple convergence norms, acceleration methods (Aitken,
fixed relaxation, IQN-ILS, IQN-IMVJ), Jacobi iteration mode, and
subcycling for mixed-timestep coupling groups.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import (
    Any,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)


@dataclass(frozen=True)
class CouplingGroup:
    """Configuration for an iteratively-coupled group of nodes.

    Parameters
    ----------
    nodes : frozenset[str]
        Names of the nodes that participate in the coupling group.
        All must belong to the same graph and form (part of) a cycle.
    max_iterations : int
        Upper bound on iterations per timestep.
    tolerance : float
        Convergence threshold on the L2 norm of state change between
        successive iterations.  Used when ``convergence_norm="l2"``.
    convergence_norm : {"l2", "mixed", "interface"}
        Norm used to check convergence.  ``"l2"`` uses a global L2
        norm with ``tolerance`` as threshold.  ``"mixed"`` uses a
        per-field mixed absolute/relative norm (converged when the
        norm <= 1.0).  ``"interface"`` checks consistency of
        coupling-edge values between iterations.
    atol : float
        Absolute tolerance for the ``"mixed"`` norm.
    rtol : float
        Relative tolerance for the ``"mixed"`` norm.
    diagnostics : bool
        If True, store iteration count and final residual in the
        ``_meta`` key of the state dict after each step.
    acceleration : str
        Acceleration method.  ``"none"`` is plain fixed-point,
        ``"aitken"`` uses Aitken delta-squared relaxation,
        ``"fixed"`` uses constant under-relaxation with ``relaxation``
        as the omega parameter, ``"iqn-ils"`` uses Interface
        Quasi-Newton with Inverse Least Squares, ``"iqn-imvj"``
        uses IQN-ILS with multi-timestep Jacobian reuse.
    relaxation : float
        Constant relaxation factor for ``acceleration="fixed"``.
        ``1.0`` is no relaxation, ``< 1`` is under-relaxation,
        ``> 1`` is over-relaxation.
    iteration_mode : str
        ``"gauss-seidel"`` updates nodes sequentially (each sees
        current-iteration values from earlier nodes).
        ``"jacobi"`` updates all nodes from the frozen
        previous-iteration state.
    accelerated_fields : dict or None
        For ``"iqn-ils"``: which fields per node participate in the
        quasi-Newton problem.  ``None`` auto-detects from coupling
        edges (interface fields only).  Otherwise a mapping
        ``{node: (field, ...)}`` naming at least one field on at least
        one node *of this group*.  A non-mapping, a value given as a
        bare field name instead of a one-element tuple, an empty
        mapping and a mapping naming only foreign nodes all raise
        ``ValueError`` here rather than failing inside the traced
        coupling loop.
    subcycling : bool
        If True, allow mixed timesteps within the coupling group.
        Fast nodes take multiple sub-steps per coupling iteration.
    boundary_interpolation : str
        Time interpolation of boundary conditions during subcycling.
        ``"constant"`` holds values constant, ``"linear"`` linearly
        interpolates between previous and current iteration values,
        ``"quadratic"`` uses quadratic Lagrange interpolation through
        three successive iteration values (falls back to linear on
        the first iteration).
    jacobian_reuse : int
        For ``"iqn-imvj"``: number of V/W columns retained from the
        previous timestep.  ``0`` means no reuse (same as IQN-ILS).
    waveform_iterations : int
        For subcycling groups: number of waveform relaxation
        iterations.  ``1`` is current behaviour (single pass),
        ``> 1`` iterates over entire sub-step windows.
    predictor : str
        Extrapolation of the coupling initial guess from previous
        converged states.  ``"none"`` uses the current state (default),
        ``"linear"`` uses linear extrapolation from the last two
        converged states, ``"quadratic"`` uses quadratic extrapolation
        from the last three converged states.  Reduces iteration count
        for smoothly varying problems.
    solver : {"ift", "fori"}
        How the fixed-point iteration is solved.  ``"ift"`` (default)
        runs a ``jax.lax.while_loop`` that exits as soon as the
        group's convergence norm meets its threshold, and
        differentiates via the implicit function theorem at the fixed
        point — a ``custom_jvp`` rule, so ``jax.jvp`` / ``jacfwd`` and
        ``jax.grad`` / ``jacrev`` both work through the step.  Every
        ``acceleration``, ``iteration_mode``, ``convergence_norm`` and
        ``diagnostics`` setting is supported.  The IFT derivative is
        exact only at a converged fixed point; see
        ``strict_convergence``.

        ``"fori"`` is the legacy path — a static ``jax.lax.fori_loop``
        that always runs ``max_iterations`` passes (freezing the state
        once converged) and differentiates straight through the
        iterates.  Deprecated: emits ``DeprecationWarning`` and will be
        removed in the next minor release.  Forward-mode AD does not
        work through it.

        The two return the *same state*: both stop on the iterate whose
        residual met the criterion rather than on the update it went on
        to produce, so migrating a graph off ``"fori"`` does not change
        the forward answer.  The *gradients* differ by design —
        ``"ift"`` gives the derivative of the fixed point, ``"fori"``
        the derivative of the iterate it returned — and they agree to
        round-off once the criterion is tight enough for the two to be
        the same point.
    strict_convergence : bool
        For ``solver="ift"``: if True, raise a runtime error (via
        ``equinox.error_if``, jit-safe) when the group exits at
        ``max_iterations`` and the state it returns is still outside
        its threshold, since the gradient through that step is then
        invalid.  The test is on that state's own residual, so a group
        that arrives on its last pass runs rather than raising (see
        ``GraphManager.coupling_diagnostics``).  Default False:
        the condition is only reported via
        ``GraphManager.coupling_diagnostics()["converged"]`` when
        ``diagnostics=True``.  Recommended True for training and
        calibration runs.
    linear_solver : {"gmres", "dense"}
        Backend used by the ``"ift"`` derivative rule to solve the
        tangent system ``(I - dF/dx) x_dot = rhs`` (and, transposed,
        the adjoint system).  ``"gmres"`` (default) is the
        matrix-free GMRES path via lineax — the safe default for the
        non-symmetric coupling Jacobians MADDENING produces.
        ``"dense"`` is the legacy ``jacrev + jnp.linalg.solve`` path,
        promoted from an env-var-gated fallback to a first-class
        config option (O(N^2) memory, O(N^3) compute — only viable
        for small groups).  The ``MADDENING_IFT_DENSE_SOLVE=1`` env
        var forces ``"dense"`` regardless of this setting and
        overrides for triage.

        Note on BiCGStab: lineax 0.0.7 ships ``lineax.BiCGStab``, but
        it returns NaN when driving a ``FunctionLinearOperator`` (the
        matrix-free shape our backward uses) — including on
        well-conditioned operators like ``0.5*I``.  The
        ``"bicgstab"`` option is therefore *not* exposed here.  See
        ``_ift_linear_solve`` and
        ``tests/core/test_coupling_ift_lineax.py`` for the
        investigation notes.
    """
    nodes: frozenset[str]
    max_iterations: int = 10
    tolerance: float = 1e-6
    convergence_norm: Literal["l2", "mixed", "interface"] = "l2"
    atol: float = 1e-8
    rtol: float = 1e-6
    diagnostics: bool = False
    acceleration: Literal[
        "none", "aitken", "fixed", "iqn-ils", "iqn-imvj"
    ] = "none"
    relaxation: float = 1.0
    iteration_mode: Literal["gauss-seidel", "jacobi"] = "gauss-seidel"
    accelerated_fields: Optional[dict[str, tuple[str, ...]]] = None
    subcycling: bool = False
    boundary_interpolation: Literal[
        "constant", "linear", "quadratic"
    ] = "linear"
    jacobian_reuse: int = 0
    waveform_iterations: int = 1
    predictor: Literal["none", "linear", "quadratic"] = "none"
    solver: Literal["fori", "ift"] = "ift"
    strict_convergence: bool = False
    linear_solver: Literal["gmres", "dense"] = "gmres"

    def to_dict(self) -> dict[str, Any]:
        """Every field of this group as JSON-compatible plain data.

        Driven by ``dataclasses.fields`` rather than a hand-written list
        of keys: a solver setting added to this class in future is
        carried by the config the moment it exists, and the failure this
        method was written to close — a group that serialises to *some*
        of its configuration and silently solves differently when it
        comes back — cannot return by omission.

        Two fields are not already plain data:

        * ``nodes`` is a ``frozenset``, written as a **sorted** list so
          one group always produces one spelling;
        * ``accelerated_fields`` maps a node to a tuple of field names,
          written as a dict of lists.

        The other seventeen are ``int``, ``float``, ``bool`` or ``str``
        and are written as they are.  :func:`coupling_group_kwargs` is
        the inverse.
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "nodes":
                value = sorted(value)
            elif f.name == "accelerated_fields" and value is not None:
                value = {node: list(flds) for node, flds in sorted(value.items())}
            out[f.name] = value
        return out

    def __post_init__(self) -> None:
        """Validate that ``Literal``-typed string fields hold permitted values.

        Without this check a typo like ``acceleration="aitkin"`` silently
        sets the field to that string; runtime dispatch (``if group.acceleration
        == "aitken": ...``) simply fails to match and the group quietly falls
        back to whatever the default branch is.  The Literal annotation is
        purely a type-checker hint at runtime, so we re-derive the valid sets
        from ``typing.get_args`` on the annotation and raise ``ValueError`` on
        anything outside the permitted set.  Adding a new option to a Literal
        automatically extends the validator.
        """
        # ``from __future__ import annotations`` is active at module top, so
        # annotations are stringified.  ``get_type_hints`` resolves them
        # against this module's globals.
        hints = get_type_hints(type(self))
        for f in fields(self):
            ann = hints.get(f.name)
            valid = _literal_options(ann)
            if valid is None:
                continue  # not a Literal (or Optional[Literal]) — skip
            value = getattr(self, f.name)
            # Optional[Literal[...]] allows None as a valid sentinel.
            if value is None and _is_optional(ann):
                continue
            if value not in valid:
                raise ValueError(
                    f"CouplingGroup.{f.name}={value!r} is not a valid "
                    f"option; expected one of {valid!r}"
                )
        if self.accelerated_fields is not None:
            # Shape, before anything reads the mapping.  A non-mapping
            # used to reach ``.values()`` and surface as
            # ``AttributeError: 'list' object has no attribute
            # 'values'`` from inside ``__post_init__``; a bare string
            # value passed every check here and only failed at compile
            # time, as ``accelerated_fields['a'] names ['p', 'o', 's',
            # ...]`` -- the field name iterated character by character.
            if not isinstance(self.accelerated_fields, Mapping):
                raise ValueError(
                    "CouplingGroup.accelerated_fields must be a mapping of "
                    "{node: (field, ...)}, got "
                    f"{type(self.accelerated_fields).__name__}: "
                    f"{self.accelerated_fields!r}."
                )
            bad = sorted(
                k for k, v in self.accelerated_fields.items()
                if isinstance(v, str)
            )
            if bad:
                raise ValueError(
                    "CouplingGroup.accelerated_fields values must be "
                    "sequences of field names, but "
                    f"{bad} map to a bare string.  Wrap a single field "
                    'in a tuple: {"node": ("field",)}.'
                )
            # Content.  An empty mapping, or one naming only nodes
            # outside the group, leaves the quasi-Newton problem with
            # zero degrees of freedom.  That surfaces as
            # ``ValueError: Need at least one array to concatenate``
            # from ``jnp.concatenate`` deep inside the traced coupling
            # loop, with no mention of the setting that caused it.
            foreign = sorted(set(self.accelerated_fields) - set(self.nodes))
            if foreign:
                raise ValueError(
                    f"CouplingGroup.accelerated_fields names node(s) "
                    f"{foreign} that are not in the group "
                    f"({sorted(self.nodes)})."
                )
            if not any(self.accelerated_fields.values()):
                raise ValueError(
                    "CouplingGroup.accelerated_fields selects no field: "
                    f"{self.accelerated_fields!r}.  Use None to let the "
                    "group auto-detect its interface fields, or name at "
                    "least one field on one node in the group."
                )
        if self.solver == "fori":
            warnings.warn(
                "CouplingGroup solver='fori' is deprecated and will be "
                "removed in the next minor release; the default "
                "solver='ift' exits early on convergence and supports "
                "forward- and reverse-mode AD.",
                DeprecationWarning,
                stacklevel=3,
            )


def coupling_group_kwargs(d: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Split a serialised group into the ``(nodes, kwargs)`` pair that
    :meth:`~maddening.core.graph_manager.GraphManager.add_coupling_group`
    takes.

    The inverse of :meth:`CouplingGroup.to_dict`, and the one place both
    readers (config and USD) turn stored data back into constructor
    arguments.  Only the two fields that are not plain data are
    converted: ``accelerated_fields``'s lists become the tuples the
    dataclass declares, and ``nodes`` comes back out as the positional
    argument.

    Nothing else is validated here on purpose.  ``add_coupling_group`` checks
    the node names against the graph and the group against the groups
    already registered, and ``CouplingGroup.__post_init__`` checks every
    enum, so a hand-edited file is rejected by exactly the code that
    rejects a hand-written call — with one loader, not two.
    """
    kwargs = dict(d)
    nodes = kwargs.pop("nodes")
    if isinstance(nodes, str):
        # ``list("rod_a")`` is five one-letter node names, and the failure
        # that follows talks about a node called 'r'.  The one conversion
        # here that can go quietly wrong, so it is the one thing checked.
        raise TypeError(
            f"'nodes' is a list of node names, not the string {nodes!r}"
        )
    accelerated = kwargs.get("accelerated_fields")
    if accelerated is not None:
        kwargs["accelerated_fields"] = {
            node: tuple(flds) for node, flds in accelerated.items()
        }
    return list(nodes), kwargs


def _literal_options(ann: object) -> Optional[tuple]:
    """Return the ``Literal`` options of ``ann``, or ``None`` if not a Literal.

    Handles both bare ``Literal[...]`` and ``Optional[Literal[...]]``
    (which normalises to ``Union[Literal[...], None]``).
    """
    if ann is None:
        return None
    if get_origin(ann) is Literal:
        return get_args(ann)
    if get_origin(ann) is Union:
        for arg in get_args(ann):
            if get_origin(arg) is Literal:
                return get_args(arg)
    return None


def _is_optional(ann: object) -> bool:
    """True iff ``ann`` is ``Union[..., None]`` (i.e. ``Optional[...]``)."""
    return get_origin(ann) is Union and type(None) in get_args(ann)
