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

Any one configuration reads only some of a group's eighteen settings.
A knob the rest of the configuration never reads warns at construction
rather than turning silently; the rules are in ``_INERT_RULES``.
"""

from __future__ import annotations

import sys
import warnings
from collections.abc import Callable, Mapping
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
        successive iterations.  Read **only** when
        ``convergence_norm="l2"``; the other two norms carry their
        tolerances in ``atol`` / ``rtol`` and test against a fixed
        threshold of ``1.0``.  Setting it away from its default under
        those norms is inert and warns (``UserWarning``).
    convergence_norm : {"l2", "mixed", "interface"}
        Norm used to check convergence.  ``"l2"`` uses a global L2
        norm with ``tolerance`` as threshold.  ``"mixed"`` uses a
        per-field mixed absolute/relative norm (converged when the
        norm <= 1.0).  ``"interface"`` checks consistency of
        coupling-edge values between iterations.
    atol : float
        Absolute tolerance for the ``"mixed"`` and ``"interface"``
        norms.  Read **only** by those two; setting it away from its
        default under ``convergence_norm="l2"`` is inert and warns
        (``UserWarning``) — tighten ``tolerance`` instead.
    rtol : float
        Relative tolerance for the ``"mixed"`` and ``"interface"``
        norms.  Read **only** by those two, on the same terms as
        ``atol``.
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
        ``> 1`` is over-relaxation.  Read **only** under that
        acceleration; setting it away from its default under any other
        is inert and warns (``UserWarning``).
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
        coupling loop.  Read **only** under the two IQN accelerations;
        supplying it under any other is inert and warns
        (``UserWarning``).
    subcycling : bool
        If True, allow mixed timesteps within the coupling group.
        Fast nodes take multiple sub-steps per coupling iteration.
    boundary_interpolation : str
        Time interpolation of boundary conditions during subcycling.
        Read **only** when ``subcycling=True``; setting it away from
        its default otherwise is inert and warns (``UserWarning``).
        ``"constant"`` holds values constant, ``"linear"`` linearly
        interpolates between previous and current iteration values,
        ``"quadratic"`` uses quadratic Lagrange interpolation through
        three successive iteration values (falls back to linear on
        the first iteration).
    jacobian_reuse : int
        For ``"iqn-imvj"``: number of V/W columns retained from the
        previous timestep.  ``0`` means no reuse (same as IQN-ILS).
        Read **only** under that acceleration; setting it away from its
        default under any other is inert and warns (``UserWarning``).
    waveform_iterations : int
        For subcycling groups: number of waveform relaxation
        iterations.  ``1`` is current behaviour (single pass),
        ``> 1`` iterates over entire sub-step windows.  Read **only**
        when ``subcycling=True``; setting it away from its default
        otherwise is inert and warns (``UserWarning``).
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
        calibration runs.  Read **only** under ``solver="ift"``;
        setting it True under ``"fori"`` is inert and warns
        (``UserWarning``).
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
        overrides for triage.  Read **only** under ``solver="ift"``,
        the only path that solves a tangent system; setting it away
        from its default under ``"fori"`` is inert and warns
        (``UserWarning``).

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
        self._warn_about_inert_settings()
        if self.solver == "fori":
            warnings.warn(
                "CouplingGroup solver='fori' is deprecated and will be "
                "removed in the next minor release; the default "
                "solver='ift' exits early on convergence and supports "
                "forward- and reverse-mode AD.",
                DeprecationWarning,
                stacklevel=_caller_stacklevel(),
            )

    def _warn_about_inert_settings(self) -> None:
        """Warn about every knob this group's configuration never reads.

        A ``CouplingGroup`` carries eighteen settings and any one
        configuration reads only some of them.  The convergence norm
        picks one of two tolerance knobs; the acceleration method picks
        whether ``relaxation``, ``jacobian_reuse`` and
        ``accelerated_fields`` mean anything; ``subcycling`` gates
        ``waveform_iterations`` and ``boundary_interpolation``; and the
        solver gates ``linear_solver`` and ``strict_convergence``.
        Nothing rejects the unread setting, and nothing used to report
        it, so the knob turned silently.

        Tightening ``tolerance`` from 1e-4 to 1e-14 on an
        ``"interface"`` group changes no digit of the answer -- which
        reads exactly like a solver converging to a different fixed
        point, and has already been written up as one, in a decisions
        document, and acted on.  An inert control proves nothing; this
        says so at the call site while the user can still act on it.
        The same failure was available through six more knobs, so the
        rules live in one table (:data:`_INERT_RULES`) keyed off the
        *read sites* in ``graph_manager``, not off this docstring.

        Only a *deliberate* setting warns.  A group that names a norm,
        an acceleration or a solver and leaves the knobs the others
        would have read alone has done nothing wrong, so the test is
        against the field's declared default rather than a record of
        what the caller passed -- a frozen dataclass keeps no such
        record, and a sentinel default would have to survive
        :meth:`to_dict`, the USD schema and every ``float(...)`` read of
        these fields.  The one case it cannot see is an explicit value
        that equals the default, which is also the one case where the
        warning would tell the user nothing they could act on.
        Comparing against the default is what makes the round trip
        through :meth:`to_dict` / :func:`coupling_group_kwargs` quiet:
        it re-passes every field by name, defaults included.
        """
        for rule in _INERT_RULES:
            if rule.live(self):
                continue
            named = tuple(
                name for name in rule.fields
                if getattr(self, name) != _FIELD_DEFAULTS[name]
            )
            if not named:
                continue
            warnings.warn(
                rule.message(self, named),
                UserWarning,
                stacklevel=_caller_stacklevel(),
            )


#: Declared default of every :class:`CouplingGroup` field, used by
#: ``_warn_about_inert_settings`` to tell a deliberately-set knob from
#: one the caller never touched.  ``nodes`` has no default and maps to
#: ``dataclasses.MISSING``; nothing looks it up.
_FIELD_DEFAULTS: dict[str, Any] = {
    f.name: f.default for f in fields(CouplingGroup)
}


@dataclass(frozen=True)
class _InertRule:
    """One knob -- or one inseparable pair -- and the setting that reads it.

    Attributes
    ----------
    fields : tuple of str
        The :class:`CouplingGroup` field names this rule governs.
        ``atol`` and ``rtol`` share a rule because they share a fate and
        a message: they go live and dead together under the same norm,
        and two warnings for one mistake is one too many.
    live : callable
        ``live(group)`` is True when this group's configuration actually
        reads those fields.  Each predicate mirrors a read site in
        ``maddening.core.graph_manager``; see :data:`_INERT_RULES` for
        where.
    message : callable
        ``message(group, names)`` builds the warning for the subset of
        ``fields`` the caller set away from its default.  Every message
        names the field, the value it was given, the setting that makes
        it inert and the setting that would make it live -- the four
        things needed to act on it without opening the source.
    """

    fields: tuple[str, ...]
    live: Callable[[CouplingGroup], bool]
    message: Callable[[CouplingGroup, tuple[str, ...]], str]


def _inert_atol_rtol_message(
    group: CouplingGroup, names: tuple[str, ...]
) -> str:
    """``atol`` / ``rtol`` under the one norm that is never handed them."""
    setting = " and ".join(f"{n}={getattr(group, n)!r}" for n in names)
    return (
        f"CouplingGroup.{setting} "
        f"{'are' if len(names) > 1 else 'is'} ignored under "
        "convergence_norm='l2', which tests the global L2 norm "
        "of the state change against tolerance alone.  Set "
        "tolerance to control convergence under this norm, or "
        "choose convergence_norm='mixed' or 'interface' to make "
        "atol and rtol live."
    )


def _inert_tolerance_message(
    group: CouplingGroup, names: tuple[str, ...]
) -> str:
    """``tolerance`` under a norm whose threshold is hard-coded to 1.0."""
    return (
        f"CouplingGroup.tolerance={group.tolerance!r} is ignored "
        f"under convergence_norm={group.convergence_norm!r}, whose "
        "residual is already scaled by atol and rtol and is tested "
        "against a fixed threshold of 1.0.  Set atol and rtol to "
        "control convergence under this norm, or choose "
        "convergence_norm='l2' to make tolerance live."
    )


def _gated_on(
    gate: str, reason: str, remedy: str
) -> Callable[[CouplingGroup, tuple[str, ...]], str]:
    """Message builder for a knob read only under one setting of ``gate``.

    Every knob below is gated on exactly one other field, so one
    sentence shape serves them all and the messages read alike:
    *what you set*, *what made it inert*, *what to set instead*.
    """
    def build(group: CouplingGroup, names: tuple[str, ...]) -> str:
        (name,) = names
        return (
            f"CouplingGroup.{name}={getattr(group, name)!r} is ignored "
            f"under {gate}={getattr(group, gate)!r}, which {reason}.  "
            f"{remedy} to make {name} live, or leave {name} at its "
            f"default ({_FIELD_DEFAULTS[name]!r})."
        )

    return build


#: Every :class:`CouplingGroup` knob that only some configurations read.
#: Each ``live`` predicate mirrors the read site that decides it, so a
#: rule is wrong only if the solver changed under it:
#:
#: * ``atol`` / ``rtol`` -- ``_compute_residual`` passes them to the
#:   mixed and interface residuals only; ``coupling_residual_l2`` does
#:   not take them.
#: * ``tolerance`` -- ``conv_threshold_value`` is ``1.0`` for the mixed
#:   and interface norms and ``float(group.tolerance)`` otherwise.
#: * ``relaxation`` -- ``_accelerate`` and the fori path both read it
#:   under ``acceleration == "fixed"`` alone; Aitken and the IQN
#:   methods derive their own factor.
#: * ``jacobian_reuse`` -- ``_iqn_warm_start`` returns zeros before
#:   reaching it unless ``acceleration == "iqn-imvj"``.
#: * ``accelerated_fields`` -- ``accel_fields`` is ``None`` unless
#:   ``acceleration`` is one of the two IQN methods.  (``compile()``
#:   still validates the names whatever the acceleration, so a typo is
#:   caught either way; it just does not change the solve.)
#: * ``waveform_iterations`` -- ``n_waveform`` is ``1`` unless the group
#:   subcycles.
#: * ``boundary_interpolation`` -- the interpolation flags are only
#:   computed inside the ``if use_subcycling:`` block.
#: * ``linear_solver`` / ``strict_convergence`` -- both are read inside
#:   ``_run_ift_forward``, which only ``solver="ift"`` calls.
_INERT_RULES: tuple[_InertRule, ...] = (
    _InertRule(
        fields=("atol", "rtol"),
        live=lambda g: g.convergence_norm != "l2",
        message=_inert_atol_rtol_message,
    ),
    _InertRule(
        fields=("tolerance",),
        live=lambda g: g.convergence_norm == "l2",
        message=_inert_tolerance_message,
    ),
    _InertRule(
        fields=("relaxation",),
        live=lambda g: g.acceleration == "fixed",
        message=_gated_on(
            "acceleration",
            "applies no constant relaxation factor; only "
            "acceleration='fixed' scales the update by relaxation, and "
            "aitken and the IQN methods derive their own factor",
            "Choose acceleration='fixed'",
        ),
    ),
    _InertRule(
        fields=("jacobian_reuse",),
        live=lambda g: g.acceleration == "iqn-imvj",
        message=_gated_on(
            "acceleration",
            "starts every timestep from empty secant matrices; only "
            "acceleration='iqn-imvj' carries V/W columns across "
            "timesteps",
            "Choose acceleration='iqn-imvj'",
        ),
    ),
    _InertRule(
        fields=("accelerated_fields",),
        live=lambda g: g.acceleration in ("iqn-ils", "iqn-imvj"),
        message=_gated_on(
            "acceleration",
            "solves no quasi-Newton problem to select fields for; only "
            "acceleration='iqn-ils' and 'iqn-imvj' accelerate over an "
            "interface subset",
            "Choose acceleration='iqn-ils' or 'iqn-imvj'",
        ),
    ),
    _InertRule(
        fields=("waveform_iterations",),
        live=lambda g: g.subcycling,
        message=_gated_on(
            "subcycling",
            "takes one pass per coupling iteration; waveform relaxation "
            "repeats a whole sub-step window, which only a subcycled "
            "group has",
            "Set subcycling=True",
        ),
    ),
    _InertRule(
        fields=("boundary_interpolation",),
        live=lambda g: g.subcycling,
        message=_gated_on(
            "subcycling",
            "resolves boundary inputs once per coupling iteration, with "
            "no intermediate time to interpolate to",
            "Set subcycling=True",
        ),
    ),
    _InertRule(
        fields=("linear_solver",),
        live=lambda g: g.solver == "ift",
        message=_gated_on(
            "solver",
            "differentiates straight through the iterates and solves no "
            "tangent system",
            "Choose solver='ift'",
        ),
    ),
    _InertRule(
        fields=("strict_convergence",),
        live=lambda g: g.solver == "ift",
        message=_gated_on(
            "solver",
            "returns the derivative of the iterate it stopped on rather "
            "than of a fixed point, so there is no IFT gradient to "
            "guard",
            "Choose solver='ift'",
        ),
    ),
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


#: Frames belonging to this package are construction machinery, never
#: the line a reader of a warning can change.  Compared against a
#: frame's ``__name__``, not its filename, so an editable install, a
#: worktree and a wheel all answer the same.
_PACKAGE = "maddening"


def _caller_stacklevel() -> int:
    """``stacklevel`` landing a warning on the first frame outside MADDENING.

    A constant cannot do this job.  The frame that wrote
    ``CouplingGroup(...)`` is three above a warning raised in
    :meth:`CouplingGroup._warn_about_inert_settings` (the helper,
    ``__post_init__``, the generated ``__init__``); the frame that wrote
    ``gm.add_coupling_group(...)`` is four, ``gm.auto_couple()`` five,
    and a config or USD load more again.  A constant tuned for one of
    them points every other case at library source -- and
    ``graph_manager.py``'s construction line is not something the reader
    can act on, which is most of what makes a warning worth emitting.

    So the stack is walked instead: the first frame whose module is
    outside the ``maddening`` package is the user's.  A stack that is
    package frames all the way up (an example module run as the entry
    point) falls back to its outermost frame rather than off the end,
    where :mod:`warnings` would attribute the message to ``sys``.
    """
    try:
        frame = sys._getframe(1)  # the frame that will call warnings.warn
    except (AttributeError, ValueError):  # pragma: no cover - not CPython
        # Without frame introspection, point at the caller of whatever
        # called us -- right for a direct ``CouplingGroup(...)``.
        return 3
    level = 1
    while True:
        module = frame.f_globals.get("__name__", "")
        if module != _PACKAGE and not module.startswith(_PACKAGE + "."):
            return level
        parent = frame.f_back
        if parent is None:
            return level
        frame = parent
        level += 1


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
